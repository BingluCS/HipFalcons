"""Parent-owned HDF5 readers for packing in forked DataLoader workers."""

import ctypes
from multiprocessing.connection import wait
from numbers import Integral
import os
import time
import traceback

import torch
import torch.multiprocessing  # Register shared tensor reducers for Connection.send.
from torch.utils.data import get_worker_info


_TIMEOUT = 120.0
_POLL_INTERVAL = 0.25
_SHUTDOWN_GRACE = 2.0
_MAX_SIZE = min((1 << 63) - 1, ctypes.c_size_t(-1).value)


def _positive_int(name, value):
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _align8(value):
    if value < 0 or value > _MAX_SIZE - 7:
        raise OverflowError("packed batch exceeds int64 indexing")
    return (value + 7) & ~7


def _decode_cpu_planes(dataset, codec, output, times, channels, first, last, height, width):
    count = len(times) * len(channels)
    if (not isinstance(output, torch.Tensor) or output.device.type != "cpu" or
            output.dtype != torch.float32 or not output.is_contiguous() or not output.is_shared() or
            tuple(output.shape) != (len(times), len(channels), height, width) or
            not 0 <= first < last <= count):
        raise ValueError("invalid CPU reader output range")
    plane = height * width
    address = output.data_ptr()
    plane_bytes = plane * ctypes.sizeof(ctypes.c_float)
    for i in range(first, last):
        frame, channel = times[i // len(channels)], channels[i % len(channels)]
        try:
            mask, chunk = dataset.id.read_direct_chunk((frame, channel, 0, 0))
        except Exception as error:
            raise RuntimeError("Could not read HDF5 chunk at time %d, channel %d"
                               % (frame, channel)) from error
        if mask != 0:
            raise ValueError("Falcons filter was skipped for chunk at time %d, channel %d (mask %d)"
                             % (frame, channel, mask))
        if not chunk:
            raise ValueError("Empty Falcons chunk at time %d, channel %d" % (frame, channel))
        codec.cpu(chunk, address + i * plane_bytes, plane)


def _reader_main(conn, inherited, shutdown, retired, worker_id, owner_pid,
                 path, name, shape, library, cpu_decode):
    for other in inherited:
        other.close()
    handle = None
    try:
        # Each helper opens its own dataset and native library after fork.
        from .loader import _Codec, _open_dataset

        handle, dataset, actual_shape = _open_dataset(path, name)
        if tuple(actual_shape) != shape:
            raise RuntimeError("HDF5 dataset changed after metadata validation")
        if cpu_decode:
            codec = _Codec(library, "cpu")
        else:
            from .fastpack import _native, pack_planes
            _native(library)
        conn.send(("ready", os.getpid()))
        worker_pid = None
        while not shutdown.value and not retired[worker_id] and os.getppid() == owner_pid:
            if not conn.poll(_POLL_INTERVAL):
                if worker_pid is not None:
                    try:
                        os.kill(worker_pid, 0)
                    except ProcessLookupError:
                        break
                    except PermissionError:
                        pass
                continue
            try:
                request = conn.recv()
            except EOFError:
                break
            if cpu_decode:
                seq, worker_pid, output, times, channels, first, last = request
                _decode_cpu_planes(dataset, codec, output, times, channels, first, last,
                                   shape[2], shape[3])
                conn.send(("decoded", seq))
                del request, output
            else:
                (seq, worker_pid, slab, destination, capacity,
                 times, channels, first, last) = request
                planes = ((times[i // len(channels)], channels[i % len(channels)])
                          for i in range(first, last))
                fallback, offsets, groups, used = pack_planes(
                    dataset, planes, shape[2], shape[3], library,
                    destination=(slab.data_ptr() + destination, capacity))
                conn.send(("packed", seq, offsets, groups, used, fallback))
                del request, slab, fallback
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()[-16384:]))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if handle is not None:
            handle.close()
        conn.close()


class ReaderPool:
    """Parent-started helpers; reader_processes includes the DataLoader worker."""

    def __init__(self, context, path, dataset_name, shape, cpu_library,
                 num_workers, reader_processes, *, cpu_decode=False):
        if context.get_start_method() != "fork":
            raise ValueError("ReaderPool requires a fork multiprocessing context")
        self._context = context
        self._path = os.fspath(path)
        self._name = dataset_name
        self._shape = tuple(_positive_int("shape dimension", x) for x in shape)
        if len(self._shape) != 4:
            raise ValueError("shape must be (time, channel, height, width)")
        self._library = os.fspath(cpu_library)
        self._cpu_decode = cpu_decode
        self._num_workers = _positive_int("num_workers", num_workers)
        self._reader_processes = _positive_int("reader_processes", reader_processes)
        if self._reader_processes <= 1:
            raise ValueError("reader_processes must include a worker and at least one helper")
        self._owner_pid = os.getpid()
        self._shutdown = context.RawValue(ctypes.c_byte, 0)
        self._retired = context.RawArray(ctypes.c_byte, self._num_workers)
        self._groups = [[] for _ in range(self._num_workers)]
        self._processes = []
        self._started = False
        self._closed = False
        self._worker_pid = None
        self._worker_id = None
        self._connections = None
        self._sequence = 0
        self._in_pack = False

    def start(self):
        if os.getpid() != self._owner_pid or get_worker_info() is not None:
            raise RuntimeError("ReaderPool.start() must run in its parent process")
        if self._closed:
            raise RuntimeError("ReaderPool is closed")
        if self._started:
            return self
        try:
            connections = []
            for worker_id in range(self._num_workers):
                for _ in range(self._reader_processes - 1):
                    worker_end, reader_end = self._context.Pipe(duplex=True)
                    process = self._context.Process(
                        target=_reader_main,
                        args=(reader_end, tuple(connections) + (worker_end,), self._shutdown,
                              self._retired, worker_id, self._owner_pid, self._path, self._name,
                              self._shape, self._library, self._cpu_decode), daemon=True)
                    try:
                        process.start()
                    except BaseException:
                        worker_end.close()
                        reader_end.close()
                        raise
                    reader_end.close()
                    self._groups[worker_id].append(worker_end)
                    connections.append(worker_end)
                    self._processes.append(process)
            pending = {conn: time.monotonic() + _TIMEOUT for conn in connections}
            while pending:
                for conn, message in self._receive(pending):
                    if len(message) != 2 or message[0] != "ready":
                        raise RuntimeError("invalid reader startup response")
                    del pending[conn]
            self._started = True
            return self
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _receive(self, pending):
        if self._shutdown.value:
            raise RuntimeError("ReaderPool is shutting down")
        remaining = min(pending.values()) - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("timed out waiting for a CPU reader")
        responses = []
        for conn in wait(pending, min(remaining, _POLL_INTERVAL)):
            try:
                message = conn.recv()
            except (EOFError, OSError, RuntimeError) as error:
                raise RuntimeError("CPU reader disconnected") from error
            if not isinstance(message, tuple) or not message:
                raise RuntimeError("invalid CPU reader response")
            if message[0] == "error":
                raise RuntimeError("CPU reader failed:\n" + str(message[1]))
            responses.append((conn, message))
        return responses

    def _worker(self):
        if not self._started or self._closed or self._shutdown.value:
            raise RuntimeError("ReaderPool is not running")
        info = get_worker_info()
        if info is None or os.getpid() == self._owner_pid:
            raise RuntimeError("ReaderPool.pack() must run in a DataLoader worker")
        worker_id = int(info.id)
        if worker_id < 0 or worker_id >= self._num_workers:
            raise RuntimeError("DataLoader worker has no assigned reader group")
        if self._worker_pid is None:
            self._worker_pid, self._worker_id = os.getpid(), worker_id
            self._connections = self._groups[worker_id]
            for i, group in enumerate(self._groups):
                if i != worker_id:
                    for conn in group:
                        conn.close()
            self._groups = []
        if self._worker_pid != os.getpid() or self._worker_id != worker_id:
            raise RuntimeError("ReaderPool cannot be inherited by another worker")
        if self._retired[worker_id]:
            raise RuntimeError("DataLoader worker's reader group is no longer available")
        return self._connections

    def pack(self, times, channels, dataset):
        connections = self._worker()
        if self._in_pack:
            raise RuntimeError("concurrent pack calls in one worker are unsupported")
        times, channels = tuple(times), tuple(channels)
        if not times or not channels:
            raise ValueError("cannot pack an empty set of planes")
        for name, values, limit in (("time", times, self._shape[0]),
                                    ("channel", channels, self._shape[1])):
            if any(not isinstance(v, Integral) or isinstance(v, bool) or v < 0 or v >= limit
                   for v in values):
                raise IndexError(f"{name} index out of range")
        self._in_pack = True
        try:
            return self._pack(connections, tuple(map(int, times)), tuple(map(int, channels)), dataset)
        except BaseException:
            self._retired[self._worker_id] = 1
            for conn in connections:
                conn.close()
            raise
        finally:
            self._in_pack = False

    def decode_cpu(self, times, channels, dataset, codec, output):
        if not self._cpu_decode:
            raise RuntimeError("ReaderPool was not started for CPU decoding")
        connections = self._worker()
        if self._in_pack:
            raise RuntimeError("concurrent reader calls in one worker are unsupported")
        times, channels = tuple(times), tuple(channels)
        if not times or not channels:
            raise ValueError("cannot decode an empty set of planes")
        self._in_pack = True
        try:
            count = len(times) * len(channels)
            participants = min(self._reader_processes, count)
            quotient, remainder = divmod(count, participants)
            self._sequence += 1
            seq = self._sequence
            pending = {}
            for i, conn in enumerate(connections[:participants - 1], 1):
                first = i * quotient + min(i, remainder)
                last = first + quotient + (i < remainder)
                conn.send((seq, os.getpid(), output, times, channels, first, last))
                pending[conn] = None
            _decode_cpu_planes(dataset, codec, output, times, channels, 0,
                               quotient + (remainder != 0), self._shape[2], self._shape[3])
            pending = {conn: time.monotonic() + _TIMEOUT for conn in pending}
            while pending:
                for conn, message in self._receive(pending):
                    if message != ("decoded", seq):
                        raise RuntimeError("invalid CPU reader response")
                    del pending[conn]
        except BaseException:
            self._retired[self._worker_id] = 1
            for conn in connections:
                conn.close()
            raise
        finally:
            self._in_pack = False

    def _pack(self, connections, times, channels, dataset):
        # This helper creates an already-shared CPU tensor without a full-size copy.
        from .fastpack import pack_planes
        from .loader import _shared_bytes

        count = len(times) * len(channels)
        estimate = 2 * self._shape[2] * self._shape[3] + 16384
        if count > _MAX_SIZE or estimate > _MAX_SIZE:
            raise OverflowError("packed batch exceeds int64 indexing")
        participants = min(self._reader_processes, count)
        active = connections[:participants - 1]
        quotient, remainder = divmod(count, participants)
        shards, capacity, first = [], 0, 0
        for i in range(participants):
            last = first + quotient + (i < remainder)
            size = _align8((last - first) * estimate)
            if capacity > _MAX_SIZE - size:
                raise OverflowError("packed batch exceeds int64 indexing")
            shards.append({"first": first, "last": last,
                           "base": capacity, "capacity": size})
            capacity += size
            first = last
        slab = _shared_bytes(capacity)
        self._sequence += 1
        seq = self._sequence
        pending = {}
        for conn, shard in zip(active, shards[1:]):
            conn.send((seq, os.getpid(), slab, shard["base"], shard["capacity"],
                       times, channels, shard["first"], shard["last"]))
            pending[conn] = shard

        def record(shard, offsets, groups, used, fallback):
            if (not isinstance(used, int) or not 0 < used <= _MAX_SIZE or
                    not isinstance(groups, int) or groups < 0 or
                    len(offsets) != shard["last"] - shard["first"] or
                    any(not isinstance(v, int) or v < 0 or v >= used or v % 8 for v in offsets) or
                    any(a >= b for a, b in zip(offsets, offsets[1:]))):
                raise RuntimeError("invalid reader packed output")
            if fallback is None:
                if used > shard["capacity"]:
                    raise RuntimeError("reader exceeded its destination")
            elif (not isinstance(fallback, torch.Tensor) or fallback.device.type != "cpu" or
                  fallback.dtype != torch.uint8 or not fallback.is_contiguous() or
                  not fallback.is_shared() or fallback.numel() < used or used <= shard["capacity"]):
                raise RuntimeError("invalid reader fallback slab")
            shard.update(offsets=offsets, groups=groups, used=used, fallback=fallback)

        local = shards[0]
        planes = ((times[i // len(channels)], channels[i % len(channels)])
                  for i in range(local["first"], local["last"]))
        fallback, offsets, groups, used = pack_planes(
            dataset, planes, self._shape[2], self._shape[3], self._library,
            destination=(slab.data_ptr() + local["base"], local["capacity"]))
        record(local, offsets, groups, used, fallback)

        pending = {conn: (time.monotonic() + _TIMEOUT, shard) for conn, shard in pending.items()}
        while pending:
            deadlines = {conn: value[0] for conn, value in pending.items()}
            for conn, message in self._receive(deadlines):
                shard = pending.pop(conn)[1]
                if len(message) != 6 or message[:2] != ("packed", seq):
                    raise RuntimeError("invalid reader packing response")
                _, _, offsets, groups, used, fallback = message
                record(shard, offsets, groups, used, fallback)

        if any(shard["fallback"] is not None for shard in shards):
            final_size = 0
            for shard in shards:
                shard["final_base"] = final_size
                size = _align8(shard["used"])
                if final_size > _MAX_SIZE - size:
                    raise OverflowError("packed fallback exceeds int64 indexing")
                final_size += size
            result = _shared_bytes(final_size)
            for shard in shards:
                fallback = shard["fallback"]
                source = fallback.data_ptr() if fallback is not None else slab.data_ptr() + shard["base"]
                ctypes.memmove(result.data_ptr() + shard["final_base"], source, shard["used"])
            slab = result
        else:
            for shard in shards:
                shard["final_base"] = shard["base"]
        offsets = [shard["final_base"] + offset for shard in shards for offset in shard["offsets"]]
        if any(a >= b for a, b in zip(offsets, offsets[1:])):
            raise RuntimeError("packed plane offsets are not strictly increasing")
        used = shards[-1]["final_base"] + shards[-1]["used"]
        return slab[:used], offsets, max(shard["groups"] for shard in shards)

    def close(self):
        if os.getpid() != self._owner_pid:
            for group in self._groups:
                for conn in group:
                    conn.close()
            if self._connections is not None:
                for conn in self._connections:
                    conn.close()
            self._closed = True
            return
        if self._closed:
            return
        self._closed = True
        self._shutdown.value = 1
        for group in self._groups:
            for conn in group:
                conn.close()

        def join_until(deadline):
            for process in self._processes:
                process.join(max(0.0, deadline - time.monotonic()))

        join_until(time.monotonic() + _SHUTDOWN_GRACE)
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        join_until(time.monotonic() + _SHUTDOWN_GRACE)
        for process in self._processes:
            if process.is_alive():
                process.kill()
        join_until(time.monotonic() + _SHUTDOWN_GRACE)
        alive = [process.pid for process in self._processes if process.is_alive()]
        for process in self._processes:
            if not process.is_alive():
                process.close()
        self._processes.clear()
        if alive:
            raise RuntimeError("could not stop CPU readers: " + ", ".join(map(str, alive)))

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()
