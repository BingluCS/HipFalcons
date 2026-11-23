"""Fork-safe raw-chunk ERA5 loader for a PyTorch ROCm/DCU training process.

The input is a native-endian float32 HDF5 dataset ``fields[T,C,H,W]`` with
chunks ``(1,1,H,W)`` and exactly one filter, Falcons (40024). Every requested
chunk must actually have that filter applied. All routes use HDF5's raw
chunk API and an explicitly supplied native decoder; they do not need the
HDF5 Falcons plugin. In contrast, normal h5py slicing invokes the HDF5 filter
pipeline and requires the plugin to be installed and discoverable.

Call ``start()`` before any CUDA/DCU initialization in the training process.
It probes metadata in a short-lived child and starts persistent fork workers;
the parent never opens an HDF5 handle. The CPU route decodes native Falcons
streams in workers, then pins and transfers each batch in the parent. The GPU
route reads compressed chunks in workers and decodes them directly into a
parent-owned device tensor. The native GPU decoder supports only its own
documented subset (serial float32 LINEAR); errors never fall back to CPU.
The DCU route packs each unique plane in the workers into one shared byte slab,
then asynchronously uploads and launches the packed reconstruction on the
current PyTorch stream. ``library_path`` names the GPU library; CPU decoding
and DCU packing use the CPU-only sibling ``libfalcons_loader_cpu.so`` inside
each worker. A benchmark can save its winning ``'cpu'``, ``'gpu'`` or ``'dcu'``
string and pass it back as ``route``.

Sequential ``cpu_stream`` and ``dcu_stream`` routes schedule one task per
batch and retain only the previous window tail on the device between tasks.
The retained raw data is separate from the returned, possibly normalized
batch. Shuffled stream routes use independent CPU/DCU batches instead.
Sequential ``dcu_stream`` can pack into a bounded parent-created shared IPC
pool; unavailable or undersized slots use an independent shared slab instead.
The parent copies a completed slot into its own pinned staging buffer before
recycling it and starting the asynchronous device transfer.
Consumers must order their own cross-stream use of returned batches, as with
other PyTorch tensors.

Each iteration yields ``(history, future)`` views of one contiguous float32
device batch shaped ``[B, history + future, C_selected, H, W]``. Repeated
times in a batch are read and decoded once, then copied within the batch.
``iter_into`` reuses
a caller-owned device batch; allocate it only after ``start()`` and do not
reuse it concurrently on another stream. Persistent workers support repeated
epochs. Close an abandoned iterator before starting another epoch, and close
the loader, preferably using a context manager, when finished.
"""

import ctypes
import multiprocessing as mp
from numbers import Integral
import os
import traceback
import weakref

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler


_FALCONS_FILTER = 40024
_FLOAT_BYTES = ctypes.sizeof(ctypes.c_float)
_METADATA_TIMEOUT_SECONDS = 120
_IPC_SLOT = "hipfalcons_ipc_slot"
_IPC_CANCELLED = "hipfalcons_ipc_cancelled"


def _identity(value):
    return value


def _dataset_metadata(dataset):
    shape = tuple(int(value) for value in dataset.shape)
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError("ERA5 dataset must have positive shape (T, C, H, W)")
    dtype = dataset.dtype
    if dtype.kind != "f" or dtype.itemsize != 4 or not dtype.isnative:
        raise ValueError("ERA5 dataset must contain native-endian float32 values")
    if dataset.chunks != (1, 1, shape[2], shape[3]):
        raise ValueError("ERA5 chunks must be exactly (1, 1, H, W)")
    properties = dataset.id.get_create_plist()
    try:
        if properties.get_nfilters() != 1 or properties.get_filter(0)[0] != _FALCONS_FILTER:
            raise ValueError("ERA5 dataset must have exactly one Falcons filter (40024)")
    finally:
        properties.close()
    return shape


def _open_dataset(path, name):
    import h5py

    handle = h5py.File(path, "r")
    try:
        dataset = handle[name]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError("HDF5 path does not name a dataset")
        shape = _dataset_metadata(dataset)
        return handle, dataset, shape
    except BaseException:
        handle.close()
        raise


def _probe_metadata(path, name, sender):
    try:
        handle, _, shape = _open_dataset(path, name)
        handle.close()
        sender.send((True, shape))
    except BaseException:
        try:
            sender.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        sender.close()


def _metadata_in_child(context, path, name):
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_probe_metadata, args=(path, name, sender), daemon=True)
    started = False
    try:
        process.start()
        started = True
        sender.close()
        if not receiver.poll(_METADATA_TIMEOUT_SECONDS):
            raise RuntimeError("Timed out probing the HDF5 dataset in a child process")
        try:
            ok, value = receiver.recv()
        except EOFError as error:
            raise RuntimeError("HDF5 metadata probe exited without a result") from error
        process.join(timeout=_METADATA_TIMEOUT_SECONDS)
        if process.is_alive():
            raise RuntimeError("HDF5 metadata probe did not exit")
        if not ok:
            raise RuntimeError("HDF5 metadata validation failed:\n" + value)
        if process.exitcode != 0:
            raise RuntimeError("HDF5 metadata probe exited with code %s" % process.exitcode)
        return tuple(value)
    finally:
        receiver.close()
        sender.close()
        if started:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
            process.join()
            process.close()


class _Codec:
    def __init__(self, path, route):
        self._library = ctypes.CDLL(path)
        if route == "cpu":
            self._library.hf_cpu_f32.argtypes = (
                ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t
            )
            self._library.hf_cpu_f32.restype = ctypes.c_int
        else:
            self._library.hf_gpu_f32.argtypes = (
                ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
                ctypes.c_size_t, ctypes.c_void_p,
            )
            self._library.hf_gpu_f32.restype = ctypes.c_int
        self._library.hf_last_error.argtypes = ()
        self._library.hf_last_error.restype = ctypes.c_char_p

    def _decode(self, function, chunk, destination, count, stream=None):
        # c_char_p holds the bytes object alive and points at its existing buffer.
        source = ctypes.c_char_p(chunk)
        output = ctypes.cast(ctypes.c_void_p(destination), ctypes.POINTER(ctypes.c_float))
        args = (ctypes.cast(source, ctypes.c_void_p), len(chunk), output, count)
        status = function(*args) if stream is None else function(*args, ctypes.c_void_p(stream))
        if status != 0:
            message = self._library.hf_last_error()
            detail = message.decode("utf-8", errors="replace") if message else "no native error detail"
            raise RuntimeError("Falcons decoder failed (%d): %s" % (status, detail))

    def cpu(self, chunk, destination, count):
        self._decode(self._library.hf_cpu_f32, chunk, destination, count)

    def gpu(self, chunk, destination, count, stream):
        self._decode(self._library.hf_gpu_f32, chunk, destination, count, stream)


class _PackedCodec:
    def __init__(self, path):
        self._library = ctypes.CDLL(path)
        self._library.hf_pack_plane.argtypes = (
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        )
        self._library.hf_pack_plane.restype = ctypes.c_int
        self._library.hf_free_plane.argtypes = (ctypes.c_void_p,)
        self._library.hf_free_plane.restype = None
        self._library.hf_last_error.argtypes = ()
        self._library.hf_last_error.restype = ctypes.c_char_p

    def pack(self, chunk, height, width):
        source = ctypes.c_char_p(chunk)
        output = ctypes.c_void_p()
        size = ctypes.c_size_t()
        groups = ctypes.c_size_t()
        status = self._library.hf_pack_plane(
            ctypes.cast(source, ctypes.c_void_p), len(chunk), height, width,
            ctypes.byref(output), ctypes.byref(size), ctypes.byref(groups))
        if status != 0 or not output.value or size.value == 0:
            message = self._library.hf_last_error() if status != 0 else None
            detail = message.decode("utf-8", errors="replace") if message else "invalid packed plane"
            if output.value:
                self.free(output.value)
            raise RuntimeError("hf_pack_plane failed (%d): %s" % (status, detail))
        return output.value, size.value, groups.value

    def free(self, address):
        self._library.hf_free_plane(ctypes.c_void_p(address))


class _PackedLauncher:
    def __init__(self, path):
        self._library = ctypes.CDLL(path)
        self._library.hf_launch_packed.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
        )
        self._library.hf_launch_packed.restype = ctypes.c_int

    def launch(self, payload, offsets, output, planes, elements, max_groups, stream):
        status = self._library.hf_launch_packed(
            ctypes.c_void_p(payload), ctypes.cast(ctypes.c_void_p(offsets), ctypes.POINTER(ctypes.c_int64)),
            ctypes.cast(ctypes.c_void_p(output), ctypes.POINTER(ctypes.c_float)),
            planes, elements, max_groups, ctypes.c_void_p(stream))
        if status != 0:
            raise RuntimeError("hf_launch_packed failed (HIP status %d)" % status)


def _shared_bytes(size):
    # Allocate shared storage directly, avoiding share_memory_'s full-size copy.
    factory = getattr(getattr(torch, "UntypedStorage", None), "_new_shared", None)
    if factory is None:
        slab = torch.empty(size, dtype=torch.uint8, device="cpu").share_memory_()
    else:
        storage = factory(size, device="cpu")
        slab = torch.empty(0, dtype=torch.uint8, device="cpu").set_(storage, 0, (size,), (1,))
    if slab.data_ptr() % 8:
        raise RuntimeError("Shared packed slab is not 8-byte aligned")
    return slab


def _shared_float_frames(count, channels, height, width):
    shape = (count, channels, height, width)
    factory = getattr(getattr(torch, "UntypedStorage", None), "_new_shared", None)
    if factory is None:
        return torch.empty(shape, dtype=torch.float32, device="cpu").share_memory_()
    storage = factory(count * channels * height * width * _FLOAT_BYTES, device="cpu")
    stride = (channels * height * width, height * width, width, 1)
    return torch.empty(0, dtype=torch.float32, device="cpu").set_(storage, 0, shape, stride)


class _SharedBatchPool:
    FREE, WRITING, READY, READING = range(4)
    _LOCK_TIMEOUT = 1.0

    @classmethod
    def create(cls, context, budget, workers, batch_size, channels, height, width):
        # Match fastpack's normal per-plane estimate; unusually large initial
        # batches and escape-heavy planes fall back to a separate slab.
        capacity = (batch_size * channels * (2 * height * width + 16384) + 7) & ~7
        if capacity == 0 or capacity > budget:
            return None
        try:
            stat = os.statvfs("/dev/shm")
            budget = min(budget, max(0, stat.f_bavail * stat.f_frsize - 8 * 1024 * 1024))
        except OSError:
            pass
        slots = min(2 * workers, budget // capacity)
        if slots == 0:
            return None
        try:
            return cls(context, _shared_bytes(slots * capacity), slots, capacity)
        except (MemoryError, OSError, RuntimeError):
            return None

    def __init__(self, context, slab, slots, capacity):
        self.slab = slab
        self.slots = slots
        self.capacity = capacity
        self._lock = context.Lock()
        self._state = context.RawArray(ctypes.c_byte, slots)
        self._generation = context.RawArray(ctypes.c_uint64, slots)
        self._task = context.RawArray(ctypes.c_uint64, slots)
        self._current = context.RawValue(ctypes.c_uint64, 1)
        self._enabled = context.RawValue(ctypes.c_byte, 1)

    def is_current(self, generation):
        return self._current.value == generation

    def advance(self, generation):
        if not self._lock.acquire(timeout=self._LOCK_TIMEOUT):
            self._enabled.value = 0
            self._current.value = generation
            return
        try:
            self._current.value = generation
            for slot in range(self.slots):
                if self._state[slot] == self.READY and self._generation[slot] != generation:
                    self._state[slot] = self.FREE
        finally:
            self._lock.release()

    def claim(self, generation, task):
        if not self._enabled.value or not self.is_current(generation):
            return None
        if not self._lock.acquire(False):
            return None
        try:
            if not self._enabled.value or not self.is_current(generation):
                return None
            for slot in range(self.slots):
                if self._state[slot] == self.FREE:
                    self._state[slot] = self.WRITING
                    self._generation[slot] = generation
                    self._task[slot] = task
                    return slot
            return None
        finally:
            self._lock.release()

    def finish_write(self, slot, generation, task, publish):
        if not self._lock.acquire(timeout=self._LOCK_TIMEOUT):
            self._enabled.value = 0
            return "disabled"
        try:
            if (self._state[slot] != self.WRITING or self._generation[slot] != generation or
                    self._task[slot] != task):
                raise RuntimeError("IPC pool lost ownership of a writing slot")
            if not self.is_current(generation):
                self._state[slot] = self.FREE
                return "cancelled"
            if not self._enabled.value:
                self._state[slot] = self.FREE
                return "disabled"
            self._state[slot] = self.READY if publish else self.FREE
            return "ready" if publish else "released"
        finally:
            self._lock.release()

    def address(self, slot):
        return self.slab.data_ptr() + slot * self.capacity

    def stage(self, slot, generation, task, used):
        if (not isinstance(slot, int) or not 0 <= slot < self.slots or
                not isinstance(used, int) or not 0 < used <= self.capacity):
            raise RuntimeError("Invalid IPC pool descriptor")
        if not self._lock.acquire(timeout=self._LOCK_TIMEOUT):
            self._enabled.value = 0
            raise RuntimeError("IPC pool lock unavailable while staging")
        try:
            if (not self.is_current(generation) or self._state[slot] != self.READY or
                    self._generation[slot] != generation or self._task[slot] != task):
                raise RuntimeError("Stale or invalid IPC pool descriptor")
            self._state[slot] = self.READING
        finally:
            self._lock.release()
        try:
            staging = torch.empty(used, dtype=torch.uint8, device="cpu", pin_memory=True)
            ctypes.memmove(staging.data_ptr(), self.address(slot), used)
            return staging
        finally:
            if self._lock.acquire(timeout=self._LOCK_TIMEOUT):
                try:
                    if (self._state[slot] == self.READING and self._generation[slot] == generation and
                            self._task[slot] == task):
                        self._state[slot] = self.FREE
                finally:
                    self._lock.release()
            else:
                self._enabled.value = 0


class _SlidingBatchSampler:
    """Add the exact entering times to each otherwise unchanged batch task."""
    def __init__(self, batches, frames, pool=None):
        self.batches = batches
        self.frames = frames
        self.pool = pool
        self.generation = 1

    def __iter__(self):
        first = True
        generation = self.generation
        for indices in self.batches:
            start = indices[0] if first else indices[0] + self.frames - 1
            times = tuple(range(start, indices[-1] + self.frames))
            task = (tuple(indices), times, first)
            yield (*task, generation) if self.pool is not None else task
            first = False

    def __len__(self):
        return len(self.batches)


class _ChunkBatches(Dataset):
    def __init__(self, path, name, shape, channels, frames, route, library_path, pool=None):
        self.path = path
        self.name = name
        self.shape = shape
        self.channels = channels
        self.frames = frames
        self.route = route
        self.library_path = library_path
        self._pool = pool
        self._handle = None
        self._dataset = None
        self._pid = None
        self._codec = None

    def __len__(self):
        return self.shape[0] - self.frames + 1

    def _get_dataset(self):
        pid = os.getpid()
        if self._handle is not None and self._pid != pid:
            raise RuntimeError("Refusing an HDF5 handle inherited across fork")
        if self._handle is None:
            handle, dataset, shape = _open_dataset(self.path, self.name)
            if shape != self.shape:
                handle.close()
                raise RuntimeError("HDF5 dataset changed after metadata validation")
            self._handle, self._dataset, self._pid = handle, dataset, pid
        return self._dataset

    def __getitem__(self, indices):
        if (self.route == "dcu_stream" and self._pool is not None and len(indices) == 4 and
                not self._pool.is_current(indices[3])):
            starts, times, first, generation = indices
            return tuple(starts), first, len(times), (_IPC_CANCELLED,), generation
        dataset = self._get_dataset()
        channels = len(self.channels)
        height, width = self.shape[2:]
        if self.route in ("cpu_stream", "dcu_stream"):
            return self._stream_batch(dataset, indices, height, width)
        if self.route == "dcu":
            return self._packed_batch(dataset, indices, height, width)
        plane = height * width
        frame_bytes = channels * plane * _FLOAT_BYTES
        if self.route == "cpu":
            if self._codec is None:
                self._codec = _Codec(self.library_path, "cpu")
            batch = torch.empty((len(indices), self.frames, channels, height, width),
                                dtype=torch.float32).share_memory_()
            address = batch.data_ptr()
        else:
            batch = []
            time_map = []

        seen = {}
        frame = 0
        for index in indices:
            if index < 0 or index >= len(self):
                raise IndexError("ERA5 window index out of range")
            for time in range(index, index + self.frames):
                previous = seen.get(time)
                if previous is not None:
                    if self.route == "cpu":
                        ctypes.memmove(address + frame * frame_bytes,
                                       address + previous * frame_bytes, frame_bytes)
                    else:
                        time_map.append(previous)
                    frame += 1
                    continue
                seen[time] = frame if self.route == "cpu" else len(seen)
                if self.route == "gpu":
                    time_map.append(seen[time])
                for channel_index, channel in enumerate(self.channels):
                    try:
                        mask, chunk = dataset.id.read_direct_chunk((time, channel, 0, 0))
                    except Exception as error:
                        raise RuntimeError("Could not read HDF5 chunk at time %d, channel %d"
                                           % (time, channel)) from error
                    if mask != 0:
                        raise ValueError("Falcons filter was skipped for chunk at time %d, channel %d (mask %d)"
                                         % (time, channel, mask))
                    if not chunk:
                        raise ValueError("Empty Falcons chunk at time %d, channel %d" % (time, channel))
                    if self.route == "cpu":
                        offset = (frame * channels + channel_index) * plane * _FLOAT_BYTES
                        self._codec.cpu(chunk, address + offset, plane)
                    else:
                        batch.append(chunk)
                frame += 1
        return batch if self.route == "cpu" else (len(indices), batch, time_map)

    def _stream_batch(self, dataset, task, height, width):
        if len(task) == 4:
            indices, times, first, generation = task
        else:
            indices, times, first = task
            generation = None
        if (not indices or any(index < 0 or index >= len(self) for index in indices) or
                any(index != indices[0] + i for i, index in enumerate(indices))):
            raise ValueError("Invalid sequential ERA5 batch indices")
        start = indices[0] if first else indices[0] + self.frames - 1
        if tuple(times) != tuple(range(start, indices[-1] + self.frames)):
            raise ValueError("Invalid entering times for sequential ERA5 batch")
        if self.route == "dcu_stream":
            if __package__:
                from .fastpack import pack_times
            else:
                from .fastpack import pack_times
            payload = self._pack_stream_times(pack_times, dataset, indices[0], times,
                                              height, width, generation)
        else:
            if self._codec is None:
                self._codec = _Codec(self.library_path, "cpu")
            channels = len(self.channels)
            plane = height * width
            payload = _shared_float_frames(len(times), channels, height, width)
            address = payload.data_ptr()
            for frame, time in enumerate(times):
                for channel_index, channel in enumerate(self.channels):
                    try:
                        mask, chunk = dataset.id.read_direct_chunk((time, channel, 0, 0))
                    except Exception as error:
                        raise RuntimeError("Could not read HDF5 chunk at time %d, channel %d"
                                           % (time, channel)) from error
                    if mask != 0:
                        raise ValueError("Falcons filter was skipped for chunk at time %d, channel %d (mask %d)"
                                         % (time, channel, mask))
                    if not chunk:
                        raise ValueError("Empty Falcons chunk at time %d, channel %d" % (time, channel))
                    offset = (frame * channels + channel_index) * plane * _FLOAT_BYTES
                    self._codec.cpu(chunk, address + offset, plane)
        result = (tuple(indices), first, len(times), payload)
        return (*result, generation) if generation is not None else result

    def _pack_stream_times(self, pack_times, dataset, task, times, height, width, generation):
        pool = self._pool
        if pool is None:
            return pack_times(dataset, times, self.channels, height, width, self.library_path)
        if not pool.is_current(generation):
            return (_IPC_CANCELLED,)
        estimate = len(times) * len(self.channels) * (2 * height * width + 16384)
        slot = pool.claim(generation, task) if estimate <= pool.capacity else None
        if slot is None:
            if not pool.is_current(generation):
                return (_IPC_CANCELLED,)
            payload = pack_times(dataset, times, self.channels, height, width, self.library_path)
            return payload if pool.is_current(generation) else (_IPC_CANCELLED,)

        owned = True
        try:
            slab, offsets, groups, used = pack_times(
                dataset, times, self.channels, height, width, self.library_path,
                destination=(pool.address(slot), pool.capacity))
            state = pool.finish_write(slot, generation, task, slab is None)
            owned = False
            if state == "cancelled" or not pool.is_current(generation):
                return (_IPC_CANCELLED,)
            if slab is not None:
                return slab, offsets, groups
            if state == "ready":
                return _IPC_SLOT, slot, generation, task, used, offsets, groups
            if state == "disabled":
                # A retired pool never reuses this storage, even if its lock
                # was abandoned; preserve this batch in the normal slab path.
                slab = _shared_bytes(used)
                ctypes.memmove(slab.data_ptr(), pool.address(slot), used)
                return slab, offsets, groups
            raise RuntimeError("IPC pool did not publish a packed batch")
        finally:
            if owned:
                pool.finish_write(slot, generation, task, False)

    def _packed_batch(self, dataset, indices, height, width):
        if self._codec is None:
            self._codec = _PackedCodec(self.library_path)
        packed = []
        offsets = []
        time_map = []
        seen = {}
        size = 0
        max_groups = 0
        try:
            for index in indices:
                if index < 0 or index >= len(self):
                    raise IndexError("ERA5 window index out of range")
                for time in range(index, index + self.frames):
                    previous = seen.get(time)
                    if previous is not None:
                        time_map.append(previous)
                        continue
                    seen[time] = len(seen)
                    time_map.append(seen[time])
                    for channel in self.channels:
                        try:
                            mask, chunk = dataset.id.read_direct_chunk((time, channel, 0, 0))
                        except Exception as error:
                            raise RuntimeError("Could not read HDF5 chunk at time %d, channel %d"
                                               % (time, channel)) from error
                        if mask != 0:
                            raise ValueError("Falcons filter was skipped for chunk at time %d, channel %d (mask %d)"
                                             % (time, channel, mask))
                        if not chunk:
                            raise ValueError("Empty Falcons chunk at time %d, channel %d" % (time, channel))
                        address, length, groups = self._codec.pack(chunk, height, width)
                        try:
                            packed.append((address, length))
                        except BaseException:
                            self._codec.free(address)
                            raise
                        offset = (size + 7) & ~7
                        if offset > (1 << 63) - 1 or length > (1 << 63) - 1 - offset:
                            raise OverflowError("Packed batch exceeds int64 indexing")
                        offsets.append(offset)
                        size = offset + length
                        max_groups = max(max_groups, groups)
            slab = _shared_bytes(size)
            destination = slab.data_ptr()
            for index, ((address, length), offset) in enumerate(zip(packed, offsets)):
                ctypes.memmove(destination + offset, address, length)
                self._codec.free(address)
                packed[index] = (0, 0)
            return len(indices), slab, offsets, time_map, max_groups
        finally:
            for address, _ in packed:
                if address:
                    self._codec.free(address)

    def close(self):
        if self._handle is not None and self._pid == os.getpid():
            self._handle.close()
        self._handle = None
        self._dataset = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _EpochIterator:
    def __init__(self, owner, iterator, destination):
        self._owner = owner
        self._iterator = iterator
        self._destination = destination
        self._tail = None
        self._tail_event = None
        self._tail_stream = None
        self._next_start = None
        self._pool_generation = owner._pool_generation if owner._pool is not None else None

    def __iter__(self):
        return self

    def __next__(self):
        if self._owner is None:
            raise StopIteration
        try:
            data = next(self._iterator)
            if self._owner._streaming:
                batch = self._owner._to_device_stream(data, self._destination, self)
            else:
                batch = self._owner._to_device(data, self._destination)
            split = self._owner.history
            return batch[:, :split], batch[:, split:]
        except BaseException:
            self.close()
            raise

    def close(self):
        owner, self._owner = self._owner, None
        self._iterator = None
        self._destination = None
        self._tail = None
        self._tail_event = None
        self._tail_stream = None
        self._next_start = None
        if owner is not None:
            owner._release_epoch(self)

    def __del__(self):
        self.close()


class ERA5Loader:
    """Load overlapping ERA5 windows into a contiguous CUDA-namespace tensor.

    Args are explicit: ``route`` is ``'cpu'``, ``'gpu'``, ``'dcu'``,
    ``'cpu_stream'`` or ``'dcu_stream'`` and may be a saved benchmark winner;
    ``library_path`` names the HIP library and CPU workers load its
    ``libfalcons_loader_cpu.so`` sibling. ``device`` is a PyTorch
    CUDA-namespace device such as ``'cuda:0'`` on ROCm; ``num_workers`` must be
    positive. ``shared_pool_bytes`` bounds the sequential ``dcu_stream`` IPC
    pool; zero disables it. ``channels`` is an optional sequence of unique source-channel
    indices, in output order. ``mean`` and ``std`` are either both absent or
    CPU scalars/per-selected-channel values in that order. Normalization is
    in-place on the GPU. ``shuffle`` shuffles window starts, not frames
    within a window. ``shape`` is the input shape and ``channels`` is the
    selected index tuple after ``start()``.

    ``start()`` must run before CUDA initialization. ``iter(loader)`` also
    calls it, but an output passed to ``iter_into`` must be allocated after an
    explicit ``start()``. Only one epoch iterator may be active at a time.
    ``iter_into`` overwrites its destination on every batch; the final batch
    may be a smaller leading slice. After ``close()``, the loader is unusable.
    """

    def __init__(self, path, *, library_path, route, device, num_workers,
                 batch_size, history, future, dataset="fields", mean=None,
                 std=None, shuffle=False, drop_last=False, channels=None,
                 shared_pool_bytes=64 * 1024 * 1024):
        if route not in ("cpu", "gpu", "dcu", "cpu_stream", "dcu_stream"):
            raise ValueError("route must be 'cpu', 'gpu', 'dcu', 'cpu_stream' or 'dcu_stream'")
        for name, value in (("num_workers", num_workers), ("batch_size", batch_size),
                            ("history", history), ("future", future)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if (not isinstance(shared_pool_bytes, Integral) or isinstance(shared_pool_bytes, bool) or
                shared_pool_bytes < 0 or shared_pool_bytes > (1 << 63) - 1):
            raise ValueError("shared_pool_bytes must be a nonnegative int64 value")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("dataset must be a nonempty HDF5 path")
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be provided together")
        if channels is not None:
            channels = tuple(channels)
            if (not channels or any(not isinstance(c, Integral) or isinstance(c, bool) or c < 0
                                    for c in channels) or len(set(channels)) != len(channels)):
                raise ValueError("channels must be nonempty, unique, nonnegative integer indices")
            channels = tuple(int(c) for c in channels)
        for value in (mean, std):
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                raise ValueError("normalization values must be on the CPU before start()")
        self.path = os.path.abspath(os.fsdecode(path))
        self.library_path = os.path.abspath(os.fsdecode(library_path))
        if not os.path.isfile(self.library_path):
            raise ValueError("native codec library does not exist: " + self.library_path)
        self.cpu_library_path = os.path.join(os.path.dirname(self.library_path),
                                             "libfalcons_loader_cpu.so")
        if route in ("cpu", "dcu", "cpu_stream", "dcu_stream") and not os.path.isfile(self.cpu_library_path):
            raise ValueError("CPU-only codec library does not exist: " + self.cpu_library_path)
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("device must be a PyTorch CUDA/ROCm device")
        self.route = route
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.history = history
        self.future = future
        self.frames = history + future
        self.dataset = dataset
        self.mean = mean
        self.std = std
        self.shuffle = bool(shuffle)
        self._streaming = route in ("cpu_stream", "dcu_stream") and not self.shuffle
        self.drop_last = bool(drop_last)
        self.shared_pool_bytes = int(shared_pool_bytes)
        self.channels = channels
        self.channel_count = None
        self.shape = None
        self._owner_pid = os.getpid()
        self._loader = None
        self._pending = None
        self._active = None
        self._codec = None
        self._normalization = None
        self._pool = None
        self._pool_generation = 1
        self._stream_sampler = None
        self._closed = False

    def _check_owner(self):
        if os.getpid() != self._owner_pid:
            raise RuntimeError("ERA5Loader may only be used in its creating process")
        if self._closed:
            raise RuntimeError("ERA5Loader is closed")

    def start(self):
        """Start the metadata child and persistent fork workers, before CUDA."""
        self._check_owner()
        if self._loader is not None:
            return self
        if torch.cuda.is_initialized():
            raise RuntimeError("Start ERA5Loader workers before initializing CUDA/DCU")
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("ERA5Loader requires the fork multiprocessing start method")
        context = mp.get_context("fork")
        shape = _metadata_in_child(context, self.path, self.dataset)
        if shape[0] < self.frames:
            raise ValueError("ERA5 time dimension is shorter than history + future")
        channels = self.channels if self.channels is not None else tuple(range(shape[1]))
        if any(channel >= shape[1] for channel in channels):
            raise ValueError("channel index exceeds the ERA5 channel dimension")
        worker_route = self.route
        if self.shuffle and worker_route in ("cpu_stream", "dcu_stream"):
            worker_route = "cpu" if worker_route == "cpu_stream" else "dcu"
        pool = None
        if self._streaming and worker_route == "dcu_stream" and self.shared_pool_bytes:
            pool = _SharedBatchPool.create(context, self.shared_pool_bytes, self.num_workers,
                                           self.batch_size, len(channels), shape[2], shape[3])
        samples = _ChunkBatches(self.path, self.dataset, shape, channels, self.frames,
                                worker_route, self.cpu_library_path, pool)
        sampler = RandomSampler(samples) if self.shuffle else SequentialSampler(samples)
        batches = BatchSampler(sampler, self.batch_size, self.drop_last)
        if self._streaming:
            batches = _SlidingBatchSampler(batches, self.frames, pool)
        loader = DataLoader(samples, batch_size=None, sampler=batches,
                            num_workers=self.num_workers, multiprocessing_context=context,
                            collate_fn=_identity, persistent_workers=True, pin_memory=False)
        # DataLoader forks only in iter(). Pinning and all CUDA calls are later.
        if torch.cuda.is_initialized():
            raise RuntimeError("CUDA/DCU initialized before the workers were started")
        try:
            iterator = iter(loader)
        except BaseException:
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
            raise
        self.shape, self.channels, self.channel_count = shape, channels, len(channels)
        self._loader, self._pending = loader, iterator
        self._pool = pool
        self._stream_sampler = batches if self._streaming else None
        return self

    def __len__(self):
        self.start()
        return len(self._loader)

    def _new_epoch(self, destination):
        self.start()
        if self._active is not None and self._active() is not None:
            raise RuntimeError("Close or exhaust the current ERA5 epoch before starting another")
        if destination is not None:
            self._validate_destination(destination)
        if self._pending is not None:
            iterator, self._pending = self._pending, None
        else:
            iterator = iter(self._loader)
        epoch = _EpochIterator(self, iterator, destination)
        self._active = weakref.ref(epoch)
        return epoch

    def __iter__(self):
        return self._new_epoch(None)

    def iter_into(self, destination):
        """Iterate using a caller-owned contiguous device batch as output."""
        return self._new_epoch(destination)

    def _release_epoch(self, epoch):
        if self._active is not None and self._active() is epoch:
            self._active = None
            self._advance_pool()

    def _advance_pool(self):
        if self._pool is not None:
            self._pool_generation += 1
            self._pool.advance(self._pool_generation)
            self._stream_sampler.generation = self._pool_generation

    def _validate_destination(self, destination):
        expected = (self.batch_size, self.frames, self.channel_count, *self.shape[2:])
        if not isinstance(destination, torch.Tensor) or destination.dtype != torch.float32:
            raise ValueError("destination must be a float32 torch tensor")
        if tuple(destination.shape) != expected or not destination.is_contiguous():
            raise ValueError("destination must be contiguous with shape %s" % (expected,))
        index = self.device.index
        if destination.device.type != "cuda" or (
            index is not None and destination.device.index != index
        ):
            raise ValueError("destination must be on the requested CUDA/ROCm device")
        if index is None and destination.device.index != torch.cuda.current_device():
            raise ValueError("destination must be on the current CUDA/ROCm device")

    def _stats(self):
        if self.mean is None:
            return None
        if self._normalization is None:
            values = []
            for name, value in (("mean", self.mean), ("std", self.std)):
                stat = torch.as_tensor(value, dtype=torch.float32, device="cpu").detach().reshape(-1)
                if stat.numel() not in (1, self.channel_count):
                    raise ValueError("%s must be scalar or have one value per channel" % name)
                if not bool(torch.isfinite(stat).all()) or (name == "std" and not bool((stat > 0).all())):
                    raise ValueError("normalization values must be finite and std must be positive")
                values.append(stat.reshape(1, 1, -1, 1, 1).to(self.device))
            self._normalization = tuple(values)
        return self._normalization

    def _to_device(self, data, destination):
        self._check_owner()
        if destination is not None:
            self._validate_destination(destination)
        with torch.cuda.device(self.device):
            if self.route in ("cpu", "cpu_stream"):
                if (not isinstance(data, torch.Tensor) or data.device.type != "cpu" or
                        data.dtype != torch.float32 or not data.is_contiguous() or
                        len(data.shape) != 5 or not 0 < data.shape[0] <= self.batch_size or
                        tuple(data.shape[1:]) != (self.frames, self.channel_count, *self.shape[2:])):
                    raise RuntimeError("Invalid decoded batch received from a worker")
                pinned = data.pin_memory()
                if destination is None:
                    batch = pinned.to(self.device, non_blocking=True)
                else:
                    batch = destination[:pinned.shape[0]]
                    batch.copy_(pinned, non_blocking=True)
            elif self.route == "gpu":
                batch_size, chunks, time_map = data
                if (not isinstance(batch_size, int) or not 0 < batch_size <= self.batch_size or
                        len(time_map) != batch_size * self.frames or
                        not chunks or len(chunks) % self.channel_count):
                    raise RuntimeError("Invalid compressed batch received from a worker")
                unique_times = len(chunks) // self.channel_count
                if (any(not isinstance(index, int) or not 0 <= index < unique_times
                        for index in time_map) or len(set(time_map)) != unique_times):
                    raise RuntimeError("Invalid compressed batch time mapping")
                shape = (batch_size, self.frames, self.channel_count, *self.shape[2:])
                batch = (torch.empty(shape, dtype=torch.float32, device=self.device)
                         if destination is None else destination[:batch_size])
                if self._codec is None:
                    self._codec = _Codec(self.library_path, "gpu")
                stream = torch.cuda.current_stream(self.device).cuda_stream
                plane = self.shape[2] * self.shape[3]
                address = batch.data_ptr()
                frame_views = batch.view(batch_size * self.frames, self.channel_count, *self.shape[2:])
                first_frame = [-1] * unique_times
                for frame, unique in enumerate(time_map):
                    if first_frame[unique] >= 0:
                        frame_views[frame].copy_(frame_views[first_frame[unique]], non_blocking=True)
                    else:
                        first_frame[unique] = frame
                        for channel in range(self.channel_count):
                            chunk = chunks[unique * self.channel_count + channel]
                            offset = (frame * self.channel_count + channel) * plane * _FLOAT_BYTES
                            self._codec.gpu(chunk, address + offset, plane, stream)
            else:
                batch = self._to_device_packed(data, destination)
            stats = self._stats()
            if stats is not None:
                batch.sub_(stats[0]).div_(stats[1])
            return batch

    def _to_device_stream(self, data, destination, epoch):
        self._check_owner()
        if destination is not None:
            self._validate_destination(destination)
        if len(data) == 5:
            indices, first, entering, payload, generation = data
            if (generation != epoch._pool_generation or self._pool is None or
                    not self._pool.is_current(generation)):
                raise RuntimeError("Stale streaming batch generation")
        else:
            indices, first, entering, payload = data
            generation = None
        if isinstance(payload, tuple) and len(payload) == 1 and payload[0] == _IPC_CANCELLED:
            raise RuntimeError("Cancelled streaming batch")
        count = len(indices)
        if (not isinstance(first, bool) or not 0 < count <= self.batch_size or
                any(not isinstance(index, int) or index != indices[0] + i
                    for i, index in enumerate(indices)) or
                first != (epoch._tail is None) or
                (not first and indices[0] != epoch._next_start) or
                entering != count + (self.frames - 1 if first else 0)):
            raise RuntimeError("Invalid sequential streaming batch")

        channels, height, width = self.channel_count, self.shape[2], self.shape[3]
        tail_count = self.frames - 1
        unique_count = count + tail_count
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            if not first:
                if epoch._tail_stream != stream.cuda_stream:
                    stream.wait_event(epoch._tail_event)
                epoch._tail.record_stream(stream)

            shape = (count, self.frames, channels, height, width)
            batch = (torch.empty(shape, dtype=torch.float32, device=self.device)
                     if destination is None else destination[:count])
            batch.record_stream(stream)
            if count == 1:
                if first:
                    target = batch[0]
                else:
                    batch[0, :tail_count].copy_(epoch._tail, non_blocking=True)
                    target = batch[0, tail_count:]
            else:
                unique = torch.empty((unique_count, channels, height, width),
                                     dtype=torch.float32, device=self.device)
                unique.record_stream(stream)
                if first:
                    target = unique
                else:
                    unique[:tail_count].copy_(epoch._tail, non_blocking=True)
                    target = unique[tail_count:]

            if self.route == "cpu_stream":
                if (not isinstance(payload, torch.Tensor) or payload.device.type != "cpu" or
                        payload.dtype != torch.float32 or not payload.is_contiguous() or
                        tuple(payload.shape) != (entering, channels, height, width)):
                    raise RuntimeError("Invalid streaming CPU batch received from a worker")
                target.copy_(payload.pin_memory(), non_blocking=True)
            else:
                self._launch_stream_packed(payload, target, entering, stream, generation, indices[0])

            if count == 1:
                next_tail = batch[0, 1:].clone()
            else:
                for index in range(count):
                    batch[index].copy_(unique[index:index + self.frames], non_blocking=True)
                next_tail = torch.empty((tail_count, channels, height, width),
                                        dtype=torch.float32, device=self.device)
                next_tail.copy_(unique[-tail_count:], non_blocking=True)
            next_tail.record_stream(stream)
            event = torch.cuda.Event()
            event.record(stream)
            epoch._tail = next_tail
            epoch._tail_event = event
            epoch._tail_stream = stream.cuda_stream
            epoch._next_start = indices[-1] + 1

            stats = self._stats()
            if stats is not None:
                batch.sub_(stats[0]).div_(stats[1])
            return batch

    def _launch_stream_packed(self, payload, target, entering, stream, generation, task):
        pooled = (isinstance(payload, tuple) and len(payload) == 7 and
                  isinstance(payload[0], str) and payload[0] == _IPC_SLOT)
        if pooled:
            _, slot, slot_generation, slot_task, used, offsets, max_groups = payload
            if (self._pool is None or slot_generation != generation or slot_task != task or
                    not isinstance(used, int) or not 0 < used <= self._pool.capacity):
                raise RuntimeError("Invalid IPC pool batch identity")
        else:
            slab, offsets, max_groups = payload
            if (not isinstance(slab, torch.Tensor) or slab.dtype != torch.uint8 or
                    slab.device.type != "cpu" or len(slab.shape) != 1 or not slab.is_contiguous()):
                raise RuntimeError("Invalid streaming packed batch received from a worker")
            used = slab.numel()
        if (len(offsets) != entering * self.channel_count or
                not isinstance(max_groups, int) or max_groups < 0 or
                any(not isinstance(offset, int) or offset < 0 or offset % 8 or offset >= used
                    for offset in offsets) or
                any(left >= right for left, right in zip(offsets, offsets[1:]))):
            raise RuntimeError("Invalid streaming packed batch received from a worker")
        pinned = self._pool.stage(slot, generation, task, used) if pooled else slab.pin_memory()
        if generation is not None and (self._pool is None or not self._pool.is_current(generation)):
            raise RuntimeError("Streaming batch was cancelled during staging")
        device_bytes = pinned.to(self.device, non_blocking=True)
        device_offsets = torch.tensor(offsets, dtype=torch.int64, device="cpu").pin_memory().to(
            self.device, non_blocking=True)
        if self._codec is None:
            self._codec = _PackedLauncher(self.library_path)
        for tensor in (device_bytes, device_offsets, target):
            tensor.record_stream(stream)
        self._codec.launch(device_bytes.data_ptr(), device_offsets.data_ptr(), target.data_ptr(),
                           len(offsets), self.shape[2] * self.shape[3], max_groups, stream.cuda_stream)

    def _to_device_packed(self, data, destination):
        batch_size, slab, offsets, time_map, max_groups = data
        if (not isinstance(batch_size, int) or not 0 < batch_size <= self.batch_size or
                not isinstance(slab, torch.Tensor) or slab.dtype != torch.uint8 or
                slab.device.type != "cpu" or len(slab.shape) != 1 or not slab.is_contiguous() or
                not offsets or len(offsets) % self.channel_count or
                len(time_map) != batch_size * self.frames or
                not isinstance(max_groups, int) or max_groups < 0):
            raise RuntimeError("Invalid packed batch received from a worker")
        if (any(not isinstance(offset, int) or offset < 0 or offset % 8 or offset >= slab.numel()
                for offset in offsets) or
                any(left >= right for left, right in zip(offsets, offsets[1:]))):
            raise RuntimeError("Invalid packed plane offsets")
        unique_times = len(offsets) // self.channel_count
        if (any(not isinstance(index, int) or not 0 <= index < unique_times for index in time_map) or
                len(set(time_map)) != unique_times):
            raise RuntimeError("Invalid packed batch time mapping")

        shape = (batch_size, self.frames, self.channel_count, *self.shape[2:])
        batch = (torch.empty(shape, dtype=torch.float32, device=self.device)
                 if destination is None else destination[:batch_size])
        direct = unique_times == len(time_map) and all(i == index for i, index in enumerate(time_map))
        output = batch if direct else torch.empty(
            (unique_times, self.channel_count, *self.shape[2:]), dtype=torch.float32, device=self.device)
        pinned = slab.pin_memory()
        pinned_offsets = torch.tensor(offsets, dtype=torch.int64, device="cpu").pin_memory()
        device_bytes = pinned.to(self.device, non_blocking=True)
        device_offsets = pinned_offsets.to(self.device, non_blocking=True)
        if self._codec is None:
            self._codec = _PackedLauncher(self.library_path)
        stream = torch.cuda.current_stream(self.device)
        for tensor in (device_bytes, device_offsets, output):
            tensor.record_stream(stream)
        if not direct:
            batch.record_stream(stream)
        self._codec.launch(device_bytes.data_ptr(), device_offsets.data_ptr(), output.data_ptr(),
                           len(offsets), self.shape[2] * self.shape[3], max_groups, stream.cuda_stream)
        if not direct:
            frames = batch.view(batch_size * self.frames, self.channel_count, *self.shape[2:])
            for frame, unique in enumerate(time_map):
                frames[frame].copy_(output[unique], non_blocking=True)
        return batch

    def close(self):
        if os.getpid() != self._owner_pid or self._closed:
            return
        self._closed = True
        active = self._active() if self._active is not None else None
        if active is not None:
            active.close()
        else:
            self._advance_pool()
        self._active = None
        loader, self._loader = self._loader, None
        self._pending = None
        if loader is not None:
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
                loader._iterator = None
        self._pool = None
        self._stream_sampler = None
        self._codec = None
        self._normalization = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
