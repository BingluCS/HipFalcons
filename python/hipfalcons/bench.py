#!/usr/bin/env python3
"""Measure the real ERA5Loader-to-device path on a small synthetic HDF5 file.

Example (Linux with a ROCm/HIP PyTorch build and the native codec library):
    python -m hipfalcons.bench --library /absolute/path/libhipfalcons_loader.so
    python -m hipfalcons.bench --library /absolute/path/libhipfalcons_loader.so --shape 192 320

Every route/window gets a new process, starts fork workers before GPU
initialization, and checks every output against the same CPU-decoded oracle.
The temporary fixture and oracle are removed on exit.
The first batch is excluded from steady timing. The first-plus-steady average
combines their separately timed durations and excludes intervening validation.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import tempfile
import time


from . import _route_readers
from ._hdf5 import FALCONS_FILTER, FLOAT_PTR, create_fields, encode_plane, native_error, native_libraries


CHANNELS = 4
HISTORY = 1
FUTURE = 1
FRAMES = HISTORY + FUTURE
MAX_BYTES = 512 * 1024 * 1024
RESULT_PREFIX = "__HIPFALCONS_BENCH_RESULT__="


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--library", required=True,
                        help="GPU codec library; libfalcons_loader_cpu.so must be in the same directory")
    parser.add_argument("--encoder-library", help="optional compressor library; default is the CPU-only sibling")
    parser.add_argument("--shape", nargs=2, type=int, metavar=("H", "W"), default=(128, 256))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--batches", type=int, default=6, help="timed batches per window, after the first batch")
    parser.add_argument("--windows", type=int, default=3, help="independent windows per route, at most 3")
    parser.add_argument("--baseline", choices=("cpu", "dcu", "cpu_stream"), default="cpu")
    parser.add_argument("--candidate", choices=("gpu", "dcu", "dcu_stream"), default="dcu_stream")
    parser.add_argument("--extra", choices=("cpu_stream",), help="also measure this control route")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader fork workers per route")
    parser.add_argument("--reader-processes", type=int, default=1,
                        help="total readers per worker for cpu_stream, dcu and dcu_stream (default: 1; 0 aliases 1)")
    parser.add_argument("--gate-prefetch", action="store_true",
                        help="allow only the warmup batch to be prefetched before timing starts")
    parser.add_argument("--limit-mib", type=int, default=MAX_BYTES // (1024 * 1024),
                        help="maximum MiB for the uncompressed fixture and retained outputs, each (default: 512)")
    parser.add_argument("--abs-error", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--device", default="cuda:0", help="PyTorch HIP device")
    parser.add_argument("--temp-dir", help="parent directory for temporary fixture and oracle")
    parser.add_argument("--report", help="write a JSON report after all comparisons pass")
    parser.add_argument("--timeout", type=float, default=600, help="seconds per route subprocess")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_builder", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_work-dir", help=argparse.SUPPRESS)
    parser.add_argument("--_route", choices=("cpu", "gpu", "dcu", "cpu_stream", "dcu_stream"), help=argparse.SUPPRESS)
    parser.add_argument("--_fixture", help=argparse.SUPPRESS)
    parser.add_argument("--_oracle", help=argparse.SUPPRESS)
    args = parser.parse_args()
    h, w = args.shape
    if not (2 <= h <= 4096 and 2 <= w <= 4096):
        parser.error("both dimensions must be between 2 and 4096")
    if not (1 <= args.batch_size <= 16 and 1 <= args.batches <= 4096 and 1 <= args.windows <= 3):
        parser.error("batch-size must be 1..16, batches 1..4096, and windows 1..3")
    if args.baseline == args.candidate:
        parser.error("baseline and candidate must differ")
    if args.extra is not None and args.extra in (args.baseline, args.candidate):
        parser.error("extra must differ from baseline and candidate")
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.reader_processes < 0:
        parser.error("reader-processes must be nonnegative")
    if args.limit_mib < 1:
        parser.error("limit-mib must be positive")
    if not (math.isfinite(args.abs_error) and args.abs_error > 0):
        parser.error("abs-error must be finite and positive")
    if not (math.isfinite(args.timeout) and args.timeout > 0):
        parser.error("timeout must be finite and positive")
    if not (0 <= args.seed < 2**32):
        parser.error("seed must be an unsigned 32-bit integer")
    if args._worker and not (args._route and args._fixture and args._oracle):
        parser.error("incomplete internal worker arguments")
    if args._builder and not args._work_dir:
        parser.error("incomplete internal fixture-builder arguments")
    if args._worker and args._builder:
        parser.error("conflicting internal modes")
    return args


def fixture(args, root):
    import h5py
    import numpy as np

    h, w = args.shape
    samples = args.batch_size * (args.batches + 1)
    times = samples + FRAMES - 1
    shape = (times, CHANNELS, h, w)
    raw_bytes = math.prod(shape) * 4
    output_bytes = samples * FRAMES * CHANNELS * h * w * 4
    limit_bytes = args.limit_mib * 1024 * 1024
    if raw_bytes > limit_bytes or output_bytes > limit_bytes:
        raise ValueError(f"fixture or retained device output would exceed the {args.limit_mib} MiB per-item limit")
    cpu_library = Path(args.library).resolve().parent / "libfalcons_loader_cpu.so"
    encoder, decoder = native_libraries(Path(args.encoder_library).resolve() if args.encoder_library else cpu_library,
                                        cpu_library)
    path = root / "fields.h5"
    oracle_path = root / "cpu_oracle.npy"
    oracle = np.lib.format.open_memmap(oracle_path, mode="w+", dtype="<f4", shape=shape)
    rng = np.random.default_rng(args.seed)
    y = np.arange(h, dtype=np.float32)[:, None] / np.float32(h)
    x = np.arange(w, dtype=np.float32)[None, :] / np.float32(w)
    original_hash = hashlib.sha256()
    max_error = 0.0
    compressed_bytes = 0

    def make_plane(t, c):
        noise = rng.standard_normal((h, w), dtype=np.float32) * np.float32(0.005)
        return np.ascontiguousarray(
            (c - 1.5) * 4.0 + 0.025 * t
            + (1.0 + 0.15 * c) * np.sin(6.2831853 * (x + 0.007 * t))
            + 0.6 * np.cos(9.424778 * (y - 0.005 * t)) + noise,
            dtype="<f4",
        )

    first_plane = make_plane(0, 0)
    first_payload = encode_plane(encoder, first_plane, args.abs_error)
    with h5py.File(path, "w") as f:
        dataset = create_fields(f, shape, first_payload)
        for t in range(times):
            for c in range(CHANNELS):
                plane = first_plane if (t, c) == (0, 0) else make_plane(t, c)
                original_hash.update(plane.tobytes(order="C"))
                payload = first_payload if (t, c) == (0, 0) else encode_plane(encoder, plane, args.abs_error)
                compressed_bytes += len(payload)
                offset = (t, c, 0, 0)
                dataset.id.write_direct_chunk(offset, payload, filter_mask=0)
                mask, stored = dataset.id.read_direct_chunk(offset)
                if mask != 0 or stored != payload:
                    raise RuntimeError(f"direct chunk write/mask mismatch at {offset}")
                decoded = np.empty((h, w), dtype="<f4")
                src = ctypes.create_string_buffer(payload, len(payload))
                if decoder.hf_cpu_f32(src, len(payload), decoded.ctypes.data_as(FLOAT_PTR), h * w):
                    raise native_error(decoder, "hf_cpu_f32")
                if not np.isfinite(decoded).all():
                    raise RuntimeError(f"non-finite CPU reference at {offset}")
                err = float(np.max(np.abs(decoded.astype(np.float64) - plane.astype(np.float64))))
                max_error = max(max_error, err)
                if err > args.abs_error * (1.0 + 1e-6):
                    raise RuntimeError(f"CPU reference exceeds absolute bound at {offset}: {err} > {args.abs_error}")
                oracle[t, c] = decoded
        dataset.id.close()
    oracle.flush()
    del oracle
    return path, oracle_path, {"shape": shape, "raw_bytes": raw_bytes, "compressed_bytes": compressed_bytes,
                               "max_abs_error": max_error, "abs_bound": args.abs_error,
                               "original_sha256": original_hash.hexdigest(), "filter_id": FALCONS_FILTER,
                               "config_bytes": first_payload[0], "plugin_threads": 1,
                               "numpy_version": np.__version__, "h5py_version": h5py.__version__,
                               "hdf5_version": h5py.version.hdf5_version}


def verify_batch(history, future, oracle, start, batch_size, h, w, device, digest):
    import numpy as np
    import torch

    shape = (batch_size, 1, CHANNELS, h, w)
    for name, tensor in (("history", history), ("future", future)):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise RuntimeError(f"{name} has unexpected type or shape, expected {shape}")
        if tensor.dtype != torch.float32 or not tensor.is_cuda or tensor.device != device:
            raise RuntimeError(f"{name} must be a float32 tensor on {device}")
    actual = torch.cat((history, future), dim=1).cpu().contiguous().numpy()
    expected = np.stack([oracle[i:i + FRAMES] for i in range(start, start + batch_size)])
    unequal = actual.view(np.uint32) != expected.view(np.uint32)
    if np.any(unequal):
        first = tuple(int(v) for v in np.argwhere(unequal)[0])
        raise RuntimeError(f"full tensor mismatch at sample start {start}, index {first}: "
                           f"actual={actual[first]!r}, cpu={expected[first]!r}")
    digest.update(actual.tobytes(order="C"))


def start_for_timing(loader, gate, timeout):
    if gate is None:
        loader.start()
        return
    from unittest.mock import patch
    from .loader import _ChunkBatches

    get_batch = _ChunkBatches.__getitem__

    def gated_batch(dataset, task):
        if not getattr(dataset, "_bench_gate_open", False):
            first = task[2] if dataset.route in ("cpu_stream", "dcu_stream") else task[0] == 0
            if not first:
                if not gate.wait(timeout):
                    raise RuntimeError("benchmark prefetch gate timed out")
                dataset._bench_gate_open = True
        return get_batch(dataset, task)

    # Only the benchmark's forked workers inherit the gate. Restore the parent immediately.
    with patch.object(_ChunkBatches, "__getitem__", gated_batch):
        loader.start()


def measure(args):
    import multiprocessing as mp
    import numpy as np
    import torch
    from .loader import ERA5Loader

    h, w = args.shape
    oracle = np.load(args._oracle, mmap_mode="r", allow_pickle=False)
    samples = args.batch_size * (args.batches + 1)
    if oracle.shape != (samples + FRAMES - 1, CHANNELS, h, w) or oracle.dtype != np.dtype("<f4"):
        raise RuntimeError("CPU oracle has an unexpected shape or dtype")
    reader_processes = _route_readers(args._route, args.reader_processes)
    loader = ERA5Loader(args._fixture, library_path=str(Path(args.library).resolve()), route=args._route,
                        device=args.device, num_workers=args.workers, batch_size=args.batch_size,
                        history=HISTORY, future=FUTURE, dataset="fields", mean=None, std=None,
                        shuffle=False, drop_last=True, reader_processes=reader_processes)
    setup_start = time.perf_counter()
    gate = mp.get_context("fork").Event() if args.gate_prefetch else None
    try:
        # This must fork the persistent workers before any CUDA/HIP runtime use.
        start_for_timing(loader, gate, args.timeout)
        setup_ms = (time.perf_counter() - setup_start) * 1000.0
        iterator = iter(loader)
        if torch.version.hip is None:
            raise RuntimeError("this benchmark requires a ROCm/HIP PyTorch build")
        device = torch.device(args.device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.synchronize(device)
        first_start = time.perf_counter()
        first_history, first_future = next(iterator)
        first_sink = first_history.sum() + first_future.sum()
        torch.cuda.synchronize(device)
        first_end = time.perf_counter()
        first_ms = (first_end - first_start) * 1000.0
        cold_first_ms = (first_end - setup_start) * 1000.0
        digest = hashlib.sha256()
        verify_batch(first_history, first_future, oracle, 0, args.batch_size, h, w, device, digest)
        if not math.isfinite(float(first_sink.item())):
            raise RuntimeError("first-batch device consumption produced a non-finite sum")
        del first_history, first_future, first_sink

        records = []
        sums = []
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        if gate is not None:
            gate.set()
        for _ in range(args.batches):
            history, future = next(iterator)
            sums.append(history.sum() + future.sum())
            records.append((history, future))
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        if seconds <= 0:
            raise RuntimeError("invalid timing interval")
        # All device-to-host comparisons and hashes happen after the timed window.
        for i, (history, future) in enumerate(records, start=1):
            verify_batch(history, future, oracle, i * args.batch_size, args.batch_size, h, w, device, digest)
        if not bool(torch.isfinite(torch.stack(sums)).all().item()):
            raise RuntimeError("device consumption produced a non-finite sum")
        try:
            next(iterator)
        except StopIteration:
            pass
        else:
            raise RuntimeError("loader produced unexpected extra samples")
        count = args.batch_size * args.batches
        byte_count = count * FRAMES * CHANNELS * h * w * 4
        return {"route": args._route, "reader_processes": reader_processes,
                "gate_prefetch": args.gate_prefetch, "setup_ms": setup_ms, "first_batch_ms": first_ms,
                "cold_first_batch_ms": cold_first_ms,
                "timed_batches": args.batches, "samples": count, "elapsed_ms": seconds * 1000.0,
                "ms_per_batch": seconds * 1000.0 / args.batches, "samples_per_s": count / seconds,
                "first_plus_steady_ms_per_batch": (first_ms + seconds * 1000.0) / (args.batches + 1),
                "logical_gb_per_s": byte_count / seconds / 1e9, "sha256": digest.hexdigest(),
                "pid": os.getpid(), "torch_version": torch.__version__, "torch_hip_version": torch.version.hip}
    finally:
        if gate is not None:
            gate.set()
        loader.close()


def subprocess_result(args, label, cmd):
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=env, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            out, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
        raise RuntimeError(f"{label} subprocess timed out after {args.timeout}s\n{out}\n{err}")
    if proc.returncode:
        raise RuntimeError(f"{label} subprocess failed with exit code {proc.returncode}\n{out}\n{err}")
    lines = [line for line in out.splitlines() if line.startswith(RESULT_PREFIX)]
    if len(lines) != 1:
        raise RuntimeError(f"{label} subprocess returned no unique result\n{out}\n{err}")
    result = json.loads(lines[0][len(RESULT_PREFIX):])
    if err.strip():
        print(f"{label} subprocess stderr:\n{err.rstrip()}", file=sys.stderr)
    other = [line for line in out.splitlines() if not line.startswith(RESULT_PREFIX)]
    if other:
        print(f"{label} subprocess output:\n" + "\n".join(other))
    return result


def run_child(args, route, path, oracle):
    cmd = [sys.executable, "-m", "hipfalcons.bench", "--_worker", "--_route", route,
           "--_fixture", str(path), "--_oracle", str(oracle), "--library", str(Path(args.library).resolve()),
           "--shape", str(args.shape[0]), str(args.shape[1]), "--batch-size", str(args.batch_size),
           "--batches", str(args.batches), "--device", args.device, "--workers", str(args.workers),
           "--reader-processes", str(args.reader_processes),
           "--limit-mib", str(args.limit_mib), "--timeout", str(args.timeout),
           "--baseline", args.baseline, "--candidate", args.candidate]
    if args.gate_prefetch:
        cmd.append("--gate-prefetch")
    result = subprocess_result(args, route, cmd)
    if result["route"] != route:
        raise RuntimeError("subprocess returned the wrong route")
    effective_readers = _route_readers(route, args.reader_processes)
    if result.get("reader_processes") != effective_readers:
        raise RuntimeError("subprocess returned the wrong reader process count")
    return result


def build_fixture(args, directory):
    cmd = [sys.executable, "-m", "hipfalcons.bench", "--_builder", "--_work-dir", str(directory),
           "--library", str(Path(args.library).resolve()), "--shape", str(args.shape[0]), str(args.shape[1]),
           "--batch-size", str(args.batch_size), "--batches", str(args.batches), "--limit-mib", str(args.limit_mib),
           "--abs-error", repr(args.abs_error), "--seed", str(args.seed)]
    if args.encoder_library:
        cmd.extend(("--encoder-library", str(Path(args.encoder_library).resolve())))
    details = subprocess_result(args, "fixture", cmd)
    return directory / "fields.h5", directory / "cpu_oracle.npy", details


def write_report(path, report):
    target = Path(path).expanduser().resolve()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=".hipfalcons-report-", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    args = arguments()
    if sys.platform != "linux":
        raise RuntimeError("this benchmark requires Linux fork workers and a HIP-capable PyTorch runtime")
    if sys.byteorder != "little":
        raise RuntimeError("this benchmark requires a little-endian CPU")
    # Only raw chunk I/O is used; prevent an installed filter's set_local hook
    # from changing the deliberately optional fixture filter.
    os.environ["HDF5_PLUGIN_PRELOAD"] = "::"
    if args._worker:
        print(RESULT_PREFIX + json.dumps(measure(args), sort_keys=True), flush=True)
        return
    if args._builder:
        _, _, details = fixture(args, Path(args._work_dir))
        print(RESULT_PREFIX + json.dumps(details, sort_keys=True), flush=True)
        return
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    routes = [args.baseline, args.candidate] + ([args.extra] if args.extra else [])
    results = {route: [] for route in routes}
    with tempfile.TemporaryDirectory(prefix="hipfalcons-loader-", dir=args.temp_dir) as tmp:
        path, oracle, details = build_fixture(args, Path(tmp))
        print("fixture " + json.dumps(details, sort_keys=True), flush=True)
        print(f"workload batch={args.batch_size} frames={FRAMES} channels={CHANNELS} workers={args.workers} "
              f"requested_reader_processes={args.reader_processes} "
              f"timed_batches={args.batches} windows={args.windows} device={args.device} "
              f"per_fixture_and_output_limit_MiB={args.limit_mib}", flush=True)
        expected_hash = None
        for window in range(args.windows):
            # Each route runs once per window; rotate who goes first.
            rotation = window % len(routes)
            for route in routes[rotation:] + routes[:rotation]:
                result = run_child(args, route, path, oracle)
                if expected_hash is None:
                    expected_hash = result["sha256"]
                elif result["sha256"] != expected_hash:
                    raise RuntimeError(f"{route} full-output hash differs from the verified baseline")
                results[route].append(result)
                print(f"window={window + 1} route={route} reader_processes={result['reader_processes']} "
                      f"worker_setup_ms={result['setup_ms']:.3f} "
                      f"first_batch_ms={result['first_batch_ms']:.3f} "
                      f"cold_first_batch_ms={result['cold_first_batch_ms']:.3f} "
                      f"first_plus_steady_ms_per_batch={result['first_plus_steady_ms_per_batch']:.3f} "
                      f"ms_per_batch={result['ms_per_batch']:.3f} samples_per_s={result['samples_per_s']:.3f} "
                      f"logical_GB_per_s={result['logical_gb_per_s']:.4f} identical=yes", flush=True)
        medians = {route: statistics.median(r["ms_per_batch"] for r in results[route]) for route in routes}
        first_plus = {route: statistics.median(r["first_plus_steady_ms_per_batch"] for r in results[route])
                      for route in routes}
        baseline, candidate = medians[args.baseline], medians[args.candidate]
        winner = "tie" if baseline == candidate else args.baseline if baseline < candidate else args.candidate
        cpu_library = Path(args.library).resolve().parent / "libfalcons_loader_cpu.so"
        report = {"schema_version": 1, "winner": winner, "baseline": args.baseline, "candidate": args.candidate,
                  "fixture": details, "library": str(Path(args.library).resolve()), "cpu_library": str(cpu_library),
                  "encoder_library": str(Path(args.encoder_library).resolve() if args.encoder_library else cpu_library),
                  "workload": {"height": args.shape[0], "width": args.shape[1],
                               "batch_size": args.batch_size, "history": HISTORY, "future": FUTURE,
                               "channels": CHANNELS, "workers": args.workers, "timed_batches": args.batches,
                               "requested_reader_processes": args.reader_processes,
                               "reader_processes_by_route": {
                                   route: _route_readers(route, args.reader_processes)
                                   for route in routes},
                               "windows": args.windows, "abs_error": args.abs_error, "seed": args.seed,
                               "gate_prefetch": args.gate_prefetch,
                               "device": args.device, "max_fixture_and_output_bytes": args.limit_mib * 1024 * 1024},
                  "results": results,
                  "median_ms_per_batch": medians, "median_first_plus_steady_ms_per_batch": first_plus,
                  "full_tensor_sha256": expected_hash}
        if args.extra:
            report["extra"] = args.extra
        if args.report:
            write_report(args.report, report)
        median_text = " ".join(f"{route}={medians[route]:.3f}" for route in routes)
        print(f"median_ms_per_batch {median_text} empirically_faster({args.baseline}_vs_{args.candidate})={winner} "
              f"full_tensor_sha256={expected_hash}", flush=True)
        if args.report:
            print(f"report={Path(args.report).expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"bench: {exc}", file=sys.stderr)
        sys.exit(1)
