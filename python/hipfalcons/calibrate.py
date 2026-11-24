#!/usr/bin/env python3
"""Calibrate two selected ERA5Loader routes on an HDF5 dataset.

Requires Linux, ROCm/HIP PyTorch, and the loader's two native libraries. The
dataset must meet ERA5Loader's constraints: native float32 [T,C,H,W], chunks
(1,1,H,W), and a single applied Falcons filter (40024). The legacy GPU route
supports only serial 2D LINEAR streams with the supported metadata; the DCU
route has its own supported stream constraints.

Each route/window uses a fresh process and fork workers before GPU init. It
repeats the same initial sequential window without shuffling: one warmup and
the requested full timed batches. It times synchronized DataLoader-to-device
delivery plus a small GPU reduction. No model is timed. The subprocesses set
OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and NUMEXPR_NUM_THREADS
to 1; num-workers controls DataLoader parallelism. Optional reader processes
are configured per worker for cpu_stream, dcu and dcu_stream.
The result is a short, warm-prefetched calibration, not a general performance
guarantee; cache state, worker count, workload, and shared load matter.
Optional sampled comparisons retain whole device batches until after timing;
they compare baseline/candidate bytes, not the original data or an error bound, and do
not validate unsampled chunks. Use --verify-batches 0 to skip comparison.
JSON is emitted only after both routes and all requested checks pass. A device
format rejection fails with a CPU recommendation; other failures do not.
The first batch is excluded from steady timing; first-plus-steady combines the
two separately measured durations and excludes the intervening validation.
"""

import argparse
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

MARKER = "__ERA5_CALIBRATION__="
INCOMPATIBLE = (
    "requires a serial 1D/2D/3D float32 Falcons interpolation stream",
    "requires LINEAR interpolation with matching serialized dimensions and block size",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="existing, stable HDF5 input file")
    p.add_argument("--library", required=True, help="HIP codec library; libfalcons_loader_cpu.so must be adjacent")
    p.add_argument("--dataset", default="fields", help="HDF5 dataset path")
    p.add_argument("--channels", type=int, nargs="+", help="ordered source-channel indices; default all")
    p.add_argument("--history", type=int, default=1)
    p.add_argument("--future", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--batches", type=int, default=8, help="timed full batches per route after one warmup")
    p.add_argument("--windows", type=int, default=3, help="paired windows, at most 3")
    p.add_argument("--baseline", choices=("cpu", "dcu", "cpu_stream"), default="cpu")
    p.add_argument("--candidate", choices=("gpu", "dcu", "dcu_stream"), default="dcu_stream")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--reader-processes", type=int, default=1,
                   help="total readers per worker for cpu_stream, dcu and dcu_stream (default: 1; 0 aliases 1)")
    p.add_argument("--device", default="cuda:0", help="PyTorch ROCm/HIP device")
    p.add_argument("--verify-batches", type=int, default=1, help="number of full timed batches to compare (0..3)")
    p.add_argument("--timeout", type=float, default=600, help="seconds per subprocess")
    p.add_argument("--report", help="optional JSON output, atomically replaced only on success")
    p.add_argument("--_child", choices=("cpu", "gpu", "dcu", "cpu_stream", "dcu_stream"), help=argparse.SUPPRESS)
    a = p.parse_args()
    if any(x < 1 for x in (a.history, a.future, a.batch_size, a.batches, a.num_workers)):
        p.error("history, future, batch-size, batches and num-workers must be positive")
    if a.reader_processes < 0:
        p.error("reader-processes must be nonnegative")
    if not 1 <= a.windows <= 3:
        p.error("windows must be 1..3")
    if a.baseline == a.candidate:
        p.error("baseline and candidate must differ")
    if not 0 <= a.verify_batches <= min(3, a.batches):
        p.error("verify-batches must be in 0..min(3, batches)")
    if a.channels is not None and (min(a.channels) < 0 or len(set(a.channels)) != len(a.channels)):
        p.error("channels must be unique nonnegative indices")
    if not math.isfinite(a.timeout) or a.timeout <= 0:
        p.error("timeout must be finite and positive")
    return a


def selected_batches(count, total):
    if count == 0:
        return set()
    if count == 1:
        return {0}
    return {i * (total - 1) // (count - 1) for i in range(count)}


def batch_hash(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return {"shape": list(array.shape), "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}


def measure(a):
    import torch
    from .loader import ERA5Loader

    reader_processes = _route_readers(a._child, a.reader_processes)
    loader = ERA5Loader(a.file, library_path=a.library, route=a._child, device=a.device,
                        num_workers=a.num_workers, batch_size=a.batch_size,
                        history=a.history, future=a.future, dataset=a.dataset,
                        channels=a.channels, shuffle=False, drop_last=True,
                        reader_processes=reader_processes)
    started = time.perf_counter()
    try:
        loader.start()  # Forks all workers before any CUDA/HIP initialization.
        setup_ms = (time.perf_counter() - started) * 1000
        if len(loader) < a.batches + 1:
            raise RuntimeError("dataset has fewer full batches than warmup + requested timed batches")
        if torch.version.hip is None:
            raise RuntimeError("ROCm/HIP PyTorch is required")
        shape = tuple(loader.shape)
        channels = tuple(loader.channels)
        device = torch.device(a.device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        iterator = iter(loader)

        def consume(history, future):
            for tensor, frames in ((history, a.history), (future, a.future)):
                expected = (a.batch_size, frames, len(channels), shape[2], shape[3])
                if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected
                        or tensor.dtype != torch.float32 or tensor.device != device or not tensor.is_cuda):
                    raise RuntimeError("loader returned an unexpected shape, dtype, or device")
            return history[..., 0, 0].sum() + future[..., 0, 0].sum()

        torch.cuda.synchronize(device)
        warm_start = time.perf_counter()
        h, f = next(iterator)
        warm_sink = consume(h, f)
        torch.cuda.synchronize(device)
        first_ms = (time.perf_counter() - warm_start) * 1000
        warm_sink.item()
        del h, f, warm_sink

        keep = []
        samples = selected_batches(a.verify_batches, a.batches)
        sink = None
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for i in range(a.batches):
            history, future = next(iterator)
            value = consume(history, future)
            sink = value if sink is None else sink + value
            if i in samples:
                keep.append((i, history, future))
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        if seconds <= 0:
            raise RuntimeError("invalid timing interval")
        sink.item()  # Small sink and all sample transfers occur after the timer.
        hashes = [{"batch": i, "history": batch_hash(history), "future": batch_hash(future)}
                  for i, history, future in keep]
        logical_bytes = (a.batches * a.batch_size * (a.history + a.future)
                         * len(channels) * shape[2] * shape[3] * 4)
        return {"route": a._child, "reader_processes": reader_processes,
                "pid": os.getpid(), "shape": shape, "channels": channels,
                "worker_setup_ms": setup_ms, "first_batch_ms": first_ms,
                "elapsed_ms": seconds * 1000, "ms_per_batch": seconds * 1000 / a.batches,
                "first_plus_steady_ms_per_batch": (first_ms + seconds * 1000) / (a.batches + 1),
                "samples_per_s": a.batches * a.batch_size / seconds,
                "logical_gb_per_s": logical_bytes / seconds / 1e9,
                "sample_hashes": hashes, "torch_version": torch.__version__, "hip_version": torch.version.hip}
    finally:
        loader.close()


def run_route(a, route):
    cmd = [sys.executable, "-m", "hipfalcons.calibrate", a.file, "--library", a.library,
           "--dataset", a.dataset, "--history", str(a.history), "--future", str(a.future),
           "--batch-size", str(a.batch_size), "--batches", str(a.batches),
           "--num-workers", str(a.num_workers), "--device", a.device,
           "--reader-processes", str(a.reader_processes),
           "--verify-batches", str(a.verify_batches), "--baseline", a.baseline,
           "--candidate", a.candidate, "--_child", route]
    if a.channels is not None:
        cmd += ["--channels", *(str(c) for c in a.channels)]
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               env=env, start_new_session=True)
    try:
        out, err = process.communicate(timeout=a.timeout)
    except subprocess.TimeoutExpired:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=5)
                break
            except subprocess.TimeoutExpired:
                continue
        raise RuntimeError(f"{route} subprocess timed out after {a.timeout}s")
    if process.returncode:
        if route in ("gpu", "dcu", "dcu_stream") and process.returncode == 3:
            raise RuntimeError(f"{route} rejected the stream as unsupported or inconsistent with its "
                               "metadata requirements; use CPU only after independent validation. "
                               "No fallback was performed.\n" + err.strip())
        raise RuntimeError(f"{route} subprocess failed (exit {process.returncode}):\n{err.strip()}\n{out.strip()}")
    lines = [line[len(MARKER):] for line in out.splitlines() if line.startswith(MARKER)]
    if len(lines) != 1:
        raise RuntimeError(f"{route} subprocess did not return one result: {out}\n{err}")
    result = json.loads(lines[0])
    if result.get("route") != route:
        raise RuntimeError("subprocess returned an unexpected route")
    effective_readers = _route_readers(route, a.reader_processes)
    if result.get("reader_processes") != effective_readers:
        raise RuntimeError("subprocess returned the wrong reader process count")
    extra = "\n".join(line for line in out.splitlines() if line and not line.startswith(MARKER))
    if extra or err.strip():
        print(f"{route} subprocess output:\n{extra}\n{err.strip()}", file=sys.stderr)
    return result


def signature(path):
    s = os.stat(path)
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def save_report(path, result):
    target = Path(path).expanduser().resolve()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                         prefix=".calibrate-", delete=False) as f:
            temporary = Path(f.name)
            json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    a = parse_args()
    if sys.platform != "linux":
        raise RuntimeError("Linux fork and a ROCm/HIP PyTorch runtime are required")
    if a._child:
        try:
            print("\n" + MARKER + json.dumps(measure(a), allow_nan=False), flush=True)
        except Exception as exc:
            message = str(exc)
            known_rejection = ((a._child == "gpu" and message.startswith("Falcons decoder failed")) or
                               (a._child in ("dcu", "dcu_stream") and
                                any(name in message for name in ("hf_pack_plane failed", "hf_pack_batch failed"))))
            if known_rejection and any(s in message for s in INCOMPATIBLE):
                print(message, file=sys.stderr)
                sys.exit(3)
            raise
        return

    a.file = str(Path(a.file).expanduser().resolve(strict=True))
    a.library = str(Path(a.library).expanduser().resolve(strict=True))
    if not Path(a.file).is_file() or not Path(a.library).is_file():
        raise RuntimeError("input and GPU library must be regular files")
    cpu_library = Path(a.library).parent / "libfalcons_loader_cpu.so"
    if not cpu_library.is_file():
        raise RuntimeError("missing CPU-only companion library: " + str(cpu_library))
    if a.report and Path(a.report).expanduser().resolve() in (Path(a.file), Path(a.library), cpu_library, Path(__file__).resolve()):
        raise RuntimeError("report must not overwrite the input, a codec library, or this script")
    original = signature(a.file)
    measurements = {a.baseline: [], a.candidate: []}
    for window in range(a.windows):
        pair = {}
        for route in ((a.baseline, a.candidate) if window % 2 == 0 else (a.candidate, a.baseline)):
            if signature(a.file) != original:
                raise RuntimeError("input file changed during calibration")
            pair[route] = run_route(a, route)
            if signature(a.file) != original:
                raise RuntimeError("input file changed during calibration")
        baseline, candidate = pair[a.baseline], pair[a.candidate]
        if baseline["shape"] != candidate["shape"] or baseline["channels"] != candidate["channels"]:
            raise RuntimeError(f"{a.baseline}/{a.candidate} metadata mismatch; no winner")
        if baseline["sample_hashes"] != candidate["sample_hashes"]:
            raise RuntimeError(f"{a.baseline}/{a.candidate} sampled tensor correctness mismatch; no winner or fallback")
        for route in measurements:
            measurements[route].append(pair[route])
    for route in measurements:
        if any((r["shape"], r["channels"]) != (measurements[a.baseline][0]["shape"], measurements[a.baseline][0]["channels"])
               for r in measurements[route]):
            raise RuntimeError("input shape or channel selection changed between windows")
    baseline = statistics.median(r["ms_per_batch"] for r in measurements[a.baseline])
    candidate = statistics.median(r["ms_per_batch"] for r in measurements[a.candidate])
    if baseline == candidate:
        raise RuntimeError("measured medians are identical; no winner")
    first_plus = {route: statistics.median(r["first_plus_steady_ms_per_batch"] for r in values)
                  for route, values in measurements.items()}
    result = {"winner": a.baseline if baseline < candidate else a.candidate,
              "baseline": a.baseline, "candidate": a.candidate,
              "median_ms_per_batch": {a.baseline: baseline, a.candidate: candidate},
              "median_first_plus_steady_ms_per_batch": first_plus,
              "file": a.file, "dataset": a.dataset, "library": a.library,
              "history": a.history, "future": a.future, "batch_size": a.batch_size, "batches": a.batches,
              "windows": a.windows, "num_workers": a.num_workers, "device": a.device,
              "requested_reader_processes": a.reader_processes,
              "reader_processes_by_route": {
                  route: _route_readers(route, a.reader_processes)
                  for route in measurements},
              "verify_batches": a.verify_batches, "measurements": measurements}
    if a.report:
        save_report(a.report, result)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"calibrate: {exc}", file=sys.stderr)
        sys.exit(1)
