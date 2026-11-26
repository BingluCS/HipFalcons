# HDF5 to a DCU training step

`dcu_dataloader.py` loads overlapping pairs of frames and runs a small
PyTorch model through forward, backward, and an optimizer step. It is a
training integration example, not an accuracy model or a benchmark.

## Requirements

- Linux, a DCU/HIP runtime, and a matching HIP-enabled PyTorch installation.
  PyTorch exposes the DCU through names such as `cuda:0`.
- The `hipfalcons` package installed in the Python environment, including the
  runnable `hipfalcons.bench` module. With a source checkout, putting its
  `python` directory on `PYTHONPATH` is an alternative.
- `h5py` for input; the demo also needs NumPy. The native HIP codec library
  and `libfalcons_loader_cpu.so` must be built together, with the CPU library
  beside the resolved `--library` path and their dependencies discoverable.
- A stable input file, readable by the worker processes, for the duration of
  training. The reader uses raw HDF5 chunks and does not need the HDF5 Falcons
  plugin installed; ordinary h5py slicing of compressed data does need it.

Run from the repository root, replacing the library path:

```bash
python examples/dcu_dataloader.py --demo \
  --library /path/to/build/libhipfalcons_loader.so \
  --route dcu_stream --num-workers 2 --batch-size 2 --epochs 2 --steps 4
```

The demo uses `hipfalcons.bench.build_fixture` to generate a temporary
96 x 120, four-channel file, requesting serial LINEAR compression and ABS
`1e-3`.
The builder also writes a CPU-decoded oracle and checks the synthetic planes
against that bound. Both files are removed after the loader is closed. It
creates one more full batch than `--steps` so that the example exercises
closing an early-stopped iterator. The builder limits the uncompressed
fixture and its calculated retained outputs to 128 MiB each; demo batch size
must be at most 16 and steps at most 4096. The demo requires a little-endian
host. By default the two DCU DataLoader workers each read and pack inline,
with no extra reader processes. `--reader-processes 0` is an alias for this
default inline mode.

For a prepared file:

```bash
python examples/dcu_dataloader.py --input /data/era5_falcons.h5 \
  --dataset fields --library /path/to/build/libhipfalcons_loader.so \
  --route dcu_stream --num-workers 1 --reader-processes 16 --batch-size 1 --channels 0 5 9 \
  --epochs 2 --steps 10
```

`--route` selects `cpu_stream` (CPU decode, then transfer), or `dcu` /
`dcu_stream` (CPU unpacking, then DCU reconstruction). All return batches
on the DCU and use one route for all workers in a run. Use a separate process
to compare routes; do not mix CPU and DCU worker pools in this example.
`--device` defaults to `cuda:0`. `--channels` selects unique zero-based source
indices in the specified order; omitted means all channels. `--steps` is a
maximum per epoch, so a shorter file ends the epoch earlier and its last batch
may be partial. By default, the example uses one history frame and one future
frame, with windows starting at consecutive time indices. `--shuffle` shuffles
window starts and disables sequential frame reuse; it also works with reader
processes on either DCU route. Shuffled `cpu_stream` stays inline; use
`--reader-processes 1` or `0` for that combination.

`--reader-processes` specifies total participants per DataLoader worker,
including the worker itself. The default `1` works inline with no helpers;
`0` is an inline alias. For `R > 1`, the worker handles a shard and `R-1`
helpers handle the rest. This is available on `dcu` and `dcu_stream` with or
without shuffle, and on `cpu_stream` only with `shuffle=False`. The command
above uses one worker plus 15 helpers. CPU participants decode disjoint planes
directly into one shared float32 tensor before the usual upload; DCU
participants pack their planes for reconstruction on the device. The legacy
`cpu` and `gpu` routes remain inline. Helpers are started by the loader's
parent before HIP initialization; each participating process opens its own
HDF5 handle, and helpers do not create GPU contexts. More
participants can increase startup, CPU and storage contention, host memory
and `/dev/shm` use; measure the target workload rather than assuming a speedup.

## Input and encoding

The named dataset must be native-endian float32 `[T, C, H, W]`, with `T >= 2`,
chunks exactly `(1, 1, H, W)`, and exactly one filter: Falcons, ID `40024`.
Every requested chunk must exist and have the filter applied, not skipped.
For `dcu` and `dcu_stream`, each chunk must contain a serial, effectively 2D
float32 Falcons interpolation stream using LINEAR or CUBIC interpolation and
quantizer radius 32768. OpenMP and lossless-fallback streams are not supported by
these routes, and automatic tuning can produce incompatible streams. `cpu_stream`
uses the native CPU decoder and can handle other formats that decoder
supports, while still requiring the same HDF5 layout.
There is no automatic DCU-to-CPU fallback for an unsupported chunk.

To prepare an uncompressed HDF5 file with a float32 `[T, C, H, W]` dataset,
for example:

```bash
python -m hipfalcons.encode /data/era5.h5 /data/era5_falcons.h5 \
  --dataset fields --library /path/to/build/libfalcons_loader_cpu.so --abs-error 1e-3
```

The encoder writes a new file and refuses to overwrite an existing one. It
reads one plane at a time and writes the result as `fields`; it does not copy
other datasets or attributes. Spatial dimensions must both be at least 2, and
this training example needs at least two time steps. Use finite input values
and retain the source for validation. The command applies the same scalar ABS
bound to every channel; variables with different units or scales may need a
different encoding policy. It does not verify every reconstructed chunk or
guarantee every chunk is compatible with `dcu_stream`.

For a real ERA5 float32 NumPy array in `[T, C, H, W]` order, first create an
ordinary HDF5 input and then use a per-plane relative bound:

```bash
python - /data/surface4-era5-20200101-24times-f32.npy /data/era5-2020-f32.h5 <<'PY'
import sys
import h5py
import numpy as np

data = np.load(sys.argv[1], mmap_mode="r", allow_pickle=False)
if data.ndim != 4 or data.dtype != np.float32:
    raise ValueError("expected float32 [T, C, H, W]")
with h5py.File(sys.argv[2], "x") as f:
    fields = f.create_dataset("fields", shape=data.shape, dtype=np.float32,
                              chunks=(1, 1, data.shape[2], data.shape[3]))
    for t in range(data.shape[0]):
        for c in range(data.shape[1]):
            fields[t, c] = data[t, c]
PY

python -m hipfalcons.encode /data/era5-2020-f32.h5 /data/era5-2020-rel.h5 \
  --dataset fields --library /path/to/build/libfalcons_loader_cpu.so --rel-error 1e-3
```

`--rel-error 1e-3` means an absolute error bound of
`1e-3 * (plane_max - plane_min)` for each `(time, channel)` plane. The range
is computed in float32, and the bound is in that variable's native units; it
is not a per-value relative bound. This matches the original H5Z-Falcons
per-chunk REL range for `(1, 1, H, W)` chunks; the converter emits supported
ABS streams. REL conversion rejects non-finite input or ranges and zero bounds
rather than silently switching to lossless. In contrast, `--abs-error` uses
the same absolute bound for every plane. Keep the original for validation.

The converter uses the original Falcons CPU 2D tuner to choose LINEAR or
CUBIC and its parameters per chunk, with OpenMP encoding disabled. No
interpolation flag is needed. An encoder under your control can also write
either serial mode with radius 32768 (65536 quantization bins).
Compress one contiguous `(H, W)` plane per `(time, channel)`. The included
[HDF5 converter](../docs/dataloader.md#convert-an-ordinary-hdf5-file)
does this using the native CPU encoder and `write_direct_chunk`. Run
conversion separately from training. An ABS bound is in the
units of each value, not a fraction of its range; choose it for the scientific
variables and verify decoded values against the originals.

For files written by another encoder, do not assume the HDF5 plugin's
defaults select a compatible stream. Validate
the actual stream mode and filter mask of **every requested chunk** before
relying on `dcu_stream`, and validate reconstruction error against the source.
A short `--steps` run checks only the chunks it reads. To exercise all times
for the chosen channels, use `--epochs 1` and set `--steps` to at least
`ceil((T - 1) / batch_size)`; compare decoded data to the original separately
to check the error bound. For an incompatible file, use `cpu_stream` if the
CPU decoder supports its streams, or re-encode.
The original [H5Z-Falcons documentation](../third_party/Falcons/tools/H5Z-Falcons/README.md)
describes the broader filter and its per-chunk REL mode; the DCU loader accepts
only the subset above, not every valid H5Z-Falcons stream.

For a paired sequential comparison on the prepared file:

```bash
HIP_VISIBLE_DEVICES=0 python -m hipfalcons.calibrate /data/era5-2020-rel.h5 \
  --library /path/to/build/libhipfalcons_loader.so --history 1 --future 1 \
  --batch-size 1 --num-workers 1 --channels 0 1 2 3 \
  --baseline cpu_stream --candidate dcu_stream --reader-processes 8 \
  --batches 8 --windows 3 --report route.json
```

The paired sequential routes use the same requested reader count, including
the worker; the default is `1`. The calibrator defaults to the inline legacy
`cpu` baseline, so select `--baseline cpu_stream` for matched multi-reader
calibration. It compares sampled outputs and does not replace
source-versus-decoded error checks.

## Lifetime and sizing

The example calls `loader.start()` before checking HIP availability, creating
the model, or allocating any device tensor. `start()` probes metadata in a
child and starts persistent Linux fork workers and any requested helper
processes. Each participating worker or helper opens its own HDF5 handle.
Do not initialize HIP or open an HDF5 handle in the training parent before
starting workers. Keep the loader alive across epochs, close an abandoned
iterator before creating the next one, and always close the loader.
The example does both in `finally` blocks, including on an exception.

Each returned pair consists of device float32 views shaped
`[batch, 1, selected_channels, H, W]`; do not call `.to(device)` on them
again. The example takes the single time index for its 2D convolution and uses
the current PyTorch stream. Consumers using another stream must arrange their
own stream ordering and tensor lifetimes. `--steps` and the printed loss are
for a small demonstration; printing the scalar synchronizes the device.
Production models should use appropriate channel scaling and statistics;
`ERA5Loader` also accepts `mean` and `std` together for device normalization.

At 257 channels and 721 x 1440, just one history/future pair at batch size 1
is about 2.13 GB (1.99 GiB) of float32 output. Model activations, gradients,
the streaming tail, worker prefetch, pinned memory and shared IPC require
additional space. Select only needed channels and size DCU, host and shared
memory for the actual batch and worker count, not the compressed file size.
`shared_pool_bytes` only controls reusable slots for inline sequential
`dcu_stream` packing with `reader_processes=1` or its `0` alias. With
`reader_processes>1`, DCU routes use independent shared output slabs and
sequential `cpu_stream` uses shared float32 tensors; that setting does not
limit either output or total RAM.
