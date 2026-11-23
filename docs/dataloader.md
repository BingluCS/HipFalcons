# DataLoader Input and Tuning

## Input layout

- One native-endian float32 dataset, `fields[T, C, H, W]`, with HDF5 chunks
  shaped `(1, 1, H, W)`. Each chunk is one time and one variable. For
  `[1460,257,721,1440]`, use `(1,1,721,1440)`.
- Use filter ID 40024 only, applied to every requested chunk. For the
  `dcu_stream` or `dcu` route, encode each chunk with the supported CPU
  Falcons revision using **2D LINEAR interpolation, float32, radius 32768,
  and OpenMP off**. Set the error bound at compression, not in the loader.
  The demo writes this format with ABS `1e-3` using a CPU encoder.
- Reading is via `read_direct_chunk`, so neither a Falcons HDF5 plugin nor
  a custom h5py build is needed to read with HipFalcons. Normal h5py
  indexing of the same file still invokes the HDF5 filter.

## Convert an ordinary HDF5 file

The converter reads and compresses one plane at a time with the CPU-only
library. It accepts an ordinary float32 HDF5 dataset readable without
external filter plugins, writes a new `fields` dataset, and never overwrites
an existing file. This is a serial example, not a high-throughput archive
builder. The absolute error is in the source variables' units and applies
to every channel in this example. The converter stops if Falcons picks a
lossless-only fallback, rather than writing a file that fails later on the
DCU.

```bash
python -m hipfalcons.encode era5-raw.h5 era5-falcons.h5 --dataset fields \
    --library build/libfalcons_loader_cpu.so --abs-error 1e-3

HIP_VISIBLE_DEVICES=0 python examples/dcu_dataloader.py --input era5-falcons.h5 \
    --library build/libhipfalcons_loader.so --channels 0 1 2 3
```

## Use in training

```python
from hipfalcons import ERA5Loader

with ERA5Loader(
    "era5-falcons.h5", dataset="fields",
    library_path="build/libhipfalcons_loader.so",
    device="cuda:0", route="dcu_stream", num_workers=4,
    batch_size=1, history=1, future=1, channels=[0, 1, 2, 3],
) as loader:
    model = model.to("cuda:0")  # The loader has already started its fork workers.
    for history, target in loader:
        prediction = model(history)
        # Both input tensors and prediction are already on the DCU.
```

Each worker opens its own HDF5 handle and prepares different batches; the
training process owns the HIP stream. `num_workers` is the number of CPU
processes, not DCU threads or additional pools of decompression processes.
For sequential training, `dcu_stream` keeps the overlapping time frames on
the DCU and reads only the new frames. `shuffle=True` disables reuse between
batches. For per-variable normalization, pass both `mean` and `std` in
selected-channel order; normalization runs on the device.

For sequential `dcu_stream`, `shared_pool_bytes` defaults to 64 MiB per
loader. Too-large or busy slots fall back to per-batch shared memory.
Use `shared_pool_bytes=0` to turn the pool off without disabling frame
reuse. One input/target pair with all 257 ERA5 channels at `721 x 1440`
occupies roughly 2 GiB **just for the returned tensors**, not counting
workers, pinned buffers, reuse buffers, the model, or gradients. Start
with a subset of channels and tune both device and `/dev/shm` usage.

## Compare routes on an existing file

```bash
HIP_VISIBLE_DEVICES=0 python -m hipfalcons.calibrate era5-falcons.h5 \
    --library build/libhipfalcons_loader.so --history 1 --future 1 \
    --batch-size 1 --num-workers 4 --channels 0 1 2 3 \
    --baseline cpu_stream --candidate dcu_stream --batches 8 --windows 3 --report route.json
```

Use the report's `winner` as `route` when the training batch layout and
worker count match the calibration. The utility checks representative
batches; it cannot guarantee that untested chunks use a supported mode.
The [runnable example](../examples/README.md) shows complete training code,
including resource cleanup.
