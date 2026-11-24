# DataLoader Input and Tuning

## Input layout and compatibility

- The loader does not fix the number of time steps, channels or grid cells
  to ERA5. It reads a native-endian float32 dataset `[T, C, H, W]`, where
  `C >= 1` and `T >= history + future`. For the DCU route, `H, W >= 2`,
  `H * W <= 2**32 - 1`, and the working set must fit available memory.
  Odd and rectangular grid sizes are supported.
- Chunk shape must be exactly `(1, 1, H, W)`: one time and one variable per
  chunk. For `[1460,257,721,1440]`, use `(1,1,721,1440)`.
- Use filter ID 40024 only, applied to every requested chunk. For the
  `dcu_stream` or `dcu` route, encode each chunk with the supported CPU
  Falcons revision using **2D LINEAR interpolation, float32, radius 32768,
  and OpenMP off**, with an interpolation block size of `max(H, W)`.
  Set the error bound at compression, not in the loader.
- Reading is via `read_direct_chunk`, so neither a Falcons HDF5 plugin nor
  a custom h5py build is needed to read with HipFalcons. Normal h5py
  indexing of the same file still invokes the HDF5 filter.

**Reuse a compatible existing HDF5 file as-is.** The DCU loader does not
rewrite it or change its error bound, and your existing CPU reader can
continue reading it. If the layout matches but the codec is unsupported
on the DCU, select `route="cpu_stream"` if the bundled CPU decoder handles
that stream. This is an explicit CPU path, not automatic per-chunk fallback.
The original [H5Z-Falcons filter](../third_party/Falcons/tools/H5Z-Falcons/README.md)
supports more configurations than the DCU subset; filter ID 40024 or a
successful ordinary HDF5 read alone does not establish DCU compatibility.

## Convert an ordinary HDF5 file

The converter reads and compresses one plane at a time with the CPU-only
library. It accepts an ordinary float32 HDF5 dataset readable without
external filter plugins, writes a new `fields` dataset, and never overwrites
an existing file. This is a serial example, not a high-throughput archive
builder. It does not copy other datasets or attributes such as timestamps,
coordinates, or normalization statistics. The absolute error is in the
source variables' units and applies to every channel in this example. The
converter stops if Falcons picks a lossless-only fallback, rather than
writing a file that fails later on the DCU.

```bash
python -m hipfalcons.encode era5-raw.h5 era5-falcons.h5 --dataset fields \
    --library build/libfalcons_loader_cpu.so --abs-error 1e-3

HIP_VISIBLE_DEVICES=0 python examples/dcu_dataloader.py --input era5-falcons.h5 \
    --library build/libhipfalcons_loader.so --channels 0 1 2 3
```

For real ERA5 variables with different units and scales, a relative bound can
be more useful than one scalar ABS bound:

```bash
python -m hipfalcons.encode /data/era5-2020-f32.h5 /data/era5-2020-rel.h5 \
    --dataset fields --library build/libfalcons_loader_cpu.so --rel-error 1e-3
```

Here `--rel-error 1e-3` selects a separate absolute bound of
`1e-3 * (plane_max - plane_min)` for each `(time, channel)` plane, with
the range computed in float32 and the bound in that variable's native units.
It is not a per-value relative error bound. For example, temperature may be
in K while pressure is in Pa. The converter rejects non-finite input or ranges
and zero bounds in REL mode rather than silently switching to lossless.
This per-plane range corresponds to the original H5Z-Falcons per-chunk REL
definition for `(1, 1, H, W)` chunks; the converter emits supported ABS streams.
The earlier `--abs-error` command instead uses the same absolute bound for
every plane. Retain the original data and independently validate errors.

## Python API

Run `python -m pip install -e .` in your HIP-enabled PyTorch environment
after building the native libraries. Pass the path to
`libhipfalcons_loader.so`; keep `libfalcons_loader_cpu.so` beside it.
Create the loader and enter its context before the process initializes HIP
or creates a DCU tensor. `ERA5Loader` already manages its own workers:
iterate it directly rather than putting it inside another DataLoader.

Here `model` is your model on the CPU; move it to the DCU after entering
the loader context.

```python
from hipfalcons import ERA5Loader

with ERA5Loader(
    "weather-falcons.h5", dataset="fields",
    library_path="build/libhipfalcons_loader.so",
    device="cuda:0", route="dcu_stream", num_workers=4,
    batch_size=3, history=2, future=2, channels=[4, 0, 2],
) as loader:
    model = model.to("cuda:0")  # The loader has already started its fork workers.
    for history, target in loader:
        prediction = model(history)  # Adapt the time dimension to your model.
        # history and target are already on the DCU; do not call .to() again.
```

With this `dcu_stream` example, the default `reader_processes=1` packs inline
in each of the four DataLoader workers, with no additional reader processes.
`reader_processes=0` is also accepted as an alias for inline packing.

| Argument | Meaning |
| --- | --- |
| `dataset` | Path within the HDF5 file, such as `fields` or `weather/fields`. |
| `history`, `future` | Positive frame counts. A sample at time `t` takes input frames `[t, t+history)` and target frames `[t+history, t+history+future)`. There are `T-history-future+1` sample windows. |
| `channels` | Zero-based indices in output order, with no duplicates. Omit for all channels. |
| `batch_size`, `drop_last` | Number of windows per batch. With the default `drop_last=False`, the final batch can be smaller. |
| `num_workers` | Positive number of CPU fork workers preparing different batches. It does not set a DCU thread count. |
| `reader_processes` | Total participants per DataLoader worker, including the worker itself. `1` (default) and `0` use the inline path. For `R > 1`, the worker and `R-1` helpers read and pack on `dcu`/`dcu_stream`, with or without shuffle, or read and decode on sequential `cpu_stream` with `shuffle=False`. Legacy `cpu`/`gpu` and shuffled `cpu_stream` use the inline path; select `0` or `1` for those routes. |
| `route` | `dcu` and `dcu_stream` run interpolation on the DCU; `cpu_stream` reconstructs on the CPU, then uploads floats. All return DCU tensors. |

Returned shapes are `[B, history, C_selected, H, W]` and
`[B, future, C_selected, H, W]`. For the example above and
`[T,C,H,W] = [10,5,65,129]`, an epoch has seven windows and batch
sizes `3, 3, 1`. The first pair of tensors both have shape
`[3,2,3,65,129]`.

To stop an epoch early and start another one, close the iterator with
`iterator.close()` or wrap it in `contextlib.closing`. The loader context
closes all workers. It returns only `(history, target)`; time stamps,
sample indices, auxiliary features and custom samplers from an existing
training reader are not supplied by this API.

With the default `reader_processes=1` on a DCU route, each worker opens its
own HDF5 handle and packs its batches inline; `0` selects the same path.
Larger values add helpers while the worker also processes a shard. On
sequential `cpu_stream`, participants decode disjoint planes directly into one
shared float32 tensor, which follows the existing CPU-to-DCU upload path.
Legacy `cpu` and `gpu` routes remain inline without helpers. The training
process owns the HIP stream; `num_workers` does not set a DCU thread count.
For sequential training, `cpu_stream` and `dcu_stream` keep overlapping time
frames on the DCU and read only the new frames.
`shuffle=True` disables reuse between batches, but still supports reader
processes on `dcu` and `dcu_stream`. Shuffled `cpu_stream` uses the inline
CPU path; use `reader_processes=1` or `0`. For per-variable normalization,
pass both `mean` and `std` in selected-channel order; normalization runs on
the device.

For example, run a sequential CPU baseline with eight participants per worker:

```bash
HIP_VISIBLE_DEVICES=0 python examples/dcu_dataloader.py --input era5-falcons.h5 \
    --library build/libhipfalcons_loader.so --route cpu_stream --num-workers 1 \
    --reader-processes 8 --batch-size 1 --channels 0 1 2 3
```

## Reader processes

For one HDF5 file, this selects 16 read-and-pack participants for one
DataLoader worker: the worker itself plus 15 helpers. This reads all selected
frames in channel order (all channels when `channels` is omitted) and retains
overlapping frames between batches:

```python
from contextlib import closing
from hipfalcons import ERA5Loader

with ERA5Loader(
    "weather-falcons.h5", dataset="fields",
    library_path="build/libhipfalcons_loader.so",
    device="cuda:0", route="dcu_stream", num_workers=1, reader_processes=16,
    batch_size=1, history=1, future=1, shuffle=False,
) as loader:
    with closing(iter(loader)) as batches:
        history, target = next(batches)
```

For a DCU route or sequential `cpu_stream` and `R > 1`, the loader's parent
creates `R-1` helpers per worker before HIP initialization; the worker itself
also reads and packs or decodes its own shard.
For example, `reader_processes=8` uses one worker and seven helpers; two
workers with `reader_processes=8` provide 16 participants,
including 14 additional helpers. Each participating process opens its own
HDF5 handle; handles are not shared across processes. The configured count
is not necessarily the number active in a batch: a batch with fewer chunks
can use fewer participants. Helpers do not create GPU contexts. Keep the
file stable and start the loader before initializing HIP or opening an HDF5
handle in the training process. Extra processes can increase startup, CPU
scheduling, storage contention and memory costs; more participants do not
necessarily improve throughput. Test the actual file, channel count and
batch size within the job's CPU allocation.

`shared_pool_bytes` defaults to 64 MiB per loader and controls only the
reusable IPC pool for **inline**, sequential `dcu_stream` packing
(`reader_processes=1` or `0`, `shuffle=False`), which is the default
participant configuration. Too-large or busy pool slots fall back to
per-batch shared memory. Set `shared_pool_bytes=0` to disable this pool
without disabling frame reuse. With `reader_processes>1` on a DCU route,
packing instead uses independent shared output slabs. Sequential `cpu_stream`
uses shared float32 tensors. Neither is limited by `shared_pool_bytes`.
Prefetch and multiple workers can increase the number of live outputs.

One input/target pair with all 257 ERA5 channels at `721 x 1440` occupies
roughly 2 GiB **just for the returned tensors**, not counting processes,
shared output slabs, pinned buffers, reuse buffers, the model or gradients.
Start with a subset of channels and check host, device and `/dev/shm` usage.

## Compare routes on an existing file

```bash
HIP_VISIBLE_DEVICES=0 python -m hipfalcons.calibrate era5-falcons.h5 \
    --library build/libhipfalcons_loader.so --history 1 --future 1 \
    --batch-size 1 --num-workers 1 --channels 0 1 2 3 \
    --baseline cpu_stream --candidate dcu_stream --reader-processes 16 \
    --batches 8 --windows 3 --report route.json
```

Use the report's `winner` as `route` when the training batch layout and
worker and reader-participant counts match the calibration. Calibration runs
without shuffle. For the explicit `cpu_stream`/`dcu_stream` pair above,
both routes use the requested `--reader-processes` count, which defaults to
`1`; `0` is an inline alias. Each worker in this example uses 15 helpers.
The calibrator's default baseline is legacy `cpu`, which remains inline;
specify `--baseline cpu_stream` for a matched multi-reader comparison. The
legacy `gpu` control also remains inline. The report records the configuration
per route, not a count of participants active on every batch.
The utility checks representative batches; it cannot guarantee that untested
chunks use a supported mode.
The [runnable example](../examples/README.md) shows complete training code,
including resource cleanup.
