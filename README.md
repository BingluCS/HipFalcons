# HipFalcons

HipFalcons decodes CPU Falcons streams on AMD-compatible HIP devices. Its
PyTorch DataLoader reads Falcons-compressed HDF5 and yields history and target
tensors already resident on the DCU.

**Training path:** HDF5 raw chunks → CPU Zstd and entropy decoding →
packed int16 codes → DCU interpolation → PyTorch tensors. The CPU encoder and
file format do not change. The loader does not require the HDF5 filter plugin
for reading, because it reads the stored chunk bytes instead of invoking h5py
slicing.

For a compatible [CPU Falcons](third_party/README.md) file, use the existing
HDF5 dataset directly; no recompression or replacement of your original reader
is needed. For an uncompressed float32 dataset, the [HDF5 conversion guide](docs/dataloader.md#convert-an-ordinary-hdf5-file)
shows how to select a fixed ABS or per-chunk REL bound.

The [standalone HIP codec](codec/README.md), `hipfalcons_codec`, can
compress and decompress individual Falcons streams with LINEAR or CUBIC
interpolation. The HDF5 DataLoader also reconstructs LINEAR and CUBIC chunks
on the DCU, for the serial 2D float32 layout below. It does not require the
optional standalone codec to be built.

## Build

**Clone only HipFalcons.** The tested CPU Falcons and fallback Zstd sources
are included as ordinary files. No second clone, submodule initialization,
or dependency download is needed during the CMake build.

Use Linux, CMake 3.18+, a C++17 compiler, and an active HIP/DTK environment.
Set up HIP using your server's instructions before building. If CMake needs
the installation location, pass `-DDTK_ROOT=<HIP-install-prefix>`.
The commands below target a BW150 (`gfx936`) with an AVX2/FMA host CPU.
Set `HIP_ARCH` for your device and omit `-DENABLE_AVX2=ON` when the host does
not support AVX2/FMA.

```bash
git clone https://github.com/BingluCS/HipFalcons.git
cd HipFalcons

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHIP_ARCH=gfx936 -DENABLE_AVX2=ON
cmake --build build -j4
build/hipfalcons --self-test
```

For the standalone codec, configure with `-DHF_BUILD_HIP_CODEC=ON` in
addition to the options above, then run
`cmake --build build --target hipfalcons_codec -j4`. See
[standalone examples](codec/README.md#build-and-use) for CUBIC encode and
decode commands.

On DTK 26.04, a HIP compile error such as `'cmath' file not found` can
mean that `hipcc` did not discover the host GCC installation. With the
intended `g++` active, retry the build with:

```bash
HIPCC_COMPILE_FLAGS_APPEND="${HIPCC_COMPILE_FLAGS_APPEND:+$HIPCC_COMPILE_FLAGS_APPEND }--gcc-install-dir=$(dirname "$(g++ -print-file-name=libgcc.a)")" \
  cmake --build build -j4
```

Falcons uses system Zstd when available, otherwise it builds
`third_party/zstd`. No system Zstd development package is required.
If you built an earlier checkout, configure a **new build directory** so
the old CMake cache does not keep pointing to `deps/Falcons`.
Pass `-DFALCONS_SOURCE=/path/to/Falcons` only to use a compatible external
checkout instead of the [bundled source](third_party/README.md).
Check that `rocminfo` sees the DCU in the job that will run the example;
management-tool visibility alone does not grant compute access.

## DataLoader Quick Start

Use the PyTorch build provided for your DTK/ROCm installation. Install the
Python package in that environment; it does not install or replace PyTorch.
Keep both `libhipfalcons_loader.so` and `libfalcons_loader_cpu.so` together in
the build directory.

```bash
python -m pip install -e .
HIP_VISIBLE_DEVICES=0 python examples/dcu_dataloader.py --demo \
    --library build/libhipfalcons_loader.so
```

The demo generates a small compatible HDF5 file and runs a real forward,
backward, and optimizer step on the DCU. To use an existing compatible file
from Python:

```python
from contextlib import closing
from hipfalcons import ERA5Loader

with ERA5Loader(
    "fields.h5", dataset="fields",
    library_path="build/libhipfalcons_loader.so",
    route="dcu_stream", device="cuda:0", num_workers=2,
    batch_size=1, history=1, future=1, channels=[0],
) as loader:
    with closing(iter(loader)) as batches:
        history, target = next(batches)
        print(history.shape, target.shape, history.device)
```

PyTorch uses the `cuda:0` device namespace on HIP; no NVIDIA GPU is required.
Start the loader before moving the model or anything else to the DCU, so
the workers fork before HIP initialization. See the
[Python guide](docs/dataloader.md#python-api) for shapes and arguments,
and the [training example](examples/README.md) for a complete loop.

Read an existing compatible CUBIC HDF5 file unchanged with
`route="dcu_stream"` or `"dcu"`; the loader detects the interpolator
per chunk. To create a new file with the CPU Falcons encoder:

```bash
python -m hipfalcons.encode era5-raw.h5 era5-cubic.h5 --dataset fields \
    --library build/libfalcons_loader_cpu.so --rel-error 1e-3
```

The standard Falcons 2D tuner chooses LINEAR or CUBIC for each chunk;
no interpolation flag is needed. Rebuild both loader libraries before
using this version.

To use more CPU readers for a single existing file, for example:

```bash
HIP_VISIBLE_DEVICES=0 python examples/dcu_dataloader.py --input fields.h5 \
    --library build/libhipfalcons_loader.so --route dcu_stream \
    --num-workers 1 --reader-processes 16
```

On `dcu`, `dcu_stream`, and sequential `cpu_stream`, `reader_processes` counts
the **total CPU readers per DataLoader worker**, including the worker itself.
The default `1` uses no helpers; `0` is an alias for `1`. The example above
uses one worker plus 15 helpers, not 16 extra workers. To compare with the
original Falcons CPU decoder, change only `--route dcu_stream` to
`--route cpu_stream`: readers decode their chunks into a shared float32
buffer and the training process uploads the frames. Multiple readers with
`--shuffle` are supported on the DCU routes only. Each reader opens its own HDF5
handle, and helpers do not create GPU contexts. Additional processes can
increase CPU, host-memory and `/dev/shm` use; measure the workload and size
resources accordingly. See [reader process tuning](docs/dataloader.md#reader-processes)
for lifecycle and memory details.

## Performance on real ERA5

On one BW150 with one worker and **8 readers**, the following times measure
delivery from a warm HDF5 file to ready device tensors. Both routes read
the same CPU Falcons streams and return one history frame and one target frame.

| Real ERA5 input (0.25-degree grid) | File CR | CPU decode + upload | CPU unpack + DCU reconstruction | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 4 surface fields, 24 time steps | 16.65x | 11.21 ms/sample | 8.16 ms/sample | 1.37x |
| Same fields + temperature at 13 pressure levels, 10 time steps | 24.33x | 35.55 ms/sample | 22.45 ms/sample | 1.58x |

The CPU encoder used a per-chunk tolerance of `1e-3 * (max - min)`, as in
[Falcons HDF5 REL mode](third_party/Falcons/tools/H5Z-Falcons/README.md).
All 266 planes met that bound; device outputs matched the native CPU decoder
bit for bit. Going from 8 to 16 readers gave no consistent benefit on
these subsets. These are not training, cold-disk or full 257-channel results.
[All reader counts, data source and timings](docs/performance.md#real-era5).

## Supported Streams

| Entry point | Supported interpolation streams | Input / output |
| --- | --- | --- |
| [`build/hipfalcons_codec encode/decode`](codec/README.md) | LINEAR and CUBIC; float32 and float64; 1D through 4D; serial and OpenMP-format substreams | Standalone Falcons streams and raw arrays; decoded output is on the host |
| `build/hipfalcons input.falcons output.f32` | LINEAR only; float32; 1D through 3D; serial; radius 32768 | Standalone stream to raw float32 file on the host |
| `ERA5Loader` with `route="dcu_stream"` or `"dcu"` | LINEAR and CUBIC; float32; 2D per chunk; serial; radius 32768 | Native-endian `float32 [T,C,H,W]` HDF5 with chunks `(1,1,H,W)` and filter 40024; returns history and target tensors on the DCU |

The standalone codec also decodes Falcons 1.1.0 lossless-only streams
(CPU Zstd in `hybrid`, supported device Zstd frames in `pure`). It does not
encode lossless-only streams or implement Lorenzo/regression, no-prediction,
or lossy integer interpolation. OpenMP substreams describe the file layout;
they are not threads running on the DCU.

The HDF5 DCU routes do not accept OpenMP, Lorenzo or lossless-only chunks.
An automatically tuned CPU encoder may write an unsupported stream even
with the right HDF5 layout. For a compatible float32 HDF5 file that the CPU
Falcons decoder can read, explicitly select `route="cpu_stream"` to decode
on the CPU and upload the result. The DCU routes do not silently fall back.
See the [DataLoader input requirements](docs/dataloader.md#input-layout-and-compatibility).
Use trusted input files.

[Performance and reproducible comparisons](docs/performance.md) ·
[Optional HIP entropy experiment](entropy_trial/README.md)

## License

See [LICENSE](LICENSE), the preserved
[Falcons notices](copyright-and-BSD-license.txt), and the notices
shipped with the [bundled dependencies](third_party/README.md).
