# HipFalcons

HipFalcons decodes CPU Falcons streams on AMD-compatible HIP devices. Its
PyTorch DataLoader reads Falcons-compressed HDF5 and yields history and target
tensors already resident on the DCU.

**Training path:** HDF5 raw chunks → CPU Zstd and entropy decoding →
packed int16 codes → DCU interpolation → PyTorch tensors. The CPU encoder and
file format do not change. The loader does not require the HDF5 filter plugin
for reading, because it reads the stored chunk bytes instead of invoking h5py
slicing.

The repository also has a [standalone HIP codec](codec/README.md) for
CPU-compatible Falcons interpolation compression and decompression. It is a
separate build target and does not change the HDF5 or Python reader.

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

To use more CPU readers for a single existing file, for example:

```bash
HIP_VISIBLE_DEVICES=0 python examples/dcu_dataloader.py --input fields.h5 \
    --library build/libhipfalcons_loader.so --route dcu_stream \
    --num-workers 1 --reader-processes 16
```

For `dcu` and `dcu_stream`, `reader_processes` counts the **total CPU
read-and-pack participants per DataLoader worker**, including the worker
itself and with `--shuffle`. The default `1` packs inline without extra
processes; `0` remains an alias for inline packing. With `R > 1`, the worker
handles a shard and `R-1` helper processes handle the rest: the example above
uses the worker plus 15 helpers. CPU and legacy GPU routes remain
single-reader without helpers. Each participating process opens its own HDF5
handle, and helpers do not create GPU contexts. Additional processes can
increase CPU, host-memory and `/dev/shm` use; measure the workload and size
resources accordingly. See [reader process tuning](docs/dataloader.md#reader-processes)
for lifecycle and memory details.

## Supported Streams

| Entry point | Input | Output |
| --- | --- | --- |
| `build/hipfalcons input.falcons output.f32` | CPU-generated serial float32 LINEAR interpolation, 1D/2D/3D, radius 32768 | Raw float32 file in host storage |
| `ERA5Loader`, `route="dcu_stream"` or `"dcu"` | Native-endian `float32 [T,C,H,W]`, not fixed to an ERA5 grid. Uses chunks `(1,1,H,W)`, one Falcons filter (40024), and a supported 2D stream in each chunk. | Two tensors `[batch,time,channel,height,width]` already on the DCU |

The two reader entry points above do not accept cubic, Lorenzo, OpenMP,
lossless-only or non-float32 streams. An automatically tuned CPU encoder may choose an
unsupported mode. `route="cpu_stream"` explicitly selects the CPU Falcons
decoder and uploads float32 tensors; it still requires the specified HDF5
layout. The HIP path does not silently fall back. This is not a drop-in
replacement for readers that return additional fields such as timestamps.
Use trusted input files.

[Performance and reproducible comparisons](docs/performance.md) ·
[Optional HIP entropy experiment](entropy_trial/README.md)

## License

See [LICENSE](LICENSE), the preserved
[Falcons notices](copyright-and-BSD-license.txt), and the notices
shipped with the [bundled dependencies](third_party/README.md).
