# HipFalcons

HipFalcons decodes CPU Falcons streams on AMD-compatible HIP devices. Its
PyTorch DataLoader reads Falcons-compressed HDF5 and yields history and target
tensors already resident on the DCU.

**Training path:** HDF5 raw chunks → CPU Zstd and entropy decoding →
packed int16 codes → DCU interpolation → PyTorch tensors. The CPU encoder and
file format do not change. The loader does not require the HDF5 filter plugin
for reading, because it reads the stored chunk bytes instead of invoking h5py
slicing.

## Build

**Clone only HipFalcons.** The tested CPU Falcons and fallback Zstd sources
are included as ordinary files. No second clone, submodule initialization,
or dependency download is needed during the CMake build.

Use Linux, a working HIP/DTK installation, CMake 3.18+, and a C++17 compiler.
The commands below target a BW150 (`gfx936`) with an AVX2/FMA host CPU.
On a different system, adjust `DTK_ROOT` and `HIP_ARCH`; omit
`-DENABLE_AVX2=ON` when the host does not support AVX2/FMA.

```bash
git clone https://github.com/BingluCS/HipFalcons.git
cd HipFalcons

source /opt/dtk/env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHIP_ARCH=gfx936 -DENABLE_AVX2=ON
cmake --build build -j4
build/hipfalcons --self-test
```

Falcons uses system Zstd when available, otherwise it builds
`third_party/zstd`. No system Zstd development package is required.
If you built an earlier checkout, configure a **new build directory** so
the old CMake cache does not keep pointing to `deps/Falcons`.
Pass `-DFALCONS_SOURCE=/path/to/Falcons` only to use a compatible external
checkout instead of the [bundled source](third_party/README.md).
If an older DTK installation selects the wrong GCC headers, export
`HIPCC_COMPILE_FLAGS_APPEND="--gcc-install-dir=$(dirname "$(g++ -print-file-name=libgcc.a)")"`
before building. Check that `rocminfo` sees the DCU in the job that will run
the example; management-tool visibility alone does not grant compute access.

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
backward, and optimizer step on the DCU. For your own file, see the
[DataLoader example](examples/README.md) and [input and tuning guide](docs/dataloader.md).
PyTorch uses the `cuda:0` device namespace on HIP; no NVIDIA GPU is required.
Start the loader before moving the model or anything else to the DCU, so
the workers fork before HIP initialization.

## Supported Streams

| Entry point | Input | Output |
| --- | --- | --- |
| `build/hipfalcons input.falcons output.f32` | CPU-generated serial float32 LINEAR interpolation, 1D/2D/3D, radius 32768 | Raw float32 file in host storage |
| `ERA5Loader`, `route="dcu_stream"` or `"dcu"` | One HDF5 dataset `[T,C,H,W]` with chunks `(1,1,H,W)`, one Falcons filter (40024), each chunk a supported 2D stream | Two tensors `[batch,time,channel,height,width]` already on the DCU |

The HIP path does not accept cubic, Lorenzo, OpenMP, lossless-only or
non-float32 streams. An automatically tuned CPU encoder may choose an
unsupported mode. `route="cpu_stream"` explicitly selects the CPU Falcons
decoder and uploads float32 tensors; the HIP path does not silently fall
back. Use trusted input files.

[Performance and reproducible comparisons](docs/performance.md) ·
[Optional HIP entropy experiment](entropy_trial/README.md)

## License

See [LICENSE](LICENSE), the preserved
[Falcons notices](copyright-and-BSD-license.txt), and the notices
shipped with the [bundled dependencies](third_party/README.md).
