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

Use Linux, a working HIP/DTK installation, CMake 3.18+, a C++17 compiler and
the [CPU Falcons source](https://github.com/opensdc-org/Falcons). The commands
below use the tested CPU revision and a BW150 (`gfx936`) with AVX2 host CPUs.
On a different system, set `DTK_ROOT` and `HIP_ARCH` accordingly and remove
`-DENABLE_AVX2=ON` when your CPU does not support AVX2/FMA.

```bash
git clone https://github.com/opensdc-org/Falcons.git deps/Falcons
git -C deps/Falcons checkout --detach 6312f4ce2642ba9efab7dcd0f0caf6a94c6278d5

source /opt/dtk/env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHIP_ARCH=gfx936 -DENABLE_AVX2=ON
cmake --build build -j4
build/hipfalcons --self-test
```

Pass `-DFALCONS_SOURCE=/path/to/Falcons` to use a checkout you already have.
Falcons uses system Zstd when available; an offline build can pass
`-DFETCHCONTENT_SOURCE_DIR_ZSTDFETCHED=/path/to/zstd-1.4.5`.
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

See [LICENSE](LICENSE) and the preserved
[Falcons notices](copyright-and-BSD-license.txt). Falcons is a separate
build dependency; its third-party components retain their own notices.
