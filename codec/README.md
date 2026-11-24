# Standalone HIP codec

`hipfalcons_codec` reads and writes individual Falcons streams, not HDF5
containers. It does not change the `hipfalcons` or `ERA5Loader` readers.
The stream contains the shape, data type and interpolation parameters;
no sidecar configuration is needed for decompression.

## Build and use

Set up the server's HIP environment first. Set `HIP_ARCH` to the device
architecture and use `DTK_ROOT` if CMake cannot find the HIP installation.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DHIP_ARCH=gfx936 -DHF_BUILD_HIP_CODEC=ON
cmake --build build --target hipfalcons_codec -j4
build/hipfalcons_codec --self-test
build/hipfalcons_codec --self-test-pure

build/hipfalcons_codec encode field.f32 field.falcons \
    --dtype f32 --shape 721 1440 --rel 1e-4 --cubic --streams 32
build/hipfalcons_codec decode field.falcons field.reconstructed.f32
```

Add `--bench` to either command to warm up once and report the median of three
in-process codec calls. Output files are written once, outside the timer. The
decode endpoint is a reconstructed host buffer, not a PyTorch device tensor.
The reported reconstruction stage is part of total time, not additional time.
See the [BW150 before/after measurements](../docs/codec-performance.md) for
compression, decompression, and the separate PyTorch reader comparison.

Raw arrays are contiguous, with the rightmost dimension varying fastest.
Use `--abs` for an absolute bound or `--rel` for a bound relative to the
data range. `--streams 1` writes a serial stream; larger counts request
logical OpenMP substreams in the output format, not OpenMP reconstruction
threads on the DCU.

## Execution modes

- `--mode hybrid` (default) does prediction, quantization and reconstruction
  on the DCU. The CPU handles Zstd and the original Falcons entropy codec
  (RLE/FSE, order-1 tANS or high/low-byte split FSE).
- `--mode pure` uses the DCU for interpolation and the supported RLE/FSE,
  order-1 tANS and split-FSE decoding paths. The host still assembles or
  inspects metadata, including reading Zstd data to locate split-FSE headers.
  The writer uses RLE/FSE or order-1 tANS, not split-FSE, and wraps its output
  in valid **raw** Zstd blocks. It does not run the CPU's Zstd compressor.
  Compression ratios and throughput can differ from the hybrid path.

Both modes support float32 and float64, 1D through 4D, and LINEAR and CUBIC
interpolation. For lossless-only Falcons 1.1.0 streams, `hybrid` uses CPU Zstd
directly into the host output (there is no interpolation or device transfer);
`pure` explicitly runs the device Zstd decoder for supported frames. Neither
interpolation mode falls back to CPU reconstruction. The codec does not yet implement lossy integer
interpolation, Lorenzo/regression, no-prediction, lossless-only encoding or
the CPU encoder's selection/tuning logic. Only pass trusted input streams.

On BW150 (`gfx936`), each mode passed 64 cross-codec cases for the two float
types, four dimensionalities, serial and four OpenMP substreams, and both
interpolators. These cases check bitwise equality against the CPU decoder and
the error bound. Also verified: REL at 32 substreams, existing CPU RLE/FSE
and order-1 tANS streams, method-2 split-FSE with one and four substreams,
and the three Zstd block types for lossless-only decoding. This is a
correctness matrix, not a throughput claim. Prefer hybrid for performance
until workloads have been measured; in particular the device Zstd decoder
processes each frame serially.

The HIP kernels were ported from the
[Falcons CUDA codec](https://github.com/JLiu-1/Falcons/tree/b564b454e98f732284fda6cbd194ae597c698d65/gpu).
See [upstream notices](notices/MODIFICATIONS.txt) for provenance and licensing.
