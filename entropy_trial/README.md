# Experimental HIP entropy decoding

This optional standalone verifier accepts existing CPU Falcons files. It
does not modify the encoder, file format, HDF5 filter, or DataLoader.
Use native-endian float32 LINEAR interpolation streams with the default
radius, as required by the main HipFalcons decoder.

- Method 0 (RLE/FSE): the CPU still decodes the FSE class blocks. The
  DCU reads the gamma mantissas and expands/scatters the runs. This
  prototype supports up to 32,768 codes per plane and excludes delta.
- Method 1 (order-1 tANS): the CPU builds the existing FSE tables and
  decodes the escape bypass. One DCU lane decodes each of the four
  independent streams, using the previous code to select its context.
  This prototype supports one four-stream segment per plane.
- Both leave Zstd on the CPU. For 2D, the prototype writes codes directly
  to the existing packed interpolation buffer. A rocPRIM select on the
  device finds the ordered quantizer escape positions; it is separate
  from the entropy escape bypass.

Configure HipFalcons as in the parent [build instructions](../README.md#build)
with `-DHF_BUILD_ENTROPY_TRIAL=ON`, then build `hipfalcons_entropy_trial`.
Each invocation takes one or more files with a single entropy method:

```sh
build-entropy-trial/hipfalcons_entropy_trial --host-check plane-*.falcons
HIP_VISIBLE_DEVICES=0 build-entropy-trial/hipfalcons_entropy_trial --device plane-*.falcons
```

The device check first compares every signed int16 code against
`Falcons::RleFseEncoder`. With homogeneous 2D inputs it also verifies the
final float32 fields against the CPU decoder and alternates three timed runs
per route. Inputs are in
host memory and output is synchronized on the DCU. The timed path includes
CPU preparation, pageable transfer, and reconstruction, but excludes
file I/O, initial buffer allocation, output download, and model execution.
These timings do not measure DataLoader throughput.

## BW150 results

HPDPS-6, one BW150 (gfx936), synthetic float32 fields, ABS 0.001.
The baseline is the AVX2 CPU batch packer plus the same DCU interpolation.
Each number below is the median of three runs in milliseconds.

| Input to a run | CPU entropy to device output | HIP prototype to device output | HIP entropy kernels |
| --- | ---: | ---: | ---: |
| 16 x (128 x 256), method 0 | 3.850 | 3.803 | 0.441 |
| 4 x (721 x 1440), method 1 | 19.655 | 249.613 | 244.107 |

The codes and final float32 fields matched the CPU decoder bit for bit.
Method 0 still decodes FSE classes on the CPU; offloading the expansion
largely trades CPU time for device time. Method 1 reduces upload size,
but only has 16 independent bitstreams in this batch. Four GPU blocks
each process four long state chains; dispersing the blocks was not
enough to approach the CPU time. These results do not justify enabling
either prototype in the training loader.

Raw timings: [RLE/FSE](../results/entropy-trial-20261123/rle-final.log),
[order-1 tANS/FSE](../results/entropy-trial-20261123/h1-final.log).
