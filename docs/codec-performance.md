# BW150 codec and reader performance

Retested on HPDPS-6, November 24, 2026. The baseline is the standalone HIP
codec before this optimization, with the same in-process timing harness.
Both builds used Release, AVX2, one gfx936 device and four host OpenMP
threads. Times below are milliseconds, median of three after one warm-up.
The codec starts from memory and returns to host memory; file I/O is not timed.
[Measurements, binary hashes and PyTorch results](../results/bw150-codec-ab-20261124.json)

## Compression

Synthetic float32, 4,152,960 raw bytes, CUBIC, ABS 0.0005, with sparse
outliers. The 2D field is 721 x 1440; the same bytes are interpreted as
8 x 721 x 180 for the 3D case. These are not production climate samples.

| Input | Mode | Before, ms | After, ms | Speedup | CR |
| --- | --- | ---: | ---: | ---: | ---: |
| 2D serial | hybrid | 7.932 | 6.868 | 1.15x | 2.99 |
| 2D serial | pure | 742.481 | 246.851 | 3.01x | 2.95 |
| 3D, 4 substreams | hybrid | 7.856 | 3.083 | 2.55x | 2.94 |
| 3D, 4 substreams | pure | 198.568 | 73.205 | 2.71x | 2.86 |

All four compressed outputs were byte-identical before/after and decoded by
CPU Falcons within the requested error bound (max 0.000499994). A second
input above the order-1 histogram sampling threshold also produced an
identical pure stream. The work distribution changed, not the histogram.

## Decompression

Same-stream comparisons to a host output buffer. A substream is a partition
in the file, not a physical GPU or host worker. CPU and DCU outputs matched
bit for bit; the independent CPU reference is also timed on the same files.

| Input | CPU, ms | Hybrid before / after, ms | Pure before / after, ms |
| --- | ---: | ---: | ---: |
| RLE, serial | 1.370 | 1.382 / 1.226 | 2.724 / 2.714 |
| RLE, 32 substreams | 3.966 | 2.038 / 1.936 | 2.276 / 2.305 |
| Order-1 tANS, serial | 6.094 | 16.948 / 9.472 | 124.030 / 122.973 |
| Split-FSE, serial | 3.796 | 4.612 / 4.315 | 10.685 / 10.685 |
| Split-FSE, 4 substreams | 1.181 | 5.162 / 4.004 | 10.524 / 10.613 |
| 3D with outliers, 4 substreams | 1.465 | 4.757 / 4.064 | 10.864 / 10.993 |
| Lossless-only, compressed Zstd | 5.803 | 4191.200 / 5.830 | same baseline / 4203.150 |

Hybrid now decodes lossless-only streams directly to host memory with CPU
Zstd. This row measures a routing change, not a faster device Zstd kernel.
For interpolation streams, the new hybrid reconstruction stage took
0.157–0.173 ms (24–26 GB/s of logical output); complete hybrid decoding
including entropy, transfers and reconstruction reached **0.44–3.39 GB/s**.
The CPU reference was faster for order-1 and split-FSE host output. Pure
decode did not consistently improve; changes near 1% should not be
interpreted as a reliable gain with three samples.

## Unchanged PyTorch reader

Synthetic HDF5 `[50,4,721,1440]`, four workers per route, batch size 1,
one history and one target frame. Each run timed 48 samples after the first.
The endpoint is synchronized tensors on the training device, not completion
of a training step. Median of three independent runs:

| Route | ms/batch | Samples/s |
| --- | ---: | ---: |
| CPU decode, then upload | 11.349 | 88.1 |
| CPU entropy, DCU reconstruction | 5.692 | 175.7 |

The existing DCU reader delivered 1.99x the CPU route's throughput. All
samples matched the CPU oracle. Neither reader uses the new standalone
codec; this is a regression check, not a reader speedup caused by this patch.
