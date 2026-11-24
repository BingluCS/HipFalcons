# Reference Measurements

For the standalone codec and its separate reader regression check, see the
[BW150 before/after measurements](codec-performance.md).

## Real ERA5

These inputs are real [WeatherBench 2 ERA5 reanalysis](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)
fields, not synthetic replicas. Both subsets start at 2020-01-01 00:00 UTC
and use consecutive six-hourly fields on the global 721 x 1440 grid.
The smaller channel set contains 2 m temperature, 10 m U/V wind and mean
sea-level pressure (24 time steps). The wider set adds temperature at all
13 pressure levels from 50 to 1000 hPa (10 time steps).

| Channels | Readers per worker | CPU decode + upload (ms/sample) | DCU route (ms/sample) | CPU time / DCU time |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 1 | 32.15 | 17.06 | 1.88x |
| 4 | 8 | 11.21 | 8.16 | 1.37x |
| 4 | 16 | 11.33 | 8.20 | 1.38x |
| 17 | 1 | 133.72 | 77.40 | 1.73x |
| 17 | 8 | 35.55 | 22.45 | 1.58x |
| 17 | 16 | 34.53 | 22.01 | 1.57x |

One BW150, 16 host cores, one fork worker, batch size one. Both routes start
at the same warm HDF5 file and end with a synchronized reduction on the
device. Sequential samples reuse one frame; each timed batch reads one new
frame. Each row is the median of three runs with one warm-up batch and gated
prefetch, timing 22 batches (4 channels) or 8 (17 channels) per run.
The CPU multi-reader measurements used the sequential reader implementation
subsequently integrated into this package; reconstruction calls the bundled
Falcons CPU decoder. The DCU route decodes entropy on the CPU. Neither model
training nor cold-storage bandwidth is measured here.

File CR is uncompressed float32 bytes divided by the entire compressed HDF5
file size: 398,684,160 / 23,942,640 = **16.65x** and 706,003,200 / 29,023,011
= **24.33x**, respectively. All 266 planes met the per-plane range-relative
bound of `1e-3`, and every returned tensor matched the CPU reconstruction
bit for bit. The small difference between 8 and 16 readers is within the
observed run-to-run variation. [Per-run timings, hashes and channel errors](../results/bw150-real-era5-20261124.json)

The timing harness subtracted the float32 extrema in double precision. The
released converter's `--rel-error` subtracts them in float32, as CPU Falcons
does, so it need not reproduce those compressed bytes exactly. In a separate
release check, the native CPU decoder recovered all 85 planes of a five-step,
17-channel subset within their per-plane bounds. At eight readers, three
timed batches from the released CPU and DCU routes matched bit for bit.

## Synthetic reader-count comparison

One BW150, one DataLoader worker, CPU affinity 0-15, batch size one.
Each sample returns one history and one target frame. The worker reads and
packs chunks itself, alongside any helpers. Rows are medians of three
interleaved runs with prefetch gated until the timer starts.

| Total readers | Additional helpers | 4 channels: delivery (ms/batch) | 257 channels: delivery (ms/batch) | 257 channels: DCU reconstruction (ms/batch) |
| ---: | ---: | ---: | ---: | ---: |
| 1 (default) | 0 | 18.91 | 1316.23 | 11.48 |
| 8 | 7 | 9.07 | 230.77 | 11.48 |
| 16 | 15 | 9.16 | 233.21 | 11.49 |

Both grids are 721 x 1440. The 257-channel file **is synthetic, not ERA5**:
it replays planes of the four-channel fixture as separate physical HDF5
chunks. After warm-up, each batch reads four or 257 new chunks respectively.
All history and target tensors matched the CPU oracle bit for bit. The
4-channel runs timed 48 batches; the 257-channel runs timed eight.

With 257 channels, eight readers delivered 5.70x the throughput of one.
Going from eight to sixteen did not improve it under this allocation.
HIP events around the batched reconstruction call exclude H2D and CPU
work; that stage took essentially the same time at all three counts.
End-to-end delivery includes a synchronized device reduction, not model
training. [Raw windows and fixture mapping](../results/bw150-reader-participants-20261124.json)

## DataLoader worker scaling on one BW150

All rows use the same synthetic HDF5 file: 50 time steps, four float32
channels, a 721 x 1440 grid, one Falcons-compressed chunk per plane. One
sample returns one history and one target frame on the DCU. The table shows
the median of three fresh processes, each measuring 48 batches of size one
after one warm-up batch. Prefetch of timed batches begins at the timer. DCU
packing in these historical runs was inline, like the current default
participant layout (`reader_processes=1`; `0` is an alias). These timings
were not measured with the current implementation. `Default` in the table
refers to CPU affinity.

| Reconstruction | Fork workers | CPU affinity | ms/batch | Samples/s |
| --- | ---: | --- | ---: | ---: |
| CPU, then upload | 4 | Default | 11.65 | 85.8 |
| DCU | 4 | Default | 5.78 | 173.0 |
| DCU | 8 | Default | 5.57 | 179.7 |
| DCU | 16 | Default | 5.81 | 172.2 |
| DCU | 8 | 16 physical cores on the DCU's NUMA node | **4.43** | **225.7** |

At equal worker count and default affinity, DCU reconstruction delivered
2.02x the CPU route's throughput. Restricting the 8-worker job to local
cores improved its throughput by a further 25.6% over unbound 8 workers.
Affinity was applied to the entire benchmark process and inherited by its
workers; this test does not separate NUMA locality from scheduling effects.
The affinity runs followed the unbound runs rather than alternating with
them. Only the DCU route was tested with affinity. All 15 windows matched the
CPU reconstruction bit for bit. Separately, the 4-worker reader supplied
device tensors for a Conv2d forward, backward and SGD update. These are
reader delivery rates, not disk bandwidth or model training rates.
[Per-window times and verification](../results/bw150-loader-scaling-20261124.json)

To run the large-grid inline configuration, select physical cores local to
the target DCU **within the job's CPU allocation**. Set `LOCAL_CPUS` to that
list; `0-15` was correct on the test host, not a portable default. Run
without the `taskset` prefix for the unbound comparison. This command does
not imply the historical timings will be reproduced by the current code.

```bash
HIP_VISIBLE_DEVICES=0 taskset -c "$LOCAL_CPUS" python -m hipfalcons.bench \
    --library build/libhipfalcons_loader.so --baseline cpu_stream --candidate dcu_stream \
    --shape 721 1440 --batch-size 1 --batches 48 --workers 8 --windows 3 \
    --reader-processes 0 --limit-mib 2048 --gate-prefetch --report bench-local.json
```

## Historical Helper-Reader Experiment

On the same four-channel synthetic grid, a separate run held the DCU,
CPU affinity (cores 0-15), native libraries and DataLoader worker count
fixed. The inline baseline ran first, before the Python packing helper
was factored out for readers; the two configurations were not interleaved.
Each row is the median of three gated, 48-batch runs. The historical
16-helper configuration used the `b8f55e8` design: 16 helper readers plus a
DataLoader worker that only coordinated and did not decode chunks. It is not
a measurement of the current implementation: today `reader_processes=16`
means the worker participates alongside 15 helpers.

| DataLoader workers | Historical helpers per worker | ms/batch | Samples/s |
| ---: | ---: | ---: | ---: |
| 1 | 0 (worker packs inline) | 18.12 | 55.2 |
| 1 | 16 | 8.84 | 113.2 |

In that historical comparison, parallel chunk preparation raised throughput
by **2.05x**.
All outputs matched the CPU reconstruction bit for bit. At batch size one,
the first sample contains eight chunks; after that only four new chunks
enter each batch. The sixteen historical helpers were configured, not all
kept busy.
The 8-worker inline configuration in the previous experiment was faster;
the best choice depends on both the channel count and the worker budget.
Separately, a smaller 257-channel file (32 x 64 planes) was used for
correctness, not throughput: reversed channel order, shuffled subsets,
early epoch close and restart, and two independent worker groups all
matched the CPU reference. No readers remained after loader shutdown.
[Raw windows and measurement settings](../results/bw150-reader-processes-20261124.json)

## Earlier reader measurements

These are synthetic runs on HPDPS-6 with one BW150, not full ERA5 runs or
training throughput. Both routes use one history and one target frame,
four float32 channels and one serial Falcons LINEAR stream per plane. The
endpoint is synchronized device output; no device-to-host copy is timed.
These historical DCU runs used inline packing, like today's participant
layout at `reader_processes=1` or `0`; they do not measure the current code.

| Grid | Batch | CPU entropy and reconstruction, 8 workers | HipFalcons, 4 workers | Throughput ratio |
| --- | ---: | ---: | ---: | ---: |
| 128 x 256 | 4 | 3.66 ms | 2.61 ms | 1.40x |
| 721 x 1440 | 1 | 10.72 ms | 6.10 ms | 1.76x |

The windows above prevent timed batches from being decoded before the
timer begins. The small and large grids used two and three windows,
respectively. Every batch matched the CPU reference bit for bit.
[Small grid](../results/cpu-pool-gated-small.json) ·
[Large grid](../results/cpu-pool-gated-era5.json)

In a different warm-prefetch run, the large-grid HipFalcons path reached
4.38 ms/batch, or 228 samples/s. The reported 7.59 GB/s counts uncompressed
bytes in the returned tensors, including the overlapping history and target
frames; it is not disk bandwidth.
[Warm-prefetch results](../results/delivery-ab-era5-4w.json)

The latest isolated CPU worker profile for that grid measured 17.09 ms
for order-1 entropy decoding out of 18.16 ms total worker time. Earlier
isolated device measurements were 0.34 ms for host-to-device transfer
and 0.39 ms for interpolation. These are different stages and runs, not
an additive breakdown of the pipelined time.
[CPU stages](../results/delivery-current-stages.json) ·
[Device stages](../results/delivery-baseline-stages.json)

To run the included paired benchmark with a temporary synthetic file:

```bash
HIP_VISIBLE_DEVICES=0 python -m hipfalcons.bench \
    --library build/libhipfalcons_loader.so --baseline cpu_stream --candidate dcu_stream \
    --shape 128 256 --batch-size 4 --batches 24 --workers 4 --windows 3 \
    --reader-processes 0 --gate-prefetch --report bench.json
```

With `--gate-prefetch` workers may prepare the warm-up batch, but no timed
batch starts reading until the timer opens the gate. Worker prefetch then
operates normally. Omit the flag to measure warm-prefetched delivery instead.
To evaluate a different number of CPU readers, change `--workers 4` to
`--workers 1` and `--reader-processes 0` to `--reader-processes 16` (or
choose another nonnegative reader count). Omitted, the count defaults to one
read-and-pack participant per DataLoader worker for `dcu` and `dcu_stream`:
the worker itself, inline. `0` is an inline alias. For `R > 1`, the worker
participates alongside `R-1` helpers. The sequential `cpu_stream` baseline
uses the same reader count; legacy `cpu` and `gpu` routes stay inline.
The output and report record the configuration
per route; fewer participants may be active when a batch has fewer chunks.
Helpers consume additional processes, host memory and shared-memory slabs,
and the best setting depends on the workload and available resources.
