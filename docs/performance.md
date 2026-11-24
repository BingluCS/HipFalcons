# Reference Measurements

For the standalone codec and its separate reader regression check, see the
[BW150 before/after measurements](codec-performance.md).

## DataLoader worker scaling on one BW150

All rows use the same synthetic HDF5 file: 50 time steps, four float32
channels, a 721 x 1440 grid, one Falcons-compressed chunk per plane. One
sample returns one history and one target frame on the DCU. The table shows
the median of three fresh processes, each measuring 48 batches of size one
after one warm-up batch. Prefetch of timed batches begins at the timer.

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

To repeat the large-grid test, select physical cores local to the target
DCU **within the job's CPU allocation**. Set `LOCAL_CPUS` to that list;
`0-15` was correct on the test host, not a portable default. Run without
the `taskset` prefix for the unbound comparison.

```bash
HIP_VISIBLE_DEVICES=0 taskset -c "$LOCAL_CPUS" python -m hipfalcons.bench \
    --library build/libhipfalcons_loader.so --baseline cpu_stream --candidate dcu_stream \
    --shape 721 1440 --batch-size 1 --batches 48 --workers 8 --windows 3 \
    --limit-mib 2048 --gate-prefetch --report bench-local.json
```

## Earlier reader measurements

These are synthetic runs on HPDPS-6 with one BW150, not full ERA5 runs or
training throughput. Both routes use one history and one target frame,
four float32 channels and one serial Falcons LINEAR stream per plane. The
endpoint is synchronized device output; no device-to-host copy is timed.

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
    --shape 128 256 --batch-size 4 --batches 24 --workers 4 --windows 3 --gate-prefetch --report bench.json
```

With `--gate-prefetch` workers may prepare the warm-up batch, but no timed
batch starts reading until the timer opens the gate. Worker prefetch then
operates normally. Omit the flag to measure warm-prefetched delivery instead.
