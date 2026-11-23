# Reference Measurements

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
    --shape 128 256 --batch-size 4 --batches 24 --workers 4 --windows 3 --report bench.json
```

The included benchmark uses ordinary worker prefetch. It does not
reproduce the gated timing window in the table exactly.
