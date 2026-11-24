"""Convert a regular float32 HDF5 field to a serial LINEAR Falcons HDF5."""

import argparse
import math
import os

from ._hdf5 import create_fields, encode_plane, native_libraries


def _relative_bound(plane, rel_error, time, channel):
    import numpy as np

    location = f"REL chunk (time={time}, channel={channel})"
    if not np.isfinite(plane).all():
        raise ValueError(f"{location} contains non-finite values; REL conversion does not modify them")
    # Falcons data_range<float> rounds the subtraction to float32 before multiplying by REL.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        value_range = float(np.subtract(plane.max(), plane.min(), dtype=np.float32))
    if not math.isfinite(value_range):
        raise ValueError(f"{location} has a non-finite float32 range; the DCU helper requires finite ABS")
    bound = rel_error * value_range
    if bound == 0:
        raise ValueError(f"{location} produces zero ABS; bundled Falcons would select lossless, "
                         "which the DCU helper cannot encode")
    if not math.isfinite(bound) or bound < 0:
        raise ValueError(f"{location} produces a non-finite or negative ABS; "
                         "the DCU helper requires finite positive ABS")
    return bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="ordinary HDF5 file readable by h5py")
    parser.add_argument("output", help="new HDF5 file (never overwritten)")
    parser.add_argument("--dataset", default="fields", help="source dataset name; output is fields")
    parser.add_argument("--library", required=True, help="path to libfalcons_loader_cpu.so")
    errors = parser.add_mutually_exclusive_group(required=True)
    errors.add_argument("--abs-error", type=float, help="fixed absolute error bound")
    errors.add_argument("--rel-error", type=float, help="relative bound times each float32 plane's range")
    args = parser.parse_args()
    if args.abs_error is not None and (not math.isfinite(args.abs_error) or args.abs_error <= 0):
        parser.error("--abs-error must be finite and positive")
    if args.rel_error is not None and (not math.isfinite(args.rel_error) or args.rel_error <= 0):
        parser.error("--rel-error must be finite and positive")

    # Direct writes need only the filter ID and client data, not the plugin.
    # Do not let an available plugin recompute those values in its set_local.
    os.environ["HDF5_PLUGIN_PRELOAD"] = "::"
    import h5py
    import numpy as np

    encoder, _ = native_libraries(args.library)
    created = False
    try:
        with h5py.File(args.source, "r") as src:
            data = src[args.dataset]
            if (not isinstance(data, h5py.Dataset) or data.ndim != 4 or
                    not all(data.shape) or min(data.shape[2:]) < 2 or
                    data.dtype.kind != "f" or data.dtype.itemsize != 4):
                raise ValueError("source must be float32 [T,C,H,W] with H,W >= 2")
            min_bound, max_bound = math.inf, 0.0

            def encode_at(time, channel):
                nonlocal min_bound, max_bound
                plane = np.ascontiguousarray(data[time, channel], dtype=np.float32)
                if args.rel_error is None:
                    return encode_plane(encoder, plane, args.abs_error)
                bound = _relative_bound(plane, args.rel_error, time, channel)
                try:
                    payload = encode_plane(encoder, plane, bound)
                except Exception as error:
                    raise RuntimeError(f"REL chunk (time={time}, channel={channel}, ABS={bound:.17g}) "
                                       f"failed in the DCU-compatible encoder: {error}") from error
                min_bound, max_bound = min(min_bound, bound), max(max_bound, bound)
                return payload

            first = encode_at(0, 0)
            stored = 0
            with h5py.File(args.output, "x") as dst:
                created = True
                fields = create_fields(dst, data.shape, first)
                for time in range(data.shape[0]):
                    for channel in range(data.shape[1]):
                        payload = first if (time, channel) == (0, 0) else encode_at(time, channel)
                        fields.id.write_direct_chunk((time, channel, 0, 0), payload, filter_mask=0)
                        stored += len(payload)
            mode = (f"ABS={args.abs_error:g}" if args.rel_error is None else
                    f"REL={args.rel_error:.17g}, per-chunk ABS=[{min_bound:.17g}, {max_bound:.17g}]")
            print(f"{args.output}: shape={data.shape}, {mode}, "
                  f"compressed chunk payload={stored} bytes")
    except Exception as e:
        if created:
            os.unlink(args.output)
        parser.exit(1, f"hipfalcons.encode: {e}\n")


if __name__ == "__main__":
    main()
