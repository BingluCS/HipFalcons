"""Convert a regular float32 HDF5 field to a serial LINEAR Falcons HDF5."""

import argparse
import math
import os

from ._hdf5 import create_fields, encode_plane, native_libraries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="ordinary HDF5 file readable by h5py")
    parser.add_argument("output", help="new HDF5 file (never overwritten)")
    parser.add_argument("--dataset", default="fields", help="source dataset name; output is fields")
    parser.add_argument("--library", required=True, help="path to libfalcons_loader_cpu.so")
    parser.add_argument("--abs-error", type=float, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.abs_error) or args.abs_error <= 0:
        parser.error("--abs-error must be finite and positive")

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
            first = encode_plane(encoder, np.ascontiguousarray(data[0, 0], dtype=np.float32),
                                 args.abs_error)
            stored = 0
            with h5py.File(args.output, "x") as dst:
                created = True
                fields = create_fields(dst, data.shape, first)
                for time in range(data.shape[0]):
                    for channel in range(data.shape[1]):
                        payload = first if (time, channel) == (0, 0) else encode_plane(
                            encoder, np.ascontiguousarray(data[time, channel], dtype=np.float32),
                            args.abs_error)
                        fields.id.write_direct_chunk((time, channel, 0, 0), payload, filter_mask=0)
                        stored += len(payload)
            print(f"{args.output}: shape={data.shape}, ABS={args.abs_error:g}, "
                  f"compressed chunk payload={stored} bytes")
    except Exception as e:
        if created:
            os.unlink(args.output)
        parser.exit(1, f"hipfalcons.encode: {e}\n")


if __name__ == "__main__":
    main()
