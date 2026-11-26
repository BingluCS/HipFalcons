"""Shared CPU encoding and HDF5 layout helpers for the example and benchmark."""

import ctypes
import struct
from pathlib import Path


FALCONS_FILTER = 40024
FLOAT_PTR = ctypes.POINTER(ctypes.c_float)


def native_libraries(encoder_path, decoder_path=None):
    encoder = ctypes.CDLL(str(Path(encoder_path).resolve()))
    encoder.hf_compress_2d_f32.argtypes = [FLOAT_PTR, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double,
                                          ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
    encoder.hf_compress_2d_f32.restype = ctypes.c_int
    encoder.hf_release.argtypes = [ctypes.c_void_p]
    encoder.hf_release.restype = None
    encoder.hf_last_error.argtypes = []
    encoder.hf_last_error.restype = ctypes.c_char_p
    if decoder_path is None:
        return encoder, None
    decoder = ctypes.CDLL(str(Path(decoder_path).resolve()))
    decoder.hf_cpu_f32.argtypes = [ctypes.c_void_p, ctypes.c_size_t, FLOAT_PTR, ctypes.c_size_t]
    decoder.hf_cpu_f32.restype = ctypes.c_int
    decoder.hf_last_error.argtypes = []
    decoder.hf_last_error.restype = ctypes.c_char_p
    return encoder, decoder


def native_error(lib, operation):
    message = lib.hf_last_error()
    return RuntimeError(f"{operation} failed: " +
                        (message.decode("utf-8", "replace") if message else "no native error message"))


def encode_plane(encoder, plane, abs_error, *, interp="linear"):
    h, w = plane.shape
    output = ctypes.c_void_p()
    size = ctypes.c_size_t()
    source = plane.ctypes.data_as(FLOAT_PTR)
    if interp == "linear":
        operation = "hf_compress_2d_f32"
        rc = encoder.hf_compress_2d_f32(source, h, w, abs_error, ctypes.byref(output), ctypes.byref(size))
    elif interp in ("auto", "cubic"):
        operation = "hf_compress_2d_f32_interp"
        encode = getattr(encoder, operation, None)
        if encode is None:
            raise RuntimeError("automatic or CUBIC encoding requires a rebuilt libfalcons_loader_cpu.so")
        encode.argtypes = [FLOAT_PTR, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_double, ctypes.c_int,
                           ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t)]
        encode.restype = ctypes.c_int
        rc = encode(source, h, w, abs_error, -1 if interp == "auto" else 1,
                    ctypes.byref(output), ctypes.byref(size))
    else:
        raise ValueError("interp must be auto, linear, or cubic")
    try:
        if rc:
            raise native_error(encoder, operation)
        if not output.value or not size.value:
            raise RuntimeError("compressor returned an empty stream")
        return ctypes.string_at(output, size.value)
    finally:
        if output.value:
            encoder.hf_release(output)


def create_fields(file, shape, first_payload):
    import h5py
    import numpy as np

    header_size = first_payload[0]
    if not 0 < header_size < len(first_payload):
        raise ValueError("invalid Falcons stream header")
    header = first_payload[:header_size]
    padded = header + bytes((-header_size) % 4)
    cd_values = struct.unpack(f"={len(padded) // 4}I", padded) + (1,)
    space = h5py.h5s.create_simple(shape)
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dtype = h5py.h5t.py_create(np.dtype("float32"))
    try:
        dcpl.set_chunk((1, 1, shape[2], shape[3]))
        dcpl.set_filter(FALCONS_FILTER, h5py.h5z.FLAG_OPTIONAL, cd_values)
        fields = h5py.Dataset(h5py.h5d.create(file.id, b"fields", dtype, space, dcpl=dcpl))
        properties = fields.id.get_create_plist()
        try:
            if properties.get_nfilters() != 1:
                raise RuntimeError("expected exactly one Falcons filter")
            filter_id, flags, values, _ = properties.get_filter(0)
            if (filter_id != FALCONS_FILTER or not flags & h5py.h5z.FLAG_OPTIONAL or
                    tuple(values) != cd_values):
                raise RuntimeError("HDF5 filter metadata changed; disable plugin discovery while writing")
        finally:
            properties.close()
        return fields
    finally:
        dtype.close()
        dcpl.close()
        space.close()
