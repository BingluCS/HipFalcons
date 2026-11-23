"""CPU-only worker interface for packing multiple HDF5 planes directly into IPC storage."""

import ctypes
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _native(path):
    lib = ctypes.CDLL(path)
    lib.hf_pack_batch.argtypes = (
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
    )
    lib.hf_pack_batch.restype = ctypes.c_int
    lib.hf_last_error.restype = ctypes.c_char_p
    return lib


def pack_times(dataset, ordered_times, channels, height, width, library_path, *, destination=None):
    chunks = []
    for time in ordered_times:
        for channel in channels:
            mask, chunk = dataset.id.read_direct_chunk((time, channel, 0, 0))
            if mask or not chunk:
                raise ValueError(f"Missing or unfiltered Falcons chunk at ({time}, {channel})")
            chunks.append(chunk)
    n = len(chunks)
    if not n:
        raise ValueError("cannot pack an empty set of planes")
    src = (ctypes.c_void_p * n)(*(ctypes.cast(ctypes.c_char_p(b), ctypes.c_void_p) for b in chunks))
    lengths = (ctypes.c_size_t * n)(*(len(b) for b in chunks))
    offsets = (ctypes.c_int64 * n)()
    used, groups = ctypes.c_size_t(), ctypes.c_size_t()
    lib = _native(library_path)
    # A supplied destination is a borrowed pool slot. If it is too small,
    # reuse the already-read chunks and pack into an independently owned slab.
    if destination is not None:
        address, capacity = destination
        status = lib.hf_pack_batch(src, lengths, n, height, width, address, capacity,
                                   offsets, ctypes.byref(used), ctypes.byref(groups))
        if status == 0 and 0 < used.value <= capacity:
            return None, list(offsets), groups.value, used.value
        if status != 1 or used.value <= capacity:
            msg = lib.hf_last_error() or b"invalid size returned by native packer"
            raise RuntimeError("hf_pack_batch failed: " + msg.decode(errors="replace"))
        capacity = used.value
    else:
        # Typical case is a dense int16 array with a small plan, few anchors/escapes.
        capacity = n * (2 * height * width + 16384)
    for attempt in range(2):
        storage = torch.UntypedStorage._new_shared(capacity, device="cpu")
        slab = torch.empty(0, dtype=torch.uint8, device="cpu").set_(storage, 0, (capacity,), (1,))
        status = lib.hf_pack_batch(src, lengths, n, height, width, slab.data_ptr(), capacity,
                                   offsets, ctypes.byref(used), ctypes.byref(groups))
        if status == 0 and 0 < used.value <= capacity:
            result = slab[:used.value], list(offsets), groups.value
            return (*result, used.value) if destination is not None else result
        if status == 1 and attempt == 0 and used.value > capacity:
            capacity = used.value
            del slab, storage
            continue
        msg = lib.hf_last_error() or b"invalid size returned by native packer"
        raise RuntimeError("hf_pack_batch failed: " + msg.decode(errors="replace"))
