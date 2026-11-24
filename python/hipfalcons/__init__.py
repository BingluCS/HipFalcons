"""Load Falcons-compressed HDF5 batches directly onto a HIP device."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import ERA5Loader as ERA5Loader

__all__ = ["ERA5Loader"]


def _route_readers(route, requested):
    # Both benchmarks use sequential windows; compare like-for-like reader counts.
    return max(1, requested) if route in ("cpu_stream", "dcu", "dcu_stream") else 1


def __getattr__(name):
    if name == "ERA5Loader":
        from .loader import ERA5Loader

        return ERA5Loader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
