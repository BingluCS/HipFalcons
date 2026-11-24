"""Load Falcons-compressed HDF5 batches directly onto a HIP device."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import ERA5Loader as ERA5Loader

__all__ = ["ERA5Loader"]


def _route_readers(route, requested):
    # Paired benchmarks use one inline reader for non-hybrid control routes.
    return max(1, requested) if route in ("dcu", "dcu_stream") else 1


def __getattr__(name):
    if name == "ERA5Loader":
        from .loader import ERA5Loader

        return ERA5Loader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
