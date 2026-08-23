"""Modal GPU list pricing for training Sandboxes.

Never fall back to ``GpuClass.hourly_usd``; that field is a RunPod price snapshot.
"""

from __future__ import annotations

_MODAL_RATES: dict[str, float] = {
    "A10": 1.10,
    "A100 SXM 40GB": 2.10,
    "A100 SXM": 2.50,
    "H100": 3.95,
    "H200": 4.54,
    "B200": 6.25,
}


def hourly_rate(gpu_name: str) -> float:
    """Return Modal's per-card hourly list price for one managed GPU class."""
    from flash.providers.base import UnsupportedGpuError, canonical_gpu, get_gpu_info

    name = canonical_gpu(gpu_name)
    if not get_gpu_info(name).modal_name:
        raise UnsupportedGpuError(f"modal does not offer managed gpu class {name!r}")
    try:
        return _MODAL_RATES[name]
    except KeyError:
        raise UnsupportedGpuError(f"modal pricing is unavailable for gpu class {name!r}") from None
