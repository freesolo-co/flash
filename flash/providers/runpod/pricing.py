"""Static per-GPU hourly rates for RunPod-provisionable Flash classes."""

from __future__ import annotations


def static_rates() -> dict[str, float]:
    """Friendly GPU name -> static $/hr snapshot."""
    from flash.providers.base import GPU_INFO

    return {name: info.hourly_usd for name, info in GPU_INFO.items() if info.enum_member}


def hourly_rate(gpu_name: str) -> float:
    """Static $/hr for one friendly GPU name."""
    from flash.providers.base import canonical_gpu

    name = canonical_gpu(gpu_name)
    return static_rates()[name]
