"""Static per-GPU hourly rates for RunPod-provisionable Flash classes."""

from __future__ import annotations


def static_rates() -> dict[str, float]:
    """Friendly GPU name -> static $/hr snapshot."""
    from flash.providers.base import static_rates_for

    return static_rates_for("enum_member")


def hourly_rate(gpu_name: str) -> float:
    """Static $/hr for one friendly GPU name."""
    from flash.providers.base import canonical_gpu, get_gpu_info

    name = canonical_gpu(gpu_name)
    rates = static_rates()
    return rates[name] if name in rates else get_gpu_info(name).hourly_usd
