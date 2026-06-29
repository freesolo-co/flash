"""Static per-GPU hourly rates for RunPod-provisionable Flash classes."""

from __future__ import annotations


def static_rates() -> dict[str, float]:
    """Friendly GPU name -> static $/hr snapshot."""
    from flash.providers.base import GPU_INFO

    return {name: info.hourly_usd for name, info in GPU_INFO.items() if info.enum_member}


def hourly_rate(gpu_name: str) -> float:
    """Static $/hr for one friendly GPU name."""
    from flash.providers.base import canonical_gpu, get_gpu_info

    name = canonical_gpu(gpu_name)
    rates = static_rates()
    # Retired classes aren't in the static table (built from the catalog) but stay priceable for the
    # teardown/billing of an in-flight run via their retained metadata.
    return rates[name] if name in rates else get_gpu_info(name).hourly_usd
