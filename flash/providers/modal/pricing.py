"""Modal reservations and list-price estimates, retrieved 2026-10-18.

https://modal.com/pricing lists Sandbox CPU/memory separately from Function compute.
These are estimates, not an invoice: CPU/RAM bursts, storage and network usage are not attributed to a run.
"""

from flash.providers.core.base import get_gpu_info

# minimum reservations, not limits: model loading/compilation may still burst and is billed.
CPU_PER_GPU = 2.0
MEMORY_MIB_PER_GPU = 8192
CPU_HOURLY = 0.00003942 * 3600
MEMORY_GIB_HOURLY = 0.00000667 * 3600


def hourly_rate(gpu: str) -> float:
    return (
        get_gpu_info(gpu).hourly_usd
        + CPU_PER_GPU * CPU_HOURLY
        + MEMORY_MIB_PER_GPU / 1024 * MEMORY_GIB_HOURLY
    )


def static_rates():
    from flash.providers.core.base import GPU_CLASSES

    return {g.name: hourly_rate(g.name) for g in GPU_CLASSES}
