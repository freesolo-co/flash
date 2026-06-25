"""Hyperstack's GPU classes (its rows of the shared GPU table).

The class table is provider-agnostic and lives in ``providers/base.py``. This module carves out
Hyperstack's rows (``gpu_classes()`` == every class with a ``hyperstack_name``) and owns the
friendly-name -> Hyperstack flavor translation.
"""

from __future__ import annotations

from flash.providers.base import GpuClass, UnsupportedGpuError, get_gpu_info, providers_for

__all__ = ["flavor_for", "gpu_classes"]


def gpu_classes() -> list[GpuClass]:
    """The GPU classes Hyperstack can provision (those with a ``hyperstack_name``)."""
    from flash.providers.base import GPU_INFO

    return [g for g in GPU_INFO.values() if g.hyperstack_name]


def flavor_for(name: str) -> str:
    """Hyperstack single-GPU flavor name (e.g. 'n3-L40x1') for a friendly GPU class name."""
    info = get_gpu_info(name)
    if not info.hyperstack_name:
        raise UnsupportedGpuError(
            f"{info.name} is not available on Hyperstack (providers: {', '.join(providers_for(name))})"
        )
    return info.hyperstack_name
