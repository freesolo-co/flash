"""RunPod's GPU classes + the Flash-specific bits of the shared GPU table.

The class table itself is provider-agnostic and lives in ``providers/base.py`` (one
canonical row per friendly name). This module carves out RunPod's rows
(``gpu_classes()`` == every class with a Flash ``enum_member``) and owns the
RunPod-only translation: friendly name -> Flash ``GpuType``.
"""

from __future__ import annotations

from autoslm.providers.base import (
    GpuClass,
    UnsupportedGpuError,
    get_gpu_info,
    providers_for,
)


# Lazy import so unit tests that only exercise the mapping don't pull the whole SDK
# graph unless needed. ``runpod_flash`` is a hard dependency, so this import is safe.
def _gpu_enum():
    from runpod_flash import GpuType

    return GpuType


def gpu_classes() -> list[GpuClass]:
    """The GPU classes RunPod Flash can provision (those with a ``GpuType`` member)."""
    from autoslm.providers.base import GPU_INFO

    return [g for g in GPU_INFO.values() if g.enum_member]


def flash_gpu(name: str):
    """Return the RunPod Flash ``GpuType`` for a friendly GPU name."""
    info = get_gpu_info(name)
    if not info.enum_member:
        raise UnsupportedGpuError(
            f"{info.name} is not available on RunPod (providers: {', '.join(providers_for(name))})"
        )
    return getattr(_gpu_enum(), info.enum_member)


def gpu_api_id(name: str) -> str:
    """RunPod API GPU id (the ``GpuType`` enum value, e.g. 'NVIDIA GeForce RTX 4090')."""
    return flash_gpu(name).value
