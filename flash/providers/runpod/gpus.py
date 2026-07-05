"""RunPod GPU classes and friendly-name → GpuType translation."""

from __future__ import annotations

from flash.providers.base import (
    GpuClass,
    UnsupportedGpuError,
    get_gpu_info,
    providers_for,
)


def _gpu_enum():
    from runpod_flash import GpuType

    return GpuType


def gpu_classes() -> list[GpuClass]:
    """The GPU classes RunPod Flash can provision (those with a ``GpuType`` member)."""
    from flash.providers.base import gpu_classes_for

    return gpu_classes_for("enum_member")


def flash_gpu(name: str):
    """Return the RunPod Flash ``GpuType`` for a friendly GPU name."""
    info = get_gpu_info(name)
    if not info.enum_member:
        raise UnsupportedGpuError(
            f"{info.name} is not available on RunPod (providers: {', '.join(providers_for(name))})"
        )
    return getattr(_gpu_enum(), info.enum_member)
