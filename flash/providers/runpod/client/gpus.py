"""RunPod REST GPU type identities for managed Secure Cloud Pods."""

from __future__ import annotations

from flash.providers.core.base import GpuClass, UnsupportedGpuError, get_gpu_info, providers_for


def gpu_classes() -> list[GpuClass]:
    """Return managed classes with an explicit RunPod REST GPU type id."""
    from flash.providers.core.base import gpu_classes_for

    return gpu_classes_for("runpod_gpu_type_id")


def gpu_type_id(name: str) -> str:
    """Return the exact RunPod REST ``gpuTypeId`` for a friendly GPU name."""
    info = get_gpu_info(name)
    if not info.runpod_gpu_type_id:
        raise UnsupportedGpuError(
            f"{info.name} is not available on RunPod (providers: {', '.join(providers_for(name))})"
        )
    return info.runpod_gpu_type_id
