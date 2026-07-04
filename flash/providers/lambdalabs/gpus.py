"""Lambda GPU classes and friendly-name -> instance-type translation."""

from __future__ import annotations

from flash.providers.base import UnsupportedGpuError, get_gpu_info, providers_for

__all__ = ["instance_type_for"]


def instance_type_for(name: str) -> str:
    """Lambda instance-type name (e.g. 'gpu_1x_a10') for a friendly GPU class name."""
    info = get_gpu_info(name)
    if not info.lambda_name:
        raise UnsupportedGpuError(
            f"{info.name} is not available on Lambda (providers: {', '.join(providers_for(name))})"
        )
    return info.lambda_name
