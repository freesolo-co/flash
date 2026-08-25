"""RunPod REST GPU type identities for managed training classes."""

from __future__ import annotations

from flash.providers.base import UnsupportedGpuError, get_gpu_info, providers_for

_SERVERLESS_ENUM_MEMBERS = {
    "RTX 4090": "NVIDIA_GEFORCE_RTX_4090",
    "RTX 5090": "NVIDIA_GEFORCE_RTX_5090",
    "A100 PCIe": "NVIDIA_A100_80GB_PCIe",
    "A100 SXM": "NVIDIA_A100_SXM4_80GB",
    "H100": "NVIDIA_H100_80GB_HBM3",
    "H200": "NVIDIA_H200",
    "RTX Pro 6000": "NVIDIA_RTX_PRO_6000_BLACKWELL_SERVER_EDITION",
    "B200": "NVIDIA_B200",
}

# the deferred serverless paths still require exact pool negations until their later migration.
_POOL_MEMBERS_MISSING_FROM_SDK = {
    "ADA_80_PRO": ("NVIDIA H100 PCIe", "NVIDIA H100 NVL"),
}


class _UnenumeratedPoolMember:
    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value


def _complete_sdk_pool_tables() -> None:
    from runpod_flash.core.resources.gpu import POOLS_TO_TYPES, GpuGroup

    for pool_name, missing in _POOL_MEMBERS_MISSING_FROM_SDK.items():
        pool = getattr(GpuGroup, pool_name, None)
        if pool is None or pool not in POOLS_TO_TYPES:
            continue
        members = POOLS_TO_TYPES[pool]
        known = {getattr(member, "value", member) for member in members}
        absent = [_UnenumeratedPoolMember(value) for value in missing if value not in known]
        if absent:
            POOLS_TO_TYPES[pool] = list(members) + absent


def gpu_classes():
    """Return managed classes with an explicit RunPod REST GPU type id."""
    from flash.providers.base import gpu_classes_for

    return gpu_classes_for("runpod_gpu_type_id")


def flash_gpu(name: str):
    """Return the deferred Serverless enum while its later deletion remains out of scope."""
    info = get_gpu_info(name)
    member = _SERVERLESS_ENUM_MEMBERS.get(info.name)
    if member is None:
        raise UnsupportedGpuError(f"{info.name} is not available on RunPod Serverless")
    from runpod_flash import GpuType

    _complete_sdk_pool_tables()
    return getattr(GpuType, member)


def gpu_type_id(name: str) -> str:
    """Return the exact RunPod REST ``gpuTypeId`` for a friendly GPU name."""
    info = get_gpu_info(name)
    if not info.runpod_gpu_type_id:
        raise UnsupportedGpuError(
            f"{info.name} is not available on RunPod (providers: {', '.join(providers_for(name))})"
        )
    return info.runpod_gpu_type_id
