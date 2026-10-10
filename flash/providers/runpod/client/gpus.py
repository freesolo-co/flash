"""RunPod GPU classes and friendly-name → GpuType translation."""

from __future__ import annotations

from flash.providers.core.base import (
    GpuClass,
    UnsupportedGpuError,
    get_gpu_info,
    providers_for,
)


def _gpu_enum():
    from runpod_flash import GpuType

    return GpuType


# ADA_80_PRO contains H100 PCIe, HBM3, and NVL, but the SDK omits members when serializing pins.
# emit live RunPod gpu-type ids as negations so an exact H100 request cannot widen to another
# member.
# only pools flash pins are corrected; SDK enum docstrings are unavailable at runtime.
_POOL_MEMBERS_MISSING_FROM_SDK = {
    "ADA_80_PRO": ("NVIDIA H100 PCIe", "NVIDIA H100 NVL"),
}


class _UnenumeratedPoolMember:
    """A pool member the SDK's GpuType enum lacks, carried only so its negation is emitted.

    to_gpu_ids_str reads ``.value`` off each pool member and emits ``-{value}`` for the ones the
    caller did not ask for, which is all this needs to participate in. It deliberately does NOT
    compare equal to any GpuType, so _pool_from_gpu_type still resolves real types to their pool.
    """

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"_UnenumeratedPoolMember({self.value!r})"


def _complete_sdk_pool_tables() -> None:
    """Add the pool members the SDK omits, so an explicit GPU pin cannot widen to its whole pool.

    Idempotent, and a no-op for any member the installed SDK already knows: runpod-flash is an
    unpinned dependency, so a release that fixes its own table must not produce a double negation.
    """
    from runpod_flash.core.resources.gpu import POOLS_TO_TYPES, GpuGroup

    for pool_name, missing in _POOL_MEMBERS_MISSING_FROM_SDK.items():
        pool = getattr(GpuGroup, pool_name, None)
        if pool is None or pool not in POOLS_TO_TYPES:
            continue
        members = POOLS_TO_TYPES[pool]
        known = {getattr(m, "value", m) for m in members}
        absent = [_UnenumeratedPoolMember(v) for v in missing if v not in known]
        if absent:
            POOLS_TO_TYPES[pool] = list(members) + absent


def gpu_classes() -> list[GpuClass]:
    """The GPU classes RunPod Flash can provision (those with a ``GpuType`` member)."""
    from flash.providers.core.base import gpu_classes_for

    return gpu_classes_for("enum_member")


def flash_gpu(name: str):
    """Return the RunPod Flash ``GpuType`` for a friendly GPU name.

    Completes the SDK's pool tables first so the returned type serializes to a pin that names
    exactly this card, not its whole pool -- see _POOL_MEMBERS_MISSING_FROM_SDK.
    """
    info = get_gpu_info(name)
    if not info.enum_member:
        raise UnsupportedGpuError(
            f"{info.name} is not available on RunPod (providers: {', '.join(providers_for(name))})"
        )
    _complete_sdk_pool_tables()
    return getattr(_gpu_enum(), info.enum_member)
