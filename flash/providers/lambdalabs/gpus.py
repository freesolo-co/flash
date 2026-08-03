"""Lambda GPU classes and friendly-name -> instance-type translation."""

from __future__ import annotations

import re

from flash.providers.base import UnsupportedGpuError, get_gpu_info, providers_for

__all__ = ["instance_type_for"]

# Lambda encodes cards-per-box in the instance-type NAME (``gpu_8x_h100_sxm5``), not as a launch
# parameter, so an N-card type is found by matching the count segment against the live catalog.
_COUNT_PREFIX = re.compile(r"^gpu_(\d+)x_")
_VRAM_GB = re.compile(r"(\d+)\s*GB", re.IGNORECASE)


def _catalog_vram_gb(entry: object) -> int | None:
    """Per-card VRAM Lambda advertises for one catalog entry, when present."""
    if not isinstance(entry, dict):
        return None
    instance = entry.get("instance_type")
    if not isinstance(instance, dict):
        return None
    for key in ("gpu_description", "description"):
        match = _VRAM_GB.search(str(instance.get(key) or ""))
        if match:
            return int(match.group(1))
    return None


def instance_type_for(name: str, gpu_count: int = 1, catalog=None) -> str:
    """Lambda instance-type name (e.g. 'gpu_1x_a10') for a friendly GPU class name.

    ``gpu_count`` > 1 rewrites the count segment (``gpu_1x_h100_pcie`` -> ``gpu_4x_h100_pcie``).

    A multi-card SKU does not always share its 1x counterpart's suffix -- the registry maps H100 to
    ``gpu_1x_h100_pcie`` while the multi-card family is ``gpu_8x_h100_sxm5`` -- so the rewrite alone
    can name a type that does not exist and make real capacity disappear. Pass ``catalog`` (the
    ``/instance-types`` map) to resolve the count against Lambda's own spelling instead. This
    function never fetches it: callers that already hold the catalog supply it, and everyone else
    stays offline. Whether the result is currently RENTABLE is still the caller's check
    (``usable_instances``).
    """
    info = get_gpu_info(name)
    if not info.lambda_name:
        raise UnsupportedGpuError(
            f"{info.name} is not available on Lambda (providers: {', '.join(providers_for(name))})"
        )
    count = int(gpu_count)
    if count <= 1:
        return info.lambda_name
    if not _COUNT_PREFIX.match(info.lambda_name):
        raise UnsupportedGpuError(
            f"lambda instance type {info.lambda_name!r} for {info.name} carries no card-count "
            f"prefix, so a {count}-card variant cannot be named"
        )
    rewritten = _COUNT_PREFIX.sub(f"gpu_{count}x_", info.lambda_name, count=1)
    if not catalog or rewritten in catalog:
        return rewritten
    # same family and card count can still name DIFFERENT memory classes (gpu_8x_a100 is 40 GB,
    # gpu_8x_a100_80gb_sxm4 is 80 GB). Match the managed class's per-card VRAM before accepting a
    # renamed suffix; otherwise catalog order can silently rent and bill the wrong box.
    family = _COUNT_PREFIX.sub("", info.lambda_name, count=1).split("_")[0]
    family_matches = []
    for entry, entry_info in catalog.items():
        match = _COUNT_PREFIX.match(entry)
        if (
            match
            and int(match.group(1)) == count
            and _COUNT_PREFIX.sub("", entry, count=1).split("_")[0] == family
        ):
            family_matches.append((entry, _catalog_vram_gb(entry_info)))
    memory_matches = sorted(entry for entry, vram_gb in family_matches if vram_gb == info.vram_gb)
    if memory_matches:
        return memory_matches[0]
    if len(family_matches) == 1:
        return family_matches[0][0]
    return rewritten
