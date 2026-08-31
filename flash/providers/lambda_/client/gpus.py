"""Lambda GPU classes and friendly-name -> instance-type translation."""

from __future__ import annotations

import re

from flash.providers.core._decoding import (
    MISSING_PROVIDER_FIELD,
    MalformedProviderFieldError,
    decode_finite_number,
)
from flash.providers.core.base import UnsupportedGpuError, get_gpu_info, providers_for

__all__ = ["instance_type_disk_gb", "instance_type_for"]

# Lambda encodes cards-per-box in the instance-type NAME (``gpu_8x_h100_sxm5``), not as a launch
# parameter, so an N-card type is found by matching the count segment against the live catalog.
_COUNT_PREFIX = re.compile(r"^gpu_(\d+)x_")
_VRAM_GB = re.compile(r"(\d+)\s*GB", re.IGNORECASE)


def _catalog_positive_int(text: str, field: str) -> int:
    try:
        value = int(text)
    except (OverflowError, ValueError) as exc:
        raise MalformedProviderFieldError("lambda", field, "a positive integer") from exc
    if value <= 0:
        raise MalformedProviderFieldError("lambda", field, "a positive integer")
    return value


def _catalog_object(value: object, field: str) -> dict | None:
    """One rung of a catalog walk: absent or null is unknown, any other shape is malformed."""
    if value is MISSING_PROVIDER_FIELD or value is None:
        return None
    if not isinstance(value, dict):
        raise MalformedProviderFieldError("lambda", field, "an object or null")
    return value


def _catalog_child(parent: dict | None, key: str, field: str) -> dict | None:
    """Descend one key, carrying an already-unknown parent through as unknown."""
    if parent is None:
        return None
    return _catalog_object(parent.get(key, MISSING_PROVIDER_FIELD), field)


def _catalog_vram_gb(entry: object, instance_type: str) -> int | None:
    """Per-card VRAM Lambda advertises for one catalog entry.

    None means the catalog states no memory for this entry, and nothing else. A present entry whose
    shape or digits cannot be trusted raises instead: the caller reads None as PROOF that Lambda
    published no memory class and falls back to a renamed suffix on the strength of it, so
    unparseable metadata answering to the same None would rent an unverified memory class.
    """
    instance = _catalog_child(
        _catalog_object(entry, instance_type),
        "instance_type",
        f"{instance_type}.instance_type",
    )
    if instance is None:
        return None
    for key in ("gpu_description", "description"):
        field = f"{instance_type}.{key}"
        raw = instance.get(key, MISSING_PROVIDER_FIELD)
        if raw is MISSING_PROVIDER_FIELD or raw is None:
            continue
        if not isinstance(raw, str):
            raise MalformedProviderFieldError("lambda", field, "a string or null")
        match = _VRAM_GB.search(raw)
        if match:
            return _catalog_positive_int(match.group(1), field)
    return None


def instance_type_disk_gb(catalog, instance_type: str) -> float | None:
    """Fixed disk Lambda ships with one instance type, or None when the catalog does not report it.

    Lambda sells storage WITH the SKU: unlike Vast's create-time ``disk_gb`` or RunPod's
    ``containerDiskInGb`` there is no launch parameter to raise, so this number is the only thing a
    run's ``gpu.disk_gb`` floor can be checked against. ``storage_gib`` is compared to the run's GB
    floor unconverted, which errs on the strict side (a GiB is larger) and never over-promises.

    None means unknown, never zero: a caller must not invent a refusal the catalog cannot prove.
    """
    if not isinstance(catalog, dict):
        return None
    entry = _catalog_object(catalog.get(instance_type, MISSING_PROVIDER_FIELD), instance_type)
    instance = _catalog_child(entry, "instance_type", f"{instance_type}.instance_type")
    specs = _catalog_child(instance, "specs", f"{instance_type}.specs")
    if specs is None:
        return None
    decoded: list[float] = []
    for key in ("storage_gib", "storage_gb"):
        raw = specs.get(key, MISSING_PROVIDER_FIELD)
        if raw is MISSING_PROVIDER_FIELD or raw is None:
            continue
        field = f"{instance_type}.{key}"
        storage = decode_finite_number(raw, provider="lambda", field=field)
        assert isinstance(storage, float)
        if storage <= 0:
            raise MalformedProviderFieldError("lambda", field, "a positive finite number")
        decoded.append(storage)
    return decoded[0] if decoded else None


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
        # family first: decoding raises, so testing the count ahead of the family would let a corrupt
        # entry from an UNRELATED family abort a lookup that never concerned it.
        if not match or _COUNT_PREFIX.sub("", entry, count=1).split("_")[0] != family:
            continue
        if _catalog_positive_int(match.group(1), f"{entry}.gpu_count") == count:
            family_matches.append((entry, _catalog_vram_gb(entry_info, entry)))
    memory_matches = sorted(entry for entry, vram_gb in family_matches if vram_gb == info.vram_gb)
    if memory_matches:
        return memory_matches[0]
    if len(family_matches) == 1 and family_matches[0][1] is None:
        # Preserve the suffix fallback only when the catalog PROVES it published no memory class for
        # the sole candidate -- _catalog_vram_gb raises on unparseable metadata rather than
        # answering None, so an unreadable description can no longer unlock this. An explicit
        # mismatch is a different managed class, not merely a renamed spelling.
        return family_matches[0][0]
    return rewritten
