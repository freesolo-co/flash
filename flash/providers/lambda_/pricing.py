"""Which Lambda instance types Flash rents, and what they cost.

Lambda prices whole instances (``price_cents_per_hour``) and encodes the card count in the type
name (``gpu_8x_h100_sxm5``). Live catalog prices always win; the static tables below only answer
when the catalog cannot be read (cost annotation, offline quotes). Lambda sells no H200.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from flash.providers.core.base import canonical_gpu


@dataclass(frozen=True)
class LambdaGpu:
    # static per-card list price (USD/hour) of the cheapest 1x type, for when the catalog is down
    hourly_usd: float
    # card counts Lambda sells for this class; the live catalog decides which are rentable now
    counts: tuple[int, ...]


# Priority / display order. Snapshot of Lambda list prices, 2026-11-15.
LAMBDA_GPUS: dict[str, LambdaGpu] = {
    "A10": LambdaGpu(1.29, (1,)),
    "A100 SXM 40GB": LambdaGpu(1.99, (1, 2, 4, 8)),
    "H100": LambdaGpu(3.79, (1, 2, 4, 8)),
    "B200": LambdaGpu(7.99, (1, 2, 4, 8)),
}

MAX_LAMBDA_CARDS = max(count for shape in LAMBDA_GPUS.values() for count in shape.counts)

_TYPE_NAME = re.compile(r"gpu_(\d+)x_([a-z0-9]+(?:_[a-z0-9]+)*)")
_VRAM_GB = re.compile(r"(\d+)\s*GB", re.IGNORECASE)


def classify(name: object, instance: object) -> tuple[str, int] | None:
    """``(gpu class, card count)`` for a catalog instance type Flash may rent, else None.

    Excludes anything that is not x86_64 (GH200 is ARM64: the worker image is amd64), any family
    without a clean Flash class, and any type whose advertised per-card memory or card count
    disagrees with its name.
    """
    if not isinstance(name, str) or not isinstance(instance, dict):
        return None
    match = _TYPE_NAME.fullmatch(name)
    if not match:
        return None
    count = int(match.group(1))
    tail = match.group(2)
    family = tail.split("_")[0]
    architecture = instance.get("architecture")
    if architecture is not None and architecture != "x86_64":
        return None
    if family.startswith("gh"):
        return None  # Grace Hopper is ARM64 even when the catalog omits the architecture
    described = None
    for key in ("gpu_description", "description"):
        found = _VRAM_GB.search(str(instance.get(key) or ""))
        if found:
            described = int(found.group(1))
            break
    if family == "h100":
        gpu = "H100"
    elif family == "b200":
        gpu = "B200"
    elif family == "a10":
        gpu = "A10"
    elif family == "a100":
        if "80gb" in tail or (described is not None and described != 40):
            return None  # the 80 GB A100 is not a class Lambda training rents
        gpu = "A100 SXM 40GB"
    else:
        return None
    from flash.providers.core.base import GPU_INFO

    if described is not None and described < GPU_INFO[gpu].vram_gb:
        return None
    specs = instance.get("specs")
    gpus = specs.get("gpus") if isinstance(specs, dict) else None
    if gpus is not None and (type(gpus) is not int or gpus != count):
        return None
    if count not in LAMBDA_GPUS[gpu].counts:
        return None
    return gpu, count


def static_hourly_rate(gpu: str) -> float:
    """Per-card list price for one class when the live catalog is unavailable."""
    return LAMBDA_GPUS[canonical_gpu(gpu)].hourly_usd


def hourly_rate(gpu: str, *, gpu_count: int = 1, deadline_at: float | None = None) -> float:
    """Per-card USD/hour of the cheapest ``gpu_count``-card type of ``gpu`` Lambda lists.

    Reads the live catalog (cached briefly) and falls back to the static table. The rate is per
    card, matching ``Candidate.hourly_usd``.
    """
    name = canonical_gpu(gpu)
    if name not in LAMBDA_GPUS:
        raise KeyError(f"Lambda does not rent {name}")
    count = max(1, int(gpu_count))
    try:
        from flash.providers.lambda_.api import offers

        prices = [
            offer.hourly_usd / offer.count
            for offer in offers(deadline_at=deadline_at, require_capacity=False)
            if offer.gpu == name and offer.count == count
        ]
        if prices:
            return min(prices)
    except Exception:
        pass
    return static_hourly_rate(name)


def lambda_unsupported_message(model_id: str, algorithm: str, need: float) -> str:
    """Why a run no Lambda shape can hold is refused: Lambda rents only these classes and counts."""
    from flash.providers.core.base import GPU_INFO

    widest = ", ".join(
        f"{max(shape.counts)}x {name} ({max(shape.counts) * GPU_INFO[name].vram_gb} GB)"
        for name, shape in LAMBDA_GPUS.items()
    )
    return (
        f"Lambda is not supported for {model_id} with algorithm {algorithm!r}: it needs at least "
        f"{need} GB of GPU memory, which no Lambda shape can hold (widest shapes: {widest}; "
        "Lambda sells no H200). Lower max_context_tokens / max_completion_tokens / batch size, "
        "or pick a smaller model"
    )
