"""GPU allocation: the cheapest RunPod class that comfortably fits the run.

Given a base model (+ algorithm), compute the VRAM the FULL run needs — sized for
the heavier phase, GRPO, since the typical pipeline is SFT followed by GRPO — then
rank every RunPod-provisionable class by live $/hr and pick the cheapest.

Allocation happens at SUBMIT time in the orchestrator; the parse-time resolution in
config_schema is a RunPod-static provisional for validation/dry-run display. Offline
(AUTOSLM_SKIP_NET) the allocator degrades to exactly ``cheapest_gpu``'s deterministic
static-rate answer.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from autoslm._logging import get_logger
from autoslm.flash.gpus import (
    GPU_INFO,
    UnsupportedGpuError,
    canonical_gpu,
    unvalidated_allowed,
)
from autoslm.providers import available_providers

logger = get_logger(__name__)

# "Comfortably" = the open-model VRAM estimate plus headroom, so a full SFT+GRPO run
# never lands in check_fit's "tight" band by construction. Curated catalog entries
# already carry measured minimums and are used as-is.
VRAM_HEADROOM = float(os.environ.get("AUTOSLM_VRAM_HEADROOM", "1.15"))


@dataclass(frozen=True)
class Candidate:
    provider: str
    gpu: str
    hourly_usd: float
    vram_gb: int
    validated: bool


@dataclass(frozen=True)
class Allocation:
    provider: str
    gpu: str
    hourly_usd: float
    min_vram_gb: int
    candidates: tuple[Candidate, ...]  # full ranked list (retry walks this)


def required_vram_gb(model_id: str, algorithm: str) -> int:
    """VRAM the full run needs. Catalog entries carry measured minimums; open models
    get the coarse estimate sized for GRPO (the heavier phase of the usual SFT+GRPO
    pipeline) plus headroom. Unknown sizes fall back to the 24 GB tier (same as
    ``resolve_gpu_policy``)."""
    from autoslm.catalog import MODELS

    info = MODELS.get(model_id)
    if info is not None:
        return int(info.min_vram_gb)
    from autoslm.engine.vram import estimate_vram_gb, fetch_hf_params_b

    params_b = fetch_hf_params_b(model_id)
    if params_b is None:
        return 24
    sizing_algo = algorithm if os.environ.get("AUTOSLM_SIZE_FOR_ALGO") == "job" else "grpo"
    return math.ceil(estimate_vram_gb(params_b, sizing_algo) * VRAM_HEADROOM)


def allocate(
    model_id: str,
    algorithm: str,
    *,
    gpu: str | None = None,
    provider: str = "auto",
    disk_gb: int = 60,
    allow_unvalidated: bool | None = None,
) -> Allocation:
    """Pick the cheapest RunPod GPU class able to run the job.

    ``gpu`` pins the class; ``provider`` pins the substrate (RunPod only).
    """
    if provider not in ("auto", "runpod"):
        raise UnsupportedGpuError(f"unknown provider {provider!r} (auto, runpod)")
    pinned_gpu = canonical_gpu(gpu) if gpu else None
    # The model's requirement is the floor regardless of a pin: an undersized concrete
    # pin (e.g. Qwen3-8B on a 24 GB card) must drop out of the candidate filter and
    # raise here, not provision a paid worker that OOMs. The pin only narrows WHICH
    # fitting class is chosen, never lowers the VRAM bar.
    need = required_vram_gb(model_id, algorithm)
    allow_unval = unvalidated_allowed(allow_unvalidated)
    live = available_providers()
    if provider != "auto" and provider not in live:
        raise UnsupportedGpuError(
            f"provider {provider!r} requested but not available on this control plane "
            f"(available: {', '.join(live)})"
        )

    candidates: list[Candidate] = []
    from autoslm.flash.pricing import hourly_rate

    for g in GPU_INFO.values():
        if not g.enum_member or g.vram_gb < need:
            continue
        if pinned_gpu and g.name != pinned_gpu:
            continue
        if "runpod" not in g.validated_on and not allow_unval:
            continue
        candidates.append(
            Candidate("runpod", g.name, hourly_rate(g.name), g.vram_gb, "runpod" in g.validated_on)
        )

    if not candidates:
        constraint = (
            f"gpu pinned to {pinned_gpu}" if pinned_gpu else f">= {need} GB VRAM for {model_id}"
        )
        raise UnsupportedGpuError(
            f"no allocatable GPU ({constraint}, provider={provider}, "
            f"validated_only={not allow_unval}); widen with gpu.allow_unvalidated = true "
            f"or a different gpu.type"
        )
    # Cheapest first; equal rates prefer less VRAM (don't burn a big card on a small job).
    ranked = sorted(candidates, key=lambda c: (c.hourly_usd, c.vram_gb))
    best = ranked[0]
    return Allocation(
        provider=best.provider,
        gpu=best.gpu,
        hourly_usd=best.hourly_usd,
        min_vram_gb=need,
        candidates=tuple(ranked),
    )


def allocation_summary(a: Allocation) -> str:
    head = (
        f"allocated {a.gpu} on {a.provider} at ${a.hourly_usd:.2f}/hr "
        f"(need >= {a.min_vram_gb} GB VRAM)"
    )
    if len(a.candidates) > 1:
        nxt = a.candidates[1]
        head += f"; next-best: {nxt.gpu}@{nxt.provider} ${nxt.hourly_usd:.2f}/hr"
    return head
