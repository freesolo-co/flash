"""Cross-provider GPU allocation: the cheapest class that comfortably fits the run.

Given a base model (+ algorithm), compute the VRAM the FULL run needs — sized for the
heavier phase, GRPO, since the typical pipeline is SFT followed by GRPO — then rank
every provisionable candidate across ALL registered providers by live $/hr and pick the
cheapest:

  runpod  every Flash-provisionable class (live pricing, cached; static fallback)
  vast    live verified-datacenter offers (usable_offers' quality floors applied)

Allocation happens at SUBMIT time in the runner (offers are a volatile market);
the parse-time resolution in schema is a RunPod-static provisional for
validation/dry-run display. Offline (FLASH_SKIP_NET) the allocator degrades to exactly
``cheapest_gpu``'s deterministic static-rate answer (RunPod only — Vast is offline-off).

Provider-agnostic by construction: it walks the registered providers and asks each for
its ``gpu_classes()`` + ``hourly_rate()``; the only provider-specific knowledge is that
Vast classes come from a live offer book (collected through the provider's
``usable_offers`` and carried opaquely on ``Candidate.offer``).
"""

from __future__ import annotations

import math
import os

from flash._logging import get_logger
from flash.providers import PROVIDER_NAMES, available_providers, get_provider
from flash.providers.base import (
    Allocation,
    Candidate,
    UnsupportedGpuError,
    canonical_gpu,
    unvalidated_allowed,
)

logger = get_logger(__name__)

# "Comfortably" = the open-model VRAM estimate plus headroom, so a full SFT+GRPO run
# never lands in check_fit's "tight" band by construction. Curated catalog entries
# already carry measured minimums and are used as-is. The headroom (default 1.1 ==
# model_required_vram_gb's own default) is read at call time via vram_headroom() so allocate()
# and resolve_gpu_policy size identically and pick up a value exported after import.


def vram_headroom() -> float:
    """The sizing headroom multiplier, honored by both the submit-time allocator and the
    parse-time resolve_gpu_policy so they never disagree (PR #176 review). A validated constant."""
    return 1.1


def required_vram_gb(
    model_id: str,
    algorithm: str,
    *,
    train=None,
    thinking: bool = False,
) -> int:
    """VRAM the full run needs, sized to the run's actual knobs (context length, LoRA
    rank, batch / group size, thinking) via the shared ``model_required_vram_gb`` matrix.

    Catalog GRPO floors stay hard floors (never under-provision a validated model); the
    matrix sizes up from there for big contexts/groups and down to a cheaper card for
    small runs. Unlisted open models size from HF metadata, falling back to the 24 GB tier
    when unreadable (handled inside model_required_vram_gb)."""
    from flash.engine.vram import model_required_vram_gb

    colocate = model_required_vram_gb(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        headroom=vram_headroom(),
    )
    # Disaggregated GRPO ([train].inference_gpus>0) splits memory across the node's GPUs: the
    # inference server (full bf16 weights + KV) and the trainer (quant weights + LoRA optimizer +
    # activations) live on SEPARATE cards, so no single GPU needs the colocate total. The binding
    # per-GPU need is max(server bf16 weights + KV/overhead, the trainer's share ~= colocate minus
    # the vLLM engine/KV). Sizing to that lets a big model fit a per-role card (e.g. Qwen3.6-35B-A3B
    # served bf16 on a 94GB H100 NVL, 4-bit trainer on the other) instead of demanding the colocate
    # floor (~96GB) — which no available 2-GPU node meets — while staying FLOORED by the bf16 weights
    # so the server can never be under-provisioned into an OOM. Also unblocks 4B 1:2 on a 5090 (the
    # disaggregated server/trainer each fit 32GB though colocate 4B needs ~35GB).
    # ``train`` may be a TrainSpec (attribute) or a raw [train] dict (parse-time, before the
    # TrainSpec is built) — read inference_gpus from whichever shape so the parse-time
    # resolve_gpu_policy and the submit-time allocator size identically.
    _ig = train.get("inference_gpus") if isinstance(train, dict) else getattr(train, "inference_gpus", 0)
    try:
        _ig = int(_ig or 0)
    except (TypeError, ValueError):
        _ig = 0
    if train is not None and _ig > 0:
        pb = _params_b_for_vram(model_id)
        if pb:
            infer = max(1, _ig)
            # Inference parallelism (must match build_vllm_serve_cmd's default + FLASH_DISAGG_PARALLEL):
            #   TP (DEFAULT): the rollout server shards BOTH the bf16 weights and the KV cache across the
            #     inference GPUs (--tensor_parallel_size=infer), so each card holds ~1/infer of the
            #     server -> divide by infer (a 35B TP=2 fits 2x A100-80G instead of demanding 94GB/card).
            #   DP (MoE-only, opt-in): each inference GPU is a FULL replica
            #     (--data_parallel_size=infer, tp=1), so each card needs the WHOLE server -> do NOT
            #     divide (dividing would under-provision into an OOM, esp. for the 35B MoE).
            # infer==1 collapses both to the full-weight single-card need.
            _dp = (os.environ.get("FLASH_DISAGG_PARALLEL") or "").strip().lower() == "dp"
            _shards = 1 if _dp else infer
            # math.ceil (not int(): flooring under-provisions by up to ~1GB into an avoidable OOM on a
            # tight fit) — matches vram.py's conservative `math.ceil(est * headroom)` sizing.
            server_need = math.ceil(pb * 2.0 * 1.2 / _shards)  # per-card bf16 (full for dp, shard for tp) + ~20% KV/overhead
            disagg_need = max(server_need, math.ceil(colocate * 0.7))
            # Cap at the colocate estimate (disaggregation never needs more per GPU than the whole
            # colocated total) but NEVER below server_need: for an MoE whose colocate is sized by
            # ACTIVE params (e.g. Qwen3.6-35B-A3B), colocate can be < the full-bf16-weight server,
            # and a bare min(colocate, ...) would under-provision the inference GPU into an OOM.
            return max(server_need, min(colocate, disagg_need))
    return colocate


def _params_b_for_vram(model_id: str) -> float | None:
    """Param count (billions) for disaggregated VRAM sizing: catalog first, then HF metadata."""
    from flash.engine.vram import fetch_hf_params_b, params_b_from_str

    try:
        from flash.catalog import get_model

        pb = params_b_from_str(getattr(get_model(model_id), "params", None))
        if pb:
            return pb
    except Exception:
        pass
    try:
        return fetch_hf_params_b(model_id)
    except Exception:
        return None


def _runpod_candidates(
    need: int, pinned_gpu: str | None, allow_unval: bool, gpu_count: int = 1
) -> list[Candidate]:
    """RunPod's fitting classes priced live (static fallback).

    ``hourly_rate`` is the PER-CARD price; a multi-GPU node (``gpu_count`` > 1, the disaggregated
    rollout topology) rents that many cards, so the candidate's $/hr must be the per-card rate x
    gpu_count. Without this a multi-GPU RunPod node would be ranked (and later reported) as if it
    cost a single card, underreporting spend and skewing selection against providers whose
    multi-GPU offers are priced as the TOTAL instance (e.g. Vast's dph_total)."""
    provider = get_provider("runpod")
    n = max(1, int(gpu_count))
    out: list[Candidate] = []
    for g in provider.gpu_classes():
        if g.vram_gb < need:
            continue
        if pinned_gpu and g.name != pinned_gpu:
            continue
        if "runpod" not in g.validated_on and not allow_unval:
            continue
        out.append(
            Candidate(
                "runpod",
                g.name,
                provider.hourly_rate(g.name) * n,
                g.vram_gb,
                "runpod" in g.validated_on,
            )
        )
    return out


def _vast_candidates(
    need: int,
    pinned_gpu: str | None,
    allow_unval: bool,
    disk_gb: int,
    exclude_machine_ids,
    *,
    required: bool,
    num_gpus: int = 1,
) -> tuple[list[Candidate], tuple]:
    """Vast's fitting classes from the live offer book (cheapest per class).

    Returns (candidates, full_offer_book). ``required`` (a hard ``provider="vast"``
    pin) re-raises a search failure; otherwise it degrades to RunPod-only.
    """
    from flash.providers.base import GPU_INFO
    from flash.providers.vast.jobs import MIN_DISK_GB, usable_offers

    # When a larger class is pinned for a small model, search at the PINNED class's VRAM,
    # not the (smaller) model requirement: the offer search returns the cheapest ``limit``
    # offers from a VRAM floor, so a search at ``need`` can fill that window entirely with
    # small cheap cards and never surface the pinned larger class. ``need`` is still the
    # validity floor (allocate() rejects an undersized pin before we get here).
    search_vram = max(need, GPU_INFO[pinned_gpu].vram_gb) if pinned_gpu else need
    book: list = []
    try:
        # The offer search must use the SAME disk floor instances are actually
        # provisioned with (``create_instance``/``_effective_disk_gb``); searching at a
        # smaller requested ``disk_gb`` would surface offers that then fail to rent.
        book = usable_offers(
            search_vram, max(float(disk_gb), MIN_DISK_GB),
            exclude_machine_ids=exclude_machine_ids, num_gpus=num_gpus,
        )
    except Exception as exc:
        if required:
            raise UnsupportedGpuError(f"vast offer search failed: {exc}") from exc
        logger.warning("vast offer search failed (%s); allocating on runpod only", exc)
    out: list[Candidate] = []
    seen: set[str] = set()
    for o in book:
        if pinned_gpu and o.gpu != pinned_gpu:
            continue
        info = GPU_INFO[o.gpu]
        if "vast" not in info.validated_on and not allow_unval:
            continue
        if o.gpu in seen:  # offers are price-sorted; keep the cheapest per class
            continue
        seen.add(o.gpu)
        out.append(
            Candidate(
                "vast", o.gpu, o.dph_total, info.vram_gb, "vast" in info.validated_on, offer=o
            )
        )
    return out, tuple(book)


def allocate(
    model_id: str,
    algorithm: str,
    *,
    gpu: str | None = None,
    provider: str = "auto",
    disk_gb: int = 60,
    allow_unvalidated: bool | None = None,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
    exclude_gpu_classes: set[str] | frozenset[str] = frozenset(),
    gpu_count: int = 1,
    train=None,
    thinking: bool = False,
) -> Allocation:
    """Pick the cheapest (provider, GPU class) able to run the job across providers.

    ``gpu`` pins the class (the allocator then only picks the provider); ``provider``
    pins the substrate ("auto"/"runpod"/"vast"). Both default to fully automatic.
    ``train``/``thinking`` size the requirement to the run's actual knobs (context, group,
    rank, batch) via the matrix — long context / large group route up, small runs down.
    ``exclude_gpu_classes`` drops whole GPU classes (any provider) from the candidate pool —
    the orchestrator adds a class here after it failed ``no_capacity`` (capacity-starved /
    throttled) so re-allocation walks to the next-cheapest AVAILABLE class instead of retrying
    the same starved one on another provider.
    """
    _excluded_classes = {canonical_gpu(c) for c in exclude_gpu_classes}
    if provider not in ("auto", *PROVIDER_NAMES):
        raise UnsupportedGpuError(
            f"unknown provider {provider!r} (auto, {', '.join(PROVIDER_NAMES)})"
        )
    pinned_gpu = canonical_gpu(gpu) if gpu else None
    # The model's requirement is the floor regardless of a pin: an undersized concrete
    # pin (e.g. Qwen3-8B on a 24 GB card) must drop out of the candidate filter and
    # raise here, not provision a paid worker that OOMs. The pin only narrows WHICH
    # fitting class is chosen, never lowers the VRAM bar.
    need = required_vram_gb(model_id, algorithm, train=train, thinking=thinking)
    allow_unval = unvalidated_allowed(allow_unvalidated)
    live = available_providers()
    if provider != "auto" and provider not in live:
        raise UnsupportedGpuError(
            f"provider {provider!r} requested but not available on this control plane "
            f"(available: {', '.join(live) or '(none)'}; vast needs VAST_API_KEY)"
        )

    def _gather(pin: str | None) -> tuple[list[Candidate], tuple]:
        cands: list[Candidate] = []
        book: tuple = ()
        if provider in ("auto", "runpod") and "runpod" in live:
            cands += _runpod_candidates(need, pin, allow_unval, gpu_count=gpu_count)
        if provider in ("auto", "vast") and "vast" in live:
            vcands, book = _vast_candidates(
                need, pin, allow_unval, disk_gb, exclude_machine_ids,
                required=(provider == "vast"), num_gpus=gpu_count,
            )
            cands += vcands
        if _excluded_classes:
            cands = [c for c in cands if c.gpu not in _excluded_classes]
        return cands, book

    candidates, offer_book = _gather(pinned_gpu)
    # NEVER hard-fail on availability: a pin that no live provider can serve (the class isn't
    # offered right now, or is below ``need`` so it's filtered out) escalates to the cheapest
    # FITTING class across providers instead of raising -- "one spot larger, and so on". The
    # ``need`` floor is still absolute (we never drop below it -> no OOM), and the pin is only a
    # preference. We only raise when NOTHING >= need is available anywhere (truly unsatisfiable).
    escalated_from = None
    if not candidates and pinned_gpu is not None:
        escalated_from = pinned_gpu
        candidates, offer_book = _gather(None)
    if not candidates:
        raise UnsupportedGpuError(
            f"no allocatable GPU (>= {need} GB VRAM for {model_id}, provider={provider}, "
            f"validated_only={not allow_unval}); widen with gpu.allow_unvalidated = true, add a "
            f"provider (VAST_API_KEY), or the run genuinely exceeds every available GPU class"
        )
    if escalated_from is not None:
        order0 = {n: i for i, n in enumerate(PROVIDER_NAMES)}
        _cheapest = sorted(candidates, key=lambda c: (c.hourly_usd, c.vram_gb, order0.get(c.provider, 99)))[0]
        # WARNING level so it surfaces at default `slm train` verbosity (configure_logging is
        # WARNING) — a silently-escalated pin changes cost/hardware and operators must see it;
        # still routed through the logger (stderr), so machine-readable stdout stays clean.
        logger.warning(
            "pinned GPU %r unavailable or below need (%s GB) on provider=%s; "
            "escalated to cheapest fitting class %s (%s GB, %s)",
            escalated_from,
            need,
            provider,
            _cheapest.gpu,
            _cheapest.vram_gb,
            _cheapest.provider,
        )
    # Cheapest first; equal rates prefer less VRAM (don't burn a big card on a small
    # job), then registry order (runpod is the longest-validated substrate).
    order = {n: i for i, n in enumerate(PROVIDER_NAMES)}
    ranked = sorted(candidates, key=lambda c: (c.hourly_usd, c.vram_gb, order.get(c.provider, 99)))
    best = ranked[0]
    return Allocation(
        provider=best.provider,
        gpu=best.gpu,
        hourly_usd=best.hourly_usd,
        min_vram_gb=need,
        candidates=tuple(ranked),
        offer=best.offer,
        provider_offers=offer_book,
    )


def allocation_summary(a: Allocation) -> str:
    head = (
        f"allocated {a.gpu} on {a.provider} at ${a.hourly_usd:.2f}/hr "
        f"(need >= {a.min_vram_gb} GB VRAM"
    )
    # ``a.offer`` is an OPAQUE per-provider provisioning hint, not necessarily a Vast
    # offer — only format Vast specifics when the chosen provider is vast, so a future
    # provider's hint never misformats or raises on a missing attribute.
    if a.provider == "vast" and a.offer is not None:
        head += f", vast offer {a.offer.offer_id} in {a.offer.geolocation}"
    head += ")"
    if len(a.candidates) > 1:
        nxt = a.candidates[1]
        head += f"; next-best: {nxt.gpu}@{nxt.provider} ${nxt.hourly_usd:.2f}/hr"
    return head
