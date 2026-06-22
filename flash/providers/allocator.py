"""Cross-provider GPU allocation: the cheapest class that comfortably fits the run.

Given a base model (+ algorithm), compute the VRAM the FULL run needs — sized for the
heavier phase, GRPO, since the typical pipeline is SFT followed by GRPO — then rank
every provisionable candidate across ALL registered providers by live $/hr and pick the
cheapest:

  runpod  every Flash-provisionable class (live pricing, cached; static fallback)
  vast    live verified-datacenter offers (usable_offers' quality floors applied)

Allocation happens at SUBMIT time in the runner (offers are a volatile market);
the parse-time resolution in schema is a RunPod-static provisional for
validation/dry-run display. With no live pricing/offers reachable (no network, or no
``VAST_API_KEY``) the allocator degrades to exactly ``cheapest_gpu``'s deterministic
static-rate answer (RunPod only — Vast is off without its key).

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
)

logger = get_logger(__name__)

# "Comfortably" = the open-model VRAM estimate plus headroom, so a full SFT+GRPO run
# never lands in check_fit's "tight" band by construction. Curated catalog entries
# already carry measured minimums and are used as-is. The headroom (default 1.1 ==
# model_required_vram_gb's own default) is read at call time via vram_headroom() so allocate()
# and the parse-time provisional_gpu size identically.


def vram_headroom() -> float:
    """The sizing headroom multiplier, honored by both the submit-time allocator and the
    parse-time provisional_gpu so they never disagree. A constant."""
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
    # inference server (full bf16 weights + KV) and the trainer (bf16 weights + LoRA optimizer +
    # activations) live on SEPARATE cards, so no single GPU needs the colocate total. The binding
    # per-GPU need is max(server bf16 weights + KV/overhead, the trainer's share ~= colocate minus
    # the vLLM engine/KV). Sizing to that lets a big model fit a per-role card (e.g. Qwen3.6-35B-A3B
    # served bf16 on a 94GB H100 NVL, bf16 LoRA trainer on the other) instead of demanding the colocate
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
            # Gate `dp` on the catalog's is_moe exactly as schema.py / engine.worker.run_rl do: vLLM
            # rejects offline data parallelism for DENSE models, so the worker DOWNGRADES a dense `dp`
            # request back to TP (shards the server). Sizing must mirror that downgrade — else a dense
            # split is sized as a full per-card replica (_shards=1) the worker will never run, routing
            # it to needlessly larger / costlier GPUs. Only honor dp here when the model is actually MoE.
            _dp = (
                (os.environ.get("FLASH_DISAGG_PARALLEL") or "").strip().lower() == "dp"
                and _is_moe_model(model_id)
            )
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


def _is_moe_model(model_id: str) -> bool:
    """Whether ``model_id`` is MoE, mirroring engine.worker.run_rl's `dp` gate.

    Drives the disaggregated `dp` sizing gate: vLLM only honors offline data parallelism for MoE
    models, so the worker downgrades a dense `dp` request to TP. Sizing must match that downgrade —
    if we size a `dp` MoE as TP (divide by infer) the worker keeps full replicas and OOMs.

    The catalog is consulted first, but an open model (``model_policy="allow"``, not listed) the
    worker would still detect as MoE via ``AutoConfig`` is invisible to a catalog-only test — so we
    fall back to the same expert-count HF probe the worker uses (``num_experts`` /
    ``n_routed_experts`` / ``model_type`` contains "moe"), but with ``trust_remote_code=False``:
    the control plane must not run a model's arbitrary remote config code merely to size a rollout.

    The only caller gates on ``FLASH_DISAGG_PARALLEL=dp`` already, so this only matters when ``dp``
    is requested. There the SAFE-to-over-size answer is True (size each card for a full replica),
    because the worker — which probes with ``trust_remote_code=True`` — runs a detected MoE as ``dp``
    (whole server per card): mis-sizing such a model as dense/TP would divide VRAM and OOM. So a
    custom-architecture MoE whose config ONLY loads with remote code (the probe below raises the
    "requires you to execute custom code" error) is reported MoE here — we never run the remote code,
    but we trust the worker will and size for the replica it will run. A genuinely-dense model wrongly
    caught this way is only OVER-provisioned (the worker downgrades dp->tp, needing LESS), never OOM.
    Other uncertainty (offline / FLASH_SKIP_NET / config unreadable) stays dense/TP — the worker also
    downgrades to tp on an unreadable probe, so sharded sizing matches what it actually runs."""
    try:
        from flash.catalog import get_model

        if bool(getattr(get_model(model_id), "is_moe", False)):
            return True
    except Exception:
        pass
    # Open/unlisted model: mirror engine.worker.run_rl's AutoConfig probe so submit sizes a `dp`
    # rollout as full per-card replicas (no divide) exactly as the worker will run it. Offline
    # (FLASH_SKIP_NET) we can't probe, so stay on the dense/TP side — consistent with the worker.
    if os.environ.get("FLASH_SKIP_NET"):
        return False
    try:
        from transformers import AutoConfig

        # trust_remote_code=False on the CONTROL PLANE: this sizing probe must not execute a
        # model's arbitrary remote config code just to read expert counts.
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
        return bool(
            getattr(cfg, "num_experts", 0) or getattr(cfg, "n_routed_experts", 0)
        ) or "moe" in (getattr(cfg, "model_type", "") or "").lower()
    except Exception as exc:
        # A custom-architecture MoE whose config ONLY loads with remote code raises a recognizable
        # "requires you to execute custom code" ValueError here. We won't run that code on the
        # control plane, but the worker WILL (trust_remote_code=True) and will run it as `dp` (whole
        # server per card). Report MoE so dp sizing provisions a full replica instead of dividing
        # VRAM into an OOM. (A genuinely-dense model misjudged this way is only over-provisioned —
        # the worker downgrades dp->tp, needing less.) Any other failure stays dense/TP, matching the
        # worker's own unreadable-probe downgrade to tp.
        return "execute custom code" in str(exc)


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


def _is_excluded(provider: str, gpu: str, exclude_gpu_classes: frozenset) -> bool:
    """Whether a (provider, gpu) candidate is excluded by ``exclude_gpu_classes``.

    Two exclusion forms coexist in the same set:
      - a BARE class string (e.g. ``"A100 SXM"``) excludes the class MARKET-WIDE, on every provider
        — the MIG-walk form (a class RunPod fulfilled with a MIG slice is bad everywhere).
      - a ``(provider, gpu)`` TUPLE excludes the class on ONLY that provider — the no_capacity form:
        a RunPod IN_QUEUE starvation says nothing about the SAME class on Vast, so only RunPod's
        offer of it is dropped while Vast's survives.
    """
    return gpu in exclude_gpu_classes or (provider, gpu) in exclude_gpu_classes


def _runpod_candidates(need: int, exclude_gpu_classes: frozenset) -> list[Candidate]:
    """RunPod's fitting, live-validated classes priced live (static fallback).

    Restricted to the validated pool (``g.validated``): the deployed control plane rejects a submit
    for any non-validated class, so allocating one would only fail at submit time. Any class
    excluded by ``exclude_gpu_classes`` (bare string = market-wide; ``("runpod", gpu)`` tuple =
    RunPod-scoped) is dropped — used to walk OFF a MIG-prone class market-wide, or off a
    capacity-starved RunPod class while leaving the same class on another provider selectable."""
    provider = get_provider("runpod")
    return [
        Candidate("runpod", g.name, provider.hourly_rate(g.name), g.vram_gb)
        for g in provider.gpu_classes()
        if g.vram_gb >= need
        and g.validated
        and not _is_excluded("runpod", g.name, exclude_gpu_classes)
    ]


def _vast_candidates(
    need: int, disk_gb: int, exclude_machine_ids, exclude_gpu_classes: frozenset
) -> tuple[list[Candidate], tuple]:
    """Vast's fitting, live-validated classes from the live offer book (cheapest per class).

    Returns (candidates, full_offer_book). Restricted to the validated pool (``GPU_INFO[gpu]
    .validated``) — the deployed control plane rejects a submit for any non-validated class. A Vast
    offer-search failure is caught and degrades to the other providers (RunPod): it is non-fatal AS
    LONG AS another provider can supply a fitting class. If Vast is the only available provider, the
    empty result means ``allocate`` then raises (nothing across any provider fits) — i.e. it is only
    fatal when Vast was the sole option.
    """
    from flash.providers.base import GPU_INFO
    from flash.providers.vast.jobs import MIN_DISK_GB, usable_offers

    book: list = []
    try:
        # The offer search must use the SAME disk floor instances are actually provisioned with
        # (a smaller requested ``disk_gb`` would surface offers that then fail to rent).
        book = usable_offers(
            need, max(float(disk_gb), MIN_DISK_GB), exclude_machine_ids=exclude_machine_ids
        )
    except Exception as exc:
        logger.warning("vast offer search failed (%s); allocating on runpod only", exc)
    out: list[Candidate] = []
    seen: set[str] = set()
    for o in book:
        if o.gpu in seen:  # offers are price-sorted; keep the cheapest per class
            continue
        if not GPU_INFO[o.gpu].validated:  # only offer live-validated classes the server accepts
            continue
        # walked OFF this class (MIG-prone market-wide via a bare string, or this provider's
        # capacity-starved offer via a ("vast", gpu) tuple) on the infra retry
        if _is_excluded("vast", o.gpu, exclude_gpu_classes):
            continue
        seen.add(o.gpu)
        out.append(Candidate("vast", o.gpu, o.dph_total, GPU_INFO[o.gpu].vram_gb, offer=o))
    return out, tuple(book)


def allocate(
    model_id: str,
    algorithm: str,
    *,
    disk_gb: int = 60,
    exclude_machine_ids: set[int] | frozenset[int] = frozenset(),
    exclude_gpu_classes: set | frozenset = frozenset(),
    train=None,
    thinking: bool = False,
) -> Allocation:
    """Pick the cheapest (provider, GPU class) able to run the job across ALL providers.

    There is no GPU pin and no provider pin — every fitting, LIVE-VALIDATED class on every live
    provider is eligible, and the cheapest wins. Allocation is restricted to the validated pool
    (``GpuClass.validated``) because the deployed control plane rejects a submit for any
    non-validated class, so picking the absolute-cheapest fitting class (e.g. an unvalidated "RTX
    2000 Ada") would just make the server refuse the run. ``train``/``thinking`` size the
    requirement to the run's actual knobs (context, group, rank, batch) via the matrix.

    ``exclude_gpu_classes`` drops candidates from the pool in TWO forms that coexist in one set:
    a BARE class string excludes the class MARKET-WIDE (the MIG-walk: a class RunPod fulfilled with a
    Blackwell MIG slice is bad on every provider, so the retry re-allocates to a DIFFERENT validated
    class a consumer card can't MIG-slice); a ``(provider, gpu)`` TUPLE excludes the class on ONLY
    that provider (the no_capacity walk: a RunPod IN_QUEUE starvation says nothing about the SAME
    class on Vast, so only RunPod's offer of it is dropped while Vast's stays selectable).
    """
    exclude_gpu_classes = frozenset(exclude_gpu_classes)
    need = required_vram_gb(model_id, algorithm, train=train, thinking=thinking)
    live = available_providers()
    candidates: list[Candidate] = []
    offer_book: tuple = ()
    if "runpod" in live:
        candidates += _runpod_candidates(need, exclude_gpu_classes)
    if "vast" in live:
        vcands, offer_book = _vast_candidates(
            need, disk_gb, exclude_machine_ids, exclude_gpu_classes
        )
        candidates += vcands
    # Authoritative exclusion at the allocate level (the per-provider helpers also pre-filter, so
    # the real path never even fetches an excluded offer; this re-applies it on the combined list so
    # a bare-string market-wide ban or a (provider, gpu)-scoped ban holds regardless of how the
    # candidates were produced).
    if exclude_gpu_classes:
        candidates = [
            c for c in candidates if not _is_excluded(c.provider, c.gpu, exclude_gpu_classes)
        ]
    if not candidates:
        excluded = (
            f" (excluding GPU classes {sorted(map(repr, exclude_gpu_classes))})"
            if exclude_gpu_classes
            else ""
        )
        raise UnsupportedGpuError(
            f"no allocatable GPU (>= {need} GB VRAM for {model_id}) on any live provider "
            f"({', '.join(live) or '(none)'}){excluded}; add VAST_API_KEY for more classes, or the "
            "run genuinely exceeds every available GPU class"
        )
    # Cheapest first; equal rates prefer less VRAM (don't burn a big card on a small job),
    # then registry order.
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
