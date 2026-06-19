"""Cross-provider GPU allocation: the cheapest class that comfortably fits the run.

Given a base model (+ algorithm), compute the VRAM the FULL run needs — sized for the
heavier phase, GRPO, since the typical pipeline is SFT followed by GRPO — then rank
every provisionable candidate across ALL registered providers by live $/hr and pick the
cheapest:

  runpod  every Flash-provisionable class (live pricing, cached; static fallback)
  vast    live verified-datacenter offers (usable_offers' quality floors applied)

Allocation happens at SUBMIT time in the runner (offers are a volatile market);
the parse-time resolution in schema is a RunPod-static provisional for
validation/dry-run display. Offline (AUTOSLM_SKIP_NET) the allocator degrades to exactly
``cheapest_gpu``'s deterministic static-rate answer (RunPod only — Vast is offline-off).

Provider-agnostic by construction: it walks the registered providers and asks each for
its ``gpu_classes()`` + ``hourly_rate()``; the only provider-specific knowledge is that
Vast classes come from a live offer book (collected through the provider's
``usable_offers`` and carried opaquely on ``Candidate.offer``).
"""

from __future__ import annotations

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

    return model_required_vram_gb(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        headroom=vram_headroom(),
    )


def _runpod_candidates(need: int, pinned_gpu: str | None, allow_unval: bool) -> list[Candidate]:
    """RunPod's fitting classes priced live (static fallback)."""
    provider = get_provider("runpod")
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
                provider.hourly_rate(g.name),
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
            search_vram, max(float(disk_gb), MIN_DISK_GB), exclude_machine_ids=exclude_machine_ids
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
    train=None,
    thinking: bool = False,
) -> Allocation:
    """Pick the cheapest (provider, GPU class) able to run the job across providers.

    ``gpu`` pins the class (the allocator then only picks the provider); ``provider``
    pins the substrate ("auto"/"runpod"/"vast"). Both default to fully automatic.
    ``train``/``thinking`` size the requirement to the run's actual knobs (context, group,
    rank, batch) via the matrix — long context / large group route up, small runs down.
    """
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
            cands += _runpod_candidates(need, pin, allow_unval)
        if provider in ("auto", "vast") and "vast" in live:
            vcands, book = _vast_candidates(
                need, pin, allow_unval, disk_gb, exclude_machine_ids, required=(provider == "vast")
            )
            cands += vcands
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
