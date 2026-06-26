"""GPU allocation: the cheapest fitting class across the active providers.

Given a base model (+ algorithm), compute the VRAM the FULL run needs — sized for the
heavier phase, GRPO, since the typical pipeline is SFT followed by GRPO — then rank every
fitting candidate by $/hr and pick the cheapest:

  runpod      every validated Flash-provisionable class (static $/hr)
  lambda      every fitting class that currently has LIVE regional capacity (live $/hr); opt-in,
              available only when LAMBDA_API_KEY is set on the control plane
  hyperstack  every fitting class whose single-GPU flavor currently has STOCK (static $/hr); opt-in,
              available only when HYPERSTACK_API_KEY is set on the control plane

RunPod's cheaper static rates almost always win on price, so the instance providers (Lambda,
Hyperstack) join the ranked list as capacity COMPLEMENTS: when RunPod's cheapest fitting class is
out of capacity (THROTTLED / queue backstop), the runner's gpu-walk steps down the ranked list and
reaches an in-capacity instance class. Both instance providers are capacity-filtered up front
(``_lambda_candidates`` / ``_hyperstack_candidates`` only offer a class a region/flavor can supply
right now), so the walk never lands on a class that would just fail to launch.

Allocation happens at SUBMIT time in the runner. The parse-time resolution in schema is a
RunPod-static provisional for validation/dry-run display.
"""

from __future__ import annotations

from dataclasses import replace

from flash._logging import get_logger
from flash.providers import PROVIDER_NAMES, available_providers, get_provider
from flash.providers.base import (
    Allocation,
    Candidate,
    UnsupportedGpuError,
)

logger = get_logger(__name__)

# Classes that stay in the catalog (the name resolves; provider runner machinery/tests use them) but
# are NEVER auto-allocated. L40 is here on PRICE alone: it is strictly dominated by RTX A6000 (same
# 48 GB, half the price, on all three providers), so the cheapest-fitting walk should never pick it.
# (Its only stock is Hyperstack's CANADA-1 fleet, once hard-banned for a broken driver but re-verified
# healthy and un-banned on 2026-06-26; the price domination is independent of that and stands.) See
# base.py's L40 row.
_POOL_EXCLUDED = frozenset({"L40"})

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

    return model_required_vram_gb(
        model_id,
        algorithm,
        train=train,
        thinking=thinking,
        headroom=vram_headroom(),
    )


def _runpod_candidates(need: int) -> list[Candidate]:
    """RunPod's fitting, validated classes priced by the static table.

    Restricted to the validated pool (``g.validated``): the deployed control plane rejects a submit
    for any non-validated class, so allocating one would only fail at submit time.
    """
    provider = get_provider("runpod")
    return [
        Candidate("runpod", g.name, provider.hourly_rate(g.name), g.vram_gb)
        for g in provider.gpu_classes()
        if g.vram_gb >= need and g.validated and g.name not in _POOL_EXCLUDED
    ]


def _lambda_candidates(need: int, ignore_sick: bool = False) -> list[Candidate]:
    """Lambda's fitting classes that currently have LIVE capacity, priced live.

    Capacity-aware by design: a Lambda class with no region advertising capacity is EXCLUDED, so
    the allocator never hands the runner a Lambda class that would immediately fail to launch (and
    burn a retry) — directly the "GPU allocation is good, doesn't randomly die" property. A Lambda
    capacity-lookup failure (no key / network blip) degrades to the other providers: it is
    non-fatal as long as another provider can supply a fitting class. ``ignore_sick`` is the
    last-resort pass: it keeps quarantined regions in so a region quarantine never zeroes out the
    only capacity (see ``allocate``).
    """
    from flash.providers.lambdalabs.jobs import usable_instances

    provider = get_provider("lambda")
    out: list[Candidate] = []
    try:
        for g in provider.gpu_classes():
            if g.vram_gb < need or g.name in _POOL_EXCLUDED:
                continue
            # usable_instances reads the cached /instance-types, so only the first call hits the API.
            if usable_instances(g.name, ignore_sick=ignore_sick):
                out.append(Candidate("lambda", g.name, provider.hourly_rate(g.name), g.vram_gb))
    except Exception as exc:
        logger.warning("lambda capacity lookup failed (%s); allocating without lambda", exc)
        return []
    return out


def _hyperstack_candidates(need: int, ignore_sick: bool = False) -> list[Candidate]:
    """Hyperstack's fitting classes that currently have flavor STOCK, priced statically.

    Capacity-aware, exactly like Lambda: a class with no in-stock flavor is excluded so the runner
    never walks onto a class that would immediately fail to launch. A capacity-lookup failure
    degrades to the other providers. ``ignore_sick`` is the last-resort pass: it keeps quarantined
    regions in so a region quarantine never zeroes out the only capacity (see ``allocate``).
    """
    from flash.providers.hyperstack.jobs import usable_instances

    provider = get_provider("hyperstack")
    out: list[Candidate] = []
    try:
        for g in provider.gpu_classes():
            if g.vram_gb < need or g.name in _POOL_EXCLUDED:
                continue
            # usable_instances reads the cached /core/flavors, so only the first call hits the API.
            if usable_instances(g.name, ignore_sick=ignore_sick):
                out.append(Candidate("hyperstack", g.name, provider.hourly_rate(g.name), g.vram_gb))
    except Exception as exc:
        logger.warning("hyperstack capacity lookup failed (%s); allocating without hyperstack", exc)
        return []
    return out


def allocate(
    model_id: str,
    algorithm: str,
    *,
    train=None,
    thinking: bool = False,
) -> Allocation:
    """Pick the cheapest fitting (provider, GPU class) able to run the job.

    There is no GPU pin — every fitting class on every available provider is eligible, and the
    cheapest wins. RunPod is restricted to its validated pool (``GpuClass.validated``) because the
    deployed control plane rejects a submit for any non-validated class; the instance providers
    (Lambda via LAMBDA_API_KEY, Hyperstack via HYPERSTACK_API_KEY — both opt-in) each contribute
    their fitting classes that currently have live capacity/stock. RunPod's cheaper static rates
    usually win, with Lambda and Hyperstack joining as capacity complements lower in the ranked list.
    ``train``/``thinking`` size the requirement to the run's actual knobs (context, group, rank,
    batch) via the matrix.
    """
    need = required_vram_gb(model_id, algorithm, train=train, thinking=thinking)
    available = available_providers()
    candidates: list[Candidate] = []
    if "runpod" in available:
        candidates += _runpod_candidates(need)
    if "lambda" in available:
        candidates += _lambda_candidates(need)
    if "hyperstack" in available:
        candidates += _hyperstack_candidates(need)
    # Quarantine's contract is bounded DEMOTION, never removal: an instance-provider class whose
    # capacity exists ONLY in currently-quarantined regions is kept as a LAST-RESORT candidate,
    # appended AFTER every healthy candidate so it is only ever reached once healthy capacity is
    # exhausted. Without this, RunPod's always-present static candidates keep the global list non-empty,
    # so a run that has burned every RunPod class on retries would hard-fail rather than walk to the
    # quarantined instance capacity that still exists (a "relax only when the list is globally empty"
    # fallback never fires while RunPod offers a class). A still-sick region just host_faults again
    # (re-quarantine + cross-provider escape); the run is never killed while ANY capacity remains.
    healthy_keys = {(c.provider, c.gpu) for c in candidates}
    sick: list[Candidate] = []
    if "lambda" in available:
        sick += [replace(c, sick=True) for c in _lambda_candidates(need, ignore_sick=True) if (c.provider, c.gpu) not in healthy_keys]
    if "hyperstack" in available:
        sick += [replace(c, sick=True) for c in _hyperstack_candidates(need, ignore_sick=True) if (c.provider, c.gpu) not in healthy_keys]
    if not candidates and not sick:
        raise UnsupportedGpuError(
            f"no allocatable GPU (>= {need} GB VRAM for {model_id}) on any available provider "
            f"({', '.join(available) or '(none)'}); the run genuinely exceeds every active GPU class"
        )
    if not candidates:
        logger.warning(
            "every fitting instance-provider region is quarantined (sick); allocating into a "
            "quarantined region rather than hard-failing the run (quarantine is bounded-demotion)"
        )
    # Healthy candidates first (a SICK quarantine-only candidate is never preferred over a healthy one,
    # even when cheaper); within each tier cheapest first, equal rates prefer less VRAM (don't burn a big
    # card on a small job), then registry order. The runner's _select_candidate applies the SAME sick-last
    # tie-break per attempt, so the demotion survives its per-attempt min()-by-price re-selection.
    order = {n: i for i, n in enumerate(PROVIDER_NAMES)}

    def rank_key(c: Candidate) -> tuple:
        return (c.sick, c.hourly_usd, c.vram_gb, order.get(c.provider, 99))

    ranked = sorted(candidates + sick, key=rank_key)
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
        f"(need >= {a.min_vram_gb} GB VRAM"
    )
    head += ")"
    if len(a.candidates) > 1:
        nxt = a.candidates[1]
        head += f"; next-best: {nxt.gpu}@{nxt.provider} ${nxt.hourly_usd:.2f}/hr"
    return head
