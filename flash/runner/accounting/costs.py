"""Run cost estimation, charging, and billing persistence."""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import replace

from flash.runner.lifecycle import reporting, state
from flash.runner.lifecycle.state import RunStatus

_BILLING_FIELDS = frozenset({"billing_state", "billing_error", "billing_charge"})


def _get_status(run_id: str):
    from flash.runner.lifecycle.status import get_status

    return get_status(run_id)


def _gpu_rate(gpu_type: str, provider: str = "") -> float:
    """Static representative $/hr for cost projection.

    Falls back to any configured provider that offers the class, so a plane without RunPod still
    prices its runs. Never raises: this feeds cost ANNOTATION on an already-finished run, so a
    provider-registry problem must degrade to the flat estimate rather than fail the metrics write.
    """
    try:
        from flash.providers.core.registry import available_providers, get_provider

        # use the billing substrate first when known, then any other configured provider that offers
        # the class, so a plane without runpod still prices its runs.
        names = [provider.strip().lower()] if provider.strip() else []
        names += [n for n in available_providers() if n not in names]
    except Exception:
        return 0.80
    for name in names:
        try:
            rate = get_provider(name).hourly_rate(gpu_type)
        except Exception:
            continue
        if rate:
            return float(rate)
    return 0.80


def _pinned_offline_allocation(provider: str, gpu_type: str, gpu_count: int):
    """An offline-rate allocation pinning a rented shape, or ``None`` when there is nothing to pin.

    the offline shape search treats the spec's card count as a ceiling and re-optimizes under it,
    so a live allocator's 2/4/8-card rental would be repriced on whatever smaller shape the static
    rate table prefers. an explicit allocation is the same mechanism the accepted quote was built
    on (see seed submission), so an estimate that consumes it reproduces the quote's exact
    geometry. the pin fixes geometry, not price: rates stay offline, mirroring the shape search's
    own rate choice, and cancel out of any partial/full ratio. never raises -- an unpriceable pin
    returns ``None`` so the caller degrades to the spec-derived shape instead of failing a charge.
    """
    if not gpu_type or gpu_count < 1:
        return None
    try:
        from flash.providers.core.base import GPU_INFO, Allocation, canonical_gpu

        gpu = canonical_gpu(gpu_type)
        if provider == "lambda":
            from flash.providers.lambda_.client.pricing import static_hourly_rate

            hourly = static_hourly_rate(gpu)
        else:
            hourly = GPU_INFO[gpu].hourly_usd
        return Allocation(
            provider=provider,
            gpu=gpu,
            hourly_usd=float(hourly),
            min_vram_gb=0,
            candidates=(),
            gpu_count=int(gpu_count),
        )
    except Exception:
        return None


def _spec_pinned_to_horizon(spec, steps: int):
    """``spec`` with its update horizon stated as ``steps``, for pricing a run that already ran.

    ``max_steps`` is the one field that states a grpo/opd horizon outright, so pinning it is what
    lets a completed step count stand in for the prompt-pool size the estimator would otherwise
    have to derive one from. floored at 1 because ``RunConfig`` rejects a zero horizon; the callers
    all return $0 for a zero-step cancel before the estimate is reached, so the floor never prices
    anything. only reached when that derivation failed, which by construction means no save
    schedule to rewrite: ``save_at_steps`` requires a positive ``max_steps`` (core/spec.py), and a
    positive ``max_steps`` is itself a stated horizon that never needed a pool size.
    """
    from dataclasses import replace as _replace

    return _replace(spec, train=_replace(spec.train, max_steps=max(1, int(steps))))


def charge_usd_for_spec(
    spec,
    *,
    steps: int | None = None,
    fallback: float = 0.0,
    provider: str = "",
    gpu_type: str = "",
    gpu_count: int = 0,
) -> float:
    """Return the estimated customer charge, prorated by completed steps when requested.

    ``provider`` overrides the pricing substrate for a training estimate. the persisted worker spec
    of an auto-allocated run carries no provider, so its offline reprice would credit nvlink
    scaling on a substrate (vast) that cannot deliver it; the cancel path passes the provider the
    run actually rented so the estimate reproduces the accepted quote's work model. ``gpu_type``
    and ``gpu_count`` together pin the rented card shape the same way: the spec's count is only the
    ceiling the allocator searched under, so without the pin a 4-card rental could be repriced on
    the 1-card offline optimum.
    """
    try:
        from flash.cost.analytical import estimate_cost
        from flash.cost.spec import UnknownPromptPoolSize, runconfig_from_spec

        try:
            cfg = runconfig_from_spec(spec)
        except UnknownPromptPoolSize:
            # a cancel prices work that already happened, so it must not need a PREDICTED horizon.
            # grpo/opd derive theirs from a stated prompt-pool size and refuse to guess without one,
            # which would make an unbounded run unpriceable at exactly the moment its cost is known
            # exactly. the completed count answers that question with the measurement instead --
            # spec_steps reads max_steps first, so the pool size is never consulted. only the
            # prorating callers may do this: with no steps to state, there is still no horizon.
            if steps is None:
                raise
            cfg = runconfig_from_spec(_spec_pinned_to_horizon(spec, int(steps)))
        if provider:
            # suppressed ValueError: the registry no longer maps this provider to the spec's
            # class. the combination was rentable when the run launched, so degrade to the spec's
            # own pricing substrate rather than turning a chargeable cancel into a pricing failure.
            with contextlib.suppress(ValueError):
                cfg = replace(cfg, provider=provider)
        allocation = _pinned_offline_allocation(provider, gpu_type, gpu_count)
        if steps is None:
            return float(estimate_cost(cfg, allocation=allocation).total_usd)
        n = max(0, int(steps))
        if n == 0:
            return 0.0
        planned = int(cfg.steps or 0)
        if planned > 0:
            n = min(n, planned)
        # a partial (cancelled) reprice only counts required saves that could already have landed by
        # the completed step. keeping a save beyond the reduced horizon would also trip the run
        # config's save_at_steps <= steps guard and drop the whole estimate to the fallback.
        reached_saves = tuple(s for s in cfg.save_at_steps if s <= n)
        if not cfg.is_grpo and cfg.train_tokens and planned > 0:
            scaled_tokens = max(1, int(cfg.train_tokens * n / planned))
            cfg = replace(cfg, steps=n, train_tokens=scaled_tokens, save_at_steps=reached_saves)
        else:
            cfg = replace(cfg, steps=n, save_at_steps=reached_saves)
        return float(estimate_cost(cfg, allocation=allocation).total_usd)
    except Exception:
        return float(fallback)


def _rented_basis(remote) -> tuple[str, str, int]:
    """The substrate a run actually rented: (provider, gpu type, card count) from its handle.

    the provider handle is the only durable record of the rented shape -- seed submission persists
    ``allocated_gpu``/``allocated_gpu_count`` beside the provider precisely because the spec's own
    count is just the ceiling the allocator searched under. a successful cancel tears the handle
    down before billing runs, so the cancel path captures it pre-teardown and passes it here.
    every field is validated and degrades to ""/0 rather than raising: an unrecognized value would
    make the offline estimate unpriceable and turn a chargeable cancel into a billing failure, so
    it falls back to the spec's own pricing substrate and shape instead. the shape pair is
    all-or-nothing -- a card name without its count (or vice versa) cannot name a geometry.
    """
    if not isinstance(remote, dict):
        return "", "", 0
    provider = remote.get("provider")
    provider = provider.strip().lower() if isinstance(provider, str) else ""
    if provider:
        try:
            from flash.providers.core.registry import PROVIDER_NAMES

            if provider not in PROVIDER_NAMES:
                provider = ""
        except Exception:
            provider = ""
    gpu_type = ""
    raw_gpu = remote.get("allocated_gpu")
    if isinstance(raw_gpu, str) and raw_gpu.strip():
        try:
            from flash.providers.core.base import GPU_INFO, canonical_gpu

            name = canonical_gpu(raw_gpu)
            info = GPU_INFO.get(name)
            if info is not None and info.validated:
                gpu_type = name
        except Exception:
            gpu_type = ""
    raw_count = remote.get("allocated_gpu_count")
    gpu_count = raw_count if isinstance(raw_count, int) and not isinstance(raw_count, bool) else 0
    # mirror the spec layer's 1..8 bound so a corrupt stamp cannot reach the estimator.
    if not 1 <= gpu_count <= 8:
        gpu_count = 0
    if not gpu_type or not gpu_count:
        gpu_type, gpu_count = "", 0
    return provider, gpu_type, gpu_count


def cancelled_charge_usd(
    status: RunStatus,
    spec,
    *,
    steps: int,
    fallback: float = 0.0,
    rented_remote: dict | None = None,
) -> float:
    """Price a mid-training cancellation from the accepted quote, scaled by the completed work.

    the persisted quote (``estimated_cost_usd``) is the whole-cent submit-time amount shown by
    ``flash train --cost``. provider selection and allocation never refresh it. instead the quote is
    scaled by the completed share of estimated billed work: partial and full spec estimates use the
    rented topology when available, so their ratio isolates the work fraction without replacing the
    accepted quote. the fraction, not a bare ``steps / planned`` ratio, keeps the charge honest:
    the one-time compile and each reached save land whole in the partial estimate, unreached saves
    stay out of it, and a wall-capped plan caps both sides so the fraction is measured against the
    capped horizon the quote actually paid for rather than the uncapped step count. the fraction is
    capped at 1 so the charge never exceeds the quote. a run with no persisted quote has nothing to
    clamp to, so it falls back to the spec reprice.

    ``rented_remote`` is the provider handle captured before teardown cleared it: a successful
    cancel destroys ``status.remote`` before billing runs, and the handle is the only durable
    record of the rented basis. absent (legacy callers, never-allocated runs), the status's own
    handle is consulted, which still covers failed or unconfirmed teardowns.
    """
    n = max(0, int(steps))
    if n == 0:
        # cancelled before any training step: nothing rented, nothing owed.
        return 0.0
    # the persisted worker spec of an auto-allocated run names no provider and carries the card
    # count only as the ceiling the allocator searched under, so pricing it bare credits nvlink
    # scaling a vast rental cannot deliver and can re-optimize a 4-card rental onto the 1-card
    # offline optimum. either topology error is a factor on the per-step term only, so it does not
    # cancel out of partial / full whenever the estimate also carries fixed work (moe compile,
    # required saves) or a wall cap: both estimates are computed on the substrate and card shape
    # the run actually rented.
    provider, gpu_type, gpu_count = _rented_basis(
        rented_remote if rented_remote is not None else getattr(status, "remote", None)
    )
    quote = getattr(status, "estimated_cost_usd", None)
    if quote is None:
        return charge_usd_for_spec(
            spec,
            steps=n,
            fallback=fallback,
            provider=provider,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
        )
    if isinstance(quote, bool) or not isinstance(quote, (int, float)):
        # a malformed persisted quote is a pricing failure, not a license to reprice: the accepted
        # rate is unknowable, so the caller's fallback must propagate and settle the run.
        return float(fallback)
    quote = float(quote)
    if not math.isfinite(quote) or quote < 0:
        # negative and non-finite quotes cannot represent an accepted whole-cent amount, so the
        # fallback propagates and the caller records the pricing failure.
        return float(fallback)
    if quote == 0:
        # a valid quote may round to zero cents and remains authoritative for any completed work.
        return 0.0
    partial = charge_usd_for_spec(
        spec,
        steps=n,
        fallback=float("nan"),
        provider=provider,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
    )
    full = charge_usd_for_spec(
        spec,
        fallback=float("nan"),
        provider=provider,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
    )
    if math.isfinite(partial) and math.isfinite(full) and full > 0:
        return quote * min(1.0, partial / full)
    # the work fraction is unpriceable: reprice the spec but never bill a cancel above the quote.
    # a non-finite reprice is a pricing failure and must propagate so the caller records it.
    repriced = charge_usd_for_spec(
        spec, steps=n, fallback=fallback, provider=provider, gpu_type=gpu_type, gpu_count=gpu_count
    )
    if not math.isfinite(repriced):
        return repriced
    return min(repriced, quote)


def _persisted_completed_work(spec) -> tuple[int | None, dict | None]:
    """Authoritative completed steps and rented basis from terminal metrics."""
    try:
        with open(os.path.join(state.artifacts_dir(spec), "metrics.json")) as handle:
            metrics = json.load(handle)
    except (OSError, TypeError, ValueError):
        return None, None
    if not isinstance(metrics, dict):
        return None, None
    step = metrics.get("step")
    valid_step = (
        isinstance(step, (int, float))
        and not isinstance(step, bool)
        and (not isinstance(step, float) or (math.isfinite(step) and step.is_integer()))
        and step >= 0
    )
    completed_steps = int(step) if valid_step else None
    provider = metrics.get("allocated_provider")
    gpu = metrics.get("allocated_gpu")
    gpu_count = metrics.get("allocated_gpu_count")
    rented_remote = {
        "provider": provider,
        "allocated_gpu": gpu,
        "allocated_gpu_count": gpu_count,
    }
    if not any(value is not None for value in rented_remote.values()):
        rented_remote = None
    return completed_steps, rented_remote


def _status_estimated_charge(status: RunStatus, spec, *, fallback: float = 0.0) -> float:
    """Charge the quote exactly for full work, or its estimated share for an early finish."""
    completed_steps, persisted_basis = _persisted_completed_work(spec)
    if completed_steps is not None:
        # reuse cancellation's estimated-work fraction: one-time compile and reached saves stay
        # whole, unreached saves stay out, and a wall-capped plan reaches 100% at its priced cap.
        # a complete horizon produces a fraction of exactly 1 and therefore the accepted quote.
        status_remote = status.remote if isinstance(status.remote, dict) else {}
        rented_remote = {
            key: value for key, value in (persisted_basis or {}).items() if value is not None
        }
        rented_remote.update(status_remote)
        return cancelled_charge_usd(
            status,
            spec,
            steps=completed_steps,
            fallback=fallback,
            rented_remote=rented_remote or None,
        )
    quote = getattr(status, "estimated_cost_usd", None)
    if quote is not None:
        return float(quote)
    return charge_usd_for_spec(spec, fallback=fallback)


def actual_steps_run(status: RunStatus) -> int:
    """How many optimizer steps to bill a (cancelled) run for.

    Cancelled after N steps -> N. The first step reports no ``step`` until it completes, so a cancel
    mid-first-step would look like 0 steps despite real GPU time -- we floor to 1 whenever a
    training-stage heartbeat is present.
    """
    hb = status.last_heartbeat if isinstance(status.last_heartbeat, dict) else {}
    step = hb.get("step")
    if isinstance(step, (int, float)) and step > 0:
        return int(step)
    # training started (rl_step/sft_step/opd_step) but no completed step yet means mid-first-step.
    if hb.get("stage") in state._TRAINING_STAGES:
        return 1
    return 0


def record_realized_cost(run_id: str, *, realized_cost_usd: float, reconciled_at: float) -> None:
    """Persist reconciliation COGS without touching run state. No-ops if run vanished."""
    with state._status_guard(run_id):
        try:
            status = _get_status(run_id)
        except FileNotFoundError:
            return
        status.realized_cost_usd = realized_cost_usd
        status.reconciled_at = reconciled_at
        if not status.billing_context or status.billing_state == "charged":
            status.realized_cost_remote = None
        status.updated_at = time.time()
        state._save_status_unlocked(status)
    reporting._report_status(status)


def record_billing_state(run_id: str, **fields) -> None:
    """Persist billing fields without touching run state. Never downgrades a charged run."""
    bad = set(fields) - _BILLING_FIELDS
    if bad:
        raise ValueError(f"record_billing_state only writes billing fields, got: {sorted(bad)}")
    with state._status_guard(run_id):
        try:
            status = _get_status(run_id)
        except FileNotFoundError:
            return
        new_billing_state = fields.get("billing_state")
        if (
            status.billing_state == "charged"
            and "billing_state" in fields
            and new_billing_state != "charged"
        ):
            return
        for key, value in fields.items():
            setattr(status, key, value)
        if status.billing_state == "charged" and status.reconciled_at is not None:
            status.realized_cost_remote = None
        status.updated_at = time.time()
        state._save_status_unlocked(status)
    reporting._report_status(status)
