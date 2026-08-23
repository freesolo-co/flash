"""Run cost estimation, charging, and billing persistence."""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
from dataclasses import replace

import flash.runner as runner
from flash.runner import RunStatus


def _gpu_rate(gpu_type: str, provider: str = "") -> float:
    """Static representative $/hr for cost projection.

    Falls back to any configured provider that offers the class, so a plane without RunPod still
    prices its runs. Never raises: this feeds cost ANNOTATION on an already-finished run, so a
    provider-registry problem must degrade to the flat estimate rather than fail the metrics write.
    """
    try:
        from flash.providers import available_providers, get_provider

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
        from flash.providers.base import GPU_INFO, Allocation, canonical_gpu

        gpu = canonical_gpu(gpu_type)
        if provider == "lambda":
            from flash.providers.lambda_.pricing import static_hourly_rate

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
            from flash.providers import PROVIDER_NAMES

            if provider not in PROVIDER_NAMES:
                provider = ""
        except Exception:
            provider = ""
    gpu_type = ""
    raw_gpu = remote.get("allocated_gpu")
    if isinstance(raw_gpu, str) and raw_gpu.strip():
        try:
            from flash.providers.base import GPU_INFO, canonical_gpu

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

    the persisted quote (``estimated_cost_usd``) carries the exact live rate the user accepted: the
    lifecycle refreshes it from the selected candidate before provisioning. a spec reprice through
    ``estimate_cost`` takes the offline static-rate path, and on live-market providers (vast,
    lambda) those rates differ materially from the accepted one, so pricing a near-complete cancel
    that way can bill above what the run would have been charged on success. instead the quote is
    scaled by the completed share of the estimated billed work: partial and full spec estimates use
    the same offline rates, so the rate cancels out of their ratio and only the work fraction
    remains. the fraction, not a bare ``steps / planned`` ratio, is what keeps the charge honest:
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
        return runner.charge_usd_for_spec(
            spec,
            steps=n,
            fallback=fallback,
            provider=provider,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
        )
    try:
        quote = float(quote)
    except (TypeError, ValueError):
        # a malformed persisted quote is a pricing failure, not a license to reprice: the accepted
        # rate is unknowable, so the caller's fallback must propagate and settle the run.
        return float(fallback)
    if quote <= 0:
        # a non-positive quote is malformed the same way: prorating it would persist a charge the
        # billing retry predicate (cost_usd > 0) can never settle, silently stranding the run
        # unbilled, so the fallback propagates and the caller records the pricing failure.
        return float(fallback)
    partial = runner.charge_usd_for_spec(
        spec,
        steps=n,
        fallback=float("nan"),
        provider=provider,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
    )
    full = runner.charge_usd_for_spec(
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
    repriced = runner.charge_usd_for_spec(
        spec, steps=n, fallback=fallback, provider=provider, gpu_type=gpu_type, gpu_count=gpu_count
    )
    if not math.isfinite(repriced):
        return repriced
    return min(repriced, quote)


def _persisted_training_metrics(spec) -> dict | None:
    """Read the sanitized worker metrics that _persist_metrics wrote before settlement."""
    try:
        with open(os.path.join(runner.artifacts_dir(spec), "metrics.json")) as handle:
            metrics = json.load(handle)
    except (OSError, TypeError, ValueError):
        return None
    return metrics if isinstance(metrics, dict) else None


def _nonnegative_seconds(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    seconds = float(value)
    return seconds if math.isfinite(seconds) and seconds > 0.0 else 0.0


def _completed_quote_charge(status: RunStatus, spec, quote: float) -> float:
    """Subtract measured unbillable train-wall seconds from a completed run's accepted quote.

    cancellation does not compose through this helper. it already settles once through
    cancelled_charge_usd using the completed-work fraction, and cancelled workers need not produce
    terminal metrics. keeping the paths disjoint avoids applying a second discount or clamp to the
    partial quote.
    """
    metrics = _persisted_training_metrics(spec)
    if metrics is None:
        return quote
    measured_keys = ("framework_init_seconds", "reward_seconds")
    if not any(metrics.get(key) is not None for key in measured_keys):
        # no measurement means no adjustment. this is the normal outcome for a worker that died
        # before terminal metrics and for records written before these measurements existed.
        return quote
    step = metrics.get("step")
    if isinstance(step, (int, float)) and not isinstance(step, bool) and step <= 0:
        return 0.0
    excluded_seconds = sum(_nonnegative_seconds(metrics.get(key)) for key in measured_keys)
    if excluded_seconds <= 0.0:
        return max(0.0, quote)

    remote = status.remote if isinstance(status.remote, dict) else {}
    provider, gpu_type, gpu_count = _rented_basis(
        {
            "provider": remote.get("provider") or metrics.get("allocated_provider"),
            "allocated_gpu": remote.get("allocated_gpu") or metrics.get("allocated_gpu"),
            "allocated_gpu_count": remote.get("allocated_gpu_count")
            or metrics.get("allocated_gpu_count"),
        }
    )
    if not provider or not gpu_type or gpu_count < 1:
        # never guess a different shape or substrate: the billed train seconds below are only the
        # ones the quote paid for on THIS shape. absence preserves the accepted charge exactly.
        return quote
    whole_instance_rate = remote.get("hourly_usd")
    if (
        isinstance(whole_instance_rate, (int, float))
        and not isinstance(whole_instance_rate, bool)
        and math.isfinite(float(whole_instance_rate))
        and float(whole_instance_rate) >= 0.0
    ):
        # the handle carries the live rate the quote was REBUILT from: seed submission replaces the
        # provisional offline quote with `estimate_for_spec(..., hourly_usd=chosen.hourly_usd)`
        # before provisioning. on vast/lambda that rate differs materially from the static table
        # (the same trap cancelled_charge_usd documents), so it is the only coherent basis here.
        # convert to per-card; multiplying by gpu_count below restores the whole-box basis.
        rate = float(whole_instance_rate) / gpu_count
    else:
        allocation = _pinned_offline_allocation(provider, gpu_type, gpu_count)
        if allocation is None:
            return quote
        try:
            from flash.cost.spec import estimate_for_spec

            basis = estimate_for_spec(spec, allocation=allocation)
            quoted_count = int(basis.gpu_count)
            train_seconds = float(basis.train_seconds)
        except Exception:
            return quote
        if quoted_count != gpu_count or not math.isfinite(train_seconds) or train_seconds <= 0.0:
            return quote
        # a runpod handle persists no rate. the quote is `train_seconds / 3600 * hourly * count`
        # (analytical.estimate_cost), so inverting it recovers the per-card rate the customer
        # actually accepted -- strictly better than reading a static table that may have moved
        # since, which could discount at a rate the quote never used.
        rate = quote * 3600.0 / (train_seconds * gpu_count)
    if not math.isfinite(rate) or rate < 0.0:
        return quote
    adjustment = excluded_seconds / 3600.0 * rate * gpu_count
    return max(0.0, min(quote, quote - adjustment))


def _status_estimated_charge(status: RunStatus, spec, *, fallback: float = 0.0) -> float:
    quote = getattr(status, "estimated_cost_usd", None)
    if quote is None:
        quote = runner.charge_usd_for_spec(spec, fallback=fallback)
    return _completed_quote_charge(status, spec, float(quote))


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
    if hb.get("stage") in runner._TRAINING_STAGES:
        return 1
    return 0


def record_realized_cost(run_id: str, *, realized_cost_usd: float, reconciled_at: float) -> None:
    """Persist reconciliation COGS without touching run state. No-ops if run vanished."""
    with runner._status_guard(run_id):
        try:
            status = runner.get_status(run_id)
        except FileNotFoundError:
            return
        status.realized_cost_usd = realized_cost_usd
        status.reconciled_at = reconciled_at
        status.updated_at = time.time()
        runner._save_status_unlocked(status)
    runner._report_status(status)


def record_billing_state(run_id: str, **fields) -> None:
    """Persist billing fields without touching run state. Never downgrades a charged run."""
    bad = set(fields) - runner._BILLING_FIELDS
    if bad:
        raise ValueError(f"record_billing_state only writes billing fields, got: {sorted(bad)}")
    with runner._status_guard(run_id):
        try:
            status = runner.get_status(run_id)
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
        status.updated_at = time.time()
        runner._save_status_unlocked(status)
    runner._report_status(status)
