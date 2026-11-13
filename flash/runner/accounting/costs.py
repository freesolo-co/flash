"""Run cost estimation, charging, and billing persistence."""

from __future__ import annotations

import contextlib
import json
import math
import os
import time

from flash.runner.lifecycle import reporting, state
from flash.runner.lifecycle.state import RunStatus

_BILLING_FIELDS = frozenset({"billing_state", "billing_error", "billing_charge"})


def _get_status(run_id: str):
    from flash.runner.lifecycle.status import get_status

    return get_status(run_id)


def _gpu_rate(gpu_type: str, provider: str = "") -> float:
    """Static representative $/hr for cost projection.

    Falls back to the configured provider that offers the class, so recovery still
    prices its runs. Never raises: this feeds cost ANNOTATION on an already-finished run, so a
    provider-registry problem must degrade to the flat estimate rather than fail the metrics write.
    """
    try:
        from flash.providers.core.registry import available_providers, get_provider

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


def cancelled_charge_usd(
    status: RunStatus,
    spec,
    *,
    steps: int,
    fallback: float = 0.0,
    rented_remote: dict | None = None,
) -> float:
    """Price a cancellation: the training time it metered, at the shapes it rented.

    ``rented_remote`` is the provider handle captured before teardown cleared it; it names the
    shape and attempt of the live rental. a worker that predates the meter falls back to the
    quote's share of completed steps (``_legacy_quote_share``).
    """
    if getattr(spec, "algorithm", None) == "custom":
        from flash.custom.billing import session_charge

        return session_charge(status, spec)
    metered = metered_charge_usd(status, spec, rented_remote=rented_remote)
    if metered is not None:
        return metered
    return _legacy_quote_share(status, spec, steps=steps, fallback=fallback)


def _legacy_quote_share(status: RunStatus, spec, *, steps: int, fallback: float) -> float:
    """A cancelled run whose worker predates the meter: the quote x its completed-step share.

    only runs in flight across the deploy that introduced metering reach this. an invalid persisted
    quote returns ``fallback`` so the caller records the pricing failure.
    """
    n = max(0, int(steps))
    if n == 0:
        return 0.0
    quote = getattr(status, "estimated_cost_usd", None)
    if (
        isinstance(quote, bool)
        or not isinstance(quote, (int, float))
        or not math.isfinite(quote)
        or quote < 0
    ):
        return float(fallback)
    try:
        from flash.cost.spec import spec_steps

        planned = int(spec_steps(spec))
    except Exception:
        planned = 0
    return float(quote) * (min(1.0, n / planned) if planned > 0 else 1.0)


def _finite_seconds(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _persisted_metrics(spec) -> dict:
    try:
        with open(os.path.join(state.artifacts_dir(spec), "metrics.json")) as handle:
            metrics = json.load(handle)
    except Exception:
        # unreadable or not-yet-written metrics: the heartbeat meter (or the legacy quote) prices.
        return {}
    return metrics if isinstance(metrics, dict) else {}


def _attempt_meters(status: RunStatus, spec, *, completed_attempt=None) -> dict[str, dict] | None:
    """Per-attempt metered seconds and rented shape, or ``None`` when nothing was metered.

    each worker attempt meters from the start of its first optimizer update to the exit of its
    training loop (``training_timing.billable_train_s``), so model download, data loading, trainer
    / model / vLLM initialization and final adapter publication are never included. the status
    record ``billable_by_attempt`` keeps every attempt's highest observed meter and the shape it
    rented. the frozen terminal meter belongs to ``completed_attempt``, the authoritative identity
    of the attempt whose artifacts completed the run, never ``last_heartbeat.attempt``: the final
    heartbeat read is best effort, so the last persisted heartbeat can still belong to an earlier
    attempt, and merging the two would drop one attempt's time. ``None`` means a legacy worker that
    reports no meter.
    """
    from flash.runner.lifecycle.status import BILLABLE_ATTEMPTS_KEY, _heartbeat_billable_seconds

    hb = status.last_heartbeat if isinstance(status.last_heartbeat, dict) else {}
    attempts: dict[str, dict] = {}
    raw = hb.get(BILLABLE_ATTEMPTS_KEY)
    for key, entry in raw.items() if isinstance(raw, dict) else ():
        seconds = _finite_seconds(entry.get("s")) if isinstance(entry, dict) else None
        if isinstance(key, str) and seconds is not None:
            shape = entry.get("shape")
            attempts[key] = {"s": seconds, "shape": shape if isinstance(shape, str) else None}
    notes = _persisted_metrics(spec).get("notes")
    timing = notes.get("training_timing") if isinstance(notes, dict) else None
    frozen = _finite_seconds(timing.get("billable_train_s")) if isinstance(timing, dict) else None
    live = _finite_seconds(_heartbeat_billable_seconds(hb))
    if live is not None:
        entry = attempts.setdefault(str(hb.get("attempt")), {"s": 0.0, "shape": None})
        entry["s"] = max(entry["s"], live)
    if frozen is not None:
        owner = completed_attempt if completed_attempt is not None else hb.get("attempt")
        # an attempt first seen here ran on the final rental, which the caller prices it at.
        entry = attempts.setdefault(str(owner), {"s": 0.0, "shape": None})
        entry["s"] = max(entry["s"], frozen)
    return attempts or None


def metered_training_seconds(status: RunStatus, spec) -> float | None:
    """Measured GPU training seconds across every attempt, or ``None`` for a legacy worker."""
    attempts = _attempt_meters(status, spec)
    return None if attempts is None else sum(entry["s"] for entry in attempts.values())


def metered_charge_usd(
    status: RunStatus, spec, *, rented_remote: dict | None = None, completed_attempt=None
):
    """Exact charge: metered training seconds x the rented card's $/hr x rented cards.

    returns ``None`` when the run carries no meter (legacy workers), so callers fall back. the rate is the same per-card rate card the quote prices with, applied
    to the provider/card/count the run actually rented rather than the quote's assumptions.
    """
    if completed_attempt is None and isinstance(rented_remote, dict):
        # a cancel prices the attempt whose handle was captured before teardown.
        completed_attempt = rented_remote.get("attempt")
    attempts = _attempt_meters(status, spec, completed_attempt=completed_attempt)
    if attempts is None:
        return None
    metrics = _persisted_metrics(spec)
    merged = {
        "provider": metrics.get("allocated_provider"),
        "allocated_gpu": metrics.get("allocated_gpu"),
        "allocated_gpu_count": metrics.get("allocated_gpu_count"),
    }
    remote = rented_remote if isinstance(rented_remote, dict) else getattr(status, "remote", None)
    if isinstance(remote, dict):
        merged.update({key: value for key, value in remote.items() if value is not None})
    spot = False
    try:
        from flash.core.checkpoint_policy import is_spot_spec

        spot = bool(is_spot_spec(spec))
    except Exception:
        pass
    from flash.runner.lifecycle.status import rented_shape_key

    final = _parse_shape(rented_shape_key(merged), spot=spot)
    if final is None:
        # no usable rented handle: the spec's own class and card count.
        gpu_spec = getattr(spec, "gpu", None)
        final = (
            "spot" if spot else str(merged.get("provider") or ""),
            str(getattr(gpu_spec, "type", "") or ""),
            max(1, int(getattr(gpu_spec, "count", 1) or 1)),
        )
    final_hourly = _per_card_rate(final[0], final[1], status) * final[2]
    # each attempt bills at the shape it actually rented (an OOM retry moves to a larger card or
    # count; a Spot resume can land on another class), never repriced at the final rental. an
    # attempt whose shape was never observed bills at the final rental.
    total = 0.0
    for entry in attempts.values():
        shape = _parse_shape(entry["shape"], spot=spot)
        hourly = (
            final_hourly if shape is None else _per_card_rate(shape[0], shape[1], status) * shape[2]
        )
        total += entry["s"] / 3600.0 * hourly
    return total


def _per_card_rate(provider: str, gpu_type: str, status: RunStatus) -> float:
    """$/hr for one card: Spot bills the tariff the run was quoted under, else the rate card.

    a Spot run's tariff is recorded at submission (``submitted_spot_rates``; older runs use the
    legacy table), so a later rate change never reprices a run. an unknown class prices at that
    tariff's costliest class, the quote's upper bound.
    """
    if provider == "spot":
        from flash.providers.core.base import canonical_gpu
        from flash.runner.accounting.spot_usage import submitted_rates

        rates = submitted_rates(status)
        with contextlib.suppress(Exception):
            gpu_type = canonical_gpu(gpu_type)
        return rates.get(gpu_type, max(rates.values()))
    return _gpu_rate(gpu_type, provider)


def _parse_shape(shape, *, spot: bool) -> tuple[str, str, int] | None:
    """``provider|gpu|cards`` -> ``(provider, gpu, cards)``; ``None`` when unusable."""
    parts = shape.split("|") if isinstance(shape, str) else []
    if len(parts) != 3 or not parts[1] or not parts[2].isdigit():
        return None
    # the authored Spot choice is authoritative, as for the final rental.
    return ("spot" if spot else parts[0], parts[1], max(1, int(parts[2])))


def _status_estimated_charge(
    status: RunStatus, spec, *, fallback: float = 0.0, completed_attempt=None
) -> float:
    """Charge a completed run: the training time it metered, at the shapes it rented.

    ``completed_attempt`` is the attempt whose terminal metrics completed the run. a worker that
    predates the meter is charged ``fallback``, the measured wall-clock cost the caller computed
    from its terminal metrics.
    """
    if getattr(spec, "algorithm", None) == "custom":
        from flash.custom.billing import session_charge

        return session_charge(status, spec)
    metered = metered_charge_usd(status, spec, completed_attempt=completed_attempt)
    return float(fallback) if metered is None else metered


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
    # training started (rl_step/sft_step_step) but no completed step yet means mid-first-step.
    if hb.get("stage") in state._TRAINING_STAGES:
        return 1
    return 0


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
        if status.billing_state == "charged":
            status.billing_remote = None
        status.updated_at = time.time()
        state._save_status_unlocked(status)
    reporting._report_status(status)
