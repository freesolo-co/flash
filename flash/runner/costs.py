"""Run cost estimation, charging, and billing persistence."""

from __future__ import annotations

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


def charge_usd_for_spec(spec, *, steps: int | None = None, fallback: float = 0.0) -> float:
    """Return the estimated customer charge, prorated by completed steps when requested."""
    try:
        from flash.cost.analytical import estimate_cost
        from flash.cost.spec import estimate_for_spec, runconfig_from_spec

        if getattr(spec, "workload_profile_kind", ""):
            # a profile job has no optimizer steps to prorate, so its charge is all-or-nothing: the
            # bounded wall it rented, or zero if it never started. the caller passes steps=0 for the
            # latter (see profile_steps_run), and honouring it matters because the id is derived
            # from the workload rather than the account. a profile cancelled before launch would
            # otherwise bill the full wall cap to whichever submitter happened to win the claim.
            if steps is not None and int(steps) <= 0:
                return 0.0
            return float(estimate_for_spec(spec).total_usd)
        if steps is None:
            return float(estimate_for_spec(spec).total_usd)
        n = max(0, int(steps))
        if n == 0:
            return 0.0
        cfg = runconfig_from_spec(spec)
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
        return float(estimate_cost(cfg).total_usd)
    except Exception:
        return float(fallback)


def _status_estimated_charge(status: RunStatus, spec, *, fallback: float = 0.0) -> float:
    quote = getattr(status, "estimated_cost_usd", None)
    if quote is not None:
        return float(quote)
    return runner.charge_usd_for_spec(spec, fallback=fallback)


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


def profile_steps_run(status: RunStatus) -> int:
    """Whether a cancelled profile job rented anything: 1 if it started, 0 if it never did.

    The signal is that a worker spoke, but a profile's run id is derived from the workload, so a
    relaunch REUSES it and the stored word may belong to the previous lifecycle. Billing the stored
    stage there charges a relaunch cancelled in the queue for a machine it never rented. So a
    relaunch -- and only a relaunch, marked by the attempt floor its takeover records -- is
    billed on the arm, which is written only for a heartbeat that passed
    ``_heartbeat_attempt_is_current``. A first lifecycle has no earlier worker to be confused with
    and bills on the stored word as before.
    """
    hb = status.last_heartbeat if isinstance(status.last_heartbeat, dict) else {}
    if not hb.get("stage"):
        return 0
    try:
        raw = runner._load_status_json(status.run_id)
    except (FileNotFoundError, ValueError):
        return 1
    if runner._PROFILE_ATTEMPT_FLOOR_KEY not in raw:
        return 1
    return 1 if runner._profile_wall_armed_at(raw) is not None else 0


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
        # backfill finished_at before bumping updated_at so reconcile._terminal_ts is not skewed.
        if (
            status.state in runner._FINISHED_AT_PRESERVED_STATES
            and status.finished_at is None
            and not status.reconciled_at
        ):
            status.finished_at = status.updated_at
        for key, value in fields.items():
            setattr(status, key, value)
        status.updated_at = time.time()
        runner._save_status_unlocked(status)
    runner._report_status(status)
