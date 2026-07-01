"""Retry/recover completion-time CUSTOMER charges so a transient blip never leaks revenue.

The run hot path charges a completed run exactly ONCE, inline, right after it goes ``done``
(runner.lifecycle._charge_completed_run_best_effort). On a transient backend blip there -- or a
control-plane crash between the ``done`` write and the charge -- the run is left billed-never:
``billing_state`` stuck in ``pending``/``charging``/``failed`` while the user still holds a
deployable adapter (a silent revenue leak). recover_runs cannot help: it only re-attaches in-flight
runs and deliberately EXCLUDES terminal ``done`` (and is aliased to the instance-protection set, so
``done`` must not be added to it).

This sweep is the durable backstop. It scans local runs for completed-but-uncharged runs and
re-invokes the same charge hook. The backend charge route is IDEMPOTENT by ``runId``, so a retry
that races or duplicates a charge the inline path (or a prior sweep) already landed is a safe no-op
(replay) -- there is no way to double-charge.

It also bills runs CANCELLED mid-training: those never reach the inline completion charge. The cancel
path (deploy.cancel_run) re-prices such a run to the steps it actually ran -- the same flash.cost
estimate at ``actual_steps`` instead of the planned steps -- and stores it as ``cost_usd``, so the
sweep charges it through the SAME path as a completed run.

It runs once at startup (so a crash between the ``done`` write and the charge is recovered promptly,
even though those runs are terminal and outside recover_runs) and then on a fixed interval, which is
the bounded backoff. Gated on the operator internal key (the charge needs it); off entirely without
FREESOLO_INTERNAL_KEY, exactly like realized-cost reconciliation.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable

from flash import runner
from flash.server.auth import INTERNAL_KEY_ENV

# States that produced a final, deployable adapter -> a completion charge applies. `deployed` is a
# `done` run with serving stood up on top -> still billable (its done-time charge may have failed
# before the deploy). A run still in one of these states whose charge is `pending` from submit but
# never ran (a crash between the `done` write and the charge) is recovered via this set.
_BILLABLE_STATES = frozenset({"done", "deployed"})

# billing_state values that mean a charge attempt already STARTED for this run. Paired with a
# `cost_usd > 0` price (see `_needs_charge`), they recover a run whose charge was attempted but never
# landed -- e.g. a `done`/`deployed` run later `cancelled` while its charge was still `charging`/
# `failed`. `pending` is intentionally NOT here: it is the submit-time default a run cancelled BEFORE
# running any step also carries, and such a run has `cost_usd` 0 -- it is never charged.
_CHARGE_STARTED_STATES = frozenset({"charging", "failed"})


def charge_retry_enabled() -> bool:
    """Retrying completion charges needs the operator internal key (the same key the charge uses)."""
    return bool(os.environ.get(INTERNAL_KEY_ENV))


def _needs_charge(status: runner.RunStatus) -> bool:
    """True for a terminal run that carries customer billing context, has a price, and isn't charged.

    ``cost_usd`` is the amount we charge: the flash.cost quote for a completed run, or the estimate at
    the steps actually run for a run cancelled mid-training (set by deploy.cancel_run). Eligible when
    the run carries a billing context, isn't already `charged`, and:
      - it is in a billable terminal state (`done`/`deployed`) -- charge its quote; or
      - it is `cancelled` with `cost_usd > 0` -- charge the estimate-at-actual-steps the cancel path
        priced (a run cancelled before any step has `cost_usd` 0 -> nothing to charge); or
      - its charge machine already started (`charging`/`failed`) and it has a price (`cost_usd > 0`)
        -- recover a charge that was attempted but never landed.
    The backend route is idempotent by runId, so the inline charge and this sweep never double-charge.

    A run cancelled BEFORE running any step keeps the submit-time `pending`, a non-billable state, and
    `cost_usd` 0, so it stays ineligible -- a run that never trained is never charged."""
    if not status.billing_context:
        return False
    if status.billing_state == "charged":
        return False
    if status.state in _BILLABLE_STATES:
        return True
    cost = float(status.cost_usd or 0.0)
    if status.state == "cancelled" and cost > 0:
        return True
    return status.billing_state in _CHARGE_STARTED_STATES and cost > 0


def retry_completion_charges_once(should_stop: Callable[[], bool] | None = None) -> int:
    """One sweep: charge every terminal-but-uncharged run from its ``cost_usd`` (the quote for a
    completed run, or the estimate-at-actual-steps for a cancelled one).

    Returns how many runs ended this sweep ``charged``. Reuses the runner's charge hook (the single
    source of truth for the billing state machine -- it sets ``charging``, charges, then records
    ``charged``/``failed``), and the backend's idempotency-by-runId makes every retry safe.

    Resilient to one bad record: runs are listed by id and loaded ONE AT A TIME, so a single
    corrupt/legacy status file (which would make ``list_runs`` raise) is skipped instead of aborting
    the whole sweep and blocking charge recovery for every other run.

    ``should_stop`` is an optional cooperative-cancel callback checked BETWEEN runs. The startup/
    periodic sweeps run in a worker thread (the charge is blocking urllib), which ``task.cancel()``
    cannot interrupt; at shutdown the caller sets a stop flag so a backlog of slow charges can't keep
    the thread (and the default executor) alive long after the server was told to stop.
    """
    if not charge_retry_enabled():
        return 0
    from flash.runner.lifecycle import _charge_completed_run_by_id

    charged = 0
    for run_id in runner.list_run_ids():
        if should_stop is not None and should_stop():
            break
        with contextlib.suppress(Exception):
            # Load per-run so one unreadable/legacy status file is skipped (caught here), never
            # aborting recovery of the rest.
            status = runner.get_status(run_id)
            if not _needs_charge(status):
                continue
            # Charge by run id -- the charge reads everything it needs from the persisted RunStatus
            # (billing_context + cost_usd + the raw spec dict), so we never reparse the JobSpec. A
            # legacy/stale persisted spec that `JobSpec.from_dict` would reject must NOT block recovery
            # of a real pending/failed charge. Append to the run log so retries surface in
            # `flash log`.
            with open(runner.runs_file_path(run_id, ".log"), "a") as log:
                _charge_completed_run_by_id(run_id, log)
            if runner.get_status(run_id).billing_state == "charged":
                charged += 1
    return charged
