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

It runs once at startup (so a crash between the ``done`` write and the charge is recovered promptly,
even though those runs are terminal and outside recover_runs) and then on a fixed interval, which is
the bounded backoff. Gated on the operator internal key (the charge needs it); off entirely without
FREESOLO_INTERNAL_KEY, exactly like realized-cost reconciliation.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from flash import runner
from flash.server._internal_client import enabled

# States that produced a final, deployable adapter -> a completion charge applies. `deployed` is a
# `done` run with serving stood up on top -> still billable (its done-time charge may have failed
# before the deploy). A run still in one of these states whose charge is `pending` from submit but
# never ran (a crash between the `done` write and the charge) is recovered via this set.
_BILLABLE_STATES = frozenset({"done", "deployed"})

# billing_state values that PROVE the completion charge machine already ran for this run, which only
# happens after a run reached `done` (lifecycle._charge_completed_run_best_effort). So a run carrying
# one of these has DEFINITELY completed training and incurred a charge, no matter what its current
# state is now. This recovers a run that completed (`done`/`deployed`) and was LATER `cancelled`
# (cancel flips a deployed run to `cancelled`, deploy.py:116) while its charge was still pending in
# the backend -- without it, the widened cancel state would drop out of `_BILLABLE_STATES` and leak.
# `pending` is intentionally NOT here: it is the submit-time default that a run cancelled BEFORE ever
# completing also carries, so charging on `pending`+`cancelled` would bill a run that never trained.
_CHARGE_STARTED_STATES = frozenset({"charging", "failed"})


def charge_retry_enabled() -> bool:
    """Retrying completion charges needs the operator internal key (the same key the charge uses)."""
    return enabled()


def _needs_charge(status: runner.RunStatus) -> bool:
    """True for a completed run that carries customer billing context but isn't `charged` yet.

    Eligible when the run carries a customer billing context, isn't already `charged`, and EITHER it
    is still in a billable terminal state (`done`/`deployed`) OR its charge machine already started
    (`charging`/`failed`). The second arm recovers a run that completed training and was then
    `cancelled` (e.g. a deployed run gets cancelled) while its charge was pending -- such a run leaves
    `_BILLABLE_STATES` but provably completed, so its pending/failed charge must still be recovered.
    A run cancelled BEFORE completing keeps the submit-time `pending` and a non-billable state, so it
    stays ineligible: a run that never trained is never charged.

    KNOWN RESIDUAL LEAK (intentionally NOT closed here): a run that completed ALL training but whose
    FIRST charge attempt never ran (e.g. attach_run completes the last seed via deploy.py:255 without
    the inline charge, so billing_state stays `pending`) and is THEN deployed and cancelled ends up
    `cancelled`+`pending` and is skipped. It cannot be safely recovered without wrongly charging a
    PARTIAL multi-seed run: attach_run also writes `artifacts_dir`/partial `cost_usd` for an
    intermediate seed while still `running` (deploy.py:232-238), so no durable field distinguishes
    "fully completed then cancelled" from "one seed recovered, still running, then cancelled" once the
    state is overwritten by cancel. Closing this needs a new persistent "training completed" column
    (set ONLY at the terminal done hook, lifecycle.py:551). Flagged for a schema follow-up; charging
    a partial/incomplete run is worse than this narrow leak, so the predicate stays conservative."""
    if not status.billing_context:
        return False
    if status.billing_state == "charged":
        return False
    if status.state in _BILLABLE_STATES:
        return True
    return status.billing_state in _CHARGE_STARTED_STATES


def retry_completion_charges_once(should_stop: Callable[[], bool] | None = None) -> int:
    """One sweep: re-invoke the completion charge for every completed-but-uncharged run.

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
            # legacy/stale persisted spec that `JobSpec.from_dict` would reject must NOT block
            # recovery of a real pending/failed charge.
            # Append to the run log so retry attempts surface in `flash status --logs`.
            with open(runner.runs_file_path(run_id, ".log"), "a") as log:
                _charge_completed_run_by_id(run_id, log)
            if runner.get_status(run_id).billing_state == "charged":
                charged += 1
    return charged
