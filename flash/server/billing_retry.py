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
import os

from flash import runner
from flash.server.auth import INTERNAL_KEY_ENV

# States that produced a final, deployable adapter -> a completion charge applies. `deployed` is a
# `done` run with serving stood up on top -> still billable (its done-time charge may have failed
# before the deploy). A run still in one of these states whose charge is `pending` from submit but
# never ran (a crash between the `done` write and the charge) is recovered via this set.
_BILLABLE_STATES = frozenset({"done", "deployed"})

# billing_state values that PROVE the completion charge machine already ran, which only happens after
# a run reached `done` (lifecycle._charge_completed_run_best_effort). Kept as a defensive backup
# completion signal alongside the primary `artifacts_dir` anchor below.
_CHARGE_STARTED_STATES = frozenset({"charging", "failed"})


def charge_retry_enabled() -> bool:
    """Retrying completion charges needs the operator internal key (the same key the charge uses)."""
    return bool(os.environ.get(INTERNAL_KEY_ENV))


def _completed_training(status: runner.RunStatus) -> bool:
    """True iff the run reached `done` -- i.e. it finished training and produced a final adapter.

    `artifacts_dir` is the DURABLE completion marker: it is written ONLY on the `done` transition
    (runner.lifecycle line 555, runner.deploy lines 236/255) and never on submit/queued/running/
    failed/cancelled, and no later transition clears it. So a run that completed training carries it
    forever, even after it is deployed and then cancelled. This is the anchor that distinguishes a
    completed-then-cancelled run (owes its completion charge) from a run cancelled BEFORE it ever
    finished training (never owes one) -- neither `billing_context` nor `cost_usd` can, since both are
    set at submit / accumulated per-seed mid-run (runner.__init__ line 429, runner.lifecycle line 538)
    and so are present on a run cancelled before completion too.

    `billing_state in {charging, failed}` is a defensive backup: those are written only by the
    post-`done` charge machine, so they also imply completion (covers any legacy run missing
    artifacts_dir whose inline charge already attempted)."""
    return bool(status.artifacts_dir) or status.billing_state in _CHARGE_STARTED_STATES


def _needs_charge(status: runner.RunStatus) -> bool:
    """True for a completed run that carries customer billing context but isn't `charged` yet.

    Eligible when the run carries a customer billing context, isn't already `charged`, and EITHER it
    is still in a billable terminal state (`done`/`deployed`) OR it provably completed training
    (`_completed_training`). The second arm recovers a run that completed training and was then
    `cancelled` (e.g. a deployed run gets cancelled, or a recovery-path completion via attach_run that
    never ran the inline charge) while its charge was still `pending` -- such a run leaves
    `_BILLABLE_STATES` but provably finished, so its pending charge must still be recovered. A run
    cancelled BEFORE completing never reached `done`, so it has no `artifacts_dir` and stays
    ineligible: a run that never trained is never charged, even with a submit-time `pending`."""
    if not status.billing_context:
        return False
    if status.billing_state == "charged":
        return False
    if status.state in _BILLABLE_STATES:
        return True
    return _completed_training(status)


def retry_completion_charges_once() -> int:
    """One sweep: re-invoke the completion charge for every completed-but-uncharged run.

    Returns how many runs ended this sweep ``charged``. Reuses the runner's charge hook (the single
    source of truth for the billing state machine -- it sets ``charging``, charges, then records
    ``charged``/``failed``), and the backend's idempotency-by-runId makes every retry safe.
    """
    if not charge_retry_enabled():
        return 0
    from flash.runner.lifecycle import _charge_completed_run_by_id

    charged = 0
    for status in runner.list_runs():
        if not _needs_charge(status):
            continue
        with contextlib.suppress(Exception):
            # Charge by run id -- the charge reads everything it needs from the persisted RunStatus
            # (billing_context + cost_usd + the raw spec dict), so we never reparse the JobSpec. A
            # legacy/stale persisted spec that `JobSpec.from_dict` would reject must NOT block
            # recovery of a real pending/failed charge.
            # Append to the run log so retry attempts surface in `flash status --logs`.
            with open(runner.runs_file_path(status.run_id, ".log"), "a") as log:
                _charge_completed_run_by_id(status.run_id, log)
            if runner.get_status(status.run_id).billing_state == "charged":
                charged += 1
    return charged
