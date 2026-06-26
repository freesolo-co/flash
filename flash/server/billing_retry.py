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

# States that produced a final, deployable adapter -> the only states a completion charge applies to.
# A failed/cancelled run never completed, so it is never charged even if its billing_state is still
# `pending` from submit. `deployed` is a `done` run with serving stood up on top -> still billable
# (its done-time charge may have failed before the deploy).
_BILLABLE_STATES = frozenset({"done", "deployed"})


def charge_retry_enabled() -> bool:
    """Retrying completion charges needs the operator internal key (the same key the charge uses)."""
    return bool(os.environ.get(INTERNAL_KEY_ENV))


def _needs_charge(status: runner.RunStatus) -> bool:
    """True for a completed run that carries customer billing context but isn't `charged` yet."""
    if status.state not in _BILLABLE_STATES:
        return False
    if not status.billing_context:
        return False
    return status.billing_state != "charged"


def retry_completion_charges_once() -> int:
    """One sweep: re-invoke the completion charge for every completed-but-uncharged run.

    Returns how many runs ended this sweep ``charged``. Reuses the runner's charge hook (the single
    source of truth for the billing state machine -- it sets ``charging``, charges, then records
    ``charged``/``failed``), and the backend's idempotency-by-runId makes every retry safe.
    """
    if not charge_retry_enabled():
        return 0
    from flash.runner.lifecycle import _charge_completed_run_best_effort
    from flash.spec import JobSpec

    charged = 0
    for status in runner.list_runs():
        if not _needs_charge(status):
            continue
        try:
            spec = JobSpec.from_dict(status.spec)
        except Exception:
            # A malformed spec can't be charged; leave it for operator follow-up rather than
            # aborting the whole sweep.
            continue
        with contextlib.suppress(Exception):
            # Append to the run log so retry attempts surface in `flash status --logs`.
            with open(runner.runs_file_path(status.run_id, ".log"), "a") as log:
                _charge_completed_run_best_effort(spec, log)
            if runner.get_status(status.run_id).billing_state == "charged":
                charged += 1
    return charged
