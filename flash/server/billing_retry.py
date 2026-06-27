"""Retry completion-time customer charges for runs whose inline charge blipped or never ran."""

from __future__ import annotations

import contextlib
from collections.abc import Callable

from flash import runner
from flash.server._internal_client import enabled

# `deployed` is a `done` run with serving layered on top; its done-time charge may have failed before deploy.
_BILLABLE_STATES = frozenset({"done", "deployed"})

# `pending` intentionally NOT here: it's also the submit-time default for runs cancelled before training,
# so `pending`+`cancelled` would bill a run that never trained.
_CHARGE_STARTED_STATES = frozenset({"charging", "failed"})


def charge_retry_enabled() -> bool:
    """Requires operator internal key."""
    return enabled()


def _needs_charge(status: runner.RunStatus) -> bool:
    """True for a completed run with billing context that isn't `charged` yet.

    KNOWN RESIDUAL LEAK: a run completed via attach_run (billing_state stays `pending`) that is then
    deployed+cancelled ends up `cancelled`+`pending` and is skipped. Closing it requires a new
    "training completed" column (lifecycle.py:551); charging partial runs is worse than this narrow leak.
    """
    if not status.billing_context:
        return False
    if status.billing_state == "charged":
        return False
    if status.state in _BILLABLE_STATES:
        return True
    return status.billing_state in _CHARGE_STARTED_STATES


def retry_completion_charges_once(should_stop: Callable[[], bool] | None = None) -> int:
    """One sweep: re-invoke the completion charge for every completed-but-uncharged run. Returns count charged."""
    if not charge_retry_enabled():
        return 0
    from flash.runner.lifecycle import _charge_completed_run_by_id

    charged = 0
    for run_id in runner.list_run_ids():
        if should_stop is not None and should_stop():
            break
        with contextlib.suppress(Exception):
            status = runner.get_status(run_id)
            if not _needs_charge(status):
                continue
            # Charge by run_id so a legacy spec that JobSpec.from_dict would reject can't block recovery.
            with open(runner.runs_file_path(run_id, ".log"), "a") as log:
                _charge_completed_run_by_id(run_id, log)
            if runner.get_status(run_id).billing_state == "charged":
                charged += 1
    return charged
