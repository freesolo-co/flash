"""Daily realized-cost reconciliation: pull what the GPU provider actually billed for each
finished run and report it to the freesolo backend for estimator accuracy tracking.

Flash charges customer-facing training usage from the completed run's final ``cost_usd``. This
job is the COGS side: the realized provider invoice (RunPod /v1/billing/endpoints).
The backend's training_cost_accuracy view joins the two per run to surface
charged-vs-realized error.

Best-effort and entirely off the run hot path: it runs in a background loop (see the server
lifespan), never blocks request handling, and any failure is swallowed and retried next cycle.
Realized cost is reported with the operator INTERNAL key (this is COGS, not a customer charge),
which also gates the whole feature -- with no FREESOLO_INTERNAL_KEY set, reconciliation is off.

Scope note (v1): cost is attributed from the run's last persisted handle (RunStatus.remote),
which is exact for the common single-seed run. A multi-seed run keeps only its final seed's
handle, so its realized cost is currently under-counted -- a known limitation to extend by
persisting every seed's resource id.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import urllib.error
import urllib.request

from flash import runner
from flash.providers.realized import realized_cost_for_remote
from flash.server.auth import INTERNAL_KEY_ENV, freesolo_base_url

_REPORT_PATH = "/api/billing/training-cost"
_REPORT_TIMEOUT_S = 10.0
# Provider billing lags; wait this long after a run goes terminal before pulling (so the
# invoice has settled) and stop retrying once a run is older than the window.
_SETTLE_SECONDS = 3600.0  # 1h
_WINDOW_SECONDS = 7 * 86400.0  # only reconcile runs that finished within the last 7 days
# States that incur no GPU cost -> never reconciled.
_FREE_TERMINAL_STATES = frozenset({"dry_run"})
# States whose training is finished and whose GPU cost is therefore final -> eligible for
# reconciliation. The terminal billable states plus `deployed`: a deployed run finished
# training (its training invoice has settled) before serving was stood up on top of it, so
# its realized training cost is final and must be reconciled like any other finished run.
# (`deployed` is intentionally NOT in runner.TERMINAL_STATES -- it's a live, undeployable-back
# state -- so it has to be added explicitly here.) Excludes the free states (e.g. dry_run).
_RECONCILABLE_STATES = (runner.TERMINAL_STATES | {"deployed"}) - _FREE_TERMINAL_STATES


def reconcile_enabled() -> bool:
    """Reconciliation (and its reporting) is on only when the operator internal key is set."""
    return bool(os.environ.get(INTERNAL_KEY_ENV))


def _report(body: dict) -> bool:
    """POST realized cost to the backend with the internal key (Bearer). Best-effort: returns
    True on a 2xx, False on any failure (never raises). Mirrors ``billing._post_billing`` but
    swallows errors -- a metering report must never affect anything."""
    key = os.environ.get(INTERNAL_KEY_ENV)
    if not key:
        return False
    req = urllib.request.Request(
        f"{freesolo_base_url()}{_REPORT_PATH}",
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_REPORT_TIMEOUT_S) as resp:
            resp.read()
        return True
    except (urllib.error.URLError, OSError):
        return False


def _terminal_ts(status: runner.RunStatus) -> float:
    """The run's training-teardown time, used for both billing (``run_end``) and eligibility
    (settle delay + window). Prefer the frozen ``finished_at`` over the mutable ``updated_at``:
    deploy / late heartbeat / reconcile all move ``updated_at`` past teardown, which would both
    DELAY the settle gate (it counts from the bump, not the finish) and let a long-finished run
    that was merely bumped look "recent" and slip back inside ``_WINDOW_SECONDS``. ``finished_at``
    is stamped once at the terminal transition and never moved; falls back to ``updated_at`` for
    pre-feature runs. ``is not None`` (not truthiness) so a legitimate ``finished_at == 0.0`` is
    honored rather than silently falling back to ``updated_at``."""
    return float(status.finished_at if status.finished_at is not None else status.updated_at)


def _due(status: runner.RunStatus, now: float) -> bool:
    """Whether a run should be reconciled this pass: a billable run whose training is finished
    (a terminal billable state, or `deployed` -- see _RECONCILABLE_STATES), not yet reconciled,
    past the settle delay, still within the window, and carrying a provider handle."""
    if status.state not in _RECONCILABLE_STATES:
        return False
    if status.reconciled_at:
        return False
    age = now - _terminal_ts(status)  # from teardown, not a later updated_at bump (see _terminal_ts)
    if age < _SETTLE_SECONDS or age > _WINDOW_SECONDS:
        return False
    return bool(status.remote)


def reconcile_run(status: runner.RunStatus, *, now: float | None = None) -> bool:
    """Pull + report realized cost for one run; mark it reconciled on success. Returns True when
    a positive realized cost was reported. A zero/None result leaves the run unreconciled so a
    later cycle (within the window) retries once the provider invoice settles."""
    now = time.time() if now is None else now
    remote = status.remote or {}
    # Truthiness (`or`), NOT `is not None`: this started_ts comes from a persisted provider handle
    # whose from_dict coerces a MISSING started_ts to 0.0 (see LambdaJobHandle.from_dict),
    # so 0.0 means "unknown launch", not a 1970 epoch launch. Billing the flat $/hr from 0.0 would
    # massively inflate realized cost, so fall back to created_at when started_ts is falsey/missing.
    start = float(remote.get("started_ts") or status.created_at)
    # The run's true terminal time (~teardown / billing stop); see _terminal_ts for why this is
    # the frozen finished_at rather than the mutable updated_at (which deploy/heartbeat move past
    # teardown and would make the instance providers' flat $/hr bill until that later event).
    run_end = _terminal_ts(status)
    # RunPod's billing query pads past run end so the settled invoice is in range; the instance
    # providers bill flat $/hr to teardown, so they get the UN-padded run_end (no extra settle hour).
    realized = realized_cost_for_remote(remote, start=start, end=run_end + _SETTLE_SECONDS, run_end=run_end)
    if realized is None or realized.realized_usd <= 0:
        return False

    body = {
        "runId": status.run_id,
        "realizedCostUsd": realized.realized_usd,
        "provider": realized.provider,
        "gpu": remote.get("allocated_gpu") or remote.get("gpu"),
        "costByResource": realized.by_resource,
        "wallSeconds": realized.wall_seconds,
        "costBasis": "realized",
        "source": realized.source,
    }
    if not _report(body):
        return False

    # Persist locally so we don't re-pull/re-report, and so `flash status` can show realized vs
    # estimated. COST-FIELDS-ONLY: record_realized_cost re-reads the run under the lock and writes
    # only the realized-cost columns, never `state`. The `status` here is an earlier snapshot, so
    # writing its `state` back could REVERT a run that advanced since (e.g. to `deployed`) -- which
    # the terminal-sticky CAS does NOT protect against, since `deployed` is non-terminal. Updating
    # only the cost columns keeps the run's current state intact.
    with contextlib.suppress(Exception):
        runner.record_realized_cost(
            status.run_id,
            realized_cost_usd=realized.realized_usd,
            reconciled_at=now,
        )
    return True


def reconcile_once(*, now: float | None = None) -> int:
    """One sweep over local runs: reconcile every due run. Returns how many were reported."""
    if not reconcile_enabled():
        return 0
    now = time.time() if now is None else now
    reported = 0
    for status in runner.list_runs():
        if not _due(status, now):
            continue
        with contextlib.suppress(Exception):
            if reconcile_run(status, now=now):
                reported += 1
    return reported
