"""Daily realized-cost reconciliation for finished runs.

Reports provider COGS with the operator key, best-effort and off-path. Attribution uses the last
`RunStatus.remote`, so multi-resource retries may be undercounted.
"""

from __future__ import annotations

import contextlib
import json
import time
import urllib.error
import urllib.request

from flash import runner
from flash.providers.realized import realized_cost_for_remote
from flash.server._internal_client import internal_key
from flash.server.auth import freesolo_base_url

_REPORT_PATH = "/api/billing/training-cost"
_REPORT_TIMEOUT_S = 10.0
# Provider billing lags; wait this long after a run goes terminal before pulling (so the
# invoice has settled) and stop retrying once a run is older than the window.
_SETTLE_SECONDS = 3600.0  # 1h
_WINDOW_SECONDS = 7 * 86400.0  # only reconcile runs that finished within the last 7 days
# States that incur no GPU cost -> never reconciled.
_FREE_TERMINAL_STATES = frozenset({"dry_run"})
# reconcile terminal billable states plus `deployed`, whose training invoice is final even though
# it is intentionally absent from `runner.TERMINAL_STATES`. exclude free states such as `dry_run`.
_RECONCILABLE_STATES = (runner.TERMINAL_STATES | {"deployed"}) - _FREE_TERMINAL_STATES


def reconcile_enabled() -> bool:
    """Reconciliation (and its reporting) is on only when the operator internal key is set.

    ``internal_key()`` is the shared gate for every backend reporter, so this is also off in
    standalone mode: realized cost is reported to a Freesolo billing backend a self-hosted plane
    has none of, and the loop would poll providers only to fail every POST."""
    return internal_key() is not None


def _report(body: dict) -> bool:
    """POST realized cost to the backend with the internal key (Bearer). Best-effort: returns
    True on a 2xx, False on any failure (never raises). Mirrors ``billing._post_billing`` but
    swallows errors -- a metering report must never affect anything."""
    key = internal_key()
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
    """The run's training-teardown time, used for both billing and eligibility.

    Prefer immutable `finished_at`; deploys and late updates move `updated_at` and distort settle
    delay and window. Fall back for pre-feature runs, preserving `finished_at == 0.0`.
    """
    return float(status.finished_at if status.finished_at is not None else status.updated_at)


def _due(status: runner.RunStatus, now: float) -> bool:
    """Whether a run should be reconciled this pass: a billable run whose training is finished
    (a terminal billable state, or `deployed` -- see _RECONCILABLE_STATES), not yet reconciled,
    past the settle delay, still within the window, and carrying a provider handle."""
    if status.state not in _RECONCILABLE_STATES:
        return False
    if status.reconciled_at:
        return False
    age = now - _terminal_ts(
        status
    )  # from teardown, not a later updated_at bump (see _terminal_ts)
    if age < _SETTLE_SECONDS or age > _WINDOW_SECONDS:
        return False
    return bool(status.remote)


def reconcile_run(status: runner.RunStatus, *, now: float | None = None) -> bool:
    """Pull + report realized cost for one run; mark it reconciled on success. Returns True when
    a positive realized cost was reported. A zero/None result leaves the run unreconciled so a
    later cycle (within the window) retries once the provider invoice settles."""
    now = time.time() if now is None else now
    remote = status.remote or {}
    spec = status.spec or {}
    # raw persisted RunStatus.remote may omit started_ts or contain a falsey value. 0.0 means an
    # unknown launch rather than the epoch; falling back to created_at prevents inflated flat-rate
    # instance billing.
    start = float(remote.get("started_ts") or status.created_at)
    # The run's true terminal time (~teardown / billing stop); see _terminal_ts for why this is
    # the frozen finished_at rather than the mutable updated_at (which deploy/heartbeat move past
    # teardown and would make the instance providers' flat $/hr bill until that later event).
    run_end = _terminal_ts(status)
    # RunPod's billing query pads past run end so the settled invoice is in range; the instance
    # providers bill flat $/hr to teardown, so they get the UN-padded run_end (no extra settle hour).
    realized = realized_cost_for_remote(
        remote, start=start, end=run_end + _SETTLE_SECONDS, run_end=run_end
    )
    if realized is None or realized.realized_usd <= 0:
        return False

    body = {
        "runId": status.run_id,
        "realizedCostUsd": realized.realized_usd,
        "provider": realized.provider,
        "gpu": remote.get("allocated_gpu") or (spec.get("gpu") or {}).get("type"),
        "costByResource": realized.by_resource,
        "wallSeconds": realized.wall_seconds,
        "costBasis": "realized",
        "source": realized.source,
    }
    if not _report(body):
        return False

    # persist realized-cost fields only. `status` is stale, and writing its state could revert a run
    # that advanced to nonterminal `deployed`, which terminal-sticky CAS would not protect.
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
