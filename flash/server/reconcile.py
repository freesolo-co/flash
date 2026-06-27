"""Daily realized-cost reconciliation: pull provider billing for finished runs and report to
the freesolo backend. Off the hot path; requires FREESOLO_INTERNAL_KEY to be enabled.
"""

from __future__ import annotations

import contextlib
import time
import urllib.error
import urllib.request

from flash import runner
from flash.providers.realized import realized_cost_for_remote
from flash.server._internal_client import (
    DEFAULT_TIMEOUT_S,
    build_internal_request,
    enabled,
    internal_key,
)

_REPORT_PATH = "/api/billing/training-cost"
_SETTLE_SECONDS = 3600.0  # 1h — wait for invoice to settle
_WINDOW_SECONDS = 7 * 86400.0  # only reconcile runs finished within last 7 days
_FREE_TERMINAL_STATES = frozenset({"dry_run"})
# `deployed` is not in runner.TERMINAL_STATES but training is done so its cost is final.
_RECONCILABLE_STATES = (runner.TERMINAL_STATES | {"deployed"}) - _FREE_TERMINAL_STATES


def reconcile_enabled() -> bool:
    """Return True when the operator internal key is set."""
    return enabled()


def _report(body: dict) -> bool:
    """POST realized cost to the backend. Returns True on 2xx, False on any failure."""
    key = internal_key()
    if not key:
        return False
    req = build_internal_request(_REPORT_PATH, body, token=key)
    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
            resp.read()
        return True
    except (urllib.error.URLError, OSError):
        return False


def _terminal_ts(status: runner.RunStatus) -> float:
    # Use finished_at (stamped once at teardown) not updated_at (moved by deploy/heartbeat/reconcile).
    # `is not None` not truthiness: finished_at==0.0 is valid, not a fallback sentinel.
    return float(status.finished_at if status.finished_at is not None else status.updated_at)


def _due(status: runner.RunStatus, now: float) -> bool:
    """True if this run should be reconciled: reconcilable state, not yet done, past settle delay, within window."""
    if status.state not in _RECONCILABLE_STATES:
        return False
    if status.reconciled_at:
        return False
    age = now - _terminal_ts(status)
    if age < _SETTLE_SECONDS or age > _WINDOW_SECONDS:
        return False
    return bool(status.remote)


def reconcile_run(status: runner.RunStatus, *, now: float | None = None) -> bool:
    """Pull and report realized cost for one run; returns True when a positive cost was reported."""
    now = time.time() if now is None else now
    remote = status.remote or {}
    # Truthiness not `is not None`: from_dict coerces missing started_ts to 0.0 (unknown launch,
    # not epoch), so fall back to created_at to avoid massively inflating flat $/hr cost.
    start = float(remote.get("started_ts") or status.created_at)
    run_end = _terminal_ts(status)
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

    # COST-FIELDS-ONLY: status is a stale snapshot; writing state back could revert a run that
    # advanced to `deployed` (not terminal, so CAS won't catch it).
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
