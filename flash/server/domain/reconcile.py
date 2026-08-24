"""Daily realized-cost reconciliation for finished runs.

Reports provider COGS with the operator key, best-effort and off-path. Attribution uses the last
`RunStatus.remote`, so multi-resource retries may be undercounted.

The reported figure is the provider bill with the run's measured startup removed: the quote bills
training wall only (cold start is reported, never charged), so comparing it against a bill that
includes boot, model load and engine init would misread the unbilled policy as estimator error.
The gross bill, the provider wall and the startup seconds ride along in ``source`` for audit.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import time
import urllib.request
from dataclasses import replace

from flash import runner
from flash.providers.realized import RealizedCost, realized_cost_for_remote
from flash.server.platform.auth import freesolo_base_url
from flash.server.platform.internal_client import internal_key

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
    except OSError:
        return False


def _measured_setup_seconds(status: runner.RunStatus) -> float | None:
    """Worker-measured cold start (worker start -> first optimizer step) for this run.

    Read from the ``metrics.json`` the lifecycle persists at ``status.artifacts_dir`` when the run
    finishes; the worker stamps ``setup_seconds`` disjoint from its training-loop ``wall_seconds``.
    None when the record is missing or unparseable (pre-instrumentation runs, runs that never
    reached training), in which case the bill is reported un-stripped."""
    if not status.artifacts_dir:
        return None
    try:
        with open(os.path.join(status.artifacts_dir, "metrics.json")) as f:
            metrics = json.load(f)
    except (OSError, ValueError):
        return None
    value = metrics.get("setup_seconds") if isinstance(metrics, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _exclude_startup(
    realized: RealizedCost, *, setup_seconds: float | None, provider_wall: float | None
) -> RealizedCost:
    """Strip the measured startup from the provider bill at the run's realized $/s.

    The bill is prorated by ``(wall - setup) / wall``, so the result is what the provider would
    have charged for the training loop alone. ``wall`` is the provider's own billed wall when the
    realized pull carries one, else the instance lifetime (``started_ts`` -> teardown). Startup is
    clamped to the wall, so a run whose cold start swallowed the whole instance nets to $0 and is
    still reported (the quote for such a run is the real miss, and hiding it would be worse).
    Every itemized resource is scaled by the same share; the gross figures stay in ``source``."""
    wall = realized.wall_seconds if realized.wall_seconds else provider_wall
    audit = {
        **realized.source,
        "grossRealizedUsd": realized.realized_usd,
        "providerWallSeconds": wall,
        "setupSeconds": setup_seconds,
    }
    if setup_seconds is None or not wall or wall <= 0:
        return replace(realized, source={**audit, "startupExcluded": False})
    setup = min(setup_seconds, wall)
    share = (wall - setup) / wall
    return replace(
        realized,
        realized_usd=round(realized.realized_usd * share, 6),
        by_resource={k: round(float(v) * share, 6) for k, v in realized.by_resource.items()},
        wall_seconds=wall - setup,
        source={**audit, "startupExcluded": True},
    )


def _terminal_ts(status: runner.RunStatus) -> float:
    """Return the immutable training-teardown time used for billing and eligibility."""
    if status.finished_at is None:
        raise ValueError(f"run {status.run_id} is missing finished_at")
    return float(status.finished_at)


def _due(status: runner.RunStatus, now: float) -> bool:
    """Whether a run should be reconciled this pass: a billable run whose training is finished
    (a terminal billable state, or `deployed` -- see _RECONCILABLE_STATES), not yet reconciled,
    past the settle delay, still within the window, and carrying a provider handle."""
    if status.state not in _RECONCILABLE_STATES:
        return False
    if status.reconciled_at or status.finished_at is None:
        return False
    age = now - _terminal_ts(
        status
    )  # from teardown, not a later updated_at bump (see _terminal_ts)
    if age < _SETTLE_SECONDS or age > _WINDOW_SECONDS:
        return False
    return bool(status.remote)


def reconcile_run(status: runner.RunStatus, *, now: float | None = None) -> bool:
    """Pull + report realized cost for one run; mark it reconciled on success. Returns True when
    a positive provider bill was pulled and reported (net of startup, which may round the
    reported figure to $0). A zero/None pull leaves the run unreconciled so a later cycle
    (within the window) retries once the provider invoice settles."""
    now = time.time() if now is None else now
    remote = status.remote or {}
    spec = status.spec or {}
    # runpod's billing query needs a lower bound even though its endpoint invoice is authoritative.
    # instance cost attribution independently requires a valid persisted started_ts and returns none
    # when it is absent or malformed rather than substituting this bound.
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
    # Drop the measured startup so the reported figure matches what the quote prices (training
    # wall only). The gross bill is preserved in ``source``; see _exclude_startup.
    realized = _exclude_startup(
        realized,
        setup_seconds=_measured_setup_seconds(status),
        provider_wall=max(0.0, run_end - start),
    )

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
