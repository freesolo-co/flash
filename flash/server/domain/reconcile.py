"""Daily realized-cost reconciliation for finished runs.

Reports provider COGS with the operator key, best-effort and off-path. Attribution settles over the
durable attempt-resource ledger (flash/runner/attempt_resources.py), so a retried run bills every
resource it created and a cancelled run whose handle was cleared still bills the one it burned.

Settlement is partial-now, top-up-later: the settled subset is reported immediately, and the run is
left unreconciled while any resource is still within its settle delay so a later sweep can restate a
higher total. `reconciled_at` closes the run only once every resource has settled or the window has
closed. Per-resource usage is persisted, so a restatement never double-counts.
"""

from __future__ import annotations

import contextlib
import json
import time
import urllib.request

from flash import runner
from flash.providers.realized import realized_cost_for_remote
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


def _terminal_ts(status: runner.RunStatus) -> float:
    """The run's training-teardown time, used for both billing and eligibility.

    Prefer immutable `finished_at`; deploys and late updates move `updated_at` and distort settle
    delay and window. Fall back for pre-feature runs, preserving `finished_at == 0.0`.
    """
    return float(status.finished_at if status.finished_at is not None else status.updated_at)


def _resource_terminal_ts(record: dict, status: runner.RunStatus) -> float:
    """When one resource stopped costing money: its own teardown, else the run's terminal time.

    A retried run's earlier resource was destroyed long before the run ended, so billing it to the
    run's `finished_at` invents rental nobody was charged for.
    """
    terminal = record.get("terminal_at")
    if isinstance(terminal, (int, float)) and not isinstance(terminal, bool) and terminal > 0:
        return float(terminal)
    return _terminal_ts(status)


def _resource_start_ts(record: dict, status: runner.RunStatus) -> float:
    """When one resource started costing money, falling back the way `reconcile_run` always has.

    A persisted handle may omit `started_ts` or hold a falsey value; 0.0 means an unknown launch
    rather than the epoch, and billing flat-rate instances from 1970 would be catastrophic.
    """
    launched = record.get("launched_at")
    if isinstance(launched, (int, float)) and not isinstance(launched, bool) and launched > 0:
        return float(launched)
    identity = record.get("identity") if isinstance(record.get("identity"), dict) else {}
    return float(identity.get("started_ts") or status.created_at)


def _resource_settled(record: dict, status: runner.RunStatus, now: float) -> bool:
    """Whether this resource's provider invoice has had time to settle."""
    return (now - _resource_terminal_ts(record, status)) >= _SETTLE_SECONDS


def _window_closed(status: runner.RunStatus, now: float) -> bool:
    """Whether the run has aged out, after which its total is final however partial it is."""
    return (now - _terminal_ts(status)) > _WINDOW_SECONDS


def _due(status: runner.RunStatus, now: float) -> bool:
    """Whether a run should be reconciled this pass: a billable run whose training is finished
    (a terminal billable state, or `deployed` -- see _RECONCILABLE_STATES), not yet reconciled,
    still within the window, and holding at least one settled paid resource.

    Eligibility comes from the ledger rather than `status.remote`, which is why a cleanly cancelled
    run -- whose handle was cleared by its own confirmed teardown -- is still reconciled.
    """
    if status.state not in _RECONCILABLE_STATES:
        return False
    if status.reconciled_at:
        return False
    # window bounds the RUN (from teardown, not a later updated_at bump -- see _terminal_ts); the
    # settle delay is per-resource, since each one stopped billing at its own teardown.
    if _window_closed(status, now):
        return False
    try:
        records = runner.attempt_resources_for_status(status)
    except Exception:
        return False
    return any(_resource_settled(record, status, now) for record in records)


def _settle_resource(record: dict, status: runner.RunStatus):
    """Pull realized cost for exactly one paid resource, over ITS OWN billing window."""
    identity = record.get("identity") if isinstance(record.get("identity"), dict) else {}
    if not identity:
        return None
    start = _resource_start_ts(record, status)
    resource_end = _resource_terminal_ts(record, status)
    # RunPod's billing query pads past the resource's end so the settled invoice is in range; the
    # instance providers bill flat $/hr to teardown, so they get the UN-padded end (no extra hour).
    return realized_cost_for_remote(
        identity, start=start, end=resource_end + _SETTLE_SECONDS, run_end=resource_end
    )


def _resource_audit(record: dict, realized) -> dict:
    """Per-resource audit detail, nested under the report's free-form `source`."""
    return {
        "attempt": record.get("attempt"),
        "provider": realized.provider,
        "allocation": record.get("allocation"),
        "acceptedRateUsdHr": record.get("accepted_rate_usd_hr"),
        "launchedAt": record.get("launched_at"),
        "terminalAt": record.get("terminal_at"),
        "teardownRequestedAt": record.get("teardown_requested_at"),
        "deletionConfirmedAt": record.get("deletion_confirmed_at"),
        "outcome": record.get("outcome"),
        "realizedCostUsd": realized.realized_usd,
        "costByResource": realized.by_resource,
        "wallSeconds": realized.wall_seconds,
        "source": realized.source,
    }


def _settled_totals(records: list[dict], status: runner.RunStatus, now: float) -> dict:
    """Settle every resource whose invoice has had time to land, and total them."""
    total = 0.0
    wall = 0.0
    by_resource: dict[str, float] = {}
    providers: set[str] = set()
    gpus: set[str] = set()
    audit: list[dict] = []
    usage: dict[tuple, dict] = {}
    unsettled = 0
    for record in records:
        if not _resource_settled(record, status, now):
            unsettled += 1
            continue
        realized = _settle_resource(record, status)
        if realized is None or realized.realized_usd <= 0:
            # nothing billed yet: treat as unsettled so a later sweep can still top it up rather
            # than closing the run at a total that silently omits it.
            unsettled += 1
            continue
        total += realized.realized_usd
        wall += float(realized.wall_seconds or 0.0)
        for key, value in (realized.by_resource or {}).items():
            by_resource[key] = by_resource.get(key, 0.0) + float(value)
        providers.add(realized.provider)
        allocation = record.get("allocation") if isinstance(record.get("allocation"), dict) else {}
        if allocation.get("gpu"):
            gpus.add(str(allocation["gpu"]))
        audit.append(_resource_audit(record, realized))
        key = runner._attempt_resource_key(record.get("identity"))
        if key is not None:
            usage[key] = _resource_audit(record, realized)
    return {
        "total": round(total, 6),
        "wall_seconds": wall or None,
        "by_resource": by_resource,
        "providers": providers,
        "gpus": gpus,
        "audit": audit,
        "usage": usage,
        "unsettled": unsettled,
    }


def _single(values: set[str], fallback=None):
    """One value when every resource agrees, "mixed" when they genuinely differ."""
    if not values:
        return fallback
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def reconcile_run(status: runner.RunStatus, *, now: float | None = None) -> bool:
    """Pull + report realized cost for one run. Returns True when a positive total was reported.

    Every paid resource in the run's ledger is queried over its own billing window and summed, so a
    retried run reports what it actually spent rather than only its last handle. The run is marked
    reconciled ONLY once nothing is left to settle (or the window closed): while any resource is
    still unsettled the total is reported but left open, so a later sweep restates it upward. A
    zero/None total leaves the run unreconciled and is retried within the window.
    """
    now = time.time() if now is None else now
    spec = status.spec or {}
    try:
        records = runner.attempt_resources_for_status(status)
    except Exception:
        return False
    if not records:
        return False

    settled = _settled_totals(records, status, now)
    if settled["total"] <= 0:
        return False

    remote = status.remote or {}
    body = {
        "runId": status.run_id,
        "realizedCostUsd": settled["total"],
        "provider": _single(settled["providers"]),
        "gpu": _single(
            settled["gpus"],
            remote.get("allocated_gpu") or (spec.get("gpu") or {}).get("type"),
        ),
        "costByResource": settled["by_resource"],
        "wallSeconds": settled["wall_seconds"],
        "costBasis": "realized",
        # per-resource audit detail rides inside the existing free-form `source`, so the top-level
        # report fields keep the names and types the backend already reads.
        "source": {"attemptResources": settled["audit"]},
    }
    if not _report(body):
        return False

    # persist per-resource usage first: a later top-up restates from these records, so they have to
    # survive even if the aggregate write below is lost.
    with contextlib.suppress(Exception):
        runner._record_attempt_resource_usage(status.run_id, settled["usage"])
    # persist realized-cost fields only. `status` is stale, and writing its state could revert a run
    # that advanced to nonterminal `deployed`, which terminal-sticky CAS would not protect.
    # `reconciled_at` closes the run for good, so it is withheld while anything can still be billed:
    # an unsettled resource must be able to raise this total on a later sweep.
    fully_settled = settled["unsettled"] == 0 or _window_closed(status, now)
    with contextlib.suppress(Exception):
        if fully_settled:
            runner.record_realized_cost(
                status.run_id,
                realized_cost_usd=settled["total"],
                reconciled_at=now,
            )
        else:
            runner.record_partial_realized_cost(status.run_id, realized_cost_usd=settled["total"])
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
