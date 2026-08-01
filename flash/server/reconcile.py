"""Daily realized-cost reconciliation: pull what the GPU provider actually billed for each
finished run and report it to the freesolo backend for estimator accuracy tracking.

Flash charges customer-facing training usage from the run's ``cost_usd`` (the flash.cost estimate).
This job is the COGS side: the realized provider invoice (RunPod /v1/billing/endpoints). What it
REPORTS (via ``/api/billing/training-cost``) is COGS, not a customer charge. The backend's
training_cost_accuracy view joins the two per run to surface charged-vs-realized error.

Best-effort and entirely off the run hot path: it never blocks request handling, and any failure is
swallowed and retried next cycle. Realized cost is reported with the operator INTERNAL key (this is
COGS, not a customer charge), which also gates the whole feature -- with no FREESOLO_INTERNAL_KEY
set, reconciliation is off.

DURABILITY (why this is a queue, not a timer). The invariant is that every billable terminal run
eventually gets a realized cost recorded, or is explicitly marked ``unattributable`` -- and that this
must NOT depend on any process being alive at a particular moment. The persisted run records ARE the
queue: a run is owed a pull until ``reconcile_state`` says otherwise. Concretely:

  * the sweep runs at STARTUP and then on an interval, so a control plane that restarts more often
    than the interval still reconciles (the old loop slept a full hour before its first sweep, so a
    plane that was up for less than that swept nothing, ever);
  * there is NO time window -- a run that was missed while the plane was down is picked up whenever
    it next comes up, instead of silently aging out;
  * failures are RECORDED (attempt count + reason), not silently skipped, and a run that exhausts
    _MAX_RECONCILE_ATTEMPTS is marked ``unattributable`` so it stops being retried and becomes a
    visible gap;
  * instance-provider runs are reported INLINE at completion (see
    runner.lifecycle._reconcile_completed_run_best_effort) because their cost is exact at teardown;
    this sweep is their backstop, not their primary path.

The backend route is an idempotent upsert by runId, so re-reporting a run is always safe -- which is
what makes unbounded retry sound.

Scope note (v1): cost is attributed from the run's last persisted handle (RunStatus.remote). This is
exact for the common single-attempt run; runs that retried across multiple resources may be
under-counted until every attempt's resource id is persisted.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from flash import runner
from flash.providers.realized import realized_cost_for_remote
from flash.server.auth import INTERNAL_KEY_ENV, freesolo_base_url

_REPORT_PATH = "/api/billing/training-cost"
_REPORT_TIMEOUT_S = 10.0
# RunPod bills through an invoice that lags teardown, so its realized pull waits this long after a
# run goes terminal. Instance providers (Lambda/Vast) bill a FLAT $/hr against a lease whose rate and
# launch time are already stamped on the handle, so their realized cost is exact arithmetic the
# moment the run ends -- nothing to settle, hence _SETTLE_BY_PROVIDER = 0 for them (see
# _settle_seconds). That is what lets the completion path report them inline.
_RUNPOD_SETTLE_SECONDS = 3600.0  # 1h
# There is deliberately NO time window. A realized pull is OWED for every billable run, so a run that
# has not been reconciled stays due until it either succeeds or exhausts _MAX_RECONCILE_ATTEMPTS. The
# old 7-day window silently converted "still owed" into "never" for any run whose control plane was
# down during the window -- which is exactly how the estimator accuracy dataset ended up empty.
# Bounded attempts (not elapsed time) is what stops an unrecoverable run from being retried forever.
# Budgets PROVIDER-ATTRIBUTION failures ONLY (no cost came back). Backend-delivery failures are
# counted separately and are never budgeted -- see runner.record_reconcile_attempt: a run whose cost
# we successfully computed must not be dropped because the backend happened to be down.
_MAX_RECONCILE_ATTEMPTS = 24
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


def _settle_seconds(status: runner.RunStatus) -> float:
    """How long after teardown this run's realized cost can be pulled.

    RunPod needs its invoice to settle; instance providers (Lambda/Vast) bill a flat $/hr against a
    lease whose rate and launch time are already on the handle, so their cost is final at teardown and
    settles instantly. Unknown/missing provider falls back to the conservative RunPod delay."""
    from flash.providers import INSTANCE_PROVIDERS

    provider = (status.remote or {}).get("provider")
    return 0.0 if provider in INSTANCE_PROVIDERS else _RUNPOD_SETTLE_SECONDS


def _due(status: runner.RunStatus, now: float) -> bool:
    """Whether a run should be reconciled this pass: a billable run whose training is finished
    (a terminal billable state, or `deployed` -- see _RECONCILABLE_STATES), still owed a realized
    pull, past its provider's settle delay, and carrying a provider handle.

    Deliberately UNBOUNDED in age -- see _MAX_RECONCILE_ATTEMPTS. A run stays due until it reconciles
    or is marked `unattributable`, so a control plane that was down when the run finished still picks
    it up on its next startup sweep instead of losing the data point forever."""
    if status.state not in _RECONCILABLE_STATES:
        return False
    if status.reconciled_at or status.reconcile_state in ("reconciled", "unattributable"):
        return False
    # `delivering`: the provider already gave us a cost and it is staged on the run. Only the backend
    # POST is owed, so NONE of the provider-side gates apply -- not the settle delay, not the
    # attribution budget, not even still having a handle. Retried until the backend accepts it.
    if status.reconcile_state == "delivering":
        return bool(status.reconcile_report)
    if int(status.reconcile_attempts or 0) >= _MAX_RECONCILE_ATTEMPTS:
        return False
    age = now - _terminal_ts(
        status
    )  # from teardown, not a later updated_at bump (see _terminal_ts)
    if age < _settle_seconds(status):
        return False
    return bool(status.remote)


def _fail(run_id: str, reason: str, *, stage: str) -> bool:
    """Record a failed realized-cost attempt and return False (nothing reported this pass).

    ``stage`` is "attribution" (the provider returned no cost -> counts against the budget, and can
    end in ``unattributable``) or "report" (the cost WAS attributed, only the backend POST failed ->
    always retryable, never budgeted). Terminality is decided inside record_reconcile_attempt from
    the incremented persisted count, NOT from a snapshot here -- see its docstring for the race."""
    with contextlib.suppress(Exception):
        runner.record_reconcile_attempt(
            run_id,
            error=reason,
            stage=stage,
            max_attempts=_MAX_RECONCILE_ATTEMPTS,
        )
    return False


def _deliver(run_id: str, body: dict, *, now: float) -> bool:
    """POST an already-attributed cost and mark the run reconciled. Returns True on delivery.

    Shared by the fresh-pull path and the ``delivering`` retry path so both record the same states.
    The backend route is an idempotent upsert by runId, so re-delivering a payload a previous attempt
    may have actually landed (a response we never saw) is a safe no-op.

    COST-FIELDS-ONLY: record_realized_cost re-reads the run under the guard and writes only the
    realized-cost columns, never ``state`` -- a caller's snapshot could otherwise REVERT a run that
    advanced since (e.g. to ``deployed``), which the terminal-sticky CAS does not protect against."""
    if not _report(body):
        return _fail(run_id, "backend report failed", stage="report")
    with contextlib.suppress(Exception):
        runner.record_realized_cost(
            run_id,
            realized_cost_usd=float(body.get("realizedCostUsd") or 0.0),
            reconciled_at=now,
        )
    return True


def reconcile_run(status: runner.RunStatus, *, now: float | None = None) -> bool:
    """Pull + report realized cost for one run; mark it reconciled on success. Returns True when
    a positive realized cost was reported.

    Every failure is RECORDED (never a silent skip) so the run stays in the queue and is retried on
    the next sweep. Failures are split by stage: a provider pull that yields no cost counts against
    _MAX_RECONCILE_ATTEMPTS and eventually marks the run ``unattributable`` (a visible gap), whereas a
    failed backend POST is retried indefinitely -- the cost WAS attributed, only delivery failed, so a
    backend outage must not strand a run we successfully priced.

    A run already in ``delivering`` skips the provider entirely and re-POSTs its staged payload, so a
    backend retry can never be defeated by the provider becoming unreachable in the meantime."""
    now = time.time() if now is None else now
    if status.reconcile_state == "delivering" and status.reconcile_report:
        return _deliver(status.run_id, dict(status.reconcile_report), now=now)
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
    try:
        realized = realized_cost_for_remote(
            remote, start=start, end=run_end + _RUNPOD_SETTLE_SECONDS, run_end=run_end
        )
    except Exception as exc:  # provider billing API down / credentials rotated / rate limited
        return _fail(
            status.run_id,
            f"provider billing pull failed: {type(exc).__name__}",
            stage="attribution",
        )
    if realized is None:
        return _fail(
            status.run_id, "no realized cost attributable to this handle", stage="attribution"
        )
    if realized.realized_usd <= 0:
        return _fail(
            status.run_id, "provider reported no cost yet (invoice unsettled)", stage="attribution"
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
    # Stage the attribution BEFORE attempting delivery. From here the cost is ours: a failed POST
    # retries against this persisted payload rather than re-querying the provider, so provider access
    # lapsing during a backend outage can no longer destroy a cost we already computed.
    with contextlib.suppress(Exception):
        runner.record_attributed_cost(
            status.run_id, realized_cost_usd=realized.realized_usd, report=body
        )
    return _deliver(status.run_id, body, now=now)


def reconcile_once(
    *, now: float | None = None, should_stop: Callable[[], bool] | None = None
) -> int:
    """One sweep over local runs: reconcile every due run. Returns how many were reported.

    Resilient to one bad record: runs are listed by id and loaded ONE AT A TIME (like
    ``billing_retry.retry_completion_charges_once``), so a single corrupt/legacy status file is
    skipped instead of aborting the whole sweep. The old ``list_runs()`` parsed every record up front,
    so one unreadable file blocked reconciliation for every OTHER run too.

    ``should_stop`` is an optional cooperative-cancel callback checked BETWEEN runs. The sweep runs in
    a worker thread (the provider billing pull is blocking urllib) which ``task.cancel()`` cannot
    interrupt, so at shutdown the caller sets a stop flag rather than leaving a backlog of slow pulls
    holding the thread alive."""
    if not reconcile_enabled():
        return 0
    now = time.time() if now is None else now
    reported = 0
    for run_id in runner.list_run_ids():
        if should_stop is not None and should_stop():
            break
        with contextlib.suppress(Exception):
            status = runner.get_status(run_id)
            if not _due(status, now):
                continue
            if reconcile_run(status, now=now):
                reported += 1
    return reported
