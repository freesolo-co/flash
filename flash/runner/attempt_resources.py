"""Durable per-run ledger of every paid provider resource an attempt created.

`RunStatus.remote` is the ACTIVE resource pointer and `cleanup_remotes` is the actionable teardown
queue whose entries vanish on confirmed deletion, so neither is an accounting record: a retried run
overwrites `remote`, and a cancelled run clears it entirely. This module owns the third thing --
history. One record per paid resource, keyed WITHOUT the attempt so the same endpoint can never be
billed twice, carrying identity, allocation, accepted rate, launch/terminal timestamps, teardown
request, deletion confirmation, outcome, and realized usage.

Stored under the private `attempt_resources` status key, so it is carried by `_save_status` but is
not a `RunStatus` field and never reaches `to_dict()` / `/v1/runs`. Providers never touch it: every
mutation goes through the primitives here, inside the run's `_status_guard`.
"""

from __future__ import annotations

import contextlib
import time

import flash.runner as runner
from flash.runner import RunStatus

# terminal dispositions a resource can end in. `succeeded`/`failed` are the attempt's own outcome;
# `cancelled` is stamped when the run is cancelled while the resource was still open.
_OUTCOMES = frozenset({"succeeded", "failed", "cancelled"})
# write-once lifecycle fields. re-stamping keeps the FIRST value: teardown was requested once, and
# a later drain retrying the same resource must not move the billing clock forward or overwrite the
# disposition the resource actually ended in.
_WRITE_ONCE_FIELDS = frozenset(
    {"teardown_requested_at", "deletion_confirmed_at", "outcome", "outcome_at"}
)


def _attempt_resource_key(remote: object) -> tuple | None:
    """The paid-resource identity, deliberately EXCLUDING attempt.

    A resource is billed once no matter how many attempt records reference it, so the key is what
    the provider charges for: the endpoint, or the instance. This is NOT
    `_remote_resource_identity` (which includes attempt and job_id and serves active-handle
    compare-and-set); using that here would let one endpoint be billed once per attempt that
    touched it. `key_fingerprint` is deliberately excluded too -- it identifies the CREDENTIAL the
    resource was rented with, not the resource, and it is absent from older handles.
    """
    if not isinstance(remote, dict):
        return None
    provider = remote.get("provider")
    if provider == "runpod":
        endpoint_id = remote.get("endpoint_id")
        if not isinstance(endpoint_id, str) or not endpoint_id:
            return None
        return (provider, endpoint_id)
    from flash.providers import INSTANCE_PROVIDERS

    if provider in INSTANCE_PROVIDERS:
        instance_id = remote.get("instance_id")
        if isinstance(instance_id, bool) or not isinstance(instance_id, (str, int)):
            return None
        if isinstance(instance_id, str) and not instance_id:
            return None
        return (provider, str(instance_id))
    return None


def _attempt_resource_rate(remote: object, gpu: object) -> float | None:
    """The accepted WHOLE-RESOURCE hourly rate, or None when the handle does not carry one.

    Instance handles stamp `hourly_usd` for the whole box. RunPod prices per card, so the selected
    candidate rate is multiplied by the cards actually allocated -- the same reason
    `_persist_metrics` multiplies by `allocated_gpu_count`.
    """
    if not isinstance(remote, dict):
        return None
    rate = remote.get("hourly_usd")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool) and rate > 0:
        return float(rate)
    if not isinstance(gpu, dict):
        return None
    candidate = gpu.get("hourly_usd")
    if not isinstance(candidate, (int, float)) or isinstance(candidate, bool) or candidate <= 0:
        return None
    count = gpu.get("count") or remote.get("allocated_gpu_count") or 1
    with contextlib.suppress(TypeError, ValueError):
        return float(candidate) * max(1, int(count))
    return float(candidate)


def _attempt_resource_record(remote: dict, *, gpu: object = None, now: float) -> dict | None:
    """Build the launch record for one paid resource from its persisted handle."""
    key = runner._attempt_resource_key(remote)
    if key is None:
        return None
    # prefer the strict canonical handle, which is complete enough for teardown to name the exact
    # worker. an incomplete handle is still a BILLABLE resource though, so accounting keeps what it
    # was given rather than discarding the record: money spent is not conditional on the handle
    # being rich enough to destroy.
    identity = runner._canonical_cleanup_remote(remote) or dict(remote)
    allocation = {
        "gpu": remote.get("allocated_gpu"),
        "gpu_count": remote.get("allocated_gpu_count"),
    }
    if isinstance(gpu, dict):
        allocation = {
            "gpu": allocation["gpu"] or gpu.get("name"),
            "gpu_count": allocation["gpu_count"] or gpu.get("count"),
        }
    launched_at = remote.get("started_ts")
    if not isinstance(launched_at, (int, float)) or isinstance(launched_at, bool):
        launched_at = now
    return {
        "attempt": identity.get("attempt"),
        "provider": identity.get("provider"),
        "identity": identity,
        "allocation": allocation,
        "accepted_rate_usd_hr": runner._attempt_resource_rate(remote, gpu),
        "launched_at": float(launched_at or now),
        "outcome": None,
        "outcome_at": None,
        "teardown_requested_at": None,
        "deletion_confirmed_at": None,
        "terminal_at": None,
        "realized_usage": None,
    }


def _attempt_resources_from_raw(raw: dict) -> list[dict]:
    """Parse the stored ledger, failing closed on anything malformed.

    A missing key is a legacy run (there is nothing to read yet); a present-but-invalid value is
    corrupt financial history and must never be silently dropped, exactly like
    `_cleanup_remotes_from_raw`.
    """
    value = raw.get(runner._ATTEMPT_RESOURCES_KEY, [])
    if not isinstance(value, list):
        raise RuntimeError("stored attempt resources are invalid")
    records: list[dict] = []
    seen = set()
    for item in value:
        if not isinstance(item, dict):
            raise RuntimeError("stored attempt resource is invalid")
        key = runner._attempt_resource_key(item.get("identity"))
        if key is None:
            raise RuntimeError("stored attempt resource identity is invalid")
        if key in seen:
            raise RuntimeError("stored attempt resources contain a duplicate paid resource")
        seen.add(key)
        records.append(dict(item))
    return records


def _find_attempt_resource(records: list[dict], key: tuple) -> dict | None:
    for record in records:
        if runner._attempt_resource_key(record.get("identity")) == key:
            return record
    return None


def _merged_attempt_resource(existing: dict, incoming: dict) -> dict:
    """Fold a repeated launch into the record already held for that paid resource.

    Re-publishing the same handle happens legitimately (recovery re-persists an adopted remote), so
    a replay of the SAME resource enriches the record rather than duplicating or resetting it. A
    replay under a different attempt is a different billable claim on one resource and is refused.
    """
    if existing.get("attempt") != incoming.get("attempt"):
        raise RuntimeError("attempt resource is already recorded under a different attempt")
    merged = dict(existing)
    # job_id and friends are monotonic enrichment: the canonical handle only gains locator fields
    # as an attempt progresses, and dropping them would leave cleanup unable to name the worker.
    merged["identity"] = {
        **(existing.get("identity") or {}),
        **{k: v for k, v in (incoming.get("identity") or {}).items() if v is not None},
    }
    for field in ("allocation", "accepted_rate_usd_hr"):
        if not merged.get(field) and incoming.get(field):
            merged[field] = incoming[field]
    # the FIRST launch is when billing started; a later republish of the same box is not a relaunch.
    if incoming.get("launched_at") and merged.get("launched_at"):
        merged["launched_at"] = min(float(merged["launched_at"]), float(incoming["launched_at"]))
    return merged


def _upsert_attempt_resource_unlocked(raw: dict, record: dict) -> list[dict]:
    records = runner._attempt_resources_from_raw(raw)
    key = runner._attempt_resource_key(record.get("identity"))
    if key is None:
        raise RuntimeError("attempt resource identity is invalid")
    existing = runner._find_attempt_resource(records, key)
    if existing is None:
        records.append(record)
        return records
    records[records.index(existing)] = runner._merged_attempt_resource(existing, record)
    return records


def _record_attempt_resource_launch(
    run_id: str, remote: dict, *, gpu: object = None, now: float | None = None
) -> list[dict] | None:
    """Ledger records for `remote`, to be written in the SAME status write that publishes it.

    Returns the new record list for `_save_status_unlocked(..., _attempt_resources=...)`, or None
    when the handle names no billable resource. The caller must already hold `_status_guard`: the
    ledger entry and `status.remote` have to land in one atomic write, or a crash between them
    loses the only durable proof that a paid resource exists.
    """
    now = time.time() if now is None else now
    record = runner._attempt_resource_record(remote, gpu=gpu, now=now)
    if record is None:
        return None
    return runner._upsert_attempt_resource_unlocked(runner._load_status_json(run_id), record)


def _record_launched_attempt_resource(
    run_id: str, remote: dict, *, gpu: object = None, now: float | None = None
) -> bool:
    """Record a launch on its own, for a resource whose handle never reached `status.remote`.

    The normal path writes the ledger inside the status write that publishes the handle. When that
    write is REFUSED (the run went terminal mid-launch), the resource still exists and was still
    paid for, so it is recorded here before teardown. Never touches run state.
    """
    report_status: RunStatus | None = None
    with runner._status_guard(run_id):
        try:
            records = runner._record_attempt_resource_launch(run_id, remote, gpu=gpu, now=now)
        except FileNotFoundError:
            return False
        if records is None:
            return False
        status = runner._runstatus_from_json(runner._load_status_json(run_id))
        status.updated_at = time.time()
        runner._save_status_unlocked(status, _attempt_resources=records)
        report_status = status
    if report_status is not None:
        with contextlib.suppress(Exception):
            runner._report_status(report_status)
    return True


def _stamp_attempt_resource(run_id: str, remote: object, **stamps) -> bool:
    """Apply write-once lifecycle stamps to one resource's record.

    Silently no-ops for a resource with no ledger entry (a legacy run, or a handle that names
    nothing billable) -- teardown correctness must never depend on bookkeeping succeeding.
    """
    key = runner._attempt_resource_key(remote)
    if key is None:
        return False
    outcome = stamps.get("outcome")
    if outcome is not None and outcome not in _OUTCOMES:
        raise ValueError(f"unknown attempt resource outcome: {outcome!r}")
    report_status: RunStatus | None = None
    changed = False
    with runner._status_guard(run_id):
        try:
            raw = runner._load_status_json(run_id)
        except FileNotFoundError:
            return False
        records = runner._attempt_resources_from_raw(raw)
        record = runner._find_attempt_resource(records, key)
        if record is None:
            return False
        for field, value in stamps.items():
            if value is None:
                continue
            # write-once: the first observation is the true one. a retried drain must not move a
            # billing boundary, and a second outcome must not overwrite the first disposition.
            if field in _WRITE_ONCE_FIELDS and record.get(field) is not None:
                continue
            record[field] = value
            changed = True
        if not changed:
            return False
        # deletion is the moment the provider stops charging, so it also closes the billing clock.
        if record.get("deletion_confirmed_at") and not record.get("terminal_at"):
            record["terminal_at"] = record["deletion_confirmed_at"]
        status = runner._runstatus_from_json(raw)
        status.updated_at = time.time()
        runner._save_status_unlocked(status, _attempt_resources=records)
        report_status = status
    if report_status is not None:
        with contextlib.suppress(Exception):
            runner._report_status(report_status)
    return True


def _record_attempt_resource_teardown_requested(run_id: str, remote: object) -> bool:
    """Stamp that exact teardown was ASKED for, before the provider call is made."""
    return runner._stamp_attempt_resource(run_id, remote, teardown_requested_at=time.time())


def _record_attempt_resource_deletion_confirmed(run_id: str, remote: object) -> bool:
    """Stamp CONFIRMED deletion, which also closes the resource's billing clock.

    Only ever called on a proven deletion. A RunPod endpoint whose worker is terminal but whose
    endpoint deletion is unconfirmed must not reach here -- it is still billable.
    """
    return runner._stamp_attempt_resource(run_id, remote, deletion_confirmed_at=time.time())


def _record_attempt_resource_outcome(run_id: str, remote: object, outcome: str) -> bool:
    """Stamp the attempt disposition this resource ended in."""
    return runner._stamp_attempt_resource(run_id, remote, outcome=outcome, outcome_at=time.time())


def _mark_open_attempt_resources_cancelled(run_id: str) -> bool:
    """Stamp every still-open resource cancelled, for a run cancelled after `remote` was cleared.

    Cancellation tears the resource down and clears the active handle, so without this the spend
    would have no disposition at all. Billing clocks are NOT closed here: teardown confirmation
    owns `terminal_at`, and an unconfirmed endpoint may still be running.
    """
    report_status: RunStatus | None = None
    changed = False
    with runner._status_guard(run_id):
        try:
            raw = runner._load_status_json(run_id)
        except FileNotFoundError:
            return False
        records = runner._attempt_resources_from_raw(raw)
        now = time.time()
        for record in records:
            if record.get("outcome") is None:
                record["outcome"] = "cancelled"
                record["outcome_at"] = now
                changed = True
        if not changed:
            return False
        status = runner._runstatus_from_json(raw)
        status.updated_at = now
        runner._save_status_unlocked(status, _attempt_resources=records)
        report_status = status
    if report_status is not None:
        with contextlib.suppress(Exception):
            runner._report_status(report_status)
    return True


def _record_attempt_resource_usage(run_id: str, usage_by_key: dict[tuple, dict]) -> bool:
    """Persist per-resource realized usage so a later restatement cannot double-count.

    Writes ONLY ledger usage. `status` here is re-read under the guard rather than passed in, for
    the same reason `record_realized_cost` never writes run state: a reconciliation sweep holds a
    stale snapshot and could otherwise revert a run that has since advanced to `deployed`.
    """
    if not usage_by_key:
        return False
    report_status: RunStatus | None = None
    changed = False
    with runner._status_guard(run_id):
        try:
            raw = runner._load_status_json(run_id)
        except FileNotFoundError:
            return False
        records = runner._attempt_resources_from_raw(raw)
        for key, usage in usage_by_key.items():
            record = runner._find_attempt_resource(records, key)
            if record is None:
                continue
            record["realized_usage"] = usage
            changed = True
        if not changed:
            return False
        status = runner._runstatus_from_json(raw)
        status.updated_at = time.time()
        runner._save_status_unlocked(status, _attempt_resources=records)
        report_status = status
    if report_status is not None:
        with contextlib.suppress(Exception):
            runner._report_status(report_status)
    return True


def _legacy_attempt_resources(status: RunStatus) -> list[dict]:
    """Synthesize records for a run that predates the ledger, from what still survives.

    A pre-ledger run kept exactly two things: the surviving `remote` and any unconfirmed cleanup
    entries. Resources already torn down left no trace, so this reconstructs what is still
    addressable rather than inventing history -- such a run keeps today's single-handle behavior.
    """
    records: list[dict] = []
    seen = set()
    terminal = float(status.finished_at if status.finished_at is not None else status.updated_at)
    candidates = [status.remote] if isinstance(status.remote, dict) else []
    with contextlib.suppress(Exception):
        candidates.extend(runner._snapshot_cleanup_remotes(status.run_id))
    for remote in candidates:
        key = runner._attempt_resource_key(remote)
        if key is None or key in seen:
            continue
        record = runner._attempt_resource_record(remote, now=float(status.created_at))
        if record is None:
            continue
        seen.add(key)
        # a legacy run has no per-resource teardown time; the run's own terminal time is the only
        # billing boundary it ever had, which is exactly the behavior being preserved.
        record["terminal_at"] = terminal
        records.append(record)
    return records


def attempt_resources_for_status(status: RunStatus) -> list[dict]:
    """Every paid resource of a run: the durable ledger, or the legacy reconstruction."""
    try:
        raw = runner._load_status_json(status.run_id)
    except (FileNotFoundError, ValueError):
        return runner._legacy_attempt_resources(status)
    records = runner._attempt_resources_from_raw(raw)
    if records:
        return records
    return runner._legacy_attempt_resources(status)
