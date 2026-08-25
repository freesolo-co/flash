"""Remote resource reconciliation and cleanup tracking."""

from __future__ import annotations

import contextlib
import math
import time

from flash.core.spec import JobSpec
from flash.runner.accounting import costs
from flash.runner.lifecycle import attempts, reporting, state
from flash.runner.lifecycle import status as status_ops
from flash.runner.lifecycle.state import RunStatus


def _remote_resource_identity(remote: object) -> tuple | None:
    """Return the exact strict provider resource identity used for compare-and-clear."""
    if not isinstance(remote, dict):
        return None
    provider = remote.get("provider")
    try:
        if provider == "runpod":
            from flash.providers.runpod.execution.identity import RunpodPodHandle

            handle = RunpodPodHandle.from_dict(remote)
            return (
                provider,
                handle.attempt,
                handle.phase,
                handle.instance_id,
                handle.label,
                handle.key_fingerprint,
                handle.account_id,
                handle.payload_secret_id,
                handle.payload_secret_name,
            )
        if provider == "lambda":
            from flash.providers.lambda_.jobs.builders import LambdaJobHandle

            handle = LambdaJobHandle.from_dict(remote)
            return (
                provider,
                handle.attempt,
                handle.instance_id,
                handle.instance_type,
                handle.region,
                handle.name,
            )
        if provider == "vast":
            from flash.providers.vast.jobs.builders import VastJobHandle

            handle = VastJobHandle.from_dict(remote)
            return (
                provider,
                handle.attempt,
                handle.instance_id,
                handle.offer_id,
                handle.machine_id,
                handle.label,
            )
    except (TypeError, ValueError):
        return None
    return None


def _expected_remote_matches(current: object, expected: dict | None) -> bool:
    if expected is None:
        return current is None
    expected_identity = _remote_resource_identity(expected)
    return expected_identity is not None and _remote_resource_identity(current) == expected_identity


def _provider_cost_record(remote: dict, *, terminated_ts: float) -> dict | None:
    """Return immutable non-secret billing identity for one exact managed RunPod Pod attempt."""
    if remote.get("provider") != "runpod":
        return None

    from flash.providers.runpod.execution.identity import RunpodPodHandle

    handle = RunpodPodHandle.from_dict(remote)
    if handle.pending:
        return None
    provider_fields = {
        "account_id": handle.account_id,
        "data_center_id": handle.data_center_id,
        "gpu_count": handle.gpu_count,
        "key_fingerprint": handle.key_fingerprint,
    }
    ended = float(terminated_ts)
    if not math.isfinite(ended) or ended < handle.started_ts:
        raise ValueError("provider attempt termination timestamp is invalid")
    record = {
        "provider": handle.provider,
        "instance_id": handle.instance_id,
        "hourly_usd": handle.hourly_usd,
        "started_ts": handle.started_ts,
        "terminated_ts": ended,
        "attempt": handle.attempt,
        "gpu": handle.gpu,
        **provider_fields,
    }
    allocated_gpu = remote.get("allocated_gpu")
    if type(allocated_gpu) is str and allocated_gpu:
        record["allocated_gpu"] = allocated_gpu
    allocated_gpu_count = remote.get("allocated_gpu_count")
    if type(allocated_gpu_count) is int and allocated_gpu_count > 0:
        record["allocated_gpu_count"] = allocated_gpu_count
    return record


def _provider_cost_key(record: dict) -> tuple:
    return record.get("provider"), record.get("instance_id"), record.get("attempt")


def _merge_provider_cost_history(existing: object, incoming: object) -> list[dict] | None:
    merged: list[dict] = []
    by_key: dict[tuple, dict] = {}
    for value in (existing, incoming):
        if value is None:
            continue
        if type(value) is not list or any(type(item) is not dict for item in value):
            raise RuntimeError("persisted provider cost history is invalid")
        for item in value:
            record = dict(item)
            if record.get("provider") != "runpod":
                raise RuntimeError("provider cost history contains a non-RunPod record")
            key = _provider_cost_key(record)
            prior = by_key.get(key)
            if prior is not None:
                stable_prior = {
                    key: value for key, value in prior.items() if key != "terminated_ts"
                }
                stable_record = {
                    key: value for key, value in record.items() if key != "terminated_ts"
                }
                if stable_prior != stable_record:
                    raise RuntimeError("provider cost history conflicts with the exact attempt")
                continue
            by_key[key] = record
            merged.append(record)
    return merged or None


def _append_provider_cost_record(status: RunStatus, remote: dict, *, terminated_ts: float) -> None:
    record = _provider_cost_record(remote, terminated_ts=terminated_ts)
    if record is None:
        return
    status.provider_cost_history = _merge_provider_cost_history(
        status.provider_cost_history, [record]
    )


def _compare_and_clear_remote(run_id: str, expected_remote: dict) -> bool:
    """Clear only the nonterminal remote that still names the destroyed resource."""
    if _remote_resource_identity(expected_remote) is None:
        return False
    report_status: RunStatus | None = None
    with state._status_guard(run_id):
        status = status_ops.get_status(run_id)
        if status.state in state.TERMINAL_STATES:
            return False
        if not _expected_remote_matches(status.remote, expected_remote):
            return False
        cleared_at = time.time()
        _append_provider_cost_record(status, status.remote, terminated_ts=cleared_at)
        status.remote = None
        status.updated_at = cleared_at
        state._save_status_unlocked(status)
        report_status = status
    if report_status is not None:
        reporting._report_status(report_status)
    return True


def _compare_and_replace_remote(
    run_id: str,
    expected_remote: dict,
    replacement_remote: dict,
) -> bool:
    """Replace one pending RunPod identity with its exact Pod identity atomically."""
    try:
        from flash.providers.runpod.execution.identity import PHASE_ORDER, RunpodPodHandle

        expected = RunpodPodHandle.from_dict(expected_remote)
        replacement = RunpodPodHandle.from_dict(replacement_remote)
    except (TypeError, ValueError):
        return False
    stable_expected = (
        expected.attempt,
        expected.key_fingerprint,
        expected.account_id,
        expected.payload_secret_name,
        expected.gpu,
        expected.gpu_count,
    )
    stable_replacement = (
        replacement.attempt,
        replacement.key_fingerprint,
        replacement.account_id,
        replacement.payload_secret_name,
        replacement.gpu,
        replacement.gpu_count,
    )
    if (
        not expected.pending
        or PHASE_ORDER[replacement.phase] <= PHASE_ORDER[expected.phase]
        or stable_expected != stable_replacement
    ):
        return False
    report_status: RunStatus | None = None
    with state._status_guard(run_id):
        status = status_ops.get_status(run_id)
        if status.state in state.TERMINAL_STATES:
            return False
        if not _expected_remote_matches(status.remote, expected_remote):
            return False
        status.remote = {**status.remote, **replacement.to_dict()}
        status.updated_at = time.time()
        state._save_status_unlocked(status)
        report_status = status
    confirmed = status_ops.get_status(run_id)
    if not _expected_remote_matches(confirmed.remote, replacement.to_dict()):
        raise RuntimeError("exact provider handle replacement was not durably confirmed")
    if report_status is not None:
        reporting._report_status(report_status)
    return True


def _compare_and_prepare_resubmit(
    run_id: str,
    expected_remote: dict | None,
    *,
    expected_state: str | None = None,
) -> bool:
    """Claim a nonterminal recovery launch only while its expected remote still owns the run."""
    report_status: RunStatus | None = None
    with state._status_guard(run_id):
        status = status_ops.get_status(run_id)
        if status.state in state.TERMINAL_STATES:
            return False
        if expected_state is not None and status.state != expected_state:
            return False
        if not _expected_remote_matches(status.remote, expected_remote):
            return False
        status.state = "provisioning"
        status.updated_at = time.time()
        state._save_status_unlocked(status)
        report_status = status
    if report_status is not None:
        reporting._report_status(report_status)
    return True


def _compare_and_fail_remote(
    run_id: str,
    expected_remote: dict | None,
    error: str,
) -> bool:
    """CAS a nonterminal expected remote to failed and confirm the durable write."""
    report_status: RunStatus | None = None
    with state._status_guard(run_id):
        status = status_ops.get_status(run_id)
        if status.state in state.TERMINAL_STATES:
            return False
        if not _expected_remote_matches(status.remote, expected_remote):
            return False
        status.state = "failed"
        status.error = error
        status.updated_at = time.time()
        if status.finished_at is None:
            status.finished_at = status.updated_at
        state._save_status_unlocked(status)
        report_status = status
    confirmed = status_ops.get_status(run_id)
    expected_after = expected_remote
    if (
        confirmed.state != "failed"
        or not _expected_remote_matches(confirmed.remote, expected_after)
        or confirmed.error != error
    ):
        raise RuntimeError("terminal recovery failure was not durably confirmed")
    if report_status is not None:
        reporting._report_status(report_status)
    return True


def _compare_and_complete_remote(
    run_id: str,
    expected_remote: dict | None,
    spec: JobSpec,
    metrics: dict,
) -> bool:
    """Adopt strict completed artifacts only while the captured remote still owns the run."""
    report_status: RunStatus | None = None
    with state._status_guard(run_id):
        status = status_ops.get_status(run_id)
        if status.state in state.TERMINAL_STATES:
            return False
        if not _expected_remote_matches(status.remote, expected_remote):
            return False
    expected_attempt = (
        expected_remote.get("attempt")
        if isinstance(expected_remote, dict)
        else attempts._latest_reserved_attempt(run_id)
    )
    metrics, verified_attempt = status_ops.validate_terminal_source_metrics(
        status,
        metrics,
        expected_attempt=expected_attempt,
    )
    if expected_remote is not None and not _record_cleanup_remote(run_id, expected_remote):
        return False
    recovered_cost = status_ops._persist_metrics(spec, metrics)
    with state._status_guard(run_id):
        status = status_ops.get_status(run_id)
        if status.state in state.TERMINAL_STATES:
            return False
        if not _expected_remote_matches(status.remote, expected_remote):
            return False
        measured = float(status.cost_usd or 0.0) + recovered_cost
        charge_usd = costs._status_estimated_charge(status, spec, fallback=measured)
        status.state = "done"
        status.cost_usd = charge_usd
        status.artifacts_dir = state.artifacts_dir(spec)
        status.source_verified_attempt = verified_attempt
        status.updated_at = time.time()
        if status.finished_at is None:
            status.finished_at = status.updated_at
        state._save_status_unlocked(status)
        report_status = status
    confirmed = status_ops.get_status(run_id)
    if confirmed.state != "done" or not _expected_remote_matches(confirmed.remote, expected_remote):
        raise RuntimeError("terminal recovery completion was not durably confirmed")
    if report_status is not None:
        reporting._report_status(report_status)
    return True


def _canonical_cleanup_remote(remote: object) -> dict | None:
    """Return the complete strict teardown handle for one exact resource."""
    if not isinstance(remote, dict) or _remote_resource_identity(remote) is None:
        return None
    provider = remote.get("provider")
    try:
        if provider == "runpod":
            from flash.providers.runpod.execution.identity import RunpodPodHandle

            return RunpodPodHandle.from_dict(remote).to_dict()
        if provider == "lambda":
            from flash.providers.lambda_.jobs.builders import LambdaJobHandle

            return LambdaJobHandle.from_dict(remote).to_dict()
        if provider == "vast":
            from flash.providers.vast.jobs.builders import VastJobHandle

            return VastJobHandle.from_dict(remote).to_dict()
    except (TypeError, ValueError):
        return None
    return None


def _cleanup_remote_key(remote: object) -> tuple | None:
    record = _canonical_cleanup_remote(remote)
    if record is None:
        return None
    return _remote_resource_identity(record), record["attempt"]


def _cleanup_remotes_from_raw(raw: dict) -> list[dict]:
    value = raw.get(state._CLEANUP_REMOTES_KEY, [])
    if not isinstance(value, list):
        raise RuntimeError("stored cleanup remotes are invalid")
    records = []
    seen = set()
    for item in value:
        record = _canonical_cleanup_remote(item)
        key = _cleanup_remote_key(record)
        if record is None or key is None:
            raise RuntimeError("stored cleanup remote is invalid")
        if key not in seen:
            records.append(record)
            seen.add(key)
    return records


def _snapshot_cleanup_remotes(run_id: str) -> list[dict]:
    with state._status_guard(run_id):
        return _cleanup_remotes_from_raw(status_ops._load_status_json(run_id))


def _compare_and_remove_cleanup_remote(run_id: str, expected_remote: dict) -> bool:
    expected_key = _teardown_removal_key(expected_remote)
    if expected_key is None:
        return False
    with state._status_guard(run_id):
        raw = status_ops._load_status_json(run_id)
        try:
            records = _cleanup_remotes_from_raw(raw)
        except Exception:
            # the strict reader raises on the FIRST record it cannot canonicalize, and the drain
            # suppresses that -- so one bad sibling made every CONFIRMED-DELETED record undeletable,
            # and each sweep retried a resource that is already gone, forever. removing one record
            # does not require understanding the others: keep the ones that cannot be parsed exactly
            # as they are on disk, drop only the record whose teardown was confirmed. nothing is
            # silently discarded, and the strict reader still guards every other write path.
            value = raw.get(state._CLEANUP_REMOTES_KEY, [])
            if not isinstance(value, list):
                return False
            remaining = [item for item in value if _teardown_removal_key(item) != expected_key]
            if len(remaining) == len(value):
                return False
            status = status_ops._runstatus_from_json(raw)
            removed_at = time.time()
            _append_provider_cost_record(status, expected_remote, terminated_ts=removed_at)
            status.updated_at = removed_at
            state._save_status_unlocked(status, _cleanup_remotes=remaining or None)
            return True
        remaining = [record for record in records if _teardown_removal_key(record) != expected_key]
        if len(remaining) == len(records):
            return False
        status = status_ops._runstatus_from_json(raw)
        removed_at = time.time()
        _append_provider_cost_record(status, expected_remote, terminated_ts=removed_at)
        status.updated_at = removed_at
        state._save_status_unlocked(status, _cleanup_remotes=remaining or None)
    return True


def _teardown_removal_key(record: object) -> tuple | None:
    """Return the strict identity used to select one confirmed cleanup record."""
    return _cleanup_remote_key(record)


def _drainable_cleanup_remotes(run_id: str) -> list[dict]:
    """Return each strict cleanup record independently, skipping malformed siblings."""
    with state._status_guard(run_id):
        try:
            raw = status_ops._load_status_json(run_id)
        except FileNotFoundError:
            return []
    value = raw.get(state._CLEANUP_REMOTES_KEY, [])
    if not isinstance(value, list):
        return []
    records: list[dict] = []
    seen = set()
    for item in value:
        record = _canonical_cleanup_remote(item)
        key = _teardown_removal_key(record)
        if record is None or key is None or key in seen:
            continue
        records.append(record)
        seen.add(key)
    return records


def _drain_cleanup_remotes(run_id: str) -> set[tuple]:
    """Teardown every tracked resource independently, removing only confirmed exact records."""
    # teardown is per-resource, so a malformed sibling must not strand every strict record.
    try:
        records = _snapshot_cleanup_remotes(run_id)
    except Exception:
        records = _drainable_cleanup_remotes(run_id)
    attempted = set()
    if not records:
        return attempted
    from flash.providers.core.base import JobHandle
    from flash.runner.supervise.lifecycle import _strict_teardown_handle

    for record in records:
        identity = _remote_resource_identity(record)
        if identity is None:
            continue
        attempted.add(identity)
        try:
            resource_deleted = _strict_teardown_handle(JobHandle.from_dict(record), run_id)
        except Exception:
            continue
        if resource_deleted:
            with contextlib.suppress(Exception):
                _compare_and_remove_cleanup_remote(run_id, record)
    return attempted


def _record_cleanup_remote(run_id: str, remote: dict) -> bool:
    """Persist one exact cleanup identity without changing the active remote."""
    record = _canonical_cleanup_remote(remote)
    key = _cleanup_remote_key(record)
    if record is None or key is None:
        return False
    report_status: RunStatus | None = None
    with state._status_guard(run_id):
        raw = status_ops._load_status_json(run_id)
        status = status_ops._runstatus_from_json(raw)
        records = _cleanup_remotes_from_raw(raw)
        if all(_cleanup_remote_key(existing) != key for existing in records):
            records.append(record)
        status.updated_at = time.time()
        state._save_status_unlocked(status, _cleanup_remotes=records)
        report_status = status
    if report_status is not None:
        reporting._report_status(report_status)
    return True


def _preserve_cleanup_remote(run_id: str, remote: dict) -> bool:
    """Persist cleanup identity without changing a terminal lifecycle state."""
    record = _canonical_cleanup_remote(remote)
    key = _cleanup_remote_key(record)
    if record is None or key is None:
        return False
    report_status: RunStatus | None = None
    with state._status_guard(run_id):
        raw = status_ops._load_status_json(run_id)
        status = status_ops._runstatus_from_json(raw)
        records = _cleanup_remotes_from_raw(raw)
        if all(_cleanup_remote_key(existing) != key for existing in records):
            records.append(record)
        current_identity = _remote_resource_identity(status.remote)
        identity = _remote_resource_identity(record)
        if current_identity is None or current_identity == identity:
            status.remote = dict(remote)
        status.updated_at = time.time()
        state._save_status_unlocked(status, _cleanup_remotes=records)
        report_status = status
    if report_status is not None:
        reporting._report_status(report_status)
    return True
