"""Remote resource reconciliation and cleanup tracking."""

from __future__ import annotations

import contextlib
import time

import flash.runner as runner
from flash.core.spec import JobSpec
from flash.runner import RunStatus


def _remote_resource_identity(remote: object) -> tuple | None:
    """Return the exact strict provider resource identity used for compare-and-clear."""
    if not isinstance(remote, dict):
        return None
    provider = remote.get("provider")
    try:
        if provider == "runpod":
            from flash.providers.runpod.jobs import JobHandle as RunpodJobHandle

            handle = RunpodJobHandle.from_dict(remote)
            return (
                provider,
                handle.attempt,
                handle.endpoint_id,
                handle.job_id,
                handle.key_fingerprint,
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
    expected_identity = runner._remote_resource_identity(expected)
    return (
        expected_identity is not None
        and runner._remote_resource_identity(current) == expected_identity
    )


def _compare_and_clear_remote(run_id: str, expected_remote: dict) -> bool:
    """Clear only the nonterminal remote that still names the destroyed resource."""
    if runner._remote_resource_identity(expected_remote) is None:
        return False
    report_status: RunStatus | None = None
    with runner._status_guard(run_id):
        status = runner.get_status(run_id)
        if status.state in runner.TERMINAL_STATES:
            return False
        if not runner._expected_remote_matches(status.remote, expected_remote):
            return False
        status.remote = None
        status.updated_at = time.time()
        runner._save_status_unlocked(status)
        report_status = status
    if report_status is not None:
        runner._report_status(report_status)
    return True


def _compare_and_prepare_resubmit(
    run_id: str,
    expected_remote: dict | None,
    *,
    expected_state: str | None = None,
) -> bool:
    """Claim a nonterminal recovery launch only while its expected remote still owns the run."""
    report_status: RunStatus | None = None
    with runner._status_guard(run_id):
        status = runner.get_status(run_id)
        if status.state in runner.TERMINAL_STATES:
            return False
        if expected_state is not None and status.state != expected_state:
            return False
        if not runner._expected_remote_matches(status.remote, expected_remote):
            return False
        status.state = "provisioning"
        status.updated_at = time.time()
        runner._save_status_unlocked(status)
        report_status = status
    if report_status is not None:
        runner._report_status(report_status)
    return True


def _compare_and_fail_remote(
    run_id: str,
    expected_remote: dict | None,
    error: str,
) -> bool:
    """CAS a nonterminal expected remote to failed and confirm the durable write."""
    report_status: RunStatus | None = None
    with runner._status_guard(run_id):
        status = runner.get_status(run_id)
        if status.state in runner.TERMINAL_STATES:
            return False
        if not runner._expected_remote_matches(status.remote, expected_remote):
            return False
        previous_updated_at = status.updated_at
        status.state = "failed"
        status.error = error
        status.updated_at = time.time()
        if status.finished_at is None:
            status.finished_at = status.updated_at or previous_updated_at
        runner._save_status_unlocked(status)
        report_status = status
    confirmed = runner.get_status(run_id)
    expected_after = expected_remote
    if (
        confirmed.state != "failed"
        or not runner._expected_remote_matches(confirmed.remote, expected_after)
        or confirmed.error != error
    ):
        raise RuntimeError("terminal recovery failure was not durably confirmed")
    if report_status is not None:
        runner._report_status(report_status)
    return True


def _compare_and_complete_remote(
    run_id: str,
    expected_remote: dict | None,
    spec: JobSpec,
    metrics: dict,
) -> bool:
    """Adopt strict completed artifacts only while the captured remote still owns the run."""
    report_status: RunStatus | None = None
    with runner._status_guard(run_id):
        status = runner.get_status(run_id)
        if status.state in runner.TERMINAL_STATES:
            return False
        if not runner._expected_remote_matches(status.remote, expected_remote):
            return False
    if expected_remote is not None and not runner._record_cleanup_remote(run_id, expected_remote):
        return False
    recovered_cost = runner._persist_metrics(spec, metrics)
    with runner._status_guard(run_id):
        status = runner.get_status(run_id)
        if status.state in runner.TERMINAL_STATES:
            return False
        if not runner._expected_remote_matches(status.remote, expected_remote):
            return False
        measured = float(status.cost_usd or 0.0) + recovered_cost
        charge_usd = runner._status_estimated_charge(status, spec, fallback=measured)
        status.state = "done"
        status.cost_usd = charge_usd
        status.artifacts_dir = runner.artifacts_dir(spec)
        status.updated_at = time.time()
        if status.finished_at is None:
            status.finished_at = status.updated_at
        runner._save_status_unlocked(status)
        report_status = status
    confirmed = runner.get_status(run_id)
    if confirmed.state != "done" or not runner._expected_remote_matches(
        confirmed.remote, expected_remote
    ):
        raise RuntimeError("terminal recovery completion was not durably confirmed")
    if report_status is not None:
        runner._report_status(report_status)
    return True


def _canonical_cleanup_remote(remote: object) -> dict | None:
    """Return the complete strict teardown handle for one exact resource."""
    if not isinstance(remote, dict) or runner._remote_resource_identity(remote) is None:
        return None
    provider = remote.get("provider")
    try:
        if provider == "runpod":
            from flash.providers.runpod.jobs import JobHandle as RunpodJobHandle

            return RunpodJobHandle.from_dict(remote).to_dict()
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
    record = runner._canonical_cleanup_remote(remote)
    if record is None:
        return None
    return runner._remote_resource_identity(record), record["attempt"]


def _cleanup_remotes_from_raw(raw: dict) -> list[dict]:
    value = raw.get(runner._CLEANUP_REMOTES_KEY, [])
    if not isinstance(value, list):
        raise RuntimeError("stored cleanup remotes are invalid")
    records = []
    seen = set()
    for item in value:
        record = runner._canonical_cleanup_remote(item)
        key = runner._cleanup_remote_key(record)
        if record is None or key is None:
            raise RuntimeError("stored cleanup remote is invalid")
        if key not in seen:
            records.append(record)
            seen.add(key)
    return records


def _snapshot_cleanup_remotes(run_id: str) -> list[dict]:
    with runner._status_guard(run_id):
        return runner._cleanup_remotes_from_raw(runner._load_status_json(run_id))


def _compare_and_remove_cleanup_remote(run_id: str, expected_remote: dict) -> bool:
    expected_key = runner._cleanup_remote_key(expected_remote)
    if expected_key is None:
        return False
    with runner._status_guard(run_id):
        raw = runner._load_status_json(run_id)
        records = runner._cleanup_remotes_from_raw(raw)
        remaining = [
            record for record in records if runner._cleanup_remote_key(record) != expected_key
        ]
        if len(remaining) == len(records):
            return False
        runner._save_status_unlocked(
            runner._runstatus_from_json(raw),
            _cleanup_remotes=remaining or None,
        )
    return True


def _drain_cleanup_remotes(run_id: str) -> set[tuple]:
    """Teardown every tracked resource independently, removing only confirmed exact records."""
    records = runner._snapshot_cleanup_remotes(run_id)
    attempted = set()
    if not records:
        return attempted
    from flash.providers.base import JobHandle
    from flash.runner.supervise.lifecycle import _strict_teardown_handle

    for record in records:
        identity = runner._remote_resource_identity(record)
        if identity is None:
            continue
        attempted.add(identity)
        try:
            resource_deleted = _strict_teardown_handle(JobHandle.from_dict(record), run_id)
        except Exception:
            continue
        if resource_deleted:
            with contextlib.suppress(Exception):
                runner._compare_and_remove_cleanup_remote(run_id, record)
    return attempted


def _record_cleanup_remote(run_id: str, remote: dict) -> bool:
    """Persist one exact cleanup identity without changing the active remote."""
    record = runner._canonical_cleanup_remote(remote)
    key = runner._cleanup_remote_key(record)
    if record is None or key is None:
        return False
    report_status: RunStatus | None = None
    with runner._status_guard(run_id):
        raw = runner._load_status_json(run_id)
        status = runner._runstatus_from_json(raw)
        records = runner._cleanup_remotes_from_raw(raw)
        if all(runner._cleanup_remote_key(existing) != key for existing in records):
            records.append(record)
        status.updated_at = time.time()
        runner._save_status_unlocked(status, _cleanup_remotes=records)
        report_status = status
    if report_status is not None:
        runner._report_status(report_status)
    return True


def _preserve_cleanup_remote(run_id: str, remote: dict) -> bool:
    """Persist cleanup identity without changing a terminal lifecycle state."""
    record = runner._canonical_cleanup_remote(remote)
    key = runner._cleanup_remote_key(record)
    if record is None or key is None:
        return False
    report_status: RunStatus | None = None
    with runner._status_guard(run_id):
        raw = runner._load_status_json(run_id)
        status = runner._runstatus_from_json(raw)
        records = runner._cleanup_remotes_from_raw(raw)
        if all(runner._cleanup_remote_key(existing) != key for existing in records):
            records.append(record)
        current_identity = runner._remote_resource_identity(status.remote)
        identity = runner._remote_resource_identity(record)
        if current_identity is None or current_identity == identity:
            status.remote = dict(remote)
        status.updated_at = time.time()
        runner._save_status_unlocked(status, _cleanup_remotes=records)
        report_status = status
    if report_status is not None:
        runner._report_status(report_status)
    return True
