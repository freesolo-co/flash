"""Remote resource reconciliation and cleanup tracking."""

from __future__ import annotations

import contextlib
import time

from flash.core.spec import JobSpec
from flash.runner.accounting import costs
from flash.runner.lifecycle import attempts, reporting, state
from flash.runner.lifecycle import status as status_ops
from flash.runner.lifecycle.state import RunStatus

# the provider-allocated identifier that names the billable resource itself, per provider handle
# above: runpod carries `endpoint_id`, lambda and vast carry `instance_id`. a record holding one of
# these still has something to delete even when the rest of it fails strict validation.
_RESOURCE_ID_FIELDS = ("endpoint_id", "instance_id")


def _remote_resource_identity(remote: object) -> tuple | None:
    """Return the exact strict provider resource identity used for compare-and-clear."""
    if not isinstance(remote, dict):
        return None
    provider = remote.get("provider")
    try:
        if provider == "runpod":
            from flash.providers.runpod.execution.jobs import JobHandle as RunpodJobHandle

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
    expected_identity = _remote_resource_identity(expected)
    return expected_identity is not None and _remote_resource_identity(current) == expected_identity


def _compare_and_clear_remote(run_id: str, expected_remote: dict) -> bool:
    """Clear only the nonterminal remote that still names the destroyed resource."""
    if _remote_resource_identity(expected_remote) is None:
        return False
    report_status: RunStatus | None = None
    with state._status_guard(run_id):
        status = status_ops.get_status(run_id)
        if status.state in state.TERMINAL_STATES:
            return False
        if _expected_remote_matches(status.remote, expected_remote):
            status.remote = None
        elif status.remote is None and _expected_remote_matches(
            status.cleanup_confirmed_remote, expected_remote
        ):
            status.cleanup_confirmed_remote = None
            if not _retain_remote_for_accounting(status) and _expected_remote_matches(
                status.realized_cost_remote, expected_remote
            ):
                status.realized_cost_remote = None
        else:
            return False
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
        if not _expected_remote_matches(status.remote, expected_remote) and not (
            status.remote is None
            and _expected_remote_matches(status.cleanup_confirmed_remote, expected_remote)
        ):
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
    expected_still_owned = _expected_remote_matches(confirmed.remote, expected_after) or (
        confirmed.remote is None
        and _expected_remote_matches(confirmed.cleanup_confirmed_remote, expected_after)
    )
    if confirmed.state != "failed" or not expected_still_owned or confirmed.error != error:
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
        confirmed_teardown = status.remote is None and _expected_remote_matches(
            status.cleanup_confirmed_remote, expected_remote
        )
        if not _expected_remote_matches(status.remote, expected_remote) and not confirmed_teardown:
            return False
    expected_attempt = (
        expected_remote.get("attempt")
        if isinstance(expected_remote, dict)
        else attempts.latest_reserved_attempt(run_id)
    )
    metrics, verified_attempt = status_ops.validate_terminal_source_metrics(
        status,
        metrics,
        expected_attempt=expected_attempt,
    )
    if (
        expected_remote is not None
        and not confirmed_teardown
        and not _record_cleanup_remote(run_id, expected_remote)
    ):
        return False
    recovered_cost = status_ops._persist_metrics(spec, metrics)
    with state._status_guard(run_id):
        status = status_ops.get_status(run_id)
        if status.state in state.TERMINAL_STATES:
            return False
        expected_still_owned = _expected_remote_matches(status.remote, expected_remote) or (
            status.remote is None
            and _expected_remote_matches(status.cleanup_confirmed_remote, expected_remote)
        )
        if not expected_still_owned:
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
    expected_still_owned = _expected_remote_matches(confirmed.remote, expected_remote) or (
        confirmed.remote is None
        and _expected_remote_matches(confirmed.cleanup_confirmed_remote, expected_remote)
    )
    if confirmed.state != "done" or not expected_still_owned:
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
            from flash.providers.runpod.execution.jobs import JobHandle as RunpodJobHandle

            record = RunpodJobHandle.from_dict(remote).to_dict()
        elif provider == "lambda":
            from flash.providers.lambda_.jobs.builders import LambdaJobHandle

            record = LambdaJobHandle.from_dict(remote).to_dict()
        elif provider == "vast":
            from flash.providers.vast.jobs.builders import VastJobHandle

            record = VastJobHandle.from_dict(remote).to_dict()
        else:
            return None
        return record
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


def _retain_remote_for_accounting(status: RunStatus) -> bool:
    return status.reconciled_at is None or (
        bool(status.billing_context) and status.billing_state != "charged"
    )


def _compare_and_confirm_remote_teardown(run_id: str, expected_remote: dict) -> bool:
    """clear one exact active remote after its teardown was independently confirmed."""
    if _remote_resource_identity(expected_remote) is None:
        return False
    report_status: RunStatus | None = None
    with state._status_guard(run_id):
        status = status_ops.get_status(run_id)
        if not _expected_remote_matches(status.remote, expected_remote):
            return False
        status.cleanup_confirmed_remote = dict(status.remote)
        if _retain_remote_for_accounting(status):
            status.realized_cost_remote = dict(status.remote)
        status.remote = None
        status.updated_at = time.time()
        state._save_status_unlocked(status)
        report_status = status
    if report_status is not None:
        reporting._report_status(report_status)
    return True


def _compare_and_remove_cleanup_remote(run_id: str, expected_remote: dict) -> bool:
    """Remove one confirmed cleanup target and its exact active remote atomically."""
    expected_key = _teardown_removal_key(expected_remote)
    if expected_key is None:
        return False
    report_status: RunStatus | None = None
    with state._status_guard(run_id):
        raw = status_ops._load_status_json(run_id)
        status = status_ops._runstatus_from_json(raw)
        try:
            records = _cleanup_remotes_from_raw(raw)
        except Exception:
            # the strict reader raises on the first record it cannot canonicalize. preserve every
            # unrecognized sibling verbatim and remove only the identity confirmed deleted.
            value = raw.get(state._CLEANUP_REMOTES_KEY, [])
            if not isinstance(value, list):
                return False
            remaining = [item for item in value if _teardown_removal_key(item) != expected_key]
            if len(remaining) == len(value):
                return False
        else:
            remaining = [
                record for record in records if _teardown_removal_key(record) != expected_key
            ]
            if len(remaining) == len(records):
                return False
        if _teardown_removal_key(status.remote) == expected_key:
            status.cleanup_confirmed_remote = dict(status.remote)
            if _retain_remote_for_accounting(status):
                status.realized_cost_remote = dict(status.remote)
            status.remote = None
        status.updated_at = time.time()
        state._save_status_unlocked(status, _cleanup_remotes=remaining or None)
        report_status = status
    if report_status is not None:
        reporting._report_status(report_status)
    return True


def _uncanonical_teardown_record(item: object) -> dict | None:
    """Return a record that names a deletable resource but fails strict canonicalization."""
    if not isinstance(item, dict):
        return None
    provider = item.get("provider")
    if not isinstance(provider, str) or not provider:
        return None
    if not any(isinstance(item.get(field), str) and item[field] for field in _RESOURCE_ID_FIELDS):
        return None
    return dict(item)


def _uncanonical_cleanup_remote_key(record: object) -> tuple | None:
    """Dedupe key for a record the strict reader cannot canonicalize."""
    if not isinstance(record, dict):
        return None
    identity = tuple(record.get(field) for field in _RESOURCE_ID_FIELDS)
    if not any(identity):
        return None
    return (record.get("provider"), record.get("attempt"), identity)


def _teardown_removal_key(record: object) -> tuple | None:
    """The identity the teardown path selects a record by, strict when possible.

    THE single derivation shared by the drain and the compare-and-remove that clears what the drain
    confirmed deleted. It has to be one function: the drain admits uncanonical records so a resource
    that fails strict validation still gets deleted, and if removal derived its key strictly instead
    it would return `None` for exactly those records and clear nothing -- so a confirmed-deleted
    resource would stay on disk and every later sweep would tear down something already gone,
    forever. That is the failure the lenient branch below exists to prevent, reintroduced through
    the key rather than through the reader.

    The two shapes cannot be confused for each other: the strict key is the 2-tuple
    `(identity, attempt)` and the lenient one is a 3-tuple, so they never compare equal and a record
    that canonicalizes is matched by its strict key on both sides.
    """
    return _cleanup_remote_key(record) or _uncanonical_cleanup_remote_key(record)


def _drainable_cleanup_remotes(run_id: str) -> list[dict]:
    """Every cleanup record that yields a teardown handle, skipping the ones that cannot.

    Only for the teardown path. Unlike the strict reader this never raises on a bad record, because
    raising would strand every well-formed sibling that is still billing.

    A record that fails canonicalization is still yielded verbatim when it names a resource. It is
    NOT true that such a record has nothing to delete: `key_fingerprint` is validated at exactly 68
    chars, while a deployed release writes the 16-char form, so every endpoint created by that
    release fails `from_dict` here. `_delete_runpod_endpoint` resolves precisely that case through
    `resolve_prefix_key_fingerprint`, and the teardown loop builds a base `JobHandle`, which
    validates only `provider`. Dropping these records here is what strands them -- a live RunPod
    endpoint then bills forever with nothing left to tear it down.
    """
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
        if record is None:
            record = _uncanonical_teardown_record(item)
        key = _teardown_removal_key(record)
        if record is None or key is None or key in seen:
            continue
        records.append(record)
        seen.add(key)
    return records


def _drain_cleanup_remotes(run_id: str) -> set[tuple]:
    """Teardown every tracked resource independently, removing only confirmed exact records."""
    # the strict snapshot raises on the FIRST record it cannot canonicalize, which strands every
    # other tracked resource behind it. teardown is per-resource, so read the records leniently
    # here. the strict reader stays in place for the write paths, where a malformed record must
    # not be silently dropped from the file.
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
        # a record that fails strict canonicalization still names a billable resource; the base
        # JobHandle validates only `provider`, and the runpod teardown resolves the deployed
        # 16-char fingerprint itself. skipping it here would leave that resource billing forever.
        identity = _remote_resource_identity(record) or _uncanonical_cleanup_remote_key(record)
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


def _cleanup_records_with(raw: dict, record: dict) -> list[dict] | None:
    """Append one strict record while preserving every existing cleanup sibling verbatim."""
    value = raw.get(state._CLEANUP_REMOTES_KEY, [])
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        return None
    records = list(value)
    key = _cleanup_remote_key(record)
    if key is None:
        return None
    if all(_teardown_removal_key(existing) != key for existing in records):
        records.append(record)
    return records


def _record_cleanup_remote(run_id: str, remote: dict) -> bool:
    """Persist one exact cleanup identity without changing the active remote."""
    record = _canonical_cleanup_remote(remote)
    if record is None:
        return False
    report_status: RunStatus | None = None
    with state._status_guard(run_id):
        raw = status_ops._load_status_json(run_id)
        status = status_ops._runstatus_from_json(raw)
        records = _cleanup_records_with(raw, record)
        if records is None:
            return False
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
