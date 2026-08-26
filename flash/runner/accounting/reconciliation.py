"""Remote resource reconciliation and cleanup tracking."""

from __future__ import annotations

import contextlib
import time

from flash.core.spec import JobSpec
from flash.runner.accounting import costs
from flash.runner.lifecycle import reporting, state
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


def _remote_attempt_identity(remote: object) -> tuple[int, int] | None:
    if not isinstance(remote, dict) or _remote_resource_identity(remote) is None:
        return None
    return remote["attempt"], remote["fence"]


def _expected_remote_matches(current: object, expected: dict | None) -> bool:
    if expected is None:
        return current is None
    expected_identity = _remote_resource_identity(expected)
    return expected_identity is not None and _remote_resource_identity(current) == expected_identity


def _expected_terminal_remote_matches(current: object, expected: dict | None) -> bool:
    return _expected_remote_matches(current, expected) and (
        expected is None or _remote_attempt_identity(current) == _remote_attempt_identity(expected)
    )


def _attempt_identity_matches(
    status: RunStatus,
    expected_attempt: tuple[int, int] | None,
    *,
    expected_no_attempt: bool,
) -> bool:
    if expected_no_attempt:
        if expected_attempt is not None:
            raise ValueError("expected attempt identity and absence are mutually exclusive")
        return status.attempt is None
    if expected_attempt is None:
        return True
    if not status.attempt:
        return False
    try:
        attempt = status_ops._current_attempt(status)
    except (TypeError, ValueError):
        return False
    return (attempt.attempt_id, attempt.fence) == expected_attempt


def _settle_terminal_attempt(status: RunStatus, expected_remote: dict | None) -> bool:
    if expected_remote is None:
        status_ops.settle_current_attempt(status)
        return True
    expected_attempt = _remote_attempt_identity(expected_remote)
    if expected_attempt is None or not status.attempt:
        return False
    return status_ops.transition_attempt_state(
        status,
        "settled",
        expected_attempt_id=expected_attempt[0],
        expected_fence=expected_attempt[1],
    )


def _settled_attempt_identity(status: RunStatus) -> tuple[int, int] | None:
    if status.attempt is None:
        return None
    attempt = status_ops._current_attempt(status)
    if attempt.state != "settled":
        raise RuntimeError("terminal attempt was not settled")
    return attempt.attempt_id, attempt.fence


def _confirmed_settled_attempt_matches(
    confirmed: RunStatus,
    expected_attempt: tuple[int, int] | None,
) -> bool:
    if expected_attempt is None:
        return confirmed.attempt is None
    try:
        attempt = status_ops._current_attempt(confirmed)
    except (TypeError, ValueError):
        return False
    return (
        attempt.attempt_id == expected_attempt[0]
        and attempt.fence == expected_attempt[1]
        and attempt.state == "settled"
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
        status.remote = None
        status.updated_at = time.time()
        state._save_status_unlocked(status)
        report_status = status
    if report_status is not None:
        reporting._report_status(report_status)
    return True


def _compare_and_prepare_resubmit(
    run_id: str,
    expected_remote: dict | None,
    *,
    expected_state: str | None = None,
    expected_attempt: tuple[int, int] | None = None,
    expected_no_attempt: bool = False,
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
        if not _attempt_identity_matches(
            status,
            expected_attempt,
            expected_no_attempt=expected_no_attempt,
        ):
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
    *,
    expected_attempt: tuple[int, int] | None = None,
    expected_no_attempt: bool = False,
) -> bool:
    """CAS a nonterminal expected remote to failed and confirm the durable write."""
    report_status: RunStatus | None = None
    settled_attempt: tuple[int, int] | None = None
    with state._status_guard(run_id):
        status = status_ops.get_status(run_id)
        if status.state in state.TERMINAL_STATES:
            return False
        if not _expected_terminal_remote_matches(status.remote, expected_remote):
            return False
        if not _attempt_identity_matches(
            status,
            expected_attempt,
            expected_no_attempt=expected_no_attempt,
        ):
            return False
        if not _settle_terminal_attempt(status, expected_remote):
            return False
        settled_attempt = _settled_attempt_identity(status)
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
        or not _expected_terminal_remote_matches(confirmed.remote, expected_after)
        or confirmed.error != error
        or not _confirmed_settled_attempt_matches(confirmed, settled_attempt)
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
    *,
    expected_attempt: tuple[int, int] | None = None,
    expected_no_attempt: bool = False,
) -> bool:
    """Adopt strict completed artifacts only while the captured remote still owns the run."""
    report_status: RunStatus | None = None
    settled_attempt: tuple[int, int] | None = None
    with state._status_guard(run_id):
        status = status_ops.get_status(run_id)
        if status.state in state.TERMINAL_STATES:
            return False
        if not _expected_terminal_remote_matches(status.remote, expected_remote):
            return False
        if not _attempt_identity_matches(
            status,
            expected_attempt,
            expected_no_attempt=expected_no_attempt,
        ):
            return False
    expected_attempt_id = (
        expected_remote.get("attempt") if isinstance(expected_remote, dict) else None
    )
    expected_fence = expected_remote.get("fence") if isinstance(expected_remote, dict) else None
    if expected_attempt is not None:
        expected_attempt_id, expected_fence = expected_attempt
    metrics, verified_attempt = status_ops.validate_terminal_source_metrics(
        status,
        metrics,
        expected_attempt=expected_attempt_id,
        expected_fence=expected_fence,
    )
    if expected_remote is not None and not _record_cleanup_remote(run_id, expected_remote):
        return False
    recovered_cost = status_ops._persist_metrics(spec, metrics)
    with state._status_guard(run_id):
        status = status_ops.get_status(run_id)
        if status.state in state.TERMINAL_STATES:
            return False
        if not _expected_terminal_remote_matches(status.remote, expected_remote):
            return False
        if not _attempt_identity_matches(
            status,
            expected_attempt,
            expected_no_attempt=expected_no_attempt,
        ):
            return False
        if not _settle_terminal_attempt(status, expected_remote):
            return False
        settled_attempt = _settled_attempt_identity(status)
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
    if (
        confirmed.state != "done"
        or not _expected_terminal_remote_matches(confirmed.remote, expected_remote)
        or not _confirmed_settled_attempt_matches(confirmed, settled_attempt)
    ):
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
            state._save_status_unlocked(
                status_ops._runstatus_from_json(raw),
                _cleanup_remotes=remaining or None,
            )
            return True
        remaining = [record for record in records if _teardown_removal_key(record) != expected_key]
        if len(remaining) == len(records):
            return False
        state._save_status_unlocked(
            status_ops._runstatus_from_json(raw),
            _cleanup_remotes=remaining or None,
        )
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
