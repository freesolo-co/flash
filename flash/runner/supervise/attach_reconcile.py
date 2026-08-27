"""Shared evidence and teardown phases for attached-run reconciliation."""

from __future__ import annotations

import contextlib
import time

from flash.core.spec import JobSpec


def carry_allocation_stamp(metrics: dict, remote: dict | None) -> None:
    """carry the persisted allocation stamp onto adopted metrics."""
    if not isinstance(metrics, dict) or not isinstance(remote, dict):
        return
    allocated_gpu = remote.get("allocated_gpu")
    if allocated_gpu:
        metrics.setdefault("allocated_gpu", allocated_gpu)
    allocated_count = remote.get("allocated_gpu_count")
    if allocated_count:
        metrics.setdefault("allocated_gpu_count", int(allocated_count))
    provider = remote.get("provider")
    if provider:
        metrics.setdefault("allocated_provider", provider)


def completed_metrics_for_remote(
    worker_spec: JobSpec,
    persisted_remote: dict,
    *,
    provider: str,
    attempt: int,
    deadline_at: float,
    log,
) -> dict | None:
    from flash.runner.supervise.lifecycle import (
        _completed_attempt_metrics,
        _runpod_completed_metrics,
    )

    metrics = _runpod_completed_metrics(persisted_remote, deadline_at=deadline_at)
    started_ts = persisted_remote.get("started_ts")
    if metrics is None and started_ts is not None:
        metrics = _completed_attempt_metrics(
            worker_spec,
            provider=provider,
            attempt=attempt,
            launch_floor=float(started_ts),
            deadline_at=deadline_at,
            log=log,
        )
    if metrics is not None:
        carry_allocation_stamp(metrics, persisted_remote)
    return metrics


def _reconcile_completed_remote(
    run_id: str,
    worker_spec: JobSpec,
    expected_remote: dict,
    completed_metrics: dict,
    deadline_at: float,
    log,
) -> bool:
    """retry completed-attempt adoption until it succeeds or its grace expires."""
    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.lifecycle.protocol import AttemptRecord
    from flash.runner.lifecycle.status import get_status
    from flash.runner.supervise.attach import _ATTACH_RECONCILE_INTERVAL_S
    from flash.runner.supervise.lifecycle import _adopt_completed_attempt

    result_deadline_at = AttemptRecord.from_dict(get_status(run_id).attempt).result_deadline_at

    try:
        if _adopt_completed_attempt(
            run_id,
            worker_spec,
            expected_remote,
            completed_metrics,
            log=log,
        ):
            return True
    except Exception:
        pass
    # the job completed with metrics; keep retrying adoption (e.g. across a
    # transient cleanup failure) until it sticks -- but never past the wall
    # deadline plus the recovery grace, or a persistently failing adoption
    # would leave the run non-terminal forever. past the grace, preserve the
    # remote for cost reconciliation and fail the run.
    if time.time() >= result_deadline_at:
        # best-effort: preserve the remote for cost reconciliation, but do not
        # gate termination on it -- a persistently failing cleanup-persist must
        # not leave the run non-terminal forever (the whole point of the grace
        # cutoff). attempt the cleanup record, then fail the run regardless.
        with contextlib.suppress(Exception):
            _record_cleanup_remote(run_id, expected_remote)
        try:
            if _compare_and_fail_remote(
                run_id,
                expected_remote,
                "completed attempt could not be adopted within the recovery grace window",
            ):
                return True
        except Exception:
            pass
        # past the grace window the terminal cas is the only exit; if it raised or
        # lost the compare-and-swap, rate-limit the retry at the full reconcile
        # interval. remaining grace is <= 0 here, so falling through to the shared
        # sleep below would sleep 0 and busy-spin the reconciler.
        time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
        return False
    remaining_grace = result_deadline_at - time.time()
    time.sleep(min(_ATTACH_RECONCILE_INTERVAL_S, max(0.0, remaining_grace)))
    return False


def reconcile_terminal_result(
    run_id: str,
    worker_spec: JobSpec,
    expected_remote: dict,
    expected_attempt: tuple[int, int],
    handle,
    terminal_result,
    deadline_at: float,
    next_attempt: int,
    source_snapshot: dict | None,
    status,
    log,
) -> bool:
    """adopt, fail, or safely replace one verified attached terminal result."""
    from flash.runner.accounting.reconciliation import (
        _compare_and_fail_remote,
        _record_cleanup_remote,
    )
    from flash.runner.supervise.attach import (
        _ATTACH_RECONCILE_INTERVAL_S,
        _resume_after_confirmed_teardown,
    )
    from flash.runner.supervise.lifecycle import (
        _consume_recovered_retry_state,
        _strict_teardown_handle,
        _worker_provably_gone,
    )

    if terminal_result.ok:
        completed_metrics = terminal_result.metrics
        carry_allocation_stamp(completed_metrics, expected_remote)
        return _reconcile_completed_remote(
            run_id,
            worker_spec,
            expected_remote,
            completed_metrics,
            deadline_at,
            log,
        )
    terminal_failure = (
        f"{terminal_result.failure or 'job_failed'}: "
        f"{terminal_result.detail or 'worker attempt failed'}"
    )
    retry_policy = _consume_recovered_retry_state(
        worker_spec,
        status,
        terminal_result.failure,
        expected_attempt,
        expected_remote,
    )
    if retry_policy is None:
        try:
            _compare_and_fail_remote(
                run_id,
                expected_remote,
                terminal_failure,
                expected_attempt=expected_attempt,
            )
        except Exception:
            time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
            return False
        return True
    try:
        resource_deleted = _strict_teardown_handle(handle, run_id)
        worker_gone = True
    except Exception:
        resource_deleted = False
        worker_gone = _worker_provably_gone(run_id, handle)
    if not worker_gone:
        time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
        return False
    if handle.provider == "runpod" and not resource_deleted:
        try:
            cleanup_preserved = _record_cleanup_remote(run_id, expected_remote)
        except Exception:
            cleanup_preserved = False
        if not cleanup_preserved:
            time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
            return False
    try:
        _resume_after_confirmed_teardown(
            run_id,
            worker_spec,
            expected_remote,
            next_attempt,
            source_snapshot,
            log,
            failure=terminal_failure,
            expected_attempt=expected_attempt,
        )
    except Exception:
        time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
        return False
    return True


def retry_confirmed_remote_recovery(
    run_id: str,
    worker_spec: JobSpec,
    expected_remote: dict,
    source_snapshot: dict | None,
    log,
) -> bool:
    """retry recovery after teardown was durably confirmed."""
    from flash.runner.supervise.attach import (
        _ATTACH_RECONCILE_INTERVAL_S,
        _recover_confirmed_remote,
    )

    try:
        _recover_confirmed_remote(run_id, worker_spec, expected_remote, source_snapshot, log)
    except Exception:
        time.sleep(_ATTACH_RECONCILE_INTERVAL_S)
        return False
    return True


def teardown_reconciled_remote(
    run_id: str,
    expected_remote: dict,
    handle,
    *,
    confirmed_teardown: bool,
) -> tuple[bool, bool]:
    """return whether recovery may continue and exact deletion was confirmed."""
    from flash.runner.accounting.reconciliation import _record_cleanup_remote
    from flash.runner.supervise.lifecycle import _strict_teardown_handle, _worker_provably_gone

    if confirmed_teardown:
        return True, True
    try:
        resource_deleted = _strict_teardown_handle(handle, run_id)
        worker_gone = True
    except Exception:
        resource_deleted = False
        worker_gone = _worker_provably_gone(run_id, handle)
    if not worker_gone:
        return False, resource_deleted
    if resource_deleted:
        return True, True
    if handle.provider != "runpod":
        return True, False
    try:
        cleanup_preserved = _record_cleanup_remote(run_id, expected_remote)
    except Exception:
        cleanup_preserved = False
    return cleanup_preserved, resource_deleted
