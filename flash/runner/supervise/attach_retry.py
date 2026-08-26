"""Retry verified terminal results found by attached-run reconciliation."""

from __future__ import annotations

import time

from flash.core.spec import JobSpec


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
        _carry_allocation_stamp,
        _reconcile_completed_remote,
        _resume_after_confirmed_teardown,
    )
    from flash.runner.supervise.lifecycle import (
        _consume_recovered_retry_state,
        _strict_teardown_handle,
        _worker_provably_gone,
    )

    if terminal_result.ok:
        completed_metrics = terminal_result.metrics
        _carry_allocation_stamp(completed_metrics, expected_remote)
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
    return True
