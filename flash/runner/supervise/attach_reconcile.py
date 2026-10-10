"""Shared evidence and teardown phases for attached-run reconciliation."""

from __future__ import annotations

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
