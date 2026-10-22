"""Atomic, exact-checkpoint deployment state transitions."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from flash.runner.results.deployments import (
    generation,
    is_ready,
    owns_attempt,
    require_checkpoint,
    validate_deployments,
)

if TYPE_CHECKING:
    from flash.runner.lifecycle.state import RunStatus


def _save_record(status: RunStatus, record: dict) -> RunStatus:
    from flash.runner.lifecycle.state import _save_status_unlocked

    validate_deployments(status.run_id, {record["checkpoint_id"]: record})
    status.deployments[record["checkpoint_id"]] = record
    if (
        record["state"] in {"failed", "undeploying", "undeployed"}
        and status.state == "deployed"
        and not any(is_ready(status.run_id, d) for d in status.deployments.values())
    ):
        status.state = "done"
    status.updated_at = time.time()
    _save_status_unlocked(status)
    return status


def mark_deployment_pending(
    run_id: str,
    deployment: dict,
    expect_state: str | None = None,
    *,
    owner_deployment: dict | None = None,
) -> RunStatus:
    from flash.runner.lifecycle.state import _status_guard
    from flash.runner.lifecycle.status import get_status

    require_checkpoint(run_id, deployment.get("checkpoint_id"))
    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state == "dry_run" or (expect_state is not None and status.state != expect_state):
            return status
        current = status.deployments.get(deployment["checkpoint_id"], {})
        if owner_deployment is not None:
            if not owns_attempt(current, owner_deployment):
                return status
        elif is_ready(run_id, current) or generation(deployment) != generation(current) + 1:
            return status
        return _save_record(status, deployment)


def mark_checkpoint_deployed(
    run_id: str,
    deployment: dict,
    expect_state: str | None = None,
) -> RunStatus:
    from flash.runner.lifecycle.state import _status_guard
    from flash.runner.lifecycle.status import get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state == "dry_run" or (expect_state is not None and status.state != expect_state):
            return status
        if not is_ready(run_id, deployment):
            raise ValueError("checkpoint readiness requires its successful verification result")
        current = status.deployments.get(deployment["checkpoint_id"], {})
        if not owns_attempt(current, deployment):
            return status
        if status.state in {"done", "deployed"}:
            status.state = "deployed"
        return _save_record(status, deployment)


def mark_deployed(run_id: str, deployment: dict, expect_state: str | None = None) -> RunStatus:
    from flash.runner.lifecycle.status import get_status

    status = get_status(run_id)
    if status.state not in {"done", "deployed"}:
        return status
    return mark_checkpoint_deployed(run_id, deployment, expect_state or status.state)


def mark_deployment_failed(run_id: str, deployment: dict) -> RunStatus:
    from flash.runner.lifecycle.state import _status_guard
    from flash.runner.lifecycle.status import get_status

    require_checkpoint(run_id, deployment.get("checkpoint_id"))
    with _status_guard(run_id):
        status = get_status(run_id)
        current = status.deployments.get(deployment["checkpoint_id"], {})
        if not owns_attempt(current, deployment):
            return status
        return _save_record(status, {**deployment, "state": "failed"})


def _mark_checkpoint_inactive(run_id: str, checkpoint_id: str, **fields) -> RunStatus:
    from flash.runner.lifecycle.state import _status_guard
    from flash.runner.lifecycle.status import get_status

    step = require_checkpoint(run_id, checkpoint_id)
    with _status_guard(run_id):
        status = get_status(run_id)
        current = dict(status.deployments.get(checkpoint_id, {}))
        for field in ("error", "retryable", "deployment_error", "cleanup_error"):
            current.pop(field, None)
        record = {
            **current,
            **fields,
            "run_id": run_id,
            "checkpoint_id": checkpoint_id,
            "checkpoint_step": step,
            "generation": generation(current) + 1,
            "updated_at": time.time(),
        }
        return _save_record(status, record)


def mark_deployment_undeploying(
    run_id: str, error: str, checkpoint_id: str, *, cleanup_error: str | None = None
) -> RunStatus:
    """Disable checkpoint serving while backend removal is pending or needs a retry."""
    return _mark_checkpoint_inactive(
        run_id,
        checkpoint_id,
        state="undeploying",
        deployment_error=error,
        cleanup_error=cleanup_error,
        error=f"{error}; checkpoint cleanup failed: {cleanup_error}" if cleanup_error else error,
        retryable=True,
    )


def mark_undeployed(run_id: str, checkpoint_id: str) -> RunStatus:
    """Keep an inactive generation fence for this checkpoint; leave every sibling intact."""
    return _mark_checkpoint_inactive(run_id, checkpoint_id, state="undeployed")
