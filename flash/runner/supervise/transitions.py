"""managed checkpoint deployment state transitions."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from flash.runner.supervise.deploy import (
    _DEPLOYMENT_BUSY_STATES,
    _FINAL_DEPLOYMENT_STATES,
    _RESTORABLE_DEPLOYMENT_STATES,
    _REVOCATION_RETRY_STATE,
)
from flash.schema import parse_checkpoint_ref

if TYPE_CHECKING:
    from flash.runner.lifecycle.state import RunStatus


def _deployment_attempt_is_owned(status: RunStatus, deployment: dict) -> bool:
    requested_at = deployment.get("requested_at")
    if requested_at is None:
        return True
    current = status.deployment or {}
    return current == deployment or (
        current.get("requested_at") == requested_at
        and current.get("state") in _DEPLOYMENT_BUSY_STATES
    )


def _commit_verified_deployment(
    run_id: str,
    deployment: dict,
    *,
    verification_generation: int | None,
    commit: Callable[[], None],
    advance_generation: bool = False,
) -> bool:
    if deployment.get("state") not in _RESTORABLE_DEPLOYMENT_STATES:
        raise ValueError("checkpoint deployment commit requires ready state")
    checkpoint_id = deployment.get("checkpoint_id")
    parsed = parse_checkpoint_ref(checkpoint_id) if isinstance(checkpoint_id, str) else None
    if parsed is None or parsed[0] != run_id:
        raise ValueError(
            f"checkpoint deployment commit requires a same-run permanent checkpoint for {run_id}"
        )
    if verification_generation is None:
        raise ValueError("checkpoint deployment commit requires a verification generation")
    from flash.runner.results.verified_revisions import commit_verified_checkpoint

    return commit_verified_checkpoint(
        run_id,
        checkpoint_id,
        expected_generation=verification_generation,
        commit=commit,
        advance_generation=advance_generation,
    )


def _promote_final_deployment(status: RunStatus, deployment: dict) -> None:
    status.deployment = deployment
    status.state = "deployed"


def mark_deployed(
    run_id: str,
    deployment: dict,
    expect_state: str | None = None,
    *,
    verification_generation: int | None = None,
) -> RunStatus:
    from flash.runner.lifecycle.state import (
        _UNDEPLOYABLE_STATES,
        _save_status_unlocked,
        _status_guard,
    )
    from flash.runner.lifecycle.status import get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state in _UNDEPLOYABLE_STATES:
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        if not _deployment_attempt_is_owned(status, deployment):
            return status

        def _commit() -> None:
            _promote_final_deployment(status, deployment)
            status.updated_at = time.time()
            _save_status_unlocked(status)

        if not _commit_verified_deployment(
            run_id,
            deployment,
            verification_generation=verification_generation,
            commit=_commit,
        ):
            return get_status(run_id)
        return status


def mark_checkpoint_deployed(
    run_id: str,
    deployment: dict,
    expect_state: str | None = None,
    *,
    verification_generation: int | None = None,
    owner_deployment: dict | None = None,
    advance_generation: bool = False,
) -> RunStatus:
    from flash.runner.lifecycle.state import _save_status_unlocked, _status_guard
    from flash.runner.lifecycle.status import get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state == "dry_run":
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        ownership_token = deployment if owner_deployment is None else owner_deployment
        if not _deployment_attempt_is_owned(status, ownership_token):
            return status

        def _commit() -> None:
            if status.state in _FINAL_DEPLOYMENT_STATES:
                _promote_final_deployment(status, deployment)
            else:
                status.deployment = deployment
            status.updated_at = time.time()
            _save_status_unlocked(status)

        if not _commit_verified_deployment(
            run_id,
            deployment,
            verification_generation=verification_generation,
            commit=_commit,
            advance_generation=advance_generation,
        ):
            return get_status(run_id)
        return status


def mark_deployment_pending(
    run_id: str,
    deployment: dict,
    expect_state: str | None = None,
    *,
    owner_deployment: dict | None = None,
) -> RunStatus:
    from flash.runner.lifecycle.state import _save_status_unlocked, _status_guard
    from flash.runner.lifecycle.status import get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state == "dry_run":
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        ownership_token = deployment if owner_deployment is None else owner_deployment
        expected_generation = ownership_token.get("verification_generation")
        if expected_generation is not None:
            from flash.runner.results.verified_revisions import verified_checkpoint_generation

            if verified_checkpoint_generation(run_id) != expected_generation:
                return status
        current = status.deployment if isinstance(status.deployment, dict) else {}
        if current.get("requested_at") == deployment.get("requested_at") and current.get(
            "state"
        ) in {"undeployed", _REVOCATION_RETRY_STATE}:
            return status
        if owner_deployment is not None and not _deployment_attempt_is_owned(
            status, owner_deployment
        ):
            return status
        status.deployment = deployment
        status.updated_at = time.time()
        _save_status_unlocked(status)
        return status


def mark_deployment_failed(run_id: str, deployment: dict) -> RunStatus:
    from flash.runner.lifecycle.state import _save_status_unlocked, _status_guard
    from flash.runner.lifecycle.status import get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        current = status.deployment or {}
        if current.get("state") in {"undeployed", _REVOCATION_RETRY_STATE}:
            return status
        if (
            current.get("requested_at") is not None
            and deployment.get("requested_at") is not None
            and current.get("requested_at") != deployment.get("requested_at")
        ):
            return status
        failed = dict(deployment)
        failed.pop("previous_deployment", None)
        status.deployment = {**failed, "state": "failed"}
        status.updated_at = time.time()
        _save_status_unlocked(status)
        return status


def mark_deployment_revocation_failed(
    run_id: str, error: str, checkpoint_id: str | None = None
) -> RunStatus:
    from flash.runner.lifecycle.state import _save_status_unlocked, _status_guard
    from flash.runner.lifecycle.status import get_status
    from flash.runner.results.verified_revisions import remove_verified_checkpoint

    with _status_guard(run_id):
        status = get_status(run_id)
        deployment = status.deployment if isinstance(status.deployment, dict) else {}
        target = checkpoint_id or deployment.get("checkpoint_id")
        parsed = parse_checkpoint_ref(target) if isinstance(target, str) else None

        def _commit(_retained: frozenset[str] = frozenset()) -> None:
            now = time.time()
            if deployment.get("checkpoint_id") == target:
                status.deployment = {
                    **deployment,
                    "state": "revocation_failed",
                    "error": error,
                    "retryable": True,
                    "updated_at": now,
                }
            status.updated_at = now
            _save_status_unlocked(status)

        if parsed is not None and parsed[0] == run_id:
            remove_verified_checkpoint(run_id, target, commit=_commit)
        else:
            _commit()
        return status


def mark_undeployed(run_id: str, checkpoint_id: str | None = None) -> RunStatus:
    """record exact undeploy while preserving sibling checkpoint verification."""

    from flash.runner.lifecycle.state import _save_status_unlocked, _status_guard
    from flash.runner.lifecycle.status import get_status
    from flash.runner.results.verified_revisions import remove_verified_checkpoint

    with _status_guard(run_id):
        status = get_status(run_id)
        target = checkpoint_id or (
            status.deployment.get("checkpoint_id") if isinstance(status.deployment, dict) else None
        )
        if not isinstance(target, str):
            raise ValueError("exact undeploy requires checkpoint_id")

        def _commit(retained: frozenset[str]) -> None:
            removes_summary = (
                isinstance(status.deployment, dict)
                and status.deployment.get("checkpoint_id") == target
            )
            if removes_summary:
                deployment = dict(status.deployment)
                for field in ("error", "retryable", "updated_at"):
                    deployment.pop(field, None)
                if retained:
                    checkpoint_id = min(retained)
                    status.deployment = {
                        **deployment,
                        "state": "ready",
                        "checkpoint_id": checkpoint_id,
                    }
                    if "openai_model" in deployment:
                        status.deployment["openai_model"] = checkpoint_id
                else:
                    status.deployment = {**deployment, "state": "undeployed"}
                    if status.state == "deployed":
                        status.state = "done"
            status.updated_at = time.time()
            _save_status_unlocked(status)

        remove_verified_checkpoint(run_id, target, commit=_commit)
        return status
