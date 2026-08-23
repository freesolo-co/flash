"""Deployment state transitions recorded on a run's status.

The ``mark_*`` functions are the write half of the deployment lifecycle: each takes a verified
outcome and persists it, leaving the decision of *whether* that outcome is correct to
``deploy.py``. They share no helpers with the cancellation path, which is why they move cleanly.

Split out of ``flash.runner.supervise.deploy`` to keep that module under the file-size limit.
"""

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
from flash.schema import parse_adapter_revision

if TYPE_CHECKING:
    from flash.runner import RunStatus


def _deployment_attempt_is_owned(status: RunStatus, deployment: dict) -> bool:
    requested_at = deployment.get("requested_at")
    if requested_at is None:
        return True
    current = status.deployment or {}
    if current == deployment:
        return True
    return (
        current.get("requested_at") == requested_at
        and current.get("state") in _DEPLOYMENT_BUSY_STATES
    )


def _commit_verified_deployment(
    run_id: str,
    deployment: dict,
    *,
    verification_generation: int | None,
    commit: Callable[[], None],
    retain_only_revision: bool = False,
    advance_generation: bool = False,
) -> bool:
    if deployment.get("state") not in _RESTORABLE_DEPLOYMENT_STATES:
        raise ValueError("immutable deployment commit requires ready or deployed state")
    revision = deployment.get("adapter_revision")
    parsed_revision = parse_adapter_revision(revision) if isinstance(revision, str) else None
    if parsed_revision is None or parsed_revision[0] != run_id:
        raise ValueError(
            f"immutable deployment commit requires a full same-run adapter revision for {run_id}"
        )
    if verification_generation is None:
        raise ValueError("immutable deployment commit requires a verification generation")
    from flash.runner.results.verified_revisions import commit_verified_adapter_revision

    return commit_verified_adapter_revision(
        run_id,
        revision,
        expected_generation=verification_generation,
        commit=commit,
        retain_only_revision=retain_only_revision,
        advance_generation=advance_generation,
    )


def _promote_final_deployment(status: RunStatus, deployment: dict) -> None:
    """Apply the lifecycle state for a final-adapter deployment."""
    status.deployment = deployment
    status.state = "deployed"


def mark_deployed(
    run_id: str,
    deployment: dict,
    expect_state: str | None = None,
    *,
    verification_generation: int | None = None,
) -> RunStatus:
    from flash.runner import _UNDEPLOYABLE_STATES, _save_status_unlocked, _status_guard, get_status

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
    retain_only_revision: bool = False,
    advance_generation: bool = False,
) -> RunStatus:
    """Record a checkpoint deployment using the run's current lifecycle state.

    If training has finished by the time serving registration completes, the run behaves like any
    finished deployed run. Otherwise, keep the training state and only attach the deployment record.
    """
    from flash.runner import _save_status_unlocked, _status_guard, get_status

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
            retain_only_revision=retain_only_revision,
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
    """Attach an in-progress deployment record without changing the run lifecycle state."""
    from flash.runner import _save_status_unlocked, _status_guard, get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        if status.state == "dry_run":
            return status
        if expect_state is not None and status.state != expect_state:
            return status
        ownership_token = deployment if owner_deployment is None else owner_deployment
        expected_generation = ownership_token.get("verification_generation")
        if expected_generation is not None:
            from flash.runner.results.verified_revisions import verified_adapter_revision_generation

            if verified_adapter_revision_generation(run_id) != expected_generation:
                return status
        current = status.deployment if isinstance(status.deployment, dict) else {}
        same_attempt = current.get("requested_at") == deployment.get("requested_at")
        if same_attempt and current.get("state") in {"undeployed", _REVOCATION_RETRY_STATE}:
            return status
        if owner_deployment is not None and not _deployment_attempt_is_owned(
            status, owner_deployment
        ):
            return status
        status.deployment = deployment
        status.updated_at = time.time()
        _save_status_unlocked(status)
        return status


def _confirmed_active_failed_predecessor(deployment: object, run_id: str) -> bool:
    if not isinstance(deployment, dict):
        return False
    revision = deployment.get("adapter_revision")
    parsed = parse_adapter_revision(revision) if isinstance(revision, str) else None
    return (
        deployment.get("state") == "failed"
        and deployment.get("alias_activation_confirmed") is True
        and parsed is not None
        and parsed[0] == run_id
    )


def _restorable_deployment_predecessor(deployment: object, run_id: str) -> bool:
    return (
        isinstance(deployment, dict) and deployment.get("state") in _RESTORABLE_DEPLOYMENT_STATES
    ) or _confirmed_active_failed_predecessor(deployment, run_id)


def mark_deployment_failed(run_id: str, deployment: dict) -> RunStatus:
    """Record a failed deployment attempt while preserving the run lifecycle state."""
    from flash.runner import _save_status_unlocked, _status_guard, get_status

    with _status_guard(run_id):
        status = get_status(run_id)
        current = status.deployment or {}
        # don't clobber a newer deployment attempt, explicit undeploy, or pending revocation.
        if current.get("state") in {"undeployed", _REVOCATION_RETRY_STATE}:
            return status
        if (
            current.get("requested_at") is not None
            and deployment.get("requested_at") is not None
            and current.get("requested_at") != deployment.get("requested_at")
        ):
            return status
        previous = deployment.get("previous_deployment")
        if _restorable_deployment_predecessor(previous, run_id) and (
            not deployment.get("activation_outcome_unknown")
            or (
                deployment.get("state") == "failed"
                and _confirmed_active_failed_predecessor(previous, run_id)
            )
        ):
            status.deployment = {
                **previous,
                "last_deploy_error": deployment.get("error") or "deployment failed",
                "last_deploy_failed_at": time.time(),
            }
        else:
            failed = dict(deployment)
            if not failed.get("activation_outcome_unknown"):
                failed.pop("previous_deployment", None)
            state = (
                "reconciling"
                if failed.get("activation_outcome_unknown") and failed.get("state") == "reconciling"
                else "failed"
            )
            status.deployment = {**failed, "state": state}
        status.updated_at = time.time()
        _save_status_unlocked(status)
        return status


def mark_deployment_revocation_failed(run_id: str, error: str) -> RunStatus:
    """Revoke local serving authority while retaining retryable backend cleanup state."""
    from flash.runner import _save_status_unlocked, _status_guard, get_status
    from flash.runner.results.verified_revisions import invalidate_verified_adapter_revisions

    with _status_guard(run_id):
        status = get_status(run_id)

        def _commit() -> None:
            deployment = status.deployment if isinstance(status.deployment, dict) else {}
            status.deployment = {
                **deployment,
                "state": "revocation_failed",
                "error": error,
                "retryable": True,
                "updated_at": time.time(),
            }
            status.updated_at = time.time()
            _save_status_unlocked(status)

        invalidate_verified_adapter_revisions(run_id, commit=_commit)
        return status


def mark_undeployed(run_id: str) -> RunStatus:
    """Record an explicit undeploy; live final-adapter deployments return to `done`."""
    from flash.runner import _save_status_unlocked, _status_guard, get_status
    from flash.runner.results.verified_revisions import invalidate_verified_adapter_revisions

    with _status_guard(run_id):
        status = get_status(run_id)

        def _commit() -> None:
            if status.deployment:
                deployment = dict(status.deployment)
                for field in ("error", "retryable", "updated_at"):
                    deployment.pop(field, None)
                status.deployment = {**deployment, "state": "undeployed"}
            if status.state == "deployed":
                status.state = "done"
            status.updated_at = time.time()
            _save_status_unlocked(status)

        invalidate_verified_adapter_revisions(run_id, commit=_commit)
        return status


def mark_deployment_undeployed(run_id: str) -> RunStatus:
    """Flip ONLY the deployment field to ``undeployed``, leaving the run's state untouched.

    Used by cancel_run, unlike mark_undeployed, never asserts or changes the run state,
    so it works even after a racing mark_undeployed has already written terminal `done`.
    """
    from flash.runner import _save_status_unlocked, _status_guard, get_status
    from flash.runner.results.verified_revisions import invalidate_verified_adapter_revisions

    with _status_guard(run_id):
        status = get_status(run_id)

        def _commit() -> None:
            if status.deployment is not None:
                deployment = status.deployment if isinstance(status.deployment, dict) else {}
                deployment = dict(deployment)
                for field in ("error", "retryable", "updated_at"):
                    deployment.pop(field, None)
                status.deployment = {**deployment, "state": "undeployed"}
                status.updated_at = time.time()
                _save_status_unlocked(status)

        invalidate_verified_adapter_revisions(run_id, commit=_commit)
        return status
