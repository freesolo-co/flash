"""complete and recover exact managed checkpoint deployments."""

from __future__ import annotations

import time
from threading import Event

import flash.runner.lifecycle.status as run_status
import flash.runner.results.checkpoints as run_checkpoints
import flash.runner.supervise.transitions as runner_transitions
import flash.serve.deployment.deploy as serving_deploy
import flash.server.platform.locks as deployment_locks
from flash.core.spec import JobSpec
from flash.runner.results.deployments import owns_attempt
from flash.serve.contract.errors import AdapterConfigMissing, ServingError
from flash.serve.deployment.deploy import DeploymentReadinessFacts
from flash.server.platform import db, deployment_state
from flash.server.routes import serving_smoke
from flash.server.routes.serving_revisions import (
    _DEPLOYABLE_STATES,
    _DEPLOYMENT_BUSY_STATES,
    _DEPLOYMENT_READY_STATES,
    _spec_is_unservable,
)


def recover_deployments() -> int:
    recovered = 0
    for row in db.all_runs():
        try:
            status = run_status.get_status(row["run_id"])
        except FileNotFoundError:
            continue
        lock = deployment_locks._deploy_lock(row["run_id"])
        if not lock.acquire(blocking=False):
            continue
        try:
            status = run_status.get_status(row["run_id"])
            for checkpoint_id, deployment in status.deployments.items():
                state = deployment.get("state")
                if state in _DEPLOYMENT_BUSY_STATES:
                    failed = deployment_state._deployment_state(
                        deployment,
                        "failed",
                        error="deployment lifecycle interrupted by control-plane restart",
                        detail="deployment interrupted; redeploy the exact checkpoint",
                        recovered_at=time.time(),
                    )
                    marked = runner_transitions.mark_deployment_failed(status.run_id, failed)
                elif state in _DEPLOYMENT_READY_STATES and _spec_is_unservable(status):
                    marked = runner_transitions.mark_deployment_revocation_failed(
                        status.run_id,
                        "deployment spec is no longer supported",
                        checkpoint_id,
                    )
                else:
                    continue
                deployment_state._report_persisted_transition(status, marked, persisted=True)
                recovered += 1
        finally:
            lock.release()
    return recovered


def replay_status_reports(stop: Event | None = None) -> int:
    from flash.runner.lifecycle.reporting import _report_status

    replayed = 0
    for row in db.all_runs():
        if stop is not None and stop.is_set():
            break
        try:
            status = run_status.get_status(row["run_id"])
            _report_status(status)
        except (OSError, TypeError, ValueError):
            continue
        replayed += 1
    return replayed


def _assert_deployment_fence(
    run_id: str, deployment: dict, is_checkpoint: bool, prev_state: str
) -> None:
    latest = run_status.get_status(run_id)
    latest_deployment = latest.deployments.get(deployment["checkpoint_id"], {})
    if not owns_attempt(latest_deployment, deployment):
        raise ServingError("deployment attempt was superseded before checkpoint readiness")
    if is_checkpoint:
        if prev_state in _DEPLOYABLE_STATES and latest.state != prev_state:
            raise ServingError(f"run state changed from {prev_state!r} to {latest.state!r}")
    elif latest.state != prev_state:
        raise ServingError(f"run state changed from {prev_state!r} to {latest.state!r}")


def _commit_ready_deployment(
    run_id: str,
    current: dict,
    is_checkpoint: bool,
    prev_state: str,
) -> bool:
    previous = run_status.get_status(run_id)
    if is_checkpoint:
        state_guard = prev_state if prev_state in _DEPLOYABLE_STATES else None
        marked = runner_transitions.mark_checkpoint_deployed(
            run_id,
            current,
            expect_state=state_guard,
        )
        persisted = marked.deployments.get(current["checkpoint_id"]) == current
    else:
        marked = runner_transitions.mark_deployed(
            run_id,
            current,
            expect_state=prev_state,
        )
        persisted = (
            marked.state == "deployed"
            and marked.deployments.get(current["checkpoint_id"]) == current
        )
    if persisted:
        deployment_state._report_persisted_transition(previous, marked, persisted=True)
    return persisted


def _finish_deployment_unlocked(
    *,
    run_id: str,
    spec_dict: dict,
    is_checkpoint: bool,
    deploy_kwargs: dict,
    deployment: dict,
    prev_state: str,
) -> None:
    spec = JobSpec.from_dict(spec_dict)
    active = run_status.get_status(run_id).deployments.get(deployment["checkpoint_id"], {})
    if not owns_attempt(active, deployment):
        return
    current = dict(deployment)
    smoke_result: dict = {}
    smoke_target: str | None = None

    def _before_ready(facts: DeploymentReadinessFacts) -> None:
        nonlocal current, smoke_target
        smoke_target = facts.checkpoint_id
        _assert_deployment_fence(run_id, deployment, is_checkpoint, prev_state)
        current = deployment_state._deployment_state(
            {**current, "checkpoint_id": facts.checkpoint_id, "openai_model": facts.checkpoint_id},
            "smoke_testing",
            detail="running bounded fixed-prompt smoke",
        )
        previous = run_status.get_status(run_id)
        marked = runner_transitions.mark_deployment_pending(
            run_id, current, owner_deployment=deployment
        )
        deployment_state._report_persisted_transition(
            previous, marked, persisted=marked.deployments.get(current["checkpoint_id"]) == current
        )
        smoke_result.update(
            serving_smoke._run_deployment_smoke(
                run_id,
                spec,
                serving_model=facts.checkpoint_id,
                expected_checkpoint=facts.expected_checkpoint,
                org_id=str(deploy_kwargs["org_id"]),
                advertised_capabilities=facts.advertised_capabilities,
                adapter_targets_images=facts.adapter_targets_images,
            )
        )
        _assert_deployment_fence(run_id, deployment, is_checkpoint, prev_state)

    try:
        dep = serving_deploy.deploy_adapter(**deploy_kwargs, before_ready=_before_ready)
    except Exception as exc:
        if smoke_target is not None and not _revoke_failed_smoke(
            run_id, smoke_target, str(deploy_kwargs["org_id"]), exc
        ):
            return
        _record_deployment_failure(run_id, spec, exc, current, deployment, is_checkpoint)
        return

    if not smoke_result:
        error = ServingError("deployment backend returned without checkpoint verification")
        if _revoke_failed_smoke(
            run_id, deployment["checkpoint_id"], str(deploy_kwargs["org_id"]), error
        ):
            _record_deployment_failure(run_id, spec, error, current, deployment, is_checkpoint)
        return

    current = {**current, **dep.to_dict(), "verify": True}
    current = deployment_state._deployment_state(
        current,
        "ready",
        detail="permanent checkpoint verified and ready",
        **smoke_result,
    )
    current = deployment_state._public_deployment(current)
    if not _commit_ready_deployment(run_id, current, is_checkpoint, prev_state):
        print(
            f"deploy[{run_id}]: checkpoint ready but deployment record changed concurrently",
            flush=True,
        )


def _revoke_failed_smoke(
    run_id: str, checkpoint_id: str, org_id: str, smoke_error: Exception
) -> bool:
    """disable an unverified hosted checkpoint, retaining a retryable state on cleanup failure."""
    try:
        serving_deploy.undeploy_adapter(checkpoint_id, org_id=org_id)
    except Exception as exc:
        error = f"{smoke_error}; checkpoint cleanup failed: {exc}"
        previous = run_status.get_status(run_id)
        marked = runner_transitions.mark_deployment_revocation_failed(
            run_id, error, checkpoint_id=checkpoint_id
        )
        deployment_state._report_persisted_transition(previous, marked, persisted=True)
        return False
    return True


def _record_deployment_failure(
    run_id: str,
    spec: JobSpec,
    exc: Exception,
    current: dict,
    deployment: dict,
    is_checkpoint: bool,
) -> None:
    error = str(exc)
    if not is_checkpoint and isinstance(exc, AdapterConfigMissing):
        steps = [candidate["step"] for candidate in run_checkpoints.list_checkpoints(spec)]
        if steps:
            error = (
                f"run {run_id} has no final adapter at {deployment.get('adapter_hf_prefix')}; "
                f"deploy an explicit saved checkpoint such as {run_id}/step-{steps[-1]}"
            )
    failed = deployment_state._deployment_state(
        current,
        "failed",
        error=error,
        detail="checkpoint deployment failed; sibling checkpoints are unchanged",
    )
    previous = run_status.get_status(run_id)
    marked = runner_transitions.mark_deployment_failed(run_id, failed)
    deployment_state._report_persisted_transition(
        previous, marked, persisted=deployment_state._deployment_failure_persisted(marked, failed)
    )
