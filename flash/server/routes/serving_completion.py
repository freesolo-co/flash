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
from flash.serve.contract.errors import AdapterConfigMissing, RegistrationRejected, ServingError
from flash.serve.contract.responses import matches_registration_identity
from flash.serve.deployment.deploy import DeploymentReadinessFacts
from flash.server.platform import db, deployment_state
from flash.server.platform.internal_client import run_serving_org_id
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
        lock = deployment_locks._deploy_lock(row["run_id"])
        if not lock.acquire(blocking=False):
            continue
        try:
            try:
                status = run_status.get_status(row["run_id"])
            except FileNotFoundError:
                continue
            for checkpoint_id, deployment in status.deployments.items():
                state = deployment.get("state")
                if state == "cleanup_pending":
                    recovered += 1  # Already fenced; preserve the original failure.
                    continue
                if state in _DEPLOYMENT_BUSY_STATES:
                    error = "deployment lifecycle interrupted by control-plane restart"
                elif state in _DEPLOYMENT_READY_STATES and _spec_is_unservable(status):
                    error = "deployment spec is no longer supported"
                else:
                    continue
                if state == "queued" and "registration_identity" not in deployment:
                    failed = deployment_state.with_deployment_state(
                        deployment, "failed", error=error
                    )
                    marked = runner_transitions.mark_deployment_failed(status.run_id, failed)
                else:
                    marked = runner_transitions.mark_deployment_cleanup_pending(
                        status.run_id, error, checkpoint_id
                    )
                deployment_state._report_persisted_transition(status, marked, persisted=True)
                recovered += 1
        finally:
            lock.release()
    return recovered


def cleanup_pending_deployments(stop: Event) -> bool:
    """Try every pending checkpoint once; return whether another pass is needed."""
    pending = False
    for row in db.all_runs():
        if stop.is_set():
            break
        run_id = row["run_id"]
        lock = deployment_locks._deploy_lock(run_id)
        if not lock.acquire(blocking=False):
            pending = True
            continue
        try:
            try:
                status = run_status.get_status(run_id)
            except FileNotFoundError:
                continue
            for checkpoint_id, deployment in status.deployments.items():
                if stop.is_set():
                    return pending
                if deployment.get("state") != "cleanup_pending":
                    continue
                org_id = deployment.get("serving_org_id") or run_serving_org_id(status)
                # Normalize composite errors from records written before causes were separated.
                cause = (
                    deployment.get("deployment_error")
                    or str(deployment.get("error") or "interrupted deployment").split(
                        "; checkpoint cleanup failed:", 1
                    )[0]
                )
                if _cleanup_failed_deployment(
                    run_id, checkpoint_id, org_id, ServingError(cause), timeout_s=5.0
                ):
                    marked = runner_transitions.mark_undeployed(run_id, checkpoint_id)
                    deployment_state._report_persisted_transition(status, marked, persisted=True)
                    status = marked
                else:
                    pending = True
        finally:
            lock.release()
    return pending


def run_deployment_cleanup_worker(stop: Event) -> None:
    """Retry failures with backoff and poll for new cleanup work until shutdown."""
    delay = 1.0
    while not stop.is_set():
        pending = cleanup_pending_deployments(stop)
        if stop.wait(delay if pending else 30.0):
            break
        delay = min(30.0, delay * 2) if pending else 1.0


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


def _require_current_deployment_attempt(
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

    def _persist_registration_intent(identity: dict) -> None:
        nonlocal current
        _require_current_deployment_attempt(run_id, deployment, is_checkpoint, prev_state)
        registering = {**current, "registration_identity": identity}
        marked = runner_transitions.mark_deployment_pending(
            run_id, registering, owner_deployment=deployment
        )
        if marked.deployments.get(deployment["checkpoint_id"]) != registering:
            raise ServingError("deployment attempt changed before registration")
        current = registering

    def _verify_checkpoint_readiness(facts: DeploymentReadinessFacts) -> None:
        nonlocal current
        current = deployment_state.with_deployment_state(
            {**current, "checkpoint_id": facts.checkpoint_id, "openai_model": facts.checkpoint_id},
            "smoke_testing",
            detail="running bounded fixed-prompt smoke",
        )
        _require_current_deployment_attempt(run_id, deployment, is_checkpoint, prev_state)
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
        _require_current_deployment_attempt(run_id, deployment, is_checkpoint, prev_state)

    try:
        dep = serving_deploy.deploy_adapter(
            **deploy_kwargs,
            before_register=_persist_registration_intent,
            before_ready=_verify_checkpoint_readiness,
        )
    except serving_deploy.DeploymentGenerationUnconfirmed as exc:
        # Ownership changed or could not be checked. Do not delete a possible replacement.
        _record_deployment_failure(run_id, spec, exc, current, deployment, is_checkpoint)
        return
    except Exception as exc:
        if (
            ("registration_identity" in current or current.get("state") == "smoke_testing")
            and not isinstance(exc, RegistrationRejected)
            and not _cleanup_failed_deployment(
                run_id, deployment["checkpoint_id"], str(deploy_kwargs["org_id"]), exc
            )
        ):
            return
        _record_deployment_failure(run_id, spec, exc, current, deployment, is_checkpoint)
        return

    if not smoke_result:
        error = ServingError("deployment backend returned without checkpoint verification")
        if _cleanup_failed_deployment(
            run_id, deployment["checkpoint_id"], str(deploy_kwargs["org_id"]), error
        ):
            _record_deployment_failure(run_id, spec, error, current, deployment, is_checkpoint)
        return

    current = {**current, **dep.to_dict(), "verify": True}
    current = deployment_state.with_deployment_state(
        current,
        "ready",
        detail="permanent checkpoint verified and ready",
        **smoke_result,
    )
    if not _commit_ready_deployment(run_id, current, is_checkpoint, prev_state):
        print(
            f"deploy[{run_id}]: checkpoint ready but deployment record changed concurrently",
            flush=True,
        )


def _cleanup_failed_deployment(
    run_id: str,
    checkpoint_id: str,
    org_id: str,
    deployment_error: Exception,
    *,
    timeout_s: float | None = None,
) -> bool:
    """disable an unverified hosted checkpoint, retaining a retryable state on cleanup failure."""
    try:
        if not org_id:
            raise ServingError("run has no organization scope for checkpoint cleanup")
        deadline = time.monotonic() + timeout_s if timeout_s is not None else None
        previous = run_status.get_status(run_id)
        identity = previous.deployments.get(checkpoint_id, {}).get("registration_identity")
        if identity:
            record = serving_deploy.fetch_registered_checkpoint(
                org_id, checkpoint_id, timeout_s=timeout_s
            )
            if record is not None and not matches_registration_identity(record, identity):
                return True  # This attempt never owned the immutable binding; leave it intact.
        if identity and deadline is not None:
            timeout_s = deadline - time.monotonic()
            if timeout_s <= 0:
                raise ServingError("checkpoint cleanup deadline expired")
        kwargs = {"timeout_s": timeout_s} if timeout_s is not None else {}
        serving_deploy.undeploy_adapter(checkpoint_id, org_id=org_id, **kwargs)
    except Exception as exc:
        previous = run_status.get_status(run_id)
        marked = runner_transitions.mark_deployment_cleanup_pending(
            run_id, str(deployment_error), checkpoint_id=checkpoint_id, cleanup_error=str(exc)
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
    failed = deployment_state.with_deployment_state(
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
