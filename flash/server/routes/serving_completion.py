"""complete and recover exact managed checkpoint deployments."""

from __future__ import annotations

import time
from threading import Event

import flash.runner.supervise.transitions as runner_transitions
from flash.core.spec import JobSpec
from flash.runner.results.verified_revisions import verified_checkpoint_generation
from flash.serve.contract.errors import AdapterConfigMissing, ServingError
from flash.server.asgi import app as _app
from flash.server.platform import db
from flash.server.routes.serving_revisions import (
    _DEPLOYMENT_BUSY_STATES,
    _DEPLOYMENT_READY_STATES,
    _spec_is_unservable,
)


def _serving():
    from flash.server.routes import serving

    return serving


def _deployment_state(deployment: dict, state: str, **fields) -> dict:
    return _serving()._deployment_state(deployment, state, **fields)


def _deployment_failure_persisted(status, failed: dict) -> bool:
    return _serving()._deployment_failure_persisted(status, failed)


def _public_deployment(deployment: dict) -> dict:
    return _serving()._public_deployment(deployment)


def recover_deployments() -> int:
    recovered = 0
    for row in db.all_runs():
        try:
            status = _app.get_status(row["run_id"])
        except FileNotFoundError:
            continue
        state = (status.deployment or {}).get("state")
        if state not in _DEPLOYMENT_BUSY_STATES and state not in _DEPLOYMENT_READY_STATES:
            continue
        lock = _app._deploy_lock(row["run_id"])
        if not lock.acquire(blocking=False):
            continue
        try:
            status = _app.get_status(row["run_id"])
            deployment = status.deployment or {}
            state = deployment.get("state")
            if state in _DEPLOYMENT_BUSY_STATES:
                error = "deployment lifecycle interrupted by control-plane restart"
                detail = "deployment interrupted; redeploy the exact checkpoint"
            elif state in _DEPLOYMENT_READY_STATES and _spec_is_unservable(status):
                error = "deployment spec is no longer supported by this control plane"
                detail = "deployment retired; submit a new run to deploy"
            else:
                continue
            failed = _deployment_state(
                deployment,
                "failed",
                error=error,
                detail=detail,
                recovered_at=time.time(),
            )
            marked = runner_transitions.mark_deployment_failed(status.run_id, failed)
            _serving()._report_persisted_transition(
                status,
                marked,
                persisted=_deployment_failure_persisted(marked, failed),
            )
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
            status = _app.get_status(row["run_id"])
            _report_status(status)
        except (OSError, TypeError, ValueError):
            continue
        replayed += 1
    return replayed


def _assert_deployment_fence(
    run_id: str, deployment: dict, is_checkpoint: bool, prev_state: str
) -> None:
    latest = _app.get_status(run_id)
    latest_deployment = latest.deployment or {}
    if (
        latest_deployment.get("requested_at") != deployment.get("requested_at")
        or latest_deployment.get("state") not in _DEPLOYMENT_BUSY_STATES
    ):
        raise ServingError("deployment attempt was superseded before checkpoint readiness")
    if is_checkpoint:
        if prev_state in _app._DEPLOYABLE_STATES and latest.state != prev_state:
            raise ServingError(f"run state changed from {prev_state!r} to {latest.state!r}")
    elif latest.state != prev_state:
        raise ServingError(f"run state changed from {prev_state!r} to {latest.state!r}")
    if verified_checkpoint_generation(run_id) != deployment.get("verification_generation"):
        raise ServingError("deployment verification generation changed")


def _commit_ready_deployment(
    run_id: str,
    current: dict,
    verification_generation,
    is_checkpoint: bool,
    prev_state: str,
) -> bool:
    previous = _app.get_status(run_id)
    if is_checkpoint:
        state_guard = prev_state if prev_state in _app._DEPLOYABLE_STATES else None
        marked = runner_transitions.mark_checkpoint_deployed(
            run_id,
            current,
            expect_state=state_guard,
            verification_generation=verification_generation,
        )
        persisted = marked.deployment == current
    else:
        marked = runner_transitions.mark_deployed(
            run_id,
            current,
            expect_state=prev_state,
            verification_generation=verification_generation,
        )
        persisted = marked.state == "deployed" and marked.deployment == current
    if persisted:
        _serving()._report_persisted_transition(previous, marked, persisted=True)
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
    active = _app.get_status(run_id).deployment or {}
    if (
        active.get("requested_at") != deployment.get("requested_at")
        or active.get("state") not in _DEPLOYMENT_BUSY_STATES
    ):
        return
    current = dict(deployment)
    smoke_result: dict = {}

    def _before_ready(
        checkpoint_id: str,
        checkpoint: str,
        *,
        advertised_capabilities: frozenset[str] | None = None,
        adapter_targets_images: bool | None = None,
    ) -> None:
        nonlocal current
        _assert_deployment_fence(run_id, deployment, is_checkpoint, prev_state)
        current = _deployment_state(
            {**current, "checkpoint_id": checkpoint_id, "openai_model": checkpoint_id},
            "smoke_testing",
            detail="running bounded fixed-prompt smoke",
        )
        previous = _app.get_status(run_id)
        marked = _serving().mark_deployment_pending(run_id, current, owner_deployment=deployment)
        _serving()._report_persisted_transition(
            previous, marked, persisted=marked.deployment == current
        )
        smoke_result.update(
            _serving()._run_deployment_smoke(
                run_id,
                spec,
                serving_model=checkpoint_id,
                expected_checkpoint=checkpoint,
                org_id=str(deploy_kwargs["org_id"]),
                advertised_capabilities=advertised_capabilities,
                adapter_targets_images=adapter_targets_images,
            )
        )
        _assert_deployment_fence(run_id, deployment, is_checkpoint, prev_state)

    try:
        dep = _app.deploy_adapter(**deploy_kwargs, before_ready=_before_ready)
    except Exception as exc:
        _record_deployment_failure(run_id, spec, exc, current, deployment, is_checkpoint)
        return

    current = {**current, **dep.to_dict(), "verify": True}
    current = _deployment_state(
        current,
        "ready",
        detail="permanent checkpoint verified and ready",
        **smoke_result,
    )
    verification_generation = current.get("verification_generation")
    current = _public_deployment(current)
    if not _commit_ready_deployment(
        run_id, current, verification_generation, is_checkpoint, prev_state
    ):
        print(
            f"deploy[{run_id}]: checkpoint ready but deployment record changed concurrently",
            flush=True,
        )


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
        steps = [candidate["step"] for candidate in _app.list_checkpoints(spec)]
        if steps:
            error = (
                f"run {run_id} has no final adapter at {deployment.get('adapter_hf_prefix')}; "
                f"deploy an explicit saved checkpoint such as {run_id}/step-{steps[-1]}"
            )
    failed = _deployment_state(
        current,
        "failed",
        error=error,
        detail="checkpoint deployment failed; sibling checkpoints are unchanged",
    )
    previous = _app.get_status(run_id)
    marked = runner_transitions.mark_deployment_failed(run_id, failed)
    _serving()._report_persisted_transition(
        previous, marked, persisted=_deployment_failure_persisted(marked, failed)
    )
