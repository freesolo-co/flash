"""Serving endpoints: deploy / undeploy an adapter, list deployments, and chat.

Service functions are resolved through ``flash.server.app`` at call time so test-suite patches on
``app.<name>`` are honored here.
"""

from __future__ import annotations

import contextlib
import math

# `multiprocessing` has no call site left here since the smoke validation moved to
# `.serving_smoke`, but the schema coverage tests patch `get_context` through THIS module and the
# spawned validator reads it back that way. same for `safe_regex` and `validator_for` below.
import multiprocessing  # noqa: F401
import os
import time
from threading import Event
from typing import Annotated, NoReturn

import regex as safe_regex  # noqa: F401
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from jsonschema.validators import validator_for  # noqa: F401

from flash.core.spec import JobSpec
from flash.runner import (
    effective_spec_from_status,
    mark_checkpoint_deployed,
    mark_deployed,
    mark_deployment_failed,
    mark_deployment_pending,
    mark_deployment_revocation_failed,
    mark_undeployed,
    verified_adapter_revision_generation,
)
from flash.schema import parse_adapter_revision

# `RetryableServingUnavailable` is raised by the serving-coverage tests as
# `serving.RetryableServingUnavailable`, so it stays reachable here even though the smoke path
# that catches it now lives in `.serving_smoke`.
from flash.serve.deploy import (  # noqa: F401
    ActivationOutcomeUnknown,
    AdapterConfigMissing,
    RetryableServingUnavailable,
    ServingError,
)
from flash.serve.urls import public_deployment
from flash.server import app as _app
from flash.server.platform import db
from flash.server.platform.deps import _require_bool, manageable_run, owned_run, require_key
from flash.server.platform.internal_client import run_org_id

router = APIRouter()

_DEPLOYMENT_STALE_SECONDS = 30 * 60


def _deployment_state(deployment: dict, state: str, **fields) -> dict:
    return {**deployment, **fields, "state": state, "updated_at": time.time()}


def _public_deployment(deployment: dict) -> dict:
    out = public_deployment(deployment)
    run_id = out.get("run_id")
    out.update(
        {
            "run_id": run_id,
            "checkpoint_step": out.get("checkpoint_step"),
            "adapter_revision": out.get("adapter_revision"),
            "state": out.get("state"),
            "verified_at": out.get("verified_at"),
            "openai_model": out.get("openai_model") or run_id,
        }
    )
    return out


def _enqueue_deployment_report(status) -> None:
    from flash.runner import _report_status, _report_status_async

    if os.environ.get("FLASH_DEPLOY_SYNC") == "1":
        _report_status(status)
    else:
        _report_status_async(status)


def _report_persisted_transition(previous, current, *, persisted: bool) -> None:
    if not persisted or (
        previous.state == current.state and previous.deployment == current.deployment
    ):
        return
    _enqueue_deployment_report(current)


def _deployment_failure_persisted(status, failed: dict) -> bool:
    if status.deployment == failed:
        return True
    previous = failed.get("previous_deployment")
    deployment = status.deployment
    failure_fields = {"last_deploy_error", "last_deploy_failed_at"}
    expected_error = failed.get("error") or "deployment failed"
    return bool(
        isinstance(previous, dict)
        and isinstance(deployment, dict)
        and all(
            deployment.get(key) == value
            for key, value in previous.items()
            if key not in failure_fields
        )
        and deployment.get("last_deploy_error") == expected_error
        and deployment.get("last_deploy_failed_at") is not None
    )


def _deployment_attempt_is_stale(deployment: dict, *, now: float | None = None) -> bool:
    if deployment.get("state") not in _DEPLOYMENT_BUSY_STATES:
        return False
    raw = deployment.get("updated_at") or deployment.get("requested_at")
    try:
        stamp = float(raw)
    except (TypeError, ValueError):
        return True
    return (time.time() if now is None else now) - stamp >= _DEPLOYMENT_STALE_SECONDS


def recover_deployments() -> int:
    """Clear deployment lifecycle records left busy by a control-plane restart."""
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
        # another replica mid-deploy holds the flock, so a non-blocking miss proves live ownership.
        if not lock.acquire(blocking=False):
            continue
        try:
            try:
                status = _app.get_status(row["run_id"])
            except FileNotFoundError:
                continue
            deployment = status.deployment or {}
            state = deployment.get("state")
            if state in _DEPLOYMENT_BUSY_STATES:
                if not _deployment_attempt_is_stale(deployment):
                    continue
                error = "deployment lifecycle interrupted by control-plane restart"
                detail = "deployment interrupted; retry `flash models deploy`"
            elif state in _DEPLOYMENT_READY_STATES and _spec_is_unservable(status):
                # a ready record with an unparseable spec is unservable, so fail it during startup.
                # handle both readiness spellings from persisted builds; staleness applies only to
                # busy states.
                error = "deployment spec is no longer supported by this control plane"
                detail = "deployment retired: its algorithm was removed; submit a new run to deploy"
            else:
                continue
            failed = _deployment_state(
                deployment,
                "failed",
                error=error,
                detail=detail,
                recovered_at=time.time(),
            )
            marked = mark_deployment_failed(status.run_id, failed)
            _report_persisted_transition(
                status,
                marked,
                persisted=_deployment_failure_persisted(marked, failed),
            )
            recovered += 1
        finally:
            lock.release()
    return recovered


def replay_status_reports(stop: Event | None = None) -> int:
    """Sequentially mirror persisted statuses that may have been dropped during shutdown."""
    from flash.runner import _report_status

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


def _assert_deployment_activation_fence(
    run_id: str, deployment: dict, is_checkpoint: bool, prev_state: str
) -> None:
    """Raise ``ServingError`` unless this attempt still owns the record and may activate the alias.

    Re-read on every call: the point of the fence is that a cancel or a newer deploy can land while
    smoke is blocked, so a cached status would defeat it.
    """
    latest = _app.get_status(run_id)
    latest_deployment = latest.deployment or {}
    if (
        latest_deployment.get("requested_at") != deployment.get("requested_at")
        or latest_deployment.get("state") not in _DEPLOYMENT_BUSY_STATES
    ):
        raise ServingError("deployment attempt was superseded before alias activation")
    if is_checkpoint:
        if prev_state in _app._DEPLOYABLE_STATES and latest.state != prev_state:
            raise ServingError(
                f"run state changed from {prev_state!r} to {latest.state!r} before alias activation"
            )
        if latest.state in {"cancelled", "failed", "dry_run"} and latest.state != prev_state:
            raise ServingError(f"run became {latest.state!r} before checkpoint alias activation")
    elif latest.state != prev_state:
        raise ServingError(
            f"run state changed from {prev_state!r} to {latest.state!r} before alias activation"
        )
    expected_generation = deployment.get("verification_generation")
    if verified_adapter_revision_generation(run_id) != expected_generation:
        raise ServingError("deployment verification generation changed before alias activation")


def _commit_ready_deployment(
    run_id: str,
    current: dict,
    verification_generation,
    is_checkpoint: bool,
    prev_state: str,
) -> bool:
    """Persist the ready deployment record. Returns whether the guarded write landed."""
    previous = _app.get_status(run_id)
    if is_checkpoint:
        state_guard = prev_state if prev_state in _app._DEPLOYABLE_STATES else None
        marked = mark_checkpoint_deployed(
            run_id,
            current,
            expect_state=state_guard,
            verification_generation=verification_generation,
        )
        persisted = marked.deployment == current
    else:
        marked = mark_deployed(
            run_id,
            current,
            expect_state=prev_state,
            verification_generation=verification_generation,
        )
        persisted = marked.state == "deployed" and marked.deployment == current
    if persisted:
        _report_persisted_transition(previous, marked, persisted=marked.deployment == current)
    return persisted


def _reconcile_ready_commit_miss(
    run_id: str,
    current: dict,
    verification_generation,
    is_checkpoint: bool,
    deployment: dict,
) -> None:
    # deploy_adapter already flipped the serving alias when this runs, so a lost
    # control-plane cas must never be dropped silently — and the alias is never
    # reverted here (post-promotion recovery reads the authoritative alias; a revert
    # could clobber a newer deployment).
    latest = _app.get_status(run_id)
    latest_deployment = latest.deployment or {}
    owned = latest_deployment.get("requested_at") == deployment.get("requested_at")
    if owned and latest_deployment.get("state") in _DEPLOYMENT_BUSY_STATES:
        # this attempt still owns the record; only the run state moved under the
        # guard. retry the write once against the fresh state.
        previous = latest
        if is_checkpoint:
            marked = mark_checkpoint_deployed(
                run_id,
                current,
                verification_generation=verification_generation,
            )
        else:
            marked = mark_deployed(
                run_id,
                current,
                expect_state=latest.state,
                verification_generation=verification_generation,
            )
        if marked.deployment == current:
            _report_persisted_transition(previous, marked, persisted=marked.deployment == current)
            return
        latest = marked
        latest_deployment = latest.deployment or {}
    # superseded, undeployed, or uncommittable: a newer actor owns the record now, so
    # log the divergence loudly but never write over what that actor recorded — a
    # clobber here would erase a concurrent final deploy's ready record or resurrect
    # an explicit undeploy.
    divergence = (
        "deployment_record_diverged: serving alias targets "
        f"{current.get('adapter_revision')} but the deployment record moved to "
        f"{latest_deployment.get('state')!r} (run state {latest.state!r}) during "
        "activation; serving alias left as activated"
    )
    print(f"deploy[{run_id}]: {divergence}", flush=True)



def _finish_deployment_unlocked(
    *,
    run_id: str,
    spec_dict: dict,
    checkpoint_step: int | None,
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
    activated = False

    def _assert_activation_fence() -> None:
        _assert_deployment_activation_fence(run_id, deployment, is_checkpoint, prev_state)

    def _before_activate(adapter_revision: str, checkpoint: str) -> None:
        nonlocal current
        _assert_activation_fence()
        # smoke is unconditional for real deployments: alias activation only ever follows a
        # verified generation against the immutable revision.
        current = _deployment_state(
            {**current, "adapter_revision": adapter_revision},
            "smoke_testing",
            detail="running bounded fixed-prompt smoke",
        )
        previous = _app.get_status(run_id)
        marked = mark_deployment_pending(run_id, current, owner_deployment=deployment)
        _report_persisted_transition(previous, marked, persisted=marked.deployment == current)
        smoke_result.update(
            _run_deployment_smoke(
                run_id,
                spec,
                serving_model=adapter_revision,
                expected_checkpoint=checkpoint,
            )
        )
        current = _deployment_state(
            current,
            "reconciling",
            detail="activating alias and reconciling the authoritative target",
            activation_outcome_unknown=True,
        )
        previous = _app.get_status(run_id)
        marked = mark_deployment_pending(run_id, current, owner_deployment=deployment)
        _report_persisted_transition(previous, marked, persisted=marked.deployment == current)
        # cancellation can revoke the ledger while smoke is blocked, so fence again immediately
        # before deploy_adapter issues the activation request.
        _assert_activation_fence()

    try:
        dep = _app.deploy_adapter(**deploy_kwargs, before_activate=_before_activate)
        activated = True
        current = {**current, **dep.to_dict()}
        current.pop("activation_outcome_unknown", None)
        current["verify"] = True
        current = _deployment_state(
            current,
            "ready",
            detail="immutable revision verified and alias activated",
            **smoke_result,
        )
        verification_generation = current.get("verification_generation")
        current = _public_deployment(current)

        def _commit_ready() -> bool:
            return _commit_ready_deployment(
                run_id, current, verification_generation, is_checkpoint, prev_state
            )

        def _reconcile_commit_miss() -> None:
            _reconcile_ready_commit_miss(
                run_id, current, verification_generation, is_checkpoint, deployment
            )

        if not _commit_ready():
            _reconcile_commit_miss()
            return
    except Exception as exc:
        if isinstance(exc, ActivationOutcomeUnknown):
            reconciling = _deployment_state(
                current,
                "reconciling",
                error=str(exc),
                detail="alias activation outcome is unknown; authoritative reconciliation required",
                activation_outcome_unknown=True,
            )
            previous = _app.get_status(run_id)
            marked = mark_deployment_failed(run_id, reconciling)
            _report_persisted_transition(
                previous, marked, persisted=marked.deployment == reconciling
            )
            return
        if activated:
            try:
                latest = _app.get_status(run_id)
                latest_deployment = latest.deployment or {}
                if (
                    latest_deployment.get("adapter_revision") == current.get("adapter_revision")
                    and latest_deployment.get("state") in _DEPLOYMENT_READY_STATES
                ):
                    return
                if not _commit_ready():
                    _reconcile_commit_miss()
            except Exception as recovery_exc:
                divergence = (
                    "deployment_record_diverged: serving alias was activated for "
                    f"{current.get('adapter_revision')} but ready-state recovery failed after "
                    f"{exc!r}: {recovery_exc!r}"
                )
                print(f"deploy[{run_id}]: {divergence}", flush=True)
            return
        _record_deployment_failure(run_id, spec, exc, current, deployment, is_checkpoint)


def _record_deployment_failure(
    run_id: str,
    spec: JobSpec,
    exc: Exception,
    current: dict,
    deployment: dict,
    is_checkpoint: bool,
) -> None:
    """Persist the failed record for an attempt that never activated the alias."""
    error = str(exc)
    if not is_checkpoint and isinstance(exc, AdapterConfigMissing):
        steps = [c["step"] for c in _app.list_checkpoints(spec)]
        if steps:
            error = (
                f"run {run_id} has no run-level adapter at "
                f"{deployment.get('adapter_hf_prefix')} (the run likely never finalized); "
                f"deploy a saved checkpoint instead, e.g. `flash models deploy "
                f"{run_id}/step-{steps[-1]}` (available steps: "
                f"{', '.join(str(step) for step in steps)})"
            )
    failed_source = dict(current)
    if not deployment.get("activation_outcome_unknown"):
        failed_source.pop("activation_outcome_unknown", None)
    failed = _deployment_state(
        failed_source,
        "failed",
        error=error,
        detail="deployment failed; previous working alias was preserved",
    )
    previous = _app.get_status(run_id)
    marked = mark_deployment_failed(run_id, failed)
    _report_persisted_transition(
        previous, marked, persisted=_deployment_failure_persisted(marked, failed)
    )


def _finish_deployment(*, deploy_lock, **kwargs) -> None:
    try:
        _finish_deployment_unlocked(**kwargs)
    finally:
        deploy_lock.release()


def _validate_hf_repo_id(repository: str) -> None:
    """Validate HF repo id grammar early — malformed ids only 502 AFTER downloading the private source adapter."""
    try:
        from huggingface_hub.utils import HFValidationError, validate_repo_id
    except ModuleNotFoundError:
        return
    try:
        validate_repo_id(repository)
    except HFValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"repository is not a valid HuggingFace repo id: {exc}",
        ) from exc


@router.get("/v1/runs/{run_id}/deploy")
def deployment(
    run_id: str,
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    status = manageable_run(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
    persisted = (
        status.deployment if isinstance(status.deployment, dict) else {"state": "undeployed"}
    )
    return _public_deployment({**persisted, "run_id": run_id})


def _reject_contended_deploy(
    run_id: str,
    key: dict,
    x_freesolo_org_id: str | None,
    x_freesolo_project_id: str | None,
) -> NoReturn:
    """Raise the 409 for a deploy that could not take the per-run lock.

    Always raises. The lock was NOT acquired on this path, so the caller must not release it.
    """
    status = manageable_run(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
    current_deployment = status.deployment or {}
    current_deployment_state = current_deployment.get("state")
    if current_deployment_state in _DEPLOYMENT_BUSY_STATES:
        detail = (
            f"run {run_id} already has a deployment in "
            f"{current_deployment_state} state; run `flash models deployments` "
            "to check progress"
        )
    else:
        # the per-run lock is shared with undeploy, export, and startup recovery, so a
        # contended acquire cannot claim a deployment is running unless the state says so.
        detail = f"another operation is in progress for run {run_id}; retry shortly"
    raise HTTPException(status_code=409, detail=detail)


def _queued_deployment_record(
    run_id: str,
    spec: JobSpec,
    effective_spec: JobSpec,
    deploy_prefix: str,
    checkpoint_step,
    is_checkpoint: bool,
    current_deployment: dict,
    previous_deployment,
) -> dict:
    """Build the ``queued`` deployment record that gets persisted before the job starts."""
    # Validate the cheap configured-rank part synchronously so obvious spec errors return 400
    # instead of becoming background deployment failures.
    try:
        from flash.serve.deploy import validate_serving_lora_rank

        validate_serving_lora_rank(
            spec.model,
            effective_spec.train.lora_rank,
            rank_source="effective prepared LoRA rank",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        dep_dict = _app.deployment_record(
            run_id=run_id,
            model=spec.model,
            adapter_prefix=deploy_prefix,
            state="queued",
            checkpoint_step=checkpoint_step,
        ).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dep_dict = _deployment_state(
        dep_dict,
        "queued",
        detail="deployment queued",
        verify=True,
        requested_at=time.time(),
    )
    dep_dict["verification_generation"] = verified_adapter_revision_generation(run_id)
    if current_deployment.get("activation_outcome_unknown"):
        dep_dict["activation_outcome_unknown"] = True
    if is_checkpoint:
        dep_dict["checkpoint_step"] = checkpoint_step
    if previous_deployment:
        dep_dict["previous_deployment"] = previous_deployment
    return dep_dict


def _validate_deploy_request(
    run_id: str, status, spec: JobSpec, payload: dict, dry_run: bool
) -> tuple[JobSpec, dict]:
    """Reject a deploy that cannot proceed, before anything is queued or registered.

    Returns the effective spec and the current deployment record for the caller to work from.
    """
    if spec.model_revision:
        raise HTTPException(
            status_code=400,
            detail=(
                "deployment does not support revision-pinned base models; "
                "train without model_revision to deploy this run"
            ),
        )
    try:
        effective_spec = effective_spec_from_status(status)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # smoke verification is mandatory for every real deployment: a loadable-but-broken
    # revision must never become the bare-run alias target. reject an explicit opt-out
    # before anything is queued or registered (dry runs never register or activate, so
    # the flag is meaningless there too).
    if _require_bool(payload, "verify", True) is False:
        raise HTTPException(
            status_code=400,
            detail=(
                "verify=false is not supported: deployment smoke verification is "
                "mandatory before alias activation"
            ),
        )
    current_deployment = status.deployment or {}
    current_deployment_state = current_deployment.get("state")
    completed_unknown_activation = (
        current_deployment_state == "reconciling"
        and current_deployment.get("activation_outcome_unknown") is True
    )
    if (
        not dry_run
        and current_deployment_state in _DEPLOYMENT_BUSY_STATES
        and not completed_unknown_activation
        and not _deployment_attempt_is_stale(current_deployment)
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                f"run {run_id} already has a deployment in "
                f"{current_deployment.get('state')} state; run `flash models deployments` "
                "to check progress"
            ),
        )
    return effective_spec, current_deployment


@router.post("/v1/runs/{run_id}/deploy")
def deploy(
    run_id: str,
    key: Annotated[dict, Depends(require_key)],
    payload: dict | None = None,
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    payload = payload or {}
    deploy_lock = _app._deploy_lock(run_id)
    if not deploy_lock.acquire(blocking=False):
        _reject_contended_deploy(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
    job_owns_lock = False
    try:
        status = manageable_run(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
        spec = JobSpec.from_dict(status.spec)
        dry_run = _require_bool(payload, "dry_run", False)
        effective_spec, current_deployment = _validate_deploy_request(
            run_id, status, spec, payload, dry_run
        )
        checkpoint_step, is_checkpoint, deploy_prefix = _resolve_deployable_target(
            run_id,
            effective_spec,
            status,
            payload.get("step"),
            action="deploy",
            enforce_state=not dry_run,
        )
        if not dry_run and not effective_spec.train.hf_repo:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} has no [train].hf_repo; its adapter artifacts "
                    "cannot be located, so it cannot be deployed"
                ),
            )
        # CAS guard: /cancel's worker + provider teardown runs outside this lock (only its status
        # write is lock-serialized), so capture state before deploy and re-verify it on the write.
        prev_state = status.state
        # Prefer org from the run's own context over the caller's key (operator deploys land on run's owner).
        deploy_org_id = run_org_id(status) or str(key.get("org_id") or "").strip() or None
        previous_deployment = _deployment_predecessor(current_deployment)
        expected_adapter_revision = None
        if not dry_run:
            try:
                expected_adapter_revision, previous_deployment = _activation_predecessor(
                    run_id, current_deployment
                )
            except ServingError as exc:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "code": "alias_reconciliation_failed",
                        "run_id": run_id,
                        "retryable": True,
                        "message": str(exc),
                    },
                ) from exc
        deploy_kwargs = {
            "run_id": run_id,
            "model": spec.model,
            "hf_repo": effective_spec.train.hf_repo,
            "adapter_prefix": deploy_prefix,
            "dry_run": dry_run,
            "lora_rank": effective_spec.train.lora_rank,
            # a run trained with thinking serves with thinking (per-run parity)
            "thinking": spec.thinking,
            # a run trained with structured_outputs serves under the same grammar. thinking runs
            # are registered only after serving advertises deferred post-reasoning constraints.
            "structured_outputs": spec.train.structured_outputs,
            "org_id": deploy_org_id,
            "checkpoint_step": checkpoint_step,
            "expected_adapter_revision": expected_adapter_revision,
        }
        if dry_run:
            try:
                dep = _app.deploy_adapter(**deploy_kwargs)
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                raise
            return dep.to_dict()

        dep_dict = _queued_deployment_record(
            run_id,
            spec,
            effective_spec,
            deploy_prefix,
            checkpoint_step,
            is_checkpoint,
            current_deployment,
            previous_deployment,
        )
        marked = mark_deployment_pending(run_id, dep_dict, expect_state=prev_state)
        if marked.deployment != dep_dict:
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} became {marked.state!r} during deploy; aborted",
            )
        _report_persisted_transition(status, marked, persisted=True)

        job_kwargs = {
            "run_id": run_id,
            "spec_dict": status.spec,
            "checkpoint_step": checkpoint_step,
            "is_checkpoint": is_checkpoint,
            "deploy_kwargs": deploy_kwargs,
            "deployment": dep_dict,
            "prev_state": prev_state,
            "deploy_lock": deploy_lock,
        }
        # the job owns the lock from here on and releases it when the lifecycle ends, so a
        # booting replica's non-blocking probe keeps failing for the whole deploy. a start
        # failure means the job never ran, so ownership stays with this request.
        job_owns_lock = True
        try:
            ran_sync = _app.start_deployment_job(_finish_deployment, **job_kwargs)
        except _app.DeploymentJobStartError as exc:
            job_owns_lock = False
            error = f"deployment job could not start: {exc}"
            failed = _deployment_state(
                dep_dict,
                "failed",
                error=error,
                detail="deployment was not started; retry when the control plane is available",
                retryable=True,
            )
            previous = _app.get_status(run_id)
            marked = mark_deployment_failed(run_id, failed)
            _report_persisted_transition(
                previous, marked, persisted=_deployment_failure_persisted(marked, failed)
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "deployment_job_unavailable",
                    "run_id": run_id,
                    "retryable": True,
                    "message": error,
                },
            ) from exc
        if ran_sync:
            return _public_deployment(_app.get_status(run_id).deployment or dep_dict)
        return _public_deployment(dep_dict)
    finally:
        if not job_owns_lock:
            deploy_lock.release()


@router.delete("/v1/runs/{run_id}/deploy")
def undeploy(
    run_id: str,
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    with _app._deploy_lock(run_id):
        status = manageable_run(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
        try:
            result = _app.undeploy_adapter(run_id)
        except ServingError as exc:
            marked = mark_deployment_revocation_failed(run_id, str(exc))
            persisted = isinstance(marked.deployment, dict) and (
                marked.deployment.get("state") == "revocation_failed"
                and marked.deployment.get("error") == str(exc)
            )
            _report_persisted_transition(status, marked, persisted=persisted)
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "deployment_revocation_failed",
                    "run_id": run_id,
                    "retryable": True,
                    "message": str(exc),
                },
            ) from exc
        marked = mark_undeployed(run_id)
        persisted = isinstance(marked.deployment, dict) and (
            marked.deployment.get("state") == "undeployed"
        )
        _report_persisted_transition(status, marked, persisted=persisted)
        deployment = (
            marked.deployment if isinstance(marked.deployment, dict) else {"state": "undeployed"}
        )
        response = _public_deployment({**deployment, "run_id": run_id})
        response.update(
            {
                field: result[field]
                for field in ("disabled_aliases", "disabled_revisions", "serving_deregistered")
                if field in result
            }
        )
        return response


@router.post("/v1/runs/{run_id}/export")
def export(run_id: str, key: Annotated[dict, Depends(require_key)], payload: dict | None = None):
    """Copy a run's trained adapter into a user-owned HuggingFace repo."""
    payload = payload or {}
    with _app._deploy_lock(run_id):
        repository = str(payload.get("repository") or "").strip()
        if not repository:
            raise HTTPException(
                status_code=400,
                detail="repository is required: the destination HuggingFace repo 'owner/name'",
            )
        if len(parts := repository.strip("/").split("/")) != 2 or not all(parts):
            raise HTTPException(
                status_code=400,
                detail=f"repository must be a HuggingFace repo of the form 'owner/name', got {repository!r}",
            )
        repository = "/".join(parts)
        _validate_hf_repo_id(repository)
        hf_token = str(payload.get("hf_token") or "").strip()
        if not hf_token:
            raise HTTPException(
                status_code=400,
                detail="hf_token is required: a HuggingFace token with write access to the destination repo",
            )
        private = _require_bool(payload, "private", True)

        status = owned_run(run_id, key)
        spec = JobSpec.from_dict(status.spec)
        try:
            effective_spec = effective_spec_from_status(status)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if not effective_spec.train.hf_repo:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} has no [train].hf_repo; its adapter artifacts "
                    "cannot be located, so it cannot be exported"
                ),
            )
        checkpoint_step, is_checkpoint, prefix = _resolve_deployable_target(
            run_id, effective_spec, status, payload.get("step"), action="export", enforce_state=True
        )
        subfolder = f"{prefix}/adapter"
        try:
            url = _app.export_adapter(
                source_repo=effective_spec.train.hf_repo,
                source_subfolder=subfolder,
                dest_repo=repository,
                dest_token=hf_token,
                private=private,
                base_model=spec.model,
                base_model_revision=spec.model_revision,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ServingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    # best-effort product-analytics report: exports never otherwise touch the
    # platform backend (the copy is hf-to-hf inside flash).
    with contextlib.suppress(Exception):
        from flash.server.domain.run_registry import record_model_exported

        record_model_exported(
            status=status,
            key=key,
            repository=repository,
            url=url,
            step=checkpoint_step if is_checkpoint else None,
        )
    result = {
        "run_id": run_id,
        "adapter_id": run_id,
        "repository": repository,
        "url": url,
        "source": f"{run_id}/step-{checkpoint_step}" if is_checkpoint else run_id,
    }
    if is_checkpoint:
        result["step"] = checkpoint_step
    return result


@router.get("/v1/deployments")
def deployments(key: Annotated[dict, Depends(require_key)]):
    out = []
    for row in db.runs_for_key(key["id"]):
        try:
            status = _app.get_status(row["run_id"])
        except FileNotFoundError:
            continue
        if status.deployment and status.deployment.get("state") not in (
            "undeployed",
            "dry_run",
        ):
            data = status.to_dict()
            data["deployment"] = _public_deployment(data["deployment"])
            out.append(data)
    return {"deployments": out}


@router.post("/v1/runs/{run_id}/chat")
def chat(run_id: str, payload: dict, key: Annotated[dict, Depends(require_key)]):
    messages = _chat_messages_from_payload(payload)
    status = owned_run(run_id, key)
    adapter_revision = payload.get("adapter_revision")
    step = payload.get("step")
    verified_revisions = _verified_adapter_revisions(status)
    deployment = status.deployment or {}
    ready_deployment = _previous_ready_deployment(deployment)
    ready_revision = (
        ready_deployment.get("adapter_revision") if ready_deployment is not None else None
    )
    pinned_revision = _resolve_explicit_chat_revision(
        run_id,
        adapter_revision,
        step,
        verified_revisions,
        preferred_revision=ready_revision if isinstance(ready_revision, str) else None,
    )
    serving_model = pinned_revision or run_id
    spec = JobSpec.from_dict(status.spec)
    try:
        effective_spec = effective_spec_from_status(status)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    deployment_state = deployment.get("state")
    has_ready_deploy = pinned_revision is not None or ready_deployment is not None
    if pinned_revision is None and ready_deployment is not None:
        ready_revision = ready_deployment.get("adapter_revision")
        parsed_ready_revision = (
            parse_adapter_revision(ready_revision) if isinstance(ready_revision, str) else None
        )
        has_ready_deploy = bool(
            parsed_ready_revision is not None
            and parsed_ready_revision[0] == run_id
            and ready_revision in verified_revisions
        )
    # A cancelled run can still serve a per-step checkpoint it deployed: checkpoint deploy records
    # a live adapter that /v1/deployments lists as active without requiring a final adapter.
    # Only block chat when there's no active deployment to serve.
    if not has_ready_deploy:
        if deployment_state in _DEPLOYMENT_BUSY_STATES:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} deployment is {deployment_state}; run "
                    "`flash models deployments` to check progress"
                ),
            )
        if deployment_state == "failed":
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} deployment failed: {deployment.get('error') or 'unknown error'}"
                ),
            )
        if status.state == "cancelled":
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} was cancelled; deploy a checkpoint with "
                f"`flash models deploy {run_id}/step-<N>` first",
            )
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} has no active deployment; `flash models deploy {run_id}` first",
        )
    if not effective_spec.train.hf_repo:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} has no [train].hf_repo; its adapter cannot be served",
        )
    # Parse sampling params before the broad try so bad values are 400, not 502.
    try:
        temperature = float(payload.get("temperature") or 0.0)
        # Avoid `or 512`: that silently coerces an explicit 0 to 512.
        raw_max_tokens = payload.get("max_tokens")
        # OverflowError (int(inf), an ArithmeticError) is NOT a TypeError/ValueError — catch it too so a
        # JSON `Infinity`/`1e400` max_tokens is a clean 400, not an uncaught 500.
        max_tokens = 512 if raw_max_tokens is None else int(raw_max_tokens)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid temperature/max_tokens: {exc}"
        ) from exc
    if not math.isfinite(temperature):
        raise HTTPException(
            status_code=400, detail=f"temperature must be a finite number, got {temperature}"
        )
    if max_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens must be a positive integer, got {max_tokens}",
        )
    # the same stops the deployment smoke verified with: a run trained to terminate on a delimiter
    # rather than EOS would otherwise pass verification and then run to max_tokens, or emit trailing
    # text past its answer, on every real request.
    stop_sequences = [
        str(value) for value in (getattr(spec.train, "stop_sequences", ()) or ())
    ] or None
    try:
        if payload.get("stream") is True:
            return StreamingResponse(
                _app.serve_chat_stream(
                    run_id=serving_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking=spec.thinking,
                    stop=stop_sequences,
                ),
                media_type="text/plain; charset=utf-8",
            )
        return _app.serve_chat(
            run_id=serving_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=spec.thinking,
            stop=stop_sequences,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"inference failure: {exc}") from exc


# re-exported at the bottom rather than imported at the top: both modules resolve names back
# through this one, so a top import would be circular. the server tests address these helpers as
# attributes of `flash.server.routes.serving`, which is why they stay on it after the move.
from flash.server.routes.serving_revisions import (  # noqa: E402,F401
    _DEPLOYMENT_BUSY_STATES,
    _DEPLOYMENT_READY_STATES,
    _activation_predecessor,
    _chat_messages_from_payload,
    _deployment_predecessor,
    _format_deployed_steps,
    _parse_checkpoint_step,
    _previous_ready_deployment,
    _resolve_deploy_step,
    _resolve_deployable_target,
    _resolve_explicit_chat_revision,
    _spec_is_unservable,
    _verified_adapter_revisions,
    _verified_step_index,
)
from flash.server.routes.serving_smoke import (  # noqa: E402,F401
    _JSON_SCHEMA_PROCESS_NAME,
    _SMOKE_BUDGET_SECONDS,
    _SMOKE_PROMPT,
    _bounded_call,
    _bounded_regex_fullmatch,
    _json_schema_validation_worker,
    _reap_schema_validation_process,
    _run_deployment_smoke,
    _sanitized_schema_error,
    _smoke_provenance,
    _smoke_timeout_error,
    _strict_json_loads,
    _thinking_answer,
    _thinking_tag_is_guaranteed,
    _validate_json_schema,
    _validate_structured_smoke,
)
