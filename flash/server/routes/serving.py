"""Serving endpoints: deploy / undeploy an adapter, list deployments, and chat.

Services are called through their owning modules.
"""

from __future__ import annotations

import contextlib
import time
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, Header, HTTPException

import flash.runner.lifecycle.status as run_status
import flash.serve.deployment.deploy as serving_deploy
import flash.serve.deployment.export as serving_export
import flash.server.platform.deployment_jobs as deployment_jobs
import flash.server.platform.deployment_state as deployment_state
import flash.server.platform.locks as deployment_locks
import flash.server.routes.serving_chat as serving_chat
import flash.server.routes.serving_completion as serving_completion
import flash.server.routes.serving_revisions as serving_revisions
from flash.core.spec import JobSpec, require_project_id
from flash.runner.lifecycle.status import effective_spec_from_status
from flash.runner.results.deployments import generation, is_ready, require_checkpoint
from flash.runner.supervise.transitions import (
    mark_deployment_cleanup_pending,
    mark_deployment_failed,
    mark_deployment_pending,
    mark_undeployed,
)
from flash.serve.contract.errors import ServingError
from flash.serve.deployment.readiness import DEPLOYMENT_TIMEOUT_SECONDS
from flash.server.platform import auth, db
from flash.server.platform.deps import _require_bool, manageable_run, owned_run, require_key
from flash.server.platform.internal_client import run_org_id, run_serving_org_id

router = APIRouter()

_DEPLOYMENT_STALE_SECONDS = DEPLOYMENT_TIMEOUT_SECONDS


def _deployment_attempt_is_stale(deployment: dict, *, now: float | None = None) -> bool:
    if deployment.get("state") not in serving_revisions._DEPLOYMENT_BUSY_STATES:
        return False
    raw = deployment.get("updated_at") or deployment.get("requested_at")
    try:
        stamp = float(raw)
    except (TypeError, ValueError):
        return True
    return (time.time() if now is None else now) - stamp >= _DEPLOYMENT_STALE_SECONDS


def _finish_deployment(*, deploy_lock, **kwargs) -> None:
    try:
        serving_completion._finish_deployment_unlocked(**kwargs)
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
    checkpoint_id: str | None = None,
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    status = manageable_run(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
    if checkpoint_id is None:
        return {"deployments": status.to_dict()["deployments"]}
    try:
        require_checkpoint(run_id, checkpoint_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    persisted = status.deployments.get(
        checkpoint_id, {"state": "undeployed", "checkpoint_id": checkpoint_id}
    )
    return deployment_state._public_deployment({**persisted, "run_id": run_id})


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
    current_deployment = next(
        (
            record
            for record in status.deployments.values()
            if record.get("state") in serving_revisions._DEPLOYMENT_BUSY_STATES
        ),
        {},
    )
    current_deployment_state = current_deployment.get("state")
    if current_deployment_state in serving_revisions._DEPLOYMENT_BUSY_STATES:
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
    effective_spec: JobSpec,
    deploy_prefix: str,
    checkpoint_step,
    is_checkpoint: bool,
    current_deployment: dict,
) -> dict:
    """Build the ``queued`` deployment record that gets persisted before the job starts."""
    # Validate the cheap configured-rank part synchronously so obvious spec errors return 400
    # instead of becoming background deployment failures.
    try:
        from flash.serve.deployment.adapter_check import validate_serving_lora_rank

        validate_serving_lora_rank(
            effective_spec.model,
            effective_spec.train.lora_rank,
            rank_source="effective prepared LoRA rank",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        dep_dict = serving_deploy.deployment_record(
            run_id=run_id,
            model=effective_spec.model,
            adapter_prefix=deploy_prefix,
            state="queued",
            checkpoint_step=checkpoint_step,
        ).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dep_dict = deployment_state._deployment_state(
        dep_dict,
        "queued",
        detail="deployment queued",
        verify=True,
        requested_at=time.time(),
    )
    dep_dict["generation"] = generation(current_deployment) + 1
    if is_checkpoint:
        dep_dict["checkpoint_step"] = checkpoint_step
    return dep_dict


def _validate_deploy_request(
    run_id: str, status, payload: dict, dry_run: bool
) -> tuple[JobSpec, dict]:
    """Reject a deploy that cannot proceed, before anything is queued or registered.

    Returns the effective spec and current deployment record.
    """
    try:
        effective_spec = effective_spec_from_status(status)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    # smoke verification is mandatory for every real checkpoint deployment. reject an opt-out
    # before anything is queued or registered (dry runs never register or activate, so
    # the flag is meaningless there too).
    if _require_bool(payload, "verify", True) is False:
        raise HTTPException(
            status_code=400,
            detail=(
                "verify=false is not supported: deployment smoke verification is "
                "mandatory before checkpoint readiness"
            ),
        )
    target = payload.get("checkpoint_id")
    current_deployment = status.deployments.get(target, {}) if isinstance(target, str) else {}
    current_deployment_state = current_deployment.get("state")
    if (
        not dry_run
        and current_deployment_state in serving_revisions._DEPLOYMENT_BUSY_STATES
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


def _require_deploy_org(run_id: str, deploy_org_id: str | None) -> None:
    """Fail closed when a managed-plane deploy would register an adapter with no owning org.

    Serving authorizes external chat requests against the org that owns the adapter (see
    platform docs), so registering a revision without an org would leave the field's
    enforcement to whatever the serving backend does with an unowned adapter. A standalone
    plane is single-tenant by definition and has no organization directory, so it keeps
    deploying without one (same escape hatch as ``deps.manageable_run``).
    """
    if deploy_org_id is not None or auth.standalone():
        return
    raise HTTPException(
        status_code=409,
        detail=(
            f"run {run_id} has no owning organization on record and the caller's key carries "
            "none; refusing to register an adapter without an org, because serving authorizes "
            "chat requests against the org that owns it. deploy with a key that belongs to the "
            "run's organization, or re-submit the run through the platform so it carries an "
            "org context."
        ),
    )


@router.post("/v1/runs/{run_id}/deploy")
def deploy(
    run_id: str,
    key: Annotated[dict, Depends(require_key)],
    payload: dict | None = None,
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    payload = payload or {}
    deploy_lock = deployment_locks._deploy_lock(run_id)
    if not deploy_lock.acquire(blocking=False):
        _reject_contended_deploy(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
    job_owns_lock = False
    try:
        status = manageable_run(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
        dry_run = _require_bool(payload, "dry_run", False)
        effective_spec, current_deployment = _validate_deploy_request(
            run_id, status, payload, dry_run
        )
        checkpoint_step, is_checkpoint, deploy_prefix = (
            serving_revisions._resolve_deployable_target(
                run_id,
                effective_spec,
                status,
                payload.get("checkpoint_id"),
                action="deploy",
                enforce_state=not dry_run,
            )
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
        _require_deploy_org(run_id, deploy_org_id)
        deploy_org_id = auth.serving_org_id(deploy_org_id)
        deploy_kwargs = {
            "run_id": run_id,
            "model": effective_spec.model,
            "hf_repo": effective_spec.train.hf_repo,
            "adapter_prefix": deploy_prefix,
            "dry_run": dry_run,
            "lora_rank": effective_spec.train.lora_rank,
            # a run trained with thinking serves with thinking (per-run parity)
            "thinking": effective_spec.thinking,
            # a run trained with structured_outputs serves under the same grammar. thinking runs
            # are registered only after serving advertises deferred post-reasoning constraints.
            "structured_outputs": effective_spec.train.structured_outputs,
            "org_id": deploy_org_id,
            "checkpoint_step": checkpoint_step,
        }
        if dry_run:
            try:
                dep = serving_deploy.deploy_adapter(**deploy_kwargs)
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                raise
            return dep.to_dict()

        if is_ready(run_id, current_deployment):
            return deployment_state._public_deployment(current_deployment)

        dep_dict = _queued_deployment_record(
            run_id,
            effective_spec,
            deploy_prefix,
            checkpoint_step,
            is_checkpoint,
            current_deployment,
        )
        marked = mark_deployment_pending(run_id, dep_dict, expect_state=prev_state)
        if marked.deployments.get(dep_dict["checkpoint_id"]) != dep_dict:
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} became {marked.state!r} during deploy; aborted",
            )
        deployment_state._report_persisted_transition(status, marked, persisted=True)

        job_kwargs = {
            "run_id": run_id,
            "spec_dict": status.spec,
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
            ran_sync = deployment_jobs.start_deployment_job(_finish_deployment, **job_kwargs)
        except deployment_jobs.DeploymentJobStartError as exc:
            job_owns_lock = False
            error = f"deployment job could not start: {exc}"
            failed = deployment_state._deployment_state(
                dep_dict,
                "failed",
                error=error,
                detail="deployment was not started; retry when the control plane is available",
                retryable=True,
            )
            previous = run_status.get_status(run_id)
            marked = mark_deployment_failed(run_id, failed)
            deployment_state._report_persisted_transition(
                previous,
                marked,
                persisted=deployment_state._deployment_failure_persisted(marked, failed),
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
            return deployment_state._public_deployment(
                run_status.get_status(run_id).deployments.get(dep_dict["checkpoint_id"], dep_dict)
            )
        return deployment_state._public_deployment(dep_dict)
    finally:
        if not job_owns_lock:
            deploy_lock.release()


@router.delete("/v1/runs/{run_id}/deploy")
def undeploy(
    run_id: str,
    checkpoint_id: str,
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    manageable_run(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
    with deployment_locks._deploy_lock(run_id):
        status = manageable_run(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
        try:
            from flash.schema import parse_checkpoint_ref

            parsed = parse_checkpoint_ref(checkpoint_id)
            if parsed is None or parsed[0] != run_id:
                raise HTTPException(
                    status_code=400,
                    detail="checkpoint_id must be a canonical checkpoint belonging to the route run",
                )
            org_id = run_serving_org_id(status)
            if not org_id:
                raise HTTPException(
                    status_code=409, detail=f"run {run_id} has no organization scope"
                )
            result = serving_deploy.undeploy_adapter(checkpoint_id, org_id=org_id)
        except ServingError as exc:
            marked = mark_deployment_cleanup_pending(run_id, str(exc), checkpoint_id=checkpoint_id)
            record = marked.deployments.get(checkpoint_id, {})
            persisted = record.get("state") == "cleanup_pending" and record.get("error") == str(exc)
            deployment_state._report_persisted_transition(status, marked, persisted=persisted)
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "deployment_cleanup_failed",
                    "run_id": run_id,
                    "retryable": True,
                    "message": str(exc),
                },
            ) from exc
        marked = mark_undeployed(run_id, checkpoint_id)
        deployment_state._report_persisted_transition(status, marked, persisted=True)
        response = deployment_state._public_deployment(marked.deployments[checkpoint_id])
        response.update(
            {
                field: result[field]
                for field in ("disabled_checkpoints", "serving_deregistered")
                if field in result
            }
        )
        return response


@router.post("/v1/runs/{run_id}/export")
def export(run_id: str, key: Annotated[dict, Depends(require_key)], payload: dict | None = None):
    """Copy a run's trained adapter into a user-owned HuggingFace repo."""
    payload = payload or {}
    owned_run(run_id, key)
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

    with deployment_locks._deploy_lock(run_id):
        status = owned_run(run_id, key)
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
        checkpoint_step, is_checkpoint, prefix = serving_revisions._resolve_deployable_target(
            run_id,
            effective_spec,
            status,
            payload.get("checkpoint_id"),
            action="export",
            enforce_state=True,
        )
        subfolder = f"{prefix}/adapter"
        try:
            url = serving_export.export_adapter(
                source_repo=effective_spec.train.hf_repo,
                source_subfolder=subfolder,
                dest_repo=repository,
                dest_token=hf_token,
                private=private,
                base_model=effective_spec.model,
                # the effective half carries the runner-assigned revision stripped from the public
                # spec. the worker stamps that sha into adapter_config.json from its internal spec,
                # and export refuses a stamped revision that disagrees with what it is handed --
                # so reading the public half here 404s every auto-pinned sft run and every warm
                # start that inherited its pin.
                base_model_revision=effective_spec.model_revision,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ServingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    # best-effort product-analytics report: exports never otherwise touch the
    # platform backend (the copy is hf-to-hf inside flash).
    with contextlib.suppress(Exception):
        from flash.server.domain.registry.runs import record_model_exported

        record_model_exported(
            status=status,
            key=key,
            repository=repository,
            url=url,
            step=checkpoint_step if is_checkpoint else None,
        )
    result = {
        "run_id": run_id,
        "checkpoint_id": payload.get("checkpoint_id"),
        "repository": repository,
        "url": url,
        "source": payload.get("checkpoint_id"),
    }
    if is_checkpoint:
        result["step"] = checkpoint_step
    return result


def _deployment_listing_scope(
    key: dict, org_id: str | None, project_id: str | None
) -> tuple[str, str] | None:
    """The (org, project) filter for ``/v1/deployments``, or None for an exact-key listing.

    Mirrors ``deps.manageable_run``: on a managed plane the internal key is the platform proxy
    and owns the runs it submitted on every org's behalf, so an unscoped listing would cross
    orgs. It must name the org AND project it lists for, exactly as it must to manage a single
    deployment. The headers are honored only for the internal key; a user key can only ever see
    its own runs, and a standalone plane is single-tenant and keeps the exact-key listing.
    """
    if key.get("auth_kind") != "internal" or auth.standalone():
        return None
    org = str(org_id or "").strip()
    try:
        project = require_project_id(project_id)
    except (TypeError, ValueError):
        project = None
    if not org or project is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "an internal-key deployment listing must be scoped: send X-Freesolo-Org-Id "
                "and X-Freesolo-Project-Id for the org and project being listed"
            ),
        )
    return org, project


def _in_deployment_listing_scope(status, org: str, project: str) -> bool:
    """Whether a run belongs to the requested org AND project (manageable_run's predicate)."""
    from flash.runner.lifecycle.preparation import _status_org_id

    if _status_org_id(status) != org:
        return False
    persisted_project = status.spec.get("project") if isinstance(status.spec, dict) else None
    try:
        return require_project_id(persisted_project) == project
    except (TypeError, ValueError):
        return False


@router.get("/v1/deployments")
def deployments(
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    scope = _deployment_listing_scope(key, x_freesolo_org_id, x_freesolo_project_id)
    out = []
    for row in db.runs_for_key(key["id"]):
        try:
            status = run_status.get_status(row["run_id"])
        except FileNotFoundError:
            continue
        if scope is not None and not _in_deployment_listing_scope(status, *scope):
            continue
        for record in status.deployments.values():
            if record.get("state") in {"undeployed", "dry_run"}:
                continue
            data = status.to_dict()
            data.pop("deployments")
            data["deployment"] = deployment_state._public_deployment(record)
            out.append(data)
    return {"deployments": out}


@router.post("/v1/runs/{run_id}/chat")
def chat(
    run_id: str,
    payload: dict,
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    return serving_chat.managed_chat(
        run_id,
        payload,
        key,
        x_freesolo_org_id,
        x_freesolo_project_id,
    )
