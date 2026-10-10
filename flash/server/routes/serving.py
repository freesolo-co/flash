"""Serving endpoints: deploy / undeploy an adapter, list deployments, and chat.

Service functions are resolved through ``flash.server.asgi.app`` at call time so test-suite patches on
``app.<name>`` are honored here.
"""

from __future__ import annotations

import contextlib

# `multiprocessing` has no call site left here since the smoke validation moved to
# `.serving_smoke`, but the schema coverage tests patch `get_context` through THIS module and the
# spawned validator reads it back that way. same for `safe_regex` and `validator_for` below.
import multiprocessing  # noqa: F401
import os
import time
from typing import Annotated, NoReturn

import regex as safe_regex  # noqa: F401
from fastapi import APIRouter, Depends, Header, HTTPException
from jsonschema.validators import validator_for  # noqa: F401

from flash.core.spec import JobSpec, require_project_id
from flash.runner.lifecycle.status import effective_spec_from_status
from flash.runner.results.verified_revisions import verified_checkpoint_generation
from flash.runner.supervise.transitions import (
    mark_deployment_failed,
    mark_deployment_pending,
    mark_deployment_revocation_failed,
    mark_undeployed,
)

# `RetryableServingUnavailable` is raised by the serving-coverage tests as
# `serving.RetryableServingUnavailable`, so it stays reachable here even though the smoke path
# that catches it now lives in `.serving_smoke`.
from flash.serve.contract.errors import (  # noqa: F401
    AdapterConfigMissing,
    RetryableServingUnavailable,
    ServingError,
)
from flash.serve.contract.urls import public_deployment
from flash.server.asgi import app as _app
from flash.server.platform import auth, db
from flash.server.platform.deps import _require_bool, manageable_run, owned_run, require_key
from flash.server.platform.internal_client import run_org_id, run_serving_org_id

router = APIRouter()

_DEPLOYMENT_STALE_SECONDS = 40 * 60


def _deployment_state(deployment: dict, state: str, **fields) -> dict:
    return {**deployment, **fields, "state": state, "updated_at": time.time()}


def _public_deployment(deployment: dict) -> dict:
    out = public_deployment(deployment)
    run_id = out.get("run_id")
    out.update(
        {
            "run_id": run_id,
            "checkpoint_step": out.get("checkpoint_step"),
            "checkpoint_id": out.get("checkpoint_id"),
            "state": out.get("state"),
            "verified_at": out.get("verified_at"),
            "openai_model": out.get("openai_model") or out.get("checkpoint_id"),
        }
    )
    return out


def _enqueue_deployment_report(status) -> None:
    from flash.runner.lifecycle.reporting import _report_status, _report_status_async

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
        from flash.serve.deployment.adapter_check import validate_serving_lora_rank

        validate_serving_lora_rank(
            effective_spec.model,
            effective_spec.train.lora_rank,
            rank_source="effective prepared LoRA rank",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        dep_dict = _app.deployment_record(
            run_id=run_id,
            model=effective_spec.model,
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
    dep_dict["verification_generation"] = verified_checkpoint_generation(run_id)
    if is_checkpoint:
        dep_dict["checkpoint_step"] = checkpoint_step
    if previous_deployment:
        dep_dict["previous_deployment"] = previous_deployment
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
    current_deployment = status.deployment or {}
    current_deployment_state = current_deployment.get("state")
    if (
        not dry_run
        and current_deployment_state in _DEPLOYMENT_BUSY_STATES
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
    deploy_lock = _app._deploy_lock(run_id)
    if not deploy_lock.acquire(blocking=False):
        _reject_contended_deploy(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
    job_owns_lock = False
    try:
        status = manageable_run(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
        dry_run = _require_bool(payload, "dry_run", False)
        effective_spec, current_deployment = _validate_deploy_request(
            run_id, status, payload, dry_run
        )
        checkpoint_step, is_checkpoint, deploy_prefix = _resolve_deployable_target(
            run_id,
            effective_spec,
            status,
            payload.get("checkpoint_id"),
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
                dep = _app.deploy_adapter(**deploy_kwargs)
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                raise
            return dep.to_dict()

        dep_dict = _queued_deployment_record(
            run_id,
            effective_spec,
            deploy_prefix,
            checkpoint_step,
            is_checkpoint,
            current_deployment,
            None,
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
    checkpoint_id: str,
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    manageable_run(run_id, key, x_freesolo_org_id, x_freesolo_project_id)
    with _app._deploy_lock(run_id):
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
            result = _app.undeploy_adapter(checkpoint_id, org_id=org_id)
        except ServingError as exc:
            marked = mark_deployment_revocation_failed(
                run_id, str(exc), checkpoint_id=checkpoint_id
            )
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
        marked = mark_undeployed(run_id, checkpoint_id)
        persisted = isinstance(marked.deployment, dict) and (
            marked.deployment.get("state") == "undeployed"
        )
        _report_persisted_transition(status, marked, persisted=persisted)
        deployment = (
            marked.deployment if isinstance(marked.deployment, dict) else {"state": "undeployed"}
        )
        removed_summary = (
            deployment.get("state") == "undeployed"
            and deployment.get("checkpoint_id") == checkpoint_id
        )
        response_state = (
            deployment
            if removed_summary
            else {
                "state": "undeployed",
                "checkpoint_id": checkpoint_id,
                "checkpoint_step": parsed[1],
            }
        )
        response = _public_deployment({**response_state, "run_id": run_id})
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

    with _app._deploy_lock(run_id):
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
        checkpoint_step, is_checkpoint, prefix = _resolve_deployable_target(
            run_id,
            effective_spec,
            status,
            payload.get("checkpoint_id"),
            action="export",
            enforce_state=True,
        )
        subfolder = f"{prefix}/adapter"
        try:
            url = _app.export_adapter(
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
            status = _app.get_status(row["run_id"])
        except FileNotFoundError:
            continue
        if scope is not None and not _in_deployment_listing_scope(status, *scope):
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
def chat(
    run_id: str,
    payload: dict,
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    return managed_chat(
        run_id,
        payload,
        key,
        x_freesolo_org_id,
        x_freesolo_project_id,
    )


from flash.server.routes.serving_chat import (  # noqa: E402,F401
    _upstream_response_headers,
    _UpstreamStreamingResponse,
    managed_chat,
)

# re-exported at the bottom rather than imported at the top: both modules resolve names back
# through this one, so a top import would be circular. the server tests address these helpers as
# attributes of `flash.server.routes.serving`, which is why they stay on it after the move.
# imported at the bottom because `serving_completion` reads the small helpers defined above: by the
# time it runs, this module is initialised far enough to satisfy it. re-exported because
# `app.py` and the server tests reach these through `serving.<name>`, and the tests patch
# `serving._finish_deployment_unlocked`, which `_finish_deployment` resolves as a global at call
# time -- so the seam survives the move.
from flash.server.routes.serving_completion import (  # noqa: E402,F401
    _commit_ready_deployment,
    _finish_deployment_unlocked,
    _record_deployment_failure,
    recover_deployments,
    replay_status_reports,
)
from flash.server.routes.serving_revisions import (  # noqa: E402,F401
    _DEPLOYMENT_BUSY_STATES,
    _DEPLOYMENT_READY_STATES,
    _authorized_chat_checkpoint,
    _chat_messages_from_payload,
    _format_deployed_steps,
    _managed_chat_messages,
    _parse_checkpoint_step,
    _resolve_deploy_step,
    _resolve_deployable_target,
    _spec_is_unservable,
    _verified_checkpoints,
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
