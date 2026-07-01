"""Serving endpoints: deploy / undeploy an adapter, list deployments, and chat.

Service functions are resolved through ``flash.server.app`` at call time so test-suite patches on
``app.<name>`` are honored here.
"""

from __future__ import annotations

import contextlib
import math
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from flash.runner import (
    adapter_prefix,
    mark_checkpoint_deployed,
    mark_deployed,
    mark_undeployed,
)
from flash.runner.checkpoints import checkpoint_adapter_prefix
from flash.serve.deploy import ServingError
from flash.server import app as _app
from flash.server import db
from flash.server._deps import _require_bool, owned_run, require_key
from flash.server._internal_client import run_org_id
from flash.spec import JobSpec

router = APIRouter()

def _chat_messages_from_payload(payload: dict) -> list[dict]:
    raw = payload.get("messages")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="messages must be a list")
    for index, message in enumerate(raw):
        if not isinstance(message, dict):
            raise HTTPException(
                status_code=400,
                detail=f"messages[{index}] must be a chat message object",
            )
    return raw


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


def _resolve_deploy_step(run_id: str, spec, raw_step) -> int | None:
    """Validate optional checkpoint step; returns int or None (final adapter). 400 on bad step, 404 on missing."""
    if raw_step is None:
        return None

    # Reject bool (True -> step 1) and non-integer floats; str path uses fullmatch not isdigit
    # (isdigit accepts unicode digits + "-5" which crash int() -> 500). The length bound also keeps a
    # 4301+ digit string from tripping Python's int-string-conversion limit (another int() -> 500).
    want: int | None = None
    if isinstance(raw_step, bool):
        want = None
    elif isinstance(raw_step, int):
        want = raw_step
    elif isinstance(raw_step, float):
        want = int(raw_step) if raw_step.is_integer() else None
    elif isinstance(raw_step, str):
        s = raw_step.strip()
        want = int(s) if re.fullmatch(r"-?[0-9]{1,18}", s) else None
    if want is not None and want < 0:
        want = None
    if want is None:
        raise HTTPException(status_code=400, detail=f"invalid checkpoint step: {raw_step!r}")
    checkpoints = _app.list_checkpoints(spec)
    if any(c["step"] == want for c in checkpoints):
        return want
    available = ", ".join(str(c["step"]) for c in checkpoints) or "none"
    raise HTTPException(
        status_code=404,
        detail=f"run {run_id} has no deployable checkpoint at step {want} (available: {available})",
    )


def _resolve_deployable_target(
    run_id: str, spec, status, raw_step, *, action: str, enforce_state: bool
) -> tuple[int | None, bool, str]:
    """Resolve the deploy/export target and gate final-adapter targets on training state."""
    checkpoint_step = _resolve_deploy_step(run_id, spec, raw_step)
    is_checkpoint = checkpoint_step is not None
    # A resolved checkpoint step has already proven a servable adapter exists; only final-adapter
    # deploy/export needs the run-state gate because the final adapter exists only after completion.
    if enforce_state and is_checkpoint and status.state == "dry_run":
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} is 'dry_run'; dry-run runs cannot be {action}ed",
        )
    if enforce_state and not is_checkpoint and status.state not in _app._DEPLOYABLE_STATES:
        detail = (
            f"run {run_id} is {status.state!r}; only finished runs with "
            f"trained adapter artifacts can be {'deployed' if action == 'deploy' else 'exported'}"
        )
        raise HTTPException(status_code=409, detail=detail)
    prefix = (
        checkpoint_adapter_prefix(spec, checkpoint_step) if is_checkpoint else adapter_prefix(spec)
    )
    return checkpoint_step, is_checkpoint, prefix


@router.post("/v1/runs/{run_id}/deploy")
def deploy(run_id: str, key: Annotated[dict, Depends(require_key)], payload: dict | None = None):
    payload = payload or {}
    with _app._deploy_lock(run_id):
        status = owned_run(run_id, key)
        spec = JobSpec.from_dict(status.spec)
        dry_run = _require_bool(payload, "dry_run", False)
        checkpoint_step, is_checkpoint, deploy_prefix = _resolve_deployable_target(
            run_id, spec, status, payload.get("step"), action="deploy", enforce_state=not dry_run
        )
        if not dry_run and not spec.train.hf_repo:
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
        try:
            dep = _app.deploy_adapter(
                run_id=run_id,
                model=spec.model,
                hf_repo=spec.train.hf_repo,
                adapter_prefix=deploy_prefix,
                gpu_name=spec.gpu.type,
                dry_run=dry_run,
                lora_rank=spec.train.lora_rank,
                # a run trained with thinking serves with thinking (per-run parity)
                thinking=spec.thinking,
                org_id=deploy_org_id,
            )
        except ServingError as exc:
            # A run-level deploy targets <prefix>/adapter, which only exists once finalize
            # completed; a cancelled/preempted/interrupted run may carry only per-step
            # checkpoints/step-N/adapter snapshots. Reading the missing run-level config
            # surfaces as "failed to read ...adapter_config.json" — turn that into actionable
            # guidance (deploy --step N) instead of an opaque 502 rank-verification error.
            if not is_checkpoint and "failed to read" in str(exc):
                steps = [c["step"] for c in _app.list_checkpoints(spec)]
                if steps:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"run {run_id} has no run-level adapter at "
                            f"{deploy_prefix}/adapter (the run likely never finalized); "
                            f"deploy a saved checkpoint instead, e.g. `flash deploy "
                            f"{run_id} --step {steps[-1]}` (available steps: "
                            f"{', '.join(str(s) for s in steps)})"
                        ),
                    ) from exc
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise
        dep_dict = dep.to_dict()
        if is_checkpoint:
            dep_dict["checkpoint_step"] = checkpoint_step
        if not dry_run:
            state_guard = prev_state
            if is_checkpoint:
                state_guard = prev_state if prev_state in _app._DEPLOYABLE_STATES else None
                marked = mark_checkpoint_deployed(run_id, dep_dict, expect_state=state_guard)
            else:
                marked = mark_deployed(run_id, dep_dict, expect_state=prev_state)
            # CAS: if /cancel or /undeploy raced us, the adapter is orphaned — deregister and 409.
            cas_failed = (
                marked.deployment != dep_dict if is_checkpoint else marked.state != "deployed"
            )
            if state_guard is not None and cas_failed:
                with contextlib.suppress(Exception):
                    _app.undeploy_adapter(run_id)
                raise HTTPException(
                    status_code=409,
                    detail=f"run {run_id} became {marked.state!r} during deploy; aborted",
                )
        return dep_dict


@router.delete("/v1/runs/{run_id}/deploy")
def undeploy(run_id: str, key: Annotated[dict, Depends(require_key)]):
    with _app._deploy_lock(run_id):
        status = owned_run(run_id, key)
        try:
            deleted = _app.undeploy_adapter(run_id)
        except ServingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        # Idempotent: clear local record even if serving side already had no adapter.
        if status.deployment:
            mark_undeployed(run_id)
        return {"run_id": run_id, "deleted_endpoints": deleted}


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
        if not spec.train.hf_repo:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} has no [train].hf_repo; its adapter artifacts "
                    "cannot be located, so it cannot be exported"
                ),
            )
        checkpoint_step, is_checkpoint, prefix = _resolve_deployable_target(
            run_id, spec, status, payload.get("step"), action="export", enforce_state=True
        )
        subfolder = f"{prefix}/adapter"
        try:
            url = _app.export_adapter(
                source_repo=spec.train.hf_repo,
                source_subfolder=subfolder,
                dest_repo=repository,
                dest_token=hf_token,
                private=private,
                base_model=spec.model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ServingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    result = {
        "run_id": run_id,
        "adapter_id": run_id,
        "repository": repository,
        "url": url,
        "source": f"{spec.train.hf_repo}:{subfolder}",
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
            out.append(status.to_dict())
    return {"deployments": out}


@router.post("/v1/runs/{run_id}/chat")
def chat(run_id: str, payload: dict, key: Annotated[dict, Depends(require_key)]):
    messages = _chat_messages_from_payload(payload)
    status = owned_run(run_id, key)
    spec = JobSpec.from_dict(status.spec)
    deployment = status.deployment or {}
    has_active_deploy = deployment.get("state") not in (None, "undeployed", "dry_run")
    # A cancelled run can still serve a per-step checkpoint it deployed: checkpoint deploy records
    # a live adapter that /v1/deployments lists as active without requiring a final adapter.
    # Only block chat when there's no active deployment to serve.
    if status.state == "cancelled" and not has_active_deploy:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} was cancelled; deploy a checkpoint with "
            f"`flash deploy {run_id} --step <N>` first",
        )
    if not has_active_deploy:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} has no active deployment; `flash deploy {run_id}` first",
        )
    if not spec.train.hf_repo:
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
    try:
        if payload.get("stream") is True:
            return StreamingResponse(
                _app.serve_chat_stream(
                    run_id=run_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking=spec.thinking,
                ),
                media_type="text/plain; charset=utf-8",
            )
        return _app.serve_chat(
            run_id=run_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=spec.thinking,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"inference failure: {exc}") from exc
