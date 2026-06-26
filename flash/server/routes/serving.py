"""Serving endpoints: deploy / undeploy an adapter, list deployments, and chat.

Service functions that the test-suite monkeypatches (``deploy_adapter``, ``undeploy_adapter``,
``serve_chat``/``serve_chat_stream``, ``list_checkpoints``, ``get_status``) plus the per-run
deploy lock and the deployable-state sets are resolved through the ``flash.server.app`` module
(``_app.<name>``) at call time, so patching ``app.<name>`` is honored here.
"""

from __future__ import annotations

import contextlib
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from flash.runner import (
    adapter_prefix,
    attach_checkpoint_deployment,
    mark_deployed,
    mark_undeployed,
)
from flash.runner.checkpoints import checkpoint_adapter_prefix
from flash.serve.deploy import ServingError
from flash.server import app as _app
from flash.server import db
from flash.server._deps import owned_run, require_key
from flash.spec import JobSpec

router = APIRouter()


def _resolve_deploy_step(run_id: str, spec, raw_step) -> int | None:
    """Validate an optional deploy ``step`` against the run's published checkpoints.

    Returns the integer step to deploy, or ``None`` when no step was requested (deploy the
    final adapter). Raises ``HTTPException(400)`` for a malformed step and ``HTTPException(404)``
    — listing the available steps — when the run has no deployable checkpoint at that step."""
    if raw_step is None:
        return None

    # Accept only an actual integer step — NOT a bool (True would coerce to step 1) and not a
    # non-integer float/string (40.9 / "40.9" must not silently round to a different checkpoint).
    want: int | None = None
    if isinstance(raw_step, bool):
        want = None
    elif isinstance(raw_step, int):
        want = raw_step
    elif isinstance(raw_step, float):
        want = int(raw_step) if raw_step.is_integer() else None
    elif isinstance(raw_step, str) and raw_step.strip().lstrip("-").isdigit():
        want = int(raw_step.strip())
    # Checkpoint steps are always non-negative (derived from ``step-<N>``); reject a negative
    # value as malformed (400) rather than letting it fall through to the 404 below.
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


@router.post("/v1/runs/{run_id}/deploy")
def deploy(run_id: str, key: Annotated[dict, Depends(require_key)], payload: dict | None = None):
    payload = payload or {}
    # Serialize deploy vs undeploy (and a second deploy) for this run: registration
    # with the freesolo serving app runs outside the status lock, so without this they
    # could interleave and leave the serving record and the control plane inconsistent.
    with _app._deploy_lock(run_id):
        status = owned_run(run_id, key)
        spec = JobSpec.from_dict(status.spec)
        # Validate ``dry_run`` is an actual JSON boolean — never ``bool(...)`` a truthy non-bool
        # (e.g. the string "false" would coerce to True and silently change deploy behavior).
        dry_run_raw = payload.get("dry_run", False)
        if not isinstance(dry_run_raw, bool):
            raise HTTPException(status_code=400, detail="dry_run must be a boolean")
        dry_run = dry_run_raw
        # Optional `step`: deploy a specific intermediate checkpoint instead of the run's
        # final adapter. We resolve it against what's actually on HF (the source of truth),
        # so a missing step 404s with the available list rather than 500ing at serve time.
        checkpoint_step = _resolve_deploy_step(run_id, spec, payload.get("step"))
        is_checkpoint = checkpoint_step is not None
        allowed_states = (
            _app._CHECKPOINT_DEPLOYABLE_STATES if is_checkpoint else _app._DEPLOYABLE_STATES
        )
        if not dry_run and status.state not in allowed_states:
            detail = (
                f"run {run_id} is {status.state!r}; deploy a checkpoint only once the run "
                "has finished or been cancelled"
                if is_checkpoint
                else f"run {run_id} is {status.state!r}; only finished runs with "
                "trained adapter artifacts can be deployed"
            )
            raise HTTPException(status_code=409, detail=detail)
        # Legacy runs persisted before [train].hf_repo was mandatory rehydrate with an
        # empty hf_repo; without this guard freesolo serving cannot locate the adapter
        # artifacts (the per-run HF dataset repo). Reject early with a clear 409.
        if not dry_run and not spec.train.hf_repo:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} has no [train].hf_repo (legacy run); its adapter artifacts "
                    "cannot be located, so it cannot be deployed"
                ),
            )
        # A checkpoint deploy serves the per-step adapter; otherwise the run's final adapter.
        deploy_prefix = (
            checkpoint_adapter_prefix(spec, checkpoint_step)
            if is_checkpoint
            else adapter_prefix(spec)
        )
        # The state the run must still be in for this deploy to finalize — a CAS guard so
        # a /cancel (NOT serialized by the deploy lock) that terminalized the run can't be
        # silently overwritten by the deployment record.
        prev_state = status.state
        # Attribute the adapter to the RUN's owning org so serving can authorize external chat
        # by org. Prefer the org persisted WITH the run — billing_context for user runs,
        # platform_context for internal/operator runs (see submit path) — over the caller's key,
        # so an operator deploy still lands on the run's owner. Each context is isinstance-guarded
        # against a non-dict legacy value (mirrors flash/server/billing.py / checkpoints.py).
        def _run_org(*contexts) -> str:
            for ctx in contexts:
                if isinstance(ctx, dict):
                    org = str(ctx.get("org_id") or "").strip()
                    if org:
                        return org
            return ""

        deploy_org_id = (
            _run_org(
                getattr(status, "billing_context", None),
                getattr(status, "platform_context", None),
            )
            or str(key.get("org_id") or "").strip()
            or None
        )
        try:
            dep = _app.deploy_adapter(
                run_id=run_id,
                model=spec.model,
                hf_repo=spec.train.hf_repo,
                adapter_prefix=deploy_prefix,
                gpu_name=spec.gpu.type,
                dry_run=dry_run,
                # a run trained with thinking serves with thinking (per-run parity)
                thinking=spec.thinking,
                org_id=deploy_org_id,
            )
        except ServingError as exc:
            # The serving backend rejected the registration or was unreachable. This is an
            # upstream/gateway failure, not a flash bug, so surface a clean 502 with the
            # real reason instead of letting httpx escape as an unhandled 500 + traceback.
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            raise
        dep_dict = dep.to_dict()
        if is_checkpoint:
            dep_dict["checkpoint_step"] = checkpoint_step
        if not dry_run:
            if is_checkpoint and status.state not in _app._DEPLOYABLE_STATES:
                # Deploying a checkpoint of a run that stopped mid-RL (cancelled/failed):
                # attach the serving deployment but KEEP the run's terminal training state
                # — flipping it to `deployed` would erase the outcome and make undeploy
                # wrongly restore it to `done`.
                attach_checkpoint_deployment(run_id, dep_dict)
            else:
                # Record the deployment. The CAS no-ops only if a /cancel raced finalization
                # — then the adapter we just registered is orphaned, so deregister it and
                # report the conflict instead of a bogus 200.
                marked = mark_deployed(run_id, dep_dict, expect_state=prev_state)
                if marked.state != "deployed":
                    with contextlib.suppress(Exception):
                        _app.undeploy_adapter(run_id)
                    raise HTTPException(
                        status_code=409,
                        detail=f"run {run_id} became {marked.state!r} during deploy; aborted",
                    )
        return dep_dict


@router.delete("/v1/runs/{run_id}/deploy")
def undeploy(run_id: str, key: Annotated[dict, Depends(require_key)]):
    # Same per-run lock as deploy: an undeploy must not interleave with an in-flight
    # deploy's provisioning/finalization.
    with _app._deploy_lock(run_id):
        status = owned_run(run_id, key)
        try:
            deleted = _app.undeploy_adapter(run_id)
        except ServingError as exc:
            # A serving-backend failure (unreachable / non-404 error) is an upstream/gateway
            # problem, not a flash bug — surface a clean 502 with the real reason (mirrors the
            # deploy handler) instead of letting the ServingError escape as an unhandled 500.
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        # Delete is idempotent: a missing serving-side adapter still means the local
        # deployment record can be cleared.
        if status.deployment:
            mark_undeployed(run_id)
        return {"run_id": run_id, "deleted_endpoints": deleted}


@router.post("/v1/runs/{run_id}/export")
def export(run_id: str, key: Annotated[dict, Depends(require_key)], payload: dict | None = None):
    """Copy a run's trained adapter into a user-owned HuggingFace repo.

    Reads the adapter from the run's private artifact repo with the operator token and
    re-uploads it to ``repository`` with the user-supplied ``hf_token`` (write access to their
    own repo). ``step`` selects an intermediate checkpoint — resolved against the same published
    checkpoints as ``flash deploy --step`` — instead of the run's final adapter.
    """
    payload = payload or {}
    repository = str(payload.get("repository") or "").strip()
    if not repository:
        raise HTTPException(
            status_code=400,
            detail="repository is required: the destination HuggingFace repo 'owner/name'",
        )
    if "/" not in repository.strip("/"):
        raise HTTPException(
            status_code=400,
            detail=f"repository must be a HuggingFace repo of the form 'owner/name', got {repository!r}",
        )
    hf_token = str(payload.get("hf_token") or "").strip()
    if not hf_token:
        raise HTTPException(
            status_code=400,
            detail="hf_token is required: a HuggingFace token with write access to the destination repo",
        )
    # Validate ``private`` is an actual JSON boolean (mirrors deploy's ``dry_run`` guard): a
    # truthy non-bool like "false" must not silently flip the destination's visibility.
    private = payload.get("private", True)
    if not isinstance(private, bool):
        raise HTTPException(status_code=400, detail="private must be a boolean")

    status = owned_run(run_id, key)
    spec = JobSpec.from_dict(status.spec)
    # Legacy runs with no artifact repo (mirrors the /deploy guard): the adapter can't be located.
    if not spec.train.hf_repo:
        raise HTTPException(
            status_code=409,
            detail=(
                f"run {run_id} has no [train].hf_repo (legacy run); its adapter artifacts "
                "cannot be located, so it cannot be exported"
            ),
        )
    # Optional `step`: export a specific intermediate checkpoint instead of the final adapter,
    # validated against what's actually on HF (a missing step 404s with the available list).
    checkpoint_step = _resolve_deploy_step(run_id, spec, payload.get("step"))
    is_checkpoint = checkpoint_step is not None
    # Same state gate as deploy: a final adapter exists only once a run finished; a per-step
    # checkpoint also survives a run that stopped mid-RL (cancelled/failed).
    allowed_states = (
        _app._CHECKPOINT_DEPLOYABLE_STATES if is_checkpoint else _app._DEPLOYABLE_STATES
    )
    if status.state not in allowed_states:
        detail = (
            f"run {run_id} is {status.state!r}; export a checkpoint only once the run "
            "has finished or been cancelled"
            if is_checkpoint
            else f"run {run_id} is {status.state!r}; only finished runs with "
            "trained adapter artifacts can be exported"
        )
        raise HTTPException(status_code=409, detail=detail)
    # The per-step adapter folder for a checkpoint, otherwise the run's final adapter folder.
    prefix = (
        checkpoint_adapter_prefix(spec, checkpoint_step)
        if is_checkpoint
        else adapter_prefix(spec)
    )
    subfolder = f"{prefix}/adapter"
    try:
        url = _app.export_adapter(
            source_repo=spec.train.hf_repo,
            source_subfolder=subfolder,
            dest_repo=repository,
            dest_token=hf_token,
            private=private,
        )
    except ValueError as exc:
        # The source has no adapter artifacts at that path — nothing to export.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ServingError as exc:
        # An HF transport/permission failure (download or upload) — an upstream problem, not a
        # flash bug, so surface a clean 502 with the real reason (mirrors deploy/undeploy).
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
    status = owned_run(run_id, key)
    spec = JobSpec.from_dict(status.spec)
    deployment = status.deployment or {}
    # A cancelled run's serve endpoint was torn down at cancel time; never let a
    # chat recreate it (closes the window before cancel marks the deployment
    # inactive, and covers a teardown that deleted nothing).
    if status.state == "cancelled":
        raise HTTPException(
            status_code=409, detail=f"run {run_id} was cancelled; redeploy is not allowed"
        )
    # Chat must ride an explicit deployment (with its cost controls), not
    # implicitly provision a serving endpoint that /v1/deployments cannot see.
    if deployment.get("state") in (None, "undeployed", "dry_run"):
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} has no active deployment; `flash deploy {run_id}` first",
        )
    # Legacy run with no artifact repo (mirrors the /deploy guard): a run that never had a
    # [train].hf_repo was never registered with freesolo serving, so reject early with a
    # clear 409 instead of an opaque downstream inference error.
    if not spec.train.hf_repo:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} has no [train].hf_repo (legacy run); its adapter cannot be served",
        )
    # Parse the client-supplied sampling params BEFORE the broad try: a bad value
    # (e.g. {"temperature": "hot"}) is a request error -> 400, not a 502 inference
    # failure (which would misclassify the bad payload as an upstream serving fault).
    try:
        temperature = float(payload.get("temperature") or 0.0)
        # Default ONLY when max_tokens is missing/None — `or 512` would silently turn an
        # explicit 0 into 512, surprising callers and inflating cost/latency. An explicit
        # non-positive value is rejected below as a request error rather than masked.
        raw_max_tokens = payload.get("max_tokens")
        max_tokens = 512 if raw_max_tokens is None else int(raw_max_tokens)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid temperature/max_tokens: {exc}"
        ) from exc
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
                    messages=payload.get("messages") or [],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # a run trained with thinking serves with thinking (per-run parity)
                    thinking=spec.thinking,
                ),
                media_type="text/plain; charset=utf-8",
            )
        return _app.serve_chat(
            run_id=run_id,
            messages=payload.get("messages") or [],
            temperature=temperature,
            max_tokens=max_tokens,
            # a run trained with thinking serves with thinking (per-run parity)
            thinking=spec.thinking,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"inference failure: {exc}") from exc
