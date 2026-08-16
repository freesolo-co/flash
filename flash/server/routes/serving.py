"""Serving endpoints: deploy / undeploy an adapter, list deployments, and chat.

Transport only. Every one of these handlers translates the request into a command for
``DeploymentService``, and translates what comes back -- a value or a ``DeploymentError`` -- into an
HTTP response. The orchestration itself lives in ``flash.server.domain.deployments``.
"""

from __future__ import annotations

import contextlib
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse

from flash.core.spec import require_project_id
from flash.server.domain.deployment_ports import (
    DeploymentConflict,
    DeploymentError,
    DeploymentNotFound,
    DeploymentUnavailable,
    DeploymentUpstreamError,
    InvalidDeploymentRequest,
)
from flash.server.domain.deployments import (
    CallerContext,
    ChatCommand,
    DeployCommand,
    DeploymentService,
    ExportCommand,
)
from flash.server.platform import auth
from flash.server.platform.deps import require_key

router = APIRouter()

_STATUS_BY_ERROR: tuple[tuple[type[DeploymentError], int], ...] = (
    (InvalidDeploymentRequest, 400),
    (DeploymentNotFound, 404),
    (DeploymentConflict, 409),
    (DeploymentUpstreamError, 502),
    (DeploymentUnavailable, 503),
)


def _http_error(exc: DeploymentError) -> HTTPException:
    """The single place a domain refusal becomes a status code."""
    for error_type, status_code in _STATUS_BY_ERROR:
        if isinstance(exc, error_type):
            return HTTPException(status_code=status_code, detail=exc.detail)
    return HTTPException(status_code=500, detail=str(exc))


def _service(request: Request) -> DeploymentService:
    """The service this app was built with (a test may inject its own)."""
    return request.app.state.deployment_service


def _caller(key: dict, org_id: str | None = None, project_id: str | None = None) -> CallerContext:
    return CallerContext(key=key, org_id=org_id, project_id=project_id)


@router.get("/v1/runs/{run_id}/deploy")
def deployment(
    request: Request,
    run_id: str,
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    try:
        return _service(request).get_deployment(
            run_id, _caller(key, x_freesolo_org_id, x_freesolo_project_id)
        )
    except DeploymentError as exc:
        raise _http_error(exc) from exc


@router.post("/v1/runs/{run_id}/deploy")
def deploy(
    request: Request,
    run_id: str,
    key: Annotated[dict, Depends(require_key)],
    payload: dict | None = None,
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    command = DeployCommand(
        run_id=run_id,
        caller=_caller(key, x_freesolo_org_id, x_freesolo_project_id),
        payload=payload or {},
    )
    try:
        return _service(request).deploy(command)
    except DeploymentError as exc:
        raise _http_error(exc) from exc


@router.delete("/v1/runs/{run_id}/deploy")
def undeploy(
    request: Request,
    run_id: str,
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    try:
        return _service(request).undeploy(
            run_id, _caller(key, x_freesolo_org_id, x_freesolo_project_id)
        )
    except DeploymentError as exc:
        raise _http_error(exc) from exc


@router.post("/v1/runs/{run_id}/export")
def export(
    request: Request,
    run_id: str,
    key: Annotated[dict, Depends(require_key)],
    payload: dict | None = None,
):
    """Copy a run's trained adapter into a user-owned HuggingFace repo."""
    command = ExportCommand(run_id=run_id, caller=_caller(key), payload=payload or {})
    try:
        result = _service(request).export(command)
    except DeploymentError as exc:
        raise _http_error(exc) from exc
    # best-effort product-analytics report: exports never otherwise touch the
    # platform backend (the copy is hf-to-hf inside flash).
    with contextlib.suppress(Exception):
        from flash.server.domain.run_registry import record_model_exported

        record_model_exported(
            status=result.status,
            key=key,
            repository=result.repository,
            url=result.url,
            step=result.checkpoint_step if result.is_checkpoint else None,
        )
    return result.to_dict()


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


@router.get("/v1/deployments")
def deployments(
    request: Request,
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    scope = _deployment_listing_scope(key, x_freesolo_org_id, x_freesolo_project_id)
    try:
        return {"deployments": _service(request).list_deployments(_caller(key), scope=scope)}
    except DeploymentError as exc:
        raise _http_error(exc) from exc


@router.post("/v1/runs/{run_id}/chat")
def chat(request: Request, run_id: str, payload: dict, key: Annotated[dict, Depends(require_key)]):
    service = _service(request)
    try:
        plan = service.plan_chat(ChatCommand(run_id=run_id, caller=_caller(key), payload=payload))
    except DeploymentError as exc:
        raise _http_error(exc) from exc
    try:
        if plan.stream:
            # chat_stream sends the upstream request and validates its status at call time, so an
            # upstream 4xx/5xx raises here, inside the try, and becomes a real 502 before the 200
            # headers are flushed. a failure after the first byte propagates out of the body
            # iterator instead, which aborts the chunked response so the client cannot mistake the
            # truncation for a finished answer.
            return StreamingResponse(
                service.chat_stream(plan), media_type="text/plain; charset=utf-8"
            )
        return service.chat(plan)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"inference failure: {exc}") from exc
