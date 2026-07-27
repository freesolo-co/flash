"""Environment publishing endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response

from flash._logging import get_logger
from flash.server._deps import require_key

logger = get_logger("flash.server.routes.envs")
router = APIRouter()

_PUBLISH_ASSOCIATION_FAILURE = (
    "environment package may already be uploaded, but its project association could not be "
    "recorded; retry the same publish to repair the association"
)


@router.post("/v1/envs")
def publish_env(
    payload: dict,
    key: Annotated[dict, Depends(require_key)],
    authorization: Annotated[str | None, Header()] = None,
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
):
    from flash.server import envs
    from flash.server.projects import require_project_access

    project_raw = payload.get("project_id")
    if not isinstance(project_raw, str):
        raise HTTPException(status_code=400, detail="project_id is required and must be a string")
    project_id = require_project_access(
        project_id=project_raw,
        key=key,
        authorization=authorization,
        org_id=x_freesolo_org_id,
    )

    # Use `if x is None` not `x or ""` so non-string falsy values reach publish_package's type checks.
    _pkg = payload.get("package_b64")
    _name = payload.get("name")
    try:
        slug = envs.publish_package(
            package_b64="" if _pkg is None else _pkg,
            name="" if _name is None else _name,
            key=key,
        )
    except envs.EnvPublishError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    from flash.server.environment_registry import record_published_environment

    try:
        recorded = record_published_environment(
            slug=slug,
            name="" if _name is None else _name,
            key={**key, "org_id": key.get("org_id") or x_freesolo_org_id},
            project_id=project_id,
        )
    except Exception as exc:
        logger.warning(
            "record_published_environment failed after package upload: %s", exc, exc_info=True
        )
        raise HTTPException(status_code=502, detail=_PUBLISH_ASSOCIATION_FAILURE) from exc
    if recorded is not True:
        raise HTTPException(status_code=502, detail=_PUBLISH_ASSOCIATION_FAILURE)
    return {"id": slug}


@router.get("/v1/envs/{env_id:path}/package")
def download_env_package(
    env_id: str,
    key: Annotated[dict, Depends(require_key)],
    authorization: Annotated[str | None, Header()] = None,
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    """Download one project's managed Freesolo environment package."""
    import base64

    from flash.server import envs
    from flash.server.environment_registry import resolve_environment_package_source
    from flash.server.projects import require_project_access
    from flash.spec import require_project_id

    try:
        project_id = require_project_id(x_freesolo_project_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc).replace("project", "X-Freesolo-Project-Id", 1),
        ) from exc
    project_id = require_project_access(
        project_id=project_id,
        key=key,
        authorization=authorization,
        org_id=x_freesolo_org_id,
    )
    try:
        env_id = envs.canonical_env_id(env_id)
    except envs.EnvPublishError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    source = resolve_environment_package_source(
        slug=env_id,
        project_id=project_id,
        key=key,
        org_id=x_freesolo_org_id,
    )
    if source["source_kind"] == "builtin":
        package = base64.b64decode(source["package_base64"], validate=True)
    else:
        try:
            package = envs.download_package(slug=env_id, key=key)
        except envs.EnvPublishError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    return Response(
        content=package,
        media_type="application/gzip",
        headers={
            "Content-Disposition": 'attachment; filename="package.tar.gz"',
            "Content-Length": str(len(package)),
        },
    )


@router.delete("/v1/envs/{env_id:path}")
def delete_env(
    env_id: str,
    key: Annotated[dict, Depends(require_key)],
    authorization: Annotated[str | None, Header()] = None,
    # the org a metadata-mirror drop targets. user keys carry their own org, so this is only
    # consulted for the internal key (which is org-agnostic): the web ui delete authenticates
    # the user, resolves their org, and passes it here. a user key never honors this header
    # (see record_deleted_environment), so a forged value can't delete another org's row.
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    # Delete a published Freesolo environment package from the managed hub. ``env_id`` is the
    # ``namespace/name`` slug and carries a slash, so the route uses the ``:path`` converter.
    # Authorization (own-namespace for user keys, any for the internal key) lives in
    # ``delete_package`` so it can't be bypassed.
    from flash.server import envs
    from flash.server.environment_registry import (
        record_deleted_environment,
        require_environment_project,
        resolve_environment_source_kind,
    )
    from flash.server.projects import require_project_access
    from flash.spec import require_project_id

    try:
        project_id = require_project_id(x_freesolo_project_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc).replace("project", "X-Freesolo-Project-Id", 1),
        ) from exc
    project_id = require_project_access(
        project_id=project_id,
        key=key,
        authorization=authorization,
        org_id=x_freesolo_org_id,
    )

    # normalize once before the strict mirror lookup so every downstream operation uses the same id.
    try:
        env_id = envs.canonical_env_id(env_id)
    except envs.EnvPublishError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    require_environment_project(
        slug=env_id,
        project_id=project_id,
        key=key,
        org_id=x_freesolo_org_id,
    )
    source_kind = resolve_environment_source_kind(
        slug=env_id,
        project_id=project_id,
        key=key,
        org_id=x_freesolo_org_id,
    )
    if source_kind == "builtin":
        try:
            deleted = record_deleted_environment(
                slug=env_id,
                project_id=project_id,
                key=key,
                org_id=x_freesolo_org_id,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Freesolo built-in environment deletion failed",
            ) from exc
        if not deleted:
            raise HTTPException(
                status_code=502,
                detail="Freesolo built-in environment deletion failed",
            )
        return {"id": env_id, "deleted": True}

    try:
        deleted = envs.delete_package(slug=env_id, key=key)
    except envs.EnvPublishError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    # github is authoritative for hub packages and is already updated; dropping the web-ui
    # metadata mirror remains best-effort and must not turn a successful hub delete into a 500.
    try:
        record_deleted_environment(
            slug=env_id,
            project_id=project_id,
            key=key,
            org_id=x_freesolo_org_id,
        )
    except Exception as exc:
        logger.warning("record_deleted_environment failed (non-fatal): %s", exc, exc_info=True)
    return {"id": env_id, "deleted": deleted}
