"""Environment publishing endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response

from flash._internal.logging import get_logger
from flash.server.platform.deps import require_key

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
    from flash.server.domain.registry import envs
    from flash.server.domain.registry.projects import require_project_access_slug

    project_raw = payload.get("project_id")
    if not isinstance(project_raw, str):
        raise HTTPException(status_code=400, detail="project_id is required and must be a string")
    project_id, project_slug = require_project_access_slug(
        project_id=project_raw,
        key=key,
        authorization=authorization,
        org_id=x_freesolo_org_id,
    )

    # Use `if x is None` not `x or ""` so non-string falsy values reach publish_package's type checks.
    _pkg = payload.get("package_b64")
    _name = payload.get("name")
    from flash.server.domain.registry.environment_registry import record_published_environment

    resolved_key = {**key, "org_id": key.get("org_id") or x_freesolo_org_id}

    try:
        slug = envs.publish_package(
            package_b64="" if _pkg is None else _pkg,
            name="" if _name is None else _name,
            key=key,
            project_slug=project_slug,
        )
    except envs.EnvPublishError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    try:
        recorded = record_published_environment(
            slug=slug,
            name="" if _name is None else _name,
            key=resolved_key,
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


@router.get("/v1/envs")
def list_envs(key: Annotated[dict, Depends(require_key)]):
    """List the published environments the caller's org owns.

    The hub is the source of truth that ``publish_env`` writes and ``download_env_package`` reads,
    so it is what this enumerates -- not the platform metadata mirror, which is best-effort on both
    publish and delete and would under-report a package that is genuinely there.

    No project filter: the hub slug is ``<org-slug>/<name>`` and carries no project, so filtering
    here would require trusting the mirror this deliberately does not read. A caller that wants one
    project's environments should read the mirror through the web UI.
    """
    from flash.server.domain.registry import envs

    try:
        slugs = envs.list_namespace_slugs(key=key)
    except envs.EnvPublishError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    return {"environments": [{"id": slug} for slug in slugs]}


@router.get("/v1/envs/{env_id:path}/package")
def download_env_package(env_id: str, key: Annotated[dict, Depends(require_key)]):
    """Download a managed Freesolo environment package from the GitHub-backed hub."""
    from flash.server.domain.registry import envs

    try:
        env_id = envs.canonical_env_id(env_id)
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
    # ``namespace/project/name`` slug and carries a slash, so the route uses the ``:path`` converter.
    # Authorization (own-namespace for user keys, any for the internal key) lives in
    # ``delete_package`` so it can't be bypassed.
    from flash.core.spec import require_project_id
    from flash.server.domain.registry import envs
    from flash.server.domain.registry.environment_registry import (
        record_deleted_environment,
        require_environment_project,
    )
    from flash.server.domain.registry.projects import require_project_access

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
    try:
        require_environment_project(
            slug=env_id,
            project_id=project_id,
            key=key,
            org_id=x_freesolo_org_id,
        )
    except HTTPException as validation_exc:
        if validation_exc.status_code != 404:
            raise
        if key.get("auth_kind") != "internal":
            try:
                caller_namespace = envs.namespace_for(key)
            except envs.EnvPublishError as namespace_exc:
                raise validation_exc from namespace_exc
            if env_id.split("/", 1)[0] != caller_namespace:
                raise validation_exc
        try:
            envs.download_package(slug=env_id, key=key)
        except envs.EnvPublishError as package_exc:
            # only a genuinely absent package is an idempotent no-op; a hub outage must keep its
            # own status so callers retry instead of reading a masked 404 as already-deleted.
            if package_exc.status != 404:
                raise HTTPException(
                    status_code=package_exc.status, detail=str(package_exc)
                ) from package_exc
            deleted = False
        else:
            raise validation_exc
    else:
        try:
            deleted = envs.delete_package(slug=env_id, key=key)
        except envs.EnvPublishError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    # github (the package store) is the source of truth and is already updated; dropping the
    # web-ui metadata mirror remains best-effort and must never turn a successful delete into a 500.
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
