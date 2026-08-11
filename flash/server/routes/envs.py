"""Environment publishing endpoint."""

from __future__ import annotations

from contextlib import suppress
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


def _name_conflict_detail(slug: str) -> str:
    """Explain a cross-project name collision and how to actually resolve it.

    Deliberately says none of what the old shared message said: the association was not
    "not recorded", it was refused; and a retry reproduces this identically, so advising one
    sends the user in a loop. The backend reports the conflict without naming the owning
    project, so this points at the place that does show it rather than guessing.
    """
    return (
        f"environment name '{slug}' already belongs to a different project in this organization. "
        "Environment names are unique per organization, not per project, so the same name cannot "
        "be published under two projects and retrying will not change this. Publish under a "
        "different name, or move the existing environment to this project from its environment "
        "page in the Freesolo dashboard (which shows the project that currently owns it)."
    )


@router.post("/v1/envs")
def publish_env(
    payload: dict,
    key: Annotated[dict, Depends(require_key)],
    authorization: Annotated[str | None, Header()] = None,
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
):
    from flash.server.domain import envs
    from flash.server.domain.projects import require_project_access

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
    from flash.server.domain.environment_registry import (
        EnvironmentProjectConflict,
        raise_if_owned_by_another_project,
        record_published_environment,
    )

    resolved_key = {**key, "org_id": key.get("org_id") or x_freesolo_org_id}
    # Check ownership BEFORE uploading: publishing replaces the whole `<org-slug>/<name>` hub
    # directory, so a colliding name would otherwise destroy the other project's package on its
    # way to failing.
    #
    # Only when the destination slug is knowable up front. Deriving it needs the caller's own org
    # namespace, which the org-agnostic internal key does not carry -- there, publish_package
    # resolves the namespace itself and raises its own error. Reporting that error from here
    # instead would change which failure a caller sees, so a slug we cannot derive simply skips
    # the guard and the conflict is still caught at the association step below.
    intended_slug = ""
    org_for_conflict = str(resolved_key.get("org_id") or "").strip()
    # `isinstance` first: a non-string name (0, False, []) sanitizes to the generic "env", and
    # guarding on that slug would answer a malformed request with a conflict about an unrelated
    # environment. An invalid name has to reach publish_package's type check and get its
    # deterministic 400, so only a real string is worth deriving a slug from.
    if isinstance(_name, str):
        with suppress(envs.EnvPublishError):
            namespace, clean_name = envs.publish_slug_for_name(_name, key)
            intended_slug = f"{namespace}/{clean_name}"
    if intended_slug and org_for_conflict:
        try:
            raise_if_owned_by_another_project(
                slug=intended_slug,
                project_id=project_id,
                key=key,
                org_id=org_for_conflict,
            )
        except EnvironmentProjectConflict as exc:
            raise HTTPException(
                status_code=409, detail=_name_conflict_detail(intended_slug)
            ) from exc

    try:
        slug = envs.publish_package(
            package_b64="" if _pkg is None else _pkg,
            name="" if _name is None else _name,
            key=key,
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
    except EnvironmentProjectConflict as exc:
        # Lost a race with a concurrent publish, or the pre-check could not run. Either way the
        # cause is ownership, so it must not be reported as an unrecorded association.
        logger.warning("environment %s belongs to another project: %s", slug, exc)
        raise HTTPException(status_code=409, detail=_name_conflict_detail(slug)) from exc
    except Exception as exc:
        logger.warning(
            "record_published_environment failed after package upload: %s", exc, exc_info=True
        )
        raise HTTPException(status_code=502, detail=_PUBLISH_ASSOCIATION_FAILURE) from exc
    if recorded is not True:
        raise HTTPException(status_code=502, detail=_PUBLISH_ASSOCIATION_FAILURE)
    return {"id": slug}


@router.get("/v1/envs/{env_id:path}/package")
def download_env_package(env_id: str, key: Annotated[dict, Depends(require_key)]):
    """Download a managed Freesolo environment package from the GitHub-backed hub."""
    from flash.server.domain import envs

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
    # ``namespace/name`` slug and carries a slash, so the route uses the ``:path`` converter.
    # Authorization (own-namespace for user keys, any for the internal key) lives in
    # ``delete_package`` so it can't be bypassed.
    from flash.core.spec import require_project_id
    from flash.server.domain import envs
    from flash.server.domain.environment_registry import (
        record_deleted_environment,
        require_environment_project,
    )
    from flash.server.domain.projects import require_project_access

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
