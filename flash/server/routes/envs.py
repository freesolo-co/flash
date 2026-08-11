"""Environment publishing endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Response

from flash._internal.logging import get_logger
from flash.server.platform.deps import require_key
from flash.server.platform.internal_client import org_id_of

logger = get_logger("flash.server.routes.envs")
router = APIRouter()

_PUBLISH_ASSOCIATION_FAILURE = (
    "environment package may already be uploaded, but its project association could not be "
    "recorded; retry the same publish to repair the association"
)


def publish_conflict_org_id(*, key: dict, internal_org_id: str | None) -> str:
    """The org whose environment names this publish must be checked against, or ``""``.

    The two auth kinds carry their trusted org in different places, and picking the wrong one is
    not a style question -- it decides whether the ownership check can be bypassed:

    * internal key: `require_project_access` REQUIRES `X-Freesolo-Org-Id` (400 without it) and
      validates the project against it, so by the time we get here that header is established
      fact. It is also the only org this key has: the internal key is org-agnostic.
    * user key: `require_project_access` validates against `key["org_id"]` and ignores the header
      entirely. Trusting the header here would check ownership in an org the caller merely
      asserted while the hub path stays namespaced by the key's own slug -- a garbage id finds
      nothing, returns 404 instead of 409, and waves the colliding write through.
    """
    if key.get("auth_kind") == "internal":
        return str(internal_org_id or "").strip()
    return org_id_of(key)


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
    org_for_conflict = publish_conflict_org_id(key=key, internal_org_id=x_freesolo_org_id)

    def _guard_destination(intended_slug: str) -> None:
        # Check ownership BEFORE the hub write: publishing replaces the whole `<org-slug>/<name>`
        # directory, so a colliding name would otherwise destroy the other project's package on
        # its way to failing.
        #
        # This runs as publish_package's pre-write hook rather than ahead of it, so a request
        # that could never publish -- bad base64, a corrupt archive, an unsafe member, no
        # environment.py -- keeps its own deterministic 400/413 instead of being answered with
        # this guard's 409. By the time the hook fires the package is fully validated and nothing
        # has been written yet.
        if not org_for_conflict:
            raise HTTPException(
                status_code=400,
                detail=(
                    "organization could not be resolved for this key, so the environment's "
                    "project ownership cannot be verified; re-run `flash login` to refresh the "
                    "key. Supplying X-Freesolo-Org-Id does not resolve this: the header is "
                    "caller-asserted and deliberately not trusted for a user key."
                ),
            )
        try:
            raise_if_owned_by_another_project(
                slug=intended_slug,
                project_id=project_id,
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
            before_write=_guard_destination,
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
    from flash.server.domain import envs

    try:
        slugs = envs.list_namespace_slugs(key=key)
    except envs.EnvPublishError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    return {"environments": [{"id": slug} for slug in slugs]}


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
