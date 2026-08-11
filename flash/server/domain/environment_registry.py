"""Best-effort reporting of published Flash environments to the Freesolo backend."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Set as AbstractSet

from fastapi import HTTPException

from flash.core.spec import require_project_id
from flash.server.platform.auth import standalone
from flash.server.platform.internal_client import (
    DEFAULT_TIMEOUT_S,
    InternalRequestError,
    build_internal_request,
    delete_internal_json,
    error_detail,
    internal_key,
    org_id_of,
    post_internal_json,
)

_LOG = logging.getLogger("flash.server.environments")
_PATH = "/api/flash/environments/internal"
_VALIDATE_PATH = "/api/flash/environments/validate/internal"
_USE_PATH = "/api/flash/environments/use/internal"
_DEFAULT_HUB_REPO = "freesolo-co/environment-hub"
_DEFAULT_HUB_REF = "main"
_REPAIR_FAILURE_DETAIL = (
    "environment package exists, but its project association could not be repaired"
)


class EnvironmentProjectConflict(Exception):
    """An environment name is already owned by a different project in the same org.

    Names are unique per ORG, not per project: the hub slug is ``<org-slug>/<name>`` with no
    project component. So this is a permanent verdict, and it is the one failure on the publish
    path that a retry can never clear -- which is exactly why it needs its own type rather than
    joining the transient failures behind a bool.
    """


def _post(
    path: str,
    body: dict,
    *,
    subject: str,
    raise_for: AbstractSet[int] | None = None,
    expected: AbstractSet[int] | None = None,
) -> bool:
    return post_internal_json(
        path,
        body,
        subject=subject,
        logger=_LOG,
        urlopen=urllib.request.urlopen,
        raise_for=raise_for,
        expected=expected,
    )


def record_published_environment(*, slug: str, name: str, key: dict, project_id: str) -> bool:
    """Persist the published environment under its validated project.

    Raises :class:`EnvironmentProjectConflict` when the name already belongs to another project
    in this org; returns the usual best-effort bool for every other outcome.
    """
    try:
        resolved_project_id = require_project_id(project_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc).replace("project", "project_id", 1)) from exc
    org_id = org_id_of(key)
    if not org_id:
        return False

    body = {
        "orgId": org_id,
        "slug": slug,
        "name": name,
        "hubRepo": _DEFAULT_HUB_REPO,
        "hubRef": _DEFAULT_HUB_REF,
        "hubPath": f"{slug}/environment.py",
        "publishedByUserId": key.get("user_id"),
        "apiKeyId": key.get("api_key_id"),
        "projectId": resolved_project_id,
        "metadata": {"source": "flash.env.push"},
    }
    try:
        return _post(
            _PATH,
            body,
            subject=f"record published environment {slug}",
            raise_for=frozenset({409}),
        )
    except InternalRequestError as exc:
        raise EnvironmentProjectConflict(exc.detail) from exc


def raise_if_owned_by_another_project(*, slug: str, project_id: str, org_id: str) -> None:
    """Raise :class:`EnvironmentProjectConflict` if ``slug`` already belongs to another project.

    Called BEFORE the hub upload. Publishing writes ``<org-slug>/<name>`` by deleting that
    directory and re-copying it, so without this check a colliding name overwrites the other
    project's environment package and only then fails to record the association -- destroying
    the other project's environment as a side effect of an error.

    Every non-conflict outcome returns normally and lets the publish proceed: a 404 is the
    ordinary first publish of a new name, and a backend that is unreachable or unconfigured must
    not block publishing (the association step afterwards still reports that failure). This is a
    guard against a specific destructive case, not a new availability dependency.
    """
    if standalone():
        # No Freesolo mirror here, so there are no cross-project rows to collide with.
        return
    try:
        _post(
            _VALIDATE_PATH,
            {"orgId": org_id, "projectId": project_id, "slug": slug},
            subject=f"check environment ownership for {slug}",
            raise_for=frozenset({409}),
            # 404 IS the ordinary answer here: this probe asks whether the name is already taken,
            # and "no" is what every first publish of a new name gets. Logging it as a failure
            # would put a warning in the logs for the most common successful path.
            expected=frozenset({404}),
        )
    except InternalRequestError as exc:
        raise EnvironmentProjectConflict(exc.detail) from exc


def require_environment_project(
    *,
    slug: str,
    project_id: str,
    key: dict,
    org_id: str | None = None,
    repair_missing: bool = False,
) -> None:
    """require the environment mirror to belong to the explicit project."""
    try:
        resolved_project_id = require_project_id(project_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if standalone():
        # The mirror this validates against is a Freesolo backend table that does not exist here.
        # Single tenant means there is no cross-org environment to guard against; the environment
        # package itself is still resolved and authorized by the normal env paths.
        return
    resolved_org_id = org_id_of(key) or str(org_id or "").strip()
    if not resolved_org_id:
        raise HTTPException(
            status_code=400,
            detail="organization is required to validate the environment project",
        )
    token = internal_key()
    if not token:
        raise HTTPException(
            status_code=503,
            detail="Freesolo environment validation is unavailable",
        )
    request = build_internal_request(
        _VALIDATE_PATH,
        {
            "orgId": resolved_org_id,
            "projectId": resolved_project_id,
            "slug": slug,
        },
        token=token,
    )
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_S) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = error_detail(exc)
        if exc.code == 404 and repair_missing:
            from flash.server.domain import envs

            not_found_detail = detail or "environment does not belong to the requested project"
            if key.get("auth_kind") == "internal":
                ownership_exc = envs.EnvPublishError(
                    "caller organization namespace is unavailable", status=403
                )
                raise HTTPException(status_code=404, detail=not_found_detail) from ownership_exc
            try:
                caller_namespace = envs.namespace_for(key)
            except envs.EnvPublishError as namespace_exc:
                raise HTTPException(status_code=404, detail=not_found_detail) from namespace_exc
            if slug.split("/", 1)[0] != caller_namespace:
                ownership_exc = envs.EnvPublishError(
                    "environment namespace does not match caller organization", status=403
                )
                raise HTTPException(status_code=404, detail=not_found_detail) from ownership_exc

            # only a missing mirror row reaches the hub clone; normal and conflicting rows stay cheap.
            try:
                envs.download_package(slug=slug, key=key)
            except envs.EnvPublishError as download_exc:
                if download_exc.status == 403:
                    raise HTTPException(
                        status_code=404,
                        detail=not_found_detail,
                    ) from download_exc
                raise HTTPException(
                    status_code=download_exc.status,
                    detail=str(download_exc),
                ) from download_exc
            try:
                recorded = record_published_environment(
                    slug=slug,
                    name=slug.rsplit("/", 1)[-1],
                    key={**key, "org_id": resolved_org_id},
                    project_id=resolved_project_id,
                )
            except Exception as record_exc:
                raise HTTPException(status_code=502, detail=_REPAIR_FAILURE_DETAIL) from record_exc
            if recorded is not True:
                raise HTTPException(status_code=502, detail=_REPAIR_FAILURE_DETAIL) from None
            return
        if exc.code in {404, 409}:
            raise HTTPException(
                status_code=exc.code,
                detail=detail or "environment does not belong to the requested project",
            ) from exc
        if exc.code in {401, 403}:
            raise HTTPException(
                status_code=503,
                detail=("Freesolo environment validation authorization failed"),
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=detail or f"Freesolo environment validation failed with HTTP {exc.code}",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail="Freesolo environment validation is unavailable",
        ) from exc
    try:
        payload = json.loads(raw) if raw else {}
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Freesolo environment validation returned invalid JSON",
        ) from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise HTTPException(
            status_code=502,
            detail="Freesolo environment validation returned an invalid response",
        )


def record_deleted_environment(
    *, slug: str, project_id: str, key: dict, org_id: str | None = None
) -> bool:
    """Remove the platform-backend metadata mirror for a deleted environment.

    Symmetric to :func:`record_published_environment`: the package store (GitHub) is the source
    of truth and is already updated by the time this runs, so dropping the row the web UI lists
    is deliberately best-effort and never blocks ``flash env delete``.

    ``key`` supplies the org for a user-key delete (``flash env delete``). The internal key is
    org-agnostic, so for the web UI delete the caller passes the authenticated user's ``org_id``
    explicitly. We prefer the key's own org and only fall back to the supplied one — so a user
    key never honors a caller-supplied override and can't drop another org's row.
    """
    try:
        resolved_project_id = require_project_id(project_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc).replace("project", "project_id", 1)) from exc
    resolved_org_id = org_id_of(key) or str(org_id or "").strip()
    if not resolved_org_id:
        return False

    return delete_internal_json(
        _PATH,
        {"orgId": resolved_org_id, "projectId": resolved_project_id, "slug": slug},
        subject=f"record deleted environment {slug}",
        logger=_LOG,
        urlopen=urllib.request.urlopen,
    )


def record_environment_use(*, slug: str, project_id: str, run_id: str, key: dict) -> bool:
    try:
        resolved_project_id = require_project_id(project_id)
    except (TypeError, ValueError) as exc:
        raise ValueError(str(exc).replace("project", "project_id", 1)) from exc
    org_id = org_id_of(key)
    if not org_id:
        return False
    body = {
        "orgId": org_id,
        "projectId": resolved_project_id,
        "slug": slug,
        "runId": run_id,
    }
    return _post(_USE_PATH, body, subject=f"record environment use {slug} for run {run_id}")
