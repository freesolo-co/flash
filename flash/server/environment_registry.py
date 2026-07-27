"""Best-effort reporting of published Flash environments to the Freesolo backend."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import re
import urllib.error
import urllib.request

from fastapi import HTTPException

from flash.spec import require_project_id

from ._internal_client import (
    DEFAULT_TIMEOUT_S,
    build_internal_request,
    delete_internal_json,
    internal_key,
    org_id_of,
    post_internal_json,
)

_LOG = logging.getLogger("flash.server.environments")
_PATH = "/api/flash/environments/internal"
_VALIDATE_PATH = "/api/flash/environments/validate/internal"
_PACKAGE_PATH = "/api/flash/environments/package/internal"
_USE_PATH = "/api/flash/environments/use/internal"
_DEFAULT_HUB_REPO = "freesolo-co/environment-hub"
_DEFAULT_HUB_REF = "main"


def _post(path: str, body: dict, *, subject: str) -> bool:
    return post_internal_json(
        path,
        body,
        subject=subject,
        logger=_LOG,
        urlopen=urllib.request.urlopen,
    )


def record_published_environment(*, slug: str, name: str, key: dict, project_id: str) -> bool:
    """Persist the published environment under its validated project."""
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
    return _post(_PATH, body, subject=f"record published environment {slug}")


def _validation_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(exc.read())
    except (TypeError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    detail = payload.get("detail") or payload.get("error")
    return str(detail).strip() if detail else ""


def _environment_request_identity(
    *,
    project_id: str,
    key: dict,
    org_id: str | None,
    missing_org_detail: str,
    unavailable_detail: str,
) -> tuple[str, str, str]:
    try:
        resolved_project_id = require_project_id(project_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    resolved_org_id = org_id_of(key) or str(org_id or "").strip()
    if not resolved_org_id:
        raise HTTPException(status_code=400, detail=missing_org_detail)
    token = internal_key()
    if not token:
        raise HTTPException(status_code=503, detail=unavailable_detail)
    return resolved_project_id, resolved_org_id, token


def require_environment_project(
    *, slug: str, project_id: str, key: dict, org_id: str | None = None
) -> None:
    """require the environment mirror to belong to the explicit project."""
    resolved_project_id, resolved_org_id, token = _environment_request_identity(
        project_id=project_id,
        key=key,
        org_id=org_id,
        missing_org_detail="organization is required to validate the environment project",
        unavailable_detail="Freesolo environment validation is unavailable",
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
        detail = _validation_error_detail(exc)
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


def resolve_environment_package_source(
    *, slug: str, project_id: str, key: dict, org_id: str | None = None
) -> dict[str, str]:
    """Fetch and validate one project's managed environment package source."""
    resolved_project_id, resolved_org_id, token = _environment_request_identity(
        project_id=project_id,
        key=key,
        org_id=org_id,
        missing_org_detail="organization is required to resolve the environment package",
        unavailable_detail="Freesolo environment package resolution is unavailable",
    )
    request = build_internal_request(
        _PACKAGE_PATH,
        {"orgId": resolved_org_id, "projectId": resolved_project_id, "slug": slug},
        token=token,
    )
    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_S) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = _validation_error_detail(exc)
        if exc.code in {404, 409}:
            raise HTTPException(
                status_code=exc.code,
                detail=detail or "environment does not belong to the requested project",
            ) from exc
        if exc.code in {401, 403}:
            raise HTTPException(
                status_code=503,
                detail="Freesolo environment package authorization failed",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=detail or f"Freesolo environment package resolution failed with HTTP {exc.code}",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=503,
            detail="Freesolo environment package resolution is unavailable",
        ) from exc
    try:
        payload = json.loads(raw) if raw else {}
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Freesolo environment package resolution returned invalid JSON",
        ) from exc
    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail="Freesolo environment package resolution returned an invalid response",
        )

    source_kind = payload.get("sourceKind")
    if source_kind == "hub":
        if set(payload) != {"sourceKind"}:
            raise HTTPException(
                status_code=502,
                detail="Freesolo Hub environment package response has an invalid shape",
            )
        return {"source_kind": "hub"}
    if source_kind != "builtin" or set(payload) != {
        "sourceKind",
        "packageBase64",
        "packageSha256",
    }:
        raise HTTPException(
            status_code=502,
            detail="Freesolo built-in environment package response has an invalid shape",
        )

    package_base64 = payload.get("packageBase64")
    package_sha256 = payload.get("packageSha256")
    if not isinstance(package_base64, str):
        raise HTTPException(
            status_code=502,
            detail="Freesolo built-in environment package response is incomplete",
        )
    if not isinstance(package_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", package_sha256) is None:
        raise HTTPException(
            status_code=502,
            detail="Freesolo built-in environment package sha256 is invalid",
        )

    from flash.envs.loader import (
        _MAX_BUILTIN_PACKAGE_BASE64_CHARS,
        _MAX_BUILTIN_PACKAGE_BYTES,
    )

    if len(package_base64) > _MAX_BUILTIN_PACKAGE_BASE64_CHARS:
        raise HTTPException(
            status_code=502,
            detail="Freesolo built-in environment package is too large",
        )
    try:
        package = base64.b64decode(package_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Freesolo built-in environment package is not valid base64",
        ) from exc
    if len(package) > _MAX_BUILTIN_PACKAGE_BYTES:
        raise HTTPException(
            status_code=502,
            detail="Freesolo built-in environment package is too large",
        )
    if hashlib.sha256(package).hexdigest() != package_sha256:
        raise HTTPException(
            status_code=502,
            detail="Freesolo built-in environment package sha256 mismatch",
        )
    return {
        "source_kind": "builtin",
        "package_base64": package_base64,
        "package_sha256": package_sha256,
    }


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
