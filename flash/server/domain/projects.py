"""Freesolo project ownership validation for paid and publication paths."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from fastapi import HTTPException

from flash.core.spec import require_project_id
from flash.server.platform.auth import freesolo_base_url, standalone
from flash.server.platform.internal_client import build_internal_request, internal_key

_PROJECT_TIMEOUT_S = 10.0
_INTERNAL_PROJECT_PATH = "/api/flash/projects/validate/internal"


def _bearer_token(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing Freesolo bearer authorization")
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(status_code=401, detail="missing Freesolo bearer authorization")
    return token


def _raw_error_detail(raw: bytes) -> str:
    try:
        body = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return ""
    if not isinstance(body, dict):
        return ""
    detail = body.get("detail") or body.get("error")
    return str(detail).strip() if detail else ""


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return _raw_error_detail(exc.read())
    except Exception:
        return ""


def _project_payload_id(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    project = payload.get("project")
    if isinstance(project, dict):
        payload = project
    try:
        return require_project_id(payload.get("id"))
    except (TypeError, ValueError):
        return ""


def _payload_project_slug(payload: Any) -> str:
    """The project slug carried by a validation response, or ``""``.

    Accepted at the top level or nested under ``project``, and under either spelling, because the
    internal and user-key validation endpoints return differently shaped bodies.

    The project directory stores no slug column, so no deployment actually sends one of those
    fields today: the slug is derived from the project's name instead, which the directory does
    return and holds ``UNIQUE (org_id, name)``. That uniqueness is what makes the derivation a
    stable identity rather than a guess -- two projects in one org cannot normalize to the same
    directory. An explicit field still wins when a backend grows one, so this needs no second
    change to hand authority back.
    """
    if not isinstance(payload, dict):
        return ""
    candidates: list[Any] = [payload]
    nested = payload.get("project")
    if isinstance(nested, dict):
        candidates.append(nested)
    for source in candidates:
        for field in ("projectSlug", "slug"):
            value = source.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    for source in candidates:
        for field in ("projectName", "name"):
            derived = _slug_from_project_name(source.get(field))
            if derived:
                return derived
    return ""


def _slug_from_project_name(value: Any) -> str:
    """The hub directory segment a project ``name`` normalizes to, or ``""``.

    Shares :func:`normalize_env_name_segment` with the name segment so both halves of a published
    ``<org>/<project>/<name>`` id obey one grammar, and so a name that survives normalization here
    is one ``publish_slug_for_name`` will accept rather than reject a step later.
    """
    from flash.schema import normalize_env_name_segment

    if not isinstance(value, str) or not value.strip():
        return ""
    return normalize_env_name_segment(value) or ""


def _internal_http_error(*, status: int, raw: bytes, project_id: str) -> HTTPException:
    detail = _raw_error_detail(raw)
    if status == 404:
        return HTTPException(
            status_code=403,
            detail=f"project {project_id!r} does not belong to the authenticated organization",
        )
    return HTTPException(
        status_code=502,
        detail=detail or f"Freesolo internal project validation failed with HTTP {status}",
    )


def _require_internal_project_access(*, project_id: str, org_id: str) -> tuple[str, str]:
    token = internal_key()
    if not token:
        raise HTTPException(
            status_code=503,
            detail="FREESOLO_INTERNAL_KEY is required for internal project validation",
        )
    request = build_internal_request(
        _INTERNAL_PROJECT_PATH,
        {"orgId": org_id, "projectId": project_id},
        token=token,
    )
    try:
        with urllib.request.urlopen(request, timeout=_PROJECT_TIMEOUT_S) as response:
            status = int(response.status)
            raw = response.read()
    except urllib.error.HTTPError as exc:
        try:
            raw = exc.read()
        except Exception:
            raw = b""
        raise _internal_http_error(status=exc.code, raw=raw, project_id=project_id) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Freesolo project validation is unavailable; no paid work was started",
        ) from exc

    if not 200 <= status < 300:
        raise _internal_http_error(status=status, raw=raw, project_id=project_id)
    try:
        payload = json.loads(raw) if raw else None
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Freesolo internal project validation returned invalid JSON",
        ) from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise HTTPException(
            status_code=502,
            detail="Freesolo internal project validation returned a malformed response",
        )
    returned_project_id = payload.get("projectId")
    returned_org_id = payload.get("orgId")
    if returned_project_id != project_id or returned_org_id != org_id:
        raise HTTPException(
            status_code=502,
            detail="Freesolo internal project validation returned a mismatched project association",
        )
    return project_id, _payload_project_slug(payload)


def require_project_access(
    *,
    project_id: str,
    key: dict[str, Any],
    authorization: str | None,
    org_id: str | None = None,
) -> str:
    """Require the explicit project to resolve under the authenticated Freesolo organization."""
    return _project_access(
        project_id=project_id,
        key=key,
        authorization=authorization,
        org_id=org_id,
    )[0]


def require_project_access_slug(
    *,
    project_id: str,
    key: dict[str, Any],
    authorization: str | None,
    org_id: str | None = None,
) -> tuple[str, str]:
    """As :func:`require_project_access`, but also returns the project's non-empty slug.

    The slug is the second segment of every environment slug this project publishes, so a caller
    that asked for one cannot proceed without it. Validation therefore ends here rather than at
    the publish itself: the destination is unknowable, and the failure has to name that.

    Two ways a slug can come back empty, both of which used to reach ``publish_slug_for_name``
    and surface as "re-run `flash login` to refresh the key" -- advice that cannot help, because
    neither cause is the caller's key. A standalone plane has no Freesolo project directory to
    resolve a slug from and cannot publish to the managed hub at all; on a managed plane the slug
    is derived from the project's name, so the only remaining way to reach here is a name with no
    character that survives normalization (``"???"``). Reporting them apart is the point: one is a
    permanent property of the deployment, the other is fixed by renaming the project -- and the
    second is actionable by the caller, so it must not be reported as an opaque upstream fault.
    """
    project_id, slug = _project_access(
        project_id=project_id,
        key=key,
        authorization=authorization,
        org_id=org_id,
    )
    if not slug:
        if standalone():
            raise HTTPException(
                status_code=501,
                detail=(
                    "publishing environments requires a Freesolo project directory to resolve "
                    "the project slug, which a standalone plane does not have; reference the "
                    "environment from git instead (see SELF_HOSTING.md)"
                ),
            )
        raise HTTPException(
            status_code=400,
            detail=(
                f"project {project_id!r} has no name that can form a publishable slug, so the "
                "environment's destination cannot be resolved; rename the project to include "
                "a letter or digit"
            ),
        )
    return project_id, slug


def _project_access(
    *,
    project_id: str,
    key: dict[str, Any],
    authorization: str | None,
    org_id: str | None = None,
) -> tuple[str, str]:
    """Validate project ownership and return ``(project_id, slug)``, slug possibly ``""``.

    The private primitive behind both public entry points, because only one of them needs a slug.
    ``require_project_access`` is the id-only path taken by runs and env delete, and standalone
    legitimately has no slug for it -- so the "a slug must exist" rule cannot live here without
    failing every standalone run. It lives in ``require_project_access_slug`` instead, which is
    the entry point whose return type promises one.
    """
    try:
        project_id = require_project_id(project_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if standalone():
        # Single tenant, no org directory: there is no OTHER organization to confuse this project
        # with, so ownership is already established by the operator key that got the caller here.
        # The id is still REQUIRED and still validated for shape above -- runs stay grouped, and a
        # malformed id fails the same way it would against the backend.
        return project_id, ""

    if key.get("auth_kind") == "internal":
        expected_org = str(org_id or "").strip()
        if not expected_org:
            raise HTTPException(
                status_code=400,
                detail="X-Freesolo-Org-Id is required for internal project validation",
            )
        return _require_internal_project_access(project_id=project_id, org_id=expected_org)

    headers = {
        "Authorization": f"Bearer {_bearer_token(authorization)}",
        "Content-Type": "application/json",
    }
    expected_org = str(key.get("org_id") or "").strip()
    quoted = urllib.parse.quote(project_id, safe="")
    req = urllib.request.Request(
        f"{freesolo_base_url()}/api/projects/{quoted}",
        method="GET",
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=_PROJECT_TIMEOUT_S) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = _error_detail(exc)
        if exc.code == 401:
            raise HTTPException(
                status_code=401, detail=detail or "invalid Freesolo bearer key"
            ) from exc
        if exc.code in {403, 404}:
            raise HTTPException(
                status_code=403,
                detail=f"project {project_id!r} does not belong to the authenticated organization",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=detail or f"Freesolo project validation failed with HTTP {exc.code}",
        ) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Freesolo project validation is unavailable; no paid work was started",
        ) from exc

    try:
        payload = json.loads(raw) if raw else {}
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Freesolo project validation returned invalid JSON",
        ) from exc
    if _project_payload_id(payload) != project_id:
        raise HTTPException(
            status_code=502,
            detail="Freesolo project validation returned a mismatched project id",
        )

    payload_project = payload.get("project") if isinstance(payload, dict) else None
    project = payload_project if isinstance(payload_project, dict) else payload
    returned_org = ""
    if isinstance(project, dict):
        for field in ("org_id", "orgId"):
            value = project.get(field)
            if isinstance(value, str) and value.strip():
                returned_org = value.strip()
                break
    if expected_org and returned_org and returned_org != expected_org:
        raise HTTPException(
            status_code=403,
            detail=f"project {project_id!r} does not belong to the authenticated organization",
        )
    return project_id, _payload_project_slug(payload)
