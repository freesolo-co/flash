"""Calls to the Freesolo backend extracted to keep ``http.py`` under 1000 lines."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from flash._internal.channel import CLI_NAME
from flash._internal.http import _urlopen_no_redirect
from flash.client.http import (
    FREESOLO_AUTH_VERIFY_PATH,
    FREESOLO_EVAL_RUNS_PATH,
    FREESOLO_PROJECTS_PATH,
    FREESOLO_TRACE_PROJECTS_PATH,
    FREESOLO_TRACES_EXPORT_PATH,
    ClientError,
    RequestTimeoutError,
    _api_error,
    freesolo_base_url,
)
from flash.core.spec import require_project_id
from flash.serve.contract.urls import displayable_url


def verify_freesolo_key(api_key: str, base_url: str | None = None) -> None:
    """Verify a freesolo API key; raises ClientError/ApiError on failure."""
    base = freesolo_base_url(base_url)
    url = f"{base}{FREESOLO_AUTH_VERIFY_PATH}"
    req = urllib.request.Request(
        url,
        method="GET",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with _urlopen_no_redirect(req, timeout=30) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            # name the url that rejected it, not just the key: the same 401 is what a valid key
            # gets when the request went to the wrong issuer (a leftover localhost from a
            # self-hosted experiment, an overridden FREESOLO_BASE_URL). Without the URL the message
            # accuses the one thing the user just copied correctly, and the actual cause -- which
            # service answered -- is never shown. name the knobs THIS url came from
            # (``freesolo_base_url``); --api-url points at the control plane and is not read here.
            raise ClientError(
                f"{displayable_url(base)} rejected this API key: check that this is the right "
                "service for the key (a stale --freesolo-url or FREESOLO_BASE_URL rejects a "
                "perfectly valid key), then create or copy a valid key at "
                f"https://freesolo.co/sign-in and pass it with `{CLI_NAME} login --api-key` "
                "(or FREESOLO_API_KEY)"
            ) from exc
        raise _api_error(exc) from exc
    except urllib.error.URLError as exc:
        raise ClientError(
            f"cannot reach the freesolo backend at {displayable_url(base)} ({exc.reason}); "
            "check your network connection and FREESOLO_BASE_URL"
        ) from exc


def _freesolo_request(
    method: str,
    path: str,
    api_key: str,
    base_url: str | None = None,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = 60.0,
):
    """Call a Freesolo bearer endpoint directly and return parsed JSON."""
    base = freesolo_base_url(base_url)
    req = urllib.request.Request(
        f"{base}{path}",
        method=method,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with _urlopen_no_redirect(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise ClientError(
                f"freesolo rejected this API key — run `{CLI_NAME} login` with a valid key "
                "(or set FREESOLO_API_KEY)"
            ) from exc
        raise _api_error(exc) from exc
    # a socket timeout surfaces as a bare timeouterror rather than a urlerror, so without this
    # it escapes as an unexpected exception. callers catch clienterror to report a failure
    # without changing their own verdict; a traceback instead would lose that.
    except TimeoutError as exc:
        # name the BACKEND, not a reconstructed route: `displayable_url` reduces the base to scheme
        # and host, so appending `path` to it would print an endpoint that was never requested
        # whenever FREESOLO_BASE_URL carries a reverse-proxy prefix (`https://host/proxy` requests
        # `/proxy{path}` but would read as `/…{path}`). the path is this client's own constant, so
        # naming it separately keeps the operator's real diagnostic without inventing a URL.
        raise RequestTimeoutError(
            f"request to {displayable_url(base)} ({path}) timed out after {timeout}s"
        ) from exc
    except urllib.error.URLError as exc:
        raise ClientError(
            f"cannot reach the freesolo backend at {displayable_url(base)} ({exc.reason}); "
            "check your network connection and FREESOLO_BASE_URL"
        ) from exc
    try:
        return json.loads(raw) if raw else {}
    except (TypeError, ValueError) as exc:
        raise ClientError(f"freesolo returned invalid JSON for {path}") from exc


def _freesolo_get(path: str, api_key: str, base_url: str | None = None, timeout: float = 60.0):
    return _freesolo_request("GET", path, api_key, base_url, timeout=timeout)


def list_projects(api_key: str, base_url: str | None = None) -> list[dict[str, Any]]:
    """List projects in the authenticated caller's Freesolo organization."""
    payload = _freesolo_get(FREESOLO_PROJECTS_PATH, api_key, base_url)
    if not isinstance(payload, list) or any(not isinstance(project, dict) for project in payload):
        raise ClientError("freesolo returned an invalid project list")
    return payload


def get_project(project_id: str, api_key: str, base_url: str | None = None) -> dict[str, Any]:
    """Fetch one project and require the requested canonical UUID in the response."""
    try:
        project_id = require_project_id(project_id)
    except (TypeError, ValueError) as exc:
        raise ClientError(str(exc).replace("project", "project id", 1)) from exc
    quoted = urllib.parse.quote(project_id, safe="")
    payload = _freesolo_get(f"{FREESOLO_PROJECTS_PATH}/{quoted}", api_key, base_url)
    project = payload.get("project") if isinstance(payload, dict) else None
    if not isinstance(project, dict):
        project = payload if isinstance(payload, dict) else None
    returned_id = project.get("id") if isinstance(project, dict) else None
    try:
        returned_id = require_project_id(returned_id)
    except (TypeError, ValueError) as exc:
        raise ClientError(f"freesolo returned no valid project for {project_id}") from exc
    if returned_id != project_id:
        raise ClientError(f"freesolo returned a mismatched project id for {project_id}")
    return {**project, "id": returned_id}


def create_project(
    name: str,
    description: str | None,
    api_key: str,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Create a project in the authenticated caller's organization."""
    name = str(name or "").strip()
    if not name:
        raise ClientError("project name must be nonblank")
    normalized_description = None
    if description is not None:
        normalized_description = str(description).strip() or None
    payload = _freesolo_request(
        "POST",
        FREESOLO_PROJECTS_PATH,
        api_key,
        base_url,
        body={"name": name, "description": normalized_description},
    )
    if not isinstance(payload, dict):
        raise ClientError("freesolo returned an invalid project response")
    try:
        project_id = require_project_id(payload.get("id"))
    except (TypeError, ValueError) as exc:
        raise ClientError("freesolo returned an invalid project id") from exc
    return {**payload, "id": project_id}


def list_trace_projects(api_key: str, base_url: str | None = None) -> list[dict[str, Any]]:
    """Projects in the caller's org that traces can be exported from."""
    payload = _freesolo_get(FREESOLO_TRACE_PROJECTS_PATH, api_key, base_url)
    projects = payload.get("projects")
    return projects if isinstance(projects, list) else []


def export_trace_records(
    project_id: str,
    api_key: str,
    base_url: str | None = None,
    limit: int | None = None,
    export_format: str | None = None,
) -> dict[str, Any]:
    """A project's traces in the requested shape, converted server-side.

    Returns ``{"records": [...], "traces": N, "skipped": N, "format": name}``. The
    shape of each record depends on ``export_format`` (see EXPORT_FORMATS); the
    conversion runs server-side, matching what the web app's export downloads."""
    query = {"project_id": project_id}
    if limit is not None:
        query["limit"] = str(int(limit))
    if export_format is not None:
        query["format"] = export_format
    path = f"{FREESOLO_TRACES_EXPORT_PATH}?{urllib.parse.urlencode(query)}"
    # a whole project's traces can be a large read; give it room beyond the default.
    return _freesolo_get(path, api_key, base_url, timeout=300.0)


def upload_eval_run(
    *,
    project_id: str,
    suite_name: str,
    environment_reference: str,
    model: str | None,
    status: str,
    error: str | None,
    started_at: str | None,
    cases: list[dict[str, Any]],
    api_key: str,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Record one `flash env eval` suite run against one explicit Freesolo project.

    The project id is required and never inferred: an API key identifies an org, not a
    project, and picking a default here would file results under a project the caller
    never named."""
    try:
        project_id = require_project_id(project_id)
    except (TypeError, ValueError) as exc:
        raise ClientError(str(exc).replace("project", "project id", 1)) from exc
    body: dict[str, Any] = {
        "project_id": project_id,
        "suite_name": suite_name,
        "environment_reference": environment_reference,
        "model": model,
        "status": status,
        "error": error,
        "started_at": started_at,
        "cases": cases,
    }
    # a large suite is a bigger write than a normal control-plane call; give it room.
    payload = _freesolo_request(
        "POST", FREESOLO_EVAL_RUNS_PATH, api_key, base_url, body=body, timeout=300.0
    )
    if not isinstance(payload, dict):
        raise ClientError("freesolo returned an invalid eval run response")
    return payload
