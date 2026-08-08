"""Stdlib HTTP client for the Flash control plane (no extra dependencies)."""

from __future__ import annotations

import codecs
import contextlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from flash.spec import require_project_id

from .config import load_credentials_with_source

ProgressCallback = Callable[[int, int], None]


class ClientError(RuntimeError):
    """Expected client-side errors (no key, unreachable server) — printed cleanly."""


class RequestTimeoutError(ClientError):
    """A request timed out before the control plane returned a response."""


class ApiError(ClientError):
    """A non-2xx response from a Flash/freesolo endpoint.

    ``detail`` keeps the server's FastAPI ``detail`` as parsed, so a caller can branch on a
    structured payload (``{"code": ..., ...}``) instead of parsing it back out of a dict's repr.
    The message is unchanged, so callers that match on ``str(exc)`` behave exactly as before.
    """

    def __init__(self, status: int, message: str, *, detail: object | None = None):
        super().__init__(message)
        self.status = status
        self.detail = message if detail is None else detail

    @property
    def code(self) -> str:
        """The server's machine-readable error code, or "" for an unstructured detail."""
        if isinstance(self.detail, dict):
            return str(self.detail.get("code") or "")
        return ""


DEFAULT_FREESOLO_BASE_URL = "https://api.freesolo.co"
FREESOLO_AUTH_VERIFY_PATH = "/api/auth/verify"
FREESOLO_PROJECTS_PATH = "/api/projects"
FREESOLO_TRACE_PROJECTS_PATH = "/api/traces/projects"
FREESOLO_TRACES_EXPORT_PATH = "/api/traces/export"
FREESOLO_EVAL_RUNS_PATH = "/api/evals/runs"
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def freesolo_base_url(override: str | None = None) -> str:
    return (override or os.environ.get("FREESOLO_BASE_URL") or DEFAULT_FREESOLO_BASE_URL).rstrip(
        "/"
    )


def _detail_from_http_error(exc: urllib.error.HTTPError) -> object:
    """Extract the server's ``detail`` from an HTTPError body, as parsed.

    A structured detail stays a dict so the caller can read its ``code``; anything else is the
    string it always was. ``str()`` it for a message, but branch on the object itself.
    """
    body = exc.read()
    try:
        detail = json.loads(body).get("detail") or body.decode()
    except (ValueError, AttributeError):
        detail = body.decode(errors="replace") if body else str(exc)
    return detail if isinstance(detail, dict) else str(detail)


def _api_error(exc: urllib.error.HTTPError) -> ApiError:
    detail = _detail_from_http_error(exc)
    return ApiError(exc.code, str(detail), detail=detail)


def _read_capped_response(resp: object, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = resp.read(_DOWNLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ClientError(
                f"response body exceeded the maximum allowed size ({max_bytes} bytes); "
                "download aborted"
            )
        chunks.append(chunk)
    return b"".join(chunks)


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
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise ClientError(
                "freesolo rejected this API key — create or copy a valid key at "
                "https://freesolo.co/sign-in and pass it with `flash login --api-key` "
                "(or FREESOLO_API_KEY)"
            ) from exc
        raise _api_error(exc) from exc
    except urllib.error.URLError as exc:
        raise ClientError(
            f"cannot reach the freesolo backend at {base} ({exc.reason}); "
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise ClientError(
                "freesolo rejected this API key — run `flash login` with a valid key "
                "(or set FREESOLO_API_KEY)"
            ) from exc
        raise _api_error(exc) from exc
    # a socket timeout surfaces as a bare TimeoutError rather than a URLError, so without this
    # it escapes as an unexpected exception. callers catch ClientError to report a failure
    # without changing their own verdict; a traceback instead would lose that.
    except TimeoutError as exc:
        raise RequestTimeoutError(f"request to {base}{path} timed out after {timeout}s") from exc
    except urllib.error.URLError as exc:
        raise ClientError(
            f"cannot reach the freesolo backend at {base} ({exc.reason}); "
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


class _ProgressReader:
    """File-like wrapper over in-memory bytes that fires a progress callback on each read()."""

    def __init__(self, data: bytes, progress: ProgressCallback):
        self._data = data
        self._total = len(data)
        self._pos = 0
        self._progress = progress

    def __len__(self) -> int:
        return self._total

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunk = self._data[self._pos :]
        else:
            chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        # a rendering hiccup must never abort an in-flight upload
        with contextlib.suppress(Exception):
            self._progress(self._pos, self._total)
        return chunk


_CHAT_STEP_SELECTOR_CAPABILITY = "chat_step_selector_v1"


def _validate_chat_messages(messages: list[dict]) -> None:
    if not isinstance(messages, list):
        raise ClientError("chat messages must be a list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ClientError(f"chat messages[{index}] must be an object")


def _parse_chat_target(target: str) -> tuple[str, str | None, int | None]:
    from flash.schema import parse_adapter_revision, parse_checkpoint_ref

    revision = parse_adapter_revision(target)
    if revision is not None:
        return revision[0], target.strip(), None
    parsed = parse_checkpoint_ref(target)
    if parsed is None:
        raise ClientError(
            "invalid run id: expected a bare RUN_ID, RUN_ID/step-N, or a full immutable adapter "
            "revision"
        )
    run_id, step = parsed
    return run_id, None, step


def _prepare_chat_request(
    target: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    *,
    stream: bool = False,
) -> tuple[str, dict[str, Any]]:
    base_run_id, adapter_revision, step = _parse_chat_target(target)
    _validate_chat_messages(messages)
    body: dict[str, Any] = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stream:
        body["stream"] = True
    if adapter_revision is not None:
        body["adapter_revision"] = adapter_revision
    elif step is not None:
        body["step"] = step
    return base_run_id, body


def _parse_adapter_target(target: str) -> tuple[str, int | None]:
    from flash.schema import parse_checkpoint_ref

    parsed = parse_checkpoint_ref(target)
    if parsed is None:
        raise ClientError(
            "invalid adapter id: expected RUN_ID for the final adapter or RUN_ID/step-N "
            "for a saved checkpoint"
        )
    return parsed


class ApiClient:
    def __init__(
        self,
        api_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        key_source: str | None = None,
    ):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.key_source = key_source
        self._chat_step_selector_available = False

    def _auth_headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def _auth_error_detail(self, status: int, detail: object) -> object:
        # only a plain-text detail is appended to. a structured detail is a machine-readable
        # payload whose keys the caller branches on, so splicing prose into it would either
        # corrupt a field or invent one.
        if status not in {401, 403} or self.key_source != "FREESOLO_API_KEY":
            return detail
        if isinstance(detail, dict):
            return detail
        return (
            f"{detail}; FREESOLO_API_KEY is set and overrides the key saved by "
            "`flash login`. Unset FREESOLO_API_KEY or update it to a valid freesolo API key."
        )

    @contextlib.contextmanager
    def _translate_http_errors(self) -> Iterator[None]:
        """Map urllib transport errors to ApiError/ClientError; other exceptions propagate."""
        try:
            yield
        except urllib.error.HTTPError as exc:
            detail = self._auth_error_detail(exc.code, _detail_from_http_error(exc))
            raise ApiError(exc.code, str(detail), detail=detail) from exc
        except urllib.error.URLError as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                raise RequestTimeoutError(
                    f"request to the Flash service at {self.api_url} timed out; "
                    "check your network connection and FLASH_API_URL"
                ) from exc
            raise ClientError(
                f"cannot reach the Flash service at {self.api_url} ({exc.reason}); "
                "check your network connection and FLASH_API_URL"
            ) from exc
        except TimeoutError as exc:
            raise RequestTimeoutError(
                f"request to the Flash service at {self.api_url} timed out; "
                "check your network connection and FLASH_API_URL"
            ) from exc

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        timeout: float | None = None,
        progress: ProgressCallback | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        headers = {
            "Content-Type": "application/json",
            **self._auth_headers(),
            **(extra_headers or {}),
        }
        if progress is not None:
            payload = json.dumps(body).encode()
            headers["Content-Length"] = str(len(payload))
            data: Any = _ProgressReader(payload, progress)
        else:
            data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.api_url}{path}",
            method=method,
            data=data,
            headers=headers,
        )
        with (
            self._translate_http_errors(),
            urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp,
        ):
            raw = resp.read()
            return json.loads(raw) if raw else {}

    def _request_bytes(
        self,
        method: str,
        path: str,
        *,
        timeout: float | None = None,
        max_bytes: int | None = None,
    ) -> bytes:
        headers = {"Accept": "application/gzip", **self._auth_headers()}
        req = urllib.request.Request(
            f"{self.api_url}{path}",
            method=method,
            headers=headers,
        )
        with (
            self._translate_http_errors(),
            urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp,
        ):
            if max_bytes is not None:
                return _read_capped_response(resp, max_bytes)
            return resp.read()

    def me(self) -> dict:
        return self._request("GET", "/v1/me")

    def health(self) -> dict:
        return self._request("GET", "/v1/health", timeout=10.0)

    def _require_chat_step_selector(self) -> None:
        # cached after it first succeeds: this is a property of the control plane, not of the
        # request. `env eval` sends one chat per case, so re-checking each time doubled the request
        # count and let a single transient /v1/health blip fail an arbitrary case while the chat
        # endpoint was healthy.
        if self._chat_step_selector_available:
            return
        capabilities = self.health().get("capabilities")
        if not isinstance(capabilities, list) or _CHAT_STEP_SELECTOR_CAPABILITY not in capabilities:
            raise ClientError(
                "chat checkpoint selectors require a control plane that advertises "
                f"{_CHAT_STEP_SELECTOR_CAPABILITY}; use a full immutable adapter revision or "
                "upgrade the control plane"
            )
        # only a successful capability check is cached, so a transient failure remains visible and
        # retryable. concurrent first calls may make the same benign request twice; a lock would add
        # coordination to every client solely to optimize that one startup race, so a caller about to
        # fan out settles it up front instead (see `warm_chat_step_selector`).
        self._chat_step_selector_available = True

    def warm_chat_step_selector(self, target: str) -> None:
        """Settle the step-selector capability now, so concurrent callers inherit the cached answer.

        A caller about to run many chats in parallel would otherwise have every worker miss the cold
        cache at once and fire its own /v1/health. Only a `RUN/step-N` target needs the
        capability, so anything else is a no-op. Raises exactly what the per-request check raises.
        """
        if _parse_chat_target(target)[2] is not None:
            self._require_chat_step_selector()

    def publish_env(
        self,
        *,
        name: str,
        package_b64: str,
        project_id: str,
        progress: ProgressCallback | None = None,
    ) -> dict:
        """Upload a packaged environment under one explicit Freesolo project."""
        try:
            project_id = require_project_id(project_id)
        except (TypeError, ValueError) as exc:
            raise ClientError(str(exc).replace("project", "project id", 1)) from exc
        body = {"name": name, "package_b64": package_b64, "project_id": project_id}
        return self._request("POST", "/v1/envs", body=body, timeout=1800.0, progress=progress)

    def delete_env(self, env_id: str, *, project_id: str) -> dict:
        """Delete a published Freesolo environment from one explicit project.

        The id carries a slash, which the server route matches with a ``:path`` converter, so the
        path segments go straight into the URL — but percent-encode everything except ``/`` first so
        a programmatic caller passing reserved characters (``?``, ``#``, spaces) can't change the
        request target or drop a fragment, turning a destructive ``team/env?x=1`` into a silent
        delete of ``team/env``. The server removes the hub package and best-effort drops the platform
        metadata mirror; deleting an already-absent env is a no-op (``deleted: false``).

        The timeout matches publish's (1800s): the server-side delete may retry the git workflow
        multiple times, and each attempt includes several git commands with a 180s per-command
        timeout (clone/pull/push can dominate). Keep the client timeout at least as large as the
        server's so the CLI doesn't time out while a destructive delete is still in progress."""
        try:
            project_id = require_project_id(project_id)
        except (TypeError, ValueError) as exc:
            raise ClientError(str(exc).replace("project", "project id", 1)) from exc
        quoted = urllib.parse.quote(env_id, safe="/")
        return self._request(
            "DELETE",
            f"/v1/envs/{quoted}",
            timeout=1800.0,
            extra_headers={"X-Freesolo-Project-Id": project_id},
        )

    def download_env_package(self, env_id: str) -> bytes:
        """Download a managed environment package through the Flash control plane."""
        from flash.envs import loader

        quoted = urllib.parse.quote(env_id, safe="/")
        return self._request_bytes(
            "GET",
            f"/v1/envs/{quoted}/package",
            timeout=1800.0,
            max_bytes=loader._MAX_ARCHIVE_BYTES,
        )

    def create_run(
        self,
        spec: dict,
        runtime_secrets: dict[str, str] | None = None,
        dry_run: bool = False,
        client_train_schema: dict | None = None,
    ) -> dict:
        if not isinstance(spec, dict):
            raise ClientError("spec must be an object")
        try:
            project_id = require_project_id(spec.get("project"))
        except (TypeError, ValueError) as exc:
            raise ClientError(str(exc)) from exc
        body: dict = {"spec": {**spec, "project": project_id}}
        if runtime_secrets:
            body["runtime_secrets"] = runtime_secrets
        if dry_run:
            # server-side preview: runs the same validation/preflights as a real submit and records
            # a state=dry_run run, but allocates no training gpu and charges nothing for training.
            # returns that status. an sft preview additionally requires an exact workload profile;
            # on a miss the server starts that separate, separately billed profile run and answers
            # 409 workload_profile_pending, so a preview is free of training spend, not all spend.
            body["dry_run"] = True
        if client_train_schema is not None:
            body["client_train_schema"] = client_train_schema
        return self._request("POST", "/v1/runs", body=body)

    def list_runs(self) -> list[dict]:
        return self._request("GET", "/v1/runs")["runs"]

    def get_run(self, run_id: str) -> dict:
        return self._request("GET", f"/v1/runs/{run_id}")

    def get_logs(self, run_id: str, offset: int = 0) -> dict:
        return self._request("GET", f"/v1/runs/{run_id}/logs?offset={int(offset)}")

    def get_worker_output(self, run_id: str) -> dict[str, str]:
        try:
            return self._request("GET", f"/v1/runs/{run_id}/worker").get("worker", {})
        except ApiError as exc:
            if exc.status == 404 and "not found" in str(exc).strip().lower():
                return {}
            raise

    def cancel_run(self, run_id: str) -> dict:
        # Cancel can block inside synchronous provider teardown. A read timeout is ambiguous: the
        # server may still have accepted the cancel and later persisted a terminal state. Resolve
        # that by polling the authoritative run status instead of surfacing a raw timeout.
        try:
            return self._request("POST", f"/v1/runs/{run_id}/cancel", timeout=60.0)
        except RequestTimeoutError as exc:
            return self._poll_cancel_status(run_id, cause=exc)

    def _poll_cancel_status(self, run_id: str, *, cause: RequestTimeoutError) -> dict:
        deadline = time.monotonic() + 120.0
        last_state = "unknown"
        while True:
            try:
                status = self.get_run(run_id)
            except ClientError:
                pass
            else:
                last_state = str(status.get("state") or "unknown")
                deployment = status.get("deployment") or {}
                terminal = last_state in {"cancelled", "done", "failed", "dry_run"}
                revocation_failed = (
                    isinstance(deployment, dict) and deployment.get("state") == "revocation_failed"
                )
                if terminal and revocation_failed:
                    error = deployment.get("error") or "unknown backend teardown error"
                    raise ClientError(
                        "cancel request reached the control plane, but backend revocation is "
                        f"unconfirmed: {error}; retry cancellation"
                    ) from cause
                if terminal:
                    return status
            if time.monotonic() >= deadline:
                raise ClientError(
                    f"cancel request timed out before confirmation; latest state={last_state!r}. "
                    f"Run `flash runs status {run_id}` to check the authoritative state before retrying."
                ) from cause
            time.sleep(2.0)

    def checkpoints(self, run_id: str) -> list[dict]:
        """Deployable per-step RL checkpoints for a run (serve one with `flash models deploy RUN/step-N`)."""
        return self._request("GET", f"/v1/runs/{run_id}/checkpoints")["checkpoints"]

    def deploy(
        self,
        run_id: str,
        dry_run: bool = False,
    ) -> dict:
        base_run_id, step = _parse_adapter_target(run_id)
        # smoke verification is mandatory server-side; there is no opt-out to forward.
        body: dict = {"dry_run": dry_run}
        if step is not None:
            body["step"] = step
        return self._request(
            "POST",
            f"/v1/runs/{base_run_id}/deploy",
            body=body,
        )

    def export(
        self,
        run_id: str,
        *,
        repository: str,
        hf_token: str,
        private: bool = True,
    ) -> dict:
        """Copy a run's adapter into a user-owned HuggingFace repo."""
        base_run_id, step = _parse_adapter_target(run_id)
        body: dict = {"repository": repository, "hf_token": hf_token, "private": private}
        if step is not None:
            body["step"] = step
        return self._request("POST", f"/v1/runs/{base_run_id}/export", body=body, timeout=30 * 60)

    def undeploy(self, run_id: str) -> dict:
        return self._request("DELETE", f"/v1/runs/{run_id}/deploy")

    def deployments(self, timeout: float | None = None) -> list[dict]:
        return self._request("GET", "/v1/deployments", timeout=timeout)["deployments"]

    def deployment_for(self, run_id: str, timeout: float | None = None) -> dict | None:
        """The current deployment record for one run, or None when it is not listed.

        ``deploy`` returns as soon as the record is persisted, which is normally before the
        requested revision is servable. This is the read side a caller needs to tell "queued"
        from "actually ready".

        ``timeout`` bounds the single request. A caller polling against its own deadline needs
        that: the default client timeout is 60s, so one stalled read inside a `--wait 5` would
        overshoot the bound the user asked for by an order of magnitude.

        Read from the run-scoped route rather than the listing. `/v1/deployments` walks every run
        the key owns and loads each one's status before this picks a single record out, so on an
        account with a long run history the poll's cost grows with that history and the wait can
        expire scanning unrelated runs while the requested revision is already ready
        `/v1/runs/{run_id}/deploy` resolves the one run directly.
        """
        base_run_id, step = _parse_adapter_target(run_id)
        try:
            deployment = self._request("GET", f"/v1/runs/{base_run_id}/deploy", timeout=timeout)
        except ApiError as exc:
            # a run the key cannot see reads the same as one that is not deployed. the listing said
            # "absent" by omitting the row; saying it by raising would turn a vanished deployment
            # into a failed command.
            if exc.status == 404:
                return None
            raise
        if not isinstance(deployment, dict):
            return None
        # the route answers for a run that was never deployed with a synthesized `undeployed`
        # record rather than 404, and the listing omits `undeployed`/`dry_run` rows entirely. keep
        # the listing's meaning: neither is a revision anyone can serve.
        if str(deployment.get("state") or "") in {"undeployed", "dry_run"}:
            return None
        # the requested step is part of the identity, not decoration. matching on the run id
        # alone lets `deploy RUN/step-40 --wait` settle on whichever revision happens to be
        # deployed -- an older one still marked ready, or a replacement another shell deployed
        # mid-wait -- and report that as this caller's own revision.
        if "checkpoint_step" in deployment:
            listed = deployment.get("checkpoint_step")
            # None is the final adapter, an int is RUN/step-N (see the deployments renderer).
            if (listed if listed is None else int(listed)) != step:
                return None
        if not deployment.get("run_id"):
            # `models deploy --wait` prints this record in place of the POST body, so without the
            # id styled output renders an empty run field and the json omits it entirely.
            deployment = {**deployment, "run_id": base_run_id}
        return deployment

    def chat(
        self,
        run_id: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float | None = None,
    ) -> dict:
        base_run_id, body = _prepare_chat_request(
            run_id,
            messages,
            temperature,
            max_tokens,
        )
        if "step" in body:
            self._require_chat_step_selector()
        return self._request(
            "POST",
            f"/v1/runs/{base_run_id}/chat",
            body=body,
            timeout=timeout if timeout is not None else 30 * 60,
        )

    def chat_stream(
        self,
        run_id: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        base_run_id, body = _prepare_chat_request(
            run_id,
            messages,
            temperature,
            max_tokens,
            stream=True,
        )
        if "step" in body:
            self._require_chat_step_selector()
        headers = {"Content-Type": "application/json", **self._auth_headers()}
        req = urllib.request.Request(
            f"{self.api_url}/v1/runs/{base_run_id}/chat",
            method="POST",
            data=json.dumps(body).encode(),
            headers=headers,
        )
        decoder = codecs.getincrementaldecoder("utf-8")()
        with (
            self._translate_http_errors(),
            urllib.request.urlopen(req, timeout=30 * 60) as resp,
        ):
            content_type = resp.headers.get("Content-Type", "")
            if "application/json" in content_type:
                payload = json.loads(resp.read() or b"{}")
                content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
                if content:
                    yield str(content)
                return
            read1 = getattr(resp, "read1", None)
            read = read1 if read1 is not None else resp.read
            read_size = 4096 if read1 is not None else 1
            while raw := read(read_size):
                state = decoder.getstate()
                try:
                    decoded = decoder.decode(raw)
                except UnicodeDecodeError as exc:
                    decoder.setstate(state)
                    prefix_end = max(0, exc.start - len(state[0]))
                    yield from decoder.decode(raw[:prefix_end])
                    # bind + re-raise explicitly: the yield above clears the active exception, so a
                    # bare `raise` here would fail with "No active exception to reraise".
                    raise exc
                yield from decoded
            yield from decoder.decode(b"", final=True)


def client_from_config(require_key: bool = True) -> ApiClient:
    """Build a client from the stored credentials; fail with a clear hint when logged out."""
    api_url, api_key, key_source = load_credentials_with_source()
    if require_key and not api_key:
        raise ClientError(
            "not logged in — run `flash login` with your freesolo API key (or set FREESOLO_API_KEY)"
        )
    return ApiClient(api_url, api_key, key_source=key_source)
