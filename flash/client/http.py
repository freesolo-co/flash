"""Stdlib HTTP client for the Flash control plane (no extra dependencies)."""

from __future__ import annotations

import codecs
import contextlib
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping
from typing import Any

from flash._internal.channel import CLI_NAME
from flash._internal.http import _urlopen_no_redirect
from flash._internal.openai_sse import (
    DeltaEvent,
    ErrorEvent,
    OpenAISSEError,
    iter_openai_sse_events,
)
from flash.client.config import load_credentials_with_source
from flash.client.shapes import RequireSpec, matches_require
from flash.client.streaming import (
    ProgressCallback,
    _capped_timeout,
    _ProgressReader,
    _read_capped_response,
    _read_response_body,
)
from flash.core.spec import require_project_id
from flash.serve.contract.urls import is_freesolo_hosted_url
from flash.serve.request.tool_calls import validate_tool_control_presence


class ClientError(RuntimeError):
    """Expected client-side errors (no key, unreachable server) — printed cleanly."""


class RequestTimeoutError(ClientError):
    """A request timed out before the control plane returned a response."""


class ServiceUnreachableError(ClientError):
    """The transport never reached the control plane (DNS, refused connection, reset).

    Distinct from a plain ``ClientError`` because a caller that retries needs to tell "nobody
    answered" apart from "something answered and it was not a Flash plane". The second is a
    permanent misconfiguration -- retrying it only hides the hint that would fix it.
    """


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

# the server performs two github reads with one retry and one bounded backoff per read.
_ENV_LIST_GITHUB_READS = 2
_ENV_LIST_ATTEMPTS_PER_READ = 2
_ENV_LIST_SOCKET_TIMEOUT_SECONDS = 20.0
_ENV_LIST_MAX_BACKOFF_PER_READ_SECONDS = 45.0
ENV_LIST_SERVER_BUDGET_SECONDS = _ENV_LIST_GITHUB_READS * (
    _ENV_LIST_ATTEMPTS_PER_READ * _ENV_LIST_SOCKET_TIMEOUT_SECONDS
    + _ENV_LIST_MAX_BACKOFF_PER_READ_SECONDS
)
ENV_LIST_CLIENT_TIMEOUT_SECONDS = ENV_LIST_SERVER_BUDGET_SECONDS + 60.0
ENV_LIST_MAX_RESPONSE_BYTES = 32 * 1024 * 1024


def freesolo_base_url(override: str | None = None) -> str:
    return (override or os.environ.get("FREESOLO_BASE_URL") or DEFAULT_FREESOLO_BASE_URL).rstrip(
        "/"
    )


def has_freesolo_backend(api_url: str) -> bool:
    """Whether the calls in this module have a Freesolo backend to reach.

    Lives beside ``freesolo_base_url`` because it answers a question about the same env var: a
    caller cannot decide whether the hosted API is reachable from the control-plane url alone.
    Two signals can supply a backend, and the absence of both means there is none:

    - the control-plane url names Freesolo. ``FLASH_STANDALONE`` is server-side and these callers
      never reach the plane, so the url is the only standalone signal available.
    - ``FREESOLO_BASE_URL`` names someone else. An operator running their own plane can point it
      at a Freesolo-compatible backend, which is what these calls resolve through.

    Note the deliberate polarity flip: the plane url qualifies by BEING Freesolo's, the backend
    url by NOT being. They are different questions -- "is the hosted backend my target" versus
    "does the operator run their own" -- and honouring a backend url that names Freesolo would
    send a self-hosted plane's operator key to it.

    The presence of ``FREESOLO_API_KEY`` deliberately does NOT count. It looks like a hosted
    signal but cannot be one: ``SELF_HOSTING.md`` has self-hosters log in with the
    plane-controlling ``FREESOLO_INTERNAL_KEY``, and ``cmd_login`` reads that env var as the login
    key, so its value is as likely to be the operator key as a hosted account key. Nothing marks
    which, and guessing wrong ships the plane credential to Freesolo. A hosted account reached
    from a self-hosted plane needs ``FREESOLO_BASE_URL`` set explicitly.

    A false positive is the safe direction: assuming a backend exists yields today's behaviour
    (an authenticated call that may fail) rather than refusing a deployment that works.

    ``client.resolve_project_id`` and the interactive branch of ``cli.env_setup`` used to decide on
    the url alone, which answered this question twice and more narrowly: they refused a configured
    backend this accepts, and skipped the ownership check for one. Both now route through here, so
    this is the single classifier.

    Takes a plain ``str`` because ``load_credentials`` falls back to ``DEFAULT_API_URL`` and so
    never yields ``None``; accepting an optional here would invent a state no caller can reach.
    """
    if is_freesolo_hosted_url(api_url):
        return True
    # an operator-controlled backend is one that is NOT Freesolo's. pointing FREESOLO_BASE_URL at
    # the hosted service (or leaving it at that default) from a self-hosted plane would send the
    # plane's operator key to Freesolo as a bearer token -- the leak this guard exists to close.
    backend = os.environ.get("FREESOLO_BASE_URL", "").strip()
    return bool(backend) and not is_freesolo_hosted_url(backend)


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


def _unexpected_response(api_url: str, path: str, problem: str) -> ClientError:
    """The one error for a 2xx body this client cannot use.

    A body that is not JSON and valid JSON of the wrong shape are the same user state (something
    other than a Flash control plane answered), so both carry the hint ``flash login`` already
    gives, see ``_verify_key_against_plane``. Nothing in the CLI turns a raw
    ``json.JSONDecodeError`` or ``KeyError`` into an error message, so neither may escape.
    """
    return ClientError(
        f"{api_url}{path} {problem}. Check that --api-url points at your Flash control plane "
        '(its /v1/health should report "service": "flash") rather than at a proxy or another '
        "service."
    )


# re-export the extracted freesolo helpers from their established import location.
from flash.client.freesolo_api import _freesolo_get as _freesolo_get  # noqa: E402
from flash.client.freesolo_api import _freesolo_request as _freesolo_request  # noqa: E402
from flash.client.freesolo_api import create_project as create_project  # noqa: E402
from flash.client.freesolo_api import export_trace_records as export_trace_records  # noqa: E402
from flash.client.freesolo_api import get_project as get_project  # noqa: E402
from flash.client.freesolo_api import list_projects as list_projects  # noqa: E402
from flash.client.freesolo_api import list_trace_projects as list_trace_projects  # noqa: E402
from flash.client.freesolo_api import upload_eval_run as upload_eval_run  # noqa: E402
from flash.client.freesolo_api import verify_freesolo_key as verify_freesolo_key  # noqa: E402


def _validate_chat_messages(messages: list[dict]) -> None:
    if not isinstance(messages, list):
        raise ClientError("chat messages must be a list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ClientError(f"chat messages[{index}] must be an object")


def _parse_chat_target(target: str) -> tuple[str, str]:
    from flash.schema import parse_checkpoint_ref

    parsed = parse_checkpoint_ref(target)
    if parsed is None:
        raise ClientError("invalid checkpoint id: expected RUN_ID/final or RUN_ID/step-N")
    return parsed[0], target


def _prepare_chat_request(
    target: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    *,
    stream: bool = False,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
    parallel_tool_calls: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    base_run_id, checkpoint_id = _parse_chat_target(target)
    _validate_chat_messages(messages)
    validate_tool_control_presence(
        tools,
        tool_choice,
        parallel_tool_calls,
        error_type=ClientError,
    )
    body: dict[str, Any] = {
        "checkpoint_id": checkpoint_id,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stream:
        body["stream"] = True
    if tools is not None:
        body["tools"] = tools
        body["tool_choice"] = "auto" if tool_choice is None else tool_choice
        body["parallel_tool_calls"] = True if parallel_tool_calls is None else parallel_tool_calls
    return base_run_id, body


def _openai_sse_text(chunks: Iterator[str]) -> Iterator[str]:
    """render normalized openai events for cli and environment evaluation."""

    reasoning_open = False
    reasoning_done = False
    try:
        for event in iter_openai_sse_events(chunks):
            if isinstance(event, ErrorEvent):
                if reasoning_open:
                    reasoning_open = False
                    yield "</think>"
                raise ClientError(event.message)
            if not isinstance(event, DeltaEvent):
                continue
            reasoning = event.reasoning_content
            if reasoning is not None:
                if not reasoning_open and not (reasoning_done and not reasoning):
                    reasoning_open = True
                    yield "<think>"
                if reasoning:
                    yield reasoning
            content = event.content
            if content:
                if reasoning_open:
                    reasoning_open = False
                    reasoning_done = True
                    yield "</think>"
                yield content
    except OpenAISSEError as exc:
        if reasoning_open:
            yield "</think>"
        raise ClientError(str(exc)) from exc
    if reasoning_open:
        yield "</think>"


def _parse_adapter_target(target: str) -> tuple[str, int | None]:
    from flash.schema import parse_checkpoint_ref

    parsed = parse_checkpoint_ref(target)
    if parsed is None:
        raise ClientError("invalid checkpoint id: expected RUN_ID/final or RUN_ID/step-N")
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
            f"`{CLI_NAME} login`. Unset FREESOLO_API_KEY or update it to a valid freesolo API key."
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
            raise ServiceUnreachableError(
                f"cannot reach the Flash service at {self.api_url} ({exc.reason}); "
                "check your network connection and FLASH_API_URL"
            ) from exc
        except TimeoutError as exc:
            raise RequestTimeoutError(
                f"request to the Flash service at {self.api_url} timed out; "
                "check your network connection and FLASH_API_URL"
            ) from exc

    def _decode_response(
        self,
        path: str,
        raw: bytes,
        content_type: str = "",
        *,
        require: Mapping[str, RequireSpec] | None = None,
    ) -> Any:
        """Parse a 2xx body and require the top-level keys, and value types, the caller reads.

        Every response body this client reads goes through here, so a proxy answering
        ``200 text/html`` and a plane answering the wrong shape both surface as the same
        ``ClientError``. An empty body decodes to ``{}``.

        The type is required alongside the key because a present-but-unusable value is the same
        user state: ``{"logs": "x", "offset": null}`` passes a presence check and then raises a
        bare ``TypeError`` out of ``int(None)``, which nothing in the CLI translates either. A
        ``[dict]`` spec extends that one level into a list, because ``{"runs": [null]}`` is the
        same story one element down.
        """
        try:
            payload = json.loads(raw) if raw else {}
        except ValueError as exc:
            raise _unexpected_response(
                self.api_url,
                path,
                f"did not return JSON (Content-Type: {content_type or 'unset'})",
            ) from exc
        # every consumer of this decoder reads an object; a bare list or scalar from a proxy
        # or wrong service would otherwise surface as an AttributeError several frames later.
        if not isinstance(payload, dict):
            raise _unexpected_response(
                self.api_url,
                path,
                f"returned JSON that is not an object ({type(payload).__name__})",
            )
        bad = [
            key
            for key, expected in (require or {}).items()
            if key not in payload or not matches_require(payload[key], expected)
        ]
        if bad:
            raise _unexpected_response(
                self.api_url,
                path,
                "returned an unexpected response shape "
                f"(missing or malformed {', '.join(repr(key) for key in bad)})",
            )
        return payload

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        timeout: float | None = None,
        progress: ProgressCallback | None = None,
        extra_headers: dict[str, str] | None = None,
        require: Mapping[str, RequireSpec] | None = None,
        max_bytes: int | None = None,
        body_deadline: float | None = None,
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
        deadline = time.monotonic() + body_deadline if body_deadline is not None else None
        with (
            self._translate_http_errors(),
            _urlopen_no_redirect(
                req, timeout=_capped_timeout(timeout or self.timeout, deadline)
            ) as resp,
        ):
            raw = _read_response_body(
                resp, max_bytes=max_bytes, deadline=deadline, path=f"{self.api_url}{path}"
            )
            return self._decode_response(
                path, raw, resp.headers.get("Content-Type", ""), require=require
            )

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
            _urlopen_no_redirect(req, timeout=timeout or self.timeout) as resp,
        ):
            if max_bytes is not None:
                return _read_capped_response(resp, max_bytes)
            return resp.read()

    def me(self) -> dict:
        return self._request("GET", "/v1/me")

    def health(self) -> dict:
        return self._request("GET", "/v1/health", timeout=10.0)

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

    def list_envs(self) -> list[str]:
        """Return the published environment ids owned by the authenticated organization."""
        payload = self._request(
            "GET",
            "/v1/envs",
            timeout=ENV_LIST_CLIENT_TIMEOUT_SECONDS,
            max_bytes=ENV_LIST_MAX_RESPONSE_BYTES,
            body_deadline=ENV_LIST_CLIENT_TIMEOUT_SECONDS,
        )
        rows = payload.get("environments") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise ClientError("control plane returned a malformed environment list")
        # the ids are printed as values the user can paste straight into `[environment]`, so a
        # nonblank but unusable one (`my-env`, `acme/env`, unsafe path characters) would be
        # advertised as usable and then fail at submit. Validated with the managed parser itself
        # rather than a second predicate here, so the two cannot drift apart.
        from flash.envs.loading.loader import _parse_managed_environment_slug

        ids: list[str] = []
        for row in rows:
            env_id = row.get("id") if isinstance(row, dict) else None
            if not isinstance(env_id, str) or not env_id.strip():
                raise ClientError("control plane returned an environment without an id")
            env_id = env_id.strip()
            if _parse_managed_environment_slug(env_id) is None:
                raise ClientError(
                    f"control plane returned an unusable environment id {env_id!r}; "
                    "expected <namespace>/<project>/<name>"
                )
            ids.append(env_id)
        return ids

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
            require={"deleted": bool},
        )

    def download_env_package(self, env_id: str) -> bytes:
        """Download a managed environment package through the Flash control plane."""
        from flash.envs.loading import loader

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
            # server-side preview: runs the same validation and preflights as a real submit, records
            # a state=dry_run run, allocates no training gpu, and charges nothing. sft also reads the
            # pinned package's dataset file and rejects a missing or unreadable file before allocation.
            body["dry_run"] = True
        if client_train_schema is not None:
            body["client_train_schema"] = client_train_schema
        return self._request("POST", "/v1/runs", body=body, require={"run_id": str})

    def list_runs(self) -> list[dict]:
        return self._request("GET", "/v1/runs", require={"runs": [dict]})["runs"]

    def get_run(self, run_id: str) -> dict:
        return self._request("GET", f"/v1/runs/{run_id}")

    def get_logs(self, run_id: str, offset: int = 0) -> dict:
        return self._request(
            "GET",
            f"/v1/runs/{run_id}/logs?offset={int(offset)}",
            require={"logs": str, "offset": int},
        )

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
            return self._request(
                "POST", f"/v1/runs/{run_id}/cancel", timeout=60.0, require={"state": str}
            )
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
                    f"Run `{CLI_NAME} runs status {run_id}` to check the authoritative state before retrying."
                ) from cause
            time.sleep(2.0)

    def checkpoints(self, run_id: str) -> list[dict]:
        """Deployable per-step RL checkpoints for a run (serve one with `flash models deploy RUN/step-N`)."""
        return self._request(
            "GET", f"/v1/runs/{run_id}/checkpoints", require={"checkpoints": [dict]}
        )["checkpoints"]

    def deploy(
        self,
        run_id: str,
        dry_run: bool = False,
    ) -> dict:
        base_run_id, _ = _parse_adapter_target(run_id)
        # smoke verification is mandatory server-side; there is no opt-out to forward.
        body: dict = {
            "dry_run": dry_run,
            "checkpoint_id": run_id,
        }
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
        base_run_id, _ = _parse_adapter_target(run_id)
        body: dict = {
            "repository": repository,
            "hf_token": hf_token,
            "private": private,
            "checkpoint_id": run_id,
        }
        return self._request("POST", f"/v1/runs/{base_run_id}/export", body=body, timeout=30 * 60)

    def undeploy(self, checkpoint_id: str) -> dict:
        run_id, checkpoint_id = _parse_chat_target(checkpoint_id)
        quoted = urllib.parse.quote(checkpoint_id, safe="")
        return self._request("DELETE", f"/v1/runs/{run_id}/deploy?checkpoint_id={quoted}")

    def deployments(self, timeout: float | None = None) -> list[dict]:
        return self._request(
            "GET", "/v1/deployments", timeout=timeout, require={"deployments": [dict]}
        )["deployments"]

    def _serving_deployment(
        self,
        base_run_id: str,
        timeout: float | None,
        *,
        body_deadline: float | None = None,
    ) -> dict | None:
        """The run's servable deployment record, or None when it has none.

        Read from the run-scoped route rather than the listing. `/v1/deployments` walks every run
        the key owns and loads each one's status before a caller picks a single record out, so on
        an account with a long run history the poll's cost grows with that history and a wait can
        expire scanning unrelated runs while the requested revision is already ready
        `/v1/runs/{run_id}/deploy` resolves the one run directly.

        ``body_deadline`` additionally bounds the read in WALL-CLOCK time. ``timeout`` alone is a
        per-socket-operation bound, so a peer that trickles bytes just inside it can hold the read
        open indefinitely; a caller that must return within a fixed budget has to pass both (see
        ``env_list``). ``--wait`` polling deliberately does not: it owns a deadline spanning many
        reads and recomputes each one's share.
        """
        try:
            deployment = self._request(
                "GET",
                f"/v1/runs/{base_run_id}/deploy",
                timeout=timeout,
                body_deadline=body_deadline,
            )
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
        if not deployment.get("run_id"):
            # `models deploy --wait` prints this record in place of the POST body, so without the
            # id styled output renders an empty run field and the json omits it entirely.
            deployment = {**deployment, "run_id": base_run_id}
        return deployment

    def deployment_for(self, run_id: str, timeout: float | None = None) -> dict | None:
        """The deployment record for the checkpoint named in ``run_id``, or None.

        ``deploy`` returns as soon as the record is persisted, which is normally before the
        requested revision is servable. This is the read side a caller needs to tell "queued"
        from "actually ready".

        ``timeout`` bounds the single request. A caller polling against its own deadline needs
        that: the default client timeout is 60s, so one stalled read inside a `--wait 5` would
        overshoot the bound the user asked for by an order of magnitude.
        """
        base_run_id, _ = _parse_adapter_target(run_id)
        deployment = self._serving_deployment(base_run_id, timeout)
        if deployment is None:
            return None
        # the exact checkpoint identity must match the caller's requested target.
        if deployment.get("checkpoint_id") != run_id:
            return None
        return deployment

    def deployed_checkpoint(
        self,
        run_id: str,
        timeout: float | None = None,
        *,
        body_deadline: float | None = None,
    ) -> dict | None:
        """Whatever checkpoint the run serves right now, whichever step that is.

        The counterpart to ``deployment_for``, which answers the narrower "is MY revision live?"
        and hides any other. A caller about to deploy needs the opposite: the record it is about
        to displace is by definition the one the step filter drops.

        ``body_deadline`` bounds the whole read in wall-clock time; ``timeout`` alone bounds each
        socket operation, which a peer trickling bytes just inside it can extend without limit.
        """
        base_run_id, _ = _parse_adapter_target(run_id)
        return self._serving_deployment(base_run_id, timeout, body_deadline=body_deadline)

    def chat(
        self,
        run_id: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float | None = None,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> dict:
        base_run_id, body = _prepare_chat_request(
            run_id,
            messages,
            temperature,
            max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
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
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str]:
        if tools is not None:
            raise ValueError("decoded chat_stream does not support tools")
        base_run_id, body = _prepare_chat_request(
            run_id,
            messages,
            temperature,
            max_tokens,
            stream=True,
        )
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
            _urlopen_no_redirect(req, timeout=30 * 60) as resp,
        ):
            content_type = resp.headers.get("Content-Type", "")
            media_type = content_type.partition(";")[0].strip().lower()
            if media_type == "application/json":
                payload = self._decode_response(
                    f"/v1/runs/{base_run_id}/chat",
                    resp.read(),
                    content_type,
                    require={"choices": [dict]},
                )
                content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
                if content:
                    yield str(content)
                return
            read1 = getattr(resp, "read1", None)
            read = read1 if read1 is not None else resp.read
            read_size = 4096 if read1 is not None else 1

            def decoded_chunks() -> Iterator[str]:
                while raw := read(read_size):
                    state = decoder.getstate()
                    try:
                        decoded = decoder.decode(raw)
                    except UnicodeDecodeError as exc:
                        decoder.setstate(state)
                        prefix_end = max(0, exc.start - len(state[0]))
                        yield from decoder.decode(raw[:prefix_end])
                        # bind + re-raise explicitly: the yield above clears the active
                        # exception, so a bare `raise` here would fail with "No active
                        # exception to reraise".
                        raise exc
                    yield from decoded
                yield from decoder.decode(b"", final=True)

            try:
                chunks = decoded_chunks()
                if media_type == "text/event-stream":
                    yield from _openai_sse_text(chunks)
                else:
                    yield from chunks
            except (http.client.IncompleteRead, ConnectionError) as exc:
                # the server aborts the chunked response when the serving backend fails
                # mid-generation; urllib reports the missing terminating chunk (or reset) here.
                # translate it so the caller raises instead of treating the truncated text as a
                # finished answer.
                raise ClientError(
                    "chat stream ended unexpectedly before completion; the serving backend "
                    "likely failed mid-generation, so any partial output is truncated"
                ) from exc


def client_from_config(require_key: bool = True) -> ApiClient:
    """Build a client from the stored credentials; fail with a clear hint when logged out."""
    api_url, api_key, key_source = load_credentials_with_source()
    if require_key and not api_key:
        raise ClientError(
            f"not logged in — run `{CLI_NAME} login` with your freesolo API key (or set FREESOLO_API_KEY)"
        )
    return ApiClient(api_url, api_key, key_source=key_source)
