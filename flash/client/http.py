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

from .config import load_credentials_with_source

ProgressCallback = Callable[[int, int], None]


class ClientError(RuntimeError):
    """Expected client-side errors (no key, unreachable server) — printed cleanly."""


class RequestTimeoutError(ClientError):
    """A request timed out before the control plane returned a response."""


class ApiError(ClientError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


DEFAULT_FREESOLO_BASE_URL = "https://api.freesolo.co"
FREESOLO_AUTH_VERIFY_PATH = "/api/auth/verify"


def freesolo_base_url(override: str | None = None) -> str:
    return (override or os.environ.get("FREESOLO_BASE_URL") or DEFAULT_FREESOLO_BASE_URL).rstrip(
        "/"
    )


def _detail_from_http_error(exc: urllib.error.HTTPError) -> str:
    """Extract the server's error message from an HTTPError body (FastAPI ``detail``)."""
    body = exc.read()
    try:
        detail = json.loads(body).get("detail") or body.decode()
    except (ValueError, AttributeError):
        detail = body.decode(errors="replace") if body else str(exc)
    return str(detail)


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
        raise ApiError(exc.code, _detail_from_http_error(exc)) from exc
    except urllib.error.URLError as exc:
        raise ClientError(
            f"cannot reach the freesolo backend at {base} ({exc.reason}); "
            "check your network connection and FREESOLO_BASE_URL"
        ) from exc


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


def _validate_chat_messages(messages: list[dict]) -> None:
    if not isinstance(messages, list):
        raise ClientError("chat messages must be a list")
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ClientError(f"chat messages[{index}] must be an object")


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

    def _auth_error_detail(self, status: int, detail: str) -> str:
        if status not in {401, 403} or self.key_source != "FREESOLO_API_KEY":
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
            raise ApiError(exc.code, detail) from exc
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
    ) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
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

    def me(self) -> dict:
        return self._request("GET", "/v1/me")

    def health(self) -> dict:
        return self._request("GET", "/v1/health", timeout=10.0)

    def publish_env(
        self,
        *,
        name: str,
        package_b64: str,
        progress: ProgressCallback | None = None,
    ) -> dict:
        """Upload a packaged Freesolo environment to the managed Environments Hub."""
        body = {"name": name, "package_b64": package_b64}
        return self._request("POST", "/v1/envs", body=body, timeout=1800.0, progress=progress)

    def delete_env(self, env_id: str) -> dict:
        """Delete a published Freesolo environment by its ``namespace/name`` id.

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
        quoted = urllib.parse.quote(env_id, safe="/")
        return self._request("DELETE", f"/v1/envs/{quoted}", timeout=1800.0)

    def create_run(self, spec: dict, runtime_secrets: dict[str, str] | None = None) -> dict:
        body = {"spec": spec}
        if runtime_secrets:
            body["runtime_secrets"] = runtime_secrets
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
            with contextlib.suppress(ClientError):
                status = self.get_run(run_id)
                last_state = str(status.get("state") or "unknown")
                if last_state in {"cancelled", "done", "failed", "dry_run"}:
                    return status
            if time.monotonic() >= deadline:
                raise ClientError(
                    f"cancel request timed out before confirmation; latest state={last_state!r}. "
                    f"Run `flash status {run_id}` to check the authoritative state before retrying."
                ) from cause
            time.sleep(2.0)

    def checkpoints(self, run_id: str) -> list[dict]:
        """Deployable per-step RL checkpoints for a run (serve one with `flash deploy RUN/step-N`)."""
        return self._request("GET", f"/v1/runs/{run_id}/checkpoints")["checkpoints"]

    def deploy(
        self,
        run_id: str,
        dry_run: bool = False,
        verify: bool = True,
    ) -> dict:
        from flash.schema import parse_checkpoint_ref

        parsed = parse_checkpoint_ref(run_id)
        if parsed is None:
            raise ClientError(
                "invalid adapter id: expected RUN_ID for the final adapter or RUN_ID/step-N "
                "for a saved checkpoint"
            )
        base_run_id, step = parsed
        body: dict = {"dry_run": dry_run, "verify": verify}
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
        from flash.schema import parse_checkpoint_ref

        parsed = parse_checkpoint_ref(run_id)
        if parsed is None:
            raise ClientError(
                "invalid adapter id: expected RUN_ID for the final adapter or RUN_ID/step-N "
                "for a saved checkpoint"
            )
        base_run_id, step = parsed
        body: dict = {"repository": repository, "hf_token": hf_token, "private": private}
        if step is not None:
            body["step"] = step
        return self._request(
            "POST", f"/v1/runs/{base_run_id}/export", body=body, timeout=30 * 60
        )

    def undeploy(self, run_id: str) -> dict:
        return self._request("DELETE", f"/v1/runs/{run_id}/deploy")

    def deployments(self) -> list[dict]:
        return self._request("GET", "/v1/deployments")["deployments"]

    def chat(
        self,
        run_id: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float | None = None,
    ) -> dict:
        from flash.schema import parse_checkpoint_ref

        parsed = parse_checkpoint_ref(run_id)
        if parsed is None:
            raise ClientError(
                "invalid run id: expected RUN_ID or RUN_ID/step-N for a deployed checkpoint"
            )
        base_run_id, _step = parsed
        _validate_chat_messages(messages)
        return self._request(
            "POST",
            f"/v1/runs/{base_run_id}/chat",
            body={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            timeout=timeout if timeout is not None else 30 * 60,
        )

    def chat_stream(
        self,
        run_id: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        from flash.schema import parse_checkpoint_ref

        parsed = parse_checkpoint_ref(run_id)
        if parsed is None:
            raise ClientError(
                "invalid run id: expected RUN_ID or RUN_ID/step-N for a deployed checkpoint"
            )
        base_run_id, _step = parsed
        _validate_chat_messages(messages)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.api_url}/v1/runs/{base_run_id}/chat",
            method="POST",
            data=json.dumps(
                {
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                }
            ).encode(),
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
                content = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content"))
                if content:
                    yield str(content)
                return
            while raw := resp.read(1):
                chunk = decoder.decode(raw)
                if chunk:
                    yield chunk
            tail = decoder.decode(b"", final=True)
            if tail:
                yield tail


def client_from_config(require_key: bool = True) -> ApiClient:
    """Build a client from the stored credentials; fail with a clear hint when logged out."""
    api_url, api_key, key_source = load_credentials_with_source()
    if require_key and not api_key:
        raise ClientError(
            "not logged in — run `flash login` with your freesolo API key (or set FREESOLO_API_KEY)"
        )
    return ApiClient(api_url, api_key, key_source=key_source)
