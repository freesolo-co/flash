"""Stdlib HTTP client for the Flash control plane (no extra dependencies).

Every CLI operation maps to one method here. Server errors (FastAPI's
``{"detail": ...}``) surface as ``ApiError`` with the server's message; connection
problems surface as ``ClientError`` with an actionable hint.
"""

from __future__ import annotations

import codecs
import contextlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from .config import load_credentials_with_source

# Called as ``progress(bytes_sent, total_bytes)`` as a request body streams to the server, so
# the CLI can draw an upload bar. ``total_bytes`` is the full Content-Length, fixed up front.
ProgressCallback = Callable[[int, int], None]


class ClientError(RuntimeError):
    """Expected client-side errors (no key, unreachable server) — printed cleanly."""


class ApiError(ClientError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# Login is handled by the freesolo backend (not the flash control plane): `flash login`
# verifies the user's freesolo API key here. The same key authenticates the flash
# control plane, which accepts freesolo-issued keys.
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
    """Verify a freesolo API key against the freesolo backend's ``/api/auth/verify``.

    Raises :class:`ClientError`/:class:`ApiError` if the key is rejected or the backend is
    unreachable; returns ``None`` on success. Keys are issued from the freesolo sign-in page.
    """
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
    """A read()-only file-like over an in-memory payload that reports bytes consumed.

    ``http.client`` sends a body exposing ``read()`` in blocksize chunks; we forward the running
    total to ``progress(sent, total)`` for each chunk so the CLI can draw an upload bar. The
    caller sets Content-Length from ``len(payload)``, so the request is NOT chunked-encoded and
    the server reads it exactly as a plain bytes body."""

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

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        timeout: float | None = None,
    ) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.api_url}{path}",
            method=method,
            data=json.dumps(body).encode() if body is not None else None,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = self._auth_error_detail(exc.code, _detail_from_http_error(exc))
            raise ApiError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise ClientError(
                f"cannot reach the Flash service at {self.api_url} ({exc.reason}); "
                "check your network connection and FLASH_API_URL"
            ) from exc

    def _post_with_progress(
        self,
        path: str,
        body: dict,
        *,
        progress: ProgressCallback,
        timeout: float,
    ) -> Any:
        """POST a JSON body while reporting upload progress (see :class:`_ProgressReader`).

        Same error mapping as :meth:`_request`; kept separate because the body is a streaming
        reader with an explicit Content-Length rather than a one-shot bytes payload."""
        payload = json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "Content-Length": str(len(payload))}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.api_url}{path}",
            method="POST",
            data=_ProgressReader(payload, progress),
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = self._auth_error_detail(exc.code, _detail_from_http_error(exc))
            raise ApiError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise ClientError(
                f"cannot reach the Flash service at {self.api_url} ({exc.reason}); "
                "check your network connection and FLASH_API_URL"
            ) from exc

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
        """Upload a packaged Freesolo environment to the managed Environments Hub.

        When ``progress`` is given the body streams to the server in chunks and
        ``progress(bytes_sent, total_bytes)`` fires for each, so the CLI can render an upload
        bar; otherwise the body is sent in one shot (the default, used off a TTY)."""
        body = {"name": name, "package_b64": package_b64}
        if progress is None:
            return self._request("POST", "/v1/envs", body=body, timeout=1800.0)
        return self._post_with_progress("/v1/envs", body, progress=progress, timeout=1800.0)

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
        # The train-subprocess console/traceback ({console_<phase>.txt, error_<phase>.txt}) from the
        # run's HF artifact repo, fetched server-side with the operator token — the real worker
        # output the offset-paged log can't carry. Kept off the hot get_logs poll path. {} if none.
        #
        # Tolerate a managed server that predates the /worker route: a CLI upgraded ahead of the
        # service rollout would otherwise hard-fail. FastAPI returns a bare 404 "Not Found" for an
        # unmatched path -> treat ONLY that as "no worker output" ({}); real 404s still surface (an
        # unknown run_id carries detail "unknown run_id: ...", not "Not Found").
        try:
            return self._request("GET", f"/v1/runs/{run_id}/worker").get("worker", {})
        except ApiError as exc:
            if exc.status == 404 and str(exc).strip().lower() == "not found":
                return {}
            raise

    def cancel_run(self, run_id: str) -> dict:
        return self._request("POST", f"/v1/runs/{run_id}/cancel")

    def checkpoints(self, run_id: str) -> list[dict]:
        """Deployable per-step RL checkpoints for a run (each `flash deploy --step N`-able)."""
        return self._request("GET", f"/v1/runs/{run_id}/checkpoints")["checkpoints"]

    def deploy(
        self,
        run_id: str,
        dry_run: bool = False,
        step: int | None = None,
    ) -> dict:
        # Deploy blocks on registration and serving warmup, which can take many minutes.
        deploy_timeout = 30 * 60 if not dry_run else None
        body: dict = {"dry_run": dry_run}
        if step is not None:
            # Deploy a specific intermediate checkpoint instead of the run's final adapter.
            # Reject a bool explicitly: `int(True)`/`int(False)` would silently coerce to step
            # 1/0, but the server guard (_resolve_deploy_step) treats a bool as an invalid step
            # and 400s — so fail fast here with a clear client-side error instead of sending a
            # bogus 0/1 that the server rejects (or, worse, that hits a real checkpoint 0/1).
            if isinstance(step, bool):
                raise ClientError(f"invalid checkpoint step: {step!r} (must be an integer)")
            body["step"] = int(step)
        return self._request(
            "POST",
            f"/v1/runs/{run_id}/deploy",
            body=body,
            timeout=deploy_timeout,
        )

    def export(
        self,
        run_id: str,
        *,
        repository: str,
        hf_token: str,
        step: int | None = None,
        private: bool = True,
    ) -> dict:
        """Export a run's trained adapter into a user-owned HuggingFace repo.

        Copies the adapter (or a specific ``--step`` checkpoint) from the platform's private
        artifact repo into ``repository``, authenticated with the user's ``hf_token`` (write
        access to their own repo). The server downloads then re-uploads the adapter, which can
        take a while for a large adapter, so the timeout matches deploy's."""
        body: dict = {"repository": repository, "hf_token": hf_token, "private": private}
        if step is not None:
            # Reject a bool explicitly: int(True)/int(False) would silently coerce to step 1/0,
            # but the server guard treats a bool as an invalid step and 400s — fail fast here
            # with a clear client-side error instead (matches deploy()'s bool guard).
            if isinstance(step, bool):
                raise ClientError(f"invalid checkpoint step: {step!r} (must be an integer)")
            # Reject a FRACTIONAL step before int() silently truncates it (e.g. 2.7 -> 2 would export
            # the wrong checkpoint). An integral float (2.0) is fine.
            if isinstance(step, float) and not step.is_integer():
                raise ClientError(
                    f"invalid checkpoint step: {step!r} (must be a whole number, not fractional)"
                )
            body["step"] = int(step)
        return self._request(
            "POST", f"/v1/runs/{run_id}/export", body=body, timeout=30 * 60
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
    ) -> dict:
        # Serving warmup can take minutes; give inference a generous timeout.
        return self._request(
            "POST",
            f"/v1/runs/{run_id}/chat",
            body={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            timeout=30 * 60,
        )

    def chat_stream(
        self,
        run_id: str,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.api_url}/v1/runs/{run_id}/chat",
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
        try:
            with urllib.request.urlopen(req, timeout=30 * 60) as resp:
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
        except urllib.error.HTTPError as exc:
            detail = self._auth_error_detail(exc.code, _detail_from_http_error(exc))
            raise ApiError(exc.code, detail) from exc
        except urllib.error.URLError as exc:
            raise ClientError(
                f"cannot reach the Flash service at {self.api_url} ({exc.reason}); "
                "check your network connection and FLASH_API_URL"
            ) from exc


def client_from_config(require_key: bool = True) -> ApiClient:
    """Build a client from the stored credentials; fail with a clear hint when logged out."""
    api_url, api_key, key_source = load_credentials_with_source()
    if require_key and not api_key:
        raise ClientError(
            "not logged in — run `flash login` with your freesolo API key (or set FREESOLO_API_KEY)"
        )
    return ApiClient(api_url, api_key, key_source=key_source)
