"""Stdlib HTTP client for the Flash control plane (no extra dependencies).

Every CLI/MCP operation maps to one method here. Server errors (FastAPI's
``{"detail": ...}``) surface as ``ApiError`` with the server's message; connection
problems surface as ``ClientError`` with an actionable hint.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .config import load_credentials_with_source


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

    # -- identity ----------------------------------------------------------------------
    def me(self) -> dict:
        return self._request("GET", "/v1/me")

    def health(self) -> dict:
        return self._request("GET", "/v1/health", timeout=10.0)

    # -- environments ------------------------------------------------------------------
    def publish_env(self, *, name: str, package_b64: str) -> dict:
        """Upload a packaged Freesolo environment to the managed Environments Hub."""
        return self._request(
            "POST",
            "/v1/envs",
            body={"name": name, "package_b64": package_b64},
            timeout=1800.0,
        )

    # -- runs --------------------------------------------------------------------------
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

    def cancel_run(self, run_id: str) -> dict:
        return self._request("POST", f"/v1/runs/{run_id}/cancel")

    # -- serving -----------------------------------------------------------------------
    def deploy(
        self,
        run_id: str,
        dry_run: bool = False,
    ) -> dict:
        # Deploy blocks on registration and serving warmup, which can take many minutes.
        deploy_timeout = 30 * 60 if not dry_run else None
        return self._request(
            "POST",
            f"/v1/runs/{run_id}/deploy",
            body={"dry_run": dry_run},
            timeout=deploy_timeout,
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


def client_from_config(require_key: bool = True) -> ApiClient:
    """Build a client from the stored credentials; fail with a clear hint when logged out."""
    api_url, api_key, key_source = load_credentials_with_source()
    if require_key and not api_key:
        raise ClientError(
            "not logged in — run `flash login` with your freesolo API key (or set FREESOLO_API_KEY)"
        )
    return ApiClient(api_url, api_key, key_source=key_source)
