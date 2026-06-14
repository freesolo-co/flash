"""Stdlib HTTP client for the AutoSLM control plane (no extra dependencies).

Every CLI/MCP operation maps to one method here. Server errors (FastAPI's
``{"detail": ...}``) surface as ``ApiError`` with the server's message; connection
problems surface as ``ClientError`` with an actionable hint.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .config import load_credentials


class ClientError(RuntimeError):
    """Expected client-side errors (no key, unreachable server) — printed cleanly."""


class ApiError(ClientError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class ApiClient:
    def __init__(self, api_url: str, api_key: str | None = None, timeout: float = 60.0):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

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
            raw = exc.read()
            try:
                detail = json.loads(raw).get("detail") or raw.decode()
            except (ValueError, AttributeError):
                detail = raw.decode(errors="replace") if raw else str(exc)
            raise ApiError(exc.code, str(detail)) from exc
        except urllib.error.URLError as exc:
            raise ClientError(
                f"cannot reach the AutoSLM service at {self.api_url} ({exc.reason}); "
                "check your network, or point AUTOSLM_API_URL at your control plane "
                "(see docs/self-hosting.md)"
            ) from exc

    # -- keys / identity ---------------------------------------------------------------
    def claim_key(self, email: str | None = None) -> dict:
        return self._request("POST", "/v1/keys", body={"email": email} if email else {})

    def me(self) -> dict:
        return self._request("GET", "/v1/me")

    def health(self) -> dict:
        return self._request("GET", "/v1/health", timeout=10.0)

    # -- catalog -----------------------------------------------------------------------
    def models(self, include_experimental: bool = False) -> list[dict]:
        path = "/v1/models" + ("?include_experimental=true" if include_experimental else "")
        return self._request("GET", path)["models"]

    # -- runs --------------------------------------------------------------------------
    def create_run(self, spec: dict, dry_run: bool = False) -> dict:
        return self._request("POST", "/v1/runs", body={"spec": spec, "dry_run": dry_run})

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
        mode: str = "dev",
        idle_timeout_s: int = 300,
        dry_run: bool = False,
    ) -> dict:
        # always-on blocks on the server until the worker has downloaded the
        # model/adapter and vLLM is healthy (the no-cold-start guarantee), which can
        # take many minutes — use the serve-scale timeout, not the default 60s.
        deploy_timeout = 30 * 60 if (mode == "always-on" and not dry_run) else None
        return self._request(
            "POST",
            f"/v1/runs/{run_id}/deploy",
            body={"mode": mode, "idle_timeout_s": idle_timeout_s, "dry_run": dry_run},
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
        # Cold starts in dev mode can take minutes; give inference a generous timeout.
        return self._request(
            "POST",
            f"/v1/runs/{run_id}/chat",
            body={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
            timeout=30 * 60,
        )


def client_from_config(require_key: bool = True) -> ApiClient:
    """Build a client from the stored credentials; fail with a clear hint when logged out."""
    api_url, api_key = load_credentials()
    if require_key and not api_key:
        raise ClientError(
            "not logged in — run `slm login` to claim your AutoSLM API key (or set AUTOSLM_API_KEY)"
        )
    return ApiClient(api_url, api_key)
