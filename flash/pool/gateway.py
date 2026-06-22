"""Async HTTP gateway to one vLLM upstream (the only place that touches the network).

The router holds a single :class:`BackendGateway` and calls it with a :class:`~flash.pool.state.Backend`
for every outbound op (health probe, load/unload LoRA, forward a generation). The ``httpx.AsyncClient``
is injectable so tests can dispatch to in-process fake-vLLM ASGI apps (no sockets) while production
uses a real pooled client. Errors are normalized to :class:`GatewayError` so the router's
retry/failover logic has one exception type to catch.
"""

from __future__ import annotations

import httpx

from flash.pool import protocol
from flash.pool.state import Backend


class GatewayError(RuntimeError):
    """A backend call failed (transport error or non-2xx)."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


class BackendGateway:
    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        *,
        request_timeout: float = 600.0,
        control_timeout: float = 120.0,
        health_timeout: float = 5.0,
    ):
        # request_timeout: generation can be long (large groups / long responses).
        # control_timeout: load/unload a LoRA (disk/remote pull). health: quick liveness probe.
        self._client = client or httpx.AsyncClient(timeout=request_timeout)
        self._owns_client = client is None
        self.request_timeout = request_timeout
        self.control_timeout = control_timeout
        self.health_timeout = health_timeout

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def health(self, be: Backend) -> bool:
        try:
            r = await self._client.get(
                protocol.join(be.url, protocol.HEALTH_PATH), timeout=self.health_timeout
            )
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    async def load_lora(self, be: Backend, lora_name: str, lora_path: str) -> None:
        await self._post(
            be,
            protocol.LOAD_LORA_PATH,
            protocol.load_lora_body(lora_name, lora_path),
            self.control_timeout,
            tolerate_already=True,
        )

    async def unload_lora(self, be: Backend, lora_name: str) -> None:
        await self._post(
            be,
            protocol.UNLOAD_LORA_PATH,
            protocol.unload_lora_body(lora_name),
            self.control_timeout,
            tolerate_already=True,
        )

    async def chat(self, be: Backend, body: dict) -> dict:
        return await self._post(be, protocol.CHAT_PATH, body, self.request_timeout)

    async def completions(self, be: Backend, body: dict) -> dict:
        return await self._post(be, protocol.COMPLETIONS_PATH, body, self.request_timeout)

    async def post(self, label: str, url: str, body: dict, timeout: float | None = None) -> dict:
        """POST JSON to an arbitrary URL (used for reward workers). ``label`` names the upstream in
        error messages."""
        return await self._post_url(label, url, body, timeout or self.request_timeout)

    async def _post(
        self,
        be: Backend,
        path: str,
        body: dict,
        timeout: float,
        *,
        tolerate_already: bool = False,
    ) -> dict:
        return await self._post_url(
            be.id, protocol.join(be.url, path), body, timeout, tolerate_already=tolerate_already
        )

    async def _post_url(
        self,
        label: str,
        url: str,
        body: dict,
        timeout: float,
        *,
        tolerate_already: bool = False,
    ) -> dict:
        try:
            r = await self._client.post(url, json=body, timeout=timeout)
        except httpx.HTTPError as e:  # transport-level (connect/read/timeout)
            raise GatewayError(f"{label} {url}: {type(e).__name__}: {e}") from e
        if r.status_code >= 400:
            # vLLM returns 400 "has already been loaded" / "not found" (the adapter, in its body) on
            # an idempotent re-load / double-unload; tolerate THOSE so reloads and double-unloads are
            # no-ops. But do NOT treat a bare 404 as success: on these endpoints a 404 means the
            # route itself is missing (wrong backend URL, or a vLLM without the dynamic-LoRA API) —
            # swallowing it would let the router believe an adapter is loaded and then fail every
            # generation. Only the message-bearing 400-style cases are benign.
            text = _safe_text(r)
            if (
                tolerate_already
                and r.status_code != 404
                and ("already" in text.lower() or "not found" in text.lower())
            ):
                return {"ok": True, "note": text[:200]}
            raise GatewayError(f"{label} {url}: HTTP {r.status_code}: {text[:300]}", status=r.status_code)
        try:
            return r.json()
        except ValueError:
            return {"ok": True}


def _safe_text(r: httpx.Response) -> str:
    try:
        return r.text
    except Exception:
        return ""
