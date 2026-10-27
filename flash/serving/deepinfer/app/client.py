"""Hosted EnginePool bridge to the authenticated Modal DeepInfer web server."""

import asyncio
import json
import os
from functools import lru_cache
from urllib.parse import urlsplit

import httpx

from flash.serving.deepinfer.app.endpoints import replica_endpoints
from flash.serving.deepinfer.app.events import chat_body
from flash.serving.deepinfer.app.retries import open_generation
from flash.serving.deepinfer.contract.config import (
    APP_NAME,
    MODEL_ID,
    REQUEST_SECONDS,
    STARTUP_SECONDS,
    STREAM_IDLE_SECONDS,
)
from flash.serving.deepinfer.contract.errors import DeepInferError

CONNECTION_KEYS = ("DEEPINFER_PROXY_KEY", "DEEPINFER_PROXY_SECRET")


@lru_cache(maxsize=1)
def endpoint_url():
    import modal

    # Resolve the fixed backend in the router's Modal environment. Credentials
    # remain secrets; neither the endpoint nor engine policy is a deploy-time knob.
    endpoint = modal.Server.from_name(APP_NAME, "Engine").get_url()
    url = urlsplit(endpoint or "")
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise DeepInferError(503, "DeepInfer backend has no valid HTTPS endpoint")
    return endpoint.rstrip("/")


async def _check_redirect(response):
    # Modal returns 303 result URLs every 150 seconds during a long cold start.
    # Never forward custom proxy credentials to a different origin.
    if response.is_redirect:
        target = response.url.join(response.headers.get("location", ""))
        source = response.url
        if response.status_code != 303 or (target.scheme, target.host, target.port) != (
            source.scheme,
            source.host,
            source.port,
        ):
            raise DeepInferError(502, "DeepInfer returned an unsafe result redirect")


def configured():
    values = [os.environ.get(key, "").strip() for key in CONNECTION_KEYS]
    if any(values) and not all(values):
        raise ValueError("DeepInfer requires both Modal proxy credentials")
    return all(values)


async def _stream_lines(response, deadline):
    # The consumer may hand this iterator to another task after its first event.
    # Keep each timeout inside one anext(), never open across a yield.
    loop = asyncio.get_running_loop()
    lines = response.aiter_lines()
    try:
        while True:
            if loop.time() >= deadline:
                raise TimeoutError("DeepInfer stream deadline exceeded")
            async with asyncio.timeout_at(min(deadline, loop.time() + STREAM_IDLE_SECONDS)):
                try:
                    line = await anext(lines)
                except StopAsyncIteration:
                    return
            yield line
    finally:
        await lines.aclose()


class HostedDeepInfer:
    def __init__(self, *, transport=None, targets=None):
        self._targets = targets or replica_endpoints
        self._next_replica = 0
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(STARTUP_SECONDS + REQUEST_SECONDS, connect=30),
            transport=transport,
            follow_redirects=True,
            max_redirects=32,
            event_hooks={"response": [_check_redirect]},
        )

    async def aclose(self):
        await self._client.aclose()

    async def stream_generate(self, base_model, payload, record, *, expected_checkpoint=None):
        if base_model != MODEL_ID or not record.serve_base_model:
            raise ValueError("DeepInfer does not support LoRA adapters")
        if expected_checkpoint is not None:
            raise ValueError("checkpoint mismatch: DeepInfer serves base weights only")
        body = chat_body(payload.model_dump())
        if not payload.generation_id:
            raise ValueError("generation id is required")
        if not configured():
            raise DeepInferError(503, "DeepInfer backend is not configured")
        headers = {
            "Modal-Key": os.environ["DEEPINFER_PROXY_KEY"],
            "Modal-Secret": os.environ["DEEPINFER_PROXY_SECRET"],
            "Accept": "text/event-stream",
        }
        terminal = False
        deadline = asyncio.get_running_loop().time() + STARTUP_SECONDS + REQUEST_SECONDS
        try:
            # A disconnect closes only this response; other streams keep their pooled connections.
            async with asyncio.timeout_at(deadline):
                start = self._next_replica
                self._next_replica += 1
                response = await open_generation(
                    self._client,
                    self._targets,
                    headers,
                    {
                        "chat": body,
                        "request_id": payload.generation_id,
                        "cache_namespace": getattr(payload, "_cache_namespace", None),
                    },
                    deadline,
                    start=start,
                )
            try:
                if response.status_code != 200:
                    code = response.status_code if response.status_code in {400, 422, 429} else 503
                    raise DeepInferError(code, "DeepInfer backend rejected the request")
                async for line in _stream_lines(response, deadline):
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        line = line[5:].lstrip()
                    event = json.loads(line)
                    if event.get("type") == "error":
                        raise DeepInferError(502, "DeepInfer generation failed")
                    if event.get("request_id") != payload.generation_id:
                        raise DeepInferError(502, "DeepInfer returned a mismatched generation id")
                    if terminal:
                        raise DeepInferError(502, "DeepInfer returned data after completion")
                    terminal = event.get("type") == "final"
                    yield event
                if not terminal:
                    raise DeepInferError(502, "DeepInfer stream ended before completion")
            finally:
                await response.aclose()
        except (httpx.HTTPError, ValueError, TimeoutError) as exc:
            raise DeepInferError(502, "DeepInfer backend transport failed") from exc

    async def generate(self, base_model, payload, record, *, expected_checkpoint=None):
        stream = self.stream_generate(
            base_model, payload, record, expected_checkpoint=expected_checkpoint
        )
        result = None
        try:
            async for event in stream:
                if event["type"] == "final":
                    result = event
        finally:
            await stream.aclose()
        if result is None:
            raise DeepInferError(502, "DeepInfer did not return a final result")
        return result
