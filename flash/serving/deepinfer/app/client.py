"""Hosted EnginePool bridge to the authenticated Modal DeepInfer web server."""

import asyncio
import json
import os
from functools import lru_cache
from urllib.parse import urlsplit

import httpx

from flash.serving.deepinfer.app.events import chat_body
from flash.serving.deepinfer.contract.config import (
    APP_NAME,
    MODEL_ID,
    REQUEST_SECONDS,
    STARTUP_SECONDS,
)
from flash.serving.deepinfer.contract.errors import DeepInferError

CONNECTION_KEYS = ("DEEPINFER_PROXY_KEY", "DEEPINFER_PROXY_SECRET")


@lru_cache(maxsize=1)
def endpoint_url():
    import modal

    # Resolve the fixed backend in the router's Modal environment. Credentials
    # remain secrets; neither the endpoint nor engine policy is a deploy-time knob.
    endpoint = modal.Function.from_name(APP_NAME, "serve").get_web_url()
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


class HostedDeepInfer:
    def __init__(self, *, transport=None):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(STARTUP_SECONDS + REQUEST_SECONDS, connect=30),
            transport=transport,
            follow_redirects=True,
            max_redirects=32,
            event_hooks={"response": [_check_redirect]},
        )

    async def aclose(self):
        await self._client.aclose()

    async def token_budget(self, payload):
        if not configured():
            raise DeepInferError(503, "DeepInfer backend is not configured")
        headers = {
            "Modal-Key": os.environ["DEEPINFER_PROXY_KEY"],
            "Modal-Secret": os.environ["DEEPINFER_PROXY_SECRET"],
        }
        url = await asyncio.wait_for(asyncio.to_thread(endpoint_url), timeout=30)
        response = await self._client.post(
            url + "/internal/token-budget",
            headers=headers,
            json=chat_body(payload.model_dump()),
        )
        if response.status_code in {400, 422}:
            try:
                detail = response.json()["detail"]
                if not isinstance(detail, str) or not detail:
                    raise ValueError("missing validation detail")
            except (ValueError, KeyError, TypeError) as exc:
                raise DeepInferError(
                    503, "DeepInfer returned malformed validation feedback"
                ) from exc
            raise DeepInferError(response.status_code, detail)
        response.raise_for_status()
        try:
            tokens = response.json()["tokens"]
            if type(tokens) is not int or tokens <= 0:
                raise ValueError("invalid token budget")
        except (ValueError, KeyError, TypeError) as exc:
            raise DeepInferError(503, "DeepInfer returned an invalid token budget") from exc
        return tokens

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
        }
        try:
            url = await asyncio.wait_for(asyncio.to_thread(endpoint_url), timeout=30)
        except Exception as exc:
            raise DeepInferError(503, "DeepInfer endpoint discovery failed") from exc
        url += "/internal/generate"
        terminal = False
        try:
            # A disconnect closes only this response; other streams keep their pooled connections.
            async with (
                asyncio.timeout(STARTUP_SECONDS + REQUEST_SECONDS),
                self._client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json={
                        "chat": body,
                        "request_id": payload.generation_id,
                        "cache_namespace": getattr(payload, "_cache_namespace", None),
                    },
                ) as response,
            ):
                if response.status_code != 200:
                    code = response.status_code if response.status_code in {400, 422, 429} else 503
                    raise DeepInferError(code, "DeepInfer backend rejected the request")
                async for line in response.aiter_lines():
                    if not line:
                        continue
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
