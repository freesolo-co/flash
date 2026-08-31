"""authenticated FastAPI surface for one immutable packaged serving runtime."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from collections.abc import Awaitable
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from flash.serve.contract.protocol import MAX_CHAT_REQUEST_BYTES, reject_non_finite_json_constant
from flash.serve.request.tool_calls import qualified_tool_parser
from flash.serve.runtime import (
    AdapterNotFoundError,
    EngineDeadError,
    PromptError,
    RuntimeConfigurationError,
    RuntimeNotReadyError,
    ServingRuntimeError,
    StreamReady,
)

from .bootstrap import ServingBootstrap
from .chat_stream import close_iterator, stream_chat_body
from .openai import (
    OpenAIRequestError,
    nonstream_response,
    parse_chat_request,
    provenance_payload,
)

_REJECTED_AUTH_DIGEST = hashlib.sha256(b"flash-rejected-authorization").digest()


async def _stream_body(
    event_stream: Any,
    ready: StreamReady,
    resolved: Any,
    provenance: dict[str, Any],
    *,
    include_usage: bool,
    choice_count: int = 1,
):
    """preserve the historical packaged stream helper seam."""

    async for chunk in stream_chat_body(
        event_stream,
        ready,
        resolved,
        provenance,
        choice_count=choice_count,
        include_usage=include_usage,
    ):
        yield chunk


class _RequestBodyTooLarge(ValueError):
    """the observed request body exceeded the application byte ceiling."""


class _HttpState:
    __slots__ = ("auth_digest", "bootstrap")

    def __init__(self, bootstrap: ServingBootstrap, auth_digest: bytes) -> None:
        self.bootstrap = bootstrap
        self.auth_digest = auth_digest


def create_app(
    bootstrap: ServingBootstrap,
    *,
    bearer_token: str | None = None,
    bearer_digest: str | None = None,
) -> FastAPI:
    """create an app retaining only the credential digest and immutable bootstrap owner."""

    digest = _auth_digest(bearer_token=bearer_token, bearer_digest=bearer_digest)
    state = _HttpState(bootstrap, digest)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        ok = state.bootstrap.ready
        return JSONResponse({"ok": ok}, status_code=200 if ok else 503)

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        unauthorized = _authorize(request, state)
        if unauthorized is not None:
            return unauthorized
        if not state.bootstrap.ready:
            return _error(503, "service_unavailable", "serving runtime is not ready")
        data = []
        for model_id in sorted(state.bootstrap.models):
            resolved = state.bootstrap.models[model_id]
            data.append(
                {
                    "id": model_id,
                    "object": "model",
                    "owned_by": "flash",
                    "flash_provenance": provenance_payload(state.bootstrap.manifest, resolved),
                }
            )
        return JSONResponse({"object": "list", "data": data})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        unauthorized = _authorize(request, state)
        if unauthorized is not None:
            return unauthorized
        if not state.bootstrap.ready:
            return _error(503, "service_unavailable", "serving runtime is not ready")
        try:
            payload = _strict_json(await _read_request_body(request))
        except _RequestBodyTooLarge:
            # both rejection branches stop before the body is drained, so the unread bytes would
            # corrupt the next request on a reused connection. closing is what makes them
            # unreachable, matching the hosted middleware's handling of the same case.
            return _error(
                413,
                "request_too_large",
                "request body exceeds the byte limit",
                headers={"connection": "close"},
            )
        except ValueError:
            return _error(400, "invalid_json", "request body is not valid json")
        if type(payload) is not dict:
            return _error(422, "invalid_request", "request body must be an object")
        model = payload.get("model")
        if type(model) is not str or not model:
            return _error(422, "invalid_request", "model is required")
        resolved = state.bootstrap.models.get(model)
        if resolved is None:
            return _error(404, "model_not_found", "requested model is not deployed")
        try:
            parsed = parse_chat_request(
                payload,
                resolved,
                tool_parser=qualified_tool_parser(state.bootstrap.manifest.logical_base_model),
            )
        except (OpenAIRequestError, PromptError, RuntimeConfigurationError, ValueError):
            return _error(422, "invalid_request", "request validation failed")
        provenance = provenance_payload(state.bootstrap.manifest, resolved)
        headers = {
            f"x-flash-{key.replace('_', '-')}": str(value) for key, value in provenance.items()
        }
        if not parsed.stream:
            try:
                result = await _await_until_disconnect(
                    request,
                    state.bootstrap.runtime.generate(parsed.generation),
                )
            except AdapterNotFoundError:
                return _error(404, "model_not_found", "requested model is not deployed")
            except (PromptError, RuntimeConfigurationError):
                return _error(400, "invalid_request", "generation request was rejected")
            except (EngineDeadError, RuntimeNotReadyError, ServingRuntimeError):
                return _error(503, "service_unavailable", "generation service is unavailable")
            except Exception:
                return _error(503, "service_unavailable", "generation service is unavailable")
            if (
                result.adapter_id != resolved.adapter.checkpoint_id
                or result.incarnation != resolved.adapter.aggregate_sha256
                or result.thinking != resolved.adapter.thinking_default
                or result.finish_reason is None
            ):
                return _error(503, "service_unavailable", "generation identity is invalid")
            return JSONResponse(
                nonstream_response(result, state.bootstrap.manifest, resolved),
                headers=headers,
            )

        event_stream = state.bootstrap.runtime.stream(parsed.generation)
        try:
            first = await _await_until_disconnect(request, anext(event_stream))
        except asyncio.CancelledError:
            await close_iterator(event_stream)
            raise
        except AdapterNotFoundError:
            await close_iterator(event_stream)
            return _error(404, "model_not_found", "requested model is not deployed")
        except (PromptError, RuntimeConfigurationError):
            await close_iterator(event_stream)
            return _error(400, "invalid_request", "generation request was rejected")
        except (EngineDeadError, RuntimeNotReadyError, ServingRuntimeError, Exception):
            await close_iterator(event_stream)
            return _error(503, "service_unavailable", "generation service is unavailable")
        if (
            type(first) is not StreamReady
            or first.adapter_id != resolved.adapter.checkpoint_id
            or first.incarnation != resolved.adapter.aggregate_sha256
            or first.thinking != resolved.adapter.thinking_default
        ):
            await close_iterator(event_stream)
            return _error(503, "service_unavailable", "generation stream did not become ready")
        body = _stream_body(
            event_stream,
            first,
            resolved,
            provenance,
            choice_count=parsed.generation.n,
            include_usage=parsed.include_usage,
        )
        return StreamingResponse(
            body,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                **headers,
            },
        )

    return app


def _auth_digest(*, bearer_token: str | None, bearer_digest: str | None) -> bytes:
    if (bearer_token is None) == (bearer_digest is None):
        raise ValueError("provide exactly one bearer token or bearer digest")
    if bearer_token is not None:
        if type(bearer_token) is not str or not bearer_token:
            raise ValueError("bearer token must be nonempty")
        return hashlib.sha256(bearer_token.encode("utf-8")).digest()
    assert bearer_digest is not None
    if (
        type(bearer_digest) is not str
        or len(bearer_digest) != 64
        or any(character not in "0123456789abcdef" for character in bearer_digest)
    ):
        raise ValueError("bearer digest must be an exact lowercase sha-256 digest")
    return bytes.fromhex(bearer_digest)


def _authorize(request: Request, state: _HttpState) -> JSONResponse | None:
    headers = request.headers.getlist("authorization")
    valid = False
    candidate = _REJECTED_AUTH_DIGEST
    if len(headers) == 1:
        parts = headers[0].split(" ")
        if (
            len(parts) == 2
            and parts[0].casefold() == "bearer"
            and parts[1]
            and not any(character.isspace() for character in parts[1])
        ):
            valid = True
            candidate = hashlib.sha256(parts[1].encode("utf-8")).digest()
    matched = hmac.compare_digest(candidate, state.auth_digest)
    if valid and matched:
        return None
    return JSONResponse(
        {"error": {"message": "authentication required", "type": "authentication_error"}},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
    )


async def _read_request_body(request: Request) -> bytes:
    lengths = request.headers.getlist("content-length")
    if len(lengths) == 1 and _decimal_exceeds_limit(lengths[0], MAX_CHAT_REQUEST_BYTES):
        raise _RequestBodyTooLarge
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_CHAT_REQUEST_BYTES:
            raise _RequestBodyTooLarge
        body.extend(chunk)
    return bytes(body)


async def _await_until_disconnect(request: Request, awaitable: Awaitable[Any]) -> Any:
    operation_task = asyncio.ensure_future(awaitable)
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
    try:
        done, _ = await asyncio.wait(
            (operation_task, disconnect_task),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if operation_task in done:
            return await operation_task
        await disconnect_task
        operation_task.cancel()
        with suppress(asyncio.CancelledError):
            await operation_task
        raise asyncio.CancelledError
    finally:
        if not operation_task.done():
            operation_task.cancel()
            with suppress(asyncio.CancelledError):
                await operation_task
        if not disconnect_task.done():
            disconnect_task.cancel()
            with suppress(asyncio.CancelledError):
                await disconnect_task


async def _wait_for_disconnect(request: Request) -> None:
    # the body is already consumed, so the receive channel now carries only disconnect lifecycle.
    while (await request.receive())["type"] != "http.disconnect":
        pass


def _decimal_exceeds_limit(value: str, limit: int) -> bool:
    if not value.isdecimal():
        return False
    normalized = value.lstrip("0") or "0"
    boundary = str(limit)
    return len(normalized) > len(boundary) or (
        len(normalized) == len(boundary) and normalized > boundary
    )


def _strict_json(raw: bytes) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_non_finite_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid json") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate json key")
        result[key] = value
    return result


def _error(
    status: int, code: str, message: str, *, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": message, "type": "invalid_request_error", "code": code}},
        status_code=status,
        headers=headers,
    )
