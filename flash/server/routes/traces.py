"""Authenticated recording proxy and trace export routes."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Annotated, Any
from urllib.parse import urljoin

import anyio
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from flash.core.spec import require_project_id
from flash.server.platform.deps import require_key
from flash.server.platform.traces import (
    MAX_EXPORT_TRACES,
    MAX_PAYLOAD_TOTAL_BYTES,
    TraceSpan,
    export_traces,
    list_projects,
    store_trace,
)
from flash.server.routes.trace_redaction import (  # noqa: F401
    _MIN_SECRET_SUBSTRING_LENGTH,
    _is_schema_definition,
    _is_secret_key,
    _local_schema_pointer,
    _redact_schema_literal,
    _redact_secret_fields,
    _redact_secret_string,
    _redact_secret_values,
    _SanitizationFlag,
    _sanitize_for_trace,
    _schema_anchor_pointers,
    _schema_resource_pointers,
    _secret_schema_definition_refs,
)
from flash.server.routes.trace_sse import SseAccumulator, SseDoneGate

logger = logging.getLogger(__name__)
router = APIRouter()

_UPSTREAM_TIMEOUT_SECONDS = 300.0
# how much of an upstream ERROR body to keep for the trace. generously above the 8 KiB the stored
# copy is truncated to, so the recorded message is never the part that got cut, while still bounding
# what one response can hold in memory.
_MAX_RECORDED_ERROR_BYTES = 64 * 1024
_UPSTREAM_TOO_LARGE_ERROR = "upstream response exceeded the 8 MiB relay limit"
_UPSTREAM_TOO_LARGE_BODY = b'{"detail":"Upstream response was too large to relay"}'
# told to the caller when the provider call succeeded but persisting its trace did not, so a
# collection run cannot look complete while its exports quietly omit calls.
_RECORD_FAILED_HEADER = "X-Freesolo-Record-Failed"
_SQLITE_INT_MIN = -(2**63)
_SQLITE_INT_MAX = 2**63 - 1
_PROVIDER_HEADER = "X-Freesolo-Provider"
_PROVIDER_KEY_HEADER = "X-Freesolo-Provider-Key"
_PROJECT_HEADER = "X-Freesolo-Project-Id"

_PROVIDER_URLS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
}
_PROVIDER_NAMES = tuple(_PROVIDER_URLS)
_FORWARDED_PROVIDER_HEADERS = frozenset(
    {
        "openai-organization",
        "openai-project",
        "openai-beta",
        "anthropic-beta",
        "anthropic-version",
    }
)
_SAFE_PROVIDER_RESPONSE_HEADERS = frozenset(
    {"request-id", "x-request-id", "retry-after", "retry-after-ms"}
)
_SAFE_PROVIDER_RESPONSE_HEADER_PREFIXES = ("x-ratelimit-", "anthropic-ratelimit-")


@dataclass
class _UpstreamRequestContext:
    url: str
    headers: dict[str, str]
    body: dict[str, Any]
    provider: str
    model: str
    key_id: int
    project_id: str | None
    metadata: dict[str, Any] | None
    secrets: tuple[str, ...]
    started_at: float
    record_trace: bool
    # set when persistence raised, so the response can tell the caller the call was NOT recorded
    # instead of leaving the gap to be discovered at export time.
    record_failed: bool = False


def _safe_provider_response_headers(
    headers: Mapping[str, str], *, status_code: int, upstream_url: str
) -> dict[str, str]:
    safe: dict[str, str] = {}
    for name, value in headers.items():
        normalized = name.casefold()
        if (
            normalized in _SAFE_PROVIDER_RESPONSE_HEADERS
            or normalized.startswith(_SAFE_PROVIDER_RESPONSE_HEADER_PREFIXES)
            or (300 <= status_code < 400 and normalized == "location")
        ):
            safe[name] = urljoin(upstream_url, value) if normalized == "location" else value
    return safe


def _usage_tokens(payload: Any) -> tuple[int | None, int | None]:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return None, None

    def _token(value: Any) -> int | None:
        return (
            value
            if isinstance(value, int)
            and not isinstance(value, bool)
            and _SQLITE_INT_MIN <= value <= _SQLITE_INT_MAX
            else None
        )

    return _token(usage.get("prompt_tokens")), _token(usage.get("completion_tokens"))


def _decoded_payload(response: httpx.Response) -> Any:
    try:
        return response.json()
    except (ValueError, UnicodeDecodeError, RecursionError):
        return response.text


def _decode_response_bytes(response: httpx.Response, body: bytes) -> str:
    encoding = response.encoding
    if encoding:
        try:
            return body.decode(encoding, errors="replace")
        except LookupError:
            pass
    return body.decode(errors="replace")


def _is_error_status(status_code: int) -> bool:
    return not 200 <= status_code < 300


def _error_for_status(status_code: int) -> str | None:
    return f"upstream returned status {status_code}" if _is_error_status(status_code) else None


def _build_trace_record(
    context: _UpstreamRequestContext,
    *,
    output_payload: Any,
    error: str | None,
    output_truncated: bool,
    usage: Any,
    duration_ms: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    input_sanitization = _SanitizationFlag()
    output_sanitization = _SanitizationFlag()
    sanitized_output = _sanitize_for_trace(
        output_payload, context.secrets, response=True, flag=output_sanitization
    )
    sanitized_usage = _sanitize_for_trace(usage, context.secrets, response=True)
    sanitized_input = _sanitize_for_trace(context.body, context.secrets, flag=input_sanitization)
    prompt_tokens, completion_tokens = _usage_tokens(
        sanitized_output if sanitized_output is not None else {"usage": sanitized_usage}
    )
    truncated_sides = [
        side
        for side, truncated in (
            ("input", input_sanitization.hit),
            ("output", output_truncated or output_sanitization.hit),
        )
        if truncated
    ]
    span = TraceSpan(
        name="chat.completions",
        provider=context.provider,
        model=_sanitize_for_trace(context.model, context.secrets),
        duration_ms=duration_ms,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        input_payload=sanitized_input,
        output_payload=sanitized_output,
        attributes={"payload_truncated": truncated_sides} if truncated_sides else None,
        status_code="ERROR" if error else "OK",
        error=_sanitize_for_trace(error, context.secrets) if error else None,
    )
    metadata = {
        "source": "recording_proxy",
        "route": context.provider,
        "tags": _sanitize_for_trace(context.metadata or {}, context.secrets),
    }
    return metadata, {"spans": [span]}


def _store_trace_record(
    context: _UpstreamRequestContext,
    *,
    project_id: str,
    output_payload: Any,
    error: str | None,
    output_truncated: bool,
    usage: Any,
    duration_ms: int,
) -> None:
    metadata, record = _build_trace_record(
        context,
        output_payload=output_payload,
        error=error,
        output_truncated=output_truncated,
        usage=usage,
        duration_ms=duration_ms,
    )
    store_trace(
        key_id=context.key_id,
        project_id=project_id,
        trace_title="chat.completions",
        metadata=metadata,
        **record,
    )


def _is_event_stream(response: httpx.Response) -> bool:
    """Whether a streamed response's body is actually SSE.

    Asked of the CONTENT TYPE rather than inferred from the status, because the two disagree: a
    gateway can answer `stream: true` with a 200 JSON error envelope or an HTML interstitial. The
    status-only test called that a stream, so the caller got `text/event-stream` for a body that is
    not one and the accumulator parsed it for deltas that never arrive -- storing `None` where the
    response bytes belong.
    """
    content_type = response.headers.get("content-type") or ""
    return content_type.split(";", 1)[0].strip().casefold() == "text/event-stream"


async def _record_trace(
    context: _UpstreamRequestContext,
    *,
    output_payload: Any,
    error: str | None,
    output_truncated: bool = False,
    usage: Any = None,
) -> None:
    if not context.record_trace or context.project_id is None:
        return
    duration_ms = max(0, round((time.perf_counter() - context.started_at) * 1000))
    try:
        await run_in_threadpool(
            _store_trace_record,
            context,
            project_id=context.project_id,
            output_payload=output_payload,
            error=error,
            output_truncated=output_truncated,
            usage=usage,
            duration_ms=duration_ms,
        )
    except Exception:
        # the provider call already happened and the caller was already billed, so failing the
        # request here would be worse than the lost trace. but staying silent lets someone finish a
        # collection run believing it was captured, and only discover the gap at export. record the
        # miss on the context so the response can say so.
        context.record_failed = True
        logger.exception("[recording-proxy] failed to persist trace")


def _response_headers(context: _UpstreamRequestContext) -> dict[str, str]:
    headers = {"content-type": "application/json"}
    if context.record_failed:
        headers[_RECORD_FAILED_HEADER] = "true"
    return headers


async def _upstream_failure_response(context: _UpstreamRequestContext) -> Response:
    await _record_trace(context, output_payload=None, error="upstream request failed")
    return Response(
        content=b'{"detail":"Upstream request failed"}',
        status_code=502,
        headers=_response_headers(context),
    )


async def _bounded_upstream_response(response: httpx.Response) -> tuple[bytes, bool]:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(chunk) > MAX_PAYLOAD_TOTAL_BYTES - len(body):
            return bytes(body), True
        body.extend(chunk)
    return bytes(body), False


async def _stream_response(
    *,
    client: httpx.AsyncClient,
    upstream_response: httpx.Response,
    context: _UpstreamRequestContext,
) -> AsyncIterator[bytes]:
    accumulator = SseAccumulator(max_accumulated_bytes=MAX_PAYLOAD_TOTAL_BYTES)
    raw_output = bytearray()
    raw_output_truncated = False
    # a body is parsed as SSE only if it says it is one. an error status is one way to get a
    # non-SSE body, but a 2xx gateway envelope is another, and feeding either to the accumulator
    # stores None in place of the bytes the caller actually received.
    raw_body = not _is_event_stream(upstream_response)
    raw_output_limit = (
        _MAX_RECORDED_ERROR_BYTES
        if _is_error_status(upstream_response.status_code)
        else MAX_PAYLOAD_TOTAL_BYTES
    )
    done_gate = SseDoneGate() if not raw_body else None
    error = _error_for_status(upstream_response.status_code)
    client_disconnected = False
    try:
        async for chunk in upstream_response.aiter_bytes():
            if context.record_trace and raw_body:
                # bounded, but by which bound depends on what the body IS. an error body is
                # stored truncated anyway, so retaining an unbounded one only to discard most
                # of it lets one response grow the plane's memory without limit. a SUCCESSFUL
                # non-SSE body keeps exactly the aggregate bound persistence accepts, so a body
                # that could be stored whole is not pre-truncated into undecodable JSON.
                remaining = raw_output_limit - len(raw_output)
                if remaining > 0:
                    raw_output.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    raw_output_truncated = True
            if done_gate is None:
                yield chunk
            else:
                forwarded_chunks = done_gate.feed(chunk)
                for forwarded in forwarded_chunks:
                    if context.record_trace:
                        accumulator.feed(forwarded)
                    yield forwarded
                if done_gate.terminated:
                    break
        if done_gate is not None:
            for forwarded in done_gate.finish():
                if context.record_trace:
                    accumulator.feed(forwarded)
                yield forwarded
    except (asyncio.CancelledError, GeneratorExit):
        client_disconnected = True
        error = "client disconnected"
        raise
    except Exception:
        error = "upstream stream interrupted"
        if done_gate is not None:
            for forwarded in done_gate.finish():
                if context.record_trace:
                    accumulator.feed(forwarded)
                yield forwarded
        raise
    finally:
        with anyio.CancelScope(shield=True):
            try:
                try:
                    await upstream_response.aclose()
                finally:
                    await client.aclose()
            finally:
                if context.record_trace:
                    if raw_body:
                        try:
                            output_payload: Any = json.loads(raw_output)
                        except (ValueError, UnicodeDecodeError, RecursionError):
                            output_payload = _decode_response_bytes(
                                upstream_response, bytes(raw_output)
                            )
                    else:
                        if done_gate is not None and done_gate.done_event is not None:
                            accumulator.feed(done_gate.done_event)
                        accumulator.finish()
                        # a stream that ended before any content arrived has NO output, which is
                        # not the same as an output with zero choices. keep an error-only envelope
                        # because raw export is the operator's record of what the provider returned,
                        # but keep suppressing usage-only and genuinely empty synthesized envelopes.
                        output_payload = (
                            accumulator.output()
                            if accumulator.received or accumulator.has_error
                            else None
                        )
                        if error is None and not accumulator.terminal:
                            error = "upstream stream ended before completion"
                        # a defect outranks a clean finish: a stream can drop a fragment or carry
                        # an error envelope and still deliver `[DONE]`, which would otherwise
                        # store the holed text as an OK reply for `records` to train on.
                        if accumulator.defect is not None:
                            error = error or accumulator.defect
                    await _record_trace(
                        context,
                        output_payload=output_payload,
                        error=error,
                        output_truncated=(
                            raw_output_truncated if raw_body else accumulator.truncated
                        ),
                        usage=None if raw_body else accumulator.usage,
                    )
    if not client_disconnected and done_gate is not None:
        if context.record_failed:
            comment = b": freesolo-record-failed\n"
            if not done_gate.done_event_has_relayed_prefix:
                comment += b"\n"
            yield comment
        if done_gate.done_event is not None:
            yield done_gate.done_event
    elif not client_disconnected and context.record_failed:
        # a non-SSE streamed body cannot carry the SSE comment -- appending one to JSON is the
        # corruption an earlier fix removed -- and the headers left before persistence ran, so the
        # header cannot say it either. log it: a silent failure would let a collection run finish
        # believing every paid call was recorded while its export quietly omits this one.
        logger.warning(
            "trace not recorded for a streamed non-SSE response (project=%s, provider=%s)",
            context.project_id,
            context.provider,
        )


async def _non_streaming_response(
    context: _UpstreamRequestContext, forwarded_body: dict[str, Any]
) -> Response:
    try:
        async with (
            httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT_SECONDS) as client,
            client.stream(
                "POST",
                context.url,
                headers=context.headers,
                json=forwarded_body,
            ) as upstream_response,
        ):
            upstream_status = upstream_response.status_code
            upstream_headers = upstream_response.headers
            upstream_body, response_too_large = await _bounded_upstream_response(upstream_response)
    except asyncio.CancelledError:
        with anyio.CancelScope(shield=True):
            await _record_trace(context, output_payload=None, error="client disconnected")
        raise
    except httpx.HTTPError:
        return await _upstream_failure_response(context)

    response_headers = _safe_provider_response_headers(
        upstream_headers, status_code=upstream_status, upstream_url=context.url
    )
    if response_too_large:
        await _record_trace(context, output_payload=None, error=_UPSTREAM_TOO_LARGE_ERROR)
        response_headers["content-type"] = "application/json"
        if context.record_failed:
            response_headers[_RECORD_FAILED_HEADER] = "true"
        return Response(content=_UPSTREAM_TOO_LARGE_BODY, status_code=502, headers=response_headers)

    buffered_response = httpx.Response(
        upstream_status,
        headers=upstream_headers,
        content=upstream_body,
    )
    await _record_trace(
        context,
        output_payload=_decoded_payload(buffered_response),
        error=_error_for_status(upstream_status),
    )
    content_type = upstream_headers.get("content-type")
    if content_type:
        response_headers["content-type"] = content_type
    if context.record_failed:
        response_headers[_RECORD_FAILED_HEADER] = "true"
    return Response(content=upstream_body, status_code=upstream_status, headers=response_headers)


async def _bounded_request_body(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header") from exc
        if declared < 0:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header")
        if declared > MAX_PAYLOAD_TOTAL_BYTES:
            raise HTTPException(status_code=413, detail="Request body exceeds the 8 MiB limit")
    body = bytearray()
    try:
        async for chunk in request.stream():
            if len(chunk) > MAX_PAYLOAD_TOTAL_BYTES - len(body):
                raise HTTPException(status_code=413, detail="Request body exceeds the 8 MiB limit")
            body.extend(chunk)
    except (HTTPException, asyncio.CancelledError):
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Failed to read request body") from exc
    return bytes(body)


def _request_context(
    *,
    request: Request,
    key: dict,
    body: dict[str, Any],
    provider: str,
    provider_key: str,
    project_id: str | None,
    model: str,
    metadata: dict[str, Any] | None,
    record_trace: bool,
) -> _UpstreamRequestContext:
    headers = {
        **{
            name: value
            for name, value in request.headers.items()
            if name.casefold() in _FORWARDED_PROVIDER_HEADERS
        },
        "Authorization": f"Bearer {provider_key}",
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
    }
    authorization = request.headers.get("authorization", "")
    secrets = tuple(
        value
        for value in (
            authorization,
            authorization.removeprefix("Bearer ").strip(),
            provider_key,
        )
        if value
    )
    if any(len(secret) < _MIN_SECRET_SUBSTRING_LENGTH for secret in secrets):
        # skipping a short substring avoids silently rewriting ordinary payload text, but it also
        # means that exact short credential is not removed when quoted outside a secret-named field.
        # warn the operator on every affected request rather than weakening redaction silently.
        logger.warning(
            "recording proxy received a credential shorter than %d characters; global substring "
            "redaction is disabled for that credential",
            _MIN_SECRET_SUBSTRING_LENGTH,
        )
    return _UpstreamRequestContext(
        url=_PROVIDER_URLS[provider],
        headers=headers,
        body=body,
        provider=provider,
        model=model,
        key_id=int(key["id"]),
        project_id=project_id,
        metadata=metadata,
        secrets=secrets,
        started_at=time.perf_counter(),
        record_trace=record_trace,
    )


def _reject_json_constant(constant: str) -> None:
    raise ValueError(f"non-finite JSON constant {constant!r} is not allowed")


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    key: Annotated[dict, Depends(require_key)],
) -> Response:
    raw_body = await _bounded_request_body(request)
    try:
        parsed_body = json.loads(raw_body, parse_constant=_reject_json_constant)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(parsed_body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    body = dict(parsed_body)
    metadata = body.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="metadata must be an object")
    raw_model = body.get("model")
    if raw_model is not None and not isinstance(raw_model, str):
        raise HTTPException(status_code=400, detail="model must be a string")
    if raw_model is None or not raw_model.strip():
        raise HTTPException(status_code=400, detail="model is required")
    model = raw_model

    provider = (request.headers.get(_PROVIDER_HEADER) or "").strip().casefold()
    if provider not in _PROVIDER_URLS:
        raise HTTPException(
            status_code=400,
            detail=f"{_PROVIDER_HEADER} must be one of: {', '.join(_PROVIDER_NAMES)}.",
        )
    provider_key = (request.headers.get(_PROVIDER_KEY_HEADER) or "").strip()
    if not provider_key:
        raise HTTPException(
            status_code=400,
            detail=f"{_PROVIDER_KEY_HEADER} is required: Flash proxies with your provider key.",
        )

    record_trace = request.headers.get("x-freesolo-record", "true").strip().casefold() != "false"
    project_id: str | None = None
    if record_trace:
        raw_project_id = (request.headers.get(_PROJECT_HEADER) or "").strip()
        if not raw_project_id:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{_PROJECT_HEADER} is required to record a trace. Send the header, or set "
                    "X-Freesolo-Record: false to proxy without recording."
                ),
            )
        try:
            project_id = require_project_id(raw_project_id)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    context = _request_context(
        request=request,
        key=key,
        body=body,
        provider=provider,
        provider_key=provider_key,
        project_id=project_id,
        model=model,
        metadata=metadata,
        record_trace=record_trace,
    )
    # the CALLER'S request, verbatim. redaction belongs to the stored copy only (`_record_trace`
    # sanitizes `context.body` itself): a proxy that rewrote the body before forwarding would send
    # the provider something the caller never wrote -- a tool schema whose `password` property got
    # replaced by the string "[redacted]", or a prompt that happens to quote the key. the caller
    # would be billed for inference on a request they did not make, and could not tell from the
    # response that it had been altered.
    forwarded_body = context.body

    if body.get("stream") is True:
        client = httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT_SECONDS)
        try:
            upstream_request = client.build_request(
                "POST", context.url, headers=context.headers, json=forwarded_body
            )
            upstream_response = await client.send(upstream_request, stream=True)
        except asyncio.CancelledError:
            with anyio.CancelScope(shield=True):
                try:
                    await client.aclose()
                finally:
                    await _record_trace(context, output_payload=None, error="client disconnected")
            raise
        except httpx.HTTPError:
            await client.aclose()
            return await _upstream_failure_response(context)
        except Exception:
            await client.aclose()
            raise
        response_headers = _safe_provider_response_headers(
            upstream_response.headers,
            status_code=upstream_response.status_code,
            upstream_url=context.url,
        )
        response_headers.update({"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})
        media_type: str | None = "text/event-stream"
        # relay the provider's own content type whenever the body is not an event stream, whatever
        # its status. a 200 gateway envelope is not SSE either, and labelling it `text/event-stream`
        # hands the caller a body their SSE reader discards as malformed.
        if not _is_event_stream(upstream_response):
            content_type = upstream_response.headers.get("content-type")
            if content_type:
                response_headers["content-type"] = content_type
            media_type = None
        return StreamingResponse(
            _stream_response(
                client=client,
                upstream_response=upstream_response,
                context=context,
            ),
            status_code=upstream_response.status_code,
            media_type=media_type,
            headers=response_headers,
        )

    return await _non_streaming_response(context, forwarded_body)


@router.get("/api/traces/projects")
def trace_projects(key: Annotated[dict, Depends(require_key)]) -> dict:
    return {"projects": list_projects(key_id=key["id"])}


@router.get("/api/traces/export")
def trace_export(
    key: Annotated[dict, Depends(require_key)],
    project_id: str,
    export_format: Annotated[str, Query(alias="format")] = "records",
    limit: Annotated[int, Query(ge=1, le=MAX_EXPORT_TRACES)] = MAX_EXPORT_TRACES,
) -> dict:
    try:
        return export_traces(
            key_id=key["id"],
            project_id=project_id,
            export_format=export_format,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
