"""Authenticated recording proxy and trace export routes."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Annotated, Any

import anyio
import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from flash.core.spec import require_project_id
from flash.server.platform.deps import require_key
from flash.server.platform.traces import (
    MAX_EXPORT_TRACES,
    TraceSpan,
    export_traces,
    list_projects,
    store_trace,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_UPSTREAM_TIMEOUT_SECONDS = 300.0
# how much of an upstream ERROR body to keep for the trace. generously above the 8 KiB the stored
# copy is truncated to, so the recorded message is never the part that got cut, while still bounding
# what one response can hold in memory.
_MAX_RECORDED_ERROR_BYTES = 64 * 1024
# told to the caller when the provider call succeeded but persisting its trace did not, so a
# collection run cannot look complete while its exports quietly omit calls.
_RECORD_FAILED_HEADER = "X-Freesolo-Record-Failed"
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
_SECRET_KEY_EXACT = frozenset({"authorization", "proxyauthorization"})
_SECRET_KEY_SUFFIXES = (
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "credential",
    "credentials",
    "privatekey",
)
_JSON_SCHEMA_STRUCTURAL_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "items",
        "prefixItems",
        "additionalItems",
        "contains",
        "enum",
        "const",
        "$ref",
        "$id",
        "$schema",
        "$anchor",
        "$dynamicRef",
        "$dynamicAnchor",
        "anyOf",
        "allOf",
        "oneOf",
        "not",
        "format",
        "additionalProperties",
        "patternProperties",
        "propertyNames",
        "unevaluatedProperties",
        "unevaluatedItems",
        "dependentRequired",
        "dependentSchemas",
        "discriminator",
        "required",
        "$defs",
        "definitions",
        "if",
        "then",
        "else",
        "contentSchema",
    }
)
_JSON_SCHEMA_ANNOTATION_KEYWORDS = frozenset(
    {
        "description",
        "title",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "minContains",
        "maxContains",
        "contentEncoding",
        "contentMediaType",
        "nullable",
        "example",
        "$comment",
    }
)
_JSON_SCHEMA_KEYWORDS = _JSON_SCHEMA_STRUCTURAL_KEYWORDS | _JSON_SCHEMA_ANNOTATION_KEYWORDS
_JSON_SCHEMA_VALUE_KEYWORDS = frozenset({"default", "examples", "example"})


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


def _is_secret_key(key: Any) -> bool:
    normalized = str(key).casefold().replace("_", "").replace("-", "")
    return normalized in _SECRET_KEY_EXACT or (
        normalized != "token"
        and any(normalized.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES)
    )


def _is_schema_definition(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if not isinstance(value, dict):
        return False
    if not value:
        # `{}` is the permissive JSON Schema ("any value"), so under a `properties` map it is a
        # declaration, not a secret. Treating it as one rewrote `{"password": {}}` into the string
        # "[redacted]" and turned a valid schema into an invalid one.
        return True
    keys = [key for key in value if not (isinstance(key, str) and key.startswith("x-"))]
    if any(key not in _JSON_SCHEMA_KEYWORDS for key in keys):
        return False
    if any(key in _JSON_SCHEMA_STRUCTURAL_KEYWORDS for key in keys):
        return True
    return bool(keys) and all(key not in _JSON_SCHEMA_VALUE_KEYWORDS for key in keys)


def _redact_secret_fields(value: Any, *, schema_property_map: bool = False) -> Any:
    if isinstance(value, dict):
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            if _is_secret_key(key) and not (schema_property_map and _is_schema_definition(item)):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = _redact_secret_fields(
                    item,
                    schema_property_map=(
                        key in {"properties", "$defs", "definitions"} and isinstance(item, dict)
                    ),
                )
        return redacted
    if isinstance(value, list | tuple):
        return [_redact_secret_fields(item) for item in value]
    return value


def _redact_secret_string(value: str, secrets: tuple[str, ...]) -> str:
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[redacted]")
    return value


def _redact_secret_values(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        # keys are redacted too. a credential used as an object key -- `{"sk-live-...": "seen"}` --
        # is still the credential, and a key-blind pass would write it into the span verbatim and
        # hand it back through `format=raw`.
        return {
            (_redact_secret_string(key, secrets) if isinstance(key, str) else key): (
                _redact_secret_values(item, secrets)
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact_secret_values(item, secrets) for item in value]
    if isinstance(value, str):
        return _redact_secret_string(value, secrets)
    return value


def _sanitize_for_trace(value: Any, secrets: tuple[str, ...]) -> Any:
    return _redact_secret_values(_redact_secret_fields(value), secrets)


def _safe_provider_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    safe: dict[str, str] = {}
    for name, value in headers.items():
        normalized = name.casefold()
        if normalized in _SAFE_PROVIDER_RESPONSE_HEADERS or normalized.startswith(
            _SAFE_PROVIDER_RESPONSE_HEADER_PREFIXES
        ):
            safe[name] = value
    return safe


def _usage_tokens(payload: Any) -> tuple[int | None, int | None]:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage, dict):
        return None, None

    def _token(value: Any) -> int | None:
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    return _token(usage.get("prompt_tokens")), _token(usage.get("completion_tokens"))


def _decoded_payload(response: httpx.Response) -> Any:
    try:
        return response.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return response.text


def _error_for_status(status_code: int) -> str | None:
    return None if status_code < 400 else f"upstream returned status {status_code}"


async def _record_trace(
    context: _UpstreamRequestContext, *, output_payload: Any, error: str | None
) -> None:
    if not context.record_trace or context.project_id is None:
        return
    duration_ms = max(0, round((time.perf_counter() - context.started_at) * 1000))
    sanitized_output = _sanitize_for_trace(output_payload, context.secrets)
    prompt_tokens, completion_tokens = _usage_tokens(sanitized_output)
    span = TraceSpan(
        name="chat.completions",
        provider=context.provider,
        model=_sanitize_for_trace(context.model, context.secrets),
        duration_ms=duration_ms,
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens,
        input_payload=_sanitize_for_trace(context.body, context.secrets),
        output_payload=sanitized_output,
        status_code="ERROR" if error else "OK",
        error=_sanitize_for_trace(error, context.secrets) if error else None,
    )
    metadata = {
        "source": "recording_proxy",
        "route": context.provider,
        "tags": _sanitize_for_trace(context.metadata or {}, context.secrets),
    }
    try:
        await run_in_threadpool(
            store_trace,
            key_id=context.key_id,
            project_id=context.project_id,
            trace_title="chat.completions",
            metadata=metadata,
            spans=[span],
        )
    except Exception:
        # the provider call already happened and the caller was already billed, so failing the
        # request here would be worse than the lost trace. but staying silent lets someone finish a
        # collection run believing it was captured, and only discover the gap at export. record the
        # miss on the context so the response can say so.
        context.record_failed = True
        logger.exception("[recording-proxy] failed to persist trace")


async def _upstream_failure_response(context: _UpstreamRequestContext) -> Response:
    await _record_trace(context, output_payload=None, error="upstream request failed")
    return Response(
        content=b'{"detail":"Upstream request failed"}',
        status_code=502,
        headers={"content-type": "application/json"},
    )


class _StringFragments:
    def __init__(self, value: str) -> None:
        self.parts = [value]

    def append(self, value: str) -> None:
        self.parts.append(value)

    def text(self) -> str:
        return "".join(self.parts)


def _materialize_fragments(value: Any) -> Any:
    if isinstance(value, _StringFragments):
        return value.text()
    if isinstance(value, dict):
        return {key: _materialize_fragments(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize_fragments(item) for item in value]
    return value


def _content_parts(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if isinstance(value, _StringFragments):
        value = value.text()
    if isinstance(value, str):
        return [{"type": "text", "text": value}] if value else []
    return [value]


def _append_fragment(target: dict[str, Any], key: str, value: Any) -> None:
    current = target.get(key)
    if isinstance(value, str):
        if isinstance(current, _StringFragments):
            current.append(value)
        elif isinstance(current, str):
            target[key] = _StringFragments(current)
            target[key].append(value)
        elif isinstance(current, list):
            current.extend(_content_parts(value))
        else:
            target[key] = _StringFragments(value)
    elif isinstance(value, list):
        if isinstance(current, list):
            current.extend(value)
        elif isinstance(current, str | _StringFragments):
            target[key] = [*_content_parts(current), *value]
        else:
            target[key] = list(value)
    elif value is not None:
        target[key] = value


def _merge_fragment_dict(target: dict[str, Any], fragment: dict[str, Any]) -> None:
    for key, value in fragment.items():
        if isinstance(value, dict):
            nested = target.setdefault(key, {})
            if isinstance(nested, dict):
                _merge_fragment_dict(nested, value)
            else:
                target[key] = dict(value)
        elif isinstance(value, str):
            current = target.get(key)
            current_text = current.text() if isinstance(current, _StringFragments) else current
            if current_text == value and key in {"id", "type"}:
                continue
            if isinstance(current, _StringFragments):
                current.append(value)
            elif isinstance(current, str):
                target[key] = _StringFragments(current)
                target[key].append(value)
            else:
                target[key] = _StringFragments(value)
        elif value is not None:
            target[key] = value


class _SseAccumulator:
    def __init__(self) -> None:
        self._buffer = b""
        self._choices: dict[int, dict[str, Any]] = {}
        self._done = False
        self.usage: Any = None

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            self._consume_line(line.rstrip(b"\r"))

    def finish(self) -> None:
        if self._buffer:
            self._consume_line(self._buffer.rstrip(b"\r"))
            self._buffer = b""

    @property
    def received(self) -> bool:
        """Whether any CHOICE content arrived, which is what makes an output an output.

        Derived from the accumulated choices rather than tracked as a flag set on each parsed
        event: a usage-only or metadata-only chunk is a parseable event carrying no reply, so a
        per-event flag reported "received" for a stream that produced nothing and re-stored the
        synthesized empty-`choices` envelope -- the row `records` exists to skip.
        """
        return bool(self._choices)

    @property
    def terminal(self) -> bool:
        return self._done or (
            bool(self._choices)
            and all(choice["finish_reason"] is not None for choice in self._choices.values())
        )

    def output(self) -> dict[str, Any]:
        choices: list[dict[str, Any]] = []
        for index in sorted(self._choices):
            state = self._choices[index]
            message = _materialize_fragments(state["message"])
            tool_calls = state["tool_calls"]
            if tool_calls:
                message["tool_calls"] = [
                    _materialize_fragments(tool_calls[i]) for i in sorted(tool_calls)
                ]
            choices.append(
                {
                    "index": index,
                    "message": message,
                    "finish_reason": state["finish_reason"],
                }
            )
        return {"choices": choices, "usage": self.usage}

    def _choice_state(self, index: int) -> dict[str, Any]:
        return self._choices.setdefault(
            index,
            {
                "message": {"role": "assistant"},
                "tool_calls": {},
                "finish_reason": None,
            },
        )

    def _consume_line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        data = line[len(b"data:") :].strip()
        if not data:
            return
        if data == b"[DONE]":
            self._done = True
            return
        try:
            payload = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(payload, dict):
            return
        if isinstance(payload.get("usage"), dict):
            self.usage = payload["usage"]
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        for position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                continue
            raw_index = choice.get("index", position)
            index = (
                raw_index
                if isinstance(raw_index, int) and not isinstance(raw_index, bool)
                else position
            )
            state = self._choice_state(index)
            delta = choice.get("delta")
            if isinstance(delta, dict):
                self._consume_delta(state, delta)
            if choice.get("finish_reason") is not None:
                state["finish_reason"] = choice["finish_reason"]

    def _consume_delta(self, state: dict[str, Any], delta: dict[str, Any]) -> None:
        message = state["message"]
        role = delta.get("role")
        if isinstance(role, str) and role:
            message["role"] = role
        # every text-shaped delta field, not a fixed pair. providers stream their own alongside the
        # standard ones -- OpenRouter's `reasoning`, audio transcripts -- and an allowlist silently
        # dropped them, so a streamed trace held less than the identical non-streaming call.
        for key, value in delta.items():
            if key in {"role", "function_call", "tool_calls"}:
                continue
            if isinstance(value, dict) and isinstance(message.get(key), dict):
                _merge_fragment_dict(message[key], value)
            elif isinstance(value, str) or (
                key in message and isinstance(message[key], str | _StringFragments)
            ):
                _append_fragment(message, key, value)
            elif key not in message:
                message[key] = value
        function_call = delta.get("function_call")
        if isinstance(function_call, dict):
            target = message.setdefault("function_call", {})
            if isinstance(target, dict):
                _merge_fragment_dict(target, function_call)
        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            return
        accumulated_calls: dict[int, dict[str, Any]] = state["tool_calls"]
        for position, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            raw_index = tool_call.get("index", position)
            index = (
                raw_index
                if isinstance(raw_index, int) and not isinstance(raw_index, bool)
                else position
            )
            target = accumulated_calls.setdefault(index, {})
            _merge_fragment_dict(
                target,
                {key: value for key, value in tool_call.items() if key != "index"},
            )


async def _stream_response(
    *,
    client: httpx.AsyncClient,
    upstream_response: httpx.Response,
    context: _UpstreamRequestContext,
) -> AsyncIterator[bytes]:
    accumulator = _SseAccumulator()
    raw_output = bytearray()
    is_error = upstream_response.status_code >= 400
    error = _error_for_status(upstream_response.status_code)
    client_disconnected = False
    try:
        async for chunk in upstream_response.aiter_bytes():
            if context.record_trace:
                if is_error:
                    # bounded: an error body is stored truncated anyway, so retaining an unbounded
                    # one only to throw most of it away lets a single upstream response grow the
                    # plane's memory without limit. the caller still receives every byte below.
                    if len(raw_output) < _MAX_RECORDED_ERROR_BYTES:
                        raw_output.extend(chunk[: _MAX_RECORDED_ERROR_BYTES - len(raw_output)])
                else:
                    accumulator.feed(chunk)
            yield chunk
    except (asyncio.CancelledError, GeneratorExit):
        client_disconnected = True
        error = "client disconnected"
        raise
    except Exception:
        error = "upstream stream interrupted"
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
                    if is_error:
                        try:
                            output_payload: Any = json.loads(raw_output)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            output_payload = bytes(raw_output).decode(errors="replace")
                    else:
                        accumulator.finish()
                        # a stream that ended before any content arrived has NO output, which is
                        # not the same as an output with zero choices. recording the synthesized
                        # `{"choices": [], "usage": null}` would make `format=records` emit a
                        # training pair whose response half is an empty envelope -- exactly the
                        # row that format exists to skip.
                        output_payload = accumulator.output() if accumulator.received else None
                        if error is None and accumulator.received and not accumulator.terminal:
                            error = "upstream stream ended before completion"
                    await _record_trace(context, output_payload=output_payload, error=error)
    if context.record_trace and context.record_failed and not client_disconnected:
        yield b": freesolo-record-failed\n\n"


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


@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    key: Annotated[dict, Depends(require_key)],
) -> Response:
    try:
        parsed_body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(parsed_body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")

    body = dict(parsed_body)
    metadata = body.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise HTTPException(status_code=400, detail="metadata must be an object")
    model = str(body.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    body["model"] = model

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
        except httpx.HTTPError:
            await client.aclose()
            return await _upstream_failure_response(context)
        response_headers = _safe_provider_response_headers(upstream_response.headers)
        response_headers.update({"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})
        media_type: str | None = "text/event-stream"
        if upstream_response.status_code >= 400:
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

    try:
        async with httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT_SECONDS) as client:
            upstream_response = await client.post(
                context.url,
                headers=context.headers,
                json=forwarded_body,
            )
    except httpx.HTTPError:
        return await _upstream_failure_response(context)

    await _record_trace(
        context,
        output_payload=_decoded_payload(upstream_response),
        error=_error_for_status(upstream_response.status_code),
    )
    response_headers = _safe_provider_response_headers(upstream_response.headers)
    content_type = upstream_response.headers.get("content-type")
    if content_type:
        response_headers["content-type"] = content_type
    if context.record_failed:
        response_headers[_RECORD_FAILED_HEADER] = "true"
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


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
