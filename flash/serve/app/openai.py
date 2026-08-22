"""strict OpenAI request translation, reasoning splitting, and response formatting."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Any

from flash.serve.runtime import GenerationRequest, GenerationResult, StreamFinished
from flash.serve.runtime.multimodal import validate_messages
from flash.serve.runtime.structured_outputs import normalize_structured_outputs

from .bootstrap import PublishedAdapter
from .manifest import ServingManifest

_REQUEST_KEYS = frozenset(
    {
        "model",
        "messages",
        "stream",
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "chat_template_kwargs",
        "structured_outputs",
        "response_format",
        "stream_options",
    }
)
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


class OpenAIRequestError(ValueError):
    """one public request failed strict shape or semantic validation."""


@dataclass(frozen=True, slots=True)
class OpenAIChatRequest:
    """strict normalized subset accepted by the packaged serving app."""

    model: str
    messages: tuple[dict[str, Any], ...]
    stream: bool
    include_usage: bool
    generation: GenerationRequest


def parse_chat_request(payload: object, resolved: PublishedAdapter) -> OpenAIChatRequest:
    """reject unknown keys and bind one request to an exact immutable adapter."""

    if type(payload) is not dict:
        raise OpenAIRequestError("request body must be a json object")
    unknown = sorted(set(payload) - _REQUEST_KEYS)
    if unknown:
        raise OpenAIRequestError("request contains unknown keys")
    model = _required_string(payload.get("model"), "model")
    if model != resolved.requested_model:
        raise OpenAIRequestError("resolved model binding does not match the request")
    messages = payload.get("messages")
    if (
        type(messages) is not list
        or not messages
        or any(type(item) is not dict for item in messages)
    ):
        raise OpenAIRequestError("messages must be a nonempty array of objects")
    # being a dict is not being a *message*. the checks above accept `{}`, `{"role": "bogus"}`, and
    # a `content` of any type, none of which generation can honor. rejecting here makes those a
    # 422 rather than an empty prompt rendered from a missing `content` (200 on nonsense) or a
    # template failure answered 503, which reads as "retry later" for a request that can never
    # succeed. this only validates: the accepted spellings are rewritten to what the template
    # understands in `PromptPreparer`, past the image dispatch, because normalizing here would
    # strip the image payloads this endpoint must forward.
    validate_messages(messages)
    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise OpenAIRequestError("stream must be a boolean")
    include_usage = parse_stream_options(payload.get("stream_options"), stream)
    structured = _structured_override(payload)
    generation = GenerationRequest(
        adapter_id=resolved.adapter_revision,
        expected_incarnation=resolved.incarnation,
        messages=messages,
        max_tokens=_positive_int(payload.get("max_tokens", 1024), "max_tokens"),
        temperature=_number(payload.get("temperature", 0.0), "temperature", minimum=0.0),
        top_p=_top_p(payload.get("top_p", 0.95)),
        stop=payload.get("stop"),
        chat_template_kwargs=_mapping(
            payload.get("chat_template_kwargs", {}), "chat_template_kwargs"
        ),
        structured_outputs=structured,
    )
    return OpenAIChatRequest(
        model=model,
        messages=tuple(dict(message) for message in messages),
        stream=stream,
        include_usage=include_usage,
        generation=generation,
    )


def _structured_override(payload: dict[str, Any]) -> Any:
    has_structured = "structured_outputs" in payload and payload["structured_outputs"] is not None
    has_response = "response_format" in payload and payload["response_format"] is not None
    if has_structured and has_response:
        raise OpenAIRequestError("structured_outputs and response_format cannot both be set")
    if has_structured:
        return normalize_structured_outputs(payload["structured_outputs"])
    if has_response:
        return _response_format(payload["response_format"])
    return None


def _response_format(value: object) -> dict[str, Any]:
    data = _mapping(value, "response_format")
    kind = data.get("type")
    if kind == "text":
        if set(data) != {"type"}:
            raise OpenAIRequestError("text response_format contains unknown keys")
        return {}
    if kind == "json_object":
        if set(data) != {"type"}:
            raise OpenAIRequestError("json_object response_format contains unknown keys")
        return {"json_object": True}
    if kind != "json_schema" or set(data) != {"type", "json_schema"}:
        raise OpenAIRequestError("response_format type is not supported")
    declaration = _mapping(data["json_schema"], "response_format.json_schema")
    allowed = {"name", "description", "schema", "strict"}
    if set(declaration) - allowed or "schema" not in declaration:
        raise OpenAIRequestError("json_schema response_format is malformed")
    schema = _mapping(declaration["schema"], "response_format.json_schema.schema")
    return {"json": schema}


def parse_stream_options(value: object, stream: bool) -> bool:
    if value is None:
        return False
    if not stream:
        raise OpenAIRequestError("stream_options requires stream=true")
    options = _mapping(value, "stream_options")
    if set(options) != {"include_usage"} or type(options["include_usage"]) is not bool:
        raise OpenAIRequestError("stream_options accepts only boolean include_usage")
    return options["include_usage"]


def _required_string(value: object, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise OpenAIRequestError(f"{name} must be a nonempty string")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise OpenAIRequestError(f"{name} must be a positive integer")
    return value


def _number(value: object, name: str, *, minimum: float) -> float:
    if type(value) not in {int, float}:
        raise OpenAIRequestError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise OpenAIRequestError(f"{name} is outside its accepted range")
    return result


def _top_p(value: object) -> float:
    result = _number(value, "top_p", minimum=0.0)
    if result <= 0 or result > 1:
        raise OpenAIRequestError("top_p must be greater than zero and at most one")
    return result


def _mapping(value: object, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise OpenAIRequestError(f"{name} must be an object")
    return dict(value)


def split_reasoning(text: str, *, thinking: bool) -> tuple[str | None, str]:
    """split at the first close only; preserve all delimiters when thinking is off."""

    if not thinking:
        return None, text
    close = text.find(_THINK_CLOSE)
    if close < 0:
        return _strip_initial_open(text), ""
    reasoning = _strip_initial_open(text[:close])
    return reasoning, text[close + len(_THINK_CLOSE) :]


class ReasoningDeltaSplitter:
    """incrementally split the first reasoning close across arbitrary delta boundaries."""

    def __init__(self, *, thinking: bool) -> None:
        self._thinking = thinking
        self._closed = not thinking
        self._buffer = ""
        self._at_start = True

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        if self._closed:
            return [("content", text)]
        self._buffer += text
        self._strip_open_when_decidable()
        close = self._buffer.find(_THINK_CLOSE)
        if close >= 0:
            events: list[tuple[str, str]] = []
            if close:
                events.append(("reasoning_content", self._buffer[:close]))
            answer = self._buffer[close + len(_THINK_CLOSE) :]
            if answer:
                events.append(("content", answer))
            self._buffer = ""
            self._closed = True
            return events
        retain = len(_THINK_CLOSE) - 1
        if len(self._buffer) <= retain:
            return []
        emit = self._buffer[:-retain]
        self._buffer = self._buffer[-retain:]
        return [("reasoning_content", emit)] if emit else []

    def finish(self) -> list[tuple[str, str]]:
        if not self._buffer:
            return []
        kind = "content" if self._closed else "reasoning_content"
        buffered = self._buffer
        self._buffer = ""
        return [(kind, buffered)]

    def _strip_open_when_decidable(self) -> None:
        if not self._at_start:
            return
        if _THINK_OPEN.startswith(self._buffer) and len(self._buffer) < len(_THINK_OPEN):
            return
        if self._buffer.startswith(_THINK_OPEN):
            self._buffer = self._buffer[len(_THINK_OPEN) :]
        self._at_start = False


def _strip_initial_open(text: str) -> str:
    return text[len(_THINK_OPEN) :] if text.startswith(_THINK_OPEN) else text


def usage_payload(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
    cached_tokens_reported: bool,
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens_reported:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return usage


def provenance_payload(
    manifest: ServingManifest,
    resolved: PublishedAdapter,
) -> dict[str, Any]:
    adapter = resolved.adapter
    return {
        "deployment_id": manifest.deployment_id,
        "spec_id": manifest.spec_id,
        "manifest_id": manifest.manifest_id,
        "engine_id": manifest.engine.engine_id,
        "image_digest": manifest.expected_oci_digest,
        "logical_base_model": manifest.logical_base_model,
        "logical_base_revision": manifest.logical_base_revision,
        "served_checkpoint": manifest.engine.served_model,
        "served_checkpoint_revision": manifest.engine.model_revision,
        "tokenizer_model": manifest.engine.tokenizer_model,
        "tokenizer_revision": manifest.engine.tokenizer_revision,
        "requested_model": resolved.requested_model,
        "adapter_revision": adapter.adapter_revision,
        "checkpoint": adapter.checkpoint,
        "source_revision": adapter.source_revision,
        "source_subfolder": adapter.source_subfolder,
        "aggregate_sha256": adapter.aggregate_sha256,
    }


def provenance_headers(provenance: dict[str, Any]) -> dict[str, str]:
    return {f"x-flash-{key.replace('_', '-')}": str(value) for key, value in provenance.items()}


def nonstream_response(
    result: GenerationResult,
    manifest: ServingManifest,
    resolved: PublishedAdapter,
) -> dict[str, Any]:
    reasoning, content = split_reasoning(result.text, thinking=bool(result.thinking))
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    return {
        "id": f"chatcmpl-{result.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resolved.requested_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": usage_payload(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cached_tokens=result.cached_tokens,
            cached_tokens_reported=result.cached_tokens_reported,
        ),
        "flash_provenance": provenance_payload(manifest, resolved),
    }


def stream_chunk(
    *,
    request_id: str,
    model: str,
    delta: dict[str, Any],
    finish_reason: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if provenance is not None:
        payload["flash_provenance"] = provenance
    return payload


def usage_stream_chunk(
    finished: StreamFinished,
    model: str,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{finished.request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": usage_payload(
            prompt_tokens=finished.prompt_tokens,
            completion_tokens=finished.completion_tokens,
            cached_tokens=finished.cached_tokens,
            cached_tokens_reported=finished.cached_tokens_reported,
        ),
        "flash_provenance": provenance,
    }


def sse_data(payload: dict[str, Any] | str) -> bytes:
    data = payload if isinstance(payload, str) else json.dumps(payload, separators=(",", ":"))
    return f"data: {data}\n\n".encode()
