"""strict OpenAI request translation, reasoning splitting, and response formatting."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from flash.serve.request.openai import (  # noqa: F401
    OpenAIRequestError,
    parse_stream_options,
    reject_thinking_logprobs,
    reject_tool_capability,
)
from flash.serve.request.openai import (
    parse_chat_request as parse_normalized_chat_request,
)
from flash.serve.runtime import GenerationRequest, GenerationResult, StreamFinished

from .bootstrap import PublishedAdapter
from .manifest import ServingManifest

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@dataclass(frozen=True, slots=True)
class OpenAIChatRequest:
    """one normalized request bound to an exact immutable adapter."""

    stream: bool
    include_usage: bool
    generation: GenerationRequest


def parse_chat_request(
    payload: object, resolved: PublishedAdapter, *, tool_parser: str | None = None
) -> OpenAIChatRequest:
    """bind the canonical request grammar to an exact immutable adapter."""

    request = parse_normalized_chat_request(
        payload,
        require_model=True,
        allow_managed_selectors=False,
    )
    if request.model != resolved.requested_model:
        raise OpenAIRequestError("resolved model binding does not match the request")
    reject_thinking_logprobs(
        thinking=resolved.adapter.thinking_default,
        logprobs=request.logprobs,
    )
    reject_tool_capability(
        tools=request.tools,
        tool_choice=request.tool_choice,
        thinking=resolved.adapter.thinking_default,
        tool_parser=tool_parser,
    )
    generation = GenerationRequest(
        adapter_id=resolved.adapter.checkpoint_id,
        expected_incarnation=resolved.adapter.aggregate_sha256,
        messages=request.messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        n=request.n,
        seed=request.seed,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        logprobs=request.logprobs,
        top_logprobs=request.top_logprobs,
        thinking=resolved.adapter.thinking_default,
        stop=request.stop,
        chat_template_kwargs=request.chat_template_kwargs,
        structured_outputs=request.structured_outputs,
        tools=request.tools,
        tool_choice=request.tool_choice,
        parallel_tool_calls=request.parallel_tool_calls,
    )
    return OpenAIChatRequest(
        stream=request.stream,
        include_usage=request.include_usage,
        generation=generation,
    )


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
        "checkpoint_id": adapter.checkpoint_id,
    }


def nonstream_response(
    result: GenerationResult,
    manifest: ServingManifest,
    resolved: PublishedAdapter,
) -> dict[str, Any]:
    choices = []
    for choice in result.choices:
        reasoning, content = split_reasoning(choice.text, thinking=bool(result.thinking))
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content if content else None if choice.tool_calls else content,
        }
        if choice.tool_calls:
            message["tool_calls"] = [call.wire() for call in choice.tool_calls]
        if reasoning is not None:
            message["reasoning_content"] = reasoning
        choices.append(
            {
                "index": choice.index,
                "message": message,
                "finish_reason": choice.finish_reason,
                "logprobs": {"content": choice.logprobs} if choice.logprobs is not None else None,
            }
        )
    return {
        "id": f"chatcmpl-{result.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resolved.requested_model,
        "choices": choices,
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
    index: int = 0,
    finish_reason: str | None = None,
    logprobs: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": index,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": {"content": logprobs} if logprobs is not None else None,
            }
        ],
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
