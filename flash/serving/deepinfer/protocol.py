"""Validate the supported chat subset before spending GPU time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flash.serve.request.openai import parse_chat_request
from flash.serving.deepinfer.config import MODEL_ID


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict[str, Any]]
    temperature: float
    top_p: float
    max_tokens: int
    seed: int | None
    thinking: bool
    stream: bool
    include_usage: bool


def parse_request(body: dict[str, Any]) -> ChatRequest:
    if not isinstance(body, dict):
        raise ValueError("chat request must be an object")
    allowed = {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "seed",
        "stream",
        "stream_options",
        "chat_template_kwargs",
        "n",
    }
    if extra := body.keys() - allowed:
        raise ValueError(
            "DeepInfer does not support these request fields: " + ", ".join(sorted(extra))
        )
    if body.get("model") != MODEL_ID:
        raise ValueError(f"model must be {MODEL_ID}")
    parsed = parse_chat_request(body, require_model=True, allow_managed_selectors=False)
    if parsed.n != 1:
        raise ValueError("DeepInfer supports n=1")
    options = parsed.chat_template_kwargs
    if options.keys() - {"enable_thinking"}:
        raise ValueError("only enable_thinking is supported in chat_template_kwargs")
    thinking = options.get("enable_thinking", False)
    if type(thinking) is not bool:
        raise ValueError("enable_thinking must be a boolean")
    for message in parsed.messages:
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("DeepInfer currently supports system, user and assistant messages")
        if not isinstance(message.get("content"), str):
            raise ValueError("DeepInfer currently supports text-only messages")
        if message.keys() - {"role", "content"}:
            raise ValueError("tool calls and additional message fields are not supported")
    return ChatRequest(
        messages=parsed.messages,
        temperature=parsed.temperature,
        top_p=parsed.top_p,
        max_tokens=parsed.max_tokens,
        seed=parsed.seed,
        thinking=thinking,
        stream=parsed.stream,
        include_usage=parsed.include_usage,
    )


def usage(prompt_tokens: int, completion_tokens: int, reasoning_tokens: int) -> dict[str, Any]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }
