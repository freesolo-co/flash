"""pure-stdlib decoding for openai-compatible server-sent events."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass


class OpenAISSEError(ValueError):
    """one decoded event stream violated the openai sse contract."""


@dataclass(frozen=True, slots=True)
class DeltaEvent:
    reasoning_content: str | None
    content: str | None


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    message: str


@dataclass(frozen=True, slots=True)
class DoneEvent:
    pass


OpenAISSEEvent = DeltaEvent | ErrorEvent | DoneEvent


def iter_openai_sse_events(chunks: Iterable[str]) -> Iterator[OpenAISSEEvent]:
    """decode arbitrary text chunks and require one terminal event."""

    buffered = ""
    terminal = False
    for chunk in chunks:
        buffered += chunk
        while "\n" in buffered:
            line, buffered = buffered.split("\n", 1)
            for event in _events_from_line(line.rstrip("\r")):
                yield event
                if isinstance(event, DoneEvent | ErrorEvent):
                    terminal = True
                    return
    if buffered.strip():
        raise OpenAISSEError("chat stream ended with an incomplete server-sent event frame")
    if not terminal:
        raise OpenAISSEError("chat stream ended before the terminal [DONE] event")


def sse_data_is_terminal(data: bytes) -> bool:
    """return whether one complete raw sse frame is terminal."""

    data_lines = [
        line.removeprefix(b"data:").strip()
        for line in data.replace(b"\r\n", b"\n").split(b"\n")
        if line.startswith(b"data:")
    ]
    if not data_lines:
        return False
    raw = b"\n".join(data_lines)
    if raw == b"[DONE]":
        return True
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    if isinstance(payload.get("error"), dict):
        return True
    choices = payload.get("choices") or []
    return isinstance(choices, list) and any(
        isinstance(choice, dict) and choice.get("finish_reason") == "error" for choice in choices
    )


def _events_from_line(line: str) -> tuple[OpenAISSEEvent, ...]:
    if not line.startswith("data:"):
        return ()
    data = line.removeprefix("data:").strip()
    if not data:
        return ()
    if data == "[DONE]":
        return (DoneEvent(),)
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise OpenAISSEError("chat stream contained invalid openai sse json") from exc
    if not isinstance(payload, dict):
        raise OpenAISSEError("chat stream contained a non-object openai sse payload")
    error = payload.get("error")
    if isinstance(error, dict):
        return (ErrorEvent(str(error.get("message") or "chat stream ended with an error")),)
    choices = payload.get("choices") or []
    if not isinstance(choices, list):
        raise OpenAISSEError("chat stream choices must be an array")
    events: list[DeltaEvent | ErrorEvent] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        if choice.get("finish_reason") == "error":
            events.append(ErrorEvent("chat stream ended with an engine error"))
            continue
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            continue
        reasoning = delta.get("reasoning_content")
        content = delta.get("content")
        normalized_reasoning = reasoning if isinstance(reasoning, str) else None
        normalized_content = content if isinstance(content, str) else None
        if normalized_reasoning is not None or normalized_content is not None:
            events.append(DeltaEvent(normalized_reasoning, normalized_content))
    if not events:
        return ()
    if len(events) > 1:
        raise OpenAISSEError("chat stream contained multiple text choices")
    return (events[0],)
