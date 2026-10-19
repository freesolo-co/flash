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

_UTF8_BOM = chr(0xFEFF)


def _next_sse_line(buffered: str, *, final: bool = False) -> tuple[str, str] | None:
    for index, character in enumerate(buffered):
        if character == "\n":
            return buffered[:index], buffered[index + 1 :]
        if character != "\r":
            continue
        if index + 1 == len(buffered) and not final:
            return None
        delimiter_length = 2 if buffered[index + 1 : index + 2] == "\n" else 1
        return buffered[:index], buffered[index + delimiter_length :]
    return None


def iter_openai_sse_events(chunks: Iterable[str]) -> Iterator[OpenAISSEEvent]:
    """decode arbitrary text chunks and require one terminal event."""

    buffered = ""
    frame_lines: list[str] = []
    first_line = True
    for chunk in chunks:
        buffered += chunk
        while parsed := _next_sse_line(buffered):
            line, buffered = parsed
            if first_line:
                line = line.removeprefix(_UTF8_BOM)
                first_line = False
            if line:
                frame_lines.append(line)
                continue
            for event in _events_from_frame(frame_lines):
                yield event
                if isinstance(event, DoneEvent | ErrorEvent):
                    return
            frame_lines = []
    while parsed := _next_sse_line(buffered, final=True):
        line, buffered = parsed
        if first_line:
            line = line.removeprefix(_UTF8_BOM)
            first_line = False
        if line:
            frame_lines.append(line)
            continue
        for event in _events_from_frame(frame_lines):
            yield event
            if isinstance(event, DoneEvent | ErrorEvent):
                return
        frame_lines = []
    if buffered or frame_lines:
        raise OpenAISSEError("chat stream ended with an incomplete server-sent event frame")
    raise OpenAISSEError("chat stream ended before the terminal [DONE] event")


def sse_data_is_terminal(data: bytes) -> bool:
    """return whether one complete raw sse frame is terminal."""

    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    data_lines = [
        line.removeprefix(b"data:").strip()
        for line in normalized.split(b"\n")
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


def _events_from_frame(lines: list[str]) -> tuple[OpenAISSEEvent, ...]:
    data_lines = []
    for line in lines:
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:")
        data_lines.append(data.removeprefix(" "))
    if not data_lines:
        return ()
    data = "\n".join(data_lines)
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
    if "error" in payload:
        error = payload["error"]
        if not isinstance(error, dict):
            raise OpenAISSEError("chat stream error must be an object")
        return (ErrorEvent(str(error.get("message") or "chat stream ended with an error")),)
    choices = payload.get("choices", [])
    if not isinstance(choices, list):
        raise OpenAISSEError("chat stream choices must be an array")
    events: list[DeltaEvent | ErrorEvent] = []
    for choice in choices:
        if not isinstance(choice, dict):
            raise OpenAISSEError("chat stream choice must be an object")
        if choice.get("finish_reason") == "error":
            events.append(ErrorEvent("chat stream ended with an engine error"))
            continue
        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            raise OpenAISSEError("chat stream delta must be an object")
        reasoning = delta.get("reasoning_content")
        content = delta.get("content")
        if reasoning is not None and not isinstance(reasoning, str):
            raise OpenAISSEError("chat stream reasoning_content must be a string or null")
        if content is not None and not isinstance(content, str):
            raise OpenAISSEError("chat stream content must be a string or null")
        normalized_reasoning = reasoning if isinstance(reasoning, str) else None
        normalized_content = content if isinstance(content, str) else None
        if normalized_reasoning is not None or normalized_content is not None:
            events.append(DeltaEvent(normalized_reasoning, normalized_content))
    if not events:
        return ()
    if len(events) > 1:
        raise OpenAISSEError("chat stream contained multiple text choices")
    return (events[0],)
