"""SSE framing helpers for the recording proxy."""

from __future__ import annotations

import json
from typing import Any

_POST_DONE_SUFFIX_LIMIT = 1024


class SseDoneGate:
    def __init__(self) -> None:
        self._buffer = b""
        self._done = bytearray()
        self.done_event: bytes | None = None

    @property
    def terminated(self) -> bool:
        return self.done_event is not None

    def feed(self, chunk: bytes) -> list[bytes]:
        if self.done_event is not None:
            return []
        self._buffer += chunk
        if self._done:
            self._consume_done_suffix()
            return []

        cursor = 0
        while True:
            newline = self._buffer.find(b"\n", cursor)
            if newline < 0:
                break
            content = self._buffer[cursor:newline].rstrip(b"\r")
            if content.startswith(b"data:") and content[len(b"data:") :].strip() == b"[DONE]":
                forwarded = self._buffer[:cursor]
                self._done.extend(self._buffer[cursor : newline + 1])
                self._buffer = self._buffer[newline + 1 :]
                self._consume_done_suffix()
                return [forwarded] if forwarded else []
            cursor = newline + 1

        trailing = self._buffer[cursor:]
        if _could_be_done_line(trailing):
            forwarded = self._buffer[:cursor]
            self._buffer = trailing
        else:
            forwarded = self._buffer
            self._buffer = b""
        return [forwarded] if forwarded else []

    def finish(self) -> list[bytes]:
        if self._done:
            self._done.extend(self._buffer)
            self._buffer = b""
            self.done_event = bytes(self._done)
            self._done.clear()
            return []
        forwarded = self._buffer
        self._buffer = b""
        return [forwarded] if forwarded else []

    def _consume_done_suffix(self) -> None:
        # a `data: [DONE]` line can be followed by further lines of the SAME event: in SSE only a
        # blank line ends one. those continuation lines belong to the terminator and are relayed
        # with it, so they are retained rather than dropped -- dropping them made the relay lossy.
        while True:
            if len(self._done) + len(self._buffer) > _POST_DONE_SUFFIX_LIMIT:
                # an upstream that sends `[DONE]` and then never delimits it would otherwise grow
                # this buffer without bound, outside the accumulator's budget. settle on what the
                # terminator already is and stop retaining the rest.
                self._settle_done()
                return
            newline = self._buffer.find(b"\n")
            if newline < 0:
                return
            blank = not self._buffer[:newline].rstrip(b"\r")
            self._done.extend(self._buffer[: newline + 1])
            self._buffer = self._buffer[newline + 1 :]
            if blank:
                # the delimiter closed the event; anything past it is post-terminator data that
                # this gate has already decided to stop at, so it must not be relayed later.
                self._settle_done()
                return

    def _settle_done(self) -> None:
        self.done_event = bytes(self._done)
        self._done.clear()
        self._buffer = b""


def _could_be_done_line(line: bytes) -> bool:
    prefix = b"data:"
    if len(line) < len(prefix):
        return prefix.startswith(line)
    if not line.startswith(prefix):
        return False
    data = line[len(prefix) :].lstrip()
    target = b"[DONE]"
    return target.startswith(data) or (
        data.startswith(target) and not data[len(target) :].strip(b" \t\r")
    )


class _StringFragments:
    def __init__(self, value: str) -> None:
        self.parts = [value] if value else []

    def append(self, value: str) -> None:
        if value:
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
        elif isinstance(value, list):
            _append_fragment(target, key, value)
        elif value is not None:
            target[key] = value


class SseAccumulator:
    def __init__(self, *, max_accumulated_bytes: int | None = None) -> None:
        self._buffer = b""
        self._event_data: list[bytes] = []
        self._event_data_bytes = 0
        self._choices: dict[int, dict[str, Any]] = {}
        self._envelope: dict[str, Any] = {}
        self._done = False
        self._max_accumulated_bytes = max_accumulated_bytes
        self._accumulated_bytes = 0
        self._overwriting_sizes: dict[str, int] = {}
        self.truncated = False
        self.usage: Any = None
        # why this stream cannot be trusted as a complete reply, if anything went wrong in it. a
        # 200 SSE stream can still fail: an unparseable `data:` event drops a fragment out of the
        # middle of the text, and a `data: {"error": ...}` envelope reports a mid-stream failure.
        # both used to leave the span `OK`, so `records` exported the surviving text as a complete
        # training target with a hole in it.
        self.defect: str | None = None

    def feed(self, chunk: bytes) -> None:
        if self.truncated:
            return
        cursor = 0
        while cursor < len(chunk):
            newline = chunk.find(b"\n", cursor)
            end = len(chunk) if newline < 0 else newline
            fragment = chunk[cursor:end]
            if self._max_accumulated_bytes is not None and len(
                fragment
            ) > self._max_accumulated_bytes - len(self._buffer):
                self._buffer = b""
                self.truncated = True
                return
            self._buffer += fragment
            if newline < 0:
                return
            line, self._buffer = self._buffer, b""
            self._consume_line(line.rstrip(b"\r"))
            if self.truncated:
                return
            cursor = newline + 1

    def finish(self) -> None:
        if self._buffer:
            self._consume_line(self._buffer.rstrip(b"\r"))
            self._buffer = b""
        self._consume_event()

    def _note_defect(self, reason: str) -> None:
        """Record the FIRST thing that went wrong; later ones are usually consequences of it."""
        if self.defect is None:
            self.defect = reason

    def _value_size(self, value: Any) -> int:
        if isinstance(value, str):
            return len(value.encode("utf-8"))
        if isinstance(value, bytes):
            return len(value)
        # measuring fragments instead of serializing the whole accumulated envelope keeps each
        # chunk constant-time. json overhead only makes this an approximate storage budget, so
        # the explicit truncation marker remains the authoritative export signal.
        return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def _reserve(self, value: Any) -> bool:
        if self.truncated or self._max_accumulated_bytes is None:
            return not self.truncated
        size = self._value_size(value)
        if size > self._max_accumulated_bytes - self._accumulated_bytes:
            self.truncated = True
            return False
        self._accumulated_bytes += size
        return True

    def _reserve_overwriting(self, key: str, value: Any) -> bool:
        if self.truncated or self._max_accumulated_bytes is None:
            return not self.truncated
        previous_size = self._overwriting_sizes.get(key, 0)
        size = self._value_size(value)
        retained_bytes = self._accumulated_bytes - previous_size
        if size > self._max_accumulated_bytes - retained_bytes:
            self.truncated = True
            return False
        self._accumulated_bytes = retained_bytes + size
        self._overwriting_sizes[key] = size
        return True

    @property
    def has_error(self) -> bool:
        return "error" in self._envelope

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
            choice = {
                "index": index,
                "message": message,
                "finish_reason": state["finish_reason"],
            }
            if state["logprobs"]:
                choice["logprobs"] = _materialize_fragments(state["logprobs"])
            choices.append(choice)
        return {**self._envelope, "choices": choices, "usage": self.usage}

    def _choice_state(self, index: int) -> dict[str, Any]:
        return self._choices.setdefault(
            index,
            {
                "message": {"role": "assistant"},
                "tool_calls": {},
                "logprobs": {},
                "finish_reason": None,
            },
        )

    def _consume_line(self, line: bytes) -> None:
        if not line:
            self._consume_event()
            return
        if not line.startswith(b"data:"):
            return
        data = line[len(b"data:") :].strip()
        added_bytes = len(data) + (1 if self._event_data else 0)
        if self._max_accumulated_bytes is not None and added_bytes > (
            self._max_accumulated_bytes - self._event_data_bytes
        ):
            self._event_data.clear()
            self._event_data_bytes = 0
            self.truncated = True
            return
        self._event_data.append(data)
        self._event_data_bytes += added_bytes

    def _consume_event(self) -> None:
        if not self._event_data:
            return
        data = b"\n".join(self._event_data)
        self._event_data.clear()
        self._event_data_bytes = 0
        if not data:
            return
        if data == b"[DONE]":
            self._done = True
            return
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            self._note_defect("stream contained an unparseable data event")
            return
        if not isinstance(payload, dict):
            self._note_defect("stream contained a non-object data event")
            return
        if payload.get("error") is not None:
            # providers report a mid-stream failure in-band, as a data event on a 200 response,
            # sometimes after real deltas have already arrived. the partial text is not a reply.
            self._note_defect("upstream reported an error mid-stream")
        for key, value in payload.items():
            if key not in {"choices", "usage"} and self._reserve_overwriting(key, value):
                self._envelope[key] = value
        if isinstance(payload.get("usage"), dict) and self._reserve_overwriting(
            "usage", payload["usage"]
        ):
            self.usage = payload["usage"]
        choices = payload.get("choices")
        if not isinstance(choices, list):
            if choices is not None:
                self._note_defect("stream contained non-list choices")
            return
        for position, choice in enumerate(choices):
            if not isinstance(choice, dict):
                self._note_defect("stream choices contained a non-object entry")
                continue
            raw_index = choice.get("index", position)
            if raw_index is None:
                index = position
            elif not isinstance(raw_index, int) or isinstance(raw_index, bool):
                self._note_defect("stream choice contained a non-integer index")
                continue
            else:
                index = raw_index
            if index not in self._choices and not self._reserve(b"x" * 64):
                continue
            state = self._choice_state(index)
            if "delta" in choice:
                delta = choice["delta"]
                if isinstance(delta, dict):
                    self._consume_delta(state, delta)
                elif delta is not None:
                    self._note_defect("stream choice contained a non-object delta")
            if "logprobs" in choice:
                logprobs = choice["logprobs"]
                if isinstance(logprobs, dict):
                    if self._reserve(logprobs):
                        _merge_fragment_dict(state["logprobs"], logprobs)
                elif logprobs is not None:
                    self._note_defect("stream choice contained non-object logprobs")
            # explicit null is the ordinary provider spelling for "no fragment", just like absence.
            # only a present non-null value of the wrong type can have silently lost response data.
            if choice.get("finish_reason") is not None:
                state["finish_reason"] = choice["finish_reason"]

    def _consume_delta(self, state: dict[str, Any], delta: dict[str, Any]) -> None:
        if self.truncated:
            return
        message = state["message"]
        role = delta.get("role")
        if isinstance(role, str) and role and self._reserve(role):
            message["role"] = role
        # every text-shaped delta field, not a fixed pair. providers stream their own alongside the
        # standard ones -- OpenRouter's `reasoning`, audio transcripts -- and an allowlist silently
        # dropped them, so a streamed trace held less than the identical non-streaming call.
        for key, value in delta.items():
            if key in {"role", "function_call", "tool_calls"}:
                continue
            if not self._reserve(value):
                return
            if isinstance(value, dict) and isinstance(message.get(key), dict):
                _merge_fragment_dict(message[key], value)
            elif isinstance(value, str | list) or (
                key in message and isinstance(message[key], str | _StringFragments)
            ):
                _append_fragment(message, key, value)
            elif key not in message:
                message[key] = value
        function_call = delta.get("function_call")
        if isinstance(function_call, dict):
            if self._reserve(function_call):
                target = message.setdefault("function_call", {})
                if isinstance(target, dict):
                    _merge_fragment_dict(target, function_call)
        elif function_call is not None:
            self._note_defect("stream function_call was not an object")
        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            if tool_calls is not None:
                self._note_defect("stream tool_calls was not a list")
            return
        if not self._reserve(tool_calls):
            return
        accumulated_calls: dict[int, dict[str, Any]] = state["tool_calls"]
        for position, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                # `tool_calls: null` means no invocation, but a null or scalar SLOT inside an
                # invocation list means one advertised assistant action was lost from the trace.
                self._note_defect("stream tool_calls contained a non-object entry")
                continue
            raw_index = tool_call.get("index", position)
            if raw_index is None:
                index = position
            elif not isinstance(raw_index, int) or isinstance(raw_index, bool):
                self._note_defect("stream tool_call contained a non-integer index")
                continue
            else:
                index = raw_index
            target = accumulated_calls.setdefault(index, {})
            _merge_fragment_dict(
                target,
                {key: value for key, value in tool_call.items() if key != "index"},
            )
