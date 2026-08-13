"""SSE framing helpers for the recording proxy."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from flash.server.platform import traces as platform_traces

_POST_DONE_SUFFIX_LIMIT = 1024
_UTF8_BOM = b"\xef\xbb\xbf"


def _line_end(data: bytes | bytearray, start: int = 0) -> tuple[int, int] | None:
    for index in range(start, len(data)):
        byte = data[index]
        if byte == 0x0A:
            return index, index + 1
        if byte == 0x0D:
            if index + 1 == len(data):
                return None
            return index, index + 2 if data[index + 1] == 0x0A else index + 1
    return None


def _resume_scan_at(data: bytes | bytearray) -> int:
    return max(0, len(data) - (1 if data.endswith(b"\r") else 0))


class SseDoneGate:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._line_start = 0
        self._scan_start = 0
        self._event_in_progress = False
        self._partial_line_in_progress = False
        self._holding_done_candidate = False
        self._at_stream_start = True
        self._leading_bom = False
        self.done_event: bytes | None = None

    @property
    def terminated(self) -> bool:
        return self.done_event is not None

    def feed(self, chunk: bytes) -> list[bytes]:
        if self.done_event is not None:
            return []
        self._buffer.extend(chunk)
        if self._at_stream_start:
            if len(self._buffer) < len(_UTF8_BOM) and _UTF8_BOM.startswith(self._buffer):
                return []
            self._at_stream_start = False
            self._leading_bom = self._buffer.startswith(_UTF8_BOM)
        forwarded = bytearray()

        while (line_end := _line_end(self._buffer, self._scan_start)) is not None:
            line, next_cursor = line_end
            content = bytes(self._buffer[self._line_start : line])
            if self._leading_bom:
                content = content[len(_UTF8_BOM) :]
                self._leading_bom = False
            continuing_partial_line = self._partial_line_in_progress
            self._partial_line_in_progress = False
            self._line_start = next_cursor
            self._scan_start = next_cursor
            if continuing_partial_line:
                forwarded.extend(self._buffer[:next_cursor])
                del self._buffer[:next_cursor]
                self._line_start = 0
                self._scan_start = 0
                continue
            if not content:
                if self._holding_done_candidate:
                    self.done_event = bytes(self._buffer[:next_cursor])
                    self._buffer.clear()
                    self._line_start = 0
                    self._scan_start = 0
                    return [bytes(forwarded)] if forwarded else []
                forwarded.extend(self._buffer[:next_cursor])
                del self._buffer[:next_cursor]
                self._line_start = 0
                self._scan_start = 0
                self._event_in_progress = False
                continue
            if content.startswith(b"data:"):
                data = _sse_data_value(content)
                if not self._event_in_progress and data == b"[DONE]":
                    self._holding_done_candidate = True
                elif self._holding_done_candidate:
                    # sse joins multiple data lines with a newline, so a second one proves that the
                    # combined event is not the single `[DONE]` terminator the gate is looking for.
                    self._holding_done_candidate = False
                self._event_in_progress = True
            if not self._holding_done_candidate:
                forwarded.extend(self._buffer[:next_cursor])
                del self._buffer[:next_cursor]
                self._line_start = 0
                self._scan_start = 0
            elif not content.startswith(b"data:") and len(self._buffer) > _POST_DONE_SUFFIX_LIMIT:
                if self._line_start < len(self._buffer):
                    forwarded.extend(self._buffer[: self._line_start])
                    del self._buffer[: self._line_start]
                    self._line_start = 0
                    self._scan_start = 0
                    self._holding_done_candidate = False
                    continue
                self._settle_bounded_done()
                return [bytes(forwarded)] if forwarded else []

        if self._holding_done_candidate:
            partial_line = bytes(self._buffer[self._line_start :])
            if partial_line.startswith(b"data:"):
                # a second data line changes the combined event even before its newline arrives.
                self._holding_done_candidate = False
                forwarded.extend(self._buffer)
                self._buffer.clear()
                self._line_start = 0
                self._scan_start = 0
                self._partial_line_in_progress = True
            elif len(self._buffer) > _POST_DONE_SUFFIX_LIMIT:
                self._settle_bounded_done()
            else:
                self._scan_start = _resume_scan_at(self._buffer)
            return [bytes(forwarded)] if forwarded else []

        trailing = bytes(self._buffer)
        parsed_trailing = (
            trailing[len(_UTF8_BOM) :]
            if self._leading_bom and trailing.startswith(_UTF8_BOM)
            else trailing
        )
        if not self._event_in_progress and _could_be_done_line(parsed_trailing):
            if len(trailing) > _POST_DONE_SUFFIX_LIMIT:
                retained = trailing[-_POST_DONE_SUFFIX_LIMIT:]
                forwarded.extend(trailing[: -len(retained)])
                self._buffer = bytearray(retained)
            self._scan_start = _resume_scan_at(self._buffer)
        elif trailing.endswith(b"\r"):
            partial_line = trailing[:-1]
            parsed_partial_line = parsed_trailing[:-1]
            if not self._event_in_progress and _could_be_done_line(parsed_partial_line):
                self._scan_start = _resume_scan_at(self._buffer)
            else:
                if partial_line:
                    forwarded.extend(partial_line)
                    del self._buffer[: len(partial_line)]
                    self._partial_line_in_progress = True
                    self._event_in_progress = (
                        self._event_in_progress or parsed_partial_line.startswith(b"data:")
                    )
                self._line_start = 0
                self._scan_start = 0
        else:
            forwarded.extend(self._buffer)
            self._buffer.clear()
            self._line_start = 0
            self._scan_start = 0
            self._partial_line_in_progress = bool(trailing)
            self._event_in_progress = self._event_in_progress or parsed_trailing.startswith(
                b"data:"
            )
            self._leading_bom = False
        return [bytes(forwarded)] if forwarded else []

    def finish(self) -> list[bytes]:
        if self.done_event is not None:
            return []
        forwarded = b""
        if self._holding_done_candidate:
            self.done_event = bytes(self._buffer)
        else:
            forwarded = bytes(self._buffer)
        self._buffer.clear()
        self._line_start = 0
        self._scan_start = 0
        self._event_in_progress = False
        self._partial_line_in_progress = False
        self._holding_done_candidate = False
        return [forwarded] if forwarded else []

    def _settle_bounded_done(self) -> None:
        self.done_event = bytes(self._buffer[: min(self._line_start, _POST_DONE_SUFFIX_LIMIT)])
        self._buffer.clear()
        self._line_start = 0
        self._scan_start = 0
        self._event_in_progress = False
        self._partial_line_in_progress = False
        self._holding_done_candidate = False


def _sse_data_value(line: bytes) -> bytes:
    data = line[len(b"data:") :]
    return data[1:] if data.startswith(b" ") else data


def _is_padded_done_value(data: bytes) -> bool:
    return data != b"[DONE]" and data.strip(b" \t") == b"[DONE]"


def _could_be_done_line(line: bytes) -> bool:
    prefix = b"data:"
    if len(line) < len(prefix):
        return prefix.startswith(line)
    if not line.startswith(prefix):
        return False
    return b"[DONE]".startswith(_sse_data_value(line))


class _StringFragments:
    def __init__(self, value: str) -> None:
        self.parts = [value] if value else []

    def append(self, value: str) -> None:
        if value:
            self.parts.append(value)

    def text(self) -> str:
        return "".join(self.parts)


def _utf8_safe_text(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return value.encode("utf-16", errors="surrogatepass").decode("utf-16", errors="replace")
    return value


def _materialize_fragments(
    value: Any,
    *,
    depth: int = 0,
    note_defect: Callable[[str], None] | None = None,
) -> Any:
    if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
        if note_defect is not None:
            note_defect("stream output exceeded the maximum nesting depth")
        return "[redacted]"
    if isinstance(value, _StringFragments):
        value = value.text()
    if isinstance(value, str):
        return _utf8_safe_text(value)
    if isinstance(value, dict):
        return {
            _utf8_safe_text(key) if isinstance(key, str) else key: _materialize_fragments(
                item, depth=depth + 1, note_defect=note_defect
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _materialize_fragments(item, depth=depth + 1, note_defect=note_defect) for item in value
        ]
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
            if key in {"id", "type"}:
                current_text = current.text() if isinstance(current, _StringFragments) else current
                if current_text == value:
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
        self._choice_extension_sizes: dict[int, dict[str, int]] = {}
        self._scan_start = 0
        self._at_stream_start = True
        self.truncated = False
        self.usage: Any = None
        # why this stream cannot be trusted as a complete reply, if anything went wrong in it. a
        # 200 SSE stream can still fail: an unparseable `data:` event drops a fragment out of the
        # middle of the text, and a `data: {"error": ...}` envelope reports a mid-stream failure.
        # both used to leave the span `OK`, so `records` exported the surviving text as a complete
        # training target with a hole in it.
        self.defect: str | None = None

    def feed(self, chunk: bytes) -> None:
        if self.truncated or self._done:
            return
        self._buffer += chunk
        if self._at_stream_start:
            if len(self._buffer) < len(_UTF8_BOM) and _UTF8_BOM.startswith(self._buffer):
                return
            self._at_stream_start = False
            if self._buffer.startswith(_UTF8_BOM):
                self._buffer = self._buffer[len(_UTF8_BOM) :]
        while (line_end := _line_end(self._buffer, self._scan_start)) is not None:
            line, next_cursor = line_end
            fragment = self._buffer[:line]
            self._buffer = self._buffer[next_cursor:]
            self._scan_start = 0
            if (
                self._max_accumulated_bytes is not None
                and len(fragment) > self._max_accumulated_bytes
            ):
                self._buffer = b""
                self._scan_start = 0
                self.truncated = True
                return
            self._consume_line(fragment)
            if self.truncated or self._done:
                return
        self._scan_start = _resume_scan_at(self._buffer)
        if (
            self._max_accumulated_bytes is not None
            and len(self._buffer) > self._max_accumulated_bytes
        ):
            self._buffer = b""
            self._scan_start = 0
            self.truncated = True

    def finish(self) -> None:
        if self._buffer:
            self._consume_line(self._buffer.rstrip(b"\r"))
            self._buffer = b""
            self._scan_start = 0
        self._consume_event()

    def _note_defect(self, reason: str) -> None:
        """Record the FIRST thing that went wrong; later ones are usually consequences of it."""
        if self.defect is None:
            self.defect = reason

    def _value_size(self, value: Any) -> int:
        if isinstance(value, bytes):
            return len(value)
        serialized = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        try:
            return len(serialized.encode("utf-8"))
        except UnicodeEncodeError:
            self._note_defect("stream contained text that is not valid utf-8")
            safe_value = _materialize_fragments(value, note_defect=self._note_defect)
            safe_serialized = (
                safe_value
                if isinstance(safe_value, str)
                else json.dumps(safe_value, ensure_ascii=False)
            )
            return len(safe_serialized.encode("utf-8"))

    def _reserve(self, value: Any, *, retained_key: str | None = None) -> bool:
        if self.truncated:
            return False
        size = self._value_size(value)
        if self._max_accumulated_bytes is None:
            return True
        if retained_key is not None:
            size += self._value_size(retained_key) + 4
        if size > self._max_accumulated_bytes - self._accumulated_bytes:
            self.truncated = True
            return False
        self._accumulated_bytes += size
        return True

    def _reserve_overwriting(self, key: str, value: Any) -> bool:
        return self._reserve_overwriting_in(self._overwriting_sizes, key, value)

    def _reserve_choice_extension(self, index: int, key: str, value: Any) -> bool:
        sizes = self._choice_extension_sizes.setdefault(index, {})
        return self._reserve_overwriting_in(sizes, key, value)

    def _reserve_overwriting_in(self, sizes: dict[str, int], key: str, value: Any) -> bool:
        if self.truncated:
            return False
        value_size = self._value_size(value)
        if self._max_accumulated_bytes is None:
            return True
        previous_size = sizes.get(key, 0)
        size = value_size
        if key not in sizes:
            size += self._value_size(key) + 4
        retained_bytes = self._accumulated_bytes - previous_size
        if size > self._max_accumulated_bytes - retained_bytes:
            self.truncated = True
            return False
        self._accumulated_bytes = retained_bytes + size
        sizes[key] = value_size
        return True

    def _tool_call_fragment_size(self, target: dict[str, Any], fragment: dict[str, Any]) -> int:
        size = 0
        for key, value in fragment.items():
            current = target.get(key)
            if isinstance(value, dict) and isinstance(current, dict):
                size += self._tool_call_fragment_size(current, value)
                continue
            if isinstance(value, str) and key in {"id", "type"}:
                current_text = current.text() if isinstance(current, _StringFragments) else current
                if current_text == value:
                    continue
            size += self._value_size(value)
            if key not in target:
                size += self._value_size(key) + 4
        return size

    def _reserve_tool_call_fragment(self, target: dict[str, Any], fragment: dict[str, Any]) -> bool:
        retained_size = self._tool_call_fragment_size(target, fragment)
        return self._reserve(b"x" * retained_size)

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
            message = _materialize_fragments(state["message"], note_defect=self._note_defect)
            tool_calls = state["tool_calls"]
            if tool_calls:
                message["tool_calls"] = [
                    _materialize_fragments(tool_calls[i], note_defect=self._note_defect)
                    for i in sorted(tool_calls)
                ]
            choice = {
                **state["extensions"],
                "index": index,
                "message": message,
                "finish_reason": state["finish_reason"],
            }
            if state["logprobs"]:
                choice["logprobs"] = _materialize_fragments(
                    state["logprobs"], note_defect=self._note_defect
                )
            choices.append(choice)
        return _materialize_fragments(
            {**self._envelope, "choices": choices, "usage": self.usage},
            note_defect=self._note_defect,
        )

    def _choice_state(self, index: int) -> dict[str, Any]:
        return self._choices.setdefault(
            index,
            {
                "message": {"role": "assistant"},
                "tool_calls": {},
                "logprobs": {},
                "extensions": {},
                "finish_reason": None,
            },
        )

    def _consume_line(self, line: bytes) -> None:
        if not line:
            self._consume_event()
            return
        if not line.startswith(b"data:"):
            return
        data = _sse_data_value(line)
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
        event_data = self._event_data
        if len(event_data) > 1 and event_data[0] == b"[DONE]":
            # a later data line proves the first one was not a terminal event. discard that candidate
            # before parsing the remaining joined payload so an oversized open event stays recordable.
            event_data = event_data[1:]
        data = b"\n".join(event_data)
        self._event_data.clear()
        self._event_data_bytes = 0
        if not data:
            return
        if data == b"[DONE]":
            self._done = True
            return
        if _is_padded_done_value(data):
            return
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeDecodeError, RecursionError):
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
            choice_entry_bytes = max(64, self._value_size(index) + 12)
            if index not in self._choices and not self._reserve(b"x" * choice_entry_bytes):
                continue
            state = self._choice_state(index)
            message = choice.get("message")
            if message is not None:
                if isinstance(message, dict):
                    self._consume_delta(state, message)
                else:
                    self._note_defect("stream choice contained a non-object message")
            for key, value in choice.items():
                if key in {"index", "delta", "logprobs", "finish_reason", "message"}:
                    continue
                if self._reserve_choice_extension(index, key, value):
                    state["extensions"][key] = value
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
            if choice.get("finish_reason") is not None and self._reserve(choice["finish_reason"]):
                state["finish_reason"] = choice["finish_reason"]

    def _consume_delta(self, state: dict[str, Any], delta: dict[str, Any]) -> None:
        if self.truncated:
            return
        message = state["message"]
        role = delta.get("role")
        if role is not None:
            if isinstance(role, str) and role:
                if self._reserve(role):
                    message["role"] = role
            else:
                self._note_defect("stream choice contained a non-string role")
        # every text-shaped delta field, not a fixed pair. providers stream their own alongside the
        # standard ones -- OpenRouter's `reasoning`, audio transcripts -- and an allowlist silently
        # dropped them, so a streamed trace held less than the identical non-streaming call.
        for key, value in delta.items():
            if key in {"role", "function_call", "tool_calls"}:
                continue
            if key == "content" and value is not None and not isinstance(value, str | list):
                self._note_defect("stream choice contained malformed content")
                continue
            retained_key = key if key not in message else None
            if not self._reserve(value, retained_key=retained_key):
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
            target = message.setdefault("function_call", {})
            if isinstance(target, dict) and self._reserve_tool_call_fragment(target, function_call):
                _merge_fragment_dict(target, function_call)
        elif function_call is not None:
            self._note_defect("stream function_call was not an object")
        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            if tool_calls is not None:
                self._note_defect("stream tool_calls was not a list")
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
            tool_call_entry_bytes = max(64, self._value_size(index) + 12)
            if index not in accumulated_calls and not self._reserve(b"x" * tool_call_entry_bytes):
                return
            fragment = {key: value for key, value in tool_call.items() if key != "index"}
            target = accumulated_calls.setdefault(index, {})
            if not self._reserve_tool_call_fragment(target, fragment):
                return
            _merge_fragment_dict(target, fragment)
