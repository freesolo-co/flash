"""strict qwen3 coder response parsing adapted from vllm 0.23.0 under apache-2.0."""

from __future__ import annotations

import re
import uuid
from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import DecimalException
from typing import Any, NamedTuple

from flash.serve.request.tool_calls import (
    _FUNCTION_END,
    _FUNCTION_START,
    _PARAMETER_END,
    _PARAMETER_START,
    _REPLAY_CONTAINER_TYPE,
    TOOL_CALL_END,
    TOOL_CALL_START,
    FunctionTool,
    _canonicalize_integer_values,
    _contains_unpaired_surrogate,
    _dump_exact_json,
    _json_values_equal,
    _load_exact_json,
    _matches_type,
    _skip_whitespace,
    _validate_tool_argument_complexity,
)
from flash.serve.request.validation import MAX_MESSAGE_NODES

_CALL_BOUNDARY_RE = re.compile(r"</function>\s*</tool_call>\s*(<tool_call>)\s*<function=([^>]+)>")
_PARAMETER_OPEN_RE = re.compile(r"<parameter=([A-Za-z0-9_-]{1,64})>")
_WHITESPACE_RE = re.compile(r"\s*")
_AMBIGUOUS, _EXHAUSTED = object(), object()
# an emitted call is only useful if the client can replay it, and the follow-up request carries
# the whole prior conversation plus the assistant turn and one result per call. the cheapest
# possible continuation costs eight nodes of fixed overhead (the root list, one minimal prior
# message, and the assistant turn) and ten nodes per call: six in the assistant turn plus a
# four-node plain-string result. that is the floor, not the typical case. the optional ``name``
# makes a result five nodes, a single text block seven, and each further block three more, and
# the prior conversation competes for the same budget, so a batch at this ceiling usually will
# not replay: with a history of 1360 text blocks there is no room for even one call. this is
# therefore only a best-case bound that stops the parser emitting a batch no continuation could
# ever carry. the request layer stays the authority on whether a follow-up is accepted.
_MIN_REPLAY_NODES_PER_CALL, _MIN_REPLAY_FIXED_NODES = 10, 8
_MAX_POTENTIALLY_REPLAYABLE_CALLS = (
    MAX_MESSAGE_NODES - _MIN_REPLAY_FIXED_NODES
) // _MIN_REPLAY_NODES_PER_CALL


def _strip_grammar_newline_wrapper(raw: str) -> str:
    """undo the ``\\n`` framing the grammar puts around a parameter value.

    the grammar writes a value as ``<parameter=name>\\n<value>\\n</parameter>``, so only a
    complete both-sided pair is framing. stripping each side independently would collapse
    ``"\\nfoo"``, ``"foo\\n"``, and ``"foo"`` onto one value and invoke the tool with data the
    model never emitted. the length guard keeps a lone ``"\\n"`` as its own single character
    rather than reading one newline as both halves of a wrapper.
    """
    if raw.startswith("\n") and raw.endswith("\n") and len(raw) >= 2:
        return raw[1:-1]
    return raw


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    id: str
    name: str
    arguments: str

    def wire(self, *, index: int | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }
        if index is not None:
            value["index"] = index
        return value


@dataclass(frozen=True, slots=True)
class ToolParseResult:
    content: str | None
    calls: tuple[ParsedToolCall, ...]

    @property
    def tools_called(self) -> bool:
        return bool(self.calls)


class ToolCallStreamParser:
    def __init__(
        self,
        tools: Sequence[FunctionTool],
        *,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._tools = tuple(tools)
        self._id_factory = id_factory
        self._pending = ""
        self._candidate = False

    def feed(self, text: str) -> str:
        if not text:
            return ""
        self._pending += text
        if self._candidate:
            return ""
        marker = self._pending.find(TOOL_CALL_START)
        if marker >= 0:
            emitted = self._pending[:marker]
            self._pending = self._pending[marker:]
            self._candidate = True
            return emitted
        retain = _partial_marker_suffix(self._pending)
        if retain:
            emitted = self._pending[:-retain]
            self._pending = self._pending[-retain:]
        else:
            emitted, self._pending = self._pending, ""
        return emitted

    def finish(self) -> ToolParseResult:
        pending, self._pending = self._pending, ""
        if not pending:
            return ToolParseResult(content=None, calls=())
        if not self._candidate:
            return ToolParseResult(content=pending, calls=())
        result = parse_qwen3_coder_output(pending, self._tools, id_factory=self._id_factory)
        if result.tools_called and result.content is None:
            return result
        return ToolParseResult(content=pending, calls=())


def parse_qwen3_coder_output(
    text: str,
    tools: Sequence[FunctionTool],
    *,
    id_factory: Callable[[], str] | None = None,
) -> ToolParseResult:
    first = text.find(TOOL_CALL_START)
    if first < 0:
        return ToolParseResult(content=text, calls=())
    content = text[:first] or None
    tool_map = {tool.name: tool for tool in tools}
    candidates = [
        match.start(1)
        for match in _CALL_BOUNDARY_RE.finditer(text, first)
        if (tool := tool_map.get(match[2])) is not None
        and _candidate_body_can_start(text, match.end(), tool)
    ]
    # ``candidates`` holds the calls after the first, so the emitted count is one more.
    if len(candidates) + 1 > _MAX_POTENTIALLY_REPLAYABLE_CALLS:
        return ToolParseResult(content=text, calls=())
    work = [min(16 * 1024 * 1024, 4 * len(text))]
    opener_positions = _index_parameter_openers(text, first, work)
    if opener_positions is _EXHAUSTED:
        return ToolParseResult(content=text, calls=())
    # call-boundary discovery advances monotonically. opener discovery limits names to the
    # declaration maximum and charges its full input span once. classification may revisit
    # candidate spans, so those scans charge their actual distance while constant-time prefix and
    # declared-name rejections charge one unit.
    confirmed = [len(text)]
    parsed: list[tuple[str, dict[str, Any]]] = []
    for start in reversed([first, *candidates]):
        parsed_call = _parse_tool_call(text, start, confirmed[-1], tool_map, opener_positions, work)
        if parsed_call is _AMBIGUOUS or parsed_call is _EXHAUSTED:
            return ToolParseResult(content=text, calls=())
        if parsed_call is not None:
            parsed_end = _bounded_whitespace_end(text, parsed_call[0], work)
            if parsed_end is _EXHAUSTED:
                return ToolParseResult(content=text, calls=())
            if parsed_end == confirmed[-1]:
                confirmed.append(start)
                parsed.append(parsed_call[1])
    if confirmed[-1] != first:
        return ToolParseResult(content=text, calls=())
    boundaries = confirmed[::-1]
    parsed.reverse()
    for index, (start, (name, _)) in enumerate(zip(boundaries, parsed, strict=False)):
        if not any(schema["type"] == "string" and "enum" not in schema for schema in tool_map[name].parameters["properties"].values()):  # fmt: skip
            continue
        for scope_end in boundaries[index + 2 :]:
            alternate = _parse_tool_call(text, start, scope_end, tool_map, opener_positions, work)
            if alternate is _AMBIGUOUS or alternate is _EXHAUSTED:
                return ToolParseResult(content=text, calls=())
            if alternate is not None:
                alternate_end = _bounded_whitespace_end(text, alternate[0], work)
                if alternate_end is _EXHAUSTED or alternate_end == scope_end:
                    return ToolParseResult(content=text, calls=())
    try:
        for _, arguments in parsed:
            _validate_tool_argument_complexity(arguments, "generated tool call", ValueError)
    except ValueError:
        return ToolParseResult(content=text, calls=())
    make_id = id_factory or (lambda: f"call_{uuid.uuid4().hex[:24]}")
    calls = tuple(
        ParsedToolCall(_validated_call_id(make_id()), name, _dump_exact_json(arguments))
        for name, arguments in parsed
    )
    return ToolParseResult(content=content, calls=calls)


class _FreeStringSpan(NamedTuple):
    start: int
    end: int


def _candidate_body_can_start(text: str, cursor: int, tool: FunctionTool) -> bool:
    cursor = _skip_whitespace(text, cursor)
    if text.startswith(_FUNCTION_END, cursor):
        return not tool.parameters["required"]
    if not text.startswith(_PARAMETER_START, cursor):
        return False
    name_end = text.find(">", cursor + len(_PARAMETER_START))
    return (
        name_end >= 0
        and text[cursor + len(_PARAMETER_START) : name_end] in tool.parameters["properties"]
    )


def _parse_tool_call(text, cursor, scope_end, tools, opener_positions, work):
    if not _consume_work(work, 1) or not text.startswith(TOOL_CALL_START, cursor):
        return _EXHAUSTED if work[0] < 0 else None
    cursor = _bounded_whitespace_end(text, cursor + len(TOOL_CALL_START), work)
    if cursor is _EXHAUSTED:
        return _EXHAUSTED
    if not text.startswith(_FUNCTION_START, cursor):
        return None
    name_end = text.find(">", cursor + len(_FUNCTION_START))
    name = text[cursor + len(_FUNCTION_START) : name_end]
    if name_end < 0 or (tool := tools.get(name)) is None:
        return None
    properties = tool.parameters["properties"]
    openers = {}
    for parameter_name in properties:
        if not _consume_work(work, 1):
            return _EXHAUSTED
        positions = opener_positions.get(parameter_name, ())
        index = bisect_left(positions, scope_end)
        if index and positions[index - 1] >= name_end:
            openers[parameter_name] = positions[index - 1]
    try:
        parsed = _parse_parameters((text, tool, openers, work), name_end + 1, {}, None)
    except RecursionError:
        # a long run of parameters descends once per value, so a schema wide enough to
        # declare hundreds of them can exhaust the interpreter stack before the work
        # budget notices. that is a candidate flash cannot finish classifying, which is
        # the same answer the budget already gives: keep the exact text.
        return _EXHAUSTED
    if parsed is _EXHAUSTED:
        return _EXHAUSTED
    count, cursor, values, _ = parsed
    if count != 1 or cursor is None or values is None:
        return _AMBIGUOUS if count == 2 else None
    for key, value in values.items():
        if isinstance(value, _FreeStringSpan):
            values[key] = _materialize_span(text, value)
            if values[key] is None:
                return None
    return (cursor, (name, values)) if _validate_value(values, tool.parameters) else None


def _consume_work(work: list[int], amount: int) -> bool:
    work[0] -= amount
    return work[0] >= 0


def _index_parameter_openers(
    text: str, start: int, work: list[int]
) -> dict[str, list[int]] | object:
    if not _consume_work(work, len(text) - start):
        return _EXHAUSTED
    positions: dict[str, list[int]] = {}
    for match in _PARAMETER_OPEN_RE.finditer(text, start, len(text)):
        positions.setdefault(match[1], []).append(match.start())
    return positions


def _bounded_whitespace_end(text: str, cursor: int, work: list[int]) -> int | object:
    limit = min(len(text), cursor + max(work[0], 0))
    match = _WHITESPACE_RE.match(text, cursor, limit)
    assert match is not None
    end = match.end()
    if end == limit and end < len(text) and text[end].isspace():
        _consume_work(work, end - cursor + 1)
        return _EXHAUSTED
    return end if _consume_work(work, end - cursor) else _EXHAUSTED


def _parse_parameters(state, cursor, values, probe):
    text, tool, _, work = state
    parsed_values = dict(values)
    while True:
        cursor = _bounded_whitespace_end(text, cursor, work)
        if cursor is _EXHAUSTED:
            return _EXHAUSTED
        if text.startswith(_FUNCTION_END, cursor):
            outer = _bounded_whitespace_end(text, cursor + len(_FUNCTION_END), work)
            if outer is _EXHAUSTED:
                return _EXHAUSTED
            if not text.startswith(TOOL_CALL_END, outer):
                return 0, None, None, None
            outer += len(TOOL_CALL_END)
            if not set(tool.parameters["required"]) - set(parsed_values):
                return 1, outer, parsed_values, None
            return 0, None, None, outer if probe else None
        if not text.startswith(_PARAMETER_START, cursor):
            return 0, None, None, None
        name_end = text.find(">", cursor + len(_PARAMETER_START))
        name = text[cursor + len(_PARAMETER_START) : name_end]
        schema = tool.parameters["properties"].get(name) if name_end >= 0 else None
        if schema is None or name in parsed_values:
            return 0, None, None, None
        parsed = _parse_parameter_value(state, name_end + 1, schema, parsed_values, name, probe)
        if not isinstance(parsed, tuple) or len(parsed) == 4:
            return parsed or (0, None, None, None)
        cursor, parsed_values[name] = parsed


def _parse_parameter_value(state, value_start, schema, values, name, probe):
    text, tool, _, work = state
    if schema["type"] == "string" and "enum" not in schema:
        missing = frozenset(set(tool.parameters["required"]) - {*values, name})
        return _classify_free_string(
            state, value_start, values, name, probe or (value_start, 0, missing)
        )
    search_from = value_start
    while True:
        value_end = (
            _find_json_container_end(text, search_from, work)
            if schema["type"] in {"array", "object", _REPLAY_CONTAINER_TYPE}
            else _bounded_parameter_end(text, search_from, work)
        )
        if value_end is _EXHAUSTED:
            return _EXHAUSTED
        if value_end < 0:
            return None
        following = _bounded_whitespace_end(text, value_end + len(_PARAMETER_END), work)
        if following is _EXHAUSTED:
            return _EXHAUSTED
        if text.startswith((_PARAMETER_START, _FUNCTION_END), following):
            break
        if schema["type"] in {"array", "object", _REPLAY_CONTAINER_TYPE}:
            return None
        search_from = value_end + len(_PARAMETER_END)
    raw = _strip_grammar_newline_wrapper(text[value_start:value_end])
    value = _coerce_value(raw, schema["type"])
    if not _validate_value(value, schema):
        return None
    try:
        return value_end + len(_PARAMETER_END), _canonicalize_integer_values(value, schema)
    except DecimalException:
        return None


def _resumes_missing_parameter(state, incomplete: int, missing: set[str]) -> bool | object:
    """report whether the span after ``incomplete`` reopens a still-missing parameter.

    a nested candidate that closes short normally ends the search, but when the very
    next value boundary hands off to a parameter this candidate still needs, a second
    valid assignment remains reachable. abandoning there would hide the competing
    interpretation and let an ambiguous candidate parse as one structured call.
    """
    text, _, _, work = state
    search_from = incomplete
    while True:
        value_end = _bounded_parameter_end(text, search_from, work)
        if value_end is _EXHAUSTED:
            return _EXHAUSTED
        if value_end < 0:
            return False
        search_from = value_end + len(_PARAMETER_END)
        following = _bounded_whitespace_end(text, search_from, work)
        if following is _EXHAUSTED:
            return _EXHAUSTED
        if not text.startswith(_PARAMETER_START, following):
            continue
        name_end = text.find(">", following + len(_PARAMETER_START))
        if name_end < 0:
            return False
        if text[following + len(_PARAMETER_START) : name_end] in missing:
            return True


def _classify_free_string(state, value_start, values, name, probe):
    text, tool, openers, work = state
    origin, depth, origin_missing = probe
    count, witness_cursor, witness_values = 0, None, None
    search_from = value_start
    while True:
        value_end = _bounded_parameter_end(text, search_from, work)
        if value_end is _EXHAUSTED:
            return _EXHAUSTED
        if value_end < 0:
            return count, witness_cursor, witness_values, None
        cursor = value_end + len(_PARAMETER_END)
        following = _bounded_whitespace_end(text, cursor, work)
        if following is _EXHAUSTED:
            return _EXHAUSTED
        is_parameter = text.startswith(_PARAMETER_START, following)
        is_function = text.startswith(_FUNCTION_END, following)
        if not is_parameter and not is_function:
            search_from = cursor
            continue
        if is_function:
            function_end = _bounded_whitespace_end(text, following + len(_FUNCTION_END), work)
            if function_end is _EXHAUSTED:
                return _EXHAUSTED
            if not text.startswith(TOOL_CALL_END, function_end):
                search_from = cursor
                continue
        next_values = {**values, name: _FreeStringSpan(value_start, value_end)}
        missing = set(tool.parameters["required"]) - set(next_values)
        content_viable = all(openers.get(field, -1) > following for field in missing) and (
            is_parameter or bool(missing)
        )
        ownership = content_viable and is_parameter and not origin_missing <= set(next_values)
        if ownership and depth >= 2:
            return 2, None, None, None
        branch_probe = (
            (origin, depth + 1, origin_missing) if ownership else probe if is_function else None
        )
        parsed = _parse_parameters(state, following, next_values, branch_probe)
        if parsed is _EXHAUSTED:
            return _EXHAUSTED
        branch_count, branch_cursor, branch_values, incomplete = parsed
        count = min(2, count + branch_count)
        if count == 1 and branch_count:
            witness_cursor, witness_values = branch_cursor, branch_values
        elif count == 2:
            return 2, None, None, None
        if incomplete is not None and count == 0:
            resumes = _resumes_missing_parameter(state, incomplete, missing)
            if resumes is _EXHAUSTED:
                return _EXHAUSTED
            if origin != value_start and not resumes:
                return count, witness_cursor, witness_values, incomplete
            search_from = incomplete
        else:
            search_from = cursor
        if not content_viable:
            return count, witness_cursor, witness_values, None
    return count, witness_cursor, witness_values, None


def _materialize_span(text: str, span: _FreeStringSpan) -> str | None:
    value = _strip_grammar_newline_wrapper(text[span.start : span.end])
    return None if _contains_unpaired_surrogate(value) else value


_find_parameter_end = str.find


def _bounded_parameter_end(text: str, cursor: int, work: list[int]) -> int | object:
    value_end = _find_parameter_end(text, _PARAMETER_END, cursor)
    scanned = len(text) - cursor if value_end < 0 else value_end - cursor + len(_PARAMETER_END)
    return value_end if _consume_work(work, scanned) else _EXHAUSTED


def _find_json_container_end(text: str, cursor: int, work: list[int]) -> int | object:
    start = cursor
    in_string = False
    escaped = False
    while cursor < len(text):
        character = text[cursor]
        if escaped:
            escaped = False
        elif in_string and character == "\\":
            escaped = True
        elif character == '"':
            in_string = not in_string
        elif not in_string and text.startswith(_PARAMETER_END, cursor):
            scanned = cursor - start + len(_PARAMETER_END)
            return cursor if _consume_work(work, scanned) else _EXHAUSTED
        cursor += 1
    return -1 if _consume_work(work, len(text) - start) else _EXHAUSTED


def _coerce_value(value: str, schema_type: str) -> Any:
    if schema_type == "string":
        return value
    if schema_type == "null":
        return None if value.strip() == "null" else value
    if schema_type == "boolean":
        stripped = value.strip()
        if stripped == "true":
            return True
        if stripped == "false":
            return False
        return value
    if schema_type in {"array", "integer", "number", "object", _REPLAY_CONTAINER_TYPE}:
        try:
            return _load_exact_json(value)
        except (RecursionError, TypeError, ValueError):
            return value
    return value


def _validate_value(value: Any, schema: Mapping[str, Any]) -> bool:
    schema_type = schema["type"]
    if schema_type == _REPLAY_CONTAINER_TYPE:
        return type(value) in {dict, list}
    if not _matches_type(value, schema_type):
        return False
    if schema_type == "string" and _contains_unpaired_surrogate(value):
        return False
    if "enum" in schema and not any(_json_values_equal(value, item) for item in schema["enum"]):
        return False
    if schema_type == "object":
        properties = schema["properties"]
        if set(value) - set(properties) or set(schema["required"]) - set(value):
            return False
        return all(_validate_value(nested, properties[name]) for name, nested in value.items())
    if schema_type == "array":
        return all(_validate_value(nested, schema["items"]) for nested in value)
    return True


def _validated_call_id(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise RuntimeError("tool call id factory returned an invalid id")
    return value


def _partial_marker_suffix(text: str) -> int:
    for size in range(min(len(text), len(TOOL_CALL_START) - 1), 0, -1):
        if TOOL_CALL_START.startswith(text[-size:]):
            return size
    return 0
