"""strict qwen3 coder response parsing adapted from vllm 0.23.0 under apache-2.0."""

from __future__ import annotations

import re
import uuid
from array import array
from bisect import bisect_left
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import DecimalException
from typing import Any, NamedTuple

from flash.serve.request import text_scan
from flash.serve.request.text_scan import TOOL_CALL_END, TOOL_CALL_START
from flash.serve.request.text_scan import skip_whitespace as _skip_whitespace
from flash.serve.request.tool_calls import (
    _REPLAY_CONTAINER_TYPE,
    FunctionTool,
    _canonicalize_integer_values,
    _contains_unpaired_surrogate,
    _dump_exact_json,
    _json_values_equal,
    _load_exact_json,
    _matches_type,
    _validate_tool_argument_complexity,
)
from flash.serve.request.validation import MAX_MESSAGE_NODES

_FUNCTION_START, _FUNCTION_END = text_scan.FUNCTION_START, text_scan.FUNCTION_END
_PARAMETER_START, _PARAMETER_END = text_scan.PARAMETER_START, text_scan.PARAMETER_END
_CALL_BOUNDARY_RE = re.compile(r"</function>\s*</tool_call>\s*(<tool_call>)\s*<function=([^>]+)>")
_WHITESPACE_RE = re.compile(r"\s*")
# the body of a json string literal up to and including its closing quote. the ordinary run and
# the escape start on disjoint characters and both quantifiers are possessive, so there is exactly
# one way to match at each position and the engine cannot backtrack: an unterminated string fails
# in one pass rather than retrying every split of the run.
_STRING_BODY_RE = re.compile(r'(?:[^"\\]++|\\.)*+"', re.DOTALL)
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
        # a candidate is buffered whole because a malformed later fragment has to fall back to the
        # exact text, and a delta already emitted cannot be retracted. concatenating onto a string
        # per delta would copy the whole buffer each time, so a candidate spanning the context
        # window costs work in the square of its length. the parts are joined once at the end.
        self._parts: list[str] = []
        self._candidate = False

    @property
    def _pending(self) -> str:
        return "".join(self._parts)

    def feed(self, text: str) -> str:
        if not text:
            return ""
        # once a candidate is open every delta is retained to the end, so nothing scans the buffer
        # again until `finish`. appending is what keeps that accumulation linear.
        if self._candidate:
            self._parts.append(text)
            return ""
        # before a candidate opens, everything but a partial marker suffix is emitted each time, so
        # the buffer stays bounded by the marker length and joining it here stays cheap.
        pending = self._pending + text
        marker = pending.find(TOOL_CALL_START)
        if marker >= 0:
            self._parts = [pending[marker:]]
            self._candidate = True
            return pending[:marker]
        retain = _partial_marker_suffix(pending)
        if retain:
            self._parts = [pending[-retain:]]
            return pending[:-retain]
        self._parts = []
        return pending

    def finish(self) -> ToolParseResult:
        pending, self._parts = self._pending, []
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
    _work_limit: int | None = 16 * 1024 * 1024,
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
    # ``candidates`` is a context-blind scan, so a boundary quoted inside a json string argument
    # looks like a call here. the emitted-call ceiling is applied after parsing, where the real
    # count is known; this budget only has to bound the work that reaching that point costs.
    # each call scans its declared parameters once, so a schema with many optional properties
    # costs work that the generated text does not pay for. that scan is bounded by the schema
    # node limit, so charging it here keeps a minimal valid call parseable under any declaration.
    declared = max((len(tool.parameters["properties"]) for tool in tools), default=0)
    scans = min(len(candidates) + 1, _MAX_POTENTIALLY_REPLAYABLE_CALLS)
    # generation caps the input-proportional budget so untrusted output cannot buy unbounded work.
    # replay passes ``None`` instead: its text is already bounded by the render ceiling, and any
    # fixed cap here would make replay the narrower side and reject a call generation just emitted.
    budget = 4 * len(text) + scans * declared
    work = [budget if _work_limit is None else min(_work_limit, budget)]
    declared_names = frozenset(name for tool in tools for name in tool.parameters["properties"])
    opener_positions = _index_parameter_openers(text, first, declared_names, work)
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
                # `parsed` only grows, so once it passes the ceiling the result below is already
                # decided and every remaining candidate is work spent on a rejected parse. this
                # counts confirmed parses rather than candidate boundaries: a single call whose
                # argument quotes many boundaries still confirms once, so quoting cannot trip it.
                if len(parsed) > _MAX_POTENTIALLY_REPLAYABLE_CALLS:
                    return ToolParseResult(content=text, calls=())
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
    text: str, start: int, declared: frozenset[str], work: list[int]
) -> dict[str, array[int]] | object:
    """offsets of every ``<parameter=name>`` opener, for the names a declaration can ask about.

    both readers look positions up by a declared parameter name, so an opener naming anything
    else can never be consulted. keeping one would let an untrusted argument spend hundreds of
    megabytes on offsets nothing reads, so the index retains only what the schema can reach.

    a declared name is whatever the schema holds, which for a replay probe is the keys of a
    historical call: those never pass `_identifier_name`, so reading the name by the charset a
    public declaration may hold left the opener a key like `weird key` spells out of the index,
    and a history the template renders exactly was rejected as unreplayable. that widens what a
    single untrusted argument can retain, since it may now quote a non-identifier declared name a
    million times, but not beyond the input-proportional worst case an identifier name already had.

    a declared name is a different matter: a single valid argument may quote its own parameter
    a million times, and every one of those offsets is genuinely reachable, because the readers
    ask for the greatest offset below a scope end that is not known until the call is parsed.
    they cannot be dropped or collapsed without changing which opener a lookup finds, so they are
    held in a compact integer array rather than a list of boxed python ints. the offsets are
    discovered in increasing order, which is exactly the order ``bisect`` needs.
    """
    if not _consume_work(work, len(text) - start):
        return _EXHAUSTED
    # one pass over the text, not one per declared name: a wide schema may declare hundreds of
    # parameters, and scanning the whole input for each of them would cost more on an ordinary
    # request than the quoted-opener shape this index is guarding against costs today.
    #
    # the name is read the way the parser reads it, by the next `>`, rather than matched by a
    # pattern. a pattern whose name run is the parser's must retry that run at every `<parameter=`
    # it cannot terminate, which is quadratic on a run of unterminated openers, and this executes
    # synchronously on model output.
    #
    # two things have to stay off the per-opener path for that to hold on a run of openers that do
    # share a delimiter. the delimiter search is not restarted, because the first `>` at or after a
    # later opener is the one already found unless that one now lies behind it: each search
    # therefore begins strictly later than the last and the scan crosses the text once. and a name
    # is cut out of the text only at a width some declaration holds: openers sharing one far
    # delimiter spell out names as long as the run itself, so copying every one of them to hash it
    # would cost the square of the run's length in slices alone.
    #
    # a name equals a declaration only if it is exactly as wide, so a width no declaration holds is
    # settled without reading the opener at all. bounding the width by the longest declared name
    # instead would be defeated by the declaration itself: a replay probe declares the keys of a
    # historical call, so one key as long as the run makes every opener in it wide enough to copy.
    #
    # the widths a declaration does hold are few, and only they can copy an opener out to hash it.
    # the openers ending at one delimiter have distinct widths, so it copies at most one name per
    # declared width, and each of those names is at most as long as the distance it spans. for a
    # fixed declaration that is proportional to the text; across declarations it is not, because
    # the number of distinct declared widths grows with the declaration itself. a replay probe
    # declaring many keys of many lengths therefore costs the text times that width count, which
    # a wide enough declaration makes materially superlinear in the input.
    #
    # the span is charged against the budget once, above, and the copies are not charged again.
    # so the work limit bounds the span, not the copying it admits: the copies exceed it by the
    # number of declared widths, and that factor is what the limit does not contain. billing the
    # same characters a second time is not the fix, because it exhausts calls that parse without
    # the width check at all; scanning once per declared name instead only relocates the cost.
    positions: dict[str, array[int]] = {}
    widths = frozenset(len(name) for name in declared)
    cursor, name_end = text.find(_PARAMETER_START, start), -1
    while cursor >= 0:
        name_start = cursor + len(_PARAMETER_START)
        if name_end < name_start:
            name_end = text.find(">", name_start)
            if name_end < 0:
                # no delimiter remains, so no opener can be completed in what is left.
                break
        if name_end - name_start in widths and (name := text[name_start:name_end]) in declared:
            if (found := positions.get(name)) is None:
                found = positions[name] = array("q")
            found.append(cursor)
        # a name may span a later `<parameter=`, exactly as the parser reads it, so the next search
        # starts after this opener's own marker rather than after the name it took. that keeps an
        # opener nested inside a rejected name discoverable while still advancing every step.
        cursor = text.find(_PARAMETER_START, name_start)
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


def _declares_next_parameter(text: str, cursor: int, tool) -> bool:
    """whether an opener at the cursor names a parameter this tool actually declares."""

    if not text.startswith(_PARAMETER_START, cursor):
        return False
    name_end = text.find(">", cursor + len(_PARAMETER_START))
    if name_end < 0:
        return False
    return text[cursor + len(_PARAMETER_START) : name_end] in tool.parameters["properties"]


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
        # a closer only ends this value if what follows can continue the call, and an opener naming
        # something the schema never declared cannot: `_parse_parameters` would reject it on sight.
        # deciding that here rather than one frame down keeps a value that quotes an undeclared
        # opener a million times from paying for a million rejected recursive branches.
        is_parameter = _declares_next_parameter(text, following, tool)
        is_function = text.startswith(_FUNCTION_END, following)
        if not is_parameter and not is_function:
            # this closer cannot end the value, and neither can any closer before the next one
            # that is followed by a parameter or the function end. a free-string argument reaches
            # the megabytes, so stepping to each inert closer in python holds the event loop for
            # seconds; the engine finds the next viable one in a single native scan.
            viable = text_scan.find_viable_parameter_end(
                text, cursor, tool.parameters["properties"]
            )
            skip_to = len(text) if viable < 0 else viable
            # the span is charged once, for the one native scan that measured it. weighing it by
            # the character is what the skip removed, so charging as if it had happened would put
            # a python loop over every closer back on the request path to buy nothing.
            if not _consume_work(work, skip_to - cursor):
                return _EXHAUSTED
            search_from = skip_to
            if viable < 0:
                return count, witness_cursor, witness_values, None
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
    """offset of the end token that closes a json value, skipping one inside a string literal.

    only a quote can change whether the end token counts, so the scan hops between strings natively
    instead of stepping per character. a replay argument runs to the megabytes and this parses on the
    request path, so stepping would hold the event loop for seconds. the end token's position is
    monotone in the cursor, so it is located once and only relocated after the cursor passes it,
    which keeps a payload of many short strings linear rather than one search per segment.
    """
    start, in_string = cursor, False
    closing = text.find(_PARAMETER_END, cursor)
    while cursor < len(text):
        if in_string:
            quote = text.find('"', cursor)
            if quote < 0:
                break
            # an escape-free segment reaches its closing quote in one native search, which is the
            # ordinary case and the cheapest way to settle it.
            if text.find("\\", cursor, quote) < 0:
                cursor, in_string = quote + 1, False
                continue
            # otherwise the escapes decide where the string ends, so consume the body natively.
            # both branches are possessive and start on disjoint characters, so the match is
            # single-pass: an unterminated segment fails immediately instead of retrying every
            # split of the run, and no slice of the segment is copied to weigh its backslashes.
            body = _STRING_BODY_RE.match(text, cursor)
            if body is None:
                break
            cursor, in_string = body.end(), False
            continue
        if 0 <= closing < cursor:
            closing = text.find(_PARAMETER_END, cursor)
        quote = text.find('"', cursor)
        if closing >= 0 and (quote < 0 or closing < quote):
            scanned = closing - start + len(_PARAMETER_END)
            return closing if _consume_work(work, scanned) else _EXHAUSTED
        if quote < 0:
            break
        cursor, in_string = quote + 1, True
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
