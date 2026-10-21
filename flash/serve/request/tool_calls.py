"""request-owned tool declaration, schema, history, and capability validation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, DecimalException
from typing import Any

from flash.serve.contract.protocol import MAX_CHAT_REQUEST_BYTES, TEXT_TYPES
from flash.serve.request import text_scan
from flash.serve.request.text_scan import TOOL_CALL_END, TOOL_CALL_START
from flash.serve.request.text_scan import skip_whitespace as _skip_whitespace
from flash.serve.request.tool_template import last_query_index, rendered_turn_prefix

_FUNCTION_START, _FUNCTION_END = text_scan.FUNCTION_START, text_scan.FUNCTION_END
_PARAMETER_START, _PARAMETER_END = text_scan.PARAMETER_START, text_scan.PARAMETER_END
_VIABLE_PARAMETER_END_RE = text_scan.VIABLE_PARAMETER_END_RE

TOOL_PARSER_QWEN3_CODER = "qwen3_coder"
_REPLAY_CONTAINER_TYPE = "replay_container"
_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
# built from chr() because a lone surrogate cannot be written as a source literal.
_SURROGATE_RANGE = re.compile(f"[{chr(0xD800)}-{chr(0xDFFF)}]")
# the node ceiling is per declaration and the tool maximum bounds how many declarations there are,
# so together they already cap a request at 128 x 512 nodes. normalizing that maximum, wiring it,
# and normalizing it again through the serving envelope measures 0.22s, so a separate smaller
# aggregate ceiling would reject valid catalogs without protecting against a measured cost.
_MAX_TOOLS, _MAX_SCHEMA_DEPTH, _MAX_SCHEMA_NODES, _MAX_ENUM_VALUES = 128, 8, 512, 128
# the parser charges four work units per input character, so a request that stays under the
# transport cap can never buy more parser work here than the fixed generation budget allows.
_MAX_REPLAY_TEMPLATE_CHARS = MAX_CHAT_REQUEST_BYTES
_MAX_FIXED_DECIMAL_DIGITS = _MAX_NUMERIC_LITERAL_DIGITS = 1024
_MAX_NUMERIC_LITERAL_EXPONENT = 1_000_000
_MAX_ENUM_INTEGER_MAGNITUDE = 10**_MAX_NUMERIC_LITERAL_DIGITS
_SCHEMA_TYPES = frozenset(["array", "boolean", "integer", "null", "number", "object", "string"])  # fmt: skip
_SCHEMA_KEYS = frozenset(["additionalProperties", "description", "enum", "items", "properties", "required", "type"])  # fmt: skip


@dataclass(frozen=True, slots=True)
class FunctionTool:
    name: str
    description: str | None
    parameters: dict[str, Any]

    def wire(self) -> dict[str, Any]:
        parameters = _json_copy(self.parameters, "parameters", ValueError)
        function = {"name": self.name, "parameters": parameters}
        if self.description is not None:
            function["description"] = self.description
        return {"type": "function", "function": function}


def qualified_tool_parser(base_model: str) -> str | None:
    return TOOL_PARSER_QWEN3_CODER if base_model == "Qwen/Qwen3.5-9B" else None


def normalize_tools(
    value: object, *, error_type: type[Exception] = ValueError
) -> tuple[FunctionTool, ...]:
    if type(value) is not list or not value:
        raise error_type("tools must be a nonempty array of function declarations")
    if len(value) > _MAX_TOOLS:
        raise error_type(f"tools may contain at most {_MAX_TOOLS} declarations")
    normalized: list[FunctionTool] = []
    names: set[str] = set()
    # the enum budget spans the list because enum values are counted against one shared ceiling,
    # while the node budget is per declaration: see the ceiling comments above.
    enum_budget = [0]
    for index, raw in enumerate(value):
        if type(raw) is not dict or set(raw) != {"type", "function"}:
            raise error_type(f"tools[{index}] must contain exactly type and function")
        if raw["type"] != "function":
            raise error_type(f"tools[{index}].type must be function")
        function = raw["function"]
        if type(function) is not dict:
            raise error_type(f"tools[{index}].function must be an object")
        allowed = {"name", "description", "parameters", "strict"}
        unknown = set(function) - allowed
        if unknown or "name" not in function or "parameters" not in function:
            raise error_type(f"tools[{index}].function is not a closed function declaration")
        strict = function.get("strict", False)
        if type(strict) is not bool:
            raise error_type(f"tools[{index}].function.strict must be a boolean")
        if strict:
            raise error_type("strict function tools are not supported")
        name = _identifier_name(function["name"], f"tools[{index}].function.name", error_type)
        if name in names:
            raise error_type(f"duplicate tool function name {name!r}")
        names.add(name)
        description = function.get("description")
        if description is not None and type(description) is not str:
            raise error_type(f"tools[{index}].function.description must be a string")
        budget = [0]
        parameters = _normalize_schema(function["parameters"], f"tools[{index}].function.parameters", error_type, depth=0, budget=budget, enum_budget=enum_budget, root=True)  # fmt: skip
        normalized.append(FunctionTool(name, description, parameters))
    if any(
        _contains_unpaired_surrogate(tool.description)
        or _contains_unpaired_surrogate(tool.parameters)
        for tool in normalized
    ):
        raise error_type("tools cannot contain an unpaired surrogate")
    return tuple(normalized)


def tools_wire(tools: Sequence[FunctionTool] | None) -> list[dict[str, Any]] | None:
    return None if tools is None else [tool.wire() for tool in tools]


def tools_active(tools: object, tool_choice: str | None) -> bool:
    return tools is not None and tool_choice == "auto"


def validate_tool_control_presence(
    tools: object,
    tool_choice: str | None,
    parallel_tool_calls: bool | None,
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    if tools is None and (tool_choice is not None or parallel_tool_calls is not None):
        raise error_type("tool controls require tools")


def validate_tool_request_contract(
    *,
    tools: Sequence[FunctionTool] | None,
    tool_choice: str | None,
    thinking: bool,
    tool_parser: str | None,
    error_type: type[Exception] = ValueError,
) -> None:
    """validate the active tool capability after authoritative thinking resolution."""

    if not tools_active(tools, tool_choice):
        return
    if thinking:
        raise error_type("tools are not supported for thinking-enabled generation")
    if tool_parser != TOOL_PARSER_QWEN3_CODER:
        raise error_type("this serving engine is not qualified for tool calling")


def validate_tool_stop_sequences(
    stop: Sequence[str],
    *,
    tools: Sequence[FunctionTool] | None,
    tool_choice: str | None,
    error_type: type[Exception] = ValueError,
) -> None:
    if not tools_active(tools, tool_choice):
        return
    markers = [TOOL_CALL_START, TOOL_CALL_END, _FUNCTION_START, _FUNCTION_END, _PARAMETER_START, _PARAMETER_END]  # fmt: skip
    for tool in tools:
        markers.append(f"{_FUNCTION_START}{tool.name}>")
        markers.extend(f"{_PARAMETER_START}{name}>" for name in tool.parameters["properties"])
    # both checks are predicates on one stop value, so a repeat can only repeat its own verdict,
    # and `merge_stop_sequences` drops it before generation. bounding the scan by the distinct
    # values keeps a caller from buying event-loop time by the copy.
    distinct_stop = dict.fromkeys(stop)
    marker_chars = sum(len(marker) for marker in markers)
    stop_chars = sum(len(stop_value) for stop_value in distinct_stop)
    if len(distinct_stop) * marker_chars + len(markers) * stop_chars > 16 * 1024 * 1024:
        raise error_type("active tool stop validation exceeds the supported complexity")
    if any(text_scan.overlaps_any(stop_value, markers) for stop_value in distinct_stop) or any(
        stop_value and _skip_whitespace(stop_value, 0) == len(stop_value)
        for stop_value in distinct_stop
    ):
        raise error_type(
            "stop sequences cannot overlap qwen tool-call grammar markers or whitespace separators "
            "when tool_choice='auto'"
        )


def validate_tool_history(
    messages: Sequence[Mapping[str, Any]], *, error_type: type[Exception] = ValueError
) -> None:
    pending: dict[str, str] = {}
    resolved: set[str] = set()
    all_ids: set[str] = set()
    for message_index, message in enumerate(messages):
        role = message.get("role")
        if "tool_calls" in message and role != "assistant":
            raise error_type(f"message {message_index} tool_calls require the assistant role")
        if "tool_call_id" in message and role != "tool":
            raise error_type(f"message {message_index} tool_call_id requires the tool role")
        if role == "assistant" and "tool_calls" in message:
            if pending:
                raise error_type(
                    f"message {message_index} starts a new turn before all tool calls were resolved"
                )
            calls = _validate_history_calls(
                message.get("tool_calls"),
                message_index,
                all_ids,
                error_type,
            )
            pending = dict(calls)
            resolved.clear()
            continue
        if role == "tool":
            if not pending:
                raise error_type(
                    f"message {message_index} tool result has no preceding tool call turn"
                )
            _validate_tool_result(message, message_index, pending, resolved, error_type)
            if resolved == set(pending):
                pending = {}
                resolved.clear()
            continue
        if pending:
            raise error_type(
                f"message {message_index} begins before all preceding tool calls were resolved"
            )
    if pending:
        raise error_type("messages end before all tool calls were resolved")


def validate_tool_history_replay(
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[FunctionTool] | None,
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    """reject historical function arguments the qwen template cannot replay exactly."""
    tool_map = {} if tools is None else {tool.name: tool for tool in tools}
    last_query = last_query_index(messages)
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        calls = tuple(message.get("tool_calls", ()))
        if not calls:
            continue
        probe: list[tuple[FunctionTool, dict[str, Any], dict[str, Any]]] = []
        for call_index, call in enumerate(calls):
            function = call["function"]
            try:
                arguments = _decode_json_object(function["arguments"])
                tool = tool_map.get(function["name"])
                if tool is None:
                    # without a declaration, this checks only whether the call is self-consistent
                    # under a schema derived from its own keys and values. a marker naming a key
                    # absent from this historical call is deliberately plain content, because no
                    # such parameter exists in the replay probe.
                    tool = _historical_replay_tool(function["name"], arguments)
                probe.append((tool, arguments, _template_json_object(function["arguments"])))
            except ValueError as exc:
                raise error_type(f"message {message_index} tool call {call_index} {exc}") from exc
        # the template renders the whole assistant turn as one block, and a competing assignment
        # in a later call can only be seen against that block. validating calls one at a time
        # would accept a turn the parser rejects as ambiguous.
        try:
            _validate_template_roundtrip(
                probe, rendered_turn_prefix(message, message_index > last_query)
            )
        except ValueError as exc:
            raise error_type(f"message {message_index} {exc}") from exc


def detached_template_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    detached: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            detached_content = [
                dict(block) if isinstance(block, dict) else block for block in content
            ]
            copied["content"] = (
                "".join(block["text"] for block in detached_content)
                if copied.get("role") == "tool"
                and all(
                    isinstance(block, dict)
                    and block.get("type") in TEXT_TYPES
                    and isinstance(block.get("text"), str)
                    for block in detached_content
                )
                else detached_content
            )
        calls = copied.get("tool_calls")
        if isinstance(calls, list):
            converted = []
            for call in calls:
                cloned = dict(call)
                function = dict(cloned["function"])
                function["arguments"] = _template_json_object(function["arguments"])
                cloned["function"] = function
                converted.append(cloned)
            copied["tool_calls"] = converted
        detached.append(copied)
    return detached


def _normalize_schema(
    raw: object,
    path: str,
    error_type: type[Exception],
    *,
    depth: int,
    budget: list[int],
    enum_budget: list[int],
    root: bool = False,
    direct: bool = False,
) -> dict[str, Any]:
    if type(raw) is not dict:
        raise error_type(f"{path} must be a JSON Schema object")
    budget[0] += 1
    if depth > _MAX_SCHEMA_DEPTH or budget[0] > _MAX_SCHEMA_NODES:
        raise error_type(f"{path} exceeds the supported schema complexity")
    non_string_keys = [key for key in raw if type(key) is not str]
    if non_string_keys:
        keywords = ", ".join(sorted(repr(key) for key in non_string_keys))
        raise error_type(f"{path} uses unsupported schema keyword(s): {keywords}")
    unknown = set(raw) - _SCHEMA_KEYS
    if unknown:
        raise error_type(f"{path} uses unsupported schema keyword(s): {', '.join(sorted(unknown))}")
    schema_type = raw.get("type")
    if type(schema_type) is not str or schema_type not in _SCHEMA_TYPES:
        raise error_type(f"{path}.type must be one supported JSON Schema type")
    if root and schema_type != "object":
        raise error_type(f"{path} must be a root object schema")
    normalized: dict[str, Any] = {"type": schema_type}
    description = raw.get("description")
    if description is not None:
        if type(description) is not str:
            raise error_type(f"{path}.description must be a string")
        normalized["description"] = description
    if "enum" in raw:
        enum = raw["enum"]
        if type(enum) is not list or not enum or len(enum) > _MAX_ENUM_VALUES:
            raise error_type(f"{path}.enum must be a nonempty bounded array")
        for enum_index, item in enumerate(enum):
            _validate_json_value_complexity(
                item,
                f"{path}.enum[{enum_index}]",
                error_type,
                budget=enum_budget,
                max_nodes=_MAX_ENUM_VALUES * _MAX_SCHEMA_NODES,
                kind="enum value",
            )
        detached = _json_copy(enum, path, error_type)
        if any(not _matches_type(item, schema_type) for item in detached):
            raise error_type(f"{path}.enum values must match {schema_type}")
        # only a value the grammar writes directly between its own delimiters can be made
        # unreadable by carrying one. a string nested inside a container is written as part of that
        # container's json, which `_find_json_container_end` delimits by brace depth and quoting,
        # so the same characters round trip there and rejecting them would refuse a valid schema.
        if (
            direct
            and schema_type == "string"
            and any(_string_enum_conflicts_with_tool_grammar(item) for item in detached)
        ):
            raise error_type(f"{path}.enum contains an unrepresentable tool grammar delimiter")
        fingerprints = [_json_value_fingerprint(item) for item in detached]
        if len(fingerprints) != len(set(fingerprints)):
            raise error_type(f"{path}.enum values must be unique")
        normalized["enum"] = detached
    if schema_type == "object":
        properties = raw.get("properties", {})
        required = raw.get("required", [])
        if type(properties) is not dict or type(required) is not list:
            raise error_type(f"{path}.properties must be an object and required must be an array")
        if raw.get("additionalProperties") is not False:
            raise error_type(f"{path}.additionalProperties must be false")
        if any(type(name) is not str or not name for name in properties):
            raise error_type(f"{path}.properties keys must be nonempty strings")
        if root:
            for name in properties:
                _identifier_name(name, f"{path}.properties key", error_type)
        if any(type(name) is not str for name in required) or len(required) != len(set(required)):
            raise error_type(f"{path}.required must contain unique property names")
        if not set(required) <= set(properties):
            raise error_type(f"{path}.required names must exist in properties")
        normalized["properties"] = {
            name: _normalize_schema(
                child,
                f"{path}.properties.{name}",
                error_type,
                depth=depth + 1,
                budget=budget,
                enum_budget=enum_budget,
                # a property of the root object is written as its own parameter value, so a string
                # one is bounded by the grammar. anything deeper is inside rendered json.
                direct=root,
            )
            for name, child in properties.items()
        }
        normalized["required"] = list(required)
        normalized["additionalProperties"] = False
        if "items" in raw:
            raise error_type(f"{path} object schema contains array-only keywords")
    elif schema_type == "array":
        if "items" not in raw:
            raise error_type(f"{path} array schemas require items")
        normalized["items"] = _normalize_schema(raw["items"], f"{path}.items", error_type, depth=depth + 1, budget=budget, enum_budget=enum_budget)  # fmt: skip
        forbidden = {"properties", "required", "additionalProperties"} & set(raw)
        if forbidden:
            raise error_type(f"{path} array schema contains object-only keywords")
    elif {"properties", "required", "additionalProperties", "items"} & set(raw):
        raise error_type(f"{path} scalar schema contains container-only keywords")
    return normalized


def _validate_history_calls(
    value: object,
    message_index: int,
    all_ids: set[str],
    error_type: type[Exception],
) -> list[tuple[str, str]]:
    if type(value) is not list or not value:
        raise error_type(f"message {message_index} tool_calls must be a nonempty list")
    calls: list[tuple[str, str]] = []
    for call_index, raw in enumerate(value):
        path = f"message {message_index} tool call {call_index}"
        if type(raw) is not dict or set(raw) != {"id", "type", "function"}:
            raise error_type(f"{path} must contain exactly id, type, and function")
        call_id = raw["id"]
        if type(call_id) is not str or not call_id or call_id != call_id.strip():
            raise error_type(f"{path} id must be a nonempty unpadded string")
        if _contains_unpaired_surrogate(call_id):
            raise error_type(f"{path} id cannot contain an unpaired surrogate")
        if call_id in all_ids:
            raise error_type(f"{path} id is duplicated")
        all_ids.add(call_id)
        if raw["type"] != "function":
            raise error_type(f"{path} type must be function")
        function = raw["function"]
        if type(function) is not dict or set(function) != {"name", "arguments"}:
            raise error_type(f"{path} function must contain exactly name and arguments")
        name = _identifier_name(function["name"], f"{path} function name", error_type)
        arguments = function["arguments"]
        if type(arguments) is not str:
            raise error_type(f"{path} function arguments must be a JSON string")
        try:
            decoded = _decode_json_object(arguments)
        except RecursionError as exc:
            raise error_type(f"{path} exceeds the supported tool argument complexity") from exc
        except ValueError as exc:
            if str(exc).startswith(("numeric literal exceeds", "numeric exponent exceeds")):
                raise error_type(f"{path} {exc}") from exc
            raise error_type(f"{path} function arguments must encode a JSON object") from exc
        _validate_tool_argument_complexity(decoded, path, error_type)
        try:
            # validate through the same rendering the template will perform, so history is
            # accepted exactly when flash can reproduce it faithfully.
            _template_json_object(arguments)
        except ValueError as exc:
            raise error_type(f"{path} {exc}") from exc
        calls.append((call_id, name))
    return calls


def _validate_tool_result(
    message: Mapping[str, Any],
    message_index: int,
    pending: Mapping[str, str],
    resolved: set[str],
    error_type: type[Exception],
) -> None:
    allowed = {"role", "content", "tool_call_id", "name"}
    if set(message) - allowed:
        raise error_type(f"message {message_index} tool result contains unsupported fields")
    call_id = message.get("tool_call_id")
    if type(call_id) is not str or not call_id:
        raise error_type(f"message {message_index} tool_call_id must be a nonempty string")
    if _contains_unpaired_surrogate(call_id):
        raise error_type(
            f"message {message_index} tool_call_id cannot contain an unpaired surrogate"
        )
    if call_id not in pending:
        raise error_type(f"message {message_index} references an unknown tool call id")
    if call_id in resolved:
        raise error_type(f"message {message_index} duplicates a tool result")
    name = message.get("name")
    if name is not None and name != pending[call_id]:
        raise error_type(f"message {message_index} tool result name does not match its call")
    content = message.get("content")
    if type(content) is not str and not (
        type(content) is list
        and all(
            type(block) is dict
            and set(block) == {"type", "text"}
            and block["type"] == "text"
            and type(block["text"]) is str
            for block in content
        )
    ):
        raise error_type(
            f"message {message_index} tool result content must be a string or text blocks"
        )
    texts = (content,) if type(content) is str else (block["text"] for block in content)
    if any(_contains_unpaired_surrogate(text) for text in texts):
        raise error_type("tool result content cannot contain an unpaired surrogate")
    resolved.add(call_id)


def _canonicalize_integer_values(value: Any, schema: Mapping[str, Any]) -> Any:
    schema_type = schema["type"]
    if schema_type == "integer" and type(value) is Decimal:
        return value.to_integral_value()
    if schema_type == "object":
        properties = schema["properties"]
        return {
            name: _canonicalize_integer_values(nested, properties[name])
            for name, nested in value.items()
        }
    if schema_type == "array":
        return [_canonicalize_integer_values(nested, schema["items"]) for nested in value]
    return value


def _matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "null":
        return value is None
    if schema_type == "boolean":
        return type(value) is bool
    if schema_type == "integer":
        return (
            type(value) is int
            or (type(value) is Decimal and _decimal_is_integral(value))
            or (type(value) is float and math.isfinite(value) and value.is_integer())
        )
    if schema_type == "number":
        if type(value) is Decimal:
            return value.is_finite()
        return type(value) is int or (type(value) is float and math.isfinite(value))
    if schema_type == "string":
        return type(value) is str
    if schema_type == "array":
        return type(value) is list
    return type(value) is dict


def _decimal_is_integral(value: Decimal) -> bool:
    if not value.is_finite():
        return False
    digits, exponent = value.as_tuple().digits, value.as_tuple().exponent
    return exponent >= 0 or all(digit == 0 for digit in digits[exponent:])


def _load_exact_json(value: str) -> Any:
    try:
        decoded = json.loads(
            value,
            parse_float=_parse_decimal_literal,
            parse_int=_parse_integer_literal,
            parse_constant=_raise_nonfinite,
            object_pairs_hook=_unique_json_object,
        )
    except DecimalException as exc:
        raise ValueError("invalid decimal number") from exc
    if _contains_unpaired_surrogate(decoded):
        raise ValueError("json contains an unpaired surrogate")
    return decoded


def _decode_json_object(value: str) -> dict[str, Any]:
    if type(decoded := _load_exact_json(value)) is not dict:
        raise ValueError("arguments are not an object")
    return decoded


def _template_json_object(value: str) -> dict[str, Any]:
    decoded = _decode_json_object(value)
    return {name: _template_argument_value(item) for name, item in decoded.items()}


def _historical_replay_tool(name: str, arguments: Mapping[str, Any]) -> FunctionTool:
    # container values are already exact json text at the template boundary, so their nested
    # schema is unknowable and irrelevant here. the private parser-only type preserves the
    # container boundary while the probe checks parameter ownership and scalar coercion.
    return FunctionTool(
        name,
        None,
        {
            "type": "object",
            "properties": {
                field: {"type": _historical_replay_type(value)}
                for field, value in arguments.items()
            },
            "required": list(arguments),
            "additionalProperties": False,
        },
    )


def _historical_replay_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int or (type(value) is Decimal and _decimal_is_integral(value)):
        return "integer"
    if type(value) in {Decimal, float}:
        return "number"
    if type(value) is str:
        return "string"
    return _REPLAY_CONTAINER_TYPE


def _validate_template_roundtrip(
    probe: Sequence[tuple[FunctionTool, dict[str, Any], dict[str, Any]]],
    prefix: str = "",
) -> None:
    from flash.serve.runtime.tool_calls import parse_qwen3_coder_output

    failure = "tool calls cannot be replayed exactly by the tool template"
    # a short numeric literal renders as up to 1024 digits, so the rendered turn can be orders
    # of magnitude larger than the request that carried it. accumulate under an explicit ceiling
    # rather than joining first, so an expanding history is rejected before it is materialized.
    blocks: list[str] = []
    size = 0
    if prefix:
        blocks.append(prefix)
        size += len(prefix)
        if size > _MAX_REPLAY_TEMPLATE_CHARS:
            raise ValueError(failure)
    for tool, decoded, rendered in probe:
        blocks.append(
            f"{TOOL_CALL_START}{_FUNCTION_START}{tool.name}>"
            + "".join(
                f"{_PARAMETER_START}{field}>\n{_render_template_argument(rendered[field])}\n"
                f"{_PARAMETER_END}"
                for field in decoded
            )
            + f"{_FUNCTION_END}{TOOL_CALL_END}"
        )
        size += len(blocks[-1])
        if size > _MAX_REPLAY_TEMPLATE_CHARS:
            raise ValueError(failure)
    text = "".join(blocks)
    replay_tools = _merged_replay_tools(probe)
    # replay must grant exactly what a generation parse of this same text would grant. restating
    # the formula here only lets the two sides drift, and whichever side is narrower rejects a turn
    # the other is willing to emit. the ceiling above already bounds this text, so replay opts out
    # of the fixed cap and closure holds by construction rather than by keeping two formulas equal.
    result = parse_qwen3_coder_output(
        text,
        replay_tools,
        id_factory=lambda: "call_replay",
        _work_limit=None,
    )
    if len(result.calls) != len(probe):
        raise ValueError(failure)
    for call, (_, decoded, _) in zip(result.calls, probe, strict=True):
        if not _json_values_equal(_decode_json_object(call.arguments), decoded):
            raise ValueError(failure)


def _merged_replay_tools(
    probe: Sequence[tuple[FunctionTool, dict[str, Any], dict[str, Any]]],
) -> tuple[FunctionTool, ...]:
    # one turn can call the same function with different optional arguments, so the parser
    # needs the union of observed keys, while only keys present in every call stay required.
    # a declared tool is one shared object for every call naming it, so only self-derived
    # probes can differ under the same name.
    merged: dict[str, FunctionTool] = {}
    for tool, _, _ in probe:
        existing = merged.get(tool.name)
        if existing is None or existing is tool:
            merged[tool.name] = tool
            continue
        properties = dict(existing.parameters["properties"])
        for name, schema in tool.parameters["properties"].items():
            prior = properties.get(name)
            if prior is None:
                properties[name] = schema
                continue
            merged_type = _merged_historical_replay_type(prior["type"], schema["type"])
            if merged_type is None:
                raise ValueError("tool calls cannot be replayed exactly by the tool template")
            properties[name] = {"type": merged_type}
        # a declared schema is normalized against the node budget, so one function can expose at
        # most this many root properties. the union of self-derived probes has no such gate, and a
        # turn repeating one name with disjoint keys could otherwise synthesize a declaration
        # orders of magnitude wider than anything generation can emit. holding the union to the
        # same budget costs no closure and keeps the parser's declaration scan proportional.
        if len(properties) > _MAX_SCHEMA_NODES - 1:
            raise ValueError("tool calls cannot be replayed exactly by the tool template")
        required = set(existing.parameters["required"]) & set(tool.parameters["required"])
        merged[tool.name] = FunctionTool(
            tool.name,
            None,
            {
                "type": "object",
                "properties": properties,
                "required": [name for name in properties if name in required],
                "additionalProperties": False,
            },
        )
    return tuple(merged.values())


def _merged_historical_replay_type(left: str, right: str) -> str | None:
    if left == right:
        return left
    if {left, right} == {"integer", "number"}:
        return "number"
    return None


def _render_template_argument(value: Any) -> str:
    return _dump_template_json(value) if type(value) in {list, dict} else str(value)


def _template_argument_value(value: Any) -> Any:
    """render one tool argument the way the grammar template will consume it.

    the template sends a mapping or sequence through ``tojson`` and every other value
    through ``string``. a number that no native python value can carry faithfully is
    therefore pre-rendered here as its exact compact text, which ``string`` passes
    through unchanged and ``tojson`` never sees. the whole container has to be
    pre-rendered together, because ``tojson`` cannot serialize a nested ``Decimal``.

    booleans and nulls need the same treatment for a different reason: ``string`` spells
    them ``True`` and ``None``, so replaying an emitted ``{"enabled": true}`` would show
    the model python syntax the grammar never produces. inside a container ``tojson``
    already spells them correctly, so only the scalar position is pre-rendered.
    """
    if value is None or type(value) is bool:
        return _dump_exact_json(value)
    try:
        return _native_json_value(value)
    except _InexactTemplateNumber:
        if type(value) in {list, dict}:
            return _dump_template_json(value)
        return _dump_exact_json(value)


def _dump_template_json(value: Any) -> str:
    """serialize a container exactly, matching what the template's ``tojson`` emits.

    transformers replaces jinja's ``tojson`` with plain ``json.dumps`` at default
    spacing and without html escaping, so the pre-rendered form has to use the same
    separators to stay byte-identical to a natively rendered container.
    """
    if type(value) is list:
        return "[" + ", ".join(_dump_template_json(item) for item in value) + "]"
    if type(value) is dict:
        members = (
            f"{json.dumps(key, ensure_ascii=False)}: {_dump_template_json(item)}"
            for key, item in value.items()
        )
        return "{" + ", ".join(members) + "}"
    try:
        # a leaf the template can carry natively must render exactly as it would have
        # without the inexact sibling that forced this container to be pre-rendered.
        # dumping every leaf exactly instead would make one argument's spelling depend
        # on another's magnitude, so ``1.2300`` would keep its trailing zeros and
        # ``-0.0`` its sign only when some unrelated value happened to be oversized.
        return json.dumps(_native_json_value(value), ensure_ascii=False)
    except _InexactTemplateNumber:
        return _dump_exact_json(value)


class _InexactTemplateNumber(ValueError):
    """raised when no native template value carries a decimal without changing it."""


def _native_json_value(value: Any) -> Any:
    if type(value) is Decimal:
        # signed decimal zero needs a native float to replay its decimal point, while signed
        # integer zero has no native carrier and stays exact text. both avoid ``int`` dropping
        # the sign, and integer schemas still canonicalize either form to a decimal integer.
        if value.is_zero() and value.is_signed() and value.as_tuple().exponent == 0:
            raise _InexactTemplateNumber("signed integer zero has no exact native template value")
        if _decimal_is_integral(value) and not (value.is_zero() and value.is_signed()):
            digits, exponent = value.as_tuple().digits, value.as_tuple().exponent
            expanded_digits = 1 if value.is_zero() else len(digits) + exponent
            if expanded_digits > _MAX_FIXED_DECIMAL_DIGITS:
                # expanding here would turn a compact literal into thousands of prompt
                # characters and eventually trip python's own integer-to-string limit.
                # the exact compact text renders identically, so hand it back instead.
                raise _InexactTemplateNumber(
                    f"expanded integer exceeds {_MAX_FIXED_DECIMAL_DIGITS}-digit template limit"
                )
            return int(value)
        try:
            converted = float(value)
        except (OverflowError, ValueError) as exc:
            raise _InexactTemplateNumber(
                "decimal number is not representable as a native template number"
            ) from exc
        if not math.isfinite(converted) or (converted == 0.0 and value != 0):
            raise _InexactTemplateNumber(
                "decimal number is not representable as a native template number"
            )
        # the template renders the float back as its shortest repr, so a value whose repr does not
        # decode to the emitted decimal would show the model a different prior call. this is the
        # rendered identity, not exact binary64 storage: 0.0125 is stored approximately but still
        # renders as itself, while 9007199254740993.1 renders as 9007199254740994.0. the integral
        # branch above already refuses what it cannot expand; refuse here rather than round silently.
        if Decimal(repr(converted)) != value:
            raise _InexactTemplateNumber(
                "decimal number is not representable as a native template number"
            )
        return converted
    if type(value) is list:
        return [_native_json_value(item) for item in value]
    if type(value) is dict:
        return {key: _native_json_value(item) for key, item in value.items()}
    return value


def _contains_unpaired_surrogate(value: Any) -> bool:
    stack = [value]
    while stack:
        nested = stack.pop()
        if type(nested) is str:
            # one native scan for the range utf-8 cannot encode. scanning `ord` per character
            # walks a multi-megabyte argument in the interpreter, and history validation reaches
            # the same string several times. encoding it would answer the same question natively
            # but allocates a whole second copy, which turns a near-cap argument into a possible
            # `MemoryError` on a path that must only ever return a verdict. every surrogate is
            # above the ascii range, so the cached ascii flag settles the common argument without
            # scanning it at all.
            if not nested.isascii() and _SURROGATE_RANGE.search(nested) is not None:
                return True
        elif type(nested) is list:
            stack.extend(nested)
        elif type(nested) is dict:
            stack.extend((*nested, *nested.values()))
    return False


def _parse_decimal_literal(value: str) -> Decimal:
    significand, separator, exponent = value.lower().lstrip("-").partition("e")
    digits = significand.replace(".", "")
    if len(digits) > _MAX_NUMERIC_LITERAL_DIGITS:
        raise ValueError(f"numeric literal exceeds {_MAX_NUMERIC_LITERAL_DIGITS}-digit limit")
    if separator:
        magnitude = exponent.lstrip("+-0")
        limit = str(_MAX_NUMERIC_LITERAL_EXPONENT)
        if len(magnitude) > len(limit) or (len(magnitude) == len(limit) and magnitude > limit):
            raise ValueError(
                f"numeric exponent exceeds {_MAX_NUMERIC_LITERAL_EXPONENT} magnitude limit"
            )
    return Decimal(value)


def _parse_integer_literal(value: str) -> int | Decimal:
    parsed = _parse_decimal_literal(value)
    return parsed if parsed.is_zero() and parsed.is_signed() else int(parsed)


def _raise_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite json constant {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise ValueError(f"duplicate json object key {key!r}")
        value[key] = nested
    return value


def _json_value_fingerprint(value: Any) -> tuple[Any, ...]:
    if _is_json_number(value):
        return ("number", _as_decimal(value))
    if type(value) is list:
        return ("array", *(_json_value_fingerprint(item) for item in value))
    if type(value) is dict:
        return ("object", *((key, _json_value_fingerprint(value[key])) for key in sorted(value)))
    return (type(value).__name__, value)


def _json_values_equal(left: Any, right: Any) -> bool:
    if _is_json_number(left) and _is_json_number(right):
        return _as_decimal(left) == _as_decimal(right)
    if type(left) is not type(right):
        return False
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if type(left) is dict:
        return set(left) == set(right) and all(
            _json_values_equal(left[key], right[key]) for key in left
        )
    return left == right


def _is_json_number(value: Any) -> bool:
    return type(value) in {Decimal, float, int}


def _as_decimal(value: Decimal | float) -> Decimal:
    if type(value) in {Decimal, int}:
        return Decimal(value)
    return Decimal(json.dumps(value, allow_nan=False))


def _dump_exact_json(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is Decimal:
        if not value.is_finite():
            raise ValueError("non-finite decimal")
        exponent = value.as_tuple().exponent
        if exponent >= 0 and len(value.as_tuple().digits) + exponent <= _MAX_FIXED_DECIMAL_DIGITS:
            return format(value, "f")
        return str(value).replace("E", "e")
    if type(value) is float:
        return json.dumps(value, allow_nan=False, separators=(",", ":"))
    if type(value) is str:
        return json.dumps(value, ensure_ascii=False)
    if type(value) is list:
        return "[" + ",".join(_dump_exact_json(item) for item in value) + "]"
    if type(value) is dict:
        members = (
            f"{json.dumps(key, ensure_ascii=False)}:{_dump_exact_json(item)}"
            for key, item in value.items()
        )
        return "{" + ",".join(members) + "}"
    raise TypeError(f"unsupported exact JSON value {type(value).__name__}")


def _validate_tool_argument_complexity(value: Any, path: str, error_type: type[Exception]) -> None:
    _validate_json_value_complexity(
        value, path, error_type, budget=[0], max_nodes=_MAX_SCHEMA_NODES, kind="tool argument"
    )


def _validate_json_value_complexity(
    value: Any,
    path: str,
    error_type: type[Exception],
    *,
    budget: list[int],
    max_nodes: int,
    kind: str,
) -> None:
    stack = [(value, 0)]
    while stack:
        nested, depth = stack.pop()
        budget[0] += 1
        if depth > _MAX_SCHEMA_DEPTH or budget[0] > max_nodes:
            raise error_type(f"{path} exceeds the supported {kind} complexity")
        if kind == "enum value" and type(nested) in {float, Decimal}:
            raise error_type(f"{path} numeric enum members must be JSON integers")
        if (
            kind == "enum value"
            and type(nested) is int
            and (nested <= -_MAX_ENUM_INTEGER_MAGNITUDE or nested >= _MAX_ENUM_INTEGER_MAGNITUDE)
        ):
            raise error_type(
                f"{path} numeric literal exceeds {_MAX_NUMERIC_LITERAL_DIGITS}-digit limit"
            )
        if type(nested) is list:
            stack.extend((item, depth + 1) for item in nested)
        elif type(nested) is dict:
            if kind == "enum value" and any(type(key) in {float, Decimal} for key in nested):
                raise error_type(f"{path} numeric enum members must be JSON integers")
            stack.extend((item, depth + 1) for item in nested.values())


def _json_copy(value: Any, path: str, error_type: type[Exception]) -> Any:
    if type(value) is list:
        return [_json_copy(item, path, error_type) for item in value]
    if type(value) is dict:
        detached: dict[str, Any] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise error_type(f"{path} must contain only string-keyed JSON objects")
            detached[key] = _json_copy(nested, path, error_type)
        return detached
    if type(value) is float:
        if not math.isfinite(value):
            raise error_type(f"{path} must contain only finite JSON values")
        return value
    if type(value) in {bool, int, str, type(None)}:
        return value
    raise error_type(f"{path} must contain only exact JSON values")


def _identifier_name(value: object, path: str, error_type: type[Exception]) -> str:
    if type(value) is str and _NAME_RE.fullmatch(value) is not None:
        return value
    raise error_type(f"{path} is invalid")


def _string_enum_conflicts_with_tool_grammar(value: str) -> bool:
    """whether an enum value could close its own parameter and be read as grammar.

    an enum value reaches the megabytes and this runs synchronously on the request path, so
    stepping to every inert delimiter in python holds the event loop. only a delimiter followed,
    after whitespace, by the next parameter or the function end can close a value, so the engine
    finds one in a single native scan instead of one python iteration per occurrence.
    """
    return _VIABLE_PARAMETER_END_RE.search(value) is not None
