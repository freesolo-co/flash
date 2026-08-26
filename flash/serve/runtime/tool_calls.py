"""strict tool declarations, history, and qwen3 coder output parsing.

The XML grammar is adapted from vLLM 0.23.0's
``vllm/tool_parsers/qwen3coder_tool_parser.py`` under Apache-2.0. Flash keeps
this implementation dependency-light and validates every parsed call against
the caller's closed function schema before exposing structured output.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, DecimalException
from typing import Any

TOOL_PARSER_QWEN3_CODER = "qwen3_coder"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
_FUNCTION_START = "<function="
_FUNCTION_END = "</function>"
_PARAMETER_START = "<parameter="
_PARAMETER_END = "</parameter>"
_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_TOOL_GRAMMAR_MARKERS = (
    TOOL_CALL_START,
    TOOL_CALL_END,
    _FUNCTION_START,
    _FUNCTION_END,
    _PARAMETER_START,
    _PARAMETER_END,
)
_MAX_TOOLS = 128
_MAX_SCHEMA_DEPTH = 8
_MAX_SCHEMA_NODES = 512
_MAX_ENUM_VALUES = 128
_MAX_ENUM_NODES = _MAX_ENUM_VALUES * _MAX_SCHEMA_NODES
_MAX_FIXED_DECIMAL_DIGITS = 1024
_MAX_TOOL_STOP_COMPARISON_CHARS = 16 * 1024 * 1024
_SCHEMA_TYPES = frozenset({"array", "boolean", "integer", "null", "number", "object", "string"})
_SCHEMA_KEYS = frozenset(
    {"additionalProperties", "description", "enum", "items", "properties", "required", "type"}
)


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """one normalized function declaration with a closed parameters schema."""

    name: str
    description: str | None
    parameters: dict[str, Any]

    def wire(self) -> dict[str, Any]:
        function: dict[str, Any] = {"name": self.name, "parameters": self.parameters}
        if self.description is not None:
            function["description"] = self.description
        return {"type": "function", "function": function}


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    """one schema-valid parsed function call."""

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
    """parsed calls and visible content, or exact fallback text when unmatched."""

    content: str | None
    calls: tuple[ParsedToolCall, ...]

    @property
    def tools_called(self) -> bool:
        return bool(self.calls)


def qualified_tool_parser(base_model: str) -> str | None:
    """return the parser qualified for one exact logical base model."""
    return TOOL_PARSER_QWEN3_CODER if base_model == "Qwen/Qwen3.5-9B" else None


class ToolCallStreamParser:
    """retain possible tool XML until it can be accepted or returned exactly."""

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
        """return text proven not to begin a tool candidate."""

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
        """parse a complete candidate or return every retained byte as content."""

        pending, self._pending = self._pending, ""
        if not pending:
            return ToolParseResult(content=None, calls=())
        if not self._candidate:
            return ToolParseResult(content=pending, calls=())
        result = parse_qwen3_coder_output(pending, self._tools, id_factory=self._id_factory)
        if result.tools_called and result.content is None:
            return result
        return ToolParseResult(content=pending, calls=())


def normalize_tools(
    value: object, *, error_type: type[Exception] = ValueError
) -> tuple[FunctionTool, ...]:
    """validate and detach the supported closed function-tool declaration list."""

    if type(value) is not list or not value:
        raise error_type("tools must be a nonempty array of function declarations")
    if len(value) > _MAX_TOOLS:
        raise error_type(f"tools may contain at most {_MAX_TOOLS} declarations")
    normalized: list[FunctionTool] = []
    names: set[str] = set()
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
        parameters = _normalize_schema(
            function["parameters"],
            f"tools[{index}].function.parameters",
            error_type,
            depth=0,
            budget=budget,
            root=True,
        )
        normalized.append(FunctionTool(name, description, parameters))
    if _contains_unpaired_surrogate([tool.wire() for tool in normalized]):
        raise error_type("tools cannot contain an unpaired surrogate")
    return tuple(normalized)


def tools_wire(tools: Sequence[FunctionTool] | None) -> list[dict[str, Any]] | None:
    """return a detached OpenAI wire representation."""
    if tools is None:
        return None
    return [tool.wire() for tool in tools]


def tools_active(tools: object, tool_choice: str | None) -> bool:
    """report whether declared tools are active for this request."""
    return tools is not None and tool_choice == "auto"


def validate_tool_control_presence(
    tools: object,
    tool_choice: str | None,
    parallel_tool_calls: bool | None,
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    """reject tool controls when no tool declarations accompany them."""

    if tools is None and (tool_choice is not None or parallel_tool_calls is not None):
        raise error_type("tool controls require tools")


def validate_tool_stop_sequences(
    stop: Sequence[str],
    *,
    tools: Sequence[FunctionTool] | None,
    tool_choice: str | None,
    error_type: type[Exception] = ValueError,
) -> None:
    """reject stop sequences that can consume active qwen tool grammar markers."""
    if not tools_active(tools, tool_choice):
        return
    markers = list(_TOOL_GRAMMAR_MARKERS)
    for tool in tools:
        markers.append(f"{_FUNCTION_START}{tool.name}>")
        markers.extend(f"{_PARAMETER_START}{name}>" for name in tool.parameters["properties"])
    marker_chars = sum(len(marker) for marker in markers)
    stop_chars = sum(len(stop_value) for stop_value in stop)
    if len(stop) * marker_chars + len(markers) * stop_chars > _MAX_TOOL_STOP_COMPARISON_CHARS:
        raise error_type("active tool stop validation exceeds the supported complexity")
    if any(
        _strings_overlap(stop_value, marker) for stop_value in stop for marker in markers
    ) or any(
        stop_value and _skip_whitespace(stop_value, 0) == len(stop_value) for stop_value in stop
    ):
        raise error_type(
            "stop sequences cannot overlap qwen tool-call grammar markers or whitespace separators "
            "when tool_choice='auto'"
        )


def validate_tool_history(
    messages: Sequence[Mapping[str, Any]], *, error_type: type[Exception] = ValueError
) -> None:
    """validate strict assistant-call and immediately-following tool-result lifecycle."""

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


def detached_template_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """clone history and decode assistant argument strings only for chat templates."""

    from flash.serve.request.validation import TEXT_TYPES

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


def parse_qwen3_coder_output(
    text: str,
    tools: Sequence[FunctionTool],
    *,
    id_factory: Callable[[], str] | None = None,
) -> ToolParseResult:
    """parse the exact qualified qwen3 coder XML grammar with exact fallback."""

    first = text.find(TOOL_CALL_START)
    if first < 0:
        return ToolParseResult(content=text, calls=())
    content = text[:first] or None
    cursor = first
    parsed: list[tuple[str, dict[str, Any]]] = []
    tool_map = {tool.name: tool for tool in tools}
    while cursor < len(text):
        cursor = _skip_whitespace(text, cursor)
        if cursor == len(text):
            break
        parsed_call = _parse_tool_call(text, cursor, tool_map)
        if parsed_call is None:
            return ToolParseResult(content=text, calls=())
        cursor, call = parsed_call
        parsed.append(call)
    if not parsed:
        return ToolParseResult(content=text, calls=())
    make_id = id_factory or (lambda: f"call_{uuid.uuid4().hex[:24]}")
    calls = tuple(
        ParsedToolCall(
            id=_validated_call_id(make_id()),
            name=name,
            arguments=_dump_exact_json(arguments),
        )
        for name, arguments in parsed
    )
    return ToolParseResult(content=content, calls=calls)


def _normalize_schema(
    raw: object,
    path: str,
    error_type: type[Exception],
    *,
    depth: int,
    budget: list[int],
    root: bool = False,
) -> dict[str, Any]:
    if type(raw) is not dict:
        raise error_type(f"{path} must be a JSON Schema object")
    budget[0] += 1
    if depth > _MAX_SCHEMA_DEPTH or budget[0] > _MAX_SCHEMA_NODES:
        raise error_type(f"{path} exceeds the supported schema complexity")
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
        enum_budget = [0]
        for enum_index, item in enumerate(enum):
            _validate_json_value_complexity(
                item,
                f"{path}.enum[{enum_index}]",
                error_type,
                budget=enum_budget,
                max_nodes=_MAX_ENUM_NODES,
                kind="enum value",
            )
        detached = _json_copy(enum, path, error_type)
        if any(not _matches_type(item, schema_type) for item in detached):
            raise error_type(f"{path}.enum values must match {schema_type}")
        if schema_type == "string" and any(
            _string_enum_conflicts_with_tool_grammar(item) for item in detached
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
        normalized["items"] = _normalize_schema(
            raw["items"], f"{path}.items", error_type, depth=depth + 1, budget=budget
        )
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
            raise error_type(f"{path} function arguments must encode a JSON object") from exc
        _validate_json_value_complexity(
            decoded,
            path,
            error_type,
            budget=[0],
            max_nodes=_MAX_SCHEMA_NODES,
            kind="tool argument",
        )
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
    resolved.add(call_id)


def _parse_tool_call(
    text: str,
    cursor: int,
    tools: Mapping[str, FunctionTool],
) -> tuple[int, tuple[str, dict[str, Any]]] | None:
    if not text.startswith(TOOL_CALL_START, cursor):
        return None
    cursor = _skip_whitespace(text, cursor + len(TOOL_CALL_START))
    if not text.startswith(_FUNCTION_START, cursor):
        return None
    name_end = text.find(">", cursor + len(_FUNCTION_START))
    if name_end < 0:
        return None
    name = text[cursor + len(_FUNCTION_START) : name_end]
    tool = tools.get(name)
    if tool is None:
        return None
    parsed_parameters = _parse_parameters(text, name_end + 1, tool)
    if parsed_parameters is None:
        return None
    cursor, values = parsed_parameters
    cursor = _skip_whitespace(text, cursor)
    if not text.startswith(TOOL_CALL_END, cursor):
        return None
    return cursor + len(TOOL_CALL_END), (name, values)


def _parse_parameters(
    text: str,
    cursor: int,
    tool: FunctionTool,
) -> tuple[int, dict[str, Any]] | None:
    values: dict[str, Any] = {}
    optional_parameters = set(tool.parameters["properties"]) - set(tool.parameters["required"])
    while True:
        cursor = _skip_whitespace(text, cursor)
        if text.startswith(_FUNCTION_END, cursor):
            cursor += len(_FUNCTION_END)
            break
        if not text.startswith(_PARAMETER_START, cursor):
            return None
        name_end = text.find(">", cursor + len(_PARAMETER_START))
        if name_end < 0:
            return None
        parameter_name = text[cursor + len(_PARAMETER_START) : name_end]
        schema = tool.parameters["properties"].get(parameter_name)
        if schema is None or parameter_name in values:
            return None
        parsed_value = _parse_parameter_value(text, name_end + 1, schema)
        if parsed_value is None:
            return None
        cursor, values[parameter_name] = parsed_value
        next_parameter = _next_parameter(text, cursor)
        if schema["type"] == "string" and "enum" not in schema and next_parameter is not None:
            next_name, next_value_start = next_parameter
            next_schema = tool.parameters["properties"].get(next_name)
            if (
                next_name in optional_parameters
                and next_schema is not None
                and _parse_parameter_value(text, next_value_start, next_schema) is not None
            ):
                return None
    if set(tool.parameters["required"]) - set(values):
        return None
    if not _validate_value(values, tool.parameters):
        return None
    return cursor, values


def _next_parameter(text: str, cursor: int) -> tuple[str, int] | None:
    cursor = _skip_whitespace(text, cursor)
    if not text.startswith(_PARAMETER_START, cursor):
        return None
    name_end = text.find(">", cursor + len(_PARAMETER_START))
    if name_end < 0:
        return None
    name = text[cursor + len(_PARAMETER_START) : name_end]
    return name, name_end + 1


def _parse_parameter_value(
    text: str,
    value_start: int,
    schema: Mapping[str, Any],
) -> tuple[int, Any] | None:
    search_from = value_start
    while True:
        value_end = (
            _find_json_container_end(text, search_from)
            if schema["type"] in {"array", "object"}
            else text.find(_PARAMETER_END, search_from)
        )
        if value_end < 0:
            return None
        following = _skip_whitespace(text, value_end + len(_PARAMETER_END))
        if (
            schema["type"] == "string"
            and "enum" not in schema
            and text.startswith(_FUNCTION_END, following)
        ):
            after_function = _skip_whitespace(text, following + len(_FUNCTION_END))
            if not text.startswith(TOOL_CALL_END, after_function):
                search_from = value_end + len(_PARAMETER_END)
                continue
        if text.startswith((_PARAMETER_START, _FUNCTION_END), following):
            raw_value = text[value_start:value_end]
            if raw_value.startswith("\n"):
                raw_value = raw_value[1:]
            if raw_value.endswith("\n"):
                raw_value = raw_value[:-1]
            value = _coerce_value(raw_value, schema["type"])
            if not _validate_value(value, schema):
                return None
            try:
                value = _canonicalize_integer_values(value, schema)
            except DecimalException:
                return None
            return value_end + len(_PARAMETER_END), value
        search_from = value_end + len(_PARAMETER_END)


def _find_json_container_end(text: str, cursor: int) -> int:
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
            return cursor
        cursor += 1
    return -1


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
    if schema_type in {"array", "integer", "number", "object"}:
        try:
            return _load_exact_json(value)
        except (TypeError, ValueError):
            return value
    return value


def _validate_value(value: Any, schema: Mapping[str, Any]) -> bool:
    schema_type = schema["type"]
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
    if exponent >= 0:
        return True
    fractional_digits = -exponent
    return all(digit == 0 for digit in digits[-fractional_digits:])


def _load_exact_json(value: str) -> Any:
    try:
        decoded = json.loads(
            value,
            parse_float=Decimal,
            parse_int=int,
            parse_constant=_raise_nonfinite,
            object_pairs_hook=_unique_json_object,
        )
    except DecimalException as exc:
        raise ValueError("invalid decimal number") from exc
    if _contains_unpaired_surrogate(decoded):
        raise ValueError("json contains an unpaired surrogate")
    return decoded


def _decode_json_object(value: str) -> dict[str, Any]:
    decoded = _load_exact_json(value)
    if type(decoded) is not dict:
        raise ValueError("arguments are not an object")
    return decoded


def _template_json_object(value: str) -> dict[str, Any]:
    return {
        key: (
            _dump_exact_json(nested)
            if type(nested) in {dict, list} and _contains_decimal(nested)
            else nested
        )
        for key, nested in _decode_json_object(value).items()
    }


def _contains_decimal(value: Any) -> bool:
    if type(value) is Decimal:
        return True
    if type(value) is list:
        return any(_contains_decimal(nested) for nested in value)
    if type(value) is dict:
        return any(_contains_decimal(nested) for nested in value.values())
    return False


def _contains_unpaired_surrogate(value: Any) -> bool:
    stack = [value]
    while stack:
        nested = stack.pop()
        if type(nested) is str:
            if any(0xD800 <= ord(character) <= 0xDFFF for character in nested):
                return True
        elif type(nested) is list:
            stack.extend(nested)
        elif type(nested) is dict:
            stack.extend(nested)
            stack.extend(nested.values())
    return False


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
        return (
            "object",
            *((key, _json_value_fingerprint(value[key])) for key in sorted(value)),
        )
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
    if type(value) is Decimal:
        return value
    if type(value) is int:
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
        return (
            "{"
            + ",".join(
                f"{json.dumps(key, ensure_ascii=False)}:{_dump_exact_json(item)}"
                for key, item in value.items()
            )
            + "}"
        )
    raise TypeError(f"unsupported exact JSON value {type(value).__name__}")


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
        if type(nested) is list:
            stack.extend((item, depth + 1) for item in nested)
        elif type(nested) is dict:
            if kind == "enum value" and any(type(key) in {float, Decimal} for key in nested):
                raise error_type(f"{path} numeric enum members must be JSON integers")
            stack.extend((item, depth + 1) for item in nested.values())


def _json_copy(value: Any, path: str, error_type: type[Exception]) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise error_type(f"{path} must contain only finite JSON values") from exc


def _identifier_name(value: object, path: str, error_type: type[Exception]) -> str:
    if type(value) is not str or _NAME_RE.fullmatch(value) is None:
        raise error_type(f"{path} is invalid")
    return value


def _string_enum_conflicts_with_tool_grammar(value: str) -> bool:
    cursor = 0
    while True:
        cursor = value.find(_PARAMETER_END, cursor)
        if cursor < 0:
            return False
        following = _skip_whitespace(value, cursor + len(_PARAMETER_END))
        if value.startswith((_PARAMETER_START, _FUNCTION_END), following):
            return True
        cursor += len(_PARAMETER_END)


def _strings_overlap(left: str, right: str) -> bool:
    if left in right or right in left:
        return True
    limit = min(len(left), len(right))
    return any(
        left[-size:] == right[:size] or right[-size:] == left[:size] for size in range(1, limit)
    )


def _skip_whitespace(text: str, cursor: int) -> int:
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return cursor


def _validated_call_id(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise RuntimeError("tool call id factory returned an invalid id")
    return value


def _partial_marker_suffix(text: str) -> int:
    for size in range(min(len(text), len(TOOL_CALL_START) - 1), 0, -1):
        if TOOL_CALL_START.startswith(text[-size:]):
            return size
    return 0
