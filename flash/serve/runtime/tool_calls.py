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
from typing import Any

TOOL_PARSER_QWEN3_CODER = "qwen3_coder"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
_FUNCTION_START = "<function="
_FUNCTION_END = "</function>"
_PARAMETER_START = "<parameter="
_PARAMETER_END = "</parameter>"
_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_MAX_TOOLS = 128
_MAX_SCHEMA_DEPTH = 8
_MAX_SCHEMA_NODES = 512
_MAX_ENUM_VALUES = 128
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
        name = _function_name(function["name"], f"tools[{index}].function.name", error_type)
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
    return tuple(normalized)


def tools_wire(tools: Sequence[FunctionTool] | None) -> list[dict[str, Any]] | None:
    """return a detached OpenAI wire representation."""

    if tools is None:
        return None
    return [tool.wire() for tool in tools]


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
                message.get("tool_calls"), message_index, all_ids, error_type
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

    detached: list[dict[str, Any]] = []
    for message in messages:
        copied = dict(message)
        content = copied.get("content")
        if isinstance(content, list):
            copied["content"] = [
                dict(block) if isinstance(block, dict) else block for block in content
            ]
        calls = copied.get("tool_calls")
        if isinstance(calls, list):
            converted = []
            for call in calls:
                cloned = dict(call)
                function = dict(cloned["function"])
                function["arguments"] = _decode_json_object(function["arguments"])
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
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text):
            break
        if not text.startswith(TOOL_CALL_START, cursor):
            return ToolParseResult(content=text, calls=())
        end = text.find(TOOL_CALL_END, cursor + len(TOOL_CALL_START))
        if end < 0:
            return ToolParseResult(content=text, calls=())
        body = text[cursor + len(TOOL_CALL_START) : end]
        call = _parse_function_body(body, tool_map)
        if call is None:
            return ToolParseResult(content=text, calls=())
        parsed.append(call)
        cursor = end + len(TOOL_CALL_END)
    if not parsed:
        return ToolParseResult(content=text, calls=())
    make_id = id_factory or (lambda: f"call_{uuid.uuid4().hex[:24]}")
    calls = tuple(
        ParsedToolCall(
            id=_validated_call_id(make_id()),
            name=name,
            arguments=json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
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
        detached = _json_copy(enum, path, error_type)
        if any(not _matches_type(item, schema_type) for item in detached):
            raise error_type(f"{path}.enum values must match {schema_type}")
        if len(
            {json.dumps(item, sort_keys=True, separators=(",", ":")) for item in detached}
        ) != len(detached):
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
        name = _function_name(function["name"], f"{path} function name", error_type)
        arguments = function["arguments"]
        if type(arguments) is not str:
            raise error_type(f"{path} function arguments must be a JSON string")
        try:
            _decode_json_object(arguments)
        except ValueError as exc:
            raise error_type(f"{path} function arguments must encode a JSON object") from exc
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
    if type(content) is not str:
        raise error_type(f"message {message_index} tool result content must be a string")
    resolved.add(call_id)


def _parse_function_body(
    body: str, tools: Mapping[str, FunctionTool]
) -> tuple[str, dict[str, Any]] | None:
    stripped = body.strip()
    if not stripped.startswith(_FUNCTION_START) or not stripped.endswith(_FUNCTION_END):
        return None
    opening_end = stripped.find(">", len(_FUNCTION_START))
    if opening_end < 0:
        return None
    name = stripped[len(_FUNCTION_START) : opening_end]
    tool = tools.get(name)
    if tool is None:
        return None
    parameters_text = stripped[opening_end + 1 : -len(_FUNCTION_END)]
    values: dict[str, Any] = {}
    cursor = 0
    while cursor < len(parameters_text):
        while cursor < len(parameters_text) and parameters_text[cursor].isspace():
            cursor += 1
        if cursor == len(parameters_text):
            break
        if not parameters_text.startswith(_PARAMETER_START, cursor):
            return None
        name_end = parameters_text.find(">", cursor + len(_PARAMETER_START))
        if name_end < 0:
            return None
        parameter_name = parameters_text[cursor + len(_PARAMETER_START) : name_end]
        value_end = parameters_text.find(_PARAMETER_END, name_end + 1)
        if value_end < 0 or parameter_name in values:
            return None
        raw_value = parameters_text[name_end + 1 : value_end]
        if raw_value.startswith("\n"):
            raw_value = raw_value[1:]
        if raw_value.endswith("\n"):
            raw_value = raw_value[:-1]
        schema = tool.parameters["properties"].get(parameter_name)
        if schema is None:
            return None
        value = _coerce_value(raw_value, schema["type"])
        if not _validate_value(value, schema):
            return None
        values[parameter_name] = value
        cursor = value_end + len(_PARAMETER_END)
    if set(tool.parameters["required"]) - set(values):
        return None
    if not _validate_value(values, tool.parameters):
        return None
    return name, values


def _coerce_value(value: str, schema_type: str) -> Any:
    if schema_type == "string":
        return value
    if schema_type == "null":
        return None if value.strip().lower() == "null" else value
    if schema_type == "boolean":
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
        return value
    if schema_type == "integer":
        try:
            return int(value)
        except ValueError:
            return value
    if schema_type == "number":
        try:
            number = float(value)
            return int(number) if math.isfinite(number) and number.is_integer() else number
        except ValueError:
            return value
    if schema_type in {"array", "object"}:
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _validate_value(value: Any, schema: Mapping[str, Any]) -> bool:
    schema_type = schema["type"]
    if not _matches_type(value, schema_type):
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    if schema_type == "object":
        properties = schema["properties"]
        if set(value) - set(properties) or set(schema["required"]) - set(value):
            return False
        return all(_validate_value(nested, properties[name]) for name, nested in value.items())
    if schema_type == "array":
        return all(_validate_value(nested, schema["items"]) for nested in value)
    return True


def _matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "null":
        return value is None
    if schema_type == "boolean":
        return type(value) is bool
    if schema_type == "integer":
        return type(value) is int
    if schema_type == "number":
        return type(value) in {int, float} and math.isfinite(float(value))
    if schema_type == "string":
        return type(value) is str
    if schema_type == "array":
        return type(value) is list
    return type(value) is dict


def _decode_json_object(value: str) -> dict[str, Any]:
    decoded = json.loads(value, parse_constant=lambda constant: _raise_nonfinite(constant))
    if type(decoded) is not dict:
        raise ValueError("arguments are not an object")
    return decoded


def _raise_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite json constant {value}")


def _json_copy(value: Any, path: str, error_type: type[Exception]) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise error_type(f"{path} must contain only finite JSON values") from exc


def _function_name(value: object, path: str, error_type: type[Exception]) -> str:
    if type(value) is not str or _NAME_RE.fullmatch(value) is None:
        raise error_type(f"{path} is invalid")
    return value


def _validated_call_id(value: object) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise RuntimeError("tool call id factory returned an invalid id")
    return value


def _partial_marker_suffix(text: str) -> int:
    for size in range(min(len(text), len(TOOL_CALL_START) - 1), 0, -1):
        if TOOL_CALL_START.startswith(text[-size:]):
            return size
    return 0
