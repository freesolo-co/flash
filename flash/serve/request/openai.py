"""canonical request grammar for the supported openai chat subset."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, cast

from flash.serve.request.tool_calls import (
    FunctionTool,
    normalize_tools,
    tools_active,
    validate_tool_history_replay,
    validate_tool_request_contract,
    validate_tool_stop_sequences,
)
from flash.serve.request.validation import (
    MAX_COMPRESSED_BYTES,
    MAX_SOURCE_CHARS,
    detached_messages,
    has_image_blocks,
    normalize_messages,
    normalize_structured_outputs,
)
from flash.serve.runtime.sampling import (
    validate_choice_count,
    validate_logprobs,
    validate_penalty,
    validate_sampling_relationships,
    validate_seed,
    validate_top_logprobs,
)

DEFAULT_MAX_TOKENS = 1024
_ALLOWED_REQUEST_KEYS = frozenset(
    {
        "checkpoint_id",
        "chat_template_kwargs",
        "frequency_penalty",
        "logprobs",
        "max_tokens",
        "messages",
        "model",
        "n",
        "parallel_tool_calls",
        "presence_penalty",
        "response_format",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "structured_outputs",
        "temperature",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
    }
)
_RESERVED_CHAT_TEMPLATE_KWARGS = frozenset(
    {
        "add_generation_prompt",
        "chat_template",
        "conversation",
        "documents",
        "enable_thinking",
        "max_length",
        "padding",
        "return_assistant_tokens_mask",
        "return_dict",
        "return_tensors",
        "tool_choice",
        "tools",
        "parallel_tool_calls",
        "tokenize",
        "truncation",
    }
)


class OpenAIRequestError(ValueError):
    """one request failed the supported openai wire grammar."""


@dataclass(frozen=True, slots=True)
class NormalizedChatRequest:
    """request-only normalized values, independent of a deployed adapter."""

    model: str | None
    messages: list[dict[str, Any]]
    temperature: float
    max_tokens: int
    top_p: float
    n: int
    seed: int | None
    frequency_penalty: float
    presence_penalty: float
    logprobs: bool
    top_logprobs: int
    stop: tuple[str, ...]
    chat_template_kwargs: dict[str, Any]
    structured_outputs: dict[str, Any] | None
    tools: tuple[FunctionTool, ...] | None
    tool_choice: str | None
    parallel_tool_calls: bool | None
    stream: bool
    stream_options: dict[str, bool] | None

    @property
    def include_usage(self) -> bool:
        return bool(self.stream_options and self.stream_options["include_usage"])


def parse_chat_request(
    payload: object,
    *,
    require_model: bool,
    allow_managed_selectors: bool,
) -> NormalizedChatRequest:
    """validate and normalize the currently supported openai request fields."""

    if type(payload) is not dict:
        raise OpenAIRequestError("request body must be a json object")
    allowed = _ALLOWED_REQUEST_KEYS
    if not allow_managed_selectors:
        allowed = allowed - {"checkpoint_id"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise OpenAIRequestError(f"unsupported chat request field(s): {', '.join(unknown)}")

    model = _optional_model(payload.get("model"), required=require_model)
    messages = _messages(payload.get("messages"))
    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise OpenAIRequestError("stream must be a boolean")
    temperature = _temperature(payload.get("temperature", 0.0))
    max_tokens = _max_tokens(payload.get("max_tokens", DEFAULT_MAX_TOKENS))
    top_p = _top_p(payload.get("top_p", 0.95))
    try:
        n = validate_choice_count(payload.get("n", 1))
        seed = validate_seed(payload.get("seed"))
        frequency_penalty = validate_penalty(
            payload.get("frequency_penalty", 0.0), "frequency_penalty"
        )
        presence_penalty = validate_penalty(
            payload.get("presence_penalty", 0.0), "presence_penalty"
        )
        logprobs = validate_logprobs(payload.get("logprobs", False))
        top_logprobs = validate_top_logprobs(payload.get("top_logprobs", 0))
        validate_sampling_relationships(
            n=n,
            temperature=temperature,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
    except ValueError as exc:
        raise OpenAIRequestError(str(exc)) from exc
    structured_outputs = _structured_outputs(payload)
    tools, tool_choice, parallel_tool_calls = _tool_controls(payload)
    replay_tools = tools if tools_active(tools, tool_choice) else None
    validate_tool_history_replay(messages, replay_tools, error_type=OpenAIRequestError)
    stop = _stop_values(payload.get("stop"))
    validate_tool_stop_sequences(
        stop,
        tools=tools,
        tool_choice=tool_choice,
        error_type=OpenAIRequestError,
    )
    if tools_active(tools, tool_choice):
        if logprobs:
            raise OpenAIRequestError("tools cannot be combined with logprobs")
        if structured_outputs:
            raise OpenAIRequestError("tools cannot be combined with structured outputs")
        response_format = payload.get("response_format")
        if response_format is not None and response_format != {"type": "text"}:
            raise OpenAIRequestError("tools require a text response_format")
        if has_image_blocks(messages, sequence_types=list):
            raise OpenAIRequestError("tools cannot be combined with image messages")

    return NormalizedChatRequest(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        n=n,
        seed=seed,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        stop=stop,
        chat_template_kwargs=_chat_template_kwargs(payload.get("chat_template_kwargs")),
        structured_outputs=structured_outputs,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        stream=stream,
        stream_options=_stream_options(payload.get("stream_options"), stream),
    )


def reject_thinking_logprobs(*, thinking: bool, logprobs: bool) -> None:
    """reject logprobs only after authoritative thinking resolution."""

    if thinking and logprobs:
        raise OpenAIRequestError("logprobs are not supported for thinking-enabled generation")


def reject_tool_capability(
    *,
    tools: tuple[FunctionTool, ...] | None,
    tool_choice: str | None,
    thinking: bool,
    tool_parser: str | None,
) -> None:
    """apply authoritative adapter and engine tool capability checks."""

    validate_tool_request_contract(
        tools=tools,
        tool_choice=tool_choice,
        thinking=thinking,
        tool_parser=tool_parser,
        error_type=OpenAIRequestError,
    )


def merge_stop_sequences(
    mandatory: list[str] | tuple[str, ...], caller: tuple[str, ...]
) -> list[str] | None:
    """merge mandatory and caller stops while preserving first occurrence order."""

    combined: list[str] = []
    seen: set[str] = set()
    for value in (*mandatory, *caller):
        if not isinstance(value, str) or not value or value in seen:
            continue
        seen.add(value)
        combined.append(value)
    return combined or None


def _optional_model(value: object, *, required: bool) -> str | None:
    if value is None and not required:
        return None
    if type(value) is not str or not value.strip() or value != value.strip():
        qualifier = "required" if required else "a non-empty string"
        raise OpenAIRequestError(f"model must be {qualifier}")
    return value


def _max_tokens(value: object) -> int:
    if type(value) is not int or value <= 0:
        raise OpenAIRequestError("max_tokens must be a positive integer")
    return value


def _finite_number(value: object, name: str) -> float:
    if type(value) not in {int, float}:
        raise OpenAIRequestError(f"{name} must be a finite number")
    try:
        normalized = float(cast("int | float", value))
    except OverflowError as exc:
        raise OpenAIRequestError(f"{name} must be a finite number") from exc
    if not math.isfinite(normalized):
        raise OpenAIRequestError(f"{name} must be a finite number")
    return normalized


def _temperature(value: object) -> float:
    normalized = _finite_number(value, "temperature")
    if normalized < 0:
        raise OpenAIRequestError("temperature must be non-negative")
    return normalized


def _top_p(value: object) -> float:
    normalized = _finite_number(value, "top_p")
    if not 0 < normalized <= 1:
        raise OpenAIRequestError("top_p must be greater than zero and at most one")
    return normalized


def _messages(value: object) -> list[dict[str, Any]]:
    if type(value) is not list or not value:
        raise OpenAIRequestError("messages must be a nonempty array of objects")
    detached = detached_messages(
        value,
        sequence_types=list,
        sequence_error="messages must be a nonempty array of objects",
        error_type=OpenAIRequestError,
    )
    try:
        normalize_messages(
            detached,
            sequence_types=list,
            sequence_error="messages must be a nonempty array of objects",
            error_type=OpenAIRequestError,
            max_source_chars=MAX_SOURCE_CHARS,
        )
    except OpenAIRequestError as exc:
        if str(exc) == "image source exceeds the per-image encoded-size limit":
            raise OpenAIRequestError(
                f"image source exceeds the {MAX_COMPRESSED_BYTES}-byte limit"
            ) from exc
        raise
    return detached


def _stop_values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list):
        raise OpenAIRequestError("stop must be a string or an array of strings")
    if any(not isinstance(item, str) or not item for item in values):
        raise OpenAIRequestError("stop values must be non-empty strings")
    return tuple(values)


def _chat_template_kwargs(value: object) -> dict[str, Any]:
    if value is None:
        caller: dict[str, Any] = {}
    elif type(value) is dict:
        caller = dict(value)
    else:
        raise OpenAIRequestError("chat_template_kwargs must be an object")
    try:
        caller = json.loads(json.dumps(caller, allow_nan=False))
        _require_finite_json(caller)
    except (TypeError, ValueError) as exc:
        raise OpenAIRequestError(
            "chat_template_kwargs must contain only finite json values"
        ) from exc
    has_enable_thinking = "enable_thinking" in caller
    enable_thinking = caller.get("enable_thinking")
    if has_enable_thinking and type(enable_thinking) is not bool:
        raise OpenAIRequestError("chat_template_kwargs.enable_thinking must be a boolean")
    normalized = {
        key: nested for key, nested in caller.items() if key not in _RESERVED_CHAT_TEMPLATE_KWARGS
    }
    if has_enable_thinking:
        normalized["enable_thinking"] = enable_thinking
    return normalized


def _require_finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite json number")
    if isinstance(value, list):
        for nested in value:
            _require_finite_json(nested)
    elif isinstance(value, dict):
        for nested in value.values():
            _require_finite_json(nested)


def _structured_outputs(payload: dict[str, Any]) -> dict[str, Any] | None:
    structured = payload.get("structured_outputs")
    response_format = payload.get("response_format")
    if structured is not None and response_format is not None:
        raise OpenAIRequestError("structured_outputs and response_format cannot both be set")
    value = structured if structured is not None else _response_format(response_format)
    try:
        return normalize_structured_outputs(
            value,
            error_type=OpenAIRequestError,
            validate_decoded_dicts=True,
        )
    except OpenAIRequestError as exc:
        raise OpenAIRequestError(f"invalid structured output: {exc}") from exc


def _response_format(value: object) -> object:
    if value is None:
        return None
    if type(value) is not dict:
        raise OpenAIRequestError("response_format must be an object")
    kind = value.get("type")
    if kind == "text" and set(value) == {"type"}:
        return {}
    if kind == "json_object" and set(value) == {"type"}:
        return {"json_object": True}
    if kind != "json_schema" or set(value) != {"type", "json_schema"}:
        raise OpenAIRequestError("response_format type is not supported")
    declaration = value.get("json_schema")
    if type(declaration) is not dict:
        raise OpenAIRequestError("response_format.json_schema must be an object")
    allowed = {"name", "description", "schema", "strict"}
    if set(declaration) - allowed or "schema" not in declaration:
        raise OpenAIRequestError("response_format.json_schema is malformed")
    if type(declaration["schema"]) is not dict:
        raise OpenAIRequestError("response_format.json_schema.schema must be an object")
    strict = declaration.get("strict", True)
    if type(strict) is not bool:
        raise OpenAIRequestError("response_format.json_schema.strict must be a boolean")
    if not strict:
        raise OpenAIRequestError("response_format.json_schema.strict=false is not supported")
    return {"json": declaration["schema"]}


def _tool_controls(
    payload: dict[str, Any],
) -> tuple[tuple[FunctionTool, ...] | None, str | None, bool | None]:
    has_tools = "tools" in payload
    has_choice = "tool_choice" in payload
    has_parallel = "parallel_tool_calls" in payload
    if not has_tools:
        if has_choice or has_parallel:
            raise OpenAIRequestError("tool_choice and parallel_tool_calls require tools")
        return None, None, None
    tools = normalize_tools(payload["tools"], error_type=OpenAIRequestError)
    choice = payload.get("tool_choice", "auto")
    if type(choice) is not str or choice not in {"auto", "none"}:
        raise OpenAIRequestError("tool_choice must be auto or none")
    parallel = payload.get("parallel_tool_calls", True)
    if parallel is not True:
        raise OpenAIRequestError("parallel_tool_calls must be true")
    return tools, choice, True


def parse_stream_options(value: object, stream: bool) -> bool:
    """validate stream_options and return whether usage chunks were requested."""

    options = _stream_options(value, stream)
    return bool(options and options["include_usage"])


def _stream_options(value: object, stream: bool) -> dict[str, bool] | None:
    if value is None:
        return None
    if not stream:
        raise OpenAIRequestError("stream_options requires stream=true")
    if type(value) is not dict:
        raise OpenAIRequestError("stream_options must be an object")
    if set(value) != {"include_usage"} or type(value["include_usage"]) is not bool:
        raise OpenAIRequestError("stream_options accepts only boolean include_usage")
    return {"include_usage": value["include_usage"]}
