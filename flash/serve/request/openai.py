"""canonical request grammar for the supported openai chat subset."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, cast

from flash.serve.request.validation import (
    MAX_COMPRESSED_BYTES,
    MAX_SOURCE_CHARS,
    normalize_messages,
    normalize_structured_outputs,
)

DEFAULT_MAX_TOKENS = 1024
_ALLOWED_REQUEST_KEYS = frozenset(
    {
        "adapter_revision",
        "chat_template_kwargs",
        "max_tokens",
        "messages",
        "model",
        "response_format",
        "step",
        "stop",
        "stream",
        "stream_options",
        "structured_outputs",
        "temperature",
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
    stop: tuple[str, ...]
    chat_template_kwargs: dict[str, Any]
    structured_outputs: dict[str, Any] | None
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
        allowed = allowed - {"adapter_revision", "step"}
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

    return NormalizedChatRequest(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=_stop_values(payload.get("stop")),
        chat_template_kwargs=_chat_template_kwargs(payload.get("chat_template_kwargs")),
        structured_outputs=_structured_outputs(payload),
        stream=stream,
        stream_options=_stream_options(payload.get("stream_options"), stream),
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
    try:
        normalize_messages(
            value,
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
    return [dict(message) for message in value]


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
    return {
        key: nested for key, nested in caller.items() if key not in _RESERVED_CHAT_TEMPLATE_KWARGS
    }


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
