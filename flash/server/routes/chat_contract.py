"""strict managed chat request normalization and immutable provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from flash.schema import format_checkpoint_ref, parse_adapter_revision
from flash.serve.request_validation import normalize_structured_outputs
from flash.serve.runtime.types import (
    RuntimeConfigurationError,
    validate_generation_max_tokens,
    validate_generation_temperature,
    validate_generation_top_p,
)

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
        "max_length",
        "padding",
        "return_assistant_tokens_mask",
        "return_dict",
        "return_tensors",
        "tokenize",
        "truncation",
    }
)


@dataclass(frozen=True, slots=True)
class ManagedChatRequest:
    """one normalized request accepted by the managed run chat route."""

    temperature: float
    max_tokens: int
    top_p: float
    stop: tuple[str, ...]
    chat_template_kwargs: dict[str, Any]
    structured_outputs: dict[str, Any] | None
    stream: bool
    stream_options: dict[str, bool] | None


def parse_managed_chat_request(payload: dict[str, Any], *, thinking: bool) -> ManagedChatRequest:
    """validate the managed subset without allowing undeclared openai fields."""

    unknown = sorted(set(payload) - _ALLOWED_REQUEST_KEYS)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported chat request field(s): {', '.join(unknown)}",
        )
    if "model" in payload:
        model = payload["model"]
        if type(model) is not str or not model.strip():
            raise HTTPException(status_code=400, detail="model must be a non-empty string")

    stream = payload.get("stream", False)
    if type(stream) is not bool:
        raise HTTPException(status_code=400, detail="stream must be a boolean")
    try:
        temperature = validate_generation_temperature(payload.get("temperature", 0.0))
        max_tokens = validate_generation_max_tokens(payload.get("max_tokens", 512))
        top_p = validate_generation_top_p(payload.get("top_p", 0.95))
    except RuntimeConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chat_template_kwargs = _chat_template_kwargs(payload.get("chat_template_kwargs"), thinking)
    structured_outputs = _structured_outputs(payload)
    stream_options = _stream_options(payload.get("stream_options"), stream)
    return ManagedChatRequest(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=_stop_values(payload.get("stop")),
        chat_template_kwargs=chat_template_kwargs,
        structured_outputs=structured_outputs,
        stream=stream,
        stream_options=stream_options,
    )


def combined_stop_sequences(
    mandatory: list[str] | tuple[str, ...], caller: tuple[str, ...]
) -> list[str] | None:
    """preserve mandatory and caller stops while deduplicating exact strings."""

    combined: list[str] = []
    seen: set[str] = set()
    for value in (*mandatory, *caller):
        if not isinstance(value, str) or not value or value in seen:
            continue
        seen.add(value)
        combined.append(value)
    return combined or None


def immutable_provenance(adapter_revision: str) -> dict[str, str]:
    """return the stable managed provenance envelope for one verified revision."""

    parsed = parse_adapter_revision(adapter_revision)
    if parsed is None:
        raise ValueError("managed chat target is not an immutable adapter revision")
    run_id, step, hf_revision = parsed
    return {
        "adapter_revision": adapter_revision,
        "checkpoint": format_checkpoint_ref(run_id, step),
        "hf_revision": hf_revision,
    }


def attach_immutable_provenance(
    payload: dict[str, Any], provenance: dict[str, str]
) -> dict[str, Any]:
    """retain backend-native metadata and expose one stable immutable envelope."""

    native = payload.get("freesolo")
    if native is not None and not isinstance(native, dict):
        raise ValueError("serving backend returned malformed immutable provenance")
    complete = False
    if isinstance(native, dict):
        _require_matching_provenance(native, provenance)
        complete = True
    packaged = payload.get("flash_provenance")
    if packaged is not None and not isinstance(packaged, dict):
        raise ValueError("serving backend returned malformed packaged provenance")
    if isinstance(packaged, dict):
        _require_matching_provenance(packaged, provenance, require_all=False)
        source_revision = packaged.get("source_revision")
        for field in ("adapter_revision", "checkpoint"):
            if packaged.get(field) is None:
                raise ValueError(f"serving backend omitted immutable provenance field {field}")
        if source_revision is None:
            raise ValueError("serving backend omitted immutable provenance field source_revision")
        if source_revision != provenance["hf_revision"]:
            raise ValueError("serving backend returned mismatched immutable source revision")
        complete = True
    if not complete:
        raise ValueError("serving backend omitted immutable provenance")
    return {**payload, "freesolo": {**(native or {}), **provenance}}


def immutable_provenance_headers(provenance: dict[str, str]) -> dict[str, str]:
    """render normalized immutable provenance without changing an sse body."""

    return {
        "X-Freesolo-Adapter-Revision": provenance["adapter_revision"],
        "X-Freesolo-Checkpoint": provenance["checkpoint"],
        "X-Freesolo-HF-Revision": provenance["hf_revision"],
    }


def _require_matching_provenance(
    native: dict[str, Any], expected: dict[str, str], *, require_all: bool = True
) -> None:
    for key, value in expected.items():
        if key not in native:
            if require_all:
                raise ValueError(f"serving backend omitted immutable provenance field {key}")
            continue
        if native[key] != value:
            raise ValueError(
                f"serving backend returned mismatched immutable provenance field {key}"
            )


def _stop_values(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list):
        raise HTTPException(status_code=400, detail="stop must be a string or an array of strings")
    if any(not isinstance(item, str) or not item for item in values):
        raise HTTPException(status_code=400, detail="stop values must be non-empty strings")
    return tuple(values)


def _chat_template_kwargs(value: object, thinking: bool) -> dict[str, Any]:
    if value is None:
        caller: dict[str, Any] = {}
    elif type(value) is dict:
        caller = dict(value)
    else:
        raise HTTPException(status_code=400, detail="chat_template_kwargs must be an object")
    try:
        caller = json.loads(json.dumps(caller, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="chat_template_kwargs must contain only finite json values",
        ) from exc
    safe = {
        key: nested
        for key, nested in caller.items()
        if key not in _RESERVED_CHAT_TEMPLATE_KWARGS and key != "enable_thinking"
    }
    safe["enable_thinking"] = thinking
    return safe


def _structured_outputs(payload: dict[str, Any]) -> dict[str, Any] | None:
    structured = payload.get("structured_outputs")
    response_format = payload.get("response_format")
    if structured is not None and response_format is not None:
        raise HTTPException(
            status_code=400,
            detail="structured_outputs and response_format cannot both be set",
        )
    value = structured if structured is not None else _response_format(response_format)
    try:
        return normalize_structured_outputs(
            value,
            error_type=ValueError,
            validate_decoded_dicts=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid structured output: {exc}") from exc


def _response_format(value: object) -> object:
    if value is None:
        return None
    if type(value) is not dict:
        raise HTTPException(status_code=400, detail="response_format must be an object")
    kind = value.get("type")
    if kind == "text" and set(value) == {"type"}:
        return {}
    if kind == "json_object" and set(value) == {"type"}:
        return {"json_object": True}
    if kind != "json_schema" or set(value) != {"type", "json_schema"}:
        raise HTTPException(status_code=400, detail="response_format type is not supported")
    declaration = value.get("json_schema")
    if type(declaration) is not dict:
        raise HTTPException(status_code=400, detail="response_format.json_schema must be an object")
    allowed = {"name", "description", "schema", "strict"}
    if set(declaration) - allowed or "schema" not in declaration:
        raise HTTPException(status_code=400, detail="response_format.json_schema is malformed")
    if type(declaration["schema"]) is not dict:
        raise HTTPException(
            status_code=400,
            detail="response_format.json_schema.schema must be an object",
        )
    strict = declaration.get("strict", True)
    if type(strict) is not bool:
        raise HTTPException(
            status_code=400,
            detail="response_format.json_schema.strict must be a boolean",
        )
    if not strict:
        raise HTTPException(
            status_code=400,
            detail="response_format.json_schema.strict=false is not supported",
        )
    return {"json": declaration["schema"]}


def _stream_options(value: object, stream: bool) -> dict[str, bool] | None:
    if value is None:
        return None
    if not stream:
        raise HTTPException(status_code=400, detail="stream_options requires stream=true")
    if type(value) is not dict or set(value) != {"include_usage"}:
        raise HTTPException(
            status_code=400,
            detail="stream_options accepts only include_usage",
        )
    if type(value["include_usage"]) is not bool:
        raise HTTPException(
            status_code=400, detail="stream_options.include_usage must be a boolean"
        )
    return {"include_usage": value["include_usage"]}
