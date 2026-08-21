"""pure-python normalization for vllm structured-output constraints."""

from __future__ import annotations

import json
from typing import Any

from .errors import StructuredOutputsError

CONSTRAINT_KEYS = ("json", "regex", "choice", "json_object")
OPTION_KEYS = ("disable_any_whitespace", "disable_additional_properties", "whitespace_pattern")
_REMOVED_KEYS = frozenset({"grammar", "structural_tag"})
_CONSTRAINT_ALIASES = {"json_schema": "json", "schema": "json", "choices": "choice"}
_OFF_STRINGS = frozenset({"", "none", "text"})
_JSON_OBJECT_STRINGS = frozenset({"json", "json_object"})
_ALLOWED_KEYS_HINT = (
    f"allowed keys: {', '.join(CONSTRAINT_KEYS)} "
    f"(aliases: {', '.join(sorted(_CONSTRAINT_ALIASES))}), "
    f"options: {', '.join(OPTION_KEYS)}"
)


def normalize_structured_outputs(value: Any) -> dict[str, Any] | None:
    """return canonical structured-output kwargs, an explicit-off dict, or none."""
    if value is None:
        return None
    if value is False:
        return {}
    if value is True:
        raise StructuredOutputsError(
            "structured outputs cannot be true; pass a constraint such as "
            f'{{"json": <schema>}}; {_ALLOWED_KEYS_HINT}'
        )
    if isinstance(value, str):
        return _normalize_string(value)
    if isinstance(value, dict):
        return _normalize_dict(value)
    raise StructuredOutputsError(
        "structured outputs must be a dict, a json string, or an off marker "
        f"(null/false/'none'/'text'), got {type(value).__name__}"
    )


def _decode_json(value: str) -> Any:
    """decode an encoded constraint, refusing the constants json does not define.

    A constraint may arrive as a *string* of json, which is itself valid outer json -- so the http
    boundary's strict parser sees a plain string and waves it through, and this inner decode is the
    only place left that can reject it. Without the same `parse_constant` policy, `NaN` / `Infinity`
    reached vllm's grammar compiler from a request the outer guard had already approved.
    """

    return json.loads(value, parse_constant=_reject_non_finite)


def _reject_non_finite(constant: str) -> Any:
    raise ValueError(f"json does not define {constant}")


def _normalize_string(value: str) -> dict[str, Any]:
    keyword = value.strip().lower()
    if keyword in _OFF_STRINGS:
        return {}
    if keyword in _JSON_OBJECT_STRINGS:
        return {"json_object": True}
    try:
        parsed = _decode_json(value)
    except ValueError as exc:
        raise StructuredOutputsError(
            "structured outputs string must be 'json'/'json_object', an off marker, or valid json: "
            f"{exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise StructuredOutputsError(
            f"structured outputs json string must decode to an object, got {type(parsed).__name__}"
        )
    return _normalize_dict(parsed)


def _normalize_dict(value: dict[str, Any]) -> dict[str, Any]:
    data = {key: item for key, item in value.items() if item is not None}
    if not data:
        return {}
    removed = sorted(key for key in data if key in _REMOVED_KEYS)
    if removed:
        raise StructuredOutputsError(
            f"structured outputs {', '.join(removed)} constraint(s) are not supported; "
            f"use one of {', '.join(CONSTRAINT_KEYS)}"
        )
    known = set(CONSTRAINT_KEYS) | set(_CONSTRAINT_ALIASES) | set(OPTION_KEYS)
    if any(key in known for key in data):
        return _normalize_canonical(data)
    return {"json": value}


def _normalize_canonical(data: dict[str, Any]) -> dict[str, Any]:
    constraints: dict[str, Any] = {}
    options: dict[str, Any] = {}
    for key, raw in data.items():
        canonical = _CONSTRAINT_ALIASES.get(key, key)
        if canonical in CONSTRAINT_KEYS:
            if canonical in constraints:
                raise StructuredOutputsError(
                    f"structured outputs sets {canonical!r} twice via aliases; {_ALLOWED_KEYS_HINT}"
                )
            constraints[canonical] = _validate_constraint(canonical, raw)
        elif canonical in OPTION_KEYS:
            options[canonical] = _validate_option(canonical, raw)
        else:
            raise StructuredOutputsError(
                f"unknown structured outputs key {key!r}; {_ALLOWED_KEYS_HINT}"
            )
    if len(constraints) != 1:
        found = ", ".join(sorted(constraints)) or "none"
        raise StructuredOutputsError(
            "structured outputs must set exactly one constraint of "
            f"{', '.join(CONSTRAINT_KEYS)}; got {found}"
        )
    return {**constraints, **{key: options[key] for key in OPTION_KEYS if key in options}}


def _validate_constraint(key: str, value: Any) -> Any:
    if key == "json":
        return _coerce_json_schema(value)
    if key == "regex":
        if not isinstance(value, str) or not value.strip():
            raise StructuredOutputsError("structured outputs 'regex' must be a non-empty string")
        return value
    if key == "choice":
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list) or not value:
            raise StructuredOutputsError(
                f"structured outputs 'choice' must be a non-empty list of strings, got {value!r}"
            )
        if not all(isinstance(item, str) for item in value):
            raise StructuredOutputsError(
                f"structured outputs 'choice' entries must all be strings, got {value!r}"
            )
        return value
    if value is not True:
        raise StructuredOutputsError(
            "structured outputs 'json_object' must be true; use null, false, or 'none' to disable"
        )
    return True


def _validate_option(key: str, value: Any) -> Any:
    if key == "whitespace_pattern":
        if not isinstance(value, str) or not value:
            raise StructuredOutputsError(
                "structured outputs option 'whitespace_pattern' must be a non-empty string"
            )
        return value
    if not isinstance(value, bool):
        raise StructuredOutputsError(
            f"structured outputs option {key!r} must be a boolean, got {type(value).__name__}"
        )
    return value


def _coerce_json_schema(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = _decode_json(value)
        except ValueError as exc:
            raise StructuredOutputsError(
                f"structured outputs 'json' string is not valid json: {exc}"
            ) from exc
        if isinstance(parsed, dict):
            return parsed
        raise StructuredOutputsError(
            "structured outputs 'json' string must decode to a json schema object"
        )
    raise StructuredOutputsError(
        "structured outputs 'json' must be a json schema object or a json string of one"
    )
