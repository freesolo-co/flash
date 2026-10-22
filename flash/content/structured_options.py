"""honor supported JSON options before vllm loses request-local option fields."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from typing import Any

_JSON_OPTIONS = ("disable_any_whitespace", "disable_additional_properties")
_SCHEMA_MAPS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")
_SCHEMA_VALUES = (
    "items",
    "additionalItems",
    "additionalProperties",
    "unevaluatedItems",
    "unevaluatedProperties",
    "contains",
    "propertyNames",
    "not",
    "if",
    "then",
    "else",
)
_SCHEMA_LISTS = ("allOf", "anyOf", "oneOf", "prefixItems")


def validate_structured_options(spec: dict[str, Any]) -> None:
    if spec.get("whitespace_pattern") is not None:
        raise ValueError(
            "whitespace_pattern is not supported by the managed decoding backend; "
            "use disable_any_whitespace=true for compact JSON"
        )
    if not ("json" in spec or spec.get("json_object")) and any(
        spec.get(key) for key in _JSON_OPTIONS
    ):
        raise ValueError("structured output whitespace/property options require a JSON constraint")


def vllm_structured_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    """lower options into the constraint itself, without mutating the public spec.

    vllm 0.23 compiles JSON using engine-global options, ignoring the same fields
    in StructuredOutputsParams. an explicit grammar carries compact formatting
    per request and preserves spaces *inside* JSON strings.
    """
    validate_structured_options(spec)
    kwargs = {key: value for key, value in spec.items() if key not in _JSON_OPTIONS}
    kwargs.pop("whitespace_pattern", None)
    compact = bool(spec.get("disable_any_whitespace"))
    closed = bool(spec.get("disable_additional_properties"))
    if not (compact or closed):
        return kwargs
    schema = copy.deepcopy(spec.get("json", {"type": "object", "additionalProperties": True}))
    if isinstance(schema, str):
        schema = json.loads(schema)
    if closed:
        _close_objects(schema)
    if compact:
        canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        return {"grammar": _compact_json_grammar(canonical)}
    return {"json": schema}


@lru_cache(maxsize=32)
def _compact_json_grammar(schema: str) -> str:
    import xgrammar

    try:
        return str(
            xgrammar.Grammar.from_json_schema(
                schema, any_whitespace=False, indent=None, separators=(",", ":")
            )
        )
    except RuntimeError as exc:
        raise ValueError(f"invalid compact JSON constraint: {exc}") from exc


def _close_objects(schema: Any) -> None:
    if not isinstance(schema, dict):
        return
    kind = schema.get("type")
    if (
        kind == "object"
        or (isinstance(kind, list) and "object" in kind)
        or any(key in schema for key in ("properties", "patternProperties", "additionalProperties"))
    ):
        schema["additionalProperties"] = False
    # descend through schema vocabulary, never example/default/enum data or property names.
    for key in _SCHEMA_MAPS:
        children = schema.get(key)
        if isinstance(children, dict):
            for child in children.values():
                _close_objects(child)
    for key in _SCHEMA_VALUES:
        child = schema.get(key)
        if isinstance(child, list):
            for item in child:
                _close_objects(item)
        else:
            _close_objects(child)
    for key in _SCHEMA_LISTS:
        children = schema.get(key)
        if isinstance(children, list):
            for child in children:
                _close_objects(child)
