"""honor supported JSON options before vllm loses request-local option fields."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

_JSON_OPTIONS = ("disable_any_whitespace", "disable_additional_properties")
_SCHEMA_MAPS = (
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
    "dependencies",
)
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
_OBJECT_KEYS = (
    "properties",
    "patternProperties",
    "additionalProperties",
    "unevaluatedProperties",
    "required",
    "minProperties",
    "maxProperties",
    "propertyNames",
    "dependentRequired",
    "dependentSchemas",
    "dependencies",
)
_COMPOSED_OBJECT_KEYS = (
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "dependentSchemas",
    "dependencies",
)


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
    if spec.get("disable_additional_properties"):
        schema = spec.get("json", {})
        if isinstance(schema, str):
            schema = json.loads(schema)
        for node in _schema_nodes(schema):
            unsupported = next((key for key in _COMPOSED_OBJECT_KEYS if key in node), None)
            if unsupported is None and _is_object_schema(node):
                unsupported = next((key for key in ("$ref", "$dynamicRef") if key in node), None)
            if unsupported is not None:
                raise ValueError(
                    "disable_additional_properties cannot safely rewrite schemas containing "
                    f"{unsupported}; set additionalProperties explicitly in the schema "
                    "and omit disable_additional_properties"
                )


def vllm_structured_kwargs(spec: dict[str, Any]) -> dict[str, Any]:
    """lower options into the constraint itself, without mutating the public spec.

    vllm 0.23 compiles JSON using engine-global options, ignoring the same fields
    in StructuredOutputsParams. an explicit grammar carries compact formatting
    per request and preserves spaces *inside* JSON strings.
    """
    spec = _plain_json_containers(spec)
    validate_structured_options(spec)
    kwargs = {key: value for key, value in spec.items() if key not in _JSON_OPTIONS}
    kwargs.pop("whitespace_pattern", None)
    compact = bool(spec.get("disable_any_whitespace"))
    closed = bool(spec.get("disable_additional_properties"))
    limited = "json" in spec and _has_length_limited_string(_loaded(spec["json"]))
    if not (compact or closed or limited):
        return kwargs
    schema = copy.deepcopy(spec.get("json", {"type": "object", "additionalProperties": True}))
    if isinstance(schema, str):
        schema = json.loads(schema)
    if closed:
        _close_objects(schema)
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    if compact:
        return {"grammar": _compact_json_grammar(canonical)}
    if limited:
        grammar = _escape_safe_json_grammar(canonical)
        if grammar is not None:
            return {"grammar": grammar}
    return {"json": schema}


# xgrammar (through 0.2.9) compiles a string with minLength/maxLength as a repetition of this
# character class, which has no escape alternative: \" \\ \n \r \t \uXXXX become undecodable, so a
# model that must quote or break a line inside such a field silently drops the quote or ends the
# string. Each rewritten repetition unit accepts one escape as one character, so the length bounds
# keep their JSON meaning (decoded characters).
_BARE_STRING_CHARACTER = r"(([^\"\\\r\n]))"
_ESCAPED_STRING_CHARACTER = r'(([^\"\\\r\n]) | ("\\" flash_json_escape))'
_JSON_ESCAPE_RULE = r'flash_json_escape ::= (([\"\\/bfnrt]) | ("u" [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9] [A-Fa-f0-9]))'


def _allow_escapes_in_limited_strings(grammar: str) -> str:
    lines = grammar.splitlines()
    patched = False
    for index, line in enumerate(lines):
        name, separator, body = line.partition(" ::= ")
        if separator and body == _BARE_STRING_CHARACTER:
            lines[index] = f"{name}{separator}{_ESCAPED_STRING_CHARACTER}"
            patched = True
    if not patched:
        return grammar
    return "\n".join([*lines, _JSON_ESCAPE_RULE]) + "\n"


def _loaded(schema: Any) -> Any:
    return json.loads(schema) if isinstance(schema, str) else schema


def _has_length_limited_string(schema: Any) -> bool:
    for node in _schema_nodes(schema):
        kind = node.get("type")
        is_string = kind == "string" or (isinstance(kind, list) and "string" in kind)
        if is_string and ("minLength" in node or "maxLength" in node):
            return True
    return False


def _xgrammar_unsupported(schema: Any) -> bool:
    from vllm.v1.structured_output.backend_xgrammar import (
        has_xgrammar_unsupported_json_features,
    )

    return any(
        has_xgrammar_unsupported_json_features(_node_features(node))
        for node in _schema_nodes(schema)
    )


@lru_cache(maxsize=32)
def _escape_safe_json_grammar(schema: str) -> str | None:
    """vLLM's default JSON compilation (any whitespace), with escapes allowed in limited strings.

    None leaves the schema to vLLM's own backend selection (features xgrammar cannot compile).
    """
    if _xgrammar_unsupported(json.loads(schema)):
        return None
    import xgrammar

    try:
        grammar = str(xgrammar.Grammar.from_json_schema(schema, any_whitespace=True))
    except RuntimeError:
        return None
    return _allow_escapes_in_limited_strings(grammar)


@lru_cache(maxsize=32)
def _compact_json_grammar(schema: str) -> str:
    if _xgrammar_unsupported(json.loads(schema)):
        raise ValueError(
            "disable_any_whitespace requires an xgrammar-compatible JSON schema; "
            "omit disable_any_whitespace to allow backend selection"
        )
    import xgrammar

    try:
        grammar = str(
            xgrammar.Grammar.from_json_schema(
                schema, any_whitespace=False, indent=None, separators=(",", ":")
            )
        )
    except RuntimeError as exc:
        raise ValueError(f"invalid compact JSON constraint: {exc}") from exc
    return _allow_escapes_in_limited_strings(grammar)


def _close_objects(schema: Any) -> None:
    for node in _schema_nodes(schema):
        if _is_object_schema(node):
            node["additionalProperties"] = False


def _is_object_schema(node: dict[str, Any]) -> bool:
    kind = node.get("type")
    return (
        kind == "object"
        or (isinstance(kind, list) and "object" in kind)
        or any(key in node for key in _OBJECT_KEYS)
    )


def _node_features(node: dict[str, Any]) -> dict[str, Any]:
    # preserve keyword presence and scalar values without exposing child objects
    # to the backend's recursive scan. schema children are checked by our walker.
    features = {}
    for key, value in node.items():
        if isinstance(value, dict):
            value = {}
        elif isinstance(value, list):
            value = [item for item in value if not isinstance(item, dict | list)]
        features[key] = value
    return features


def _schema_nodes(schema: Any):
    if not isinstance(schema, dict):
        return
    yield schema
    # descend through schema vocabulary, never example/default/enum data or property names.
    for key in _SCHEMA_MAPS:
        children = schema.get(key)
        if isinstance(children, dict):
            for child in children.values():
                yield from _schema_nodes(child)
    for key in _SCHEMA_VALUES:
        child = schema.get(key)
        if isinstance(child, list):
            for item in child:
                yield from _schema_nodes(item)
        else:
            yield from _schema_nodes(child)
    for key in _SCHEMA_LISTS:
        children = schema.get(key)
        if isinstance(children, list):
            for child in children:
                yield from _schema_nodes(child)


def _plain_json_containers(value: Any) -> Any:
    # hydra wraps nested rollout configuration in DictConfig/ListConfig. vllm/xgrammar
    # require plain JSON containers; a shallow dict() leaves schemas unserializable.
    if isinstance(value, Mapping):
        return {key: _plain_json_containers(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_plain_json_containers(item) for item in value]
    return value
