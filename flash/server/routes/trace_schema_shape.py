"""What counts as a JSON Schema declaration, and where a payload may declare one.

Split out of `trace_redaction` so the vocabulary -- the keyword sets, the wrapper and host names --
sits beside the predicates that read it, and both can be reviewed without the traversal around them.

Everything here answers one question: is this node a schema DECLARATION, or is it instance data that
merely resembles one? The answer decides whether a secret-named member's literals are a property
definition worth keeping or a credential that must not be persisted, so each predicate is written to
demand positive evidence -- a declaration is recognized by its shape AND its name, never by either
alone, because ordinary request metadata is shaped like a schema often enough to matter.
"""

from __future__ import annotations

from typing import Any

_JSON_SCHEMA_STRUCTURAL_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "items",
        "prefixItems",
        "additionalItems",
        "contains",
        "enum",
        "const",
        "$ref",
        "$id",
        "$schema",
        "$anchor",
        "$dynamicRef",
        "$dynamicAnchor",
        "$recursiveRef",
        "$recursiveAnchor",
        "anyOf",
        "allOf",
        "oneOf",
        "not",
        "format",
        "additionalProperties",
        "patternProperties",
        "propertyNames",
        "unevaluatedProperties",
        "unevaluatedItems",
        "dependentRequired",
        "dependentSchemas",
        "discriminator",
        "required",
        "$defs",
        "definitions",
        "if",
        "then",
        "else",
        "contentSchema",
    }
)
_JSON_SCHEMA_ANNOTATION_KEYWORDS = frozenset(
    {
        "description",
        "title",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minProperties",
        "maxProperties",
        "minContains",
        "maxContains",
        "contentEncoding",
        "contentMediaType",
        "nullable",
        "example",
        "$comment",
    }
)
_JSON_SCHEMA_KEYWORDS = _JSON_SCHEMA_STRUCTURAL_KEYWORDS | _JSON_SCHEMA_ANNOTATION_KEYWORDS
_JSON_SCHEMA_SECRET_LITERAL_KEYWORDS = frozenset(
    {"default", "const", "enum", "examples", "example"}
)


def _is_secret_literal_keyword(key: Any) -> bool:
    """Whether `key` carries INSTANCE DATA that a secret schema must not keep.

    The standard names above are the usual spelling, but tooling writes the same thing as an
    OpenAPI-style vendor extension: `{"password": {"type": "string", "x-example": "..."}}` states a
    sample credential exactly as `example` does. Recognizing only the standard set meant the
    extension form survived storage and raw export while the plain one was redacted.

    Only an extension OF a literal keyword qualifies. `x-internal-note` is an unknown keyword inside
    a declared schema, which the traversal deliberately preserves, and widening this to every `x-`
    key would delete legitimate schema content.
    """
    if not isinstance(key, str):
        return False
    if key in _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS:
        return True
    extension = key.removeprefix("x-").removeprefix("X-")
    return extension != key and extension in _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS


_JSON_SCHEMA_PROPERTY_MAP_KEYWORDS = frozenset(
    {"properties", "patternProperties", "dependentSchemas", "$defs", "definitions"}
)
# keyed by property name like the maps above, but its VALUES are arrays of declared property names
# rather than subschemas. both halves are declarations: `{"password": ["username"]}` names two
# fields, so redacting either corrupts the stored schema instead of protecting a credential.
_JSON_SCHEMA_PROPERTY_NAME_MAP_KEYWORDS = frozenset({"dependentRequired"})
# tool output can arrive as content PARTS instead of one string. a part's `text` is still the tool's
# output, so it needs the same parsed-or-conservative handling the scalar form gets.
_TEXT_CONTENT_PART_TYPES = frozenset({"text", "output_text", "input_text"})
_JSON_SCHEMA_VALUE_KEYWORDS = frozenset(
    {
        "items",
        "additionalItems",
        "contains",
        "not",
        "additionalProperties",
        "propertyNames",
        "unevaluatedProperties",
        "unevaluatedItems",
        "if",
        "then",
        "else",
        "contentSchema",
        "prefixItems",
        "anyOf",
        "allOf",
        "oneOf",
    }
)
_JSON_SCHEMA_WRAPPER_KEYS = frozenset({"schema", "parameters", "input_schema", "output_schema"})
# containers that genuinely declare schemas. a wrapper key is honoured only beneath one of these,
# so ordinary request metadata that happens to be shaped like a schema cannot claim the exemption.
# only these names are a request's OWN top-level declaration. the rest of the host vocabulary is
# spelled only INSIDE one of them, so accepting those at the root let an unrelated top-level
# extension named `function` or `json_schema` open a host for its whole subtree.
# `tool_choice` and a root `function_call` SELECT an already-declared function by name; neither
# carries a schema. Listing them here let `{"tool_choice": {"function": {"parameters": ...}}}` open
# a host, and the nested wrapper then read `password` as a schema property and kept its unknown
# `value` verbatim -- a credential in a selector, which the tool declaration itself never contains.
_ROOT_SCHEMA_HOST_KEYS = frozenset({"tools", "functions", "response_format", "text_format"})
# a declaration is its NAME and its SHAPE. `tools` and `functions` are arrays of tool definitions;
# the rest are single objects. accepting the name alone let `{"tools": {"parameters": {...}}}` --
# which no provider would accept -- open a host whose nested wrapper then kept a secret property's
# literal, so an upstream-rejected request still persisted a third-party credential.
_ARRAY_SHAPED_ROOT_HOST_KEYS = frozenset({"tools", "functions"})
# these two name the SAME declaration in the chat-completions and responses spellings, and both
# state a discriminator: a response format only declares a schema as `{"type": "json_schema",
# "json_schema": {...}}`. accepting any object under the name let a direct `schema` wrapper the
# provider rejects open the exemption and keep an unknown keyword's credential literal.
_SCHEMA_DECLARING_FORMAT_KEYS = frozenset({"response_format", "text_format"})
# keywords whose KEYS are property names, so a secret-looking key declares a secret member and any
# reference under it must be followed. `dependencies` is the draft-07 spelling of
# `dependentSchemas`; its array form lists property names instead and carries no reference, which
# the `_is_schema_definition` test below already separates.
_SECRET_DECLARING_PROPERTY_MAPS = (
    "properties",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
)
_NESTED_SCHEMA_HOST_KEYS = frozenset({"function", "json_schema"})
_SCHEMA_HOST_KEYS = _ROOT_SCHEMA_HOST_KEYS | _NESTED_SCHEMA_HOST_KEYS
_JSON_SCHEMA_TYPES = frozenset(
    {"null", "boolean", "object", "array", "number", "string", "integer"}
)


def _is_schema_definition(value: Any, *, allow_custom_vocabulary: bool = False) -> bool:
    if isinstance(value, bool):
        return True
    if not isinstance(value, dict):
        return False
    if not value:
        # `{}` is the permissive JSON Schema ("any value"), so under a `properties` map it is a
        # declaration, not a secret. Treating it as one rewrote `{"password": {}}` into the string
        # "[redacted]" and turned a valid schema into an invalid one.
        return True
    keys = [key for key in value if not (isinstance(key, str) and key.startswith("x-"))]
    if allow_custom_vocabulary and any(key in _JSON_SCHEMA_KEYWORDS for key in keys):
        return True
    if any(key in _JSON_SCHEMA_STRUCTURAL_KEYWORDS for key in keys):
        return True
    return bool(keys) and all(key in _JSON_SCHEMA_KEYWORDS for key in keys)


def _is_property_name_list(value: Any) -> bool:
    """Whether `value` is a draft-07 `dependencies` array: a list of declared property names.

    These are schema declarations, not instance data, so a secret-looking entry is a property name
    and replacing the list with `"[redacted]"` corrupts the stored schema.
    """
    return isinstance(value, list) and all(isinstance(entry, str) for entry in value)


def _is_schema_map_keyword(key: str, item: Any) -> bool:
    """Whether `key`'s entries are keyed by PROPERTY NAME rather than being instance data.

    Under such a keyword a secret-looking key is a declared property name, not a credential, so
    replacing the entry with `"[redacted]"` corrupts the stored schema instead of protecting
    anything. Draft-07 `dependencies` qualifies in both of its forms: a schema value states a
    conditional subschema and an array value lists required property names, and neither is a secret.
    Only its entry VALUES differ, which the ordinary schema traversal already distinguishes.
    """
    return (
        key in _JSON_SCHEMA_PROPERTY_MAP_KEYWORDS
        or key in _JSON_SCHEMA_PROPERTY_NAME_MAP_KEYWORDS
        or (key == "dependencies" and isinstance(item, dict))
    )


def _has_schema_context(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if isinstance(value.get("$schema"), str) or isinstance(value.get("$id"), str):
        return True
    property_maps = [
        value[keyword]
        for keyword in _JSON_SCHEMA_PROPERTY_MAP_KEYWORDS
        if isinstance(value.get(keyword), dict)
    ]
    property_maps_are_schemas = all(
        all(_is_unambiguous_schema_definition(item) for item in property_map.values())
        for property_map in property_maps
    )
    schema_type = value.get("type")
    if (
        isinstance(schema_type, str)
        and schema_type in _JSON_SCHEMA_TYPES
        and property_maps_are_schemas
    ):
        return True
    if (
        isinstance(schema_type, list)
        and schema_type
        and all(isinstance(item, str) and item in _JSON_SCHEMA_TYPES for item in schema_type)
        and property_maps_are_schemas
    ):
        return True
    return bool(property_maps) and property_maps_are_schemas


def _is_unambiguous_schema_definition(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if not isinstance(value, dict):
        return False
    keys = [key for key in value if not (isinstance(key, str) and key.startswith("x-"))]
    if not keys:
        return True
    if any(key not in _JSON_SCHEMA_KEYWORDS for key in keys):
        return False
    schema_type = value.get("type")
    if schema_type is None:
        return True
    if isinstance(schema_type, str):
        return schema_type in _JSON_SCHEMA_TYPES
    return (
        isinstance(schema_type, list)
        and bool(schema_type)
        and all(isinstance(item, str) and item in _JSON_SCHEMA_TYPES for item in schema_type)
    )


def _declares_applicators_its_type_forbids(value: Any) -> bool:
    """Whether a declared scalar `type` contradicts the applicators declared beside it.

    `properties` applies only to objects and `items` only to arrays, so a schema declaring exactly
    `type: "string"` alongside either is self-contradictory. Real schemas do not do this; ordinary
    metadata that merely happens to carry both keys does.
    """
    if not isinstance(value, dict):
        return False
    schema_type = value.get("type")
    types = (
        {schema_type}
        if isinstance(schema_type, str)
        else set(schema_type)
        if isinstance(schema_type, list)
        else set()
    )
    if not types or not types <= _JSON_SCHEMA_TYPES:
        return False
    object_only = any(
        isinstance(value.get(key), dict) for key in ("properties", "patternProperties")
    )
    array_only = "items" in value or "prefixItems" in value
    return (object_only and "object" not in types) or (array_only and "array" not in types)


def _has_schema_wrapper_evidence(value: Any) -> bool:
    if _has_schema_context(value):
        return True
    if not isinstance(value, dict):
        return False
    schema_type = value.get("type")
    if _declares_applicators_its_type_forbids(value):
        # a schema is not merely a dict with a valid `type`: the type has to agree with the
        # applicators alongside it. `{"type": "string", "properties": {...}}` describes a string
        # that somehow has properties, which no real schema does, and treating it as one let
        # ordinary metadata claim the exemption and keep its literals.
        return False
    if isinstance(schema_type, str):
        return schema_type in _JSON_SCHEMA_TYPES
    return (
        isinstance(schema_type, list)
        and bool(schema_type)
        and all(isinstance(item, str) and item in _JSON_SCHEMA_TYPES for item in schema_type)
    )
