"""Secret and JSON Schema redaction for recorded traces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from flash.server.platform import traces as platform_traces
from flash.server.routes.trace_schema_identity import (
    SchemaResourceScopes,
    _local_schema_pointer,
    _schema_resource_id,
)
from flash.server.routes.trace_secret_names import (  # noqa: F401
    _is_secret_key,
    _is_secret_property_pattern,
    _secret_key_candidates,
    _unwrap_pattern_groups,
)
from flash.server.routes.trace_uri import (
    _canonical_resource_uri,
    _safe_urldefrag,
    _safe_urljoin,
)

# short strings occur naturally in prompts and object keys. treating one as a global substring
# secret corrupts unrelated training text, while real bearer credentials are comfortably longer.
_MIN_SECRET_SUBSTRING_LENGTH = 16

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
_NESTED_SCHEMA_HOST_KEYS = frozenset({"function", "json_schema"})
_SCHEMA_HOST_KEYS = _ROOT_SCHEMA_HOST_KEYS | _NESTED_SCHEMA_HOST_KEYS
_JSON_SCHEMA_TYPES = frozenset(
    {"null", "boolean", "object", "array", "number", "string", "integer"}
)


@dataclass
class _SanitizationFlag:
    hit: bool = False


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


def _redact_schema_literal(value: Any, *, depth: int, flag: _SanitizationFlag | None = None) -> Any:
    if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
        if flag is not None:
            flag.hit = True
        return "[redacted]"
    if isinstance(value, dict):
        return {
            "[redacted]" if _is_secret_key(key) else key: _redact_schema_literal(
                item, depth=depth + 1, flag=flag
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact_schema_literal(item, depth=depth + 1, flag=flag) for item in value]
    return "[redacted]"


def _secret_schema_definition_refs(value: Any, *, depth: int = 0) -> set[tuple[str, ...]]:
    if not isinstance(value, dict):
        return set()
    refs: set[tuple[str, ...]] = set()
    scopes = SchemaResourceScopes.build(value, depth=depth)
    base_uri = scopes.base_uri
    resources = scopes.resources
    anchors = scopes.anchors
    dynamic_anchors = scopes.dynamic_anchors
    anchor_belongs_to_resource = scopes.anchor_belongs_to_resource

    def collect_refs(
        node: Any, node_depth: int, path: tuple[str, ...], scope_uri: str
    ) -> set[tuple[str, ...]]:
        if node_depth >= platform_traces._MAX_PAYLOAD_DEPTH:
            return set()
        found: set[tuple[str, ...]] = set()
        if isinstance(node, dict):
            resource_id = _schema_resource_id(node)
            if isinstance(resource_id, str):
                scope_uri = _safe_urljoin(scope_uri, resource_id)
            for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
                ref = node.get(keyword)
                if isinstance(ref, str):
                    # a dynamic reference leaves its own resource by design: the target is the
                    # outermost dynamic scope declaring the anchor, which depends on the evaluation
                    # entry point. restricting it to the reference's resource kept literals on the
                    # outer target exposed, so dynamic keywords consider every same-named anchor.
                    is_dynamic = keyword in {"$dynamicRef", "$recursiveRef"}
                    resolved_base, fragment = _safe_urldefrag(_safe_urljoin(scope_uri, ref))
                    canonical_base = _canonical_resource_uri(resolved_base)
                    resource_path = resources.get(canonical_base)
                    if resource_path is not None:
                        resource_ref = f"#{fragment}"
                        if keyword == "$recursiveRef" and resource_ref == "#":
                            # `$recursiveRef` resolves against the dynamic scope, so an ENCLOSING
                            # resource that also declares `$recursiveAnchor` can be the target. a
                            # sibling embedded resource cannot: it is not on this reference's
                            # evaluation path, so its anchor is never in scope. with no anchor
                            # reachable at all the target is this resource's own root.
                            pointers = frozenset(
                                pointer
                                for pointer in anchors.get("", frozenset())
                                if anchor_belongs_to_resource(pointer, canonical_base)
                                or path[: len(pointer)] == pointer
                            )
                            found.update(pointers or {resource_path})
                        elif is_dynamic and resource_ref != "#":
                            # union, not replacement: a plain-name fragment also names an ordinary
                            # `$anchor`, so dropping the static resolution would leave a same-named
                            # `$anchor` target unredacted.
                            found.update(_local_schema_pointer(resource_ref, dynamic_anchors))
                            found.update(
                                pointer
                                for pointer in _local_schema_pointer(resource_ref, anchors)
                                if anchor_belongs_to_resource(pointer, canonical_base)
                            )
                        else:
                            pointers = _local_schema_pointer(resource_ref, anchors)
                            if resource_ref == "#" or resource_ref.startswith("#/"):
                                found.update((*resource_path, *pointer) for pointer in pointers)
                            else:
                                found.update(
                                    pointer
                                    for pointer in pointers
                                    if anchor_belongs_to_resource(pointer, canonical_base)
                                )
                    else:
                        found.update(_local_schema_pointer(ref, anchors, base_uri=scope_uri))
            for key, item in node.items():
                if key not in _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS:
                    found.update(collect_refs(item, node_depth + 1, (*path, str(key)), scope_uri))
        elif isinstance(node, list | tuple):
            for index, item in enumerate(node):
                found.update(collect_refs(item, node_depth + 1, (*path, str(index)), scope_uri))
        return found

    def collect_secret_properties(
        node: Any, node_depth: int, path: tuple[str, ...], scope_uri: str
    ) -> None:
        if node_depth >= platform_traces._MAX_PAYLOAD_DEPTH:
            return
        if isinstance(node, dict):
            resource_id = _schema_resource_id(node)
            if isinstance(resource_id, str):
                scope_uri = _safe_urljoin(scope_uri, resource_id)
            # `patternProperties` declares secret-looking members too: `{"^secret_": {...}}` marks
            # every matching property secret. scanning only `properties` left a reference from such
            # an entry uncollected, so its target definition kept its credential-bearing literal.
            for map_keyword in ("properties", "patternProperties"):
                property_map = node.get(map_keyword)
                if not isinstance(property_map, dict):
                    continue
                for key, schema in property_map.items():
                    schema_path = (*path, map_keyword, str(key))
                    secret_declaration = (
                        _is_secret_property_pattern(key)
                        if map_keyword == "patternProperties"
                        else _is_secret_key(key)
                    )
                    if secret_declaration and _is_schema_definition(schema):
                        refs.update(collect_refs(schema, 0, schema_path, scope_uri))
                    collect_secret_properties(schema, node_depth + 1, schema_path, scope_uri)
            for key, item in node.items():
                if key not in {"properties", "patternProperties"}:
                    collect_secret_properties(item, node_depth + 1, (*path, str(key)), scope_uri)
        elif isinstance(node, list | tuple):
            for index, item in enumerate(node):
                collect_secret_properties(item, node_depth + 1, (*path, str(index)), scope_uri)

    def resolve(pointer: tuple[str, ...]) -> Any:
        target: Any = value
        for segment in pointer:
            if isinstance(target, dict) and segment in target:
                target = target[segment]
            elif isinstance(target, list | tuple) and segment.isdigit():
                index = int(segment)
                if index >= len(target):
                    return None
                target = target[index]
            else:
                return None
        return target

    collect_secret_properties(value, depth, (), base_uri)
    pending = list(refs)
    while pending:
        target_path = pending.pop()
        target = resolve(target_path)
        for pointer in collect_refs(target, 0, target_path, scopes.scope_for(target_path)):
            if pointer not in refs:
                refs.add(pointer)
                pending.append(pointer)
    return refs


def _looks_structured(value: str) -> bool:
    """Whether a string appears to carry JSON, whether or not it parses.

    Used to tell truncated structured output -- whose nested credentials cannot be inspected --
    from ordinary prose, which must survive verbatim.
    """
    stripped = value.strip()
    if stripped.startswith(("{", "[")):
        return True
    return '":' in stripped or '": ' in stripped


def _redact_tool_result_content(value: str, *, depth: int, flag: _SanitizationFlag | None) -> str:
    """Redact a tool message's `content`, which carries the tool's output rather than prose."""
    try:
        parsed = json.loads(value)
    except (ValueError, RecursionError):
        # a tool result is not required to be json -- prose is the ordinary case -- so unparseable
        # text is kept. but truncated or malformed structured output is common too, and its
        # credentials cannot be inspected, so anything that LOOKS structured is blanked instead:
        # keeping `{"password":"HUNTER2"` verbatim made a parse failure an exfiltration path.
        return "[redacted]" if _looks_structured(value) else value
    if not isinstance(parsed, dict | list):
        return value
    redacted = _redact_secret_fields(parsed, depth=depth + 1, flag=flag)
    return json.dumps(redacted, separators=(",", ":"))


def _opens_root_schema_host(key: Any, item: Any) -> bool:
    """Whether a root-level `key` is a real declaration container, by NAME and by SHAPE.

    `tools` and `functions` are arrays of tool definitions and the other host keys are single
    objects, so a value of the wrong shape is not the declaration its name claims -- no provider
    would accept it -- and must not open the schema exemption for its subtree.
    """
    if key not in _ROOT_SCHEMA_HOST_KEYS:
        return False
    if key in _ARRAY_SHAPED_ROOT_HOST_KEYS:
        return isinstance(item, list)
    return isinstance(item, dict)


def _carries_tool_result(container: dict[Any, Any], key: Any, *, inside_tool_result: bool) -> bool:
    """Whether `key`'s value is a tool's OUTPUT rather than ordinary prose.

    Tools routinely return serialized json, so treating the value as an opaque scalar preserved
    `{"password": "..."}` verbatim and a third-party credential a tool returned -- never in
    `context.secrets` -- reached the raw export intact. Output arrives either as `content` directly
    or, in the parts form, as a part's `text`; the parts case is narrowed to that key so a part's
    `type` or `id` is not parsed as output.
    """
    if key == "content" and container.get("role") in {"tool", "function"}:
        return True
    # the enclosing message's role already established that this is tool output; the part's `type`
    # only narrows WHICH member carries it, and `text` is that member in every spelling. requiring
    # a recognized `type` meant an absent or vendor-specific one dropped the context, and a
    # serialized credential in that part was preserved verbatim. a MALFORMED discriminator (an int,
    # a list) is no more trustworthy than a missing one -- treating it as "not a tool result" let
    # `{"type": 1, "text": "{\"password\": ...}"}` through -- so the `type` is not consulted at all.
    # the role is what decides; `type` cannot revoke it.
    return inside_tool_result and key == "text"


def _resolve_secret_schema_refs(
    value: dict[Any, Any],
    *,
    depth: int,
    secret_schema_definition: bool,
    secret_schema_refs: set[tuple[str, ...]] | None,
    schema_definition_path: tuple[str, ...],
) -> tuple[set[tuple[str, ...]], bool]:
    """Collect the pointers whose targets hold secret literals, and whether THIS node is one.

    A node that declares its own `$id` starts a new resource scope, so it inherits the flag only
    when something actually references it -- otherwise an unrelated sibling definition's secrecy
    would leak across the scope boundary.
    """
    active = (secret_schema_refs or set()) | {
        (*schema_definition_path, *pointer)
        for pointer in _secret_schema_definition_refs(value, depth=depth)
    }
    referenced = schema_definition_path in active
    if schema_definition_path and isinstance(value.get("$id"), str):
        return active, referenced
    return active, secret_schema_definition or referenced


def _child_response_shape_flags(
    key: Any,
    item: Any,
    *,
    response_root: bool,
    choice_list: bool,
    choice: bool,
    logprobs: bool,
    logprob_entries: bool,
) -> dict[str, bool]:
    """The response-envelope flags a child inherits: where it sits in `choices` and `logprobs`.

    These travel together because they describe one thing -- position inside a reply envelope --
    and `logprob_entries` in particular must stay sticky so a token whose text happens to look like
    a secret key is not redacted out of a logprob table.
    """
    return {
        "choice_list": response_root and key == "choices" and isinstance(item, list),
        "choice": choice_list,
        "logprobs": choice and key == "logprobs" and isinstance(item, dict),
        "logprob_entries": logprob_entries
        or (logprobs and key in {"content", "refusal", "top_logprobs"}),
    }


def _child_secret_schema_flags(
    key: Any,
    *,
    schema_definition: bool,
    schema_property_pattern_map: bool,
    secret_schema_definition: bool,
    secret_schema_property: bool,
    referenced_secret_definition: bool,
) -> tuple[bool, bool]:
    """The two secret-schema flags a child node inherits: definition-level and property-level.

    `propertyNames` constrains the NAMES a member may have, never its value, so its `enum` lists
    allowed property names. Propagating either flag into it rewrote that list to `["[redacted]"]`
    and silently changed the stored schema's contract while protecting nothing -- no credential is
    ever spelled there. Both flags therefore stop at that keyword.

    Under `patternProperties` the key is a REGEX rather than a name, so the secrecy test strips its
    anchors: `^password$` names exactly the field `password`.
    """
    if key == "propertyNames":
        return False, False
    declares_secret_property = schema_definition and (
        _is_secret_property_pattern(key) if schema_property_pattern_map else _is_secret_key(key)
    )
    return (
        secret_schema_definition or referenced_secret_definition,
        secret_schema_property or declares_secret_property,
    )


def _redact_secret_child(
    value: dict[Any, Any],
    key: Any,
    item: Any,
    *,
    depth: int,
    schema_property_map: bool,
    schema_property_pattern_map: bool,
    schema_property_dependencies: bool,
    schema_context: bool,
    schema_definition: bool,
    secret_schema_definition: bool,
    secret_schema_property: bool,
    referenced_secret_definition: bool,
    response_root: bool,
    choice_list: bool,
    choice: bool,
    logprobs: bool,
    logprob_entries: bool,
    function_container: bool,
    tool_result_content: bool,
    schema_host: bool,
    schema_wrapper: bool,
    tool_call: bool,
    payload_root: bool,
    active_secret_schema_refs: set[tuple[str, ...]],
    current_schema_path: tuple[str, ...],
    flag: _SanitizationFlag | None,
) -> Any:
    """Redact one member of a mapping, deciding what context the child inherits.

    This is the whole of the non-secret, non-literal case: everything about WHERE the child sits --
    schema context, host and wrapper exemptions, response-envelope position, tool-call framing --
    is decided here so the mapping walk itself stays readable.
    """
    child_schema_context = schema_context and (
        _is_schema_map_keyword(key, item)
        or key in _JSON_SCHEMA_VALUE_KEYWORDS
        or schema_property_map
        or key in _JSON_SCHEMA_KEYWORDS
    )
    # a wrapper key only grants the schema exemption inside a container that actually
    # declares schemas. `{"parameters": {"type": "object", "properties": {...}}}` is a
    # perfectly ordinary metadata shape, and honouring the wrapper anywhere let it claim
    # the exemption: its secret-named property kept an unknown `value` verbatim, so a
    # third-party credential that is not in `context.secrets` reached the raw export.
    wrapper_has_schema = (
        schema_host and key in _JSON_SCHEMA_WRAPPER_KEYS and _has_schema_wrapper_evidence(item)
    )
    if wrapper_has_schema:
        child_schema_context = True
    child_secret_definition, child_secret_property = _child_secret_schema_flags(
        key,
        schema_definition=schema_definition,
        schema_property_pattern_map=schema_property_pattern_map,
        secret_schema_definition=secret_schema_definition,
        secret_schema_property=secret_schema_property,
        referenced_secret_definition=referenced_secret_definition,
    )
    return _redact_secret_fields(
        item,
        depth=depth + 1,
        schema_property_map=(schema_context and _is_schema_map_keyword(key, item)),
        schema_property_pattern_map=(key == "patternProperties"),
        schema_property_dependencies=(
            key == "dependencies" or key in _JSON_SCHEMA_PROPERTY_NAME_MAP_KEYWORDS
        ),
        schema_context=child_schema_context,
        secret_schema_definition=child_secret_definition,
        secret_schema_property=child_secret_property,
        payload_root=False,
        response_root=False,
        **_child_response_shape_flags(
            key,
            item,
            response_root=response_root,
            choice_list=choice_list,
            choice=choice,
            logprobs=logprobs,
            logprob_entries=logprob_entries,
        ),
        function_arguments=function_container and key == "arguments",
        tool_result_content=_carries_tool_result(
            value, key, inside_tool_result=tool_result_content
        ),
        function_container=(key == "function_call" or (tool_call and key == "function")),
        # only the request's OWN declaration keys open a schema host, and only at the
        # payload root. recognizing the names anywhere let ordinary nested metadata --
        # `{"metadata": {"tools": {...}}}` -- open one for its whole subtree and keep a
        # secret-named property's literal verbatim. inside a host the flag stays set,
        # since a real declaration nests (`tools[].function.parameters`). `function`
        # and `json_schema` are spelled only INSIDE a host, so a top-level extension
        # of either name is ordinary data rather than a declaration.
        schema_host=schema_host or (payload_root and _opens_root_schema_host(key, item)),
        # `tools`/`functions` hold tool DEFINITIONS; their entries must each qualify.
        tool_definition_list=(
            payload_root and key in _ARRAY_SHAPED_ROOT_HOST_KEYS and isinstance(item, list)
        ),
        schema_wrapper=schema_wrapper
        or wrapper_has_schema
        or (schema_context and _is_schema_map_keyword(key, item)),
        tool_call_list=key == "tool_calls" and isinstance(item, list),
        secret_schema_refs=active_secret_schema_refs,
        schema_definition_path=current_schema_path,
        flag=flag,
    )


def _redact_secret_fields(
    value: Any,
    *,
    depth: int = 0,
    schema_property_map: bool = False,
    schema_property_pattern_map: bool = False,
    schema_property_dependencies: bool = False,
    schema_context: bool = False,
    secret_schema_definition: bool = False,
    secret_schema_property: bool = False,
    # the value handed to this function IS the payload root; recursion passes False explicitly.
    payload_root: bool = True,
    response_root: bool = False,
    choice_list: bool = False,
    choice: bool = False,
    logprobs: bool = False,
    logprob_entries: bool = False,
    function_arguments: bool = False,
    tool_result_content: bool = False,
    function_container: bool = False,
    schema_host: bool = False,
    schema_wrapper: bool = False,
    tool_definition_list: bool = False,
    tool_call_list: bool = False,
    tool_call: bool = False,
    secret_schema_refs: set[tuple[str, ...]] | None = None,
    schema_definition_path: tuple[str, ...] = (),
    flag: _SanitizationFlag | None = None,
) -> Any:
    if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
        if flag is not None:
            flag.hit = True
        return "[redacted]"
    if isinstance(value, dict):
        schema_context = schema_context or schema_wrapper or _has_schema_context(value)
        active_secret_schema_refs, secret_schema_definition = _resolve_secret_schema_refs(
            value,
            depth=depth,
            secret_schema_definition=secret_schema_definition,
            secret_schema_refs=secret_schema_refs,
            schema_definition_path=schema_definition_path,
        )
        redact_schema_literals = secret_schema_definition or secret_schema_property
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            schema_definition = schema_property_map and (
                _is_schema_definition(item, allow_custom_vocabulary=schema_wrapper)
                or (schema_property_dependencies and _is_property_name_list(item))
            )
            current_schema_path = (*schema_definition_path, str(key))
            referenced_secret_definition = current_schema_path in active_secret_schema_refs
            if redact_schema_literals and key in _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS:
                redacted[key] = _redact_schema_literal(item, depth=depth + 1, flag=flag)
            elif _is_secret_key(key, allow_token=logprob_entries) and not schema_definition:
                redacted[key] = "[redacted]"
            else:
                redacted[key] = _redact_secret_child(
                    value,
                    key,
                    item,
                    depth=depth,
                    schema_property_map=schema_property_map,
                    schema_property_pattern_map=schema_property_pattern_map,
                    schema_property_dependencies=schema_property_dependencies,
                    schema_context=schema_context,
                    schema_definition=schema_definition,
                    secret_schema_definition=secret_schema_definition,
                    secret_schema_property=secret_schema_property,
                    referenced_secret_definition=referenced_secret_definition,
                    response_root=response_root,
                    choice_list=choice_list,
                    choice=choice,
                    logprobs=logprobs,
                    logprob_entries=logprob_entries,
                    function_container=function_container,
                    tool_result_content=tool_result_content,
                    schema_host=schema_host,
                    schema_wrapper=schema_wrapper,
                    tool_call=tool_call,
                    payload_root=payload_root,
                    active_secret_schema_refs=active_secret_schema_refs,
                    current_schema_path=current_schema_path,
                    flag=flag,
                )
        return redacted
    if isinstance(value, list | tuple):
        return _redact_secret_sequence(
            value,
            depth=depth,
            schema_property_map=schema_property_map,
            schema_context=schema_context,
            secret_schema_definition=secret_schema_definition,
            secret_schema_property=secret_schema_property,
            choice_list=choice_list,
            logprobs=logprobs,
            logprob_entries=logprob_entries,
            tool_result_content=tool_result_content,
            schema_host=schema_host,
            tool_definition_list=tool_definition_list,
            tool_call_list=tool_call_list,
            schema_wrapper=schema_wrapper,
            secret_schema_refs=secret_schema_refs,
            schema_definition_path=schema_definition_path,
            flag=flag,
        )
    return _redact_secret_scalar(
        value,
        depth=depth,
        tool_result_content=tool_result_content,
        function_arguments=function_arguments,
        flag=flag,
    )


def _is_tool_definition(item: Any) -> bool:
    """Whether an entry of a `tools`/`functions` array is really a tool definition.

    The array shape alone is not the declaration: `{"tools": [{"parameters": {...}}]}` has the right
    container but an entry no provider would accept, and honouring it let the nested wrapper keep a
    secret property's literal. A real entry either wraps its declaration in `function`, or names the
    tool directly alongside its schema (Anthropic's `input_schema`, the legacy `functions` form).
    """
    if not isinstance(item, dict):
        return False
    # a definition NAMES the tool it declares -- that is what a later `tool_choice` selects and what
    # the provider calls back with. accepting a bare `{"function": {...}}` wrapper let an entry no
    # provider would accept open the schema exemption, so its nested wrapper kept a literal.
    function = item.get("function")
    if isinstance(function, dict):
        # the wrapper is only a declaration under the tool type that DEFINES it. `{"type":
        # "custom", "function": {...}}` is an entry no provider accepts, yet a bare name check
        # honoured it and opened the schema exemption, so the nested wrapper kept a secret
        # literal. an absent type is the pre-`type` spelling and stays valid; a present one has
        # to agree. a blank name names nothing, so it cannot be the tool a `tool_choice` selects.
        declared_type = item.get("type")
        if declared_type is not None and declared_type != "function":
            return False
        name = function.get("name")
        return isinstance(name, str) and bool(name.strip())
    name = item.get("name")
    return (
        isinstance(name, str)
        and bool(name.strip())
        and any(isinstance(item.get(wrapper), dict) for wrapper in _JSON_SCHEMA_WRAPPER_KEYS)
    )


def _redact_secret_sequence(
    value: list[Any] | tuple[Any, ...],
    *,
    depth: int,
    schema_property_map: bool,
    schema_context: bool,
    secret_schema_definition: bool,
    secret_schema_property: bool,
    choice_list: bool,
    logprobs: bool,
    logprob_entries: bool,
    tool_result_content: bool,
    schema_host: bool,
    tool_definition_list: bool,
    tool_call_list: bool,
    schema_wrapper: bool,
    secret_schema_refs: set[tuple[str, ...]] | None,
    schema_definition_path: tuple[str, ...],
    flag: _SanitizationFlag | None,
) -> list[Any]:
    """Redact each entry of an array, carrying the flags its position implies."""
    return [
        _redact_secret_fields(
            item,
            depth=depth + 1,
            schema_property_map=schema_property_map,
            schema_context=schema_context,
            secret_schema_definition=(
                secret_schema_definition
                or (*schema_definition_path, str(index)) in (secret_schema_refs or set())
            ),
            secret_schema_property=secret_schema_property,
            choice=choice_list,
            logprobs=logprobs,
            logprob_entries=logprob_entries,
            payload_root=False,
            # a tool message's `content` may be a list of parts rather than one string, and the
            # tool's output then sits in each part's `text`. dropping the flag at the list hop
            # left a serialized credential in a part verbatim.
            tool_result_content=tool_result_content,
            # inside a declaration array the host survives only for entries that really are tool
            # definitions; a fake entry beside a real one gets no exemption of its own.
            schema_host=schema_host and (not tool_definition_list or _is_tool_definition(item)),
            tool_call=tool_call_list,
            schema_wrapper=schema_wrapper,
            secret_schema_refs=secret_schema_refs,
            schema_definition_path=(*schema_definition_path, str(index)),
            flag=flag,
        )
        for index, item in enumerate(value)
    ]


def _redact_secret_scalar(
    value: Any,
    *,
    depth: int,
    tool_result_content: bool,
    function_arguments: bool,
    flag: _SanitizationFlag | None,
) -> Any:
    """Redact a leaf value, whose handling depends on the field that carries it."""
    if tool_result_content and isinstance(value, str):
        return _redact_tool_result_content(value, depth=depth, flag=flag)
    if function_arguments and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, RecursionError):
            # malformed arguments cannot be inspected for nested credentials, so preserving their
            # bytes would turn parse failure into a secret exfiltration path. keep the wire type only.
            return "[redacted]"
        redacted = _redact_secret_fields(parsed, depth=depth + 1, flag=flag)
        return json.dumps(redacted, separators=(",", ":"))
    return value


def _redact_secret_string(value: str, secrets: tuple[str, ...]) -> str:
    eligible_secrets = {secret for secret in secrets if len(secret) >= _MIN_SECRET_SUBSTRING_LENGTH}
    for secret in sorted(eligible_secrets, key=len, reverse=True):
        value = value.replace(secret, "[redacted]")
    return value


def _redact_secret_values(
    value: Any,
    secrets: tuple[str, ...],
    *,
    depth: int = 0,
    flag: _SanitizationFlag | None = None,
) -> Any:
    if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
        if flag is not None:
            flag.hit = True
        return "[redacted]"
    if isinstance(value, dict):
        # keys are redacted too. a credential used as an object key -- `{"sk-live-...": "seen"}` --
        # is still the credential, and a key-blind pass would write it into the span verbatim and
        # hand it back through `format=raw`.
        return {
            (_redact_secret_string(key, secrets) if isinstance(key, str) else key): (
                _redact_secret_values(item, secrets, depth=depth + 1, flag=flag)
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return [_redact_secret_values(item, secrets, depth=depth + 1, flag=flag) for item in value]
    if isinstance(value, str):
        return _redact_secret_string(value, secrets)
    return value


def _sanitize_for_trace(
    value: Any,
    secrets: tuple[str, ...],
    *,
    response: bool = False,
    flag: _SanitizationFlag | None = None,
) -> Any:
    redacted = _redact_secret_fields(value, payload_root=True, response_root=response, flag=flag)
    return _redact_secret_values(redacted, secrets, flag=flag)
