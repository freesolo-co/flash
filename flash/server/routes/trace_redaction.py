"""Secret and JSON Schema redaction for recorded traces."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urldefrag, urljoin, urlsplit, urlunsplit

from flash.server.platform import traces as platform_traces

# short strings occur naturally in prompts and object keys. treating one as a global substring
# secret corrupts unrelated training text, while real bearer credentials are comfortably longer.
_MIN_SECRET_SUBSTRING_LENGTH = 16

# `auth` is exact-matched, never a suffix: `author` and `oauth` end in it but carry no credential.
_SECRET_KEY_EXACT = frozenset({"authorization", "proxyauthorization", "auth", "xauth"})
_SECRET_KEY_SUFFIXES = (
    "apikey",
    # conventional cloud credential fields end in these normalized forms. bare `key` is deliberately
    # excluded because JSON schemas and tool arguments use it pervasively for harmless data.
    "accesskeyid",
    "secretkey",
    "accesskey",
    "secret",
    "token",
    "password",
    "passwd",
    "credential",
    "credentials",
    "privatekey",
)
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
_SCHEMA_HOST_KEYS = frozenset(
    {
        "tools",
        "functions",
        "tool_choice",
        "function_call",
        "function",
        "response_format",
        "json_schema",
        "text_format",
    }
)
_JSON_SCHEMA_TYPES = frozenset(
    {"null", "boolean", "object", "array", "number", "string", "integer"}
)


@dataclass
class _SanitizationFlag:
    hit: bool = False


def _is_secret_key(key: Any, *, allow_token: bool = False) -> bool:
    normalized = str(key).casefold().replace("_", "").replace("-", "")
    return normalized in _SECRET_KEY_EXACT or (
        not (allow_token and normalized == "token")
        and any(normalized.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES)
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
    return key in _JSON_SCHEMA_PROPERTY_MAP_KEYWORDS or (
        key == "dependencies" and isinstance(item, dict)
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


def _local_schema_pointer(
    ref: str,
    anchors: Mapping[str, frozenset[tuple[str, ...]]],
    *,
    base_uri: str = "",
) -> frozenset[tuple[str, ...]]:
    if not ref.startswith("#"):
        ref_base, fragment = urldefrag(urljoin(base_uri, ref))
        document_base = urldefrag(base_uri)[0]
        if not document_base or ref_base != document_base:
            return frozenset()
        ref = f"#{fragment}"
    ref = unquote(ref)
    if ref == "#":
        return frozenset({()})
    if ref.startswith("#/"):
        segments = tuple(
            segment.replace("~1", "/").replace("~0", "~") for segment in ref[2:].split("/")
        )
        return frozenset({segments}) if segments else frozenset()
    if ref.startswith("#") and len(ref) > 1:
        return anchors.get(ref[1:], frozenset())
    return frozenset()


def _normalize_percent_encoding(value: str, *, fold_decoded: bool = False) -> str:
    """Decode unreserved escapes and uppercase the rest, per RFC 3986 6.2.2.

    `fold_decoded` case-folds the characters an escape decodes to. It is for the host, which is
    case-insensitive: without it `%4A` normalizes to `J` and never matches a plain `j`. Folding the
    whole string instead would lowercase the hex digits of escapes that stay encoded.
    """
    unreserved = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~"
    normalized: list[str] = []
    index = 0
    while index < len(value):
        if index + 2 < len(value) and value[index] == "%":
            escaped = value[index + 1 : index + 3]
            try:
                decoded = chr(int(escaped, 16))
            except ValueError:
                pass
            else:
                if decoded in unreserved:
                    normalized.append(decoded.casefold() if fold_decoded else decoded)
                else:
                    normalized.append(f"%{escaped.upper()}")
                index += 3
                continue
        normalized.append(value[index])
        index += 1
    return "".join(normalized)


def _remove_dot_segments(path: str) -> str:
    input_buffer = path
    output: list[str] = []
    while input_buffer:
        if input_buffer.startswith("../"):
            input_buffer = input_buffer[3:]
        elif input_buffer.startswith("./"):
            input_buffer = input_buffer[2:]
        elif input_buffer.startswith("/./"):
            input_buffer = f"/{input_buffer[3:]}"
        elif input_buffer == "/.":
            input_buffer = "/"
        elif input_buffer.startswith("/../"):
            input_buffer = f"/{input_buffer[4:]}"
            if output:
                output.pop()
        elif input_buffer == "/..":
            input_buffer = "/"
            if output:
                output.pop()
        elif input_buffer in {".", ".."}:
            input_buffer = ""
        else:
            segment_end = input_buffer.find("/", 1 if input_buffer.startswith("/") else 0)
            if segment_end < 0:
                output.append(input_buffer)
                input_buffer = ""
            else:
                output.append(input_buffer[:segment_end])
                input_buffer = input_buffer[segment_end:]
    return "".join(output)


def _canonical_resource_uri(uri: str) -> str:
    scheme, netloc, path, query, fragment = urlsplit(uri)
    normalized_scheme = scheme.casefold()
    default_port = {"http": "80", "https": "443"}.get(normalized_scheme)
    userinfo, user_separator, hostport = netloc.rpartition("@")
    if hostport.startswith("[") and (host_end := hostport.find("]")) >= 0:
        suffix = hostport[host_end + 1 :]
        if suffix == ":" or (default_port is not None and suffix == f":{default_port}"):
            suffix = ""
        hostport = f"[{hostport[1:host_end].casefold()}]{suffix}"
    else:
        host, port_separator, port = hostport.rpartition(":")
        if port_separator and (not port or port == default_port):
            hostport = host.casefold()
        else:
            hostport = f"{host.casefold()}:{port}" if port_separator else hostport.casefold()
    # the host is case-insensitive, so a character an escape DECODES to must fold too: `%4A` and a
    # plain `j` are the same host. the casefold above cannot do it (the escape is still encoded) and
    # folding the whole string afterwards would lowercase the hex digits rfc 3986 6.2.2.1 wants
    # uppercase, so the fold is applied to decoded characters only. the userinfo is case-sensitive
    # and keeps the plain normalization. decoding is delimiter-safe here because no unreserved
    # character is a delimiter: `@`, `:` and the brackets stay encoded and the split above holds.
    normalized_netloc = (
        f"{_normalize_percent_encoding(userinfo)}@"
        f"{_normalize_percent_encoding(hostport, fold_decoded=True)}"
        if user_separator
        else _normalize_percent_encoding(hostport, fold_decoded=True)
    )
    normalized_path = _remove_dot_segments(_normalize_percent_encoding(path))
    if normalized_scheme in {"http", "https"} and normalized_netloc and not normalized_path:
        normalized_path = "/"
    return urlunsplit(
        (
            normalized_scheme,
            normalized_netloc,
            normalized_path,
            _normalize_percent_encoding(query),
            _normalize_percent_encoding(fragment),
        )
    )


def _schema_resource_pointers(value: Any, *, depth: int = 0) -> dict[str, tuple[str, ...]]:
    resources: dict[str, tuple[str, ...]] = {}

    def collect(node: Any, path: tuple[str, ...], base_uri: str, depth: int) -> None:
        if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
            return
        if isinstance(node, dict):
            resource_id = node.get("$id")
            if isinstance(resource_id, str):
                base_uri = urljoin(base_uri, resource_id)
                resources.setdefault(_canonical_resource_uri(urldefrag(base_uri)[0]), path)
            for key, item in node.items():
                if key not in _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS:
                    collect(item, (*path, str(key)), base_uri, depth + 1)
        elif isinstance(node, list | tuple):
            for index, item in enumerate(node):
                collect(item, (*path, str(index)), base_uri, depth + 1)

    collect(value, (), "", depth)
    return resources


def _schema_anchor_pointers(value: Any, *, depth: int = 0) -> dict[str, frozenset[tuple[str, ...]]]:
    anchors: dict[str, set[tuple[str, ...]]] = {}
    keywords = ("$anchor", "$dynamicAnchor")

    def collect(node: Any, path: tuple[str, ...], depth: int) -> None:
        if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
            return
        if isinstance(node, dict):
            for keyword in keywords:
                anchor = node.get(keyword)
                if isinstance(anchor, str):
                    anchors.setdefault(anchor, set()).add(path)
            if node.get("$recursiveAnchor") is True:
                anchors.setdefault("", set()).add(path)
            for key, item in node.items():
                if key not in _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS:
                    collect(item, (*path, str(key)), depth + 1)
        elif isinstance(node, list | tuple):
            for index, item in enumerate(node):
                collect(item, (*path, str(index)), depth + 1)

    collect(value, (), depth)
    return {name: frozenset(paths) for name, paths in anchors.items()}


def _schema_dynamic_anchor_pointers(
    value: Any, *, depth: int = 0
) -> dict[str, frozenset[tuple[str, ...]]]:
    """Pointers a dynamic reference can reach, keyed by anchor name (`""` for `$recursiveAnchor`).

    `$dynamicRef` and `$recursiveRef` resolve against the dynamic scope, so the target is chosen by
    the outermost resource that declares the anchor rather than by the reference's own resource.
    Which resource that is depends on the evaluation entry point, which a redactor recording a
    payload cannot know, so every same-named dynamic anchor counts as a possible target.
    """
    anchors: dict[str, set[tuple[str, ...]]] = {}

    def collect(node: Any, path: tuple[str, ...], depth: int) -> None:
        if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
            return
        if isinstance(node, dict):
            anchor = node.get("$dynamicAnchor")
            if isinstance(anchor, str):
                anchors.setdefault(anchor, set()).add(path)
            if node.get("$recursiveAnchor") is True:
                anchors.setdefault("", set()).add(path)
            for key, item in node.items():
                if key not in _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS:
                    collect(item, (*path, str(key)), depth + 1)
        elif isinstance(node, list | tuple):
            for index, item in enumerate(node):
                collect(item, (*path, str(index)), depth + 1)

    collect(value, (), depth)
    return {name: frozenset(paths) for name, paths in anchors.items()}


def _secret_schema_definition_refs(value: Any, *, depth: int = 0) -> set[tuple[str, ...]]:
    if not isinstance(value, dict):
        return set()
    refs: set[tuple[str, ...]] = set()
    document_id = value.get("$id")
    base_uri = document_id if isinstance(document_id, str) else ""
    resources = _schema_resource_pointers(value, depth=depth)
    resources.setdefault(_canonical_resource_uri(urldefrag(base_uri)[0]), ())
    resource_scopes = sorted(
        ((path, uri) for uri, path in resources.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    # `$dynamicAnchor` also declares an ordinary plain-name fragment, so a static `$ref` resolves
    # to it as well. one map serves both keywords: splitting them let `{"$ref": "#Name"}` miss a
    # `$dynamicAnchor: "Name"` target and persist its literals.
    anchors = _schema_anchor_pointers(value, depth=depth)
    dynamic_anchors = _schema_dynamic_anchor_pointers(value, depth=depth)

    def scope_for(path: tuple[str, ...]) -> str:
        return next(
            (uri for prefix, uri in resource_scopes if path[: len(prefix)] == prefix), base_uri
        )

    def anchor_belongs_to_resource(pointer: tuple[str, ...], resource_uri: str) -> bool:
        owner_uri = _canonical_resource_uri(urldefrag(scope_for(pointer))[0])
        return owner_uri == _canonical_resource_uri(resource_uri)

    def collect_refs(
        node: Any, node_depth: int, path: tuple[str, ...], scope_uri: str
    ) -> set[tuple[str, ...]]:
        if node_depth >= platform_traces._MAX_PAYLOAD_DEPTH:
            return set()
        found: set[tuple[str, ...]] = set()
        if isinstance(node, dict):
            resource_id = node.get("$id")
            if isinstance(resource_id, str):
                scope_uri = urljoin(scope_uri, resource_id)
            for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
                ref = node.get(keyword)
                if isinstance(ref, str):
                    # a dynamic reference leaves its own resource by design: the target is the
                    # outermost dynamic scope declaring the anchor, which depends on the evaluation
                    # entry point. restricting it to the reference's resource kept literals on the
                    # outer target exposed, so dynamic keywords consider every same-named anchor.
                    is_dynamic = keyword in {"$dynamicRef", "$recursiveRef"}
                    resolved_base, fragment = urldefrag(urljoin(scope_uri, ref))
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
            resource_id = node.get("$id")
            if isinstance(resource_id, str):
                scope_uri = urljoin(scope_uri, resource_id)
            properties = node.get("properties")
            if isinstance(properties, dict):
                for key, schema in properties.items():
                    schema_path = (*path, "properties", str(key))
                    if _is_secret_key(key) and _is_schema_definition(schema):
                        refs.update(collect_refs(schema, 0, schema_path, scope_uri))
                    collect_secret_properties(schema, node_depth + 1, schema_path, scope_uri)
            for key, item in node.items():
                if key != "properties":
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
        for pointer in collect_refs(target, 0, target_path, scope_for(target_path)):
            if pointer not in refs:
                refs.add(pointer)
                pending.append(pointer)
    return refs


def _redact_tool_result_content(value: str, *, depth: int, flag: _SanitizationFlag | None) -> str:
    """Redact a tool message's `content`, which carries the tool's output rather than prose."""
    try:
        parsed = json.loads(value)
    except (ValueError, RecursionError):
        # unlike function arguments, a tool result is not required to be json: prose is the
        # ordinary case. a parse failure here means "this was never structured", so the text is
        # kept as-is rather than blanked -- blanking would destroy legitimate tool output.
        return value
    if not isinstance(parsed, dict | list):
        return value
    redacted = _redact_secret_fields(parsed, depth=depth + 1, flag=flag)
    return json.dumps(redacted, separators=(",", ":"))


def _redact_secret_fields(
    value: Any,
    *,
    depth: int = 0,
    schema_property_map: bool = False,
    schema_property_dependencies: bool = False,
    schema_context: bool = False,
    secret_schema_definition: bool = False,
    secret_schema_property: bool = False,
    response_root: bool = False,
    choice_list: bool = False,
    choice: bool = False,
    logprobs: bool = False,
    logprob_entries: bool = False,
    function_arguments: bool = False,
    tool_result_content: bool = False,
    message_list: bool = False,
    function_container: bool = False,
    schema_host: bool = False,
    schema_wrapper: bool = False,
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
        local_secret_schema_refs = {
            (*schema_definition_path, *pointer)
            for pointer in _secret_schema_definition_refs(value, depth=depth)
        }
        active_secret_schema_refs = (secret_schema_refs or set()) | local_secret_schema_refs
        referenced_schema_definition = schema_definition_path in active_secret_schema_refs
        if schema_definition_path and isinstance(value.get("$id"), str):
            secret_schema_definition = referenced_schema_definition
        else:
            secret_schema_definition = secret_schema_definition or referenced_schema_definition
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
                    schema_host
                    and key in _JSON_SCHEMA_WRAPPER_KEYS
                    and _has_schema_wrapper_evidence(item)
                )
                if wrapper_has_schema:
                    child_schema_context = True
                redacted[key] = _redact_secret_fields(
                    item,
                    depth=depth + 1,
                    schema_property_map=(schema_context and _is_schema_map_keyword(key, item)),
                    schema_property_dependencies=key == "dependencies",
                    schema_context=child_schema_context,
                    secret_schema_definition=secret_schema_definition
                    or referenced_secret_definition,
                    secret_schema_property=secret_schema_property
                    or (schema_definition and _is_secret_key(key)),
                    response_root=False,
                    choice_list=response_root and key == "choices" and isinstance(item, list),
                    choice=choice_list,
                    logprobs=choice and key == "logprobs" and isinstance(item, dict),
                    logprob_entries=logprob_entries
                    or (logprobs and key in {"content", "refusal", "top_logprobs"}),
                    function_arguments=function_container and key == "arguments",
                    # a tool message's `content` carries the tool's OUTPUT, and tools routinely
                    # return serialized json there. treating it as an opaque scalar preserved
                    # `{"password": "..."}` verbatim, so a third-party credential a tool returned --
                    # never in `context.secrets` -- reached the raw export intact.
                    tool_result_content=(
                        key == "content" and value.get("role") in {"tool", "function"}
                    ),
                    message_list=key == "messages" and isinstance(item, list),
                    function_container=(
                        key == "function_call" or (tool_call and key == "function")
                    ),
                    schema_host=schema_host or key in _SCHEMA_HOST_KEYS,
                    schema_wrapper=schema_wrapper
                    or wrapper_has_schema
                    or (schema_context and _is_schema_map_keyword(key, item)),
                    tool_call_list=key == "tool_calls" and isinstance(item, list),
                    secret_schema_refs=active_secret_schema_refs,
                    schema_definition_path=current_schema_path,
                    flag=flag,
                )
        return redacted
    if isinstance(value, list | tuple):
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
                schema_host=schema_host,
                tool_call=tool_call_list,
                schema_wrapper=schema_wrapper,
                secret_schema_refs=secret_schema_refs,
                schema_definition_path=(*schema_definition_path, str(index)),
                flag=flag,
            )
            for index, item in enumerate(value)
        ]
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
    redacted = _redact_secret_fields(value, response_root=response, flag=flag)
    return _redact_secret_values(redacted, secrets, flag=flag)
