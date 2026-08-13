"""Secret and JSON Schema redaction for recorded traces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urldefrag, urljoin, urlsplit, urlunsplit

from flash.server.platform import traces as platform_traces

# short strings occur naturally in prompts and object keys. treating one as a global substring
# secret corrupts unrelated training text, while real bearer credentials are comfortably longer.
_MIN_SECRET_SUBSTRING_LENGTH = 16

_SECRET_KEY_EXACT = frozenset({"authorization", "proxyauthorization"})
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
_JSON_SCHEMA_VALUE_KEYWORDS = frozenset({"default", "examples", "example"})
_JSON_SCHEMA_SECRET_LITERAL_KEYWORDS = frozenset(
    {"default", "const", "enum", "examples", "example"}
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


def _is_schema_definition(value: Any) -> bool:
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
    if any(key in _JSON_SCHEMA_STRUCTURAL_KEYWORDS for key in keys):
        return True
    if any(key not in _JSON_SCHEMA_KEYWORDS for key in keys):
        return False
    return bool(keys) and all(key not in _JSON_SCHEMA_VALUE_KEYWORDS for key in keys)


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


def _canonical_resource_uri(uri: str) -> str:
    scheme, netloc, path, query, fragment = urlsplit(uri)
    userinfo, user_separator, hostport = netloc.rpartition("@")
    if hostport.startswith("[") and (host_end := hostport.find("]")) >= 0:
        hostport = f"[{hostport[1:host_end].casefold()}]{hostport[host_end + 1 :]}"
    else:
        host, port_separator, port = hostport.rpartition(":")
        hostport = f"{host.casefold()}:{port}" if port_separator else hostport.casefold()
    normalized_netloc = f"{userinfo}@{hostport}" if user_separator else hostport
    return urlunsplit((scheme.casefold(), normalized_netloc, path, query, fragment))


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
    resource_scopes = sorted(
        ((path, uri) for uri, path in resources.items()),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    # `$dynamicAnchor` also declares an ordinary plain-name fragment, so a static `$ref` resolves
    # to it as well. one map serves both keywords: splitting them let `{"$ref": "#Name"}` miss a
    # `$dynamicAnchor: "Name"` target and persist its literals.
    anchors = _schema_anchor_pointers(value, depth=depth)

    def scope_for(path: tuple[str, ...]) -> str:
        return next(
            (uri for prefix, uri in resource_scopes if path[: len(prefix)] == prefix), base_uri
        )

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
            for keyword in ("$ref", "$dynamicRef"):
                ref = node.get(keyword)
                if isinstance(ref, str):
                    resolved_base, fragment = urldefrag(urljoin(scope_uri, ref))
                    resource_path = resources.get(_canonical_resource_uri(resolved_base))
                    if resource_path is not None:
                        resource_ref = f"#{fragment}"
                        pointers = _local_schema_pointer(resource_ref, anchors)
                        if resource_ref == "#" or resource_ref.startswith("#/"):
                            found.update((*resource_path, *pointer) for pointer in pointers)
                        else:
                            found.update(pointers)
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


def _redact_secret_fields(
    value: Any,
    *,
    depth: int = 0,
    schema_property_map: bool = False,
    secret_schema_definition: bool = False,
    response_root: bool = False,
    choice_list: bool = False,
    choice: bool = False,
    logprobs: bool = False,
    logprob_entries: bool = False,
    secret_schema_refs: set[tuple[str, ...]] | None = None,
    schema_definition_path: tuple[str, ...] = (),
    flag: _SanitizationFlag | None = None,
) -> Any:
    if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
        if flag is not None:
            flag.hit = True
        return "[redacted]"
    if isinstance(value, dict):
        local_secret_schema_refs = {
            (*schema_definition_path, *pointer)
            for pointer in _secret_schema_definition_refs(value, depth=depth)
        }
        active_secret_schema_refs = (secret_schema_refs or set()) | local_secret_schema_refs
        secret_schema_definition = secret_schema_definition or (
            schema_definition_path in active_secret_schema_refs
        )
        redacted: dict[Any, Any] = {}
        for key, item in value.items():
            schema_definition = schema_property_map and _is_schema_definition(item)
            current_schema_path = (*schema_definition_path, str(key))
            referenced_secret_definition = current_schema_path in active_secret_schema_refs
            if secret_schema_definition and key in _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS:
                redacted[key] = _redact_schema_literal(item, depth=depth + 1, flag=flag)
            elif _is_secret_key(key, allow_token=logprob_entries) and not schema_definition:
                redacted[key] = "[redacted]"
            else:
                redacted[key] = _redact_secret_fields(
                    item,
                    depth=depth + 1,
                    schema_property_map=(
                        key in {"properties", "$defs", "definitions"} and isinstance(item, dict)
                    ),
                    secret_schema_definition=secret_schema_definition
                    or (schema_definition and _is_secret_key(key))
                    or referenced_secret_definition,
                    response_root=False,
                    choice_list=response_root and key == "choices" and isinstance(item, list),
                    choice=choice_list,
                    logprobs=choice and key == "logprobs" and isinstance(item, dict),
                    logprob_entries=logprob_entries
                    or (logprobs and key in {"content", "refusal", "top_logprobs"}),
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
                secret_schema_definition=(
                    secret_schema_definition
                    or (*schema_definition_path, str(index)) in (secret_schema_refs or set())
                ),
                choice=choice_list,
                logprobs=logprobs,
                logprob_entries=logprob_entries,
                secret_schema_refs=secret_schema_refs,
                schema_definition_path=(*schema_definition_path, str(index)),
                flag=flag,
            )
            for index, item in enumerate(value)
        ]
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
