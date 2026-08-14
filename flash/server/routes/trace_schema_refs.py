"""Collecting the schema pointers a secret-declaring property refers to.

`{"properties": {"password": {"$ref": "#/$defs/creds"}}}` says nothing secret about `password`
itself: the credential-bearing literals sit at the TARGET. Finding those targets means resolving
references against the payload's own identity graph, following them transitively, and doing so
under the dialect each embedded resource declares. That search is separable from deciding which
literals are secret once a target is known, which is what `trace_redaction` does with the result.
"""

from __future__ import annotations

from typing import Any

from flash.server.platform import traces as platform_traces
from flash.server.routes.trace_schema_identity import (
    SchemaResourceScopes,
    _bounded,
    _declares_legacy_id_dialect,
    _local_schema_pointer,
    _node_legacy_id_dialect,
    _schema_resource_id,
)
from flash.server.routes.trace_schema_shape import (
    _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS,
    _SECRET_DECLARING_PROPERTY_MAPS,
    _is_schema_definition,
)
from flash.server.routes.trace_secret_names import (
    _is_secret_key,
    _is_secret_property_pattern,
)
from flash.server.routes.trace_uri import (
    _canonical_resource_uri,
    _safe_urldefrag,
    _safe_urljoin,
)

_REFERENCE_KEYWORDS = ("$ref", "$dynamicRef", "$recursiveRef")
_DYNAMIC_REFERENCE_KEYWORDS = frozenset({"$dynamicRef", "$recursiveRef"})


def _secret_schema_definition_refs(value: Any, *, depth: int = 0) -> set[tuple[str, ...]]:
    """The pointers, inside `value`, to definitions a secret-declaring property refers to."""
    if not isinstance(value, dict):
        return set()
    scopes = SchemaResourceScopes.build(value, depth=depth)
    legacy_id_dialect = _declares_legacy_id_dialect(value, depth=depth)

    refs: set[tuple[str, ...]] = set()
    _collect_secret_properties(
        value,
        depth,
        (),
        scopes.base_uri,
        scopes=scopes,
        legacy_id_dialect=legacy_id_dialect,
        refs=refs,
    )
    # a target may itself reference a further definition, so the collected set is closed under
    # resolution rather than read once.
    pending = list(refs)
    while pending:
        target_path = pending.pop()
        target = _resolve_pointer(value, target_path)
        for pointer in _collect_refs(
            target,
            0,
            target_path,
            scopes.scope_for(target_path),
            scopes=scopes,
            legacy_id_dialect=legacy_id_dialect,
        ):
            if pointer not in refs:
                refs.add(pointer)
                pending.append(pointer)
    return refs


def _collect_secret_properties(
    node: Any,
    node_depth: int,
    path: tuple[str, ...],
    scope_uri: str,
    *,
    scopes: SchemaResourceScopes,
    legacy_id_dialect: bool,
    refs: set[tuple[str, ...]],
) -> None:
    """Walk `node`, adding to `refs` every pointer a secret-declaring property refers to."""
    if node_depth >= platform_traces._MAX_PAYLOAD_DEPTH:
        return
    if isinstance(node, dict):
        # an embedded resource may declare its own dialect, so the reading is taken at each node
        # rather than fixed for the payload.
        legacy_here = _node_legacy_id_dialect(node, legacy_id_dialect)
        resource_id = _schema_resource_id(node, legacy_id_dialect=legacy_here)
        if isinstance(resource_id, str):
            scope_uri = _safe_urljoin(scope_uri, resource_id)
        # `patternProperties` declares secret-looking members too: `{"^secret_": {...}}` marks
        # every matching property secret. scanning only `properties` left a reference from such an
        # entry uncollected, so its target definition kept its credential-bearing literal.
        # `dependentSchemas` (and its draft-07 `dependencies` spelling) is keyed by property name
        # too, so `{"password": {"$ref": ...}}` declares a secret member exactly like `properties`
        # does. scanning only the two property maps left that reference uncollected, and the target
        # definition kept its credential-bearing literal.
        for map_keyword in _SECRET_DECLARING_PROPERTY_MAPS:
            property_map = node.get(map_keyword)
            if not isinstance(property_map, dict):
                continue
            for key, schema in _bounded(property_map.items()):
                schema_path = (*path, map_keyword, str(key))
                secret_declaration = (
                    _is_secret_property_pattern(key)
                    if map_keyword == "patternProperties"
                    else _is_secret_key(key)
                )
                if secret_declaration and _is_schema_definition(schema):
                    refs.update(
                        _collect_refs(
                            schema,
                            0,
                            schema_path,
                            scope_uri,
                            scopes=scopes,
                            legacy_id_dialect=legacy_id_dialect,
                        )
                    )
                _collect_secret_properties(
                    schema,
                    node_depth + 1,
                    schema_path,
                    scope_uri,
                    scopes=scopes,
                    legacy_id_dialect=legacy_id_dialect,
                    refs=refs,
                )
        for key, item in _bounded(node.items()):
            if key not in _SECRET_DECLARING_PROPERTY_MAPS:
                _collect_secret_properties(
                    item,
                    node_depth + 1,
                    (*path, str(key)),
                    scope_uri,
                    scopes=scopes,
                    legacy_id_dialect=legacy_id_dialect,
                    refs=refs,
                )
    elif isinstance(node, list | tuple):
        for index, item in enumerate(_bounded(node)):
            _collect_secret_properties(
                item,
                node_depth + 1,
                (*path, str(index)),
                scope_uri,
                scopes=scopes,
                legacy_id_dialect=legacy_id_dialect,
                refs=refs,
            )


def _collect_refs(
    node: Any,
    node_depth: int,
    path: tuple[str, ...],
    scope_uri: str,
    *,
    scopes: SchemaResourceScopes,
    legacy_id_dialect: bool,
) -> set[tuple[str, ...]]:
    """The pointers every reference under `node` resolves to."""
    if node_depth >= platform_traces._MAX_PAYLOAD_DEPTH:
        return set()
    found: set[tuple[str, ...]] = set()
    if isinstance(node, dict):
        # an embedded resource may declare its own dialect, so the reading is taken at each node
        # rather than fixed for the payload.
        legacy_here = _node_legacy_id_dialect(node, legacy_id_dialect)
        resource_id = _schema_resource_id(node, legacy_id_dialect=legacy_here)
        if isinstance(resource_id, str):
            scope_uri = _safe_urljoin(scope_uri, resource_id)
        for keyword in _REFERENCE_KEYWORDS:
            ref = node.get(keyword)
            if isinstance(ref, str):
                found.update(_reference_targets(keyword, ref, path, scope_uri, scopes=scopes))
        for key, item in _bounded(node.items()):
            if key not in _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS:
                found.update(
                    _collect_refs(
                        item,
                        node_depth + 1,
                        (*path, str(key)),
                        scope_uri,
                        scopes=scopes,
                        legacy_id_dialect=legacy_id_dialect,
                    )
                )
    elif isinstance(node, list | tuple):
        for index, item in enumerate(_bounded(node)):
            found.update(
                _collect_refs(
                    item,
                    node_depth + 1,
                    (*path, str(index)),
                    scope_uri,
                    scopes=scopes,
                    legacy_id_dialect=legacy_id_dialect,
                )
            )
    return found


def _reference_targets(
    keyword: str,
    ref: str,
    path: tuple[str, ...],
    scope_uri: str,
    *,
    scopes: SchemaResourceScopes,
) -> set[tuple[str, ...]]:
    """The pointers one reference keyword resolves to, given the scope it appears in."""
    # a dynamic reference leaves its own resource by design: the target is the outermost dynamic
    # scope declaring the anchor, which depends on the evaluation entry point. restricting it to
    # the reference's resource kept literals on the outer target exposed, so dynamic keywords
    # consider every same-named anchor.
    is_dynamic = keyword in _DYNAMIC_REFERENCE_KEYWORDS
    resolved_base, fragment = _safe_urldefrag(_safe_urljoin(scope_uri, ref))
    canonical_base = _canonical_resource_uri(resolved_base)
    # a canonical id may be declared by several nodes. duplicates make the schema ambiguous, so
    # every declaring path is an equally valid resolution of this reference and all of them are
    # redacted; keeping one left the others' secret literals in the raw export.
    resource_paths = scopes.resources.get(canonical_base)
    if not resource_paths:
        return set(_local_schema_pointer(ref, scopes.anchors, base_uri=scope_uri))

    resource_ref = f"#{fragment}"
    if keyword == "$recursiveRef" and resource_ref == "#":
        # `$recursiveRef` resolves against the dynamic scope, so an ENCLOSING resource that also
        # declares `$recursiveAnchor` can be the target. a sibling embedded resource cannot: it is
        # not on this reference's evaluation path, so its anchor is never in scope. with no anchor
        # reachable at all the target is this resource's own root.
        pointers = frozenset(
            pointer
            for pointer in scopes.anchors.get("", frozenset())
            if scopes.anchor_belongs_to_resource(pointer, canonical_base)
            or path[: len(pointer)] == pointer
        )
        return set(pointers or resource_paths)
    if is_dynamic and resource_ref != "#":
        # union, not replacement: a plain-name fragment also names an ordinary `$anchor`, so
        # dropping the static resolution would leave a same-named `$anchor` target unredacted.
        return set(_local_schema_pointer(resource_ref, scopes.dynamic_anchors)) | {
            pointer
            for pointer in _local_schema_pointer(resource_ref, scopes.anchors)
            if scopes.anchor_belongs_to_resource(pointer, canonical_base)
        }
    pointers = _local_schema_pointer(resource_ref, scopes.anchors)
    if resource_ref == "#" or resource_ref.startswith("#/"):
        return {
            (*resource_path, *pointer) for resource_path in resource_paths for pointer in pointers
        }
    return {
        pointer
        for pointer in pointers
        if scopes.anchor_belongs_to_resource(pointer, canonical_base)
    }


def _resolve_pointer(value: Any, pointer: tuple[str, ...]) -> Any:
    """The node `pointer` names inside `value`, or None when it names nothing."""
    target: Any = value
    for segment in pointer:
        if isinstance(target, dict) and segment in target:
            target = target[segment]
        elif isinstance(target, list | tuple) and segment.isdigit():
            # an index is compared by LENGTH before it is converted. `int()` refuses a digit string
            # past the interpreter's conversion limit (4300 digits by default), and the ValueError
            # escaped post-call sanitization, failing persistence and dropping the completed
            # upstream call from the store. a segment longer than the collection it indexes cannot
            # be in range, so it needs no conversion to be rejected.
            if len(segment) > len(str(len(target))):
                return None
            index = int(segment)
            if index >= len(target):
                return None
            target = target[index]
        else:
            return None
    return target
