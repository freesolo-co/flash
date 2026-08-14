"""Locating a JSON Schema target by identity: pointers, resource ids, and anchors.

Redaction has to decide whether a `$ref` names a definition inside THIS payload. That question is
entirely about identity -- base URIs, plain-name anchors, and the two draft-04 spellings of `id` --
and is separable from deciding which literals are secret.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any
from urllib.parse import unquote

from flash.server.platform import traces as platform_traces
from flash.server.routes.trace_uri import (
    _canonical_resource_uri,
    _safe_urldefrag,
    _safe_urljoin,
)

# a secret literal keyword's VALUE is instance data, never a place a schema identity is declared,
# so the collectors below skip it -- otherwise a credential string shaped like a URI could register
# a bogus resource and steer a `$ref` at the wrong node.
_JSON_SCHEMA_SECRET_LITERAL_KEYWORDS = frozenset(
    {"default", "const", "enum", "examples", "example"}
)


def _bounded(members: Iterable[Any]) -> list[Any]:
    """The leading members of a collection, bounded by the limit storage will apply.

    The wire limit bounds BYTES, so a payload inside 8 MiB can still carry hundreds of thousands of
    compact members -- 120,000 `$defs` each with a distinct `$id` fits in under 5 MiB. Identity
    discovery ran before the walker's own bound and retained a resource-map entry and a path tuple
    for every one of them, and the anchor and scope collectors repeated the same walk, so one
    request cost hundreds of MB. Members past the bound are dropped here for the same reason the
    walker drops them: storage will not keep them either.
    """
    return list(islice(members, platform_traces._MAX_PAYLOAD_COLLECTION))


def _local_schema_pointer(
    ref: str,
    anchors: Mapping[str, frozenset[tuple[str, ...]]],
    *,
    base_uri: str = "",
) -> frozenset[tuple[str, ...]]:
    if not ref.startswith("#"):
        ref_base, fragment = _safe_urldefrag(_safe_urljoin(base_uri, ref))
        document_base = _safe_urldefrag(base_uri)[0]
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


# dialects that spell the resource identifier `id`. draft-06 renamed it to `$id`, and from there
# on a plain `id` is an unknown keyword -- an ordinary annotation, not an identity declaration.
_LEGACY_ID_DIALECT_MARKERS = ("draft-03", "draft-04")


def _declares_legacy_id_dialect(document: Any, *, depth: int = 0) -> bool:
    """Whether a payload's own `$schema` selects a dialect that spells the identifier `id`.

    A modern document may legitimately carry an `id` MEMBER -- `{"properties": {"id": ...}}` is the
    commonest property name there is, and a 2020-12 schema node may annotate itself with one.
    Reading every `id` as draft-04 started a resource scope that the dialect does not declare, which
    moved the base URI and pointed a relative `$ref` at the wrong definition: the real target kept
    its credential-bearing literals while an unrelated sibling was rewritten to "[redacted]".

    The value handed in is often an ENVELOPE rather than the schema -- the resolver runs on the
    payload root, on `tools`, and on each tool entry before it reaches `parameters` -- so the
    declaration is searched for rather than read off the top node. Finding it only at the top left
    every enclosing call reading the legacy dialect, and the outermost of those decided the result.

    Absent or unrecognized `$schema` keeps the legacy reading. A schema that declares no dialect is
    most often legacy, and the failure direction there is redacting a definition that did not need
    it rather than exporting a credential.
    """
    dialect = _declared_dialect(document, depth=depth)
    if dialect is None:
        return True
    return any(marker in dialect for marker in _LEGACY_ID_DIALECT_MARKERS)


def _declared_dialect(value: Any, *, depth: int = 0) -> str | None:
    """The `$schema` a payload declares, at whatever depth the schema document sits."""
    if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
        return None
    if isinstance(value, dict):
        dialect = value.get("$schema")
        if isinstance(dialect, str):
            return dialect
        for key, item in value.items():
            if key in _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS:
                continue
            found = _declared_dialect(item, depth=depth + 1)
            if found is not None:
                return found
    elif isinstance(value, list | tuple):
        for item in value:
            found = _declared_dialect(item, depth=depth + 1)
            if found is not None:
                return found
    return None


def _node_legacy_id_dialect(node: dict[Any, Any], inherited: bool) -> bool:
    """The dialect in force AT this node: its own `$schema` if it declares one, else the inherited.

    An embedded resource may select a different dialect from its parent, so the reading cannot be
    fixed once for the whole payload.
    """
    dialect = node.get("$schema")
    if not isinstance(dialect, str):
        return inherited
    return any(marker in dialect for marker in _LEGACY_ID_DIALECT_MARKERS)


def _schema_resource_id(node: dict[Any, Any], *, legacy_id_dialect: bool = True) -> str | None:
    """The resource identifier a schema node declares, in either the modern or draft-04 spelling.

    Draft-04 spells it `id`. Recognizing only `$id` left a legacy definition's resource unknown, so
    a `$ref` to it resolved nowhere, the target was never marked secret, and its `default` stayed
    verbatim in the raw export. A bare `#fragment` id is the draft-04 plain-name anchor form and is
    handled by the anchor collector instead, so it is not a resource here.

    `legacy_id_dialect` is false when the document's own `$schema` selects a dialect that renamed
    the keyword, where a plain `id` declares no resource at all.
    """
    resource_id = node.get("$id")
    if isinstance(resource_id, str):
        return resource_id
    if not legacy_id_dialect:
        return None
    legacy_id = node.get("id")
    if isinstance(legacy_id, str) and not legacy_id.startswith("#"):
        return legacy_id
    return None


def _legacy_id_anchor_name(node: dict[Any, Any], *, legacy_id_dialect: bool = True) -> str | None:
    """The plain-name anchor a draft-04 `id` declares, in either of its two spellings.

    Draft-04 writes an anchor as a bare `#name` OR as a relative URI carrying that fragment, as in
    `id: "defs#cred"` -- which sets the base URI to `defs` AND names the anchor `cred`. Accepting
    only the bare form left `$ref: "defs#cred"` resolving to a known resource but an unknown anchor,
    so the target was never marked secret and its `default`/`const` credential stayed in the export.

    A `#/pointer` fragment is a JSON pointer, not a plain name, and is resolved structurally
    instead. `$id` is deliberately excluded: modern schemas spell anchors with `$anchor`, and a
    fragment on `$id` is not an anchor declaration.
    """
    if not legacy_id_dialect:
        return None
    legacy_id = node.get("id")
    if not isinstance(legacy_id, str):
        return None
    _, _, fragment = legacy_id.partition("#")
    if not fragment or fragment.startswith("/"):
        return None
    return fragment


def _schema_resource_pointers(
    value: Any, *, depth: int = 0, legacy_id_dialect: bool = True
) -> dict[str, frozenset[tuple[str, ...]]]:
    """Every path declaring each canonical resource id.

    Duplicate ids make a schema ambiguous, but a recorded payload is untrusted input and may carry
    them anyway. Keeping only the first path meant a `$ref` to a secret target resolved to one of
    them, and the OTHER definition -- an equally valid resolution of the same reference -- kept its
    credential-bearing literals. Every declaring path is retained so all of them are redacted.
    """
    resources: dict[str, set[tuple[str, ...]]] = {}

    def collect(
        node: Any, path: tuple[str, ...], base_uri: str, depth: int, legacy_id_dialect: bool
    ) -> None:
        if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
            return
        if isinstance(node, dict):
            # an embedded resource may declare its OWN dialect: a 2020-12 document can hold a
            # draft-04 subschema, where `id` really is the identifier. choosing one dialect for the
            # whole payload left that resource undiscovered, so a `$ref` to it resolved nowhere and
            # its credential-bearing literals stayed in the export.
            legacy_here = _node_legacy_id_dialect(node, legacy_id_dialect)
            resource_id = _schema_resource_id(node, legacy_id_dialect=legacy_here)
            if isinstance(resource_id, str):
                base_uri = _safe_urljoin(base_uri, resource_id)
                canonical = _canonical_resource_uri(_safe_urldefrag(base_uri)[0])
                resources.setdefault(canonical, set()).add(path)
            for key, item in _bounded(node.items()):
                if key not in _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS:
                    collect(item, (*path, str(key)), base_uri, depth + 1, legacy_here)
        elif isinstance(node, list | tuple):
            for index, item in enumerate(_bounded(node)):
                collect(item, (*path, str(index)), base_uri, depth + 1, legacy_id_dialect)

    collect(value, (), "", depth, legacy_id_dialect)
    return {uri: frozenset(paths) for uri, paths in resources.items()}


def _schema_anchor_pointers(
    value: Any, *, depth: int = 0, legacy_id_dialect: bool = True
) -> dict[str, frozenset[tuple[str, ...]]]:
    anchors: dict[str, set[tuple[str, ...]]] = {}
    keywords = ("$anchor", "$dynamicAnchor")

    def collect(node: Any, path: tuple[str, ...], depth: int, legacy_id_dialect: bool) -> None:
        if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
            return
        if isinstance(node, dict):
            legacy_here = _node_legacy_id_dialect(node, legacy_id_dialect)
            for keyword in keywords:
                anchor = node.get(keyword)
                if isinstance(anchor, str):
                    anchors.setdefault(anchor, set()).add(path)
            # draft-04 spells a plain-name anchor as `id: "#name"` or as a relative URI carrying
            # that fragment (`id: "defs#cred"`). without both a `$ref` to it resolved to nothing
            # and the legacy target kept its secret literal.
            legacy_anchor = _legacy_id_anchor_name(node, legacy_id_dialect=legacy_here)
            if legacy_anchor is not None:
                anchors.setdefault(legacy_anchor, set()).add(path)
            if node.get("$recursiveAnchor") is True:
                anchors.setdefault("", set()).add(path)
            for key, item in _bounded(node.items()):
                if key not in _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS:
                    collect(item, (*path, str(key)), depth + 1, legacy_here)
        elif isinstance(node, list | tuple):
            for index, item in enumerate(_bounded(node)):
                collect(item, (*path, str(index)), depth + 1, legacy_id_dialect)

    collect(value, (), depth, legacy_id_dialect)
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
            for key, item in _bounded(node.items()):
                if key not in _JSON_SCHEMA_SECRET_LITERAL_KEYWORDS:
                    collect(item, (*path, str(key)), depth + 1)
        elif isinstance(node, list | tuple):
            for index, item in enumerate(_bounded(node)):
                collect(item, (*path, str(index)), depth + 1)

    collect(value, (), depth)
    return {name: frozenset(paths) for name, paths in anchors.items()}


@dataclass(frozen=True)
class SchemaResourceScopes:
    """Which resource each node in a payload belongs to, and the anchors each resource declares.

    Resolving a `$ref` needs three facts that are all about identity: the base URI in effect at the
    reference, which resource a candidate target sits in, and what anchors exist. Bundling them
    keeps the reference walker from threading four maps through every recursive call.
    """

    base_uri: str
    resources: dict[str, frozenset[tuple[str, ...]]]
    anchors: dict[str, frozenset[tuple[str, ...]]]
    dynamic_anchors: dict[str, frozenset[tuple[str, ...]]]
    _scopes: tuple[tuple[tuple[str, ...], str], ...]

    @classmethod
    def build(cls, value: Any, *, depth: int = 0) -> SchemaResourceScopes:
        legacy_id_dialect = _declares_legacy_id_dialect(value, depth=depth)
        document_id = (
            _schema_resource_id(value, legacy_id_dialect=legacy_id_dialect)
            if isinstance(value, dict)
            else None
        )
        base_uri = document_id if isinstance(document_id, str) else ""
        resources = _schema_resource_pointers(
            value, depth=depth, legacy_id_dialect=legacy_id_dialect
        )
        resources.setdefault(_canonical_resource_uri(_safe_urldefrag(base_uri)[0]), frozenset({()}))
        return cls(
            base_uri=base_uri,
            resources=resources,
            # `$dynamicAnchor` also declares an ordinary plain-name fragment, so a static `$ref`
            # resolves to it as well. one map serves both keywords: splitting them let
            # `{"$ref": "#Name"}` miss a `$dynamicAnchor: "Name"` target and persist its literals.
            anchors=_schema_anchor_pointers(
                value, depth=depth, legacy_id_dialect=legacy_id_dialect
            ),
            dynamic_anchors=_schema_dynamic_anchor_pointers(value, depth=depth),
            _scopes=tuple(
                sorted(
                    ((path, uri) for uri, paths in resources.items() for path in paths),
                    key=lambda item: len(item[0]),
                    reverse=True,
                )
            ),
        )

    def scope_for(self, path: tuple[str, ...]) -> str:
        return next(
            (uri for prefix, uri in self._scopes if path[: len(prefix)] == prefix), self.base_uri
        )

    def anchor_belongs_to_resource(self, pointer: tuple[str, ...], resource_uri: str) -> bool:
        owner_uri = _canonical_resource_uri(_safe_urldefrag(self.scope_for(pointer))[0])
        return owner_uri == _canonical_resource_uri(resource_uri)
