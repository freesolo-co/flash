"""Locating a JSON Schema target by identity: pointers, resource ids, and anchors.

Redaction has to decide whether a `$ref` names a definition inside THIS payload. That question is
entirely about identity -- base URIs, plain-name anchors, and the two draft-04 spellings of `id` --
and is separable from deciding which literals are secret.
"""

from __future__ import annotations

from collections.abc import Mapping
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


def _schema_resource_id(node: dict[Any, Any]) -> str | None:
    """The resource identifier a schema node declares, in either the modern or draft-04 spelling.

    Draft-04 spells it `id`. Recognizing only `$id` left a legacy definition's resource unknown, so
    a `$ref` to it resolved nowhere, the target was never marked secret, and its `default` stayed
    verbatim in the raw export. A bare `#fragment` id is the draft-04 plain-name anchor form and is
    handled by the anchor collector instead, so it is not a resource here.
    """
    resource_id = node.get("$id")
    if isinstance(resource_id, str):
        return resource_id
    legacy_id = node.get("id")
    if isinstance(legacy_id, str) and not legacy_id.startswith("#"):
        return legacy_id
    return None


def _legacy_id_anchor_name(node: dict[Any, Any]) -> str | None:
    """The plain-name anchor a draft-04 `id` declares, in either of its two spellings.

    Draft-04 writes an anchor as a bare `#name` OR as a relative URI carrying that fragment, as in
    `id: "defs#cred"` -- which sets the base URI to `defs` AND names the anchor `cred`. Accepting
    only the bare form left `$ref: "defs#cred"` resolving to a known resource but an unknown anchor,
    so the target was never marked secret and its `default`/`const` credential stayed in the export.

    A `#/pointer` fragment is a JSON pointer, not a plain name, and is resolved structurally
    instead. `$id` is deliberately excluded: modern schemas spell anchors with `$anchor`, and a
    fragment on `$id` is not an anchor declaration.
    """
    legacy_id = node.get("id")
    if not isinstance(legacy_id, str):
        return None
    _, _, fragment = legacy_id.partition("#")
    if not fragment or fragment.startswith("/"):
        return None
    return fragment


def _schema_resource_pointers(value: Any, *, depth: int = 0) -> dict[str, tuple[str, ...]]:
    resources: dict[str, tuple[str, ...]] = {}

    def collect(node: Any, path: tuple[str, ...], base_uri: str, depth: int) -> None:
        if depth >= platform_traces._MAX_PAYLOAD_DEPTH:
            return
        if isinstance(node, dict):
            resource_id = _schema_resource_id(node)
            if isinstance(resource_id, str):
                base_uri = _safe_urljoin(base_uri, resource_id)
                resources.setdefault(_canonical_resource_uri(_safe_urldefrag(base_uri)[0]), path)
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
            # draft-04 spells a plain-name anchor as `id: "#name"` or as a relative URI carrying
            # that fragment (`id: "defs#cred"`). without both a `$ref` to it resolved to nothing
            # and the legacy target kept its secret literal.
            legacy_anchor = _legacy_id_anchor_name(node)
            if legacy_anchor is not None:
                anchors.setdefault(legacy_anchor, set()).add(path)
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
