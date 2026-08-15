"""Exact metadata values carried inside archive and filesystem names.

A slash is part of the base64 alphabet, so the content scanner cannot infer that one separates path
components. A dot is not, but the extension after it stops an encoded run from occupying the whole
value. Names supply both boundaries explicitly, without widening speculative decoding in file data.
"""

from __future__ import annotations

from collections.abc import Iterator


def exact_name_values(name: bytes) -> Iterator[bytes]:
    """`name`, each complete path component, and each component without its final extension."""
    seen: set[bytes] = set()
    for component in (name, *name.split(b"/")):
        candidates = [component]
        stem, dot, extension = component.rpartition(b".")
        if dot and stem and extension:
            candidates.append(stem)
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                yield candidate
