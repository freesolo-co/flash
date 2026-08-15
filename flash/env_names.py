"""Exact metadata values carried inside archive and filesystem names.

A slash is part of the base64 alphabet, so the content scanner cannot infer that one separates path
components. A dot is not, but the extension after it stops an encoded run from occupying the whole
value. Names supply both boundaries explicitly, without widening speculative decoding in file data.
"""

from __future__ import annotations

from collections.abc import Iterator


def exact_name_values(name: bytes) -> Iterator[bytes]:
    """`name`, each complete path component, and every complete extension-stripped stem."""
    seen: set[bytes] = set()
    for component in (name, *name.split(b"/")):
        candidate = component
        while candidate:
            if candidate not in seen:
                seen.add(candidate)
                yield candidate
            slash = candidate.rfind(b"/")
            dot = candidate.rfind(b".")
            if dot <= slash + 1 or dot == len(candidate) - 1:
                break
            candidate = candidate[:dot]
