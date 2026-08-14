"""Byte predicates the credential scan applies to a buffer it already holds.

None of these open a file, expand a container, or call back into the scan: each answers one
question about bytes in hand. Kept apart from `flash.env_secrets` for that reason as much as for
the file-size limit -- the scanning module imports these, never the reverse, so they can be tested
on literal bytes with no archive, no deadline and no recursion.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

from flash.env_formats import _looks_compressed, _looks_like_tar, _overlay_offset
from flash.env_patterns import _PAIRED_PATTERNS

# Read in bounded chunks so a large dataset member is never held in memory whole. This costs no
# more I/O than the publish already pays: `_tar_b64` reads every one of these bytes to gzip them.
_SCAN_CHUNK_BYTES = 1 << 20


def _paired_markers_kind(window: bytes, seen: set[tuple[int, str]]) -> str | None:
    """The kind of two-marker credential whose halves have BOTH appeared by the end of `window`.

    `seen` accumulates across the whole stream, so the halves are paired at any distance and in
    either order. Tracking only one side was wrong for the reverse ordering that JSON permits: a
    JWK written `{"d": ..., <1 MiB of metadata>, "kty": "RSA"}` had its private member leave the
    window before the `kty` arrived, and the key published.

    Marks halves by their index in `_PAIRED_PATTERNS` rather than by the pattern object, so two
    detectors sharing a pattern cannot be confused for each other.
    """
    for index, (kind, detector) in enumerate(_PAIRED_PATTERNS):
        # `payload_match` rather than the raw pattern, so the captured body goes through the same
        # entropy test the single-buffer path applies. Calling `detector.payload` directly here
        # meant the streaming scan -- which is every file over one chunk -- paired placeholders and
        # prose that `_match` had already rejected.
        halves = (("context", detector.context.search), ("payload", detector.payload_match))
        for half, find in halves:
            if (index, half) not in seen and find(window):
                seen.add((index, half))
        if {(index, "context"), (index, "payload")} <= seen:
            return kind
    return None


def _looks_like_container(data: bytes) -> bool:
    """Whether `data` is a container worth reopening, by magic OR by zip structure.

    Nested members were tested on LEADING magic alone while top-level files got `is_zipfile`, so a
    self-extracting zip one layer in -- whose first bytes are `MZ` -- was treated as final content
    and the credential in its deflated payload published. `is_zipfile` scans for the end-of-central
    -directory record, so it recognises a zip behind any preamble; applying it here makes a nested
    member as well covered as the same bytes published directly.

    Gating the recursion on this rather than recursing unconditionally keeps `_MAX_CONTAINER_DEPTH`
    honest: the depth cap raises, so calling it for an ordinary deeply-nested *file* would refuse a
    legitimate publish over nesting that never expanded anything.

    Tar counts here too. A tar's own member bytes are literal, so a top-level one needed no special
    handling for its plain members -- but nested it is a container like any other, and `tar.gz`
    holding a tar of gzipped shards left the innermost key unreached.
    """
    return (
        _looks_compressed(data[:6])
        or _looks_like_tar(data)
        or zipfile.is_zipfile(io.BytesIO(data))
        # A self-extracting SHELL archive, whose stub is a script rather than an executable: none of
        # the tests above sees past it, since each asks what the file BEGINS with and it begins with
        # `#!/bin/sh`. `is_zipfile` covers the same shape for a zip payload; this covers the gzip,
        # bzip2 and xz payloads that `makeself` and `.run` installers actually carry.
        #
        # `False` -- the search gave up with candidates unprobed -- counts as a container too, so
        # the handler runs and turns it into a refusal rather than passing it off as ordinary bytes.
        or _overlay_offset(data) is not None
    )


def _blocks_of(source: Path | bytes) -> Iterator[bytes]:
    """`source` in bounded blocks, so a probe never allocates a second copy of a whole file.

    A path is read incrementally and bytes already in memory are yielded once: re-slicing those
    would allocate the copy this exists to avoid.
    """
    if isinstance(source, bytes):
        yield source
        return
    try:
        with source.open("rb") as handle:
            while block := handle.read(_SCAN_CHUNK_BYTES):
                yield block
    except OSError:
        return
