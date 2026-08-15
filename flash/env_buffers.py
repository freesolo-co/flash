"""Byte predicates the credential scan applies to a buffer it already holds.

None of these open a file, expand a container, or call back into the scan: each answers one
question about bytes in hand. Kept apart from `flash.env_secrets` for that reason as much as for
the file-size limit -- the scanning module imports these, never the reverse, so they can be tested
on literal bytes with no archive, no deadline and no recursion.
"""

from __future__ import annotations

import io
import re
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path

from flash.env_formats import _looks_compressed, _looks_like_tar, _overlay_offset
from flash.env_patterns import (
    _PAIRED_PATTERNS,
    SHORTEST_TOKEN_BYTES,
    _RecordHalves,
    _RecordSplitter,
)

# Read in bounded chunks so a large dataset member is never held in memory whole. This costs no
# more I/O than the publish already pays: `_tar_b64` reads every one of these bytes to gzip them.
_SCAN_CHUNK_BYTES = 1 << 20

# How much of a member is inflated to decide whether its zlib header is real before the whole thing
# is held in memory. Large enough that a genuine stream produces symbols to fail on, small enough
# that the chance case costs a page rather than a second copy of a 256 MiB shard.
_ZLIB_PROBE_BYTES = 1 << 16


def _zlib_prefix_inflates(source: Path | bytes) -> bool:
    """Whether the first `_ZLIB_PROBE_BYTES` of `source` inflate as a zlib stream.

    Deliberately weaker than "is a zlib stream": a truncated prefix of a real stream ends mid-symbol
    rather than at a record end, so the test is that `decompress` does not RAISE, not that it
    completes. That is enough to separate a genuine stream from the roughly one file in 2,000 whose
    first two bytes satisfy the header rule by chance, which is all this decides.
    """
    if isinstance(source, bytes):
        prefix = source[:_ZLIB_PROBE_BYTES]
    else:
        with source.open("rb") as handle:
            prefix = handle.read(_ZLIB_PROBE_BYTES)
    try:
        zlib.decompressobj().decompress(prefix, _ZLIB_PROBE_BYTES)
    except zlib.error:
        return False
    return True


def _paired_state() -> tuple[_RecordSplitter, tuple[_RecordHalves, ...]]:
    """The record boundary tracker and one pairing state per two-marker detector.

    One splitter shared by every detector, because where a record ends is a property of the STREAM
    rather than of what is being looked for. Giving each detector its own repeated the tokenizing
    per detector, which cost more than the pairing it fed.
    """
    return (
        _RecordSplitter(),
        tuple(_RecordHalves(detector) for _, detector in _PAIRED_PATTERNS),
    )


def _paired_markers_kind(
    window: bytes, seen: tuple[_RecordSplitter, tuple[_RecordHalves, ...]], *, overlap: int = 0
) -> str | None:
    """The kind of two-marker credential whose halves share a RECORD, by the end of `window`.

    `seen` carries each detector's pairing state across the whole stream, so halves pair at any
    distance within one record and in either order. Tracking only one side was wrong for the
    reverse ordering that JSON permits: a JWK written `{"d": ..., <1 MiB of metadata>, "kty":
    "RSA"}` had its private member leave the window before the `kty` arrived, and the key
    published. Carrying the state is what keeps that key caught however far apart its halves sit.

    Pairing is per RECORD rather than per stream. Accumulating over the whole file combined
    unrelated JSONL rows -- a public JWK in one and a high-entropy build id under a private member
    name in another -- into a private key that no row held, refusing a legitimate publish.

    One state per detector, positionally, so two detectors sharing a pattern cannot be confused
    for each other.

    `overlap` is how much of the NEXT window this one shares with it, so the splitter can rewind
    to that point instead of counting the shared bytes a second time.
    """
    splitter, states = seen
    ends = splitter.ends(window, overlap=overlap)
    for (kind, _), halves in zip(_PAIRED_PATTERNS, states, strict=True):
        # `_RecordHalves` applies `payload_match`, so the captured body goes through the same
        # entropy test the single-buffer path applies. Calling `detector.payload` directly here
        # meant the streaming scan -- which is every file over one chunk -- paired placeholders and
        # prose that `_match` had already rejected.
        if halves.paired(window, ends):
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


# Marks NUL as 1 and everything else as 0, so an unbroken stretch of padding bytes becomes a run
# `_WIDE_RUN` can find. The length floor is the shortest credential worth decoding; below it a
# chance alignment of NULs cannot carry one anyway.
#
# The floor is the SHORTEST credential any pattern admits, not a round number. At 24 it sat above
# three of them -- `pit_` matches from 20 characters, `fslo_` from 21, `hf_` from 23 -- so a real
# key of any of those lengths was detected as ASCII and missed in its UTF-16 form, which is the
# encoding this narrowing exists to cover. A run must hold the whole credential to decode it.
#
# DERIVED from the shortest token, not written as a number. Lowering the base64 floor for Slack
# left this one at a hardcoded 20, so `xoxb-` plus its 10-character body -- 15 bytes, 15 NUL columns
# in UTF-16 -- was detected as ASCII and missed in the encoding this narrowing exists to cover. The
# two floors answer the same question about the same patterns, so they move together or one of them
# silently stops matching what the other admits.
#
# The NUL-column gate is what keeps an ELF from narrowing into a token, and a shorter run is a
# weaker gate, so the floor is re-measured whenever it moves: at 15 over 500 system binaries, still
# zero false positives.
_NUL_MARKER = bytes(1 if byte == 0 else 0 for byte in range(256))
_WIDE_RUN = re.compile(rb"\x01{%d,}" % SHORTEST_TOKEN_BYTES)


def _wide_runs(data: bytes, width: int, offset: int) -> Iterator[bytes]:
    """The stretches of `data[offset::width]` whose discarded padding byte is NUL throughout.

    Wide text pads every character with NULs, so the padding column is what separates a genuine
    UTF-16/32 region from bytes that merely narrow into something readable. Only runs long enough
    to hold a credential are yielded, which is why an ELF -- whose NULs are scattered rather than
    columnar -- produces none while a wide file produces one run covering the whole of it.
    """
    pad = offset + 1 if offset + 1 < width else offset - 1
    narrow = data[offset::width]
    columns = data[pad::width].translate(_NUL_MARKER)
    for match in _WIDE_RUN.finditer(columns, 0, min(len(narrow), len(columns))):
        yield narrow[match.start() : match.end()]
