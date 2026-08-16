"""Bounded stream buffering and cross-window pairing state for the credential scan.

Split out of `flash.env_secrets` to keep that module under the file-size limit. The dependency runs
one way: nothing here knows about packages, files or the scan orchestration.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator
from pathlib import Path

from flash.env_patterns import _PAIRED_PATTERNS, _RecordHalves, _RecordSplitter

# How much of a stream is read into one scan window.
_SCAN_CHUNK_BYTES = 1 << 20

# How much of a member is inflated to decide whether its zlib header is real before the whole thing
# is held in memory. Large enough that a genuine stream produces symbols to fail on, small enough
# that the chance case costs a page rather than a second copy of a 256 MiB shard.
_ZLIB_PROBE_BYTES = 1 << 16


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


def _zlib_prefix_inflates(source: Path | bytes) -> bool:
    """Whether the first `_ZLIB_PROBE_BYTES` of `source` inflate as a zlib stream.

    Deliberately weaker than "is a zlib stream": a truncated prefix of a real stream ends mid-symbol
    rather than at a record end, so the test is that `decompress` does not raise, not that it
    completes. That is enough to separate a genuine stream from the roughly one file in 2,000 whose
    first two bytes satisfy the header rule by chance.
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

    One splitter shared by every detector, because where a record ends is a property of the stream
    rather than of what is being looked for.
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
    distance within one record and in either order. A JWK written `{"d": ..., <1 MiB of metadata>,
    "kty": "RSA"}` has its private member leave the window before the `kty` arrives, and carrying
    the state is what keeps that key caught however far apart its halves sit.

    Pairing is per RECORD rather than per stream, so unrelated JSONL rows cannot be combined into a
    private key that no row held.

    `overlap` is how much of the NEXT window this one shares with it, so the splitter can rewind to
    that point instead of counting the shared bytes a second time.
    """
    splitter, states = seen
    ends = splitter.ends(window, overlap=overlap)
    for (kind, _), halves in zip(_PAIRED_PATTERNS, states, strict=True):
        # `_RecordHalves` applies `payload_match`, so the captured body goes through the same
        # entropy test the single-buffer path applies.
        if halves.paired(window, ends):
            return kind
    return None
