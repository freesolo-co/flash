"""Expanding DEFLATE payloads that carry no magic of their own.

Two shapes the magic-based recognition in `flash.env_formats` cannot reach: a headerless RFC 1951
stream, which has no header at all, and the compressed streams inside a PDF, which begin after an
object header rather than at byte zero. Both hold their credential nowhere a pattern can see, so a
file in either shape published intact while the same payload standing alone was expanded.

Split out to keep both modules under the file-size limit. The dependency runs one way: this knows
about bytes and formats, nothing about files, packages or the scan.
"""

from __future__ import annotations

import base64
import itertools
import re
import zlib
from collections.abc import Iterator

# Where a PDF keeps its compressed content, and how many of those streams are expanded. The
# `/FlateDecode` filter names the encoding, and the `stream` keyword with its mandatory newline
# marks where the zlib record begins. The gap between them is bounded because a real object
# dictionary is short -- a `/Length`, sometimes a `/DecodeParms`, little else -- and unbounded the
# pattern would pair a filter name with a `stream` keyword arbitrarily far away in a document.
_PDF_STREAM = re.compile(rb"/FlateDecode\b[\s\S]{0,512}?\bstream\r?\n")
_MAX_PDF_STREAMS = 4096

# The filter list of the object the matched stream belongs to. A PDF may pipe a stream through
# SEVERAL filters -- `/Filter [/ASCII85Decode /FlateDecode]` is what `pdftk` and several writers
# emit -- and they apply in order, so the bytes after `stream` are ASCII85 text rather than the
# zlib record `/FlateDecode` names. Handing those to zlib fails, and the stream was skipped as
# clean while the credential inside it decoded perfectly well.
_PDF_FILTERS = re.compile(rb"/Filter\s*(?:/(\w+)|\[([^\]]{0,256})\])")
_PDF_FILTER_NAME = re.compile(rb"/(\w+)")

# The one pre-filter this can undo. ASCII85 is the common companion to FlateDecode and is pure
# syntax, so decoding it needs no parameters. Every other filter is left undone deliberately: a
# chain this cannot fully reverse is refused rather than guessed at, since a stream inspected
# through the wrong decoder is not evidence of anything.
_ASCII85_FILTER = b"ASCII85Decode"
_FLATE_FILTER = b"FlateDecode"


class _UnreadableFilterChain(Exception):
    """A stream is piped through a filter chain this cannot fully undo, so it was never inspected.

    Raised rather than skipped for the same reason every other bound here refuses: the bytes behind
    an unreadable filter are exactly as unverified as an archive that would not expand, and calling
    them clean is the fail-open this scan exists to close.
    """


class _TooManyStreams(Exception):
    """A PDF carries more compressed streams than `_MAX_PDF_STREAMS`, so it was not fully read.

    Defined here rather than reusing the scan's own refusal so the dependency stays one way: this
    module knows about bytes and formats, and the caller is what turns "not fully read" into a
    refusal. The distinction from the over-budget sentinel is what keeps the message honest --
    the document is not too large to inflate, it has too many streams to walk.
    """


def _gzip_header_unfinished(probe: bytes) -> bool:
    """Whether a gzip header declares OPTIONAL FIELDS that run past all of `probe`.

    RFC 1952 puts a fixed 10-byte header first, then up to three optional fields the flag byte
    announces: an extra field of up to 65,535 bytes, a NUL-terminated name, a NUL-terminated
    comment, and a 2-byte header CRC. A maximum-size extra field alone is longer than the 64 KiB
    probe the overlay search reads, so the deflate bits never came into view and a perfectly valid
    stream inflated to nothing -- which the caller read as "not a stream" and passed over. Undecided
    is not clean, and here the honest answer is that the payload has not been reached yet.

    The test is that the DECLARED header outruns the bytes available, never merely that the walk
    reached the end of them. Those are different questions on a short probe, and answering the
    second accepted 1,638 of 2,000 random bodies behind a chance gzip magic: a 64-byte scrap ends
    inside its own fixed header, so everything looked unfinished. The caller's false-positive
    budget is what that would have spent -- every chance magic in a model shard becoming a
    candidate to expand -- so the flags have to claim the length, not the probe merely lack it.

    True means only "header still in progress". It is not a claim that the stream is real, so a
    candidate reaching it is expanded and judged there rather than accepted here.
    """
    if len(probe) < 12 or not probe.startswith(b"\x1f\x8b\x08"):
        return False
    flags = probe[3]
    # Only FEXTRA, and only when the two reserved flag bits are clear. A name or comment is
    # NUL-terminated, so "no terminator in view" is satisfied by any run of random bytes, and
    # accepting that admitted 2 of 500 decoy magics as real streams -- the decoy stream in the
    # overlay test is nothing but gzip magics, so one landing near the end of a probe has a
    # header that genuinely runs past it. An extra field is different: it declares a LENGTH, and
    # the field is the only part of a gzip header that can exceed a 64 KiB probe on its own.
    if flags & 0b11100000 or not flags & 0b100:
        return False
    # The declared field must actually outrun the probe. A chance length is small and lands well
    # inside it, which is not the condition this exists for.
    return 12 + int.from_bytes(probe[10:12], "little") > len(probe)


def _raw_deflate_payload(data: bytes, budget: int) -> bytes | None:
    """What ALL of `data` inflates to when it is one complete raw DEFLATE stream (RFC 1951).

    Empty means "not a raw deflate stream"; None means the stream is real but larger than `budget`,
    which is undecided rather than clean and the caller turns into a refusal.

    Raw deflate is deflate with no header at all -- what `zlib.compressobj(wbits=-15)` writes and
    what a `.deflate` sidecar carries. Having no header is exactly why it slipped through:
    recognition here is magic-based, and the zlib rule needs the two header bytes raw deflate does
    not have, so the compressed bytes were scanned as content and the credential inside published.

    With no magic to key on, the decode IS the recognition, and it has to cover the WHOLE input:
    the stream must inflate, reach its end marker, and leave no trailing bytes. That is what keeps
    this off ordinary binaries -- deflate is self-terminating, so arbitrary bytes end early or run
    out mid-symbol. Measured across 2,000 random buffers at four sizes and all 308 real binaries in
    `/usr/bin`: 0 acceptances. The weaker "inflates something" rule accepted 8 of 2,000, which is
    why completeness rather than output is the test.

    Anchored at offset zero and never searched for: a headerless format cannot be located by
    scanning, and trying every offset in a large file is both unbounded and meaningless.
    """
    return _raw_deflate_from(iter((data,)), budget)


def _raw_deflate_from(blocks: Iterator[bytes], budget: int) -> bytes | None:
    """`_raw_deflate_payload` over a stream of blocks, without holding the source whole.

    Every file that reaches this handler is fed through it, including an ordinary model shard that
    is not deflate at all, so reading the source entire to answer "is this a stream" doubled the
    memory one publish costs -- the request body and the extracted member are both still live. The
    decompressor consumes blocks as they arrive, so probing costs one block plus whatever has
    actually inflated, and a non-stream is rejected on the first block.
    """
    inflate = zlib.decompressobj(-zlib.MAX_WBITS)
    plain, seen = b"", 0
    for block in blocks:
        seen += len(block)
        try:
            plain += inflate.decompress(block, max(0, budget - len(plain)))
        except zlib.error:
            return b""
        if inflate.unconsumed_tail:
            return None
        if inflate.eof:
            # The stream ended inside this block. Anything after it means the input is not ONE
            # complete stream, which is the completeness test that keeps this off ordinary binaries.
            return b"" if inflate.unused_data or any(True for _ in blocks) else plain or b""
    if seen < 2:
        return b""
    return plain if plain and inflate.eof and not inflate.unused_data else b""


def _pdf_stream_payloads(data: bytes, budget: int) -> Iterator[bytes | None]:
    """What each `/FlateDecode` stream in a PDF inflates to, or None for one over `budget`.

    A PDF keeps its content in compressed streams whose zlib record begins after the object header
    rather than at byte zero, so the head-anchored zlib check never saw one and the appended-payload
    search covers only gzip, bzip2 and xz. A credential in a document published intact even though
    the same zlib record standing alone is expanded.

    Located by the format's own grammar -- the `%PDF-` signature, the filter name, and the `stream`
    keyword with its mandatory newline -- rather than by searching for zlib headers. Searching is
    what makes this unaffordable: that rule is about eleven bits, so it trips once per 2 KiB of
    arbitrary data, measured 44,197 candidates across 310 MB of real binaries of which 15 inflated.
    Feeding those through the appended-payload machinery would exhaust its bound and refuse every
    large binary. The grammar costs nothing on a non-PDF, which is every other file in a package.

    A stream that will not inflate is skipped rather than refused: `/FlateDecode` may name an
    encoding this cannot read, and the `stream` keyword appears in ordinary PDF text too.

    A stream piped through EARLIER filters is decoded through them first where that is possible and
    refused where it is not. `/Filter [/ASCII85Decode /FlateDecode]` applies in order, so the bytes
    after `stream` are ASCII85 text; zlib rejected them and the stream was passed over as clean
    while its credential decoded perfectly well one filter further in.
    """
    if not data.startswith(b"%PDF-"):
        return
    streams = _PDF_STREAM.finditer(data)
    for found in itertools.islice(streams, _MAX_PDF_STREAMS):
        body = _after_pre_filters(data, found.start(), data[found.end() :])
        inflate = zlib.decompressobj()
        try:
            plain = inflate.decompress(body, budget)
        except zlib.error:
            continue
        yield None if inflate.unconsumed_tail else plain
    # Stopping at the bound silently reported every later stream as clean, so a document with one
    # more stream than the limit published the credential in it. Undecided is not clean, and every
    # other bound here already refuses rather than truncating -- this one returned a verdict.
    if next(streams, None) is not None:
        raise _TooManyStreams


def _after_pre_filters(data: bytes, at: int, body: bytes) -> bytes:
    """`body` with any filters applied BEFORE `/FlateDecode` undone, ready for zlib.

    The filter list belongs to the object the stream sits in, so it is read backwards from the
    match rather than forwards: `/Filter` precedes `stream` in the dictionary. Only the entry
    closest behind the match is considered, which is that object's own.

    ASCII85 is undone; anything else before the flate stage raises, because a stream that cannot be
    put back into the shape `/FlateDecode` names is one this never inspected, and skipping it is
    the fail-open the surrounding walk exists to close. Filters AFTER the flate stage are not this
    function's business -- the inflated bytes are still scanned, and a credential in them is found
    whatever encoding is layered on top.
    """
    # Searched from well BEFORE the match, not from it. `_PDF_STREAM` anchors on the filter NAME,
    # so on a chain the match begins in the middle of the array -- at `/FlateDecode]` -- and a slice
    # ending there is cut after the opening bracket, leaving `/Filter [` unmatched and the chain
    # invisible. Reading from behind the whole dictionary is what makes the array visible; the last
    # entry that starts before the stream keyword is the one this object declares.
    dictionary = data[max(0, at - 512) : at + 512]
    names = None
    for candidate in _PDF_FILTERS.finditer(dictionary):
        if candidate.start() <= min(at, 512):
            names = candidate
    if not names:
        return body
    chain = _PDF_FILTER_NAME.findall(names.group(2) or b"/" + (names.group(1) or b""))
    before = chain[: chain.index(_FLATE_FILTER)] if _FLATE_FILTER in chain else []
    if not before:
        return body
    if before != [_ASCII85_FILTER]:
        raise _UnreadableFilterChain
    try:
        return base64.a85decode(body.split(b"~>")[0], adobe=False, ignorechars=b" \t\r\n\v\f")
    except ValueError as exc:
        raise _UnreadableFilterChain from exc
