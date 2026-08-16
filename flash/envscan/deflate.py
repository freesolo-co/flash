"""Expanding DEFLATE payloads that carry no magic of their own.

Two shapes the magic-based recognition in `flash.envscan.formats` cannot reach: a headerless RFC 1951
stream, which has no header at all, and the compressed streams inside a PDF, which begin after an
object header rather than at byte zero. Both hold their credential nowhere a pattern can see, so a
file in either shape published intact while the same payload standing alone was expanded.

Split out to keep both modules under the file-size limit. The dependency runs one way: this knows
about bytes and formats, nothing about files, packages or the scan.
"""

from __future__ import annotations

import base64
import bz2
import io
import itertools
import lzma
import re
import time
import zlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO

from flash.envscan.pdf import pdf_dictionary_spans, pdf_inline_images, pdf_tokens
from flash.envscan.png import _PNG_SIGNATURE, _png_text_payloads

# Where a PDF keeps its compressed content, and how many of those streams are expanded. The
# `/FlateDecode` filter names the encoding, and the `stream` keyword with its mandatory newline
# marks where the zlib record begins.
#
# The gap between them is bounded so the pattern cannot pair a filter name with a `stream` keyword
# arbitrarily far away in a document. The bound was 512 on the reasoning that a real object
# dictionary is short -- a `/Length`, sometimes a `/DecodeParms`, little else. That is a
# convention, not a rule: a dictionary may legally carry any amount of metadata, and 600 bytes of
# it hid the stream entirely while the compact form of the same document was caught.
#
# Distance is now measured but never treated as proof of absence. A filter name the bound does not
# pair is checked again against the dictionary's OWN end -- `>>` then the keyword, at any distance
# -- and a stream found that way is refused rather than reported clean. Anchoring the second look
# on `>>` is what separates an over-long dictionary from prose: a document that merely mentions the
# word `/FlateDecode` has no dictionary to close, so it never reaches a `>> stream` of its own.
#
# The keyword's line ending is CRLF, LF, or a bare CR. The spec names the first two, but real
# producers emit the third and conforming readers accept it -- and matching only `\r?\n` meant a
# `stream\r` document was never enumerated at all, so a key in it published while byte-identical
# LF and CRLF versions were caught.
_PDF_GAP = 512
_PDF_EOL = rb"(?:\r\n|\n|\r)"
# EVERY character of a PDF name may be written as a `#XX` escape, and readers resolve the escaped
# and plain spellings identically -- so the name is matched character by character, each accepting
# either form, rather than as the one literal string. Enumerating spellings does not work: there are
# 4,096 of them for this name alone, and hardcoding the `#44` that a report happened to cite still
# published a key under `/#46lateDecode`, `/FlateDecod#65` and `/#46#6cate#44ecode`.
_FLATE_NAME = b"".join(
    rb"(?:%c|#%02X|#%02x)" % (letter, letter, letter) for letter in b"FlateDecode"
)
_PDF_STREAM = re.compile(rb"/%s\b[\s\S]{0,%d}?\bstream%s" % (_FLATE_NAME, _PDF_GAP, _PDF_EOL))
# the caller buffers at most 64 mib of one pdf, so this reaches every dictionary it can approve.
# a minimum above `_PDF_GAP` finds only streams the ordinary payload walk cannot pair, avoiding the
# previous count of all compact streams and refusing a declaration wherever it sits in the document.
_MAX_PDF_DICTIONARY_GAP = 1 << 26
_PDF_UNREACHED_STREAM = re.compile(
    rb"/%s\b[^<>]{%d,%d}?>>\s*stream%s"
    % (_FLATE_NAME, _PDF_GAP + 1, _MAX_PDF_DICTIONARY_GAP, _PDF_EOL)
)
_MAX_PDF_STREAMS = 4096

# EVERY stream keyword, whatever filter its object declares. The walk above is anchored on
# `/FlateDecode`, so a stream carrying any other filter was never enumerated at all: a conforming
# `/ASCIIHexDecode` stream holding the hex spelling of a key returned clean, while the same key
# behind flate was caught. Decoding every filter PDF defines is a document parser's job; refusing a
# stream whose declared chain this cannot undo is the bounded answer, and it is the same answer the
# flate path already gives for a chain it cannot reverse.
_PDF_ANY_STREAM = re.compile(rb"\bstream%s" % _PDF_EOL)

# The filter list of the object the matched stream belongs to. A PDF may pipe a stream through
# SEVERAL filters -- `/Filter [/ASCII85Decode /FlateDecode]` is what `pdftk` and several writers
# emit -- and they apply in order, so the bytes after `stream` are ASCII85 text rather than the
# zlib record `/FlateDecode` names. Handing those to zlib fails, and the stream was skipped as
# clean while the credential inside it decoded perfectly well.
# What a PDF begins with. Exported so the caller can decline a non-PDF from its first bytes rather
# than reading a whole file to find out here.
_PDF_SIGNATURE = b"%PDF-"

# A PDF comment is a legal token separator. The lexical helper handles comments and escaped names
# directly; these regex separators remain only for the bounded DecodeParms and Predictor searches.
_PDF_SEPARATOR = rb"(?:\s|%[^\r\n]*(?:\r\n|\r|\n))*"
_PDF_REQUIRED_SEPARATOR = rb"(?=[\s%])" + _PDF_SEPARATOR

# The one pre-filter this can undo. ASCII85 is the common companion to FlateDecode and is pure
# syntax, so decoding it needs no parameters. Every other filter is left undone deliberately: a
# chain this cannot fully reverse is refused rather than guessed at, since a stream inspected
# through the wrong decoder is not evidence of anything.
_ASCII85_FILTER = b"ASCII85Decode"
_FLATE_FILTER = b"FlateDecode"
_INLINE_FILTER_ALIASES = {
    b"A85": _ASCII85_FILTER,
    _ASCII85_FILTER: _ASCII85_FILTER,
    b"Fl": _FLATE_FILTER,
    _FLATE_FILTER: _FLATE_FILTER,
}


# the document's encryption dictionary belongs in a classic trailer or a cross-reference stream.
# lexical names are escape-decoded before comparison, while strings and comments are skipped.
def _pdf_has_encryption_dictionary(data: bytes, deadline: float | None) -> bool:
    """Whether a trailer or cross-reference stream dictionary declares `/Encrypt`."""
    frames: list[tuple[bool, list[bytes]]] = []
    trailer = False
    check = _document_checker(deadline)
    for token, _start, _end in pdf_tokens(data, check):
        if token == b"trailer" and not frames:
            trailer = True
            continue
        if token == b"<<":
            frames.append((trailer, []))
            trailer = False
            continue
        if token == b">>":
            if not frames:
                trailer = False
                continue
            is_trailer, direct = frames.pop()
            is_xref = any(
                direct[index : index + 2] == [b"/Type", b"/XRef"]
                for index in range(len(direct) - 1)
            )
            if b"/Encrypt" in direct and (is_trailer or is_xref):
                return True
            continue
        if frames:
            frames[-1][1].append(token)
        elif trailer:
            trailer = False
    return False


# `/DecodeParms` may also point at an indirect object. The predictor and its dimensions then sit
# outside the local stream dictionary, so inflating and scanning the still-predicted bytes is not a
# complete read. The key is escape-tolerant because `/#44ecodeParms` is the same PDF name.
_DECODE_PARMS_NAME = b"".join(
    rb"(?:%c|#%02X|#%02x)" % (letter, letter, letter) for letter in b"DecodeParms"
)
_PDF_INDIRECT_DECODE_PARMS = re.compile(
    rb"/%s%s\d+%s\d+%sR\b"
    % (
        _DECODE_PARMS_NAME,
        _PDF_REQUIRED_SEPARATOR,
        _PDF_REQUIRED_SEPARATOR,
        _PDF_REQUIRED_SEPARATOR,
    )
)

# A predictor declared in `/DecodeParms`. Predictor 1 is the identity and needs no undoing; any
# higher value means the inflated bytes are differences rather than content.
#
# The KEY is spelled escape-tolerantly, like `_PDF_ENCRYPT`. `/#50redictor` is the same name as
# `/Predictor` to every reader, and matching the literal bytes meant a predictor written that way
# named nothing here: the stream was inflated and scanned as content while its bytes were still
# horizontal differences, so a key that a conforming decode reconstructs published intact.
#
# its value is separated with the same grammar as every other PDF token. `%comment\n` is as legal
# there as a space: `/Predictor%comment\n12` reconstructed a key from PNG-Up differences while the
# whitespace spelling was refused, because `\s*` left the predictor undeclared here.
_PREDICTOR_NAME = b"".join(
    rb"(?:%c|#%02X|#%02x)" % (letter, letter, letter) for letter in b"Predictor"
)
_PDF_PREDICTOR = re.compile(rb"/%s%s(\d+)" % (_PREDICTOR_NAME, _PDF_SEPARATOR))


class _UnreadableFilterChain(Exception):
    """A stream is piped through a filter chain this cannot fully undo, so it was never inspected.

    Raised rather than skipped for the same reason every other bound here refuses: the bytes behind
    an unreadable filter are exactly as unverified as an archive that would not expand, and calling
    them clean is the fail-open this scan exists to close.
    """


class _UnreadablePngText(_UnreadableFilterChain):
    """A PNG text chunk declares content that could not be completely decoded."""


class _UnreachedStream(Exception):
    """A PDF object declares `/FlateDecode` but its `stream` sits beyond the dictionary gap.

    Distinct from `_UnreadableFilterChain` so the message stays honest: the filters here are
    perfectly readable, the stream was simply never located. A dictionary may legally carry any
    amount of metadata, and 600 bytes of it put the keyword out of the pattern's reach.
    """


class _EncryptedDocument(Exception):
    """A PDF declares an `/Encrypt` dictionary, so its stream bodies are ciphertext.

    Distinct from `_UnreadableFilterChain` because the filters are not the problem: they would be
    perfectly readable if the bytes underneath them were. Encryption is reversed before any filter
    runs, so zlib sees ciphertext and rejects it, and the skip for "not really a stream" turned an
    encrypted document into a clean result.
    """


class _TooManyStreams(Exception):
    """A PDF carries more compressed streams than `_MAX_PDF_STREAMS`, so it was not fully read.

    Defined here rather than reusing the scan's own refusal so the dependency stays one way: this
    module knows about bytes and formats, and the caller is what turns "not fully read" into a
    refusal. The distinction from the over-budget sentinel is what keeps the message honest --
    the document is not too large to inflate, it has too many streams to walk.
    """


class _DocumentDeadlineExceeded(Exception):
    """A bounded document walk exhausted the caller's shared deadline."""


def _check_document_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() > deadline:
        raise _DocumentDeadlineExceeded


def _document_checker(deadline: float | None) -> Callable[[], None]:
    """A callback lexical helpers can invoke without owning the deadline policy."""

    def check() -> None:
        _check_document_deadline(deadline)

    return check


class _PdfDictionaryIndex:
    """One per-document lexical index of PDF dictionary spans."""

    def __init__(self, data: bytes, deadline: float | None):
        check = _document_checker(deadline)
        self.spans = pdf_dictionary_spans(data, check)

    def span(self, at: int) -> tuple[int, int]:
        containing = [span for span in self.spans if span[0] <= at < span[1]]
        if containing:
            return max(containing, key=lambda span: span[0])
        preceding = [span for span in self.spans if span[1] <= at]
        return max(preceding, key=lambda span: span[1]) if preceding else (at, at)


# What a byte of a gzip name or comment may be. RFC 1952 makes both ISO 8859-1 text, so the C1
# control range is excluded along with the C0 one -- a name is something a person could have typed.
_LATIN1_TEXT = frozenset(range(0x20, 0x7F)) | frozenset(range(0xA0, 0x100))


def _gzip_extra_payloads(extra: bytes) -> Iterator[bytes]:
    """The payload of each complete RFC 1952 extra subfield."""
    at = 0
    while at + 4 <= len(extra):
        size = int.from_bytes(extra[at + 2 : at + 4], "little")
        end = at + 4 + size
        if end > len(extra):
            return
        yield extra[at + 4 : end]
        at = end


def _gzip_header_parts(probe: bytes) -> tuple[tuple[bytes, ...], bytes | None, bytes | None, bool]:
    """The gzip extra payloads, name, comment, and whether the header consumes the probe.

    RFC 1952 puts a fixed 10-byte header first, then four optional fields the flag byte announces: an
    extra field of up to 65,535 bytes, a NUL-terminated name, a NUL-terminated comment, and a 2-byte
    header CRC. A maximum-size extra field alone is longer than the 64 KiB probe the overlay search
    reads, so the deflate bits never came into view and a perfectly valid stream inflated to nothing
    -- which the caller read as "not a stream" and passed over. Undecided is not clean, and here the
    honest answer is that the payload has not been reached yet.

    The test is that the DECLARED fields consume the bytes available, never merely that a short probe
    ran out. Those are different questions, and answering the second accepted 1,638 of 2,000 random
    bodies behind a chance gzip magic: a 64-byte scrap ends inside its own fixed header, so everything
    looked unfinished. The flags and lengths have to claim the boundary.

    Two payload bytes are required because one cannot complete even an empty fixed-Huffman deflate
    block. A 65,523-byte FEXTRA left only one byte in the probe; with or without FHCRC, the valid
    stream behind a shell stub was dismissed while its standalone copy exposed the key.

    The boolean means only "payload not yet judgeable". It is not a claim that the stream is real,
    so a candidate reaching it is expanded and judged there rather than accepted here. Both text
    metadata fields are returned exactly as stored, so published values reach the name scanner too.
    """
    if len(probe) < 12 or not probe.startswith(b"\x1f\x8b\x08"):
        return (), None, None, False
    flags = probe[3]
    # The two reserved bits must be clear: no real header sets them, and requiring that is most of
    # what keeps a chance magic from looking like a header at all.
    if flags & 0b11100000:
        return (), None, None, False
    at = 10
    extras: tuple[bytes, ...] = ()
    name: bytes | None = None
    comment: bytes | None = None
    # The extra field declares a LENGTH, so it is decided by arithmetic. A chance length is small
    # and lands well inside the probe, which is not the condition this exists for.
    if flags & 0b100:
        at = 12 + int.from_bytes(probe[10:12], "little")
        if at >= len(probe):
            return extras, name, comment, True
        extras = tuple(_gzip_extra_payloads(probe[12:at]))
    # FNAME and FCOMMENT are NUL-terminated and unbounded, so a legal one longer than the probe
    # reaches the end with no terminator -- the same "payload not reached yet" the extra field
    # reports by arithmetic. Excluding them meant a valid stream with an 80 KiB name inflated to
    # nothing, was read as "not a stream", and published, while `gzip -dc` recovered the credential
    # from those same bytes.
    #
    # "No terminator in view" alone is not enough, which is why they were excluded before: any run
    # of bytes satisfies it, and a file of adjacent gzip magics is nothing but such runs. RFC 1952
    # makes both fields Latin-1 TEXT, so the unterminated remainder must READ like a name -- which
    # a stream of `1f 8b 08` control bytes does not. Measured over full probes: 0 of 4,000 random
    # bodies behind a chance magic, and the adjacent-magic decoy rejected, while the real long-name
    # stream is caught. Padding with printable bytes defeats the test, but that only buys an
    # expansion attempt which then judges the candidate on what it actually inflates to.
    for present in (0b1000, 0b10000):
        if flags & present:
            if at >= len(probe):
                return extras, name, comment, True
            end = probe.find(b"\0", at)
            if end < 0:
                unterminated = probe[at:]
                return extras, name, comment, all(byte in _LATIN1_TEXT for byte in unterminated)
            if present == 0b1000:
                name = probe[at:end]
            else:
                comment = probe[at:end]
            at = end + 1
    if flags & 0b10:
        at += 2
    return extras, name, comment, len(probe) - at < 2


def _gzip_header_unfinished(probe: bytes) -> bool:
    """Whether a gzip header consumes the probe before two payload bytes are visible."""
    return _gzip_header_parts(probe)[3]


class _GzipHeaderTooLarge(Exception):
    """A gzip header exceeds the bounded metadata window and cannot be approved as scanned."""


_GZIP_HEADER_READ_BYTES = 64 << 10


def _gzip_metadata(source: Path | bytes, limit: int = 4 << 20) -> tuple[bytes, ...]:
    """The exact original filename and comment in a gzip header."""
    source_size = len(source) if isinstance(source, bytes) else source.stat().st_size
    wanted = min(_GZIP_HEADER_READ_BYTES, limit)
    while wanted:
        if isinstance(source, bytes):
            probe = source[:wanted]
        else:
            with source.open("rb") as handle:
                probe = handle.read(wanted)
        extras, name, comment, unfinished = _gzip_header_parts(probe)
        if not unfinished:
            return extras + tuple(part for part in (name, comment) if part is not None)
        if len(probe) >= source_size:
            return ()
        if wanted >= limit:
            raise _GzipHeaderTooLarge
        wanted = min(limit, wanted * 2)
    return ()


class _SingleCompressedReader:
    """A bounded-output reader for one bzip2 or lzma stream that preserves its suffix."""

    def __init__(self, handle: IO[bytes], decoder: bz2.BZ2Decompressor | lzma.LZMADecompressor):
        self._handle = handle
        self._decoder = decoder
        self._unused = b""

    def read(self, size: int = -1) -> bytes:
        if size <= 0:
            return b""
        out = bytearray()
        while len(out) < size and not self._decoder.eof:
            block = b"" if not self._decoder.needs_input else self._handle.read(1 << 20)
            if not block and self._decoder.needs_input:
                raise EOFError("compressed stream ended before its end marker")
            out += self._decoder.decompress(block, size - len(out))
        if self._decoder.eof:
            self._unused = self._decoder.unused_data
        return bytes(out)

    def trailing(self, limit: int) -> bytes | None:
        """Bytes after the first stream, or None when buffering them would exceed `limit`."""
        if len(self._unused) > limit:
            return None
        rest = self._handle.read(limit + 1 - len(self._unused))
        trailing = self._unused + rest
        return None if len(trailing) > limit else trailing


_StreamScanner = Callable[[IO[bytes], float, int], str | None]


def _scan_framed_stream(
    source: Path | bytes,
    *,
    format: str,
    deadline: float,
    depth: int,
    scan: _StreamScanner,
    trailing_limit: int,
) -> tuple[str | None, bytes | None]:
    """Scan one bzip2/xz/lzma-alone stream and return its unconsumed suffix."""
    handle = source.open("rb") if isinstance(source, Path) else io.BytesIO(source)
    try:
        if format == "bz2":
            decoder: bz2.BZ2Decompressor | lzma.LZMADecompressor = bz2.BZ2Decompressor()
        else:
            lzma_format = lzma.FORMAT_ALONE if format == "lzma-alone" else lzma.FORMAT_XZ
            decoder = lzma.LZMADecompressor(format=lzma_format)
        stream = _SingleCompressedReader(handle, decoder)
        if kind := scan(stream, deadline, depth):
            return kind, b""
        return None, stream.trailing(trailing_limit)
    finally:
        handle.close()


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
        # Checked BEFORE the call, because `max_length=0` means unlimited to `decompress` rather
        # than "no output". A block boundary landing exactly on the exhausted budget therefore
        # inflated the next block whole: measured 11,024 bytes returned under a 1,024-byte budget,
        # which is the buffer cap this bound exists to enforce.
        if len(plain) >= budget:
            return None
        try:
            plain += inflate.decompress(block, budget - len(plain))
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


class _TooManyRawDeflateRecords(Exception):
    """A concatenated raw-DEFLATE sequence exceeds its bounded record count."""


def _raw_deflate_records_from(
    blocks: Iterator[bytes], budget: int, record_limit: int
) -> list[bytes] | None:
    """Complete concatenated raw-DEFLATE records under one output and record budget."""
    records: list[bytes] = []
    inflate = zlib.decompressobj(-zlib.MAX_WBITS)
    current = bytearray()
    total = 0
    for block in blocks:
        pending = block
        while pending:
            if len(records) >= record_limit:
                raise _TooManyRawDeflateRecords
            if total + len(current) >= budget:
                return None
            try:
                current += inflate.decompress(pending, budget - total - len(current))
            except zlib.error:
                if records and current:
                    records.append(bytes(current))
                return records
            if inflate.unconsumed_tail:
                return None
            if not inflate.eof:
                break
            pending = inflate.unused_data
            records.append(bytes(current))
            total += len(current)
            current = bytearray()
            inflate = zlib.decompressobj(-zlib.MAX_WBITS)
    if records and current:
        records.append(bytes(current))
    return records


def _filter_value(tokens: list[bytes], at: int) -> tuple[list[bytes], int]:
    """Read one direct scalar or array filter value from lexical tokens."""
    if at >= len(tokens):
        raise _UnreadableFilterChain
    if tokens[at].startswith(b"/"):
        return [tokens[at][1:]], at + 1
    if tokens[at] != b"[":
        raise _UnreadableFilterChain
    names: list[bytes] = []
    at += 1
    while at < len(tokens) and tokens[at] != b"]":
        if not tokens[at].startswith(b"/"):
            raise _UnreadableFilterChain
        names.append(tokens[at][1:])
        at += 1
    if at >= len(tokens) or not names:
        raise _UnreadableFilterChain
    return names, at + 1


def _inline_predictors(tokens: list[bytes], at: int) -> list[int] | None:
    """Direct inline-image decode parameters as one predictor per filter, or defaults."""

    def dictionary(start: int) -> tuple[int, int]:
        depth, predictor, seen = 1, 1, False
        index = start + 1
        while index < len(tokens):
            token = tokens[index]
            if token == b"<<":
                depth += 1
            elif token == b">>":
                depth -= 1
                if not depth:
                    return predictor, index + 1
            elif depth == 1 and token == b"/Predictor":
                if seen or index + 1 >= len(tokens) or not tokens[index + 1].isdigit():
                    raise _UnreadableFilterChain
                if (
                    index + 3 < len(tokens)
                    and tokens[index + 2].isdigit()
                    and tokens[index + 3] == b"R"
                ):
                    raise _UnreadableFilterChain
                predictor, seen = int(tokens[index + 1]), True
                index += 1
            index += 1
        raise _UnreadableFilterChain

    if at >= len(tokens):
        raise _UnreadableFilterChain
    if tokens[at] == b"null":
        return None
    if tokens[at] == b"<<":
        predictor, _after = dictionary(at)
        return [predictor]
    if tokens[at] != b"[":
        raise _UnreadableFilterChain
    predictors: list[int] = []
    at += 1
    while at < len(tokens) and tokens[at] != b"]":
        if tokens[at] == b"null":
            predictors.append(1)
            at += 1
        elif tokens[at] == b"<<":
            predictor, at = dictionary(at)
            predictors.append(predictor)
        else:
            raise _UnreadableFilterChain
    if at >= len(tokens):
        raise _UnreadableFilterChain
    return predictors


def _inline_filters(header: bytes, deadline: float | None) -> list[bytes]:
    """The normalized direct filter chain with readable, identity decode parameters."""
    tokens = [token for token, _start, _end in pdf_tokens(header, _document_checker(deadline))]
    declared: list[bytes] | None = None
    params_at: int | None = None
    depth = 0
    for at, token in enumerate(tokens):
        if token in (b"<<", b"["):
            depth += 1
        elif token in (b">>", b"]"):
            depth = max(0, depth - 1)
        elif depth == 0 and token in (b"/F", b"/Filter"):
            declared, _after = _filter_value(tokens, at + 1)
        elif depth == 0 and token in (b"/DP", b"/DecodeParms"):
            params_at = at + 1
    if declared is None:
        return []
    if any(name not in _INLINE_FILTER_ALIASES for name in declared):
        raise _UnreadableFilterChain
    if params_at is not None:
        predictors = _inline_predictors(tokens, params_at)
        if predictors is not None and (
            len(predictors) != len(declared) or any(predictor != 1 for predictor in predictors)
        ):
            raise _UnreadableFilterChain
    return [_INLINE_FILTER_ALIASES[name] for name in declared]


def _pdf_inline_payloads(
    data: bytes, budget: int, deadline: float | None
) -> Iterator[bytes | None]:
    """Decoded payloads of inline images with supported scalar or array filters."""
    check = _document_checker(deadline)
    images = pdf_inline_images(data, check)
    for count, (header, payload_at) in enumerate(itertools.islice(images, _MAX_PDF_STREAMS)):
        if count % 128 == 0:
            _check_document_deadline(deadline)
        chain = _inline_filters(header, deadline)
        if not chain:
            continue
        payload = data[payload_at:]
        if _FLATE_FILTER in chain:
            flate = chain.index(_FLATE_FILTER)
            payload = _undo_ascii85(payload, chain[:flate])
            inflate = zlib.decompressobj()
            try:
                plain = inflate.decompress(payload, budget)
            except zlib.error:
                continue
            if inflate.unconsumed_tail:
                yield None
                continue
            if not inflate.eof:
                continue
            payload = _undo_ascii85(plain, chain[flate + 1 :])
        else:
            payload = _undo_ascii85(payload, chain)
        yield None if len(payload) > budget else payload
    if next(images, None) is not None:
        raise _TooManyStreams


def _document_payloads(
    data: bytes, budget: int, *, deadline: float | None = None
) -> Iterator[bytes | None]:
    """Decoded compressed text from supported document and image containers."""
    _check_document_deadline(deadline)
    if data.startswith(_PNG_SIGNATURE):
        yield from _png_text_payloads(
            data, budget, _UnreadablePngText, check=_document_checker(deadline)
        )
        return
    yield from _pdf_stream_payloads(data, budget, deadline=deadline)


def _pdf_stream_payloads(
    data: bytes, budget: int, *, deadline: float | None = None
) -> Iterator[bytes | None]:
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

    A stream piped through filters on EITHER side of the flate stage is decoded through them where
    that is possible and refused where it is not. Both orders occur: `/Filter [/ASCII85Decode
    /FlateDecode]` leaves the bytes after `stream` as ASCII85 text, which zlib rejected outright,
    and `/Filter [/FlateDecode /ASCII85Decode]` inflates TO ASCII85 text, which scanned as
    printable noise. Either way the credential decoded perfectly well one filter further in.
    """
    if not data.startswith(_PDF_SIGNATURE):
        return
    _check_document_deadline(deadline)
    dictionaries = _PdfDictionaryIndex(data, deadline)
    # An encrypted document reverses stream encryption BEFORE the declared filters, so what follows
    # `stream` is ciphertext and zlib rejects it -- which the skip below treats as "not really a
    # stream" and the document passes as clean. That made an encrypted PDF the one container shape
    # this let through, while encrypted zip, OpenSSL and OpenPGP payloads are all refused. The
    # passphrase is not ours to have, so the only honest answer is undecided.
    if _pdf_has_encryption_dictionary(data, deadline):
        raise _EncryptedDocument
    # a dictionary too long for the payload walk is undecided, not clean. this searches only gaps
    # beyond that walk's cap, so one match proves a declared stream was never going to be expanded.
    _check_document_deadline(deadline)
    unreached = _PDF_UNREACHED_STREAM.search(data)
    _check_document_deadline(deadline)
    if unreached:
        raise _UnreachedStream
    # Indirect decode parameters hide whether a predictor must be undone. A predictor-encoded key
    # inflated into differences containing no literal credential, so unresolved parameters are
    # unreadable rather than evidence that the stream is clean.
    _check_document_deadline(deadline)
    indirect_decode_parms = _PDF_INDIRECT_DECODE_PARMS.search(data)
    _check_document_deadline(deadline)
    if indirect_decode_parms:
        raise _UnreadableFilterChain
    # A stream whose filters this cannot undo is refused BEFORE the flate walk, so a document
    # mixing one readable stream with one unreadable one does not report the readable verdict and
    # stop. The flate walk below re-reads the same objects; this pass only decides readability.
    _refuse_unreadable_streams(data, dictionaries, deadline)
    streams = _PDF_STREAM.finditer(data)
    for count, found in enumerate(itertools.islice(streams, _MAX_PDF_STREAMS)):
        if count % 128 == 0:
            _check_document_deadline(deadline)
        before, after = _filter_stages(data, found.end(), dictionaries, deadline)
        # A predictor is applied to the INFLATED bytes, so what comes out of zlib is horizontal or
        # PNG differences rather than the stream's contents: a key encoded that way inflates
        # successfully while containing none of its own literal bytes. Undoing it needs the colour
        # and column parameters, so the stream is refused rather than reconstructed and guessed at.
        _check_document_deadline(deadline)
        predictor = _PDF_PREDICTOR.search(_object_dictionary(data, found.end(), dictionaries))
        _check_document_deadline(deadline)
        if predictor and int(predictor.group(1)) > 1:
            raise _UnreadableFilterChain
        body = _undo_ascii85(data[found.end() :], before)
        inflate = zlib.decompressobj()
        try:
            plain = inflate.decompress(body, budget)
        except zlib.error:
            continue
        if inflate.unconsumed_tail:
            yield None
            continue
        decoded = _undo_ascii85(plain, after)
        yield decoded
        yield from _pdf_inline_payloads(decoded, budget, deadline)
    # Stopping at the bound silently reported every later stream as clean, so a document with one
    # more stream than the limit published the credential in it. Undecided is not clean, and every
    # other bound here already refuses rather than truncating -- this one returned a verdict.
    if next(streams, None) is not None:
        raise _TooManyStreams
    yield from _standalone_ascii85_payloads(data, budget, dictionaries, deadline)
    yield from _pdf_inline_payloads(data, budget, deadline)
    _check_document_deadline(deadline)


def _object_dictionary(data: bytes, at: int, dictionaries: _PdfDictionaryIndex) -> bytes:
    """The exact indexed dictionary belonging to the stream at `at`."""
    start, end = dictionaries.span(at)
    return data[start:end]


def _refuse_unreadable_streams(
    data: bytes, dictionaries: _PdfDictionaryIndex, deadline: float | None
) -> None:
    """Raise when any stream in `data` declares a filter chain this cannot reverse.

    The flate walk enumerates only streams naming `/FlateDecode`, which left every other filter
    unexamined rather than undecided: a conforming `/ASCIIHexDecode` stream whose payload is the hex
    spelling of a key returned clean. Reversing each of PDF's filters is a document parser's job, so
    the bounded answer is to refuse a chain that cannot be undone here -- the same answer the flate
    path already gives.

    A stream with NO `/Filter` at all is uncompressed, so its literal bytes are scanned by the
    ordinary pass over the document and need nothing from this.
    """
    streams = _PDF_ANY_STREAM.finditer(data)
    for count, found in enumerate(itertools.islice(streams, _MAX_PDF_STREAMS)):
        if count % 128 == 0:
            _check_document_deadline(deadline)
        chain = _declared_filters(data, found.start(), dictionaries, deadline)
        if any(name not in (_FLATE_FILTER, _ASCII85_FILTER) for name in chain):
            raise _UnreadableFilterChain
    # Stopping at the bound treated every stream past it as readable, which is the same fail-open
    # the flate walk already refuses: 4,096 unfiltered streams followed by an `/ASCIIHexDecode`
    # stream holding a hex-spelled key never reached the filtered one, and the flate walk counts
    # only flate streams so it did not reach it either. Undecided is not clean.
    if next(streams, None) is not None:
        raise _TooManyStreams
    _check_document_deadline(deadline)


def _declared_filters(
    data: bytes, at: int, dictionaries: _PdfDictionaryIndex, deadline: float | None
) -> list[bytes]:
    """The direct scalar or array filter chain in the indexed owning dictionary."""
    dictionary = _object_dictionary(data, at, dictionaries)
    if not dictionary:
        return []
    tokens = [token for token, _start, _end in pdf_tokens(dictionary, _document_checker(deadline))]
    depth = 0
    declared: list[bytes] | None = None
    for index, token in enumerate(tokens):
        if token == b"<<":
            depth += 1
        elif token == b">>":
            depth = max(0, depth - 1)
        elif depth == 1 and token == b"/Filter":
            declared, _after = _filter_value(tokens, index + 1)
    return declared or []


def _filter_stages(
    data: bytes, at: int, dictionaries: _PdfDictionaryIndex, deadline: float | None
) -> tuple[list[bytes], list[bytes]]:
    """The filters the object at `at` applies before and after its flate stage, in order.

    The filter list belongs to the object the stream sits in, so it is read backwards from the
    match rather than forwards: `/Filter` precedes `stream` in the dictionary. Only the entry
    closest behind the match is considered, which is that object's own.
    """
    chain = _declared_filters(data, at, dictionaries, deadline)
    if _FLATE_FILTER not in chain:
        return [], []
    flate = chain.index(_FLATE_FILTER)
    return chain[:flate], chain[flate + 1 :]


def _standalone_ascii85_payloads(
    data: bytes,
    budget: int,
    dictionaries: _PdfDictionaryIndex,
    deadline: float | None,
) -> Iterator[bytes | None]:
    """decoded payloads of streams whose complete filter chain is standalone ascii85."""
    for count, found in enumerate(_PDF_ANY_STREAM.finditer(data)):
        if count % 128 == 0:
            _check_document_deadline(deadline)
        if _declared_filters(data, found.start(), dictionaries, deadline) != [_ASCII85_FILTER]:
            continue
        decoded = _undo_ascii85(data[found.end() :], [_ASCII85_FILTER])
        yield None if len(decoded) > budget else decoded


def _undo_ascii85(payload: bytes, filters: list[bytes]) -> bytes:
    """`payload` with `filters` undone, refusing any chain this cannot read.

    ASCII85 is undone; anything else raises, because a stream that cannot be put into the shape the
    next stage names is one this never inspected, and skipping it is the fail-open the surrounding
    walk exists to close.

    Applied on BOTH sides of the flate stage. Filters after it were originally left alone, on the
    reasoning that the inflated bytes are scanned anyway and a credential is found whatever encoding
    sits on top. That holds for a layer this scans through, and ASCII85 is not one: under
    `/Filter [/FlateDecode /ASCII85Decode]` the inflated bytes are ASCII85 text, so the key's own
    bytes are re-encoded and the scan reads printable noise. Measured: that chain published a key
    that plain `/FlateDecode` caught.
    """
    if not filters:
        return payload
    if filters != [_ASCII85_FILTER]:
        raise _UnreadableFilterChain
    try:
        stripped = payload.lstrip(b" \t\r\n\v\f")
        if stripped.startswith(b"<~"):
            end = stripped.find(b"~>")
            if end < 0:
                raise ValueError
            return base64.a85decode(stripped[: end + 2], adobe=True, ignorechars=b" \t\r\n\v\f")
        return base64.a85decode(payload.split(b"~>")[0], adobe=False, ignorechars=b" \t\r\n\v\f")
    except ValueError as exc:
        raise _UnreadableFilterChain from exc
