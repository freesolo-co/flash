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
_PDF_LONG_DICTIONARY = re.compile(
    rb"/%s\b[^<>]{0,%d}?>>\s*stream%s" % (_FLATE_NAME, 1 << 16, _PDF_EOL)
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

# The dictionary KEY is spelled escape-tolerantly for the same reason its values are: `/#46ilter`
# names `Filter` to every reader, and a literal spelling here would leave a chain declared that way
# invisible -- the stream would then be handed to zlib undecoded, or its unreadable filters missed.
_FILTER_NAME = b"".join(rb"(?:%c|#%02X|#%02x)" % (letter, letter, letter) for letter in b"Filter")
# A PDF comment runs from `%` to the end of its line and is a legal token separator, so
# `/Filter%c\n[/ASCII85Decode /FlateDecode]` declares exactly the chain the spaced form does.
# Accepting whitespace alone left that chain unrecovered: the ASCII85 body went straight to zlib,
# the decompression error was skipped as "not really a stream", and the key inside published. The
# encryption-key pattern already treats a comment as a separator; this is the same rule.
_PDF_SEPARATOR = rb"(?:\s|%[^\r\n]*(?:\r\n|\r|\n))*"
# An indirect reference needs at least one separator between each token. The lookahead supplies that
# requirement while `_PDF_SEPARATOR` consumes comments as well as whitespace.
_PDF_REQUIRED_SEPARATOR = rb"(?=[\s%])" + _PDF_SEPARATOR
# How far back `_dictionary_start` will look for that opening bracket. A document is untrusted
# input: without a bound, one that never opens a dictionary would be walked from every stream in it
# back to byte zero. 64 KiB is far beyond any real object dictionary and still linear per stream.
_MAX_DICTIONARY_REACH = 1 << 16

# The two dictionary brackets, matched together so `_dictionary_start` can count depth. `<<` and
# `>>` are the only tokens that change it; PDF's array brackets are a different pair and do not.
_PDF_BRACKETS = re.compile(rb"<<|>>")

# The array body is bounded by the dictionary reach rather than a short cap. 256 was enough for the
# names a real chain chooses, but the body also carries whatever whitespace and comments sit between
# them: a legal array with a 600-byte gap between `/ASCII85Decode` and `/FlateDecode]` exceeded the
# cap, so no filters were reported at all and zlib was handed ASCII85 text. `[^\]]` cannot cross the
# closing bracket, so widening it reads more of ONE array rather than pairing across objects.
_PDF_FILTERS = re.compile(
    rb"/%s%s(?:/([\w#]+)|\[([^\]]{0,%d})\])" % (_FILTER_NAME, _PDF_SEPARATOR, _MAX_DICTIONARY_REACH)
)
_PDF_FILTER_NAME = re.compile(rb"/([\w#]+)")

# `#` followed by two hex digits inside a PDF name stands for that byte, so `/Flate#44ecode` and
# `/FlateDecode` are the SAME name to every reader -- the escape is spelling, not content. Matching
# the literal bytes meant the escaped spelling named no filter this recognised, the stream was left
# uninflated, and a key inside it published while the plain spelling was caught.
_PDF_NAME_ESCAPE = re.compile(rb"#([0-9A-Fa-f]{2})")


def _pdf_name(raw: bytes) -> bytes:
    """`raw` with its `#XX` escapes resolved, so a name compares by what it MEANS."""
    return _PDF_NAME_ESCAPE.sub(lambda hexed: bytes.fromhex(hexed.group(1).decode()), raw)


# The one pre-filter this can undo. ASCII85 is the common companion to FlateDecode and is pure
# syntax, so decoding it needs no parameters. Every other filter is left undone deliberately: a
# chain this cannot fully reverse is refused rather than guessed at, since a stream inspected
# through the wrong decoder is not evidence of anything.
_ASCII85_FILTER = b"ASCII85Decode"
_FLATE_FILTER = b"FlateDecode"

# The document's encryption dictionary. Present exactly when stream bodies are ciphertext, and
# named in the trailer rather than in any stream's own dictionary, so it is searched document-wide.
#
# A name ENDS at whitespace, at a delimiter, or at the `%` that opens a comment -- all three are
# legal separators, and a reader that sees any of them has read the name `Encrypt`. Requiring
# whitespace alone meant `/Encrypt%c\n2 0 R` named no encryption dictionary here, so the ciphertext
# went to the declared filters, failed to inflate, and was skipped as "not really a stream": the
# document published clean while its streams were, by construction, unreadable to this check.
#
# Written character by character like `_FLATE_NAME` above, and for the same reason -- every one of
# them may be spelled `#XX`, so `/Encryp#74` is the same name to every reader. Matching the escaped
# forms here rather than resolving them across the document keeps stream bytes untouched: a `#` in
# a compressed body means nothing, and rewriting it could invent a name that is not there.
_ENCRYPT_NAME = b"".join(rb"(?:%c|#%02X|#%02x)" % (letter, letter, letter) for letter in b"Encrypt")
_PDF_ENCRYPT = re.compile(rb"/%s(?:[\s/<>\[\]()%%]|$)" % _ENCRYPT_NAME)

# A `/Filter` whose value is an indirect reference (`2 0 R`) rather than a name or an array of
# names. Resolving it means following the xref table into another object. PDF comments are token
# separators too: `%comment\n2 0 R` names the same reference as the spaced form, and the comment form
# previously left the compressed credential unassociated with any filter and published it.
_PDF_INDIRECT_FILTER = re.compile(
    rb"/%s%s\d+%s\d+%sR\b"
    % (_FILTER_NAME, _PDF_REQUIRED_SEPARATOR, _PDF_REQUIRED_SEPARATOR, _PDF_REQUIRED_SEPARATOR)
)

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
_PREDICTOR_NAME = b"".join(
    rb"(?:%c|#%02X|#%02x)" % (letter, letter, letter) for letter in b"Predictor"
)
_PDF_PREDICTOR = re.compile(rb"/%s\s*(\d+)" % _PREDICTOR_NAME)

# How far FORWARD of the filter name the dictionary slice runs. Backwards it runs to the
# dictionary's own `<<` instead, which is the real boundary rather than a guessed distance.
_PDF_DICTIONARY_REACH = 512


class _UnreadableFilterChain(Exception):
    """A stream is piped through a filter chain this cannot fully undo, so it was never inspected.

    Raised rather than skipped for the same reason every other bound here refuses: the bytes behind
    an unreadable filter are exactly as unverified as an archive that would not expand, and calling
    them clean is the fail-open this scan exists to close.
    """


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


# What a byte of a gzip name or comment may be. RFC 1952 makes both ISO 8859-1 text, so the C1
# control range is excluded along with the C0 one -- a name is something a person could have typed.
_LATIN1_TEXT = frozenset(range(0x20, 0x7F)) | frozenset(range(0xA0, 0x100))


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
    # The two reserved bits must be clear: no real header sets them, and requiring that is most of
    # what keeps a chance magic from looking like a header at all.
    if flags & 0b11100000:
        return False
    at = 10
    # The extra field declares a LENGTH, so it is decided by arithmetic. A chance length is small
    # and lands well inside the probe, which is not the condition this exists for.
    if flags & 0b100:
        at = 12 + int.from_bytes(probe[10:12], "little")
        if at > len(probe):
            return True
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
            end = probe.find(b"\0", at)
            if end < 0:
                unterminated = probe[at:]
                return bool(unterminated) and all(byte in _LATIN1_TEXT for byte in unterminated)
            at = end + 1
    return False


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

    A stream piped through filters on EITHER side of the flate stage is decoded through them where
    that is possible and refused where it is not. Both orders occur: `/Filter [/ASCII85Decode
    /FlateDecode]` leaves the bytes after `stream` as ASCII85 text, which zlib rejected outright,
    and `/Filter [/FlateDecode /ASCII85Decode]` inflates TO ASCII85 text, which scanned as
    printable noise. Either way the credential decoded perfectly well one filter further in.
    """
    if not data.startswith(_PDF_SIGNATURE):
        return
    # An encrypted document reverses stream encryption BEFORE the declared filters, so what follows
    # `stream` is ciphertext and zlib rejects it -- which the skip below treats as "not really a
    # stream" and the document passes as clean. That made an encrypted PDF the one container shape
    # this let through, while encrypted zip, OpenSSL and OpenPGP payloads are all refused. The
    # passphrase is not ours to have, so the only honest answer is undecided.
    if _PDF_ENCRYPT.search(data):
        raise _EncryptedDocument
    # A dictionary too long for the gap is undecided, not clean. Raised before the walk so a
    # document carrying one such object refuses whatever its other streams inflate to. Compared
    # against the gap-bounded pattern rather than searched alone: every stream `_PDF_STREAM` pairs
    # is also found here, so only a SURPLUS means one sits beyond the bound.
    if len(_PDF_LONG_DICTIONARY.findall(data)) > len(_PDF_STREAM.findall(data)):
        raise _UnreachedStream
    # A filter named through an INDIRECT reference -- `/Filter 2 0 R`, resolved from another object
    # -- cannot be read by a pattern that matches the name directly, so the stream it belongs to was
    # never associated with flate and its credential published. Resolving object references means
    # parsing the xref table; refusing is the bounded answer, and these are rare in practice.
    if _PDF_INDIRECT_FILTER.search(data):
        raise _UnreadableFilterChain
    # Indirect decode parameters hide whether a predictor must be undone. A predictor-encoded key
    # inflated into differences containing no literal credential, so unresolved parameters are
    # unreadable rather than evidence that the stream is clean.
    if _PDF_INDIRECT_DECODE_PARMS.search(data):
        raise _UnreadableFilterChain
    # A stream whose filters this cannot undo is refused BEFORE the flate walk, so a document
    # mixing one readable stream with one unreadable one does not report the readable verdict and
    # stop. The flate walk below re-reads the same objects; this pass only decides readability.
    _refuse_unreadable_streams(data)
    streams = _PDF_STREAM.finditer(data)
    for found in itertools.islice(streams, _MAX_PDF_STREAMS):
        before, after = _filter_stages(data, found.start())
        # A predictor is applied to the INFLATED bytes, so what comes out of zlib is horizontal or
        # PNG differences rather than the stream's contents: a key encoded that way inflates
        # successfully while containing none of its own literal bytes. Undoing it needs the colour
        # and column parameters, so the stream is refused rather than reconstructed and guessed at.
        predictor = _PDF_PREDICTOR.search(_object_dictionary(data, found.start()))
        if predictor and int(predictor.group(1)) > 1:
            raise _UnreadableFilterChain
        body = _undo_ascii85(data[found.end() :], before)
        inflate = zlib.decompressobj()
        try:
            plain = inflate.decompress(body, budget)
        except zlib.error:
            continue
        yield None if inflate.unconsumed_tail else _undo_ascii85(plain, after)
    # Stopping at the bound silently reported every later stream as clean, so a document with one
    # more stream than the limit published the credential in it. Undecided is not clean, and every
    # other bound here already refuses rather than truncating -- this one returned a verdict.
    if next(streams, None) is not None:
        raise _TooManyStreams


def _dictionary_start(data: bytes, at: int) -> int:
    """Where the dictionary containing `at` opens, or as far back as this is willing to read.

    A dictionary's own `<<` is its boundary, so finding it reads exactly the object rather than a
    guessed number of bytes around it. Searched backwards from `at` and bounded by
    `_MAX_DICTIONARY_REACH`: a document is untrusted input, and an unbounded reverse scan over one
    that never opens a dictionary would walk the whole file for every stream in it.

    Nested dictionaries are why the LAST `<<` is not simply taken: `/DecodeParms << ... >>` opens
    one INSIDE the object, and starting there would cut off the `/Filter` entry written before it.
    Depth is counted instead, so the position returned is the outermost open bracket -- the object's
    own -- and every entry it declares is inside the slice.
    """
    floor = max(0, at - _MAX_DICTIONARY_REACH)
    depth = 0
    for token in reversed([found.start() for found in _PDF_BRACKETS.finditer(data, floor, at)]):
        depth += 1 if data[token : token + 2] == b">>" else -1
        if depth < 0:
            return token
    return floor


def _object_dictionary(data: bytes, at: int) -> bytes:
    """The bytes of the dictionary belonging to the stream whose filter name sits at `at`.

    Read backwards from the match for the same reason `_filter_stages` does: the dictionary
    precedes the `stream` keyword, and a slice starting at the filter name would miss the
    `/DecodeParms` entry when that entry is written before `/Filter`.

    Back to the dictionary's OWN opening `<<`, not a fixed number of bytes. The reach was 512 on the
    reasoning that a real object dictionary is short, which is a convention rather than a rule: a
    legal `/DecodeParms << /Predictor 2 ... >>` written 600 bytes ahead of `/Filter` fell outside
    the window, so the predictor named nothing, the stream was inflated, and horizontal differences
    were scanned as though they were content while a conforming decode reconstructs the key.
    """
    return data[_dictionary_start(data, at) : at + _PDF_DICTIONARY_REACH]


def _refuse_unreadable_streams(data: bytes) -> None:
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
    for found in itertools.islice(streams, _MAX_PDF_STREAMS):
        chain = _declared_filters(data, found.start())
        if any(name not in (_FLATE_FILTER, _ASCII85_FILTER) for name in chain):
            raise _UnreadableFilterChain
    # Stopping at the bound treated every stream past it as readable, which is the same fail-open
    # the flate walk already refuses: 4,096 unfiltered streams followed by an `/ASCIIHexDecode`
    # stream holding a hex-spelled key never reached the filtered one, and the flate walk counts
    # only flate streams so it did not reach it either. Undecided is not clean.
    if next(streams, None) is not None:
        raise _TooManyStreams


def _declared_filters(data: bytes, at: int) -> list[bytes]:
    """The filter names the object owning the stream at `at` declares, escapes resolved.

    Searched from well BEFORE `at`, not from it. `_PDF_STREAM` anchors on the filter NAME, so on a
    chain the match begins in the middle of the array -- at `/FlateDecode]` -- and a slice ending
    there is cut after the opening bracket, leaving `/Filter [` unmatched and the chain invisible.
    Reading from behind the whole dictionary is what makes the array visible; the last entry that
    starts before the stream keyword is the one this object declares.

    Bounded by the dictionary's own `<<` rather than a fixed 512 bytes, for the same reason
    `_object_dictionary` is: a legal array may put any amount of whitespace or comment between
    `/Filter [/ASCII85Decode` and `/FlateDecode]`, and a 600-byte gap put the opening of the
    declaration outside the window -- so no pre-filter was reported and zlib was handed ASCII85
    text, which fails to inflate and reads as "not really a stream".
    """
    start = _dictionary_start(data, at)
    dictionary = data[start : at + _PDF_DICTIONARY_REACH]
    names = None
    for candidate in _PDF_FILTERS.finditer(dictionary):
        if candidate.start() <= at - start:
            names = candidate
    if not names:
        return []
    return [
        _pdf_name(raw)
        for raw in _PDF_FILTER_NAME.findall(names.group(2) or b"/" + (names.group(1) or b""))
    ]


def _filter_stages(data: bytes, at: int) -> tuple[list[bytes], list[bytes]]:
    """The filters the object at `at` applies before and after its flate stage, in order.

    The filter list belongs to the object the stream sits in, so it is read backwards from the
    match rather than forwards: `/Filter` precedes `stream` in the dictionary. Only the entry
    closest behind the match is considered, which is that object's own.
    """
    chain = _declared_filters(data, at)
    if _FLATE_FILTER not in chain:
        return [], []
    flate = chain.index(_FLATE_FILTER)
    return chain[:flate], chain[flate + 1 :]


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
        return base64.a85decode(payload.split(b"~>")[0], adobe=False, ignorechars=b" \t\r\n\v\f")
    except ValueError as exc:
        raise _UnreadableFilterChain from exc
