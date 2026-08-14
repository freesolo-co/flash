"""Base64 decoding for the credential scan.

A Kubernetes Secret stores every value base64-encoded, and that is an ordinary file to keep beside
an environment -- so a credential can be fully present in a file while sharing no substring with any
pattern. These three functions are what make the encoded form visible.

Split from `flash.env_secrets` to keep that module under the file-size limit. The dependency runs
one way -- nothing here imports the scanning -- so these can be tested on bytes alone.
"""

from __future__ import annotations

import base64
import binascii
import re
from collections.abc import Iterator
from typing import Protocol

from flash.env_formats import _looks_like_zlib
from flash.env_patterns import SHORTEST_TOKEN_BYTES, _match


# What a caller may offer for a second look at decoded bytes: given them, it names a credential or
# returns None. Typed here rather than importing the scan, so the dependency stays one-way.
#
# `whole` says the decode was an exact, aligned, complete one rather than one of the four
# speculative alignments tried inside a window. The caller uses it to decide whether a refusal from
# the decoded bytes is trustworthy enough to propagate.
class _Inspector(Protocol):
    def __call__(self, decoded: bytes, *, whole: bool = False) -> str | None: ...


# A base64 run long enough to hold the shortest credential a pattern admits. The lower bound makes
# the scan walk past ordinary prose rather than decoding every word it meets. There is deliberately
# no upper bound: capping the run split a long encoded file into adjacent pieces, and a credential
# straddling the cut decoded into neither half -- so base64 of a whole `env.sh` published clean.
# Length is instead bounded by `_decode_windows` below, which slides a window over the run.
#
# The URL-safe alphabet (`-` and `_` for `+` and `/`, RFC 4648 section 5) is admitted too. It is
# what `base64.urlsafe_b64encode`, a JWT, and most token-in-a-URL configs emit, and accepting only
# `+/` split such a value at its first `-` so the fragments decoded to neither the whole token nor
# anything matching. The two alphabets are disjoint apart from the shared 62 characters, so one
# pattern covers both and the decode below translates whichever pair is present.
#
# The floor is DERIVED from the shortest credential the patterns admit, not chosen. At a fixed 24
# it sat above the encoding of the shortest Slack token: `xoxb-` plus its 10-character body is 15
# bytes, which encodes to 20 characters, so `eG94Yi1BYkNkRWYwMTIz` in a Kubernetes Secret or any
# other base64 config was never decoded even though the same token in plaintext was caught. Any
# future lowering of a body minimum moves this with it.
_MIN_BASE64_RUN = -(-SHORTEST_TOKEN_BYTES * 4 // 3)  # ceil, unpadded base64 length
_BASE64_RUN = re.compile(rb"[A-Za-z0-9+/\-_]{%d,}={0,2}" % _MIN_BASE64_RUN)

# Maps the URL-safe alphabet onto the standard one so a single decoder handles both. A run mixing
# the two is not valid base64 either way, and translating it simply fails to decode as before.
_URL_SAFE_ALPHABET = bytes.maketrans(b"-_", b"+/")

# A fixed-width wrapped base64 block: one or more full-width lines, then a final line of any
# length. Every line but the last shares one width, which is what leaves ordinary adjacent lines
# alone -- no arbitrary pair of values is welded into a run that decodes to something neither line
# contained. 76 (MIME: `base64.encodebytes`, mail, many `kubectl -o yaml` outputs) and 64 (PEM
# bodies) are the common conventions, but they are not the only ones in use.
#
# ONE full line is enough to qualify, not two. Requiring two meant the commonest shape of all --
# a blob just over the width, so one full line plus a short tail -- never joined, and a key
# straddling its single break decoded into neither side.
#
# `\r?\n` because a Windows checkout or a YAML export wraps with CRLF, and matching only `\n` left
# every CRLF blob unjoined. The `\r` is dropped along with the `\n` when the block is joined.
#
# `[ \t]*` after each break because a wrapped blob is routinely INDENTED: a YAML block scalar, a
# base64 value under a `data:` key, a heredoc inside a function body. Requiring the next line to
# start in column 0 missed 20 of 60 key offsets on a two-space-indented block, at both widths --
# and an indented blob is the commonest way a key is embedded in a config file.
#
# The URL-safe alphabet is admitted here for the same reason `_BASE64_RUN` admits it: `-` and `_`
# are what a JWT, a token-in-a-URL, or `base64.urlsafe_b64encode` emits, and accepting only `+/`
# meant a url-safe blob was never recognised as wrapped. Its lines were then left unjoined and a
# credential straddling a break decoded into neither side -- the exact bypass joining exists to
# close, reachable by encoding with the other alphabet.
#
# Any consistent width is joined, not just 76 and 64. Naming those two covered MIME and PEM and
# nothing else, so a blob wrapped at another column -- `base64 -w 72`, an editor reflow, a generator
# with its own convention -- arrived as independent per-line runs and a credential crossing a break
# decoded into neither side. That is the same bypass joining exists to close, reachable by choosing
# a different column.
# The range of line widths a wrapped base64 block may use. The floor keeps ordinary prose and short
# adjacent values from being welded together -- real wrapping is never narrow -- and the ceiling is
# the widest column any encoder emits before switching to a single unbroken line.
_MIN_WRAP_WIDTH = 32
_MAX_WRAP_WIDTH = 128

#
# Built as an alternation over widths rather than written out, because a regex cannot back-reference
# a LENGTH: `(\w{64})\n\1` would compare the characters, not the column. Enumerating every width in
# the range keeps each alternative exactly as strict as the two hardcoded ones were -- within one
# alternative all lines but the last are the same fixed width -- while covering the columns real
# encoders actually use. Widest first so the longest valid join wins.
_WRAP_WIDTHS = range(_MIN_WRAP_WIDTH, _MAX_WRAP_WIDTH + 1)
_WRAPPED_BLOCK = re.compile(
    b"|".join(
        rb"(?:[A-Za-z0-9+/\-_]{%d}\r?\n[ \t]*)+[A-Za-z0-9+/\-_]{1,%d}={0,2}" % (width, width)
        for width in reversed(_WRAP_WIDTHS)
    )
)
# A necessary condition for the block above: a full-width line of base64 followed by a break. Cheap
# to reject, and it fails on essentially every real file, so the expensive alternation only runs
# where a wrapped block could actually be. Same alphabet AND same minimum width as the block, or
# the guard rejects precisely the blobs the block was widened to join -- it hardcoded 64 while
# the block accepted narrower columns, which made the widening inert for every one of them.
#
# A fixed-width LOOKBEHIND on the break rather than an open-ended run before it. `{32,}` reads the
# same but is quadratic: the engine takes the longest run at every start position and backtracks it
# away one character at a time when no break follows, which is 53 seconds on 100 KB of one long
# non-matching run and hours on the megabyte chunks this actually runs over. The lookbehind asks
# the same question -- are the 32 characters before this break all base64 -- at each break only,
# which is linear and measured within noise of the single fixed-width test it replaced.
_WRAPPED_HINT = re.compile(rb"(?<=[A-Za-z0-9+/\-_]{%d})\r?\n" % _MIN_WRAP_WIDTH)
# The break itself, removed when a block is joined -- with any indent that follows it, or the
# joined run would still carry spaces outside the base64 alphabet. Both endings, so a CRLF blob
# joins into a continuous run rather than one still carrying stray `\r` bytes.
_WRAPPED_BREAK = re.compile(rb"\r?\n[ \t]*")

# How much of one base64 run to decode at a time, and how far the windows overlap. The overlap
# exceeds the encoded length of the longest possible match (4/3 of `_MAX_BODY` plus its prefix), so
# a credential anywhere in a run of any length lands whole inside some window.
_BASE64_WINDOW = 8192
_BASE64_WINDOW_OVERLAP = 1024

# How long a run may be before it is no longer decoded whole for container inspection. A container
# has to be seen entire to be expanded at all, so this is a memory bound rather than a window: 4 MiB
# of base64 decodes to 3 MiB, which the nested-buffer limit already allows a container to expand
# into. Past it the windowed pass still runs, so a literal credential is still found.
_MAX_WHOLE_RUN = 4 << 20


# What a container looks like in the first decoded bytes of a run, and how much of the run to
# decode to find out. The magics are the compressed containers the scan can expand -- gzip, bzip2,
# xz and zip -- and a header sits at the very start, so a short sniff settles it.
#
# zlib is NOT among them because it has no fixed magic: `x\x9c` is only the default level, and
# `zlib.compress(data, 9)` writes `x\xda`. Listing the one literal meant a level-9 stream whose
# base64 crossed a scan chunk was not recognised as a container and published clean, so the
# structural predicate is applied alongside these instead of a byte of it being spelled out here.
_CONTAINER_MAGIC = (b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"PK\x03\x04", b"PK\x05\x06")
_CONTAINER_SNIFF_CHARS = 64


class _RunTooLongToExpand(Exception):
    """A base64 run is longer than `_MAX_WHOLE_RUN`, so the container inside it was never expanded.

    Windowing finds a LITERAL credential in a run of any length, but a container does not survive
    being cut: only the first window carries its header, and every later one starts mid-stream. So
    past this bound the encoded bytes are unexpanded rather than clean, and reporting them clean
    made "make it bigger" a bypass of the whole container path.

    Defined here rather than reusing the scan's refusal so the dependency stays one way: this module
    knows about base64, and the caller is what turns "not expanded" into a refusal.
    """


def _match_base64(
    data: bytes, inspect: _Inspector | None = None, *, truncated: bool = False
) -> str | None:
    """The kind of credential hidden in a base64 run, or None.

    A Kubernetes Secret stores every value base64-encoded, and that is an ordinary file to keep
    beside an environment. The encoded key shares no substring with the plaintext, so none of the
    patterns can see it, and `data: <base64 of an fslo_ key>` published with exit 0.

    Only runs long enough to hold a credential are decoded, and only the base64 alphabet is
    considered, so this walks past prose. The decoded bytes go through `_match` rather than the full
    `_credential_kind`, which keeps the recursion one level deep: base64 of base64 is not a
    convention worth chasing, and unbounded re-decoding is a denial-of-service surface.

    `inspect`, when given, is offered the decoded bytes after `_match` declines. A Kubernetes
    Secret, a cloud-init document and every `kubectl -o yaml` export store their values base64, and
    a gzipped credential inside one decoded successfully here and was then pattern-matched while
    still COMPRESSED -- so `b64encode(gzip.compress(secret))` published clean even though base64 of
    the plaintext and the bare gzip are both caught. Passed in rather than imported so the
    dependency stays one-way: this module knows nothing about containers, only that the caller may
    want a second look.

    `truncated` says that `data` is a CHUNK with more bytes behind it, so a run reaching its end
    was cut rather than ended. Such a run is refused instead of inspected: the windowed pass would
    still find a literal credential in it, but a container inside it can no longer be expanded, and
    reporting that clean made a large enough blob a bypass of the container path.

    A run is decoded in overlapping windows rather than whole, so memory stays bounded on a large
    encoded blob while a credential anywhere in it still lands whole inside some window. Slicing a
    long run into ADJACENT pieces was a bypass: a key straddling the cut decoded into neither half,
    so base64 of an ordinary config file published clean.

    Measured against the false-positive risk before adopting it: 630,011 base64-shaped runs across
    8,769 real hub files decode to zero credential matches, so this costs no legitimate publish.
    """
    joined = _unwrapped(data)
    for run in _BASE64_RUN.finditer(joined):
        candidate = run.group(0)
        # A run touching the end of `data` may have been CUT there rather than ended there. The
        # caller reads a file in bounded chunks, so a long encoded blob arrives in pieces, and a
        # container needs to be seen entire to be expanded at all -- measured: the credential in a
        # gzip whose base64 crossed the 1 MiB chunk boundary was published clean, while the same
        # blob one byte shorter was caught. Only the tail run can be affected, and only when the
        # caller says more bytes follow.
        if (
            truncated
            and inspect is not None
            and run.end() == len(joined)
            and _decodes_to_container(candidate)
        ):
            raise _RunTooLongToExpand
        for window in _decode_windows(candidate):
            # base64 packs 3 bytes per 4 characters, so a run rarely starts on a boundary; all four
            # alignments are tried, each trimmed to a whole number of quartets.
            for start in range(min(len(window), 4)):
                chunk = window[start:]
                # Restore the padding rather than discarding the tail. Trimming to a whole quartet
                # threw away up to three encoded characters, which is up to two decoded bytes off
                # the END of the value -- enough to take a token below its pattern's minimum
                # length, so unpadded base64url of a 20-character `pit_` key published clean.
                # Unpadded output is what `b64encode(...).rstrip("=")`, a JWT segment, and most
                # token-in-a-URL encodings emit, so this is the common case rather than the odd one.
                chunk = _padded(chunk)
                # The same derived floor as the run pattern, not a second hardcoded 24. Lowering
                # only the pattern left this one rejecting exactly the encodings it had started
                # admitting, so the fix would have looked applied while the bypass stayed open.
                if len(chunk) < _MIN_BASE64_RUN:
                    continue
                try:
                    decoded = base64.b64decode(chunk.translate(_URL_SAFE_ALPHABET), validate=True)
                except (ValueError, binascii.Error):
                    continue
                if kind := _match(decoded):
                    return kind
                # `whole` when the run is a DELIMITED VALUE that decodes entirely at its natural
                # alignment: start 0, the whole run in one window, and a delimiter on both sides.
                # Only then is a refusal from the decoded bytes trustworthy enough to propagate.
                #
                # "Decodes exactly" alone is not enough, and testing only that made two real hub
                # datasets unpublishable. A 9.7 MB JSONL of issue text holds base64-shaped runs by
                # chance, and a 196-byte one decoded to bytes beginning `x\x9c` with the FDICT bit
                # set -- so the dictionary refusal fired on prose nobody encoded. Short runs are
                # exactly where chance alignments live, and length cannot separate them: a real
                # `zip -P` archive is 142 bytes, SMALLER than those accidents.
                #
                # Being ASSIGNED is what an encoded value has and a run inside prose does not:
                # `KEY=<base64>`, a JSON or YAML scalar, or a whole `.b64` sidecar. Bounding
                # characters are not enough -- prose is full of them, measured 8,430 such runs in
                # one real dataset -- so a run in a sentence stays speculative and its refusal is
                # swallowed as before.
                exact = (
                    start == 0
                    and window is candidate
                    and len(chunk) == len(_padded(candidate))
                    and _is_assigned_value(joined, run.start(), run.end())
                )
                if inspect is not None and (kind := inspect(decoded, whole=exact)):
                    return kind
        if inspect is not None and (kind := _inspect_whole(candidate, inspect)):
            return kind
    return None


def _decodes_to_container(run: bytes) -> bool:
    """Whether the START of `run` decodes to something carrying a compressed-container signature.

    What makes a cut run unrecoverable is a CONTAINER inside it: the header lives in the first
    window and every later piece starts mid-stream. A run that is merely long loses nothing by
    being cut, since the windowed pass reads a literal credential wherever it lands -- so refusing
    on length alone made an ordinary padded JSON value unpublishable the moment it crossed a chunk
    boundary, which a real test caught.

    Only the head is decoded, at each alignment, since a container declares itself in its first
    bytes. Deliberately narrow: this decides whether to REFUSE, so it answers "is a container
    visibly starting here", never "might these bytes hide one".
    """
    head = run[:_CONTAINER_SNIFF_CHARS]
    for start in range(min(len(head), 4)):
        aligned = head[start:]
        aligned = aligned[: len(aligned) - len(aligned) % 4]
        if len(aligned) < 4:
            continue
        try:
            decoded = base64.b64decode(aligned.translate(_URL_SAFE_ALPHABET), validate=True)
        except (ValueError, binascii.Error):
            continue
        if decoded.startswith(_CONTAINER_MAGIC) or _looks_like_zlib(decoded):
            return True
    return False


def _inspect_whole(run: bytes, inspect: _Inspector) -> str | None:
    """What `inspect` makes of the WHOLE run decoded, for a run too long to fit one window.

    Windowing is what bounds memory, but a container does not survive being cut: only the first
    window decodes to anything with a header on it, and every later window starts mid-stream, so
    the expansion sees a prefix and never reaches the tail. A 13 KB base64 of a gzip therefore
    published clean while the same gzip standing alone was expanded -- the credential lived past
    the first window, which is where a credential in a real encoded blob usually is.

    Skipped when the run already fits one window, since window zero is then the whole run and this
    would decode it a second time for the same answer. Bounded by `_MAX_WHOLE_RUN` so an enormous
    encoded blob cannot be turned into an unbounded buffer by asking for it in one piece.
    """
    if len(run) <= _BASE64_WINDOW:
        return None
    if len(run) > _MAX_WHOLE_RUN:
        # Skipping here reported a container nobody could expand as clean. Measured across 8,944
        # real hub files: the longest base64 run is 11,244 bytes, so no publishable file is near
        # this bound and refusing costs nothing that a real environment does.
        raise _RunTooLongToExpand
    # PADDED to the next multiple of four, not cut back to the previous one. An unpadded encoding
    # is ordinary -- `base64 -w0 | tr -d '='`, a JWT segment, many YAML emitters -- and truncating
    # discards the last one to three characters, which are real bytes at the END of the container:
    # for a zip that is part of the end-of-central-directory record, so the whole-run inspection
    # rejected an archive that decodes perfectly once padded. A remainder of one is not a length
    # base64 can produce, so that case still drops the stray character.
    whole = _padded(run)
    try:
        decoded = base64.b64decode(whole.translate(_URL_SAFE_ALPHABET), validate=True)
    except (ValueError, binascii.Error):
        return None
    return inspect(decoded, whole=True)


def _unwrapped(data: bytes) -> bytes:
    """`data` with the line breaks of a fixed-width base64 block removed, joining it into one run.

    MIME base64 breaks every 76 characters (`base64.encodebytes`, mail attachments, many
    `kubectl get -o yaml` outputs) and PEM bodies every 64. The run pattern stops at the newline,
    so a wrapped blob arrived as a series of per-line runs and a credential straddling a break
    decoded into neither -- measured at 20 of 60 possible key offsets missed, which is every key
    that happens to cross a line.

    Only FIXED-WIDTH sequences are joined, never any break between two base64 characters. That
    distinction matters: `KEY=aGVsbG8\\nOTHER=d29ybGQ` is two unrelated values, and welding those
    together would decode arbitrary line pairs and invent credentials that were never there.
    Wrapping is what the width identifies, and a real wrapped blob always has it.

    Guarded by a cheap substring test first. The joining pattern alternates over two widths and
    retries at every offset, which cost most of the scan's throughput when it ran over every chunk
    of every file; almost no file contains a wrapped block, and one that does always contains a
    full-width run of base64 characters. Measured on ordinary source: 9 MB/s without the guard,
    back to the pre-existing rate with it.
    """
    if b"\n" not in data or not _WRAPPED_HINT.search(data):
        return data
    return _WRAPPED_BLOCK.sub(lambda match: _WRAPPED_BREAK.sub(b"", match.group(0)), data)


# What marks a base64 run as an ASSIGNED value: the run is what something was set to, or the whole
# of a quoted string, or the entire buffer. This is the test for whether a REFUSAL from the decoded
# bytes is trustworthy, so it asks who PUT the bytes there rather than what sits beside them.
#
# Bounding characters alone cannot answer that, and using them made two real hub datasets
# unpublishable. In English a word is bounded by spaces, and `(word)` and `{word}` are bounded too
# -- measured 8,430 "delimited" runs in one 9.7 MB JSONL of issue text, one of which decoded to a
# chance FDICT zlib header and was refused. Prose is full of delimiters; what it does not have is
# an assignment.
_ASSIGNED_VALUE = re.compile(rb"""[=:]\s*["'`]?\Z""")
_QUOTE_CHARACTERS = b"\"'`"


def _is_assigned_value(data: bytes, start: int, end: int) -> bool:
    """Whether the run at `data[start:end]` was assigned or quoted rather than embedded in prose.

    Three shapes count, and they are the ones an encoded container actually arrives in:

      * assigned -- `KEY=<run>`, `data: <run>`, `"key": "<run>"`; the run follows `=` or `:` with
        optional whitespace and an optional opening quote.
      * fully quoted -- the run is the entire contents of a quoted string, which is how a JSON or
        YAML scalar carries one.
      * the whole buffer -- a `.b64` sidecar whose entire content is one encoded blob.

    A run inside a sentence matches none of them, so its refusal stays swallowed as speculative.

    The whole-buffer case ignores a TRAILING NEWLINE, which every text file has and `openssl enc
    -a` writes: requiring nothing at all after the run meant a real encrypted sidecar was treated as
    prose, and the envelope it decoded to published clean while the same bytes inside a YAML value
    were refused. A newline after the final run is the file ending, not a neighbouring token.
    """
    before = data[max(0, start - 64) : start]
    after = data[end : end + 1]
    if not before and after in (b"", b"\n", b"\r"):
        return True
    if _ASSIGNED_VALUE.search(before):
        return True
    return bool(before[-1:] and before[-1:] in _QUOTE_CHARACTERS and before[-1:] == after)


def _padded(run: bytes) -> bytes:
    """`run` with its base64 padding restored, so a whole number of quartets can be decoded.

    Restoring the padding rather than discarding the tail. Trimming to a whole quartet throws away
    up to three encoded characters, which is up to two decoded bytes off the END of the value --
    enough to take a token below its pattern's minimum length, and enough to cut a zip's
    end-of-central-directory record so the archive no longer opens. Unpadded output is what
    `b64encode(...).rstrip("=")`, a JWT segment, and most token-in-a-URL encodings emit, so this is
    the common case rather than the odd one.

    A remainder of ONE is not a length base64 can produce -- 4n+1 characters decode to no whole
    byte count -- so that character is a neighbour rather than part of the value, and dropping it
    is what leaves a decodable run behind.
    """
    if remainder := len(run) % 4:
        return run + b"=" * (4 - remainder) if remainder > 1 else run[:-1]
    return run


def _decode_windows(run: bytes) -> Iterator[bytes]:
    """Overlapping slices of a base64 run, each small enough to decode eagerly."""
    if len(run) <= _BASE64_WINDOW:
        yield run
        return
    step = _BASE64_WINDOW - _BASE64_WINDOW_OVERLAP
    for start in range(0, len(run), step):
        yield run[start : start + _BASE64_WINDOW]
