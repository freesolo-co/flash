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
from collections.abc import Callable, Iterator

from flash.env_patterns import SHORTEST_TOKEN_BYTES, _match

# What a caller may offer for a second look at decoded bytes: given them, it names a credential or
# returns None. Typed here rather than importing the scan, so the dependency stays one-way.
_Inspector = Callable[[bytes], str | None]

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
# length. The widths are the two conventions in use -- 76 for MIME (`base64.encodebytes`, mail,
# many `kubectl -o yaml` outputs) and 64 for PEM bodies. Joining ONLY these leaves ordinary
# adjacent lines alone, so no arbitrary pair of values is welded into a run that decodes to
# something neither line contained.
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
_WRAPPED_BLOCK = re.compile(
    rb"(?:[A-Za-z0-9+/\-_]{76}\r?\n[ \t]*)+[A-Za-z0-9+/\-_]{1,76}={0,2}"
    rb"|(?:[A-Za-z0-9+/\-_]{64}\r?\n[ \t]*)+[A-Za-z0-9+/\-_]{1,64}={0,2}"
)
# A necessary condition for the block above: a full-width line of base64 followed by a break. Cheap
# to reject, and it fails on essentially every real file, so the expensive alternation only runs
# where a wrapped block could actually be. Same alphabet as the block, or the guard rejects
# precisely the url-safe blobs the block was widened to join.
_WRAPPED_HINT = re.compile(rb"[A-Za-z0-9+/\-_]{64}\r?\n")
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


def _match_base64(data: bytes, inspect: _Inspector | None = None) -> str | None:
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

    A run is decoded in overlapping windows rather than whole, so memory stays bounded on a large
    encoded blob while a credential anywhere in it still lands whole inside some window. Slicing a
    long run into ADJACENT pieces was a bypass: a key straddling the cut decoded into neither half,
    so base64 of an ordinary config file published clean.

    Measured against the false-positive risk before adopting it: 630,011 base64-shaped runs across
    8,769 real hub files decode to zero credential matches, so this costs no legitimate publish.
    """
    for run in _BASE64_RUN.finditer(_unwrapped(data)):
        candidate = run.group(0)
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
                if remainder := len(chunk) % 4:
                    chunk = chunk + b"=" * (4 - remainder) if remainder > 1 else chunk[:-1]
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
                if inspect is not None and (kind := inspect(decoded)):
                    return kind
        if inspect is not None and (kind := _inspect_whole(candidate, inspect)):
            return kind
    return None


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
    if len(run) <= _BASE64_WINDOW or len(run) > _MAX_WHOLE_RUN:
        return None
    whole = run[: len(run) - len(run) % 4]
    try:
        decoded = base64.b64decode(whole.translate(_URL_SAFE_ALPHABET), validate=True)
    except (ValueError, binascii.Error):
        return None
    return inspect(decoded)


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


def _decode_windows(run: bytes) -> Iterator[bytes]:
    """Overlapping slices of a base64 run, each small enough to decode eagerly."""
    if len(run) <= _BASE64_WINDOW:
        yield run
        return
    step = _BASE64_WINDOW - _BASE64_WINDOW_OVERLAP
    for start in range(0, len(run), step):
        yield run[start : start + _BASE64_WINDOW]
