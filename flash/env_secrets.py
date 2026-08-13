"""Credential content scan for `flash env push`.

Filename filters cannot decide this question. `_ENV_PUSH_SECRET_PATTERNS` drops files *named* like
secret stores (`.env`, `*.pem`, `credentials*`), but the common convention of exporting keys from a
sourceable shell file -- `env.sh`, `setenv.sh`, `secrets.sh` -- is named like ordinary tooling, so a
plain `flash env push .` committed a live `FREESOLO_API_KEY` into the shared environment hub. A
published env repo is org-shared and its history is permanent, so the leak survives deleting the
file afterwards. Python source is exempt from the name filter entirely (so helper modules ship
instead of breaking the worker with ModuleNotFoundError), so a key pasted into a helper had nothing
between it and the hub either.

So the authoritative check reads what is about to be published and refuses on credential *shape*.
Patterns require an issuer prefix plus a long key body: `hf_[A-Za-z0-9]{20,}` is a token, while the
`hf_hub_download` in ordinary code is not, so a real environment still publishes untouched.

Split out of `flash.cli.commands.env.push` to keep that module under the file-size limit, and kept
free of any import from it so the dependency runs one way.
"""

from __future__ import annotations

import base64
import binascii
import bz2
import gzip
import io
import lzma
import os
import re
import tarfile
import time
import zipfile
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import IO

# Bodies are bounded rather than open-ended so a match has a maximum length, which is what lets
# `_SCAN_OVERLAP_BYTES` below be a real guarantee instead of a hope. The cap is far above every
# issued key format; a longer key still matches its first `_MAX_BODY` characters, which is a
# detection either way.
_MAX_BODY = 256

# (kind, pattern) for issued tokens: an issuer prefix plus a long key body, captured as a group.
# The kind names the credential in the refusal so the author knows which key to rotate; the matched
# text is NEVER echoed, since the error is printed and may reach a log.
#
# Patterns are BYTES: members are scanned as raw bytes so a credential stored inside a binary
# container (a sqlite state file, a pickle, an archive) is not skipped. Prefix-anchored patterns
# cannot realistically fire on random bytes -- matching `fslo_` alone is 256**-5 per position.
#
# No AWS entry. An `AKIA...` access key ID is a public identifier -- AWS puts it in signed URLs in
# the clear, so it turns up verbatim in any web-scraped dataset (it does, in the mbti training
# shards here) -- and the matching secret access key is 40 undifferentiated base64 characters with
# no prefix to anchor on. Matching the identifier would block real dataset publishes while still
# not catching the secret that actually matters.
_TOKEN_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("a Freesolo API key", re.compile(rb"fslo_([A-Za-z0-9_-]{16,%d})" % _MAX_BODY)),
    ("a Hugging Face token", re.compile(rb"hf_([A-Za-z0-9]{20,%d})" % _MAX_BODY)),
    (
        "a GitHub token",
        re.compile(
            rb"gh[pousr]_([A-Za-z0-9]{20,%d})|github_pat_([A-Za-z0-9_]{20,%d})"
            % (_MAX_BODY, _MAX_BODY)
        ),
    ),
    ("a Prime Intellect key", re.compile(rb"pit_([A-Za-z0-9]{16,%d})" % _MAX_BODY)),
    ("an Anthropic API key", re.compile(rb"sk-ant-([A-Za-z0-9_-]{20,%d})" % _MAX_BODY)),
    ("an OpenRouter API key", re.compile(rb"sk-or-v1-([A-Za-z0-9]{20,%d})" % _MAX_BODY)),
    # every currently-issued OpenAI family is named explicitly. `sk-svcacct-` and `sk-admin-` keys
    # carry project-wide and organization-wide authority, and neither is reachable through the bare
    # `sk-` branch below: the subtype's own hyphen ends that branch's alphanumeric run early.
    (
        "an OpenAI API key",
        re.compile(
            rb"sk-(?:proj|svcacct|admin)-([A-Za-z0-9_-]{20,%d})"
            # the bare legacy form requires a capital SOMEWHERE in the body, tested by lookahead
            # rather than by position. Demanding 31 more characters *after* the first capital
            # missed real keys: a legacy body carries `T3BlbkFJ` around index 20, leaving too few
            # behind it. The requirement is still what kills the false positive, since a
            # lowercase-hex body of the same length is a content hash, not a key --
            # `.../assets/sk-<32 hex>.js` is an ordinary CDN asset URL.
            rb"|sk-((?=[a-z0-9]*[A-Z])[A-Za-z0-9]{32,%d})" % (_MAX_BODY, _MAX_BODY)
        ),
    ),
    # `xapp-` is Slack's app-level token, a different prefix rather than another `xox` letter.
    ("a Slack token", re.compile(rb"(?:xox[baprs]|xapp)-([A-Za-z0-9-]{10,%d})" % _MAX_BODY)),
)

# Credentials with no issuer prefix, anchored on the ASSIGNMENT that names them instead. Both are
# names this repository already treats as runtime secrets (`WANDB_API_KEY` is the default in
# `flash/client/runtime_secrets.py`, and `AWS_SECRET_ACCESS_KEY` is its documented example), so a
# training environment is exactly the directory where one sits beside the config.
#
# The NAME matches case-insensitively: the same key sits in an env file as `WANDB_API_KEY` and in a
# yaml or python config as `wandb_api_key`, and it is equally live in both. The BODY is what bounds
# the false positives, not the casing of the name.
#
# Bodies are deliberately not pinned to one length. W&B issued 40-hex keys historically and now
# issues much longer ones (the SDK's own `API key must be 40 characters long, yours was 86` error
# is that migration), and both remain live -- a new-format key does not revoke an existing legacy
# one. Matching only 40-hex would have caught the legacy form and published every currently-issued
# key. AWS secret access keys are 40 characters of base64 alphabet with no prefix at all.
#
# The variable name is what makes these bytes a credential: bare 40-hex is a git sha and bare
# 40-base64 is any digest, and refusing those would block ordinary dataset publishes. So the
# assignment is required, and a `${VAR}` or `$(...)` indirection matches nothing because it is not
# a key-shaped body.
#
# These skip `_is_high_entropy`, which exists to spare hand-written placeholders in a body that is
# otherwise unmistakably a key. Here the variable name already carries that meaning, and a hex body
# can legitimately be all-letters (`abcdef...`), which the placeholder heuristic would reject.
_ASSIGNED_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "a Weights & Biases API key",
        re.compile(rb"(?i:wandb_api_key)[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_-]{40,%d})" % _MAX_BODY),
    ),
    (
        "an AWS secret access key",
        re.compile(
            rb"(?i:aws_secret_access_key)[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9/+=]{40})(?![A-Za-z0-9/+=])"
        ),
    ),
)

# A PEM header only means a key is present when the BODY follows it. The header alone is something
# documentation says -- "if you see -----BEGIN RSA PRIVATE KEY----- in a log, redact it" -- and
# refusing on it blocks a legitimate publish over prose about credentials. Requiring a base64 line
# after the header keeps every real key (they all carry one) and drops the mention of one.
#
# The second alternative is the encrypted form, whose body is preceded by RFC 1421 headers instead
# of starting with base64. It names those two headers exactly: a general `[A-Za-z-]+:` also accepts
# `Warning:` or `Note:`, which is prose about a key rather than a key, and reopens the very false
# positive the base64 requirement exists to close.
_LITERAL_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "a private key block",
        re.compile(
            rb"-----BEGIN [A-Z ]*PRIVATE KEY-----[\r\n\s]*"
            rb"(?:[A-Za-z0-9+/=]{32,}|Proc-Type:|DEK-Info:)"
        ),
    ),
)

# Read in bounded chunks so a large dataset member is never held in memory whole. This costs no
# more I/O than the publish already pays: `_tar_b64` reads every one of these bytes to gzip them.
_SCAN_CHUNK_BYTES = 1 << 20
# Carried between chunks so a credential straddling a chunk boundary is still matched. Longer than
# the longest possible match (`_MAX_BODY` plus the longest prefix), so every match is fully visible
# inside some window rather than merely likely to be.
_SCAN_OVERLAP_BYTES = 1024

# Leading bytes of the compressed containers worth expanding. Detected by magic rather than by
# extension, since the extension is the publisher's choice and a renamed archive is still an
# archive.
#
# A plain `.tar` is NOT here and does not need to be: its member bytes appear literally, so the
# ordinary scan already reads them. But a tar whose MEMBER is compressed does need expanding --
# `tar > shard.gz` is as ordinary as `zip > shard.gz`, and only the latter was reached. Tar is
# enumerated by `_credential_in_tar` instead, which is entered on structure rather than on magic
# (a tar's magic sits 257 bytes in, and an uncompressed tar has no leading signature at all).
_COMPRESSED_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00")

# The end-of-central-directory signature, and how much of a stream's tail to keep so it can be
# found. A zip's end record is last in the file, within 64 KiB of the end (the comment field is
# 16-bit), so this window always contains it. Used only to tell "an archive went past the buffer
# cap" from "an ordinary large member did", which decides refuse-versus-pass on content the scan
# could not reopen.
_ZIP_END_RECORD = b"PK\x05\x06"
_ZIP_TAIL_BYTES = (64 << 10) + 64

# A base64 run long enough to hold the shortest credential a pattern admits. The lower bound makes
# the scan walk past ordinary prose rather than decoding every word it meets. There is deliberately
# no upper bound: capping the run split a long encoded file into adjacent pieces, and a credential
# straddling the cut decoded into neither half -- so base64 of a whole `env.sh` published clean.
# Length is instead bounded by `_decode_windows` below, which slides a window over the run.
_BASE64_RUN = re.compile(rb"[A-Za-z0-9+/]{24,}={0,2}")

# How much of one base64 run to decode at a time, and how far the windows overlap. The overlap
# exceeds the encoded length of the longest possible match (4/3 of `_MAX_BODY` plus its prefix), so
# a credential anywhere in a run of any length lands whole inside some window.
_BASE64_WINDOW = 8192
_BASE64_WINDOW_OVERLAP = 1024

# How many container layers deep to expand. A zip holding a gzipped shard is an ordinary way to
# ship a dataset, and stopping at one level meant the inner member's bytes were treated as final
# content -- so a key one layer further in published untouched. The limit still exists because
# each layer multiplies the work a hostile archive can demand.
_MAX_CONTAINER_DEPTH = 4
# A nested container is buffered in memory to be reopened, so its size is capped.
#
# BOTH limits refuse rather than pass when they bite. Returning "nothing found" made the cheapest
# bypass of the entire check "make it expensive": pad an archive past the buffer cap, or bury the
# key one layer past the depth cap, and the scan reported clean. No real environment here comes
# close to either bound, so a refusal means something genuinely unusual is being published.
_MAX_NESTED_BUFFER_BYTES = 64 << 20
# Wall-clock budget for expanding one file's archives. Expansion is unbounded in principle, so it
# needs a stop -- but a stop on BYTES is the wrong one: it discards the tail of the stream, and a
# credential placed after the cutoff then publishes. That is not hypothetical, since the package
# limit bounds *compressed* size: gzip does about 1000:1 on padding, so a 294 KB member expands
# past any byte cap you would plausibly set while staying far under the 256 MB package limit.
#
# So the whole stream is scanned, chunk by chunk with bounded memory, and the budget bounds TIME.
# A pathological archive costs a bounded wait rather than an unbounded one, while every real member
# -- the largest here expands in about 3 seconds -- finishes long inside it.
#
# The budget covers a whole PACKAGE, not one file. Per-file it multiplied: a package may hold 5,000
# members (`ARCHIVE_MEMBER_LIMIT`), so a caller could split compression bombs across them and buy
# 5,000 x 60s of expansion from an authenticated `POST /v1/envs` while every individual file stayed
# inside the apparent one-minute limit.
_MAX_DECOMPRESS_SECONDS = 60.0
# How many members of ONE archive to enumerate. `ZipFile` reads the entire central directory up
# front, so a nested archive of millions of empty entries materialises millions of `ZipInfo`
# objects before any per-member budget is consulted -- and empty members never enter the read loop
# that checks the deadline, so neither bound could stop it. The package extractor counts such an
# archive as a single ordinary file, so this is the only place the inner count is bounded.
_MAX_ARCHIVE_MEMBERS = 100_000

# Everything the standard library raises for an archive it cannot read. There is no single base
# class to catch, and most of these are not OSError, so each omission crashed `flash env push` with
# a traceback on an ordinary corrupt shard: an encrypted member raises RuntimeError, an
# unimplemented compression method NotImplementedError, a corrupt deflate stream zlib.error, and a
# corrupt xz lzma.LZMAError (which inherits straight from Exception).
#
# Shallow corruption is a trap when testing this: truncating near the end of an xz stream raises
# EOFError, which was already caught, so the bug looks absent. The distinct error only appears when
# the damage is deep enough that the decompressor rejects the data rather than running out of it.
_UNREADABLE_ARCHIVE = (
    OSError,
    EOFError,
    ValueError,
    RuntimeError,
    NotImplementedError,
    zipfile.BadZipFile,
    zlib.error,
    lzma.LZMAError,
)


def _is_high_entropy(body: bytes) -> bool:
    """Whether a key body looks issued rather than hand-written.

    An issued token is random over its alphabet, so each rejected shape below is one a real key
    essentially never takes. For a 16-character base62 body -- the shortest any pattern admits --
    the chance of landing in one is about 5 in 10,000,000, and for the 45-character Freesolo bodies
    actually issued it is vanishingly smaller still. The three rejected shapes are exactly the
    placeholder conventions:

      * all lowercase letters, no digit -- `fslo_retry_after_close`, `fslo_your_api_key_here`
      * all capital letters, no digit -- `fslo_YOUR_API_KEY_HERE`, `fslo_REPLACE_ME`
      * one or two distinct characters -- `fslo_XXXXXXXXXXXXXXXX`, a masked value

    Testing only for "a digit or a capital" caught the lowercase convention and missed the other
    two, so `flash env push` refused a scaffolded environment and told the author to rotate a key
    that had never existed. A false refusal is not harmless: it is the failure mode that gets a
    check switched off.

    Mixed-case and digit-bearing bodies are still treated as issued, which keeps a real key whose
    body happens to read like a word. Erring that way is deliberate -- a false refusal is visible
    and recoverable, a missed credential is permanent in a shared repository's history.
    """
    text = body.decode("ascii", "ignore").replace("_", "").replace("-", "")
    if len(set(text)) <= 2:
        return False
    return not (text.isalpha() and (text.isupper() or text.islower()))


def _match(data: bytes) -> str | None:
    """The kind of credential the literal bytes `data` contain, or None."""
    for kind, pattern in _LITERAL_PATTERNS + _ASSIGNED_PATTERNS:
        if pattern.search(data):
            return kind
    for kind, pattern in _TOKEN_PATTERNS:
        for match in pattern.finditer(data):
            # the alternations above put the body in whichever group matched; the rest are None.
            body = next((group for group in match.groups() if group), b"")
            if _is_high_entropy(body):
                return kind
    return None


def _match_base64(data: bytes) -> str | None:
    """The kind of credential hidden in a base64 run, or None.

    A Kubernetes Secret stores every value base64-encoded, and that is an ordinary file to keep
    beside an environment. The encoded key shares no substring with the plaintext, so none of the
    patterns can see it, and `data: <base64 of an fslo_ key>` published with exit 0.

    Only runs long enough to hold a credential are decoded, and only the base64 alphabet is
    considered, so this walks past prose. The decoded bytes go through `_match` rather than the full
    `_credential_kind`, which keeps the recursion one level deep: base64 of base64 is not a
    convention worth chasing, and unbounded re-decoding is a denial-of-service surface.

    A run is decoded in overlapping windows rather than whole, so memory stays bounded on a large
    encoded blob while a credential anywhere in it still lands whole inside some window. Slicing a
    long run into ADJACENT pieces was a bypass: a key straddling the cut decoded into neither half,
    so base64 of an ordinary config file published clean.

    Measured against the false-positive risk before adopting it: 630,011 base64-shaped runs across
    8,769 real hub files decode to zero credential matches, so this costs no legitimate publish.
    """
    for run in _BASE64_RUN.finditer(data):
        candidate = run.group(0)
        for window in _decode_windows(candidate):
            # base64 packs 3 bytes per 4 characters, so a run rarely starts on a boundary; all four
            # alignments are tried, each trimmed to a whole number of quartets.
            for start in range(min(len(window), 4)):
                chunk = window[start:]
                chunk = chunk[: len(chunk) - len(chunk) % 4]
                if len(chunk) < 24:
                    continue
                try:
                    decoded = base64.b64decode(chunk, validate=True)
                except (ValueError, binascii.Error):
                    continue
                if kind := _match(decoded):
                    return kind
    return None


def _decode_windows(run: bytes) -> Iterator[bytes]:
    """Overlapping slices of a base64 run, each small enough to decode eagerly."""
    if len(run) <= _BASE64_WINDOW:
        yield run
        return
    step = _BASE64_WINDOW - _BASE64_WINDOW_OVERLAP
    for start in range(0, len(run), step):
        yield run[start : start + _BASE64_WINDOW]


def _credential_kind(data: bytes) -> str | None:
    """The kind of credential `data` contains under any of its plausible text encodings.

    A wide encoding interleaves NUL bytes between ASCII characters, so a UTF-16 `env.ps1` holding
    an otherwise-detected key matches none of the byte patterns and published intact. Rather than
    decode every member (most are not text at all), strip the NUL padding of each wide form and
    re-test: that is exact for the ASCII-range characters every one of these credentials is built
    from, and costs nothing on the ordinary UTF-8 file, which has no NULs to strip.

    Narrowed text goes through the SAME checks as literal text, base64 included. Running only the
    plain patterns over it left the two supported encodings composable: a UTF-16 config file holding
    a base64 credential passed both gates individually and published.
    """
    if kind := _decoded_kind(data):
        return kind
    if b"\x00" not in data:
        return None
    for width, keep in ((2, (0, 1)), (4, (0, 3))):
        for offset in keep:
            # take every `width`-th byte: for UTF-16 that is the ASCII half of each code unit, in
            # whichever of the two byte orders the file used.
            if kind := _decoded_kind(data[offset::width]):
                return kind
    return None


def _decoded_kind(data: bytes) -> str | None:
    """The kind of credential in `data` literally, or inside a base64 run within it."""
    return _match(data) or _match_base64(data)


class _Unscannable(Exception):
    """A member could not be scanned to the end, so the publish cannot be vouched for.

    Every limit this module imposes -- time, nesting depth, buffer size -- raises this rather than
    returning None. A limit that returns "no credential found" is indistinguishable from a clean
    scan, so the cheapest way past the whole check is to make it expensive: bury the key deeper
    than the depth cap, or behind a member too large to buffer. Refusing keeps the limits honest
    about what they mean, which is "not verified", not "verified clean".
    """


def _scan_stream(handle: IO[bytes], *, deadline: float | None = None, depth: int = 0) -> str | None:
    """The kind of credential anywhere in `handle`, or None.

    The WHOLE stream is read -- memory is bounded by the chunk size, not the total -- with an
    overlap so a credential straddling a chunk boundary is still matched. Stopping early on a byte
    count would mean a key placed after the cutoff publishes, which is the bug rather than the
    protection.

    `deadline` bounds expansion time when the bytes come from an archive. Exceeding it raises
    `_Unscannable` rather than returning None, so the caller refuses the publish.

    A stream that is ITSELF a container is expanded in turn. Nested containers are buffered to be
    reopened, so one too large to hold in memory also raises: leaving it to the scan of its literal
    bytes looked like a reasonable trade, but a deflated member's bytes hold the credential nowhere
    a pattern can see, so it was a silent bypass reachable by padding an archive past the cap.
    """
    carry = b""
    buffered = bytearray()
    tail = b""
    compressed_head = False
    overflowed = False
    while chunk := handle.read(_SCAN_CHUNK_BYTES):
        if deadline is not None and time.monotonic() > deadline:
            raise _Unscannable("takes too long to decompress")
        if not carry and depth:
            compressed_head = _looks_compressed(chunk[:6])
        if depth and not overflowed:
            buffered.extend(chunk)
            if len(buffered) > _MAX_NESTED_BUFFER_BYTES:
                if compressed_head:
                    raise _Unscannable("contains a compressed member too large to inspect")
                # Not a compressed container by its head, so the literal scan below is complete
                # coverage and the buffer is only needed to REOPEN a container. Dropping it keeps
                # memory bounded on an ordinary large member; `tail` still decides at the end
                # whether what went past was a zip hiding behind a preamble.
                overflowed = True
                buffered = bytearray()
        if depth:
            tail = (tail + chunk)[-_ZIP_TAIL_BYTES:]
        window = carry + chunk
        if kind := _credential_kind(window):
            return kind
        carry = window[-_SCAN_OVERLAP_BYTES:]
    if overflowed and _ZIP_END_RECORD in tail:
        raise _Unscannable("contains an archive too large to inspect")
    if buffered and _looks_like_container(bytes(buffered)):
        return _credential_in_container(bytes(buffered), deadline=deadline or 0.0, depth=depth + 1)
    return None


def _looks_compressed(head: bytes) -> bool:
    """Whether `head` begins a compressed container this scan can expand."""
    return head.startswith(_COMPRESSED_MAGIC)


def _looks_like_tar(source: Path | bytes) -> bool:
    """Whether `source` is a tar, by its ustar magic at offset 257.

    A tar has no leading signature -- the first 257 bytes are the first member's name and mode --
    so this cannot be a leading-magic test like every other container here. `tarfile.is_tarfile`
    would be the obvious call, but it accepts COMPRESSED tars too, and those are already routed to
    the right opener above; entering here on a `.tar.gz` would decompress it twice.
    """
    try:
        if isinstance(source, Path):
            with source.open("rb") as handle:
                handle.seek(257)
                magic = handle.read(8)
        else:
            magic = source[257:265]
    except OSError:
        return False
    return magic.startswith((b"ustar\x0000", b"ustar  \x00", b"ustar\x00"))


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
    """
    return _looks_compressed(data[:6]) or zipfile.is_zipfile(io.BytesIO(data))


def _credential_in_container(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential inside a compressed container, or None.

    A deflated zip member or a gzipped shard does not contain its credential anywhere in the file,
    so scanning the container's own bytes cannot see it -- and `flash env push` happily publishes a
    `.zip` holding an `env.sh`.

    Expansion RECURSES to `_MAX_CONTAINER_DEPTH`. Stopping at one level treated an inner member's
    bytes as final content, so a zip holding a gzipped shard -- an ordinary way to ship a dataset --
    hid a key one layer further in and published it.

    A container that will not open is not an error, and neither is one that fails partway through
    reading. An unsupported, truncated or corrupt archive falls back to the literal scan of its own
    bytes, which is the coverage it had before. Crashing on it would be a worse bug than the hole
    being closed: a half-written shard in a dataset directory is ordinary.

    Each zip member is guarded SEPARATELY. Wrapping the whole loop meant one unreadable entry
    abandoned every entry behind it, so a single encrypted member at the top of an archive hid a
    real key further down and the publish succeeded -- the same silent-pass bypass the expansion
    budget refuses, arriving through error handling instead.
    """
    if depth > _MAX_CONTAINER_DEPTH:
        raise _Unscannable("nests compressed containers too deeply to inspect")
    opened: IO[bytes] | None = None
    try:
        # `is_zipfile` scans for the end-of-central-directory record, so it recognises a zip with a
        # preamble -- a self-extracting archive carries an executable stub, and its LEADING bytes
        # are `MZ`, not `PK`. Testing the magic alone left that whole class unexpanded.
        if zipfile.is_zipfile(source if isinstance(source, Path) else io.BytesIO(source)):
            return _credential_in_zip(source, deadline=deadline, depth=depth)
        if _looks_like_tar(source):
            return _credential_in_tar(source, deadline=deadline, depth=depth)
        head = source[:6] if isinstance(source, bytes) else source.open("rb").read(6)
        opener = {b"BZh": bz2.open, b"\xfd7zXZ\x00": lzma.open}.get(
            next((magic for magic in (b"BZh", b"\xfd7zXZ\x00") if head.startswith(magic)), b""),
            gzip.open,
        )
        opened = opener(source if isinstance(source, Path) else io.BytesIO(source), "rb")
        with opened as stream:
            return _scan_stream(stream, deadline=deadline, depth=depth)
    except _UNREADABLE_ARCHIVE:
        return None


def _credential_in_zip(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential in any readable member of a zip, or None."""
    with zipfile.ZipFile(source if isinstance(source, Path) else io.BytesIO(source)) as archive:
        for count, info in enumerate(archive.infolist(), 1):
            if count > _MAX_ARCHIVE_MEMBERS:
                raise _Unscannable("contains an archive with too many members to inspect")
            if time.monotonic() > deadline:
                raise _Unscannable("takes too long to decompress")
            if info.is_dir():
                continue
            try:
                with archive.open(info) as member:
                    if kind := _scan_stream(member, deadline=deadline, depth=depth):
                        return kind
            except _UNREADABLE_ARCHIVE:
                continue  # this member is opaque; the rest of the archive still gets scanned
    return None


def _credential_in_tar(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential in any readable member of a tar, or None.

    A tar is not itself compressed, so its members' bytes are read by the ordinary scan already --
    but a COMPRESSED member inside one is not: `tar > shard.gz` holds the credential nowhere a
    pattern can see, exactly like `zip > shard.gz`, and only the zip form was ever expanded.
    Enumerating members hands each one to `_scan_stream`, which expands it if it is a container.

    Streamed with `r|*` rather than `r:*`: the streaming reader does not seek back over the archive,
    so a member's data is read once, in order. Members are guarded separately for the same reason
    they are in a zip -- one unreadable entry must not abandon the entries behind it.
    """
    handle = source.open("rb") if isinstance(source, Path) else io.BytesIO(source)
    try:
        with tarfile.open(fileobj=handle, mode="r|*") as archive:
            for count, info in enumerate(archive, 1):
                if count > _MAX_ARCHIVE_MEMBERS:
                    raise _Unscannable("contains an archive with too many members to inspect")
                if time.monotonic() > deadline:
                    raise _Unscannable("takes too long to decompress")
                if not info.isfile():
                    continue
                # the member NAME is checked too: a tar entry called `fslo_<key>.json` publishes
                # the key in the archive's listing whatever its contents are.
                if kind := credential_in_name(info.name):
                    return kind
                try:
                    member = archive.extractfile(info)
                    if member is None:
                        continue
                    if kind := _scan_stream(member, deadline=deadline, depth=depth):
                        return kind
                except _UNREADABLE_ARCHIVE:
                    continue
    finally:
        handle.close()
    return None


def credential_in_file(path: Path, *, deadline: float | None = None) -> str | None:
    """The kind of credential publishing `path` would leak, or None.

    Scanned as raw bytes, binary members included. Skipping binaries would be a hole rather than a
    saving: a credential sitting in a sqlite state file or a pickle is as published as one in a
    shell script, and the prefixes above cannot realistically collide with random bytes.

    `deadline` is the expansion budget, shared across a whole package when one is being scanned.
    A per-file budget multiplied by the member limit, which let an authenticated caller split
    compression bombs across thousands of files and buy hours of expansion. Left unset, one file
    gets its own budget, which is the right behaviour for a standalone call.

    Raises `_Unscannable` if an archive is too expensive to finish expanding, which the
    caller turns into a refusal: unverifiable is not the same as clean.
    """
    with path.open("rb") as handle:
        # no deadline on the file's own bytes: that read is bounded by the package size limit
        if kind := _scan_stream(handle):
            return kind
    # `is_zipfile` is consulted inside, so a self-extracting archive is expanded despite its stub
    if deadline is None:
        deadline = time.monotonic() + _MAX_DECOMPRESS_SECONDS
    return _credential_in_container(path, deadline=deadline, depth=1)


def credential_in_name(name: str) -> str | None:
    """The kind of credential the path `name` itself carries, or None.

    A file whose *name* is the key leaks it through the archive's member list even when its
    contents are empty, and the published repo shows that name in its tree forever.
    """
    return _credential_kind(name.encode("utf-8", "surrogateescape"))


def _redacted(name: str) -> str:
    """`name` with any credential body masked, safe to print.

    The refusal names the member so the author can find it, and when the credential is IN that name
    printing it verbatim re-leaks the key into a terminal and whatever collects its output -- the
    one thing the rest of this module is careful never to do. The issuer prefix and the surrounding
    path survive, which is all that is needed to locate the file.
    """

    def _mask(match: re.Match[bytes]) -> bytes:
        body = next((index for index, group in enumerate(match.groups(), 1) if group), None)
        if body is None:
            return match.group(0)
        return match.group(0)[: match.start(body) - match.start()] + b"***"

    masked = name.encode("utf-8", "surrogateescape")
    for _kind, pattern in _TOKEN_PATTERNS + _ASSIGNED_PATTERNS:
        masked = pattern.sub(_mask, masked)
    # A name detected only through base64 has no plaintext body to mask, so masking cannot help:
    # printing any of it prints the encoded key. Withhold the name and give the author the
    # directory instead, which is enough to find a file they just tried to publish.
    if _match_base64(masked):
        parent = name.rsplit("/", 1)[0] if "/" in name else ""
        return (
            f"{parent}/<a file whose name encodes a credential>"
            if parent
            else ("<a file whose name encodes a credential>")
        )
    return masked.decode("utf-8", "replace")


def reject_credential_bearing_package(package_root: Path, *, display: dict[str, str]) -> None:
    """Refuse the publish if any member of the staged package carries a credential.

    Takes the STAGED package rather than the source tree, so what is scanned is exactly what is
    uploaded. Scanning the source instead left three holes: the generated README (which embeds
    `--name`, so a key passed there was published verbatim), the generated entrypoint alias, and
    the window between reading the source and copying it, in which any local process rewriting a
    file put unscanned bytes into the archive.

    Raises ValueError naming the member and the credential kind. Refusing rather than quietly
    dropping the file is deliberate: the author needs to rotate a key that has been sitting in a
    directory they just tried to publish, and a silent drop teaches them nothing.

    One expansion budget covers the whole package. Giving each file its own multiplied it by the
    member limit, so splitting compression bombs across thousands of members bought hours of work
    while every individual file looked well inside the limit.
    """
    deadline = time.monotonic() + _MAX_DECOMPRESS_SECONDS
    # sorted so the member named in the refusal is the same one on every machine.
    for root, dirs, files in os.walk(package_root):
        dirs.sort()
        for name in sorted(dirs) + sorted(files):
            member = Path(root) / name
            relative = member.relative_to(package_root).as_posix()
            shown = _redacted(display.get(relative, relative))
            try:
                # the NAME is checked too, directories included: a file called `fslo_<key>.json`
                # publishes the key in the repository's file tree whatever its contents are.
                kind = credential_in_name(relative) or (
                    credential_in_file(member, deadline=deadline) if member.is_file() else None
                )
            except _Unscannable as exc:
                raise ValueError(
                    f"{shown} {exc}, so it cannot be checked for credentials. Publishing it would "
                    "commit unscanned content to a shared environment repository. Unpack the "
                    "archive before publishing."
                ) from None
            if not kind:
                continue
            raise ValueError(
                f"{shown} contains what looks like {kind}. "
                "Publishing would commit it to a shared environment repository, permanently in "
                "git history. Remove the credential from the environment directory and rotate it "
                "before publishing."
            )
