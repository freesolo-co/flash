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
from typing import IO, NoReturn

from flash.env_formats import (
    _MAX_ARCHIVE_MEMBERS,
    _MAX_OPENPGP_MARKERS,
    _UNEXPANDABLE_MAGIC,
    _ZIP_TAIL_BYTES,
    _after_skippable_frames,
    _has_zip_end_record,
    _is_openpgp_secret_key,
    _jks_private_key_entries,
    _looks_compressed,
    _looks_like_tar,
    _looks_like_textual,
    _looks_like_zlib,
    _zip_member_count,
)
from flash.env_patterns import (
    _ASSIGNED_PATTERNS,
    _LITERAL_PATTERNS,
    _MAX_BODY,
    _PAIRED_PATTERNS,
    _TOKEN_PATTERNS,
    SHORTEST_TOKEN_BYTES,
    _match,
)

# Read in bounded chunks so a large dataset member is never held in memory whole. This costs no
# more I/O than the publish already pays: `_tar_b64` reads every one of these bytes to gzip them.
_SCAN_CHUNK_BYTES = 1 << 20
# Carried between chunks so a credential straddling a chunk boundary is still matched. Derived from
# `_MAX_BODY` rather than written as a bare number, so the two cannot drift apart: the overlap must
# exceed the longest possible match (a body plus its prefix and quoting) or a credential landing on
# a boundary is fully visible in no window at all.
_SCAN_OVERLAP_BYTES = _MAX_BODY * 4

# How much of a chunk's head is walked for skippable frames. Generous for the handful a real
# seekable stream carries, and bounded so a chain of crafted frame headers cannot make this walk a
# cost of its own.
_SKIPPABLE_SCAN_BYTES = 64 << 10
# How much of a member's head the OpenPGP secret-key test reads. The test itself needs about a
# dozen bytes, but a legal marker packet may precede the real one and each consumes five, so a
# fixed 24 left too few behind four markers to reach the version and algorithm fields.
_OPENPGP_HEAD_BYTES = 24 + 5 * _MAX_OPENPGP_MARKERS

# How many concatenated zlib records are inflated before the stream is refused. A per-record cache
# or an appended log writes a handful; the bound is what stops a file of many tiny records from
# becoming an expansion cost of its own, and exceeding it refuses rather than passes.
_MAX_ZLIB_RECORDS = 64

# How long a signature must be to be searched for at an ARBITRARY offset rather than only at the
# start of a stream. Six bytes is where the two real self-extracting formats sit (7-Zip at six, RAR
# at seven and eight) and where a chance occurrence stops being plausible: measured 0 hits across
# 256 MiB of random bytes for every signature, but a 4-byte magic is only 1 in 4 billion per
# position, which a large enough model shard reaches. The short zstd and LZ4 magics stay decisive
# at offset zero, where they mean what they say.
_SFX_MAGIC_BYTES = 6

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

# Marks NUL as 1 and everything else as 0, so an unbroken stretch of padding bytes becomes a run
# `_WIDE_RUN` can find. The length floor is the shortest credential worth decoding; below it a
# chance alignment of NULs cannot carry one anyway.
#
# The floor is the SHORTEST credential any pattern admits, not a round number. At 24 it sat above
# three of them -- `pit_` matches from 20 characters, `fslo_` from 21, `hf_` from 23 -- so a real
# key of any of those lengths was detected as ASCII and missed in its UTF-16 form, which is the
# encoding this narrowing exists to cover. A run must hold the whole credential to decode it.
#
# Lowered to exactly 20 rather than further, because every character of slack admits more machine
# code: the NUL-column gate is what keeps an ELF from narrowing into a token, and a shorter run is
# a weaker gate. Re-measured at 20 over 500 system binaries -- still zero false positives.
_NUL_MARKER = bytes(1 if byte == 0 else 0 for byte in range(256))
_WIDE_RUN = re.compile(rb"\x01{20,}")

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
# How many members of ONE archive to enumerate, imported above. `ZipFile` reads the entire central
# directory up front, so a nested archive of millions of empty entries materialises millions of
# `ZipInfo` objects before any per-member budget is consulted -- and empty members never enter the
# read loop that checks the deadline, so neither bound could stop it. The package extractor counts
# such an archive as a single ordinary file, so this is the only place the inner count is bounded.
#
# Every read of it is in THIS module, and the directory walk takes it as an argument rather than
# reading it from its own, so rebinding it here still governs the whole check.


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
    # `TarError` inherits straight from `Exception`, so nothing else in this tuple covers it -- a
    # truncated tar (`ReadError: unexpected end of data`) crashed the publish outright. A
    # half-written shard in a dataset directory is ordinary, and crashing on it would be a worse
    # bug than the hole being closed.
    tarfile.TarError,
)


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
    return None


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

    Only the stretches that are ACTUALLY wide-encoded are narrowed, not the whole file. Taking
    every second byte of arbitrary data invents text that was never written: machine code narrows
    into plausible-looking tokens often enough that 5 of 500 ordinary system binaries were refused
    as holding a credential -- `/usr/bin/bash` narrowed to `fslo_eietossvdrdrcsP3`. A real wide
    character keeps its padding byte NUL, so requiring an unbroken NUL run alongside the candidate
    costs nothing on genuine UTF-16/32 and leaves machine code with nothing long enough to match.
    """
    if kind := _decoded_kind(data):
        return kind
    if b"\x00" not in data:
        return None
    for width, keep in ((2, (0, 1)), (4, (0, 3))):
        for offset in keep:
            # take every `width`-th byte: for UTF-16 that is the ASCII half of each code unit, in
            # whichever of the two byte orders the file used.
            for run in _wide_runs(data, width, offset):
                if kind := _decoded_kind(run):
                    return kind
    return None


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
    container_head = False
    overflowed = False
    seen: set[tuple[int, str]] = set()
    while chunk := handle.read(_SCAN_CHUNK_BYTES):
        if deadline is not None and time.monotonic() > deadline:
            raise _Unscannable("takes too long to decompress")
        if not carry and _jks_private_key_entries(chunk[:16]):
            # A Java KeyStore holding a private-key entry, recognised structurally. Head-anchored
            # like the OpenPGP test and for the same reason: `feedfeed` plus a plausible version
            # and count is only decisive at offset 0.
            return "a private key"
        if not carry and _is_openpgp_secret_key(chunk[:_OPENPGP_HEAD_BYTES]):
            # only ever at offset 0, and `carry` is empty only on the first chunk. Every file and
            # every archive member reaches this, so the binary export is covered wherever it sits.
            return "a private key"
        if not carry:
            # zstd and LZ4 both allow a metadata envelope before the real frame. What matters here
            # is only whether that prelude could be READ to its end: a frame declaring a payload
            # longer than the bytes available leaves the format undecided, and undecided is not
            # clean. The signature search itself runs below over the whole window, so the walked
            # bytes are not needed -- only the verdict on whether the walk ran out.
            head, truncated = _after_skippable_frames(chunk[:_SKIPPABLE_SCAN_BYTES])
            if truncated:
                raise _Unscannable("begins with a frame prelude too long to read past")
            # Anchored: every format is decisive about what a stream BEGINS with, including the
            # short zstd and LZ4 magics that are not searched for at arbitrary offsets below.
            if fmt := _unexpandable_format(head, anchored=True):
                raise _Unscannable(f"contains a {fmt} archive this check cannot expand")
        if not carry and depth:
            # tar as well as the compressed magics: a tar's own members are literal, but a
            # COMPRESSED member inside one is not, so an oversized tar that went past the buffer
            # cap is as unverifiable as an oversized gzip and must refuse rather than pass.
            container_head = _looks_compressed(chunk[:6]) or _looks_like_tar(chunk)
        if depth and not overflowed:
            buffered.extend(chunk)
            if len(buffered) > _MAX_NESTED_BUFFER_BYTES:
                if container_head:
                    raise _Unscannable("contains an archive too large to inspect")
                # Not a container by its head, so the literal scan below is complete coverage and
                # the buffer is only needed to REOPEN a container. Dropping it keeps memory bounded
                # on an ordinary large member; `tail` still decides at the end whether what went
                # past was a zip hiding behind a preamble.
                overflowed = True
                buffered = bytearray()
        if depth:
            tail = (tail + chunk)[-_ZIP_TAIL_BYTES:]
        window = carry + chunk
        # The signature of a self-extracting archive is searched over the WHOLE stream, not just
        # the head. Bounding it to the first 64 KiB only moved the bypass: a stub of at least that
        # size -- which is every real SFX module, since the smallest 7-Zip one is about 150 KiB --
        # put the signature past the window, and the opaque compressed body behind it was scanned
        # as ordinary content and published. There is no upper bound on where a stub ends, so any
        # fixed prefix is a number an attacker picks their padding to exceed.
        #
        # Affordable because these signatures are 4 to 8 bytes of fixed content: `bytes.find` is a
        # memchr-driven scan, and only the 6-to-8-byte signatures are searched this way, whose
        # expected false-positive rate on arbitrary data is vanishing. Measured 0 hits across
        # 256 MiB of random bytes for all six.
        if fmt := _unexpandable_format(window, anchored=False):
            raise _Unscannable(f"contains a {fmt} archive this check cannot expand")
        if kind := _credential_kind(window):
            return kind
        # A two-marker credential is paired across the WHOLE stream, not within one window. Those
        # detectors are order-independent and distance-free inside a single buffer, but a chunked
        # scan re-imposed a window between the halves at the chunk boundary. Remembering which
        # halves have gone past keeps memory bounded whatever the distance -- which is unbounded,
        # since JWK members and PuTTY sections may sit behind any amount of intervening data.
        if kind := _paired_markers_kind(window, seen):
            return kind
        carry = window[-_SCAN_OVERLAP_BYTES:]
    if overflowed and _has_zip_end_record(tail):
        raise _Unscannable("contains an archive too large to inspect")
    if buffered and _looks_like_container(bytes(buffered)):
        return _credential_in_container(bytes(buffered), deadline=deadline or 0.0, depth=depth + 1)
    return None


def _unexpandable_format(data: bytes, *, anchored: bool) -> str | None:
    """The name of an unexpandable archive format `data` carries, or None.

    Two questions rather than one, because the two have different error costs.

    `anchored` asks whether the stream BEGINS with such a signature. Every format is decisive
    there: a file whose first bytes are a zstd frame is a zstd frame.

    Unanchored asks whether one appears anywhere, which is what catches a self-extracting archive
    -- an executable stub, then the signature, then the opaque compressed body, as `rar a -sfx` or
    7-Zip's `-sfx` writes. Only the signatures of six bytes or more are searched that way. The
    4-byte zstd and LZ4 magics are not distinctive enough to be decisive at an arbitrary offset:
    embedded in a large model shard or high-entropy dataset they refuse a publishable file, and a
    false refusal on ordinary content is worse than the narrow bypass of an SFX built from a
    format that has no self-extracting form in the first place. RAR and 7-Zip, which do ship SFX
    modules, carry 6-to-8-byte signatures and stay searched.
    """
    magics = (
        _UNEXPANDABLE_MAGIC
        if anchored
        else [pair for pair in _UNEXPANDABLE_MAGIC if len(pair[0]) >= _SFX_MAGIC_BYTES]
    )
    if anchored:
        return next((fmt for magic, fmt in magics if data.startswith(magic)), None)
    return next((fmt for magic, fmt in magics if magic in data), None)


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
        for half, pattern in (("context", detector.context), ("payload", detector.payload)):
            if (index, half) not in seen and pattern.search(window):
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
        _looks_compressed(data[:6]) or _looks_like_tar(data) or zipfile.is_zipfile(io.BytesIO(data))
    )


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
    # EVERY applicable format is tried, not just the first one that claims a match. The detectors
    # are heuristics over untrusted bytes and one of them was decisive: `is_zipfile` searches the
    # last 64 KiB for the end-of-central-directory record, so four bytes of `PK\x05\x06` ANYWHERE
    # in a tar made it claim the file. The tar was then opened as a zip, failed, and the failure
    # was read as "nothing here" -- so adding four stray bytes to any member of a tar took a
    # gzipped credential inside it from refused to published. Falling through instead means a
    # wrong guess costs an open attempt rather than the whole scan.
    #
    # A handler that REFUSES is deferred rather than allowed to end the loop. Its limits are
    # applied to bytes that may not be its format at all -- a tar carrying a fake end-of-central
    # -directory record made the zip handler refuse for "too many members", and because
    # `_Unscannable` is deliberately not in `_UNREADABLE_ARCHIVE`, that refusal escaped before the
    # tar handler ever ran. The credential inside was real and went unreported. A handler that
    # COMPLETES has genuinely enumerated the bytes, so its answer settles the question; the
    # deferred refusal is re-raised only when no handler managed that, which keeps a real
    # oversized archive fail-closed.
    refusal: _Unscannable | None = None
    for handler in (_credential_in_zip, _credential_in_tar, _credential_in_compressed):
        try:
            if kind := handler(source, deadline=deadline, depth=depth):
                return kind
        except _Unscannable as unscannable:
            refusal = refusal or unscannable
        except _UNREADABLE_ARCHIVE:
            continue  # not this format, or corrupt in it; the remaining formats still get a turn
    if refusal is not None:
        raise refusal
    return None


def _credential_in_compressed(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential inside a gzip, bzip2, xz or zlib stream, or None."""
    head = source[:6] if isinstance(source, bytes) else source.open("rb").read(6)
    opener = {b"BZh": bz2.open, b"\xfd7zXZ\x00": lzma.open}.get(
        next((magic for magic in (b"BZh", b"\xfd7zXZ\x00") if head.startswith(magic)), b""),
        gzip.open,
    )
    # A raw zlib stream (RFC 1950) is deflate with a 2-byte header instead of gzip's 10, so none of
    # the openers above read it. `decompressobj` with the zlib window size does, and it is the same
    # deflate underneath -- which is why the stream can be expanded rather than merely refused.
    if opener is gzip.open and not head.startswith(b"\x1f\x8b") and _looks_like_zlib(head):
        raw = source.read_bytes() if isinstance(source, Path) else source
        # FDICT (bit 5 of the flag byte) means the stream was compressed against a preset
        # dictionary that is not carried in the file. Without it `decompress` raises, and treating
        # that as "not zlib after all" let the opaque bytes fall through to the literal scan and
        # publish. Refusing is the honest answer: the content cannot be inspected from here.
        #
        # Gated on the bytes not reading as text first. The zlib header rule is about eleven bits
        # of signal, and `x ` satisfies all of it -- so an ordinary `x = 1` sidecar was refused as
        # a dictionary-compressed stream and could not be published at all. A refusal needs more
        # than a heuristic behind it, and a deflate payload is not printable ASCII: measured over
        # 20 real dictionary-compressed streams none read as text, and over every innocent shape
        # that trips the header none read as compressed. Decompression cannot make this call --
        # `zlib.error` is identical for a real FDICT stream and for `x = 1`.
        if head[1] & 0x20 and not _looks_like_textual(raw):
            raise _Unscannable("contains a compressed stream needing a dictionary to inspect")
        # EVERY concatenated stream is inflated, not just the first. `decompressobj` stops at the
        # end of one zlib record and hands the rest back as `unused_data`; scanning only the first
        # plaintext meant `zlib.compress(benign) + zlib.compress(secret)` published clean, since
        # the credential lived entirely in the discarded remainder. Concatenated records are what a
        # per-record cache or an appended log writes, so this is an ordinary shape as well as a
        # reachable bypass.
        remaining, budget = raw, _MAX_NESTED_BUFFER_BYTES
        for record in range(_MAX_ZLIB_RECORDS):
            inflate = zlib.decompressobj()
            try:
                plain = inflate.decompress(remaining, budget)
            except zlib.error:
                if record:
                    # Records inflated and then one did not: the trailing bytes are a compressed
                    # stream this cannot read, and undecided is not clean. Only the FIRST record
                    # failing means "not zlib after all", which the openers below still handle.
                    raise _Unscannable(
                        "contains trailing compressed data this check cannot inspect"
                    ) from None
                break
            # `max_length` TRUNCATES rather than raising, so a credential past the cap would read
            # as a clean scan. `unconsumed_tail` is non-empty exactly when that happened.
            if inflate.unconsumed_tail:
                raise _Unscannable("contains a compressed stream too large to inspect")
            if kind := _scan_stream(io.BytesIO(plain), deadline=deadline, depth=depth):
                return kind
            # The budget is shared across the records so a chain of them cannot buy more expansion
            # than one stream of the same total size.
            budget -= len(plain)
            remaining = inflate.unused_data
            if not remaining:
                return None
            if budget <= 0:
                raise _Unscannable("contains a compressed stream too large to inspect")
        else:
            raise _Unscannable("contains more compressed records than this check can inspect")
    with opener(source if isinstance(source, Path) else io.BytesIO(source), "rb") as stream:
        return _scan_stream(stream, deadline=deadline, depth=depth)


def _credential_in_zip(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential in any readable member of a zip, or None."""
    # The member count is read from the end-of-central-directory record BEFORE `ZipFile` is
    # constructed. `ZipFile.__init__` parses the whole central directory and materializes every
    # `ZipInfo`, so a bound checked after it is charged the cost it exists to avoid -- measured at
    # 1.8 seconds and 239 MB of resident memory for 400,000 empty entries in a 35 MB file, all of
    # it spent before the per-member loop below ran once.
    if _zip_member_count(source, _MAX_ARCHIVE_MEMBERS) > _MAX_ARCHIVE_MEMBERS:
        raise _Unscannable("contains an archive with too many members to inspect")
    unreadable = ""
    with zipfile.ZipFile(source if isinstance(source, Path) else io.BytesIO(source)) as archive:
        for count, info in enumerate(archive.infolist(), 1):
            if count > _MAX_ARCHIVE_MEMBERS:
                raise _Unscannable("contains an archive with too many members to inspect")
            if time.monotonic() > deadline:
                raise _Unscannable("takes too long to decompress")
            if info.is_dir():
                continue
            # Bit 0 of the general-purpose flags marks an encrypted member. Its bytes cannot be
            # read without the password, so treating it as clean approved a package whose only copy
            # of a credential was inside -- `zip -P` around a key file published intact.
            #
            # Noted and raised AFTER the loop rather than here, because the members behind it must
            # still be scanned: refusing on sight would abandon the rest of the archive, and a
            # credential further down would go unreported in favour of a weaker message about an
            # unreadable member. A found credential is the more specific answer, so it wins. The
            # same deferral covers members whose compression this Python cannot decode, below.
            if info.flag_bits & 0x01:
                unreadable = unreadable or "an encrypted archive member this check cannot read"
                continue
            try:
                with archive.open(info) as member:
                    if kind := _scan_stream(member, deadline=deadline, depth=depth):
                        return kind
            except NotImplementedError:
                # The member is valid but uses a compression method this Python has no decompressor
                # for -- Deflate64 is the common one, and `zip -fd` writes it. Caught by the broad
                # skip below it read as "opaque member, carry on" and the credential in its payload
                # published. Recorded like an encrypted member: unverifiable is not clean.
                #
                # Reported distinctly from encryption because the remedy differs: a password is not
                # what is missing, the archive has to be rewritten with a supported method.
                unreadable = unreadable or (
                    "an archive member compressed in a way this check cannot read"
                )
            except _UNREADABLE_ARCHIVE:
                # Recorded like the two above rather than skipped silently. A member of a SPLIT
                # archive (`zip -s`) has its directory entry in the final volume and its bytes in
                # an earlier one, so opening it here raises and the member read as clean -- both
                # published parts returned None while joining the volumes recovered the key. The
                # bytes are not in this file, which is exactly the "unverifiable" case, and every
                # other unreadable member reaches the same conclusion for the same reason.
                unreadable = unreadable or "an archive member this check cannot read"
                continue  # the rest of the archive still gets scanned
    if unreadable:
        raise _Unscannable(f"contains {unreadable}")
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
    if deadline is None:
        deadline = time.monotonic() + _MAX_DECOMPRESS_SECONDS
    with path.open("rb") as handle:
        # The package budget covers the file's own bytes too. Leaving it off looked safe because
        # the read is bounded by the package size limit, but the limit bounds BYTES and this
        # bounds TIME: matching cost is not uniform per byte, so a large file of adversarial
        # near-matches held a worker far longer than its size suggested.
        if kind := _scan_stream(handle, deadline=deadline):
            return kind
    # `is_zipfile` is consulted inside, so a self-extracting archive is expanded despite its stub
    return _credential_in_container(path, deadline=deadline, depth=1)


def credential_in_name(name: str) -> str | None:
    """The kind of credential the path `name` itself carries, or None.

    A file whose *name* is the key leaks it through the archive's member list even when its
    contents are empty, and the published repo shows that name in its tree forever.

    `surrogatepass` rather than `surrogateescape`: the latter is a DECODE-only handler, so a name
    holding a lone surrogate raised `UnicodeEncodeError` out of a security check. Reached from the
    publish route that check turned a 400 into an uncaught 500, and reached from a tar member name
    it crashed the scan of an archive rather than reporting what was in it.
    """
    return _credential_kind(name.encode("utf-8", "surrogatepass"))


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

    # `surrogatepass` for the same reason as `credential_in_name`: `surrogateescape` cannot encode
    # a lone surrogate, and this runs while REPORTING a refusal, so the crash would replace the
    # message naming the credential.
    masked = name.encode("utf-8", "surrogatepass")
    for _kind, pattern in _TOKEN_PATTERNS + _ASSIGNED_PATTERNS:
        masked = pattern.sub(_mask, masked)
    # A name detected only through base64, or as a whole key structure, has no plaintext body to
    # mask, so masking cannot help: printing any of it prints the key. A compact private JWK fits
    # in a 129-character filename, and every pattern above left it untouched, so the refusal
    # printed the complete Ed25519 scalar to the terminal and any collected logs. Withhold the
    # name and give the author the directory instead, which is enough to find a file they just
    # tried to publish.
    if any(pattern.search(masked) for _kind, pattern in _LITERAL_PATTERNS) or _match_base64(masked):
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

    def unwalkable(error: OSError) -> NoReturn:
        # `os.walk` swallows descent errors by default, so a directory the scan cannot enter --
        # mode 000 in an uploaded tar, read by a non-root control plane -- had its contents skipped
        # silently and a credential inside was published intact. The same tree with the directory
        # readable is refused, so passing here was purely a function of what could be opened.
        #
        # Named relative to the package like every other refusal: the absolute path is the control
        # plane's staging directory, which is the server's business rather than the publisher's.
        failed = Path(str(error.filename or package_root))
        relative = failed.relative_to(package_root).as_posix() if failed != package_root else "."
        raise ValueError(
            f"{_redacted(display.get(relative, relative))} could not be read to check it for "
            "credentials, so publishing it would commit unscanned content to a shared environment "
            "repository. Fix the permissions on it before publishing."
        ) from None

    # sorted so the member named in the refusal is the same one on every machine.
    for root, dirs, files in os.walk(package_root, onerror=unwalkable):
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
