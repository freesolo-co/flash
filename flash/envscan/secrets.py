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

import bz2
import functools
import gzip
import io
import lzma
import os
import tarfile
import time
import zipfile
import zlib
from pathlib import Path
from typing import IO, NoReturn

from flash.envscan.archive import (
    _looks_like_cpio_header,
    credential_in_ar,
    credential_in_cpio,
    credential_in_tar,
    credential_in_zip,
)
from flash.envscan.base64 import _Inspector, _match_base64, _RunTooLongToExpand
from flash.envscan.buffers import (
    _SCAN_CHUNK_BYTES,
    _blocks_of,
    _looks_like_container,
    _looks_like_container_head,
    _looks_like_source_container,
    _paired_markers_kind,
    _paired_state,
    _zlib_prefix_inflates,
)
from flash.envscan.deflate import (
    _PDF_SIGNATURE,
    _document_payloads,
    _DocumentDeadlineExceeded,
    _EncryptedDocument,
    _gzip_metadata,
    _GzipHeaderTooLarge,
    _raw_deflate_records_from,
    _scan_framed_stream,
    _TooManyRawDeflateRecords,
    _TooManyStreams,
    _UnreachedStream,
    _UnreadableFilterChain,
)
from flash.envscan.formats import (
    _MAX_ARCHIVE_MEMBERS,
    _OVERLAY_PROBE_BYTES,
    _ZIP_TAIL_BYTES,
    OVERLAY_UNPROBED,
    _after_skippable_frames,
    _has_zip_end_record,
    _looks_like_lzma_alone,
    _looks_like_textual,
    _looks_like_zlib,
    _overlay_offset,
    _overlay_payload,
    # Re-exported rather than dropped when the archive walk moved out: the member-count limit is
    # read HERE, so the tests that rebind it read the counter from here too.
    _zip_member_count,  # noqa: F401
)
from flash.envscan.joined import _rejoined

# The two head-anchored walks and the refusal they raise, split out to keep this module under the
# file-size limit. `_Unscannable` lives THERE because those walks raise it and this module imports
# them, so the other direction would be a cycle. Re-exported because every other raise site and
# every test still reads all three from here.
from flash.envscan.keystores import (
    _decoded_key_kind,
    _keystore_undecided,
    _openpgp_kind,
    _Unscannable,
)
from flash.envscan.names import exact_name_values
from flash.envscan.opaque import OpaqueDeadlineExceeded, opaque_format
from flash.envscan.openpgp import (
    _encrypted_text_format,
    _has_openpgp_message_armor,
    _unfinished_openpgp_message_armor,
)
from flash.envscan.patterns import _MAX_BODY, _match, _unfinished_private_key_armor
from flash.envscan.policy import _unexpandable_format, _uninspectable_reason
from flash.envscan.redact import redacted_name
from flash.envscan.wide import credential_in_wide_runs

# Carried between chunks so a credential straddling a chunk boundary is still matched. Derived from
# `_MAX_BODY` rather than written as a bare number, so the two cannot drift apart: the overlap must
# exceed the longest possible match (a body plus its prefix and quoting) or a credential landing on
# a boundary is fully visible in no window at all.
_SCAN_OVERLAP_BYTES = _MAX_BODY * 4

# How much of a chunk's head is walked for skippable frames. Generous for the handful a real
# seekable stream carries, and bounded so a chain of crafted frame headers cannot make this walk a
# cost of its own.
_SKIPPABLE_SCAN_BYTES = 64 << 10
# How many concatenated zlib records are inflated before the stream is refused. A per-record cache
# or an appended log writes a handful; the bound is what stops a file of many tiny records from
# becoming an expansion cost of its own, and exceeding it refuses rather than passes.
_MAX_ZLIB_RECORDS = 64

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


def _credential_kind(
    data: bytes,
    *,
    deadline: float | None = None,
    depth: int = 0,
    truncated: bool = False,
    paired: bool = True,
    shell: bool = False,
    literal_syntax: str | None = None,
    rejoin: bool = True,
) -> str | None:
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
    if kind := _decoded_kind(
        data,
        deadline=deadline,
        depth=depth,
        truncated=truncated,
        paired=paired,
        shell=shell,
        literal_syntax=literal_syntax,
        rejoin=rejoin,
    ):
        return kind
    if b"\x00" not in data:
        return None
    return credential_in_wide_runs(
        data,
        detector=_decoded_kind,
        deadline=deadline,
        depth=depth,
        truncated=truncated,
        shell=shell,
        literal_syntax=literal_syntax,
        rejoin=rejoin,
    )


def _decoded_kind(
    data: bytes,
    *,
    deadline: float | None = None,
    depth: int = 0,
    truncated: bool = False,
    paired: bool = True,
    shell: bool = False,
    literal_syntax: str | None = None,
    rejoin: bool = True,
) -> str | None:
    """The kind of credential in `data` literally, or inside a base64 run within it."""
    if kind := _match(data, paired=paired) or _match_base64(
        data, _decoded_container(deadline, depth), truncated=truncated
    ):
        return kind
    # A file can hold a credential in pieces that no contiguous run of its bytes contains: adjacent
    # string literals, which the language concatenates at parse time, and a backslash-newline
    # continuation, which the shell removes before the value is assigned. Python source is EXEMPT
    # from the filename filter by design -- helper modules have to ship or the worker fails to
    # import -- so a key split either way had nothing between it and the hub.
    #
    # Only tried when the literal pass found nothing, and `_rejoined` returns the input unchanged
    # when no seam is present, so the ordinary file pays two cheap searches and no rematch.
    if not rejoin:
        return None
    joined = _rejoined(data, shell=shell, literal_syntax=literal_syntax)
    return _match(joined, paired=paired) if joined != data else None


def _decoded_container(deadline: float | None, depth: int) -> _Inspector | None:
    """What `_match_base64` should do with decoded bytes that match no pattern, or None to stop.

    A base64 value routinely holds a whole CONTAINER: a Kubernetes Secret, a cloud-init document
    and a `kubectl -o yaml` export all store their values encoded, so a gzipped credential inside
    one decoded here and was then matched while still compressed and published clean.

    At the depth cap the container is REFUSED rather than skipped. Returning None there switched
    the second look off and reported the member clean, so four nested zips around
    `base64(gzip(secret))` published while the same gzip added as an ordinary fifth container
    correctly raised -- the cap is a limit on what can be inspected, and every other limit in this
    module raises rather than returning a verdict it did not reach. Only bytes that actually look
    like a container are refused, so an ordinary deeply-nested file still publishes.

    A refusal from the second look is swallowed only for a SPECULATIVE decode, unlike every other
    unscannable path here, because `_match_base64` tries four alignments of every base64-shaped
    run: the "container" handed over is then a re-interpretation of bytes never claimed to be one,
    and an ELF holds enough such runs to produce one by chance. Measured on `containerd`, `ctr` and
    `dockerd`: each decoded to something tripping the dictionary-zlib refusal, making them
    unpublishable over bytes nobody encoded.

    An exact WHOLE-RUN decode is not speculative -- the run is aligned, complete, and decodes to a
    container in one piece -- so its refusal propagates. Swallowing that one turned a real
    `zip -P` archive behind base64 into a clean result while the same archive scanned directly was
    refused. `whole` carries that distinction down from `_match_base64`.

    An exact decode is also checked for a format that cannot be inspected at ALL, which is not the
    same question as whether it is a container. `openssl enc -a` writes its salted envelope in
    base64 by design, and that form published a key the binary form of the same ciphertext refused.
    """
    if deadline is None:
        return None

    def inspect(decoded: bytes, *, whole: bool = False) -> str | None:
        if time.monotonic() > deadline:
            raise _Unscannable("takes too long to decompress")
        # exact decodes receive the same anchored binary checks as standalone files.
        if whole and (kind := _decoded_key_kind(decoded)):
            return kind
        if whole and (fmt := _unexpandable_format(decoded, anchored=True)):
            raise _Unscannable(_uninspectable_reason(fmt))
        # raw deflate has no magic. probing only exact decodes avoids an inflate for every speculative
        # base64 alignment in large csv files and keeps chance decodes from becoming trusted refusals.
        if whole and (
            kind := _credential_in_raw_deflate(decoded, deadline=deadline, depth=depth + 1)
        ):
            return kind
        # Only bytes that actually look like a container are re-entered; anything else has already
        # been through `_match` and would just be scanned a second time.
        if not _looks_like_container(decoded):
            return None
        if depth >= _MAX_CONTAINER_DEPTH:
            raise _Unscannable("nests compressed containers too deeply to inspect")
        try:
            return _credential_in_container(decoded, deadline=deadline, depth=depth + 1)
        except _Unscannable:
            if whole or time.monotonic() > deadline:
                raise
            return None
        except _UNREADABLE_ARCHIVE:
            return None

    inspect.deadline = deadline  # type: ignore[attr-defined]
    return inspect


def _scan_stream_head(chunk: bytes, upcoming: bytes) -> tuple[str | None, bool, bool]:
    """Inspect the first chunk's anchored formats and classify its container syntax."""
    container_head = _looks_like_container_head(chunk)
    source_container = _looks_like_source_container(chunk)
    if kind := _openpgp_kind(chunk, truncated=bool(upcoming)):
        return kind, container_head, source_container
    head, truncated = _after_skippable_frames(chunk[:_SKIPPABLE_SCAN_BYTES])
    if truncated:
        raise _Unscannable("begins with a frame prelude too long to read past")
    if fmt := _unexpandable_format(head, anchored=True):
        raise _Unscannable(_uninspectable_reason(fmt))
    return None, container_head, source_container


def _scan_stream(handle: IO[bytes], *, deadline: float | None = None, depth: int = 0) -> str | None:
    """The kind of credential anywhere in `handle`, or None.

    The WHOLE stream is read -- memory is bounded by the chunk size, not the total -- with an
    overlap so a credential straddling a chunk boundary is still matched. Stopping early on a byte
    count would mean a key placed after the cutoff publishes, which is the bug rather than the
    protection.

    `deadline` bounds expansion time and refuses rather than returning a clean verdict.

    A stream that is ITSELF a container is expanded in turn. Nested containers are buffered to be
    reopened, so one too large to hold in memory also raises: leaving it to the scan of its literal
    bytes looked like a reasonable trade, but a deflated member's bytes hold the credential nowhere
    a pattern can see, so it was a silent bypass reachable by padding an archive past the cap.
    """
    name = str(getattr(handle, "name", "")).lower()
    shell = name.endswith(".sh")
    literal_syntax = (
        "yaml" if name.endswith((".yaml", ".yml")) else "toml" if name.endswith(".toml") else None
    )
    carry = b""
    buffered = bytearray()
    tail = b""
    container_head = source_container = False
    overflowed = False
    overlay_tail = b""
    overlay_found = False
    seen = _paired_state()
    store_head = bytearray()
    walking_store = True
    # Read one chunk AHEAD, so each pass knows whether bytes follow it. A base64 run reaching the
    # end of a chunk that is not the last one was cut by the read rather than ended by the file,
    # and a container encoded across that cut can no longer be expanded from either piece.
    chunk = handle.read(_SCAN_CHUNK_BYTES)
    while chunk:
        upcoming = handle.read(_SCAN_CHUNK_BYTES)
        if deadline is not None and time.monotonic() > deadline:
            raise _Unscannable("takes too long to decompress")
        if walking_store:
            store_head.extend(chunk)
            walking_store = _keystore_undecided(bytes(store_head))
            if walking_store is None:
                # Named for the STORE rather than for the entry inside it: the walk stops at the
                # first key entry, which may be a private key or a JCEKS symmetric key, and the
                # author has to rotate the store either way.
                return "a key store"
            if not walking_store:
                store_head = bytearray()
        if not carry:
            kind, container_head, source_container = _scan_stream_head(chunk, upcoming)
            if kind:
                return kind
        if depth:
            overlay_probe = overlay_tail + chunk
            overlay_found = overlay_found or _overlay_offset(overlay_probe) is not None
            overlay_tail = overlay_probe[-_OVERLAY_PROBE_BYTES:]
            if overflowed and overlay_found:
                raise _Unscannable("contains an appended archive too large to inspect")
        if depth and not overflowed:
            buffered.extend(chunk)
            if len(buffered) > _MAX_NESTED_BUFFER_BYTES:
                if overlay_found:
                    raise _Unscannable("contains an appended archive too large to inspect")
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
            raise _Unscannable(_uninspectable_reason(fmt))
        # Armored OpenPGP ciphertext, over the same window and for the same reason as the binary
        # form above: the body is opaque, so neither a pattern nor a base64 decode can see what is
        # inside, and treating it as ordinary text published the message intact.
        if _has_openpgp_message_armor(window):
            raise _Unscannable("contains an encrypted OpenPGP message this check cannot read")
        if bool(upcoming) and _unfinished_openpgp_message_armor(window):
            raise _Unscannable("contains an OpenPGP message armor header too long to read past")
        # encrypted text containers require body structure, not documentation markers alone.
        if encrypted := _encrypted_text_format(window):
            raise _Unscannable(f"contains a {encrypted} file this check cannot read")
        # A private-key armor whose HEADER runs past this window. The PEM pattern requires the
        # BEGIN line and the start of the base64 body in one buffer, and RFC 4880 armor headers sit
        # between them with no length limit -- so a 1.1 MB `Comment:` pushed the body into the next
        # chunk, the two halves appeared in no single window, and a real `gpg --export-secret-keys
        # --armor` key published. The body is what proves a key rather than prose about one, so it
        # cannot simply be dropped from the pattern; an armor still in its headers at the end of a
        # window is undecided instead, and undecided refuses.
        if bool(upcoming) and _unfinished_private_key_armor(window):
            raise _Unscannable("contains a private key armor header too long to read past")
        try:
            # The two-marker detectors are left out here and run below instead, over state
            # carried between windows. Inside `_match` they build a fresh record splitter per
            # window, which knows neither the depth nor the quote phase the previous window ended
            # on -- right for a whole file, wrong for one window of one.
            if kind := _credential_kind(
                window,
                deadline=deadline,
                depth=depth,
                truncated=bool(upcoming),
                paired=False,
                shell=shell,
                literal_syntax=literal_syntax,
                # archive framing is not source syntax, so its raw pass must not join bytes that
                # belong to separate names, headers, or members. exact metadata and members are
                # scanned independently by their container walkers.
                rejoin=not source_container,
            ):
                return kind
        except _RunTooLongToExpand:
            raise _Unscannable("contains a base64 run too long to expand") from None
        # A two-marker credential is paired across the WHOLE stream, not within one window. Those
        # detectors are order-independent and distance-free inside a single buffer, but a chunked
        # scan re-imposed a window between the halves at the chunk boundary. Remembering which
        # halves have gone past keeps memory bounded whatever the distance -- which is unbounded,
        # since JWK members and PuTTY sections may sit behind any amount of intervening data.
        # The overlap is declared so the splitter can rewind to where the next window starts:
        # those bytes are tokenized twice and must be counted once.
        if kind := _paired_markers_kind(
            window, seen, overlap=_SCAN_OVERLAP_BYTES if upcoming else 0
        ):
            return kind
        carry = window[-_SCAN_OVERLAP_BYTES:]
        chunk = upcoming
    if walking_store and store_head:
        # The stream ENDED with the walk still undecided, which more bytes can no longer settle --
        # either it ran off the end of the file or it exhausted the entry bound. Both mean the
        # entries behind the stopping point are unread, and unread is not clean.
        raise _Unscannable("contains a key store this check cannot finish walking")
    if overflowed and _has_zip_end_record(tail):
        raise _Unscannable("contains an archive too large to inspect")
    if buffered and _looks_like_container(bytes(buffered)):
        return _credential_in_container(bytes(buffered), deadline=deadline or 0.0, depth=depth + 1)
    return None


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
    try:
        if fmt := opaque_format(source, deadline=deadline):
            raise _Unscannable(_uninspectable_reason(fmt))
    except OpaqueDeadlineExceeded:
        raise _Unscannable("takes too long to decompress") from None
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
    # A handler that returns None has either enumerated its format and found nothing, or declined
    # bytes that were never its format -- and `None` alone cannot tell those apart, so a deferred
    # refusal was re-raised over a scan that had already answered the question. A tar whose first
    # member is named `x data.txt` reproduces it: the leading `x ` satisfies the zlib FDICT header
    # rule, that handler defers a dictionary-stream refusal, the tar walk then enumerates the
    # archive successfully, and the file was refused anyway.
    #
    # Only the archive handlers count as settling it. Their success is a statement about the WHOLE
    # file -- every member listed and read -- whereas a stream handler returning None has read one
    # payload and says nothing about bytes another handler could not reach. Ar must count too: a
    # clean Debian archive was fully walked, then refused on a false gzip guess over its outer bytes.
    refusal: _Unscannable | None = None
    settled = False
    for handler in (
        _credential_in_zip,
        _credential_in_tar,
        _credential_in_ar,
        _credential_in_compressed,
        _credential_in_overlay,
        _credential_in_raw_deflate,
        _credential_in_pdf,
    ):
        if time.monotonic() > deadline:
            raise _Unscannable("takes too long to decompress")
        try:
            if kind := handler(source, deadline=deadline, depth=depth):
                return kind
            settled = settled or handler in (
                _credential_in_zip,
                _credential_in_tar,
                _credential_in_ar,
            )
        except _Unscannable as unscannable:
            if time.monotonic() > deadline:
                raise
            refusal = refusal or unscannable
        except _UNREADABLE_ARCHIVE:
            continue  # not this format, or corrupt in it; the remaining formats still get a turn
    if time.monotonic() > deadline:
        raise _Unscannable("takes too long to decompress")
    if refusal is not None and not settled:
        raise refusal
    return None


def _credential_in_overlay(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential in a compressed payload appended after a stub, or None.

    Last of the handlers, and it re-enters the ordinary container path on the payload alone rather
    than reimplementing any format: what sits behind the stub is an ordinary gzip, bzip2 or xz
    stream, so once the offset is known there is nothing special about it.

    The stub's own bytes are NOT re-scanned here. `_scan_stream` already read them literally on the
    way in -- it is text -- so this only has to cover the part that scan could not see.
    """
    if (at := _overlay_offset(source)) is None:
        return None
    if at == OVERLAY_UNPROBED:
        raise _Unscannable("contains more appended archive candidates than this check can probe")
    payload = _overlay_payload(source, at, _MAX_NESTED_BUFFER_BYTES)
    if payload is None:
        return None
    if not payload:
        raise _Unscannable("contains an appended archive too large to inspect")
    return _credential_in_container(payload, deadline=deadline, depth=depth)


def _credential_in_raw_deflate(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """Scan complete concatenated RFC 1951 records without reading the source whole.

    With no magic, successful bounded inflation is the format recognition.
    """
    try:
        records = _raw_deflate_records_from(
            _blocks_of(source), _MAX_NESTED_BUFFER_BYTES, _MAX_ZLIB_RECORDS
        )
    except _TooManyRawDeflateRecords:
        raise _Unscannable("contains more compressed records than this check can inspect") from None
    if records is None:
        raise _Unscannable("contains a compressed stream too large to inspect")
    for plain in records:
        if kind := _scan_stream(io.BytesIO(plain), deadline=deadline, depth=depth):
            return kind
    return None


def _credential_in_pdf(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """Scan structurally declared PDF and PNG payloads under the shared bounds."""
    # decline non-documents from their head; real documents are buffered only within the same
    # nested-container cap because their trailer and object dictionaries require random access.
    if isinstance(source, Path):
        with source.open("rb") as handle:
            if not handle.read(8).startswith((_PDF_SIGNATURE, b"\x89PNG\r\n\x1a\n")):
                return None
        if source.stat().st_size > _MAX_NESTED_BUFFER_BYTES:
            raise _Unscannable("contains a document too large to inspect")
    raw = source.read_bytes() if isinstance(source, Path) else source
    try:
        for plain in _document_payloads(raw, _MAX_NESTED_BUFFER_BYTES, deadline=deadline):
            if plain is None:
                raise _Unscannable("contains a compressed stream too large to inspect")
            if kind := _scan_stream(io.BytesIO(plain), deadline=deadline, depth=depth):
                return kind
    except _DocumentDeadlineExceeded:
        raise _Unscannable("takes too long to decompress") from None
    except _EncryptedDocument:
        raise _Unscannable("contains an encrypted document this check cannot read") from None
    except _TooManyStreams:
        raise _Unscannable("contains more compressed streams than can be inspected") from None
    except _UnreachedStream:
        raise _Unscannable("contains a compressed stream this could not locate") from None
    except _UnreadableFilterChain:
        raise _Unscannable(
            "contains a compressed stream behind a filter this cannot undo"
        ) from None
    return None


def _credential_in_compressed(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential inside a gzip, bzip2, xz, lzma-alone or zlib stream, or None."""
    head = source[:13] if isinstance(source, bytes) else source.open("rb").read(13)
    opener = {b"BZh": bz2.open, b"\xfd7zXZ\x00": lzma.open}.get(
        next((magic for magic in (b"BZh", b"\xfd7zXZ\x00") if head.startswith(magic)), b""),
        gzip.open,
    )
    if head.startswith(b"\x1f\x8b\x08"):
        try:
            metadata = _gzip_metadata(source)
        except _GzipHeaderTooLarge:
            raise _Unscannable("contains a gzip header too large to inspect") from None
        for value in metadata:
            if kind := credential_in_name(
                value.decode("latin-1"), deadline=deadline
            ) or _scan_stream(io.BytesIO(value), deadline=deadline, depth=depth):
                return kind
    # A raw zlib stream (RFC 1950) is deflate with a 2-byte header instead of gzip's 10, so none of
    # the openers above read it. `decompressobj` with the zlib window size does, and it is the same
    # deflate underneath -- which is why the stream can be expanded rather than merely refused.
    # Probed on a bounded prefix before the whole member is held. The header rule is about eleven
    # bits, so roughly one arbitrary file in 2,000 trips it, and an extracted member may be 256 MiB
    # with the request body and the staged file already live -- so an ordinary model shard that
    # merely starts with the right two bytes was charged a second full copy to establish it was
    # never zlib at all. A real stream inflates its prefix; only bytes that are not deflate raise,
    # which is exactly the accidental case. Failing the probe skips this branch and falls through
    # to the openers below, the same path the first record's `zlib.error` already took.
    #
    # FDICT is exempt because it CANNOT inflate: the dictionary is not in the file, so `decompress`
    # raises for a genuine stream just as it does for an accident. That branch stays decided by the
    # textual gate below, on the whole member as before.
    if (
        opener is gzip.open
        and not head.startswith(b"\x1f\x8b")
        and _looks_like_zlib(head)
        and (head[1] & 0x20 or _zlib_prefix_inflates(source))
    ):
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
                if record and _looks_like_zlib(remaining[:2]):
                    # Records inflated and then one did not: the trailing bytes are a compressed
                    # stream this cannot read, and undecided is not clean. Only the FIRST record
                    # failing means "not zlib after all", which the openers below still handle.
                    raise _Unscannable(
                        "contains trailing compressed data this check cannot inspect"
                    ) from None
                # A remainder that does not even open like a record is a footer, a checksum or the
                # next section of a framed file -- ordinary bytes rather than something unreadable.
                # Refusing on any remainder rejected `zlib.compress(b"harmless") + b"footer"`, whose
                # single record decoded perfectly, and with it the framed and cache formats that
                # write exactly that shape. Those bytes are still SCANNED: falling through leaves
                # them to the literal pass over the file, which is where a credential in a footer
                # would be found anyway.
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
    framed = (
        "bz2"
        if head.startswith(b"BZh")
        else "xz"
        if head.startswith(b"\xfd7zXZ\x00")
        else "lzma-alone"
        if _looks_like_lzma_alone(head)
        else None
    )
    if framed:
        kind, trailing = _scan_framed_stream(
            source,
            format=framed,
            deadline=deadline,
            depth=depth,
            scan=_scan_member,
            trailing_limit=_MAX_NESTED_BUFFER_BYTES,
        )
        if trailing is None:
            raise _Unscannable("contains trailing data too large to inspect")
        if kind or not trailing:
            return kind
        return _credential_in_container(trailing, deadline=deadline, depth=depth)
    stream_source = source if isinstance(source, Path) else io.BytesIO(source)
    with opener(stream_source, "rb") as stream:
        try:
            return _scan_stream(stream, deadline=deadline, depth=depth)
        except gzip.BadGzipFile:
            # A COMPLETE member followed by bytes that are not a valid next one: the reader finishes
            # the member, looks for another where the trailer ends, and raises -- without yielding
            # one byte of the plaintext it already inflated. `BadGzipFile` is an `OSError`, so the
            # dispatch loop read that as "never this format", every remaining handler declined, and
            # the file published on its literal bytes. `gzip.compress(key) + b"x"` is that file, and
            # `gzip -dc` prints the key from it: recoverable by the ordinary tool, unseen by the
            # publish. Any non-null trailing byte does it, so concatenation reaches this too.
            #
            # The magic check is not redundant: `gzip.open` is the fallback opener here, so ordinary
            # text raises this SAME exception on its first header read. Only `zlib.error` -- damage
            # INSIDE a member, genuinely unreadable rather than merely unread -- keeps falling
            # through, which is what lets a corrupt shard publish instead of failing.
            if head.startswith(b"\x1f\x8b"):
                raise _Unscannable(
                    "contains a compressed stream this check cannot finish reading"
                ) from None
            raise


def _scan_member(handle: IO[bytes], deadline: float, depth: int) -> str | None:
    """One archive member's bytes, scanned as if they were a file.

    Positional, because `flash.envscan.archive` names neither this module nor its keywords -- handing
    the scanner in is what lets the archive walk live there without importing the scan back.
    """
    return _scan_stream(handle, deadline=deadline, depth=depth)


def _credential_in_zip(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential in any readable member of a zip, or None."""
    return credential_in_zip(
        source,
        deadline=deadline,
        depth=depth,
        scan=_scan_member,
        refusal=_Unscannable,
        named=functools.partial(credential_in_name, deadline=deadline),
        metadata=_scan_member,
        member_limit=_MAX_ARCHIVE_MEMBERS,
    )


def _credential_in_ar(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential in any readable member of an ar or SVR4 cpio archive."""
    if isinstance(source, bytes):
        head = source[:110]
    else:
        with source.open("rb") as handle:
            head = handle.read(4122)
    # settlement means this handler actually walked an archive. without this decline, every other
    # file returned none here and incorrectly suppressed a later pdf or tar refusal as verified.
    if head.startswith((b"!<arch>\n", b"!<thin>\n")):
        walker = credential_in_ar
    elif _looks_like_cpio_header(head):
        walker = credential_in_cpio
    else:
        raise zipfile.BadZipFile("not an ar or cpio archive")
    return walker(
        source,
        deadline=deadline,
        depth=depth,
        scan=_scan_member,
        refusal=_Unscannable,
        named=functools.partial(credential_in_name, deadline=deadline),
        member_limit=_MAX_ARCHIVE_MEMBERS,
        size_limit=_MAX_NESTED_BUFFER_BYTES,
    )


def _credential_in_tar(source: Path | bytes, *, deadline: float, depth: int) -> str | None:
    """The kind of credential in any readable member of a tar, or None."""
    return credential_in_tar(
        source,
        deadline=deadline,
        depth=depth,
        scan=_scan_member,
        dispatch=_credential_in_container,
        refusal=_Unscannable,
        named=functools.partial(credential_in_name, deadline=deadline),
        member_limit=_MAX_ARCHIVE_MEMBERS,
    )


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
    if path.name.lower().endswith(".br"):
        raise _Unscannable("contains a Brotli sidecar this check cannot inspect")
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


def credential_in_name(name: str, *, deadline: float | None = None) -> str | None:
    """The kind of credential the path `name` itself carries, or None.

    A file whose *name* is the key leaks it through the archive's member list even when its
    contents are empty, and the published repo shows that name in its tree forever.

    `surrogatepass` rather than `surrogateescape`: the latter is a DECODE-only handler, so a name
    holding a lone surrogate raised `UnicodeEncodeError` out of a security check. Reached from the
    publish route that check turned a 400 into an uncaught 500, and reached from a tar member name
    it crashed the scan of an archive rather than reporting what was in it.

    A name gets a deadline, which is what enables the container inspection that file contents
    already get: without one `_decoded_container` returns None, so an encoded container in a name
    was matched only in its still-compressed form. A 66-character filename holding
    `base64(gzip(key))` published clean while decoding and inflating the published path recovered
    the whole key.

    `deadline` is the PACKAGE's budget when the caller has one. A name is bounded by the filesystem
    at a few hundred bytes, which reads like nothing to multiply -- but the expansion behind it is
    not: a legal 111-character basename encodes a bzip2 stream that inflates to 64 MiB, measured at
    9.6 seconds for a single name, and a package may carry thousands of them. Charging each one to
    a fresh 60-second budget made the package-wide bound advisory. A caller with no budget of its
    own -- the publish route, checking one name -- gets a fresh one.
    """
    if name.lower().endswith(".br"):
        raise _Unscannable("contains a Brotli sidecar this check cannot inspect")
    deadline = time.monotonic() + _MAX_DECOMPRESS_SECONDS if deadline is None else deadline
    for value in exact_name_values(name.encode("utf-8", "surrogatepass")):
        if kind := _credential_kind(value, deadline=deadline):
            return kind
    return None


def _redacted(name: str, *, deadline: float | None = None) -> str:
    """`name` masked or withheld by the container-aware display helper."""
    return redacted_name(
        name,
        credential_check=credential_in_name,
        deadline=deadline,
        refusals=(_Unscannable, _RunTooLongToExpand),
    )


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
            f"{_redacted(display.get(relative, relative), deadline=deadline)} could not be read to check it for "
            "credentials, so publishing it would commit unscanned content to a shared environment "
            "repository. Fix the permissions on it before publishing."
        ) from None

    # sorted so the member named in the refusal is the same one on every machine.
    for root, dirs, files in os.walk(package_root, onerror=unwalkable):
        dirs.sort()
        for name in sorted(dirs) + sorted(files):
            member = Path(root) / name
            relative = member.relative_to(package_root).as_posix()
            shown = _redacted(display.get(relative, relative), deadline=deadline)
            try:
                # the NAME is checked too, directories included: a file called `fslo_<key>.json`
                # publishes the key in the repository's file tree whatever its contents are.
                kind = credential_in_name(relative, deadline=deadline) or (
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
