"""Container and key-format recognition for the credential scan.

Every predicate here answers "what ARE these bytes" from the bytes themselves, never from a
filename: the extension is the publisher's choice, and a renamed archive is still an archive. They
are deliberately structural rather than name-based, and several are the difference between a
credential being expanded into view and being published intact.

Split from `flash.env_secrets` to keep that module under the file-size limit. The dependency runs
one way -- nothing here imports the pattern matching, so these can be tested on bytes alone.
"""

from __future__ import annotations

import bz2
import lzma
import zlib
from collections.abc import Iterator
from pathlib import Path

# The end-of-central-directory signature, and how much of a stream's tail to keep so it can be
# found. A zip's end record is last in the file, within 64 KiB of the end (the comment field is
# 16-bit), so this window always contains it.
_ZIP_END_RECORD = b"PK\x05\x06"
_ZIP_TAIL_BYTES = (64 << 10) + 64

# A central-directory record's fixed part, before its variable-length name, extra field and
# comment. Used to step from one record to the next while counting them.
_ZIP_CENTRAL_HEADER_BYTES = 46

# The zip64 end-of-central-directory record (56 bytes) plus its locator (20). They sit between the
# directory and the classic end record on a zip64 archive, which is what makes the shift computed
# from the classic record overshoot by exactly this much.
_ZIP64_END_BYTES = 76

# How many members of one archive are inspected before it is refused as unscannable. Defined here
# because the directory walk below needs a bound of its own, and re-exported through
# `flash.env_secrets`, which is where the policy is applied and where tests rebind it.
_MAX_ARCHIVE_MEMBERS = 100_000

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

# Containers this scan can RECOGNISE but not expand, because the stdlib has no decompressor for
# them. A `.jsonl.zst` -- the ordinary way a dataset shard ships -- holds its credential nowhere a
# pattern can see, exactly like a gzip, so treating it as final content was a silent bypass.
#
# Refused rather than expanded. Adding `zstandard` to a security scanner's dependencies to inspect
# a format no environment in the hub currently uses (0 of 8944 files) trades a real supply-chain
# surface for a hypothetical one, and the refusal is honest about what it means: not verified.
_UNEXPANDABLE_MAGIC = (
    (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"\x04\x22\x4d\x18", "lz4"),
    # The LZ4 LEGACY frame, a different magic rather than a variant of the one above. It is what
    # `lz4 -l` writes and what the Linux kernel build and several dataset tools still emit, and its
    # body is opaque exactly like the modern frame -- so naming only `04 22 4d 18` meant a legacy
    # frame was scanned as ordinary bytes and published with its credential intact.
    (b"\x02\x21\x4c\x18", "lz4"),
    # The full RAR 4 and RAR 5 signatures, not the bare `Rar!` prefix. Four printable characters
    # are ordinary prose -- a README opening "Rar! archives are not supported" was refused as an
    # archive -- and a real signature always carries the version bytes that follow.
    (b"Rar!\x1a\x07\x00", "rar"),
    (b"Rar!\x1a\x07\x01\x00", "rar"),
    # 7-Zip. Its body is opaque to the stdlib exactly like zstd, so a `.7z` holding a credential
    # was neither expanded nor refused: the compressed bytes were scanned as if they were content
    # and the archive published intact.
    (b"7z\xbc\xaf\x27\x1c", "7-zip"),
)

# Skippable frames: a standardized envelope both zstd and LZ4 allow before the real frame, used by
# seekable and metadata-bearing streams. The magic is `0x184D2A5x` little-endian for any low nibble
# `x`, followed by a 4-byte payload length. A head-only format check saw the skippable magic,
# matched neither list, and treated the compressed frame behind it as ordinary content.
_SKIPPABLE_FRAME_MAGIC = tuple(bytes((0x50 | low, 0x2A, 0x4D, 0x18)) for low in range(16))
_SKIPPABLE_FRAME_HEADER = 8

# How many skippable frames are walked before the stream is given up on. A real stream carries a
# handful; an unbounded walk over crafted headers would be a scan cost of its own.
_MAX_SKIPPABLE_FRAMES = 16

# How many OpenPGP marker packets are walked before the secret-key test. One is what an
# implementation emits; the bound keeps a file of nothing but repeated markers from being a scan
# cost. Exhausting it is NOT a pass: `_after_openpgp_markers` reports that it stopped early and
# the caller refuses, because a stream still sitting on a marker at the bound is undecided rather
# than clean. Returning the remainder silently let five markers hide a secret key.
_MAX_OPENPGP_MARKERS = 8

# What a Java KeyStore and a JCEKS store begin with. Named so the scan can tell in four bytes
# whether accumulating a stream is worth it at all.
_KEYSTORE_MAGIC = (b"\xfe\xed\xfe\xed", b"\xce\xce\xce\xce")

# How many keystore entries are walked looking for a private key. Reaching it means the walk never
# settled the question, which the caller treats as unscannable rather than clean.
#
# NOT "a handful". That guess refused a file every JDK ships: `/etc/ssl/certs/java/cacerts` holds
# 146 trusted certificates and no private key at all, so a bound of 64 turned the most ordinary
# keystore in existence into an unpublishable one. The walk costs a few integer reads per entry and
# is bounded by the buffer anyway -- every step advances at least four bytes or ends -- so the
# bound only has to sit above any real store rather than above a handful.
_MAX_JKS_ENTRIES = 4096

# The expandable compressed formats worth looking for BEHIND a stub, and how much of one is read to
# prove it is really a stream rather than three bytes of coincidence. A self-extracting shell
# archive -- what `makeself` writes, and what ships as a `.run` installer -- puts a script first and
# the payload after it, so recognition anchored at byte zero saw only `#!/bin/sh` and the compressed
# credential behind it was scanned as opaque bytes and published.
#
# Distinct from the unanchored RAR/7-Zip search, which only has to REFUSE. These can be expanded, so
# the payload is read rather than the publish blocked, and an offset that is wrong costs a failed
# open rather than a false refusal.
_OVERLAY_MAGIC = (b"\x1f\x8b\x08", b"BZh", b"\xfd7zXZ\x00")
_OVERLAY_PROBE_BYTES = 1 << 16

# How many candidate offsets are probed before the search gives up. Each probe is a decompressor
# rejecting a few bytes, so the bound is only there to keep a file of nothing but fake magics from
# becoming a cost of its own -- and a REAL stream ends the search at the first hit. Measured 20
# chance occurrences of the gzip magic across 400 MiB of random bytes, of which 0 inflated.
_MAX_OVERLAY_CANDIDATES = 4096

# The "gave up with candidates unprobed" answer, distinct from both an offset and from None. A
# plain sentinel rather than a bool so no caller can confuse it with a falsy offset.
OVERLAY_UNPROBED = -1

# How much of a stream is read at a time while looking for an overlay, and while walking a zip's
# central directory. One record's variable-length fields are three 16-bit lengths, so a window this
# size always holds a whole record once it is refilled from that record's start.
_STREAM_WINDOW_BYTES = 1 << 20

# The armor line of an OpenPGP message, as `gpg --armor --symmetric` and `--armor --encrypt` write
# it. The binary form is recognised structurally by `_is_openpgp_encrypted`, but the ARMORED form is
# the one an author actually commits -- it is what survives a copy-paste into a config -- and it is
# not a key block, so the private-key armor pattern never saw it. Its base64 body is ciphertext, so
# decoding it finds nothing either: a Freesolo key inside published clean.
#
# `SIGNED MESSAGE` is deliberately excluded. A clear-signed message carries its payload in the
# CLEAR, so the ordinary scan reads it, and refusing it would block a signed README.
_OPENPGP_MESSAGE_ARMOR = b"-----BEGIN PGP MESSAGE-----"

# How much of a stream `_looks_like_textual` reads. A multi-byte UTF-8 character straddling the cut
# would decode-fail on the truncation rather than on the content, so the sample is taken at a
# 4 KiB boundary and the decode error is tolerated as "not text" -- which is the safe direction:
# it only ever costs a refusal that the heuristic already justified.
_TEXT_SAMPLE_BYTES = 4096


def _after_skippable_frames(head: bytes) -> tuple[bytes, bool]:
    """`head` advanced past any leading zstd/LZ4 skippable frames, and whether it ran out.

    The flag is the whole point of the second return value. `head` is a bounded prefix of the
    stream, so a frame declaring a payload longer than what is left slices to empty -- which
    matches no magic and read as "not a compressed stream at all". A 70 KiB skippable frame in
    front of a zstd frame was enough to publish the credential behind it. Running out is not
    evidence of anything, so it is reported and the caller refuses instead.

    Returned rather than mutated in place so the caller keeps the original bytes for the checks
    that must see the true start of the file.
    """
    for _ in range(_MAX_SKIPPABLE_FRAMES):
        if not head.startswith(_SKIPPABLE_FRAME_MAGIC) or len(head) < _SKIPPABLE_FRAME_HEADER:
            return head, False
        size = int.from_bytes(head[4:_SKIPPABLE_FRAME_HEADER], "little")
        if _SKIPPABLE_FRAME_HEADER + size > len(head):
            return b"", True
        head = head[_SKIPPABLE_FRAME_HEADER + size :]
    # More frames than any real stream carries, and the format is still undecided.
    return b"", True


def _looks_compressed(head: bytes) -> bool:
    """Whether `head` begins a compressed container this scan can expand."""
    return head.startswith(_COMPRESSED_MAGIC) or _looks_like_zlib(head)


def _jks_private_key_entries(store: bytes) -> bool | None:
    """Whether `store` is a Java KeyStore holding a private-key entry.

    `keytool -genkeypair` writes a format no other check here understands: not PEM, not DER, not a
    container -- so a store holding a complete private key returned None, and neither `.jks` nor
    `.jceks` is in the filename exclusions. The key inside is password-encrypted, but the store is
    the credential: possession plus a guessable or shared store password is the whole secret, and
    it is exactly what gets committed beside a service config.

    Structural: the magic, a version of 1 or 2, an entry count, then every entry in turn. Tag 1 is
    a `PrivateKeyEntry` and tag 2 a `TrustedCertificateEntry` -- a store holding only trusted certs
    carries no secret and stays publishable, which is what separates the two.

    EVERY entry is walked, not just the first. Reading one tag was wrong about ordering rather than
    about rare stores: `keytool -importcert` followed by `-genkeypair` writes the trusted cert
    first, so a real two-entry store led with tag 2 and its private key published intact. The
    fields are variable-length but fully determined, so the walk is exact; anything that does not
    stay on an entry boundary means this is not the format it claims and the caller falls through
    to the other checks.

    JCEKS (`cececece`) is the same layout under a different magic and is walked identically. It was
    a distinct bypass: `keytool -storetype JCEKS` produced a store whose encrypted key matched no
    textual or DER check.
    """
    if len(store) < 16 or store[:4] not in _KEYSTORE_MAGIC:
        return False
    if int.from_bytes(store[4:8], "big") not in (1, 2):
        return False
    count, at, truncated = int.from_bytes(store[8:12], "big"), 12, False

    def read(width: int) -> int | None:
        """The next `width`-byte big-endian field, or None once the walk runs off the end."""
        nonlocal at, truncated
        if at < 0 or at + width > len(store):
            truncated = True
            return None
        value = int.from_bytes(store[at : at + width], "big")
        at += width
        return value

    def skip(length: int | None) -> bool:
        """Advance over a variable-length field, refusing a length that leaves the buffer."""
        nonlocal at, truncated
        if length is None:
            return False
        if length < 0:
            return False
        if at + length > len(store):
            truncated = True
            return False
        at += length
        return True

    def unwalked() -> bool | None:
        """What a stopped walk means: undecided if it ran out of bytes, otherwise not this format.

        The distinction is the whole fix. A trusted certificate whose DER runs past the buffer made
        the skip fail, which reported "not a keystore" -- so a private key stored BEHIND that
        certificate published intact, and no later chunk re-entered the parser. Running out of bytes
        proves nothing about what follows, and undecided is not clean.
        """
        return None if truncated else False

    for walked in range(count):
        if walked >= _MAX_JKS_ENTRIES:
            # More entries than the walk inspects, and the ones behind it are unread. Reporting
            # False here was the same fail-open as the OpenPGP marker bound: a store whose key sat
            # past the limit published intact. Undecided is not clean, so the caller refuses.
            return None
        tag = read(4)
        if tag in (1, 3):
            # 1 is a `PrivateKeyEntry`; 3 is the JCEKS `SecretKeyEntry` that `keytool -genseckey`
            # writes, whose payload is a symmetric key -- as much a credential as an asymmetric
            # one, and it published intact while the store around it was recognised.
            return True
        if tag != 2:
            # not an entry tag this format defines: the walk is off the rails, so this is not the
            # structure it claimed and the other checks get their turn
            return unwalked()
        # the alias, then an 8-byte creation timestamp, then the certificate's type and its DER
        if not skip(read(2)) or not skip(8) or not skip(read(2)) or not skip(read(4)):
            return unwalked()
    return False


def _decompresses(probe: bytes) -> bool:
    """Whether `probe` really begins a compressed stream, proven by inflating some of it.

    The magic alone is not proof: three fixed bytes occur by chance in any large binary. Attempting
    the decompression is what separates a payload from a coincidence, and it is decisive -- a
    stream that yields a byte is a stream.
    """
    try:
        if probe.startswith(b"\x1f\x8b\x08"):
            return bool(zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(probe, 4096))
        if probe.startswith(b"BZh"):
            return bool(bz2.BZ2Decompressor().decompress(probe, 4096))
        if probe.startswith(b"\xfd7zXZ\x00"):
            return bool(lzma.LZMADecompressor().decompress(probe, 4096))
    except (OSError, EOFError, ValueError, zlib.error, lzma.LZMAError):
        return False
    return False


def _windows(source: Path | bytes) -> Iterator[tuple[int, bytes]]:
    """`source` in bounded windows as (absolute offset, bytes), overlapping so no magic is split."""
    if isinstance(source, bytes):
        yield 0, source
        return
    overlap = max(len(magic) for magic in _OVERLAY_MAGIC) - 1
    try:
        with source.open("rb") as handle:
            at, carry = 0, b""
            while block := handle.read(_STREAM_WINDOW_BYTES):
                yield at - len(carry), carry + block
                at += len(block)
                carry = block[-overlap:]
    except OSError:
        return


def _overlay_offset(source: Path | bytes) -> int | None:
    """Where a compressed stream sits behind a stub in `source`, or None if none does.

    `OVERLAY_UNPROBED` is returned when the search gave up with candidates still unexamined, which
    is undecided rather than clean -- the caller turns it into a refusal.

    That third answer is the whole point of the bound being here. Returning None on exhaustion made
    the cap itself the bypass: padding a stub with more failing magics than the limit sent the real
    appended stream unprobed and its credential published. Since only a candidate that actually
    inflates ends the search, an attacker chooses how many decoys sit in front of the real one.

    Every candidate in a window is probed before the bound is consulted, so the limit bites on the
    number of WINDOWS walked rather than on a decoy count an attacker sets. A probe is a
    decompressor rejecting a few bytes -- measured 20 chance gzip magics across 400 MiB of random
    data, none of which inflated -- so probing them all is cheap and a real payload is still found.

    Offset 0 is excluded deliberately: a stream that begins the file is not an overlay and the
    ordinary openers already read it.
    """
    probed = 0
    for base, window in _windows(source):
        found = sorted(
            at for magic in _OVERLAY_MAGIC for at in _offsets_of(window, magic) if base + at > 0
        )
        for at in found:
            if _decompresses(_read_at(source, base + at, _OVERLAY_PROBE_BYTES) or b""):
                return base + at
        probed += len(found)
        if probed > _MAX_OVERLAY_CANDIDATES:
            return OVERLAY_UNPROBED
    return None


def _overlay_payload(source: Path | bytes, at: int, cap: int) -> bytes | None:
    """`source`'s bytes from `at`, empty if they exceed `cap`, or None if they cannot be read.

    Empty and None are deliberately different answers: too large to buffer is undecided, which the
    caller turns into a refusal, while unreadable is not this shape and the other handlers get a
    turn.
    """
    if isinstance(source, bytes):
        return b"" if len(source) - at > cap else source[at:]
    try:
        if source.stat().st_size - at > cap:
            return b""
        with source.open("rb") as handle:
            handle.seek(at)
            return handle.read()
    except OSError:
        return None


def _offsets_of(window: bytes, magic: bytes) -> Iterator[int]:
    """Every offset in `window` where `magic` appears."""
    at = window.find(magic)
    while at >= 0:
        yield at
        at = window.find(magic, at + 1)


def _has_openpgp_message_armor(window: bytes) -> bool:
    """Whether `window` carries the armor line of an encrypted OpenPGP message.

    Searched at any offset rather than anchored: armor is text, so it is ordinarily embedded -- a
    key pasted into a YAML block or appended to a config sits well past byte zero.

    No false-positive budget is spent on this. The line is 27 fixed bytes ending in five dashes; it
    does not occur in prose that is not quoting an actual message.
    """
    return _OPENPGP_MESSAGE_ARMOR in window


def _looks_like_textual(data: bytes) -> bool:
    """Whether `data` reads as text rather than as a compressed stream.

    Used only to keep a heuristic from becoming a refusal: the zlib header rule is satisfied by
    ordinary text such as `x = 1`, and refusing that file outright is worse than the bypass the
    rule exists to close. A deflate payload is high-entropy bytes -- it holds NUL and 0x80-0xff
    almost immediately -- so requiring the sample to be printable UTF-8 separates the two cleanly.

    Deliberately not the inverse of "looks compressed". This answers a narrower question, on a
    bounded sample, and is only ever consulted where the alternative is a false refusal.
    """
    sample = data[:_TEXT_SAMPLE_BYTES]
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return all(32 <= byte < 127 or byte in (9, 10, 13) for byte in sample)


def _looks_like_zlib(head: bytes) -> bool:
    """Whether `head` begins a raw zlib stream (RFC 1950).

    A bare zlib stream is what `zlib.compress` writes, and what a `.zz` shard, a PDF `FlateDecode`
    payload, and many application caches carry. It holds its credential nowhere a pattern can see,
    exactly like a gzip -- but the format list was magic-based and zlib has no fixed magic, so the
    bytes were scanned as if they were content and the stream published intact.

    Identified by the RFC 1950 header rule rather than by a literal: the method nibble must be 8
    (deflate), the window nibble at most 7, and the two bytes together must be a multiple of 31.
    That is about 11 bits of constraint, so an arbitrary file trips it roughly once in 2,000 --
    which is why it is checked LAST and only where a container is already suspected. A false
    positive costs a decompression attempt that fails and falls through, not a refusal.
    """
    return (
        len(head) >= 2
        and head[0] & 0x0F == 8
        and head[0] >> 4 <= 7
        and int.from_bytes(head[:2], "big") % 31 == 0
    )


def _is_openpgp_encrypted(head: bytes) -> bool | None:
    """Whether `head` begins an encrypted OpenPGP message, or None if that cannot be decided.

    Marker packets are stripped first, exactly as the secret-key test strips them. Normalizing for
    one predicate and not the other meant `ca 03 50 47 50` in front of a real encrypted message --
    which GnuPG decrypts happily -- made this see tag 10, report "not encrypted", and publish the
    ciphertext. Stopping ON a marker at the bound is undecided, not clean.
    """
    head, stopped_on_marker = _after_openpgp_markers(head)
    if stopped_on_marker:
        return None
    return _encrypted_message_head(head)


def _encrypted_message_head(head: bytes) -> bool | None:
    """Whether `head` begins an encrypted OpenPGP message.

    `gpg --symmetric` and `gpg --encrypt` write a session-key packet followed by an encrypted data
    packet. None of the secret-key tags appear, and the ciphertext matches no textual or DER check,
    so a credential wrapped this way read as ordinary bytes and published intact -- while an
    encrypted ZIP member, which is the same situation, is refused as unverifiable. Treating the two
    differently was the inconsistency; both are opaque, and opaque is not clean.

    Two packets are required, not one, AND the session-key packet's own fields must be valid. A
    lone tag byte is a single common value; even the two-packet shape alone fired on 1 in 4,000
    random 64-byte heads, which is a refusal rate an ordinary model shard would hit. Requiring the
    version and, for a symmetric-key packet, the cipher and S2K specifier to come from their
    registries takes that to 0 across 20,000 -- the structure is then one ordinary data does not
    fall into. Tags 1 and 3 are the public-key and symmetric-key session keys, tags 9, 18 and 20
    the encrypted data forms.
    """
    session, encrypted = frozenset((1, 3)), frozenset((9, 18, 20))
    if len(head) < 2:
        return False
    if head[0] & 0xC0 == 0xC0:  # new format: tag in the low six bits
        # The length is decoded here rather than through `_openpgp_body_length`, which keys on the
        # secret-key tag bytes: for a `0xC1`/`0xC3` header it falls through to a slice that is
        # empty and returns 0, so a wrong length would read as a stated one.
        tag, first = head[0] & 0x3F, head[1]
        if first < 192:
            length, header = first, 2
        elif first < 224:
            if len(head) < 3:
                return False
            length, header = ((first - 192) << 8) + head[2] + 192, 3
        elif first == 255:
            if len(head) < 6:
                return False
            length, header = int.from_bytes(head[2:6], "big"), 6
        else:
            return False  # a partial-body length: the packet has no single stated length
    elif head[0] & 0xC0 == 0x80:  # old format: tag in bits 5-2, length type in the low two
        tag, width = (head[0] >> 2) & 0x0F, (1, 2, 4, 0)[head[0] & 0x03]
        if not width:
            return False
        length, header = int.from_bytes(head[1 : 1 + width], "big"), 1 + width
    else:
        return False
    if tag not in session or length is None or len(head) < header + 1:
        return False
    # RFC 4880/9580 packet versions: 3 for a public-key ESK, 4/5/6 for a symmetric-key ESK.
    if head[header] not in (3, 4, 5, 6):
        return False
    if tag == 3:
        # symmetric-key ESK: a cipher from the registry, then an S2K specifier of a defined type
        if len(head) < header + 3:
            return False
        if head[header + 1] not in (1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13):
            return False
        if head[header + 2] not in (0, 1, 3, 4):
            return False
    at = header + length
    if at + 1 > len(head):
        # The session packet is longer than the bytes available. A real PKESK for an RSA-2048 key
        # is a few hundred bytes, so a fixed head could not reach the data packet behind it and
        # reported "not encrypted" -- publishing the ciphertext. The caller passes the whole chunk
        # now; still running out means the packet is longer than a chunk, which is undecided.
        return None
    nxt = head[at]
    following = nxt & 0x3F if nxt & 0xC0 == 0xC0 else (nxt >> 2) & 0x0F if nxt & 0xC0 == 0x80 else 0
    return following in encrypted


def _is_openpgp_secret_key(head: bytes) -> bool | None:
    """Whether `head` begins an unarmoured OpenPGP secret key packet.

    `gpg --export-secret-keys` without `--armor` writes raw packets: no text header for the PEM
    pattern to match, and the key material inside is neither base64 nor DER, so every other check
    here passes it through. The armoured form of the same key is caught by its header, which made
    the binary form the way to publish a private key intact.

    Matched on the packet header rather than on a byte pattern anywhere in the file. An OpenPGP
    packet's first byte has bit 7 set, and its tag is 5 (secret key) or 7 (secret subkey) -- old
    format `0x94-0x97` and `0x95`, new format `0xc5`/`0xc7` -- followed by the length and then
    version 4 or 6. The corresponding PUBLIC key tags are 6 and 14 (`0x98`/`0x99`, `0xc6`), so the
    tag alone distinguishes a secret key from the public half that is meant to be shared.

    Anchoring at offset 0 is what keeps this from firing on ordinary binaries. These are only a
    handful of constrained bytes, and searching for them anywhere would match roughly once per
    megabyte of arbitrary data -- a false refusal on every model shard in the package.

    The ALGORITHM byte is checked as well as the tag and version. Tag plus version alone is about
    twelve bits of signal, which measured 1 in 4,400 on random bytes: high enough that a package of
    binary shards would eventually be refused over nothing. The public-key algorithm is a small
    registry, and requiring it takes the same measurement to 1 in 108,000 (8/256 * 2/256 * 12/256
    predicts 1 in 87,000). That is a head-anchored test on the FIRST bytes of a member, not a
    search, so it is one draw per file rather than one per megabyte.
    """
    # RFC 4880 and 9580 public-key algorithms: RSA, Elgamal, DSA, ECDH/ECDSA/EdDSA, and the RFC
    # 9580 curve IDs. A byte outside this registry is not a key packet.
    algorithms = frozenset((1, 2, 3, 16, 17, 18, 19, 22, 25, 26, 27, 28))
    head, stopped_on_marker = _after_openpgp_markers(head)
    if stopped_on_marker:
        # Still on a marker at the bound: what follows is unread, not absent. This module holds no
        # exception type of its own -- it is imported BY the scanner, never the reverse -- so the
        # undecided case is reported and `_scan_stream` turns it into the refusal.
        return None
    if not head:
        return False
    tag_old, tag_new = head[0] & 0xFC, head[0]
    if tag_old not in (0x94, 0x9C) and tag_new not in (0xC5, 0xC7):
        return False
    # Old format carries the length in 1, 2 or 4 bytes as selected by the low two bits.
    #
    # New format is NOT always one byte: RFC 4880 encodes it in one octet below 192, two up to
    # 8383, and five above that. Assuming one put the version byte at the wrong offset for any
    # packet of 192 bytes or more -- which is every RSA secret key, and anything Sequoia, RNP or
    # `--use-new-packet-format` writes -- so those returned false and published intact.
    lengths = {0x00: 2, 0x01: 3, 0x02: 5}
    if tag_new in (0xC5, 0xC7):
        first = head[1] if len(head) > 1 else 0
        offset = 2 if first < 192 else (3 if first < 224 else (6 if first == 0xFF else 0))
    else:
        offset = lengths.get(head[0] & 0x03, 0)
    if offset == 0 or len(head) <= offset or head[offset] not in (4, 6):
        return False
    # The declared body length must actually reach the fields read below. A packet claiming a
    # one-byte body cannot hold a version, a four-byte timestamp and an algorithm, so reading them
    # takes bytes from BEYOND the packet -- `c5 01 04 00 00 00 00 01` is an ordinary binary that
    # was refused as a private key. Old-format length type 3 is indeterminate, which declares no
    # length at all, and is already excluded by the table above.
    body = _openpgp_body_length(head, offset)
    if body is not None and body < 6:
        return False
    # then a four-byte creation timestamp, then the algorithm
    return len(head) > offset + 5 and head[offset + 5] in algorithms


def _after_openpgp_markers(head: bytes) -> tuple[bytes, bool]:
    """`head` past any leading OpenPGP marker packets.

    A marker packet (tag 10, body `PGP`) is a legal no-op that RFC 9580 requires implementations to
    skip, and GnuPG parses `<marker><secret key>` as the secret key it is. Anchoring the secret-key
    test at offset 0 meant prepending five bytes -- `ca 03 50 47 50` -- moved the real packet out
    from under the check, and the remaining key material matched no textual or DER pattern, so a
    binary secret key published intact.

    Only markers are skipped, not arbitrary packets. Walking any packet header would let a crafted
    prefix of ordinary binary lead the scan to a false secret-key match deep inside a model shard;
    the marker is a fixed five bytes with a body that must be exactly `PGP`, so recognising it
    costs no signal. Both packet formats are accepted because either may carry tag 10.

    Returns the remainder and whether the walk STOPPED on a marker. Reporting only the remainder
    made the bound its own bypass: a key behind five markers left the walk sitting on the fifth,
    the secret-key test ran against a marker header and said no, and the file published. A stream
    still on a marker at the bound is undecided, and the caller refuses it.
    """
    for _ in range(_MAX_OPENPGP_MARKERS):
        if head[:2] in (b"\xca\x03", b"\xa8\x03") and head[2:5] == b"PGP":
            head = head[5:]
        else:
            return head, False
    return head, head[:2] in (b"\xca\x03", b"\xa8\x03") and head[2:5] == b"PGP"


def _openpgp_body_length(head: bytes, offset: int) -> int | None:
    """The body length an OpenPGP packet header declares, or None if it is not stated."""
    if head[0] in (0xC5, 0xC7):
        first = head[1]
        if first < 192:
            return first
        if first < 224:
            return ((first - 192) << 8) + head[2] + 192 if len(head) > 2 else None
        return int.from_bytes(head[2:6], "big") if len(head) > 5 else None
    return int.from_bytes(head[1:offset], "big")


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
    if magic.startswith((b"ustar\x0000", b"ustar  \x00", b"ustar\x00")):
        return True
    # V7 -- the original pre-POSIX format, still written by `tar --format=v7` -- has NO magic at
    # all: offset 257 is zero padding. Testing the magic alone left it unrecognised, so a v7 tar
    # holding a gzipped credential was never enumerated and published intact. Its header is
    # verifiable anyway: the checksum at offset 148 covers the first 512 bytes with that field
    # read as spaces, which is a structural property no ordinary file satisfies by chance.
    return _has_tar_checksum(source)


def _has_tar_checksum(source: Path | bytes) -> bool:
    """Whether `source` begins a tar header whose stored checksum verifies.

    The check is what makes magic-less V7 detection safe: an arbitrary binary would have to carry
    a valid octal checksum of its own first 512 bytes at exactly offset 148 to be mistaken for one.
    """
    try:
        if isinstance(source, Path):
            with source.open("rb") as handle:
                block = handle.read(512)
        else:
            block = source[:512]
    except OSError:
        return False
    if len(block) < 512:
        return False
    field = block[148:156]
    try:
        stored = int(field.split(b"\x00")[0].split()[0] or b"-1", 8)
    except (ValueError, IndexError):
        return False
    # the checksum is computed with its own field taken as eight spaces
    blanked = block[:148] + b" " * 8 + block[156:]
    # Historic tars signed the header bytes, so both sums are accepted. The signed one is summed
    # as integers rather than rebuilt through `bytes()`: a negative value is not a byte, so any
    # header holding a byte above 127 -- a UTF-8 filename is enough -- raised `ValueError` here.
    # That propagated out of `_looks_like_tar`, where the zip handler caught it as an unreadable
    # member and skipped it, so a v7 tar with a non-ASCII name inside a zip published intact.
    signed = sum(byte - 256 if byte > 127 else byte for byte in blanked)
    return stored in (sum(blanked), signed)


def _has_zip_end_record(tail: bytes) -> bool:
    """Whether `tail` ends with a STRUCTURALLY valid zip end-of-central-directory record.

    The bare four-byte signature is not enough to refuse a publish on. Those bytes occur by chance
    about once per 4 GB of arbitrary data, and this test runs over the last 64 KiB of every member
    too large to buffer -- so a model shard that happened to contain them was refused as an
    unverifiable archive, with a message about an archive that was never there.

    A real record is self-describing: the 2-byte comment length at offset 20 states exactly how
    many bytes follow it. Requiring that to agree with what is actually there costs nothing on a
    genuine zip and drops essentially every chance hit, since random bytes would have to encode
    their own remaining length correctly.
    """
    offset = tail.rfind(_ZIP_END_RECORD)
    while offset >= 0:
        if len(tail) >= offset + 22:
            comment = int.from_bytes(tail[offset + 20 : offset + 22], "little")
            if len(tail) - (offset + 22) == comment:
                return True
        offset = tail.rfind(_ZIP_END_RECORD, 0, offset)
    return False


def _zip_concat_shift(
    source: Path | bytes, tail: bytes, offset: int, size: int, start: int
) -> int | None:
    """How far the real central directory sits past the offset the end record states, or None.

    Zero for an ordinary zip. Non-zero when bytes precede the archive -- a self-extracting stub, or
    a zip appended to another file -- which shifts every recorded offset by the same amount. This
    mirrors what `ZipFile` calls `concat`, so the walk reads the same directory the constructor
    parses rather than the one the (attacker-controlled) offset field points at.
    """
    total = len(source) if isinstance(source, bytes) else _file_size(source)
    if total is None:
        return None
    # where the end record actually is, in whole-file coordinates
    record_at = total - len(tail) + offset
    shift = record_at - size - start
    return shift if shift >= 0 else None


def _file_size(source: Path) -> int | None:
    """`source`'s size in bytes, or None if it cannot be stat'd."""
    try:
        return source.stat().st_size
    except OSError:
        return None


def _read_at(source: Path | bytes, start: int, size: int) -> bytes | None:
    """`size` bytes of `source` from `start`, or None if they cannot be read."""
    if start < 0:
        return None
    try:
        if isinstance(source, Path):
            with source.open("rb") as handle:
                handle.seek(start)
                return handle.read(size)
    except OSError:
        return None
    return source[start : start + size] if isinstance(source, bytes) else None


def _zip_directory_entries(
    source: Path | bytes, tail: bytes, offset: int, limit: int
) -> int | None:
    """How many central-directory records a zip really holds, or None if it cannot be walked.

    Counted by stepping over the records rather than by trusting the end record's count field,
    which is attacker-controlled and does not govern what `ZipFile` actually parses.

    Stops one past `limit`: the caller only needs to know the bound is exceeded, and walking an
    unbounded directory would make this counter the very cost it exists to prevent.
    """
    size = int.from_bytes(tail[offset + 12 : offset + 16], "little")
    start = int.from_bytes(tail[offset + 16 : offset + 20], "little")
    if size == 0 or size == 0xFFFFFFFF or start == 0xFFFFFFFF:
        return None
    # A self-extracting archive carries an executable stub before the zip, so every offset stored
    # inside is short by the stub's length. `ZipFile` computes that shift and still materializes
    # every entry; reading the recorded offset literally landed the walk in the stub, which is not
    # a directory record, so the walk gave up and the forged count in the end record was trusted --
    # a 102-byte stub made a 500-entry archive report one member.
    #
    # The shift is what the end record itself proves: the directory ends where the record begins,
    # so its true start is that position minus its size.
    if (shift := _zip_concat_shift(source, tail, offset, size, start)) is None:
        return None
    # Both candidate starts are tried, the recorded one first. The computed shift assumes the
    # directory ends where the classic end record begins, which a zip64 archive breaks: its own
    # end record and locator sit in between, so the shift came out as their combined length (76
    # bytes) on an archive that had no stub at all and the walk landed mid-record. Reading the
    # recorded offset settles that case, and the shifted one still covers a genuine stub.
    #
    # A third candidate covers the two TOGETHER. An SFX zip64 has both a stub (so the recorded
    # offset is short) and the zip64 record and locator between its directory and the classic end
    # record (so the computed shift overshoots by their combined length). Neither of the first two
    # candidates lands on the directory, and the walk fell back to the forged count.
    # Every candidate is WALKED, not merely sniffed, and the largest count any of them yields wins.
    # Selecting on the leading four bytes and committing to that choice was a bypass: a decoy
    # `PK\x01\x02` planted in the stub at the unshifted offset won the sniff, its walk then failed
    # on the second record, and the failure was reported as "cannot be walked" -- which makes the
    # caller fall back to the forged count in the end record. A stub decoy plus a count patched to
    # 1 made a real 500-entry archive report one member while `ZipFile` still materialized all 500.
    #
    # Taking the maximum rather than the first success is what makes a decoy useless: it can add a
    # candidate that walks to a small number, but it cannot lower what the real directory walks to.
    counts = [
        walked
        for candidate in dict.fromkeys((start, start + shift, start + shift - _ZIP64_END_BYTES))
        if (walked := _walk_directory(source, candidate, size, limit)) is not None
    ]
    return max(counts) if counts else None


def _walk_directory(source: Path | bytes, at: int, size: int, limit: int) -> int | None:
    """How many central-directory records sit at `at`, or None if that is not a directory.

    Stops one past `limit`: the caller only needs to know the bound is exceeded, and walking an
    unbounded directory would make this counter the very cost it exists to prevent.

    Read in bounded windows rather than as one `size`-byte slice. `size` comes from the end record,
    which is attacker-controlled, so a forged 32-bit value covering most of the file made this
    counter allocate roughly the whole package -- the very cost it exists to avoid, arriving through
    the preflight instead of through `ZipFile`. Each record's variable-length fields are three
    16-bit lengths, so one always fits in a window and the walk refills from the record's own start.
    """
    if at < 0 or size <= 0:
        return None
    count, cursor, window, base = 0, 0, b"", at
    while cursor < size:
        if cursor + _ZIP_CENTRAL_HEADER_BYTES > len(window):
            # Refill from THIS record's start, so a record straddling the previous window is whole.
            base += cursor
            window = _read_at(source, base, min(_STREAM_WINDOW_BYTES, size - (base - at))) or b""
            cursor = 0
            if len(window) < _ZIP_CENTRAL_HEADER_BYTES:
                break
        if window[cursor : cursor + 4] != b"PK\x01\x02":
            return None  # not a directory record, so this walk proves nothing
        name, extra, comment = (
            int.from_bytes(window[cursor + off : cursor + off + 2], "little")
            for off in (28, 30, 32)
        )
        cursor += _ZIP_CENTRAL_HEADER_BYTES + name + extra + comment
        count += 1
        if count > limit:
            break
    return count or None


def _zip_member_count(source: Path | bytes, limit: int = _MAX_ARCHIVE_MEMBERS) -> int:
    """How many members a zip holds, or 0 if that cannot be read.

    Read without parsing the archive so an absurd count can be refused before `ZipFile`
    materializes that many `ZipInfo` objects. `limit` is passed in rather than read from the module
    so the caller's bound -- which tests rebind -- governs how far the directory walk goes.

    A count of `0xffff` is the zip64 sentinel: the true count lives in the zip64 record, which is
    read instead. Treating the sentinel as a literal 65,535 would wave through the archives that
    carry more members than a 16-bit field can express, which are the ones this limit is for --
    but reporting it as over the bound would refuse an ordinary 70,000-member archive, which is
    under the bound and perfectly legitimate. Only the real count settles it.

    Reporting 0 for an unreadable record is not a bypass: the constructor below then rejects the
    archive, and the per-member loop still enforces the same bound.
    """
    tail = source[-_ZIP_TAIL_BYTES:] if isinstance(source, bytes) else b""
    if isinstance(source, Path):
        try:
            with source.open("rb") as handle:
                handle.seek(max(0, source.stat().st_size - _ZIP_TAIL_BYTES))
                tail = handle.read()
        except OSError:
            return 0
    offset = tail.rfind(_ZIP_END_RECORD)
    if offset < 0 or len(tail) < offset + 12:
        return 0
    total = int.from_bytes(tail[offset + 10 : offset + 12], "little")
    # The claimed count is not trusted on its own. `ZipFile` walks the central directory by its
    # SIZE (`while total < size_cd`), not by this count, so patching both count fields of a real
    # 500-entry archive down to 1 left all 500 entries materialized while this bound saw one
    # member. The directory itself is walked instead, which is the same bytes the constructor
    # reads but without allocating a `ZipInfo` per entry -- the cost this bound exists to avoid.
    # Only a WALKED count may exceed the bound. The claimed field is attacker-controlled and the
    # record holding it is found by searching the tail, so any file can carry one: a tar member of
    # ordinary text plus `PK\x05\x06` and a zip64 record claimed 100,001 members and made a clean
    # tar -- holding no zip at all -- unpublishable. A claim the directory does not corroborate is
    # not evidence of members; it is evidence of four bytes.
    #
    # This does not weaken the bound it exists for. A real oversized archive HAS a real directory,
    # so the walk reaches it and the refusal still fires; and an archive whose directory cannot be
    # walked is rejected by `ZipFile` itself, with the per-member loop enforcing the same limit.
    walked = _zip_directory_entries(source, tail, offset, limit)
    if walked is None:
        return min(total, limit)
    total = max(total, walked)
    if total != 0xFFFF:
        return total
    # zip64 end-of-central-directory: the 8-byte total sits 32 bytes into its own record
    zip64 = tail.rfind(b"PK\x06\x06")
    if zip64 < 0 or len(tail) < zip64 + 40:
        # Claims zip64 but carries no zip64 record, so the count is unknown. Reported as 0 like
        # every other unreadable record rather than as "over the limit": these bytes may not be a
        # zip at all, and refusing here made a tar carrying a forged end record abandon the scan
        # before the tar handler ran. `ZipFile` rejects a genuinely malformed archive on its own,
        # and the per-member loop still enforces the same bound on one that opens.
        return max(total, walked or 0)
    # The walked count wins here too. A zip64 archive states its total in 64 bits, so forging that
    # field down needs no `0xffff` sentinel and the walk above was being discarded on exactly the
    # archives large enough for the bound to matter -- a 70,000-entry zip64 patched to claim one
    # member reported one while `ZipFile` still materialized all 70,000.
    claimed = int.from_bytes(tail[zip64 + 32 : zip64 + 40], "little")
    return max(claimed, walked or 0)
