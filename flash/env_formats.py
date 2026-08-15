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

from flash.env_deflate import _gzip_header_unfinished

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

# a legacy lzma-alone stream has a 13-byte structural header rather than fixed magic: one properties
# byte, a little-endian dictionary size, and an uncompressed size. The dictionary field is not a
# canonical-size marker: liblzma accepts every 32-bit value and rounds values below its internal
# minimum up, so restricting it to powers of two or three times one rejected a stream with 4,095
# there even though the same decoder recovered its key. The properties remain structural, and the
# caller still requires the candidate to produce decompressed output before recognising it.
_LZMA_ALONE_HEADER_BYTES = 13

# The size an lzma-alone stream declares when the encoder did not know it: all bits set.
_LZMA_ALONE_UNKNOWN_SIZE = (1 << 64) - 1

# a ceiling still rejects chance headers before they trigger a decompression probe. it sits far above
# the package size because the field declares expanded output: a valid 38 kib stream can legitimately
# expand past 256 mib, while random 64-bit values overwhelmingly remain above this bound.
_LZMA_ALONE_DECLARED_SIZE_CEILING = 1 << 56

# Containers this scan can RECOGNISE but not expand, because the stdlib has no decompressor for
# them. A `.jsonl.zst` -- the ordinary way a dataset shard ships -- holds its credential nowhere a
# pattern can see, exactly like a gzip, so treating it as final content was a silent bypass.
#
# Refused rather than expanded. Adding `zstandard` to a security scanner's dependencies to inspect
# a format no environment in the hub currently uses (0 of 8944 files) trades a real supply-chain
# surface for a hypothetical one, and the refusal is honest about what it means: not verified.
_UNEXPANDABLE_MAGIC = (
    # unix compress has no stdlib decoder, so approving its opaque lzw body published a key that
    # `gzip -dc` recovered intact. refusal is the same bounded answer used for every format here.
    (b"\x1f\x9d", "Unix compress"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
    # parquet has a head-anchored magic but no stdlib reader. filenames are irrelevant: a renamed
    # dataset is still opaque, while an ordinary file merely ending in `.parquet` remains clean.
    (b"PAR1", "Parquet"),
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
    # avro ocf starts with this four-byte marker. codecs such as snappy and zstd are optional
    # dependencies, so recognising the container must fail closed before any codec is required.
    (b"Obj\x01", "Avro"),
    # the complete framed snappy stream identifier. the binary prefix and declared six-byte payload
    # distinguish a stream from prose mentioning snappy, while the stdlib has no decoder for it.
    (b"\xff\x06\x00\x00sNaPpY", "Snappy"),
)

# Formats recognised ONLY at byte zero, kept apart from the list above rather than added to it.
#
# The unanchored search there admits any signature of `_SFX_MAGIC_BYTES` or more, which stands in
# for "distinctive enough to mean something at an arbitrary offset" -- and that proxy holds only
# for signatures carrying non-printable bytes. `Salted__` is eight printable characters and a word
# documentation uses, so putting it in that list would refuse a file merely DESCRIBING the format,
# the same false refusal the bare `Rar!` prefix caused. Length is the wrong test for it; position
# is the right one, and the format defines these bytes as the start of the file.
_ANCHORED_ONLY_MAGIC = (
    # The OpenSSL salted envelope: `Salted__`, an 8-byte salt, then ciphertext. This is what
    # `openssl enc -aes-256-cbc -pbkdf2 -salt` writes, and an encrypted credential file is an
    # ordinary thing to keep beside an environment. The body is ciphertext, so neither a pattern
    # nor a base64 decode can see the key inside -- verified with a real AES-256-CBC envelope
    # around a Freesolo key, which scanned clean and decrypted back to the whole key.
    #
    # Refused rather than decrypted, for the same reason as an encrypted ZIP member and an OpenPGP
    # message: the passphrase is not ours to have, and unverifiable is not clean. That the author
    # encrypted it is not evidence the publish is safe, since the hub copy is readable by everyone
    # the environment is and the passphrase travels beside the file about as often as not.
    (b"Salted__", "OpenSSL-encrypted"),
    # age's native header defines the start of its binary form. the armored form is handled by a
    # body-gated search because yaml commonly embeds it after a scalar header, while documentation
    # may mention the armor marker without carrying ciphertext.
    (b"age-encryption.org/v1", "age-encrypted"),
)

# Which recognised-but-uninspectable formats are ENCRYPTED rather than merely unexpandable. The
# refusal is the same either way; only the advice differs. Telling someone holding an `openssl enc`
# envelope that it is "an archive this check cannot expand" sends them looking for a decompressor
# that does not exist, when what they have to do is keep the ciphertext out of the package.
ENCRYPTED_FORMATS = frozenset(fmt for _magic, fmt in _ANCHORED_ONLY_MAGIC)

# Skippable frames: a standardized envelope both zstd and LZ4 allow before the real frame, used by
# seekable and metadata-bearing streams. The magic is `0x184D2A5x` little-endian for any low nibble
# `x`, followed by a 4-byte payload length. A head-only format check saw the skippable magic,
# matched neither list, and treated the compressed frame behind it as ordinary content.
_SKIPPABLE_FRAME_MAGIC = tuple(bytes((0x50 | low, 0x2A, 0x4D, 0x18)) for low in range(16))
_SKIPPABLE_FRAME_HEADER = 8

# How many skippable frames are walked before the stream is given up on. A real stream carries a
# handful; an unbounded walk over crafted headers would be a scan cost of its own.
_MAX_SKIPPABLE_FRAMES = 16

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
_LZMA_ALONE_OVERLAY_LEAD = b"\x5d"
_OVERLAY_MAGIC = (
    b"\x1f\x8b\x08",
    b"BZh",
    b"\xfd7zXZ\x00",
    # the default lzma1 properties byte; the structural filter below accepts any dictionary size.
    _LZMA_ALONE_OVERLAY_LEAD,
)
_OVERLAY_PROBE_BYTES = 1 << 16

# How many candidate offsets are probed before the search gives up. Each probe is a decompressor
# rejecting a few bytes, so the bound is only there to keep a file of nothing but fake magics from
# becoming a cost of its own -- and a REAL stream ends the search at the first hit. Measured 20
# chance occurrences of the gzip magic across 400 MiB of random bytes, of which 0 inflated.
_MAX_OVERLAY_CANDIDATES = 4096

# how much of a candidate's probe buckets it for the exact repeat check below. This prefix is only a
# cheap lookup key: candidates in one bucket are compared over the whole probe before one is skipped.
_OVERLAY_PROBE_KEY_BYTES = 64

# The "gave up with candidates unprobed" answer, distinct from both an offset and from None. A
# plain sentinel rather than a bool so no caller can confuse it with a falsy offset.
OVERLAY_UNPROBED = -1

# How much of a stream is read at a time while looking for an overlay, and while walking a zip's
# central directory. One record's variable-length fields are three 16-bit lengths, so a window this
# size always holds a whole record once it is refilled from that record's start.
_STREAM_WINDOW_BYTES = 1 << 20

# How much of a stream `_looks_like_textual` reads. A multi-byte UTF-8 character straddling the cut
# would decode-fail on the truncation rather than on the content, so the sample is taken at a
# 4 KiB boundary and the decode error is tolerated as "not text" -- which is the safe direction:
# it only ever costs a refusal that the heuristic already justified.
_TEXT_SAMPLE_BYTES = 4096

_ANSIBLE_VAULT_GUARD = b"$ANSIBLE_VAULT;"
_ANSIBLE_VAULT_HEX = frozenset(b"0123456789abcdefABCDEF")
_ANSIBLE_VAULT_PAYLOAD_HEX = frozenset(b"0123456789abcdef")
_ANSIBLE_VAULT_FIXED_FIELD_HEX = 64


def _has_ansible_vault(data: bytes) -> bool:
    """Whether `data` carries a supported Ansible Vault header and ciphertext body."""
    if _ANSIBLE_VAULT_GUARD not in data:
        return False
    lines = data.splitlines()
    for index, raw in enumerate(lines):
        header = raw.strip()
        if not header.startswith(_ANSIBLE_VAULT_GUARD):
            continue
        parts = header.split(b";")
        valid = parts in (
            [b"$ANSIBLE_VAULT", b"1.1", b"AES256"],
            [b"$ANSIBLE_VAULT", b"1.2", b"AES256"],
        )
        valid = valid or (
            len(parts) == 4
            and parts[:3] == [b"$ANSIBLE_VAULT", b"1.2", b"AES256"]
            and bool(parts[3])
        )
        if not valid:
            continue
        wrapped = bytearray()
        for body_line in lines[index + 1 :]:
            body = body_line.strip()
            if not body or len(body) % 2 or any(byte not in _ANSIBLE_VAULT_HEX for byte in body):
                break
            wrapped.extend(body)
        try:
            payload = bytes.fromhex(wrapped.decode("ascii"))
        except ValueError:
            continue
        fields = payload.split(b"\n")
        if len(fields) != 3:
            continue
        salt, digest, ciphertext = fields
        if (
            len(salt) == _ANSIBLE_VAULT_FIXED_FIELD_HEX
            and len(digest) == _ANSIBLE_VAULT_FIXED_FIELD_HEX
            and ciphertext
            and len(ciphertext) % 2 == 0
            and all(byte in _ANSIBLE_VAULT_PAYLOAD_HEX for field in fields for byte in field)
        ):
            return True
    return False


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


def _looks_like_lzma_alone(head: bytes) -> bool:
    """Whether `head` has the structural properties of an lzma-alone stream.

    The uncompressed-size field is what keeps this cheap. The properties byte and the dictionary
    size between them accept 35.6% of the bytes in an ordinary CSV, and every acceptance costs a
    decompression probe: widening the dictionary field alone took an 87 MB spreadsheet from a 57
    second scan to the 60 second budget, so a file that had always published was refused for
    "takes too long to decompress". A ceiling still rejects almost all chance 64-bit values, but it
    must bound expanded output rather than package size: a valid 38 KiB stream can legitimately
    declare more than 256 MiB of plaintext. The 2^56 ceiling admits that expansion while preserving
    the cheap discriminator that keeps ordinary CSV traffic out of the decompressor.
    """
    if len(head) < _LZMA_ALONE_HEADER_BYTES:
        return False
    properties = head[0]
    if properties >= 9 * 5 * 5:
        return False
    lc = properties % 9
    lp = (properties // 9) % 5
    if lc + lp > 4:
        return False
    # The dictionary field itself stays unrestricted: liblzma accepts every 32-bit value and rounds
    # a small one up, so a stream declaring 4,095 decodes perfectly and rejecting it published the
    # key it holds.
    declared = int.from_bytes(head[5:_LZMA_ALONE_HEADER_BYTES], "little")
    return declared == _LZMA_ALONE_UNKNOWN_SIZE or declared < _LZMA_ALONE_DECLARED_SIZE_CEILING


def _looks_compressed(head: bytes) -> bool:
    """Whether `head` begins a compressed container this scan can expand."""
    return (
        head.startswith(_COMPRESSED_MAGIC)
        or _looks_like_zlib(head)
        or (_looks_like_lzma_alone(head) and _decompresses(head[:_OVERLAY_PROBE_BYTES]))
    )


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
    version = int.from_bytes(store[4:8], "big")
    if version not in (1, 2):
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
        # the alias and timestamp exist in both versions, but only version 2 carries a certificate
        # type string. Reading one from a version-1 truststore consumed the DER length as text length
        # and refused a certificate-only store that the equivalent version-2 walk settled as clean.
        if not skip(read(2)) or not skip(8):
            return unwalked()
        if version == 2 and not skip(read(2)):
            return unwalked()
        if not skip(read(4)):
            return unwalked()
    return False


def _decompresses(probe: bytes) -> bool:
    """Whether `probe` really begins a compressed stream, proven by inflating some of it.

    The magic alone is not proof: three fixed bytes occur by chance in any large binary. Attempting
    the decompression is what separates a payload from a coincidence, and it is decisive -- a
    stream that yields a byte is a stream.

    For bzip2, "yielded no bytes yet" is NOT a rejection. bzip2 works in blocks of up to 900 KiB and
    emits nothing until it has a whole one, so a stream compressing more than the probe reads --
    200 KiB of incompressible data is enough -- returned empty from a perfectly valid decode and the
    candidate was dismissed. What separates it from coincidence there is that the decompressor
    consumed the entire probe and asked for more (`needs_input`) rather than raising: three bytes of
    chance do not survive 64 KiB of block decoding. Measured 0 acceptances across 2,000 random
    bodies behind a real `BZh` magic.

    gzip, xz and lzma-alone are deliberately NOT given the same treatment. They emit output within
    the probe, so "no output yet" really is a rejection for them. Accepting `needs_input` instead
    measured 97 false acceptances in 2,000 random gzip-magic bodies. For lzma-alone, requiring output
    after widening the dictionary field measured 0 acceptances in each of three 2,000-probe sets:
    random bodies behind structurally valid headers, fully random bodies, and bodies beginning with
    0x5d. That proof matters because its header is structural rather than fixed magic.
    """
    try:
        if probe.startswith(b"\x1f\x8b\x08"):
            # "No output" is only a rejection once the probe has actually REACHED the payload. A
            # gzip header carries optional fields -- an extra field of up to 65,535 bytes, a name,
            # a comment -- and a legal maximum-size FEXTRA runs past this 64 KiB probe on its own,
            # so a valid stream produced no output because the deflate bits were never in view.
            # Verified with a 65,535-byte extra field: the standalone stream was expanded, while
            # the same bytes behind a stub were passed over and the credential published.
            #
            # Only for a FULL probe. A chance magic near the end of a file returns the few bytes
            # that are left, and a header cannot outrun bytes that were never there to read -- with
            # a short probe admitted, a random shard whose last bytes happened to look like a magic
            # was refused once in 400 trials. A full probe means the remainder is genuinely out of
            # view: measured 0 acceptances in 300 random 64 KiB probes behind a chance magic.
            if len(probe) >= _OVERLAY_PROBE_BYTES and _gzip_header_unfinished(probe):
                return True
            return bool(zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(probe, 4096))
        if probe.startswith(b"BZh"):
            decompressor = bz2.BZ2Decompressor()
            return bool(decompressor.decompress(probe, 4096)) or decompressor.needs_input
        if probe.startswith(b"\xfd7zXZ\x00"):
            return bool(lzma.LZMADecompressor().decompress(probe, 4096))
        if _looks_like_lzma_alone(probe):
            return bool(lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(probe, 4096))
    except (OSError, EOFError, ValueError, zlib.error, lzma.LZMAError):
        return False
    return False


def _windows(source: Path | bytes) -> Iterator[tuple[int, bytes]]:
    """`source` in bounded windows as (absolute offset, bytes), overlapping so no magic is split."""
    if isinstance(source, bytes):
        yield 0, source
        return
    overlap = max(max(len(magic) for magic in _OVERLAY_MAGIC), _LZMA_ALONE_HEADER_BYTES) - 1
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
    number of WINDOWS walked rather than on a decoy count an attacker sets. Cutting a window's list
    short instead would put a real payload behind a cap the file controls. A file of 4,146 distinct
    decoys before a payload still has that payload examined in the same bounded window.

    What the file cannot be allowed to choose is repeated decompression of the same bytes. A 1 MiB
    file of adjacent gzip magics contains 349,522 candidates but only 22 distinct probe prefixes; at
    56 microseconds per decode, probing every repeat took 18 seconds. The probe comes from the window
    already in hand, and candidates sharing `_OVERLAY_PROBE_KEY_BYTES` enter an exact repeat check.

    The prefix is not proof of equality. A corrupted bzip2 stream and a valid copy agreed for 64
    bytes, diverged at byte 70, and the old cache skipped the valid copy -- the key behind it published
    while the valid stream alone was reported. A full-probe comparison now skips only identical
    decompressor input; one mismatch disables deduplication for that bucket rather than guessing that
    later candidates are repeats.

    A probe is a decompressor rejecting a few bytes -- measured 20 chance gzip magics across 400 MiB
    of random data, none of which inflated -- so the cap is far above what any real file reaches and
    a genuine payload is still found.

    Offset 0 is excluded deliberately: a stream that begins the file is not an overlay and the
    ordinary openers already read it.
    """
    probed = 0
    for base, window in _windows(source):
        found = sorted(
            at
            for magic in _OVERLAY_MAGIC
            for at in _offsets_of(window, magic)
            if base + at > 0
            and (
                magic != _LZMA_ALONE_OVERLAY_LEAD
                or _looks_like_lzma_alone(window[at : at + _LZMA_ALONE_HEADER_BYTES])
            )
        )
        seen: dict[bytes, int | None] = {}
        for at in found:
            probe = _probe_bytes(source, window, base, at)
            key = probe[:_OVERLAY_PROBE_KEY_BYTES]
            if key in seen:
                representative = seen[key]
                if representative is not None:
                    prior = _probe_bytes(source, window, base, representative)
                    if probe == prior:
                        continue
                    seen[key] = None
            else:
                seen[key] = at
            if _decompresses(probe):
                return base + at
        probed += len(found)
        if probed > _MAX_OVERLAY_CANDIDATES:
            return OVERLAY_UNPROBED
    return None


def _probe_bytes(source: Path | bytes, window: bytes, base: int, at: int) -> bytes:
    """The bytes at `at` used to decide whether a candidate really is a compressed stream.

    Served from the window already read whenever it holds the whole probe. Re-reading the file per
    candidate is what made a window of adjacent magics expensive, and the window is the same bytes:
    the read only happens where the probe would run off its end.
    """
    if len(window) - at >= _OVERLAY_PROBE_BYTES:
        return window[at : at + _OVERLAY_PROBE_BYTES]
    return _read_at(source, base + at, _OVERLAY_PROBE_BYTES) or b""


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


def _zip_end_record(source: Path | bytes) -> tuple[bytes, int, int] | None:
    """The tail, record offset in that tail, and tail offset for a zip end record."""
    total = len(source) if isinstance(source, bytes) else _file_size(source)
    if total is None:
        return None
    tail_at = max(0, total - _ZIP_TAIL_BYTES)
    tail = (
        source[tail_at:]
        if isinstance(source, bytes)
        else _read_at(source, tail_at, total - tail_at)
    )
    if tail is None:
        return None
    offset = tail.rfind(_ZIP_END_RECORD)
    if offset < 0 or len(tail) < offset + 22:
        return None
    return tail, offset, tail_at


def _zip_end_offset(source: Path | bytes) -> int | None:
    """The first byte after the zip end record and its declared comment, or None."""
    found = _zip_end_record(source)
    if found is None:
        return None
    tail, offset, tail_at = found
    comment = int.from_bytes(tail[offset + 20 : offset + 22], "little")
    end = tail_at + offset + 22 + comment
    # A suffix after the declared comment is outside the archive. Before this boundary was exposed,
    # a zlib-compressed key appended after an otherwise valid zip was never handed back to the scan.
    return end if end <= tail_at + len(tail) else None


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
    found = _zip_end_record(source)
    if found is None:
        return 0
    tail, offset, _tail_at = found
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
