"""Container and key-format recognition for the credential scan.

Every predicate here answers "what ARE these bytes" from the bytes themselves, never from a
filename: the extension is the publisher's choice, and a renamed archive is still an archive. They
are deliberately structural rather than name-based, and several are the difference between a
credential being expanded into view and being published intact.

Split from `flash.env_secrets` to keep that module under the file-size limit. The dependency runs
one way -- nothing here imports the pattern matching, so these can be tested on bytes alone.
"""

from __future__ import annotations

from pathlib import Path

# The end-of-central-directory signature, and how much of a stream's tail to keep so it can be
# found. A zip's end record is last in the file, within 64 KiB of the end (the comment field is
# 16-bit), so this window always contains it.
_ZIP_END_RECORD = b"PK\x05\x06"
_ZIP_TAIL_BYTES = (64 << 10) + 64

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
    (b"Rar!", "rar"),
)


def _looks_compressed(head: bytes) -> bool:
    """Whether `head` begins a compressed container this scan can expand."""
    return head.startswith(_COMPRESSED_MAGIC)


def _is_openpgp_secret_key(head: bytes) -> bool:
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
    # then a four-byte creation timestamp, then the algorithm
    return len(head) > offset + 5 and head[offset + 5] in algorithms


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
    return stored in (
        sum(blanked),
        sum(bytes(byte - 256 if byte > 127 else byte for byte in blanked)),
    )


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


def _zip_member_count(source: Path | bytes) -> int:
    """The member count a zip's end-of-central-directory record claims, or 0 if unreadable.

    Read from the record rather than from a parsed archive so an absurd count can be refused
    before `ZipFile` materializes that many `ZipInfo` objects.

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
        return 0
    return int.from_bytes(tail[zip64 + 32 : zip64 + 40], "little")
