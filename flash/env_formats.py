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
# implementation emits; the bound is what keeps a file of nothing but repeated markers from being
# a scan cost, and it is checked against the 24-byte head so it can never walk past that anyway.
_MAX_OPENPGP_MARKERS = 4

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


def _jks_private_key_entries(head: bytes) -> bool:
    """Whether `head` begins a Java KeyStore holding a private-key entry.

    `keytool -genkeypair -storetype JKS` writes a format no other check here understands: not PEM,
    not DER, not a container -- so a store holding a complete private key returned None, and `.jks`
    is not in the filename exclusions either. The key inside is password-encrypted, but the store
    is the credential: possession plus a guessable or shared store password is the whole secret,
    and it is exactly what gets committed beside a service config.

    Structural: the `feedfeed` magic, a version of 1 or 2, an entry count, then each entry's tag.
    Tag 1 is a `PrivateKeyEntry` and tag 2 a `TrustedCertificateEntry` -- a store holding only
    trusted certs carries no secret and stays publishable, which is what separates the two. Only
    the FIRST entry's tag is read, since the fields after it are variable-length and walking them
    would mean parsing the whole store to answer a question the first entry usually settles.
    """
    if len(head) < 16 or head[:4] != b"\xfe\xed\xfe\xed":
        return False
    version = int.from_bytes(head[4:8], "big")
    count = int.from_bytes(head[8:12], "big")
    return version in (1, 2) and count > 0 and int.from_bytes(head[12:16], "big") == 1


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
    head = _after_openpgp_markers(head)
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


def _after_openpgp_markers(head: bytes) -> bytes:
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
    """
    for _ in range(_MAX_OPENPGP_MARKERS):
        if head[:2] in (b"\xca\x03", b"\xa8\x03") and head[2:5] == b"PGP":
            head = head[5:]
        else:
            break
    return head


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
        if (walked := _walk_directory(_read_at(source, candidate, size), limit)) is not None
    ]
    return max(counts) if counts else None


def _walk_directory(directory: bytes | None, limit: int) -> int | None:
    """How many central-directory records `directory` holds, or None if it is not one.

    Stops one past `limit`: the caller only needs to know the bound is exceeded, and walking an
    unbounded directory would make this counter the very cost it exists to prevent.
    """
    if directory is None or directory[:4] != b"PK\x01\x02":
        return None
    count, cursor = 0, 0
    while cursor + _ZIP_CENTRAL_HEADER_BYTES <= len(directory):
        if directory[cursor : cursor + 4] != b"PK\x01\x02":
            return None  # not a directory record, so this walk proves nothing
        name, extra, comment = (
            int.from_bytes(directory[cursor + at : cursor + at + 2], "little")
            for at in (28, 30, 32)
        )
        cursor += _ZIP_CENTRAL_HEADER_BYTES + name + extra + comment
        count += 1
        if count > limit:
            break
    return count


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
    walked = _zip_directory_entries(source, tail, offset, limit)
    if walked is not None:
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
