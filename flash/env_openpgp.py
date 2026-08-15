"""OpenPGP packet and armor recognition for the credential scan.

A binary `gpg --export-secret-keys` carries no text header for a pattern to match and no ASN.1 for
the DER checks, so a private key in that form had nothing between it and the hub. These functions
read the packet grammar itself: what a stream begins with, what its sequence holds, and whether an
armored message is ciphertext this check cannot see into.

Split from `flash.env_formats` to keep both modules under the file-size limit. The dependency runs
one way -- nothing here imports the scanning -- so these can be tested on bytes alone.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

_MAX_OPENPGP_MARKERS = 8

# How many packets of an OpenPGP sequence are walked looking for secret key material. A public key
# block is a key, its subkeys, their signatures and a handful of user IDs -- a dozen or so packets
# -- and a concatenated public-then-secret export doubles that. The bound is what keeps a crafted
# chain of tiny packets from being an unbounded walk; the walk itself only ever runs on bytes that
# already parsed as OpenPGP.
_MAX_OPENPGP_PACKETS = 64

# The largest body a packet may declare and still be walked as OpenPGP. Beyond it the header is read
# as ordinary binary that happens to carry the shape of one, and the walk ends rather than reporting
# the sequence undecided.
#
# Bit 7 alone is set on half of all random bytes, so the walk entered on ordinary binary, read what
# followed as a body length -- averaging two gigabytes -- and reported "this packet runs past my
# buffer". The caller turns that into a refusal, so 11.7% of megabyte chunks of ordinary binary were
# refused when more data followed: a model shard or a padded archive member blocked over noise.
#
# 64 MiB is set against `_MAX_NESTED_BUFFER_BYTES`, the most this scan ever holds: a body larger than
# that could not be examined even in principle, so calling it a packet boundary claims a precision
# the scan does not have. Real key material is nowhere near it -- a measured RSA-4096
# `gpg --export-secret-keys` declares at most 1,816 bytes -- and the margin leaves room for the large
# legitimate packets a keyring carries, such as a photo user ID, without letting noise through.
#
# This cannot hide a key behind a huge packet: a real one that large would have to be READ to reach
# whatever follows it, and nothing here can read it, so the honest statement about those bytes is
# the one the literal scan already makes rather than a boundary invented from an unverifiable length.
_MAX_OPENPGP_BODY_BYTES = 64 << 20

# The packet types a partial body length may be used on. RFC 4880 section 4.2.2.4 restricts it to
# the data packets -- literal (11), compressed (8), symmetrically encrypted (9) and its integrity
# protected form (18) -- so a partial length on any other tag is not a packet this format defines.
#
# Gating on the tag is what keeps random binary out. A partial chunk may declare up to 1 GiB, so a
# random first octet in 224-254 lands in the partial branch constantly, and treating every one of
# them as an unfinished sequence refused ordinary binaries: measured 3 of 64 random megabyte chunks
# before this, and 0 of 256 after.
_PARTIAL_LENGTH_TAGS = frozenset({8, 9, 11, 18})

# Yielded by the packet walk in place of a boundary when a packet's declared body runs past the
# bytes in hand. A distinct object rather than a flag so the sequence test can tell "no secret key
# in this sequence" from "the sequence continues somewhere this never read", which are the same
# `False` otherwise. Never a real packet: no OpenPGP packet has a first byte with bit 7 clear.
_TRUNCATED_PACKET = b"\x00"

# Yielded when the packet-count bound is reached with bytes still unwalked. Separate from
# `_TRUNCATED_PACKET` because it is undecided unconditionally: the unexamined bytes are in hand,
# not beyond the buffer, so no statement about what follows the chunk can make them absent.
_UNWALKED_REMAINDER = b"\x01"

# What a Java KeyStore and a JCEKS store begin with. Named so the scan can tell in four bytes

_MAX_PGP_RECIPIENTS = 256


# "The packet runs past the bytes available", distinct from both an offset and from None. A packet
# longer than what was read says nothing about what follows it, so the caller reports undecided.

_PACKET_PAST_BUFFER = -1

# The expandable compressed formats worth looking for BEHIND a stub, and how much of one is read to
# prove it is really a stream rather than three bytes of coincidence. A self-extracting shell
# archive -- what `makeself` writes, and what ships as a `.run` installer -- puts a script first and
# the payload after it, so recognition anchored at byte zero saw only `#!/bin/sh` and the compressed
# credential behind it was scanned as opaque bytes and published.
#
# Distinct from the unanchored RAR/7-Zip search, which only has to REFUSE. These can be expanded, so
# the payload is read rather than the publish blocked, and an offset that is wrong costs a failed
# open rather than a false refusal.

_OPENPGP_MESSAGE_ARMOR = b"-----BEGIN PGP MESSAGE-----"

# The armor line followed by an actual armored BODY. RFC 4880 §6.2 puts optional headers, then a
# blank line, then base64 -- so a real message always has a run of base64 behind the line, while a
# README saying "look for a block starting with -----BEGIN PGP MESSAGE-----" has prose or nothing.
# The line alone refused documentation that merely names it, which is a false alarm on exactly the
# file most likely to mention it.
#
# One base64 line of 32 characters is the bar. The shortest real message is a session-key packet
# and a data packet, hundreds of bytes of base64 across several lines, so this cannot exclude a
# genuine one -- and prose does not carry a 32-character base64 run on the line after the marker.
_OPENPGP_ARMORED_BODY = re.compile(
    rb"-----BEGIN PGP MESSAGE-----[^\n]*\n(?:[!-9;-~][^\n]*\n)*\s*\n?[A-Za-z0-9+/]{32}"
)

# age armor is text and is commonly embedded after a yaml scalar header, so byte-zero recognition
# misses it. the body requirement keeps a readme that merely names the marker publishable, matching
# the openpgp rule above rather than treating prose as ciphertext.
_AGE_FILE_ARMOR = b"-----BEGIN AGE ENCRYPTED FILE-----"
_AGE_ARMORED_BODY = re.compile(
    rb"-----BEGIN AGE ENCRYPTED FILE-----[^\r\n]*\r?\n[ \t]*[A-Za-z0-9+/]{32}"
)

# How much of a stream `_looks_like_textual` reads. A multi-byte UTF-8 character straddling the cut
# would decode-fail on the truncation rather than on the content, so the sample is taken at a
# 4 KiB boundary and the decode error is tolerated as "not text" -- which is the safe direction:
# it only ever costs a refusal that the heuristic already justified.


def _has_openpgp_message_armor(window: bytes) -> bool:
    """Whether `window` carries the armor line of an encrypted OpenPGP message.

    Searched at any offset rather than anchored: armor is text, so it is ordinarily embedded -- a
    key pasted into a YAML block or appended to a config sits well past byte zero.

    The BODY is required, not just the line. The line alone read as ciphertext in a README that
    merely names it -- "look for a block starting with -----BEGIN PGP MESSAGE-----" -- which is a
    false refusal on the document most likely to mention it, and on a file holding no ciphertext at
    all. A real message always has base64 behind the line, so requiring that costs a genuine one
    nothing.

    The substring test runs first because it is a memchr scan and the pattern is not: almost every
    window has no armor line at all and pays only the fast test.
    """
    return _OPENPGP_MESSAGE_ARMOR in window and _OPENPGP_ARMORED_BODY.search(window) is not None


def _has_age_file_armor(window: bytes) -> bool:
    """Whether `window` carries age armor with a plausible ciphertext body at any offset."""
    return _AGE_FILE_ARMOR in window and _AGE_ARMORED_BODY.search(window) is not None


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
    at = 0
    for _ in range(_MAX_PGP_RECIPIENTS):
        following = _session_packet_end(head, at, session)
        if following is None:
            return False
        if following == _PACKET_PAST_BUFFER:
            return None
        tag = _packet_tag(head[following])
        if tag in encrypted:
            return True
        if tag not in session:
            return False
        # another recipient's session key, so the data packet is further on: keep walking. Each
        # header is at least two bytes, so the offset always advances and the loop always ends.
        at = following
    # Bounded like every other walk here, and exhausting it is undecided rather than clean: a
    # message addressed to more recipients than this is still an encrypted message.
    return None


def _packet_tag(first: int) -> int:
    """The packet tag of an OpenPGP header byte, or 0 if it is not a packet header at all."""
    if first & 0xC0 == 0xC0:
        return first & 0x3F
    return (first >> 2) & 0x0F if first & 0xC0 == 0x80 else 0


def _session_packet_end(head: bytes, at: int, session: frozenset[int]) -> int | None:
    """Where the session-key packet at `head[at:]` ends, so the caller can read what follows.

    Three answers, each with one meaning: None is "not a valid session-key packet here", the
    `_PACKET_PAST_BUFFER` sentinel is "the packet runs past the bytes available" -- undecided rather
    than clean -- and any other value is the offset of the packet behind this one, guaranteed to be
    a readable index.

    Split out of `_encrypted_message_head` because that function now walks: `gpg --encrypt -r a -r b`
    writes one of these per recipient before the encrypted data packet, so deciding a message from
    the first packet alone reported "not encrypted" for every multi-recipient message and published
    the ciphertext.
    """
    head = head[at:]
    if len(head) < 2:
        return None
    if head[0] & 0xC0 == 0xC0:  # new format: tag in the low six bits
        # The length is decoded here rather than through `_openpgp_body_length`, which keys on the
        # secret-key tag bytes: for a `0xC1`/`0xC3` header it falls through to a slice that is
        # empty and returns 0, so a wrong length would read as a stated one.
        tag, first = head[0] & 0x3F, head[1]
        if first < 192:
            length, header = first, 2
        elif first < 224:
            if len(head) < 3:
                return None
            length, header = ((first - 192) << 8) + head[2] + 192, 3
        elif first == 255:
            if len(head) < 6:
                return None
            length, header = int.from_bytes(head[2:6], "big"), 6
        else:
            return None  # a partial-body length: the packet has no single stated length
    elif head[0] & 0xC0 == 0x80:  # old format: tag in bits 5-2, length type in the low two
        tag, width = (head[0] >> 2) & 0x0F, (1, 2, 4, 0)[head[0] & 0x03]
        if not width:
            return None
        length, header = int.from_bytes(head[1 : 1 + width], "big"), 1 + width
    else:
        return None
    if tag not in session or len(head) < header + 1:
        return None
    # RFC 4880/9580 packet versions: 3 for a public-key ESK, 4/5/6 for a symmetric-key ESK.
    if head[header] not in (3, 4, 5, 6):
        return None
    if tag == 3:
        # symmetric-key ESK: a cipher from the registry, then an S2K specifier of a defined type.
        #
        # Where the S2K sits depends on the VERSION. A version 5 or 6 packet carries an AEAD
        # algorithm byte between the cipher and the S2K, and version 6 a length byte after it, so
        # reading the version 4 layout for all of them tested the AEAD algorithm as though it were
        # the S2K type. `gpg --force-aead --aead-algo OCB` writes AEAD 2, which is not a defined
        # S2K type, so the packet was rejected as malformed and the encrypted message published --
        # `gpg --decrypt` recovered the key from the same file.
        skew = {5: 1, 6: 2}.get(head[header], 0)
        if len(head) < header + 3 + skew:
            return None
        if head[header + 1] not in (1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13):
            return None
        if head[header + 2 + skew] not in (0, 1, 3, 4):
            return None
    end = header + length
    if end + 1 > len(head):
        # The session packet is longer than the bytes available. A real PKESK for an RSA-2048 key
        # is a few hundred bytes, so a fixed head could not reach the data packet behind it and
        # reported "not encrypted" -- publishing the ciphertext. The caller passes the whole chunk
        # now; still running out means the packet is longer than a chunk, which is undecided.
        return _PACKET_PAST_BUFFER
    return at + end


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
    if offset == 0 or len(head) <= offset:
        return False
    # Version 3 as well as 4 and 6. A v3 secret key is obsolete, not unreadable: GnuPG still names
    # a structurally complete tag-5 v3 packet an (obsolete) secret key, and its MPI key material
    # matches no textual or DER detector -- so rejecting the version published the key intact.
    #
    # The v3 layout differs by two bytes: a validity period in DAYS follows the creation timestamp,
    # before the algorithm. Reading it at the v4 offset lands inside that field, so the version
    # decides the offset rather than one being assumed for all of them.
    version = head[offset]
    if version not in (3, 4, 6):
        return False
    algorithm_at = offset + (7 if version == 3 else 5)
    # The declared body length must actually reach the fields read below. A packet claiming a
    # one-byte body cannot hold a version, a four-byte timestamp and an algorithm, so reading them
    # takes bytes from BEYOND the packet -- `c5 01 04 00 00 00 00 01` is an ordinary binary that
    # was refused as a private key. Old-format length type 3 is indeterminate, which declares no
    # length at all, and is already excluded by the table above.
    body = _openpgp_body_length(head, offset)
    if body is not None and body < algorithm_at - offset + 1:
        return False
    # then a four-byte creation timestamp (and, for v3, a two-byte validity), then the algorithm
    return len(head) > algorithm_at and head[algorithm_at] in algorithms


def _openpgp_packet_starts(head: bytes) -> Iterator[bytes]:
    """`head` at each packet boundary of the OpenPGP sequence it begins with.

    A keyring is a SEQUENCE of packets, and `gpg --import` installs the secret key in one wherever
    it sits. Testing only the first packet was wrong about ordering rather than about rare files:
    `gpg --export` followed by `gpg --export-secret-keys` -- which is what a "back up my GnuPG
    keys" one-liner writes, and what `--export-options export-local-sigs` produces -- leads with a
    PUBLIC key block, so the secret material behind it matched no textual or DER pattern and
    published intact.

    The walk only ever starts from a packet that already parsed as OpenPGP, so this cannot wander
    into arbitrary binary: each step re-derives the length from the header it is standing on and
    stops the moment one does not add up. That keeps the "anchored, not searched" property the
    secret-key test depends on -- every position yielded here is a boundary the format itself
    declares, not an offset found by looking for one.

    Bounded by `_MAX_OPENPGP_PACKETS`. A real key block is a handful of packets; a file claiming
    more than this is either not a keyring or is one no publish needs, and an unbounded walk over
    attacker-chosen lengths is a denial-of-service surface.

    Reaching that bound with bytes still unwalked yields `_TRUNCATED_PACKET`, exactly as running
    past the buffer does. Ending the generator normally made the cap its own bypass: 64 valid
    literal-data packets in front of a secret key returned `False` -- a confident "no key here"
    about a remainder never examined -- while 63 of them reported the key correctly.
    """
    for _ in range(_MAX_OPENPGP_PACKETS):
        if not head:
            return
        yield head
        tag_new = head[0]
        if tag_new & 0x80 == 0:
            return
        if tag_new & 0x40:
            first = head[1] if len(head) > 1 else 0
            if 224 <= first < 255 and (tag_new & 0x3F) in _PARTIAL_LENGTH_TAGS:
                # A partial body length: the packet arrives in chunks, each declaring its own
                # length, and only the LAST one is definite. There is no single stated length to
                # read, so this used to fall to `offset = 0` and end the walk -- returning normally
                # rather than undecided, which reported the unread remainder clean. A legal
                # partial-length literal packet between a public packet and a secret-key packet
                # therefore published the secret key.
                skipped = _past_partial_chunks(head, first)
                if skipped is None:
                    # The chunks do not finish inside the buffer. Undecided only when the sequence
                    # so far is plausible: a partial chunk may declare up to 1 GiB, so random bytes
                    # trip this constantly, and yielding the sentinel unconditionally refused
                    # 3 of every 64 ordinary binaries. This is the same distinction the definite
                    # lengths below already draw with `_MAX_OPENPGP_BODY_BYTES`.
                    if (1 << (first & 0x1F)) <= _MAX_OPENPGP_BODY_BYTES:
                        yield _TRUNCATED_PACKET
                    return
                head = head[skipped:]
                continue
            offset = 2 if first < 192 else (3 if first < 224 else 6)
        else:
            offset = {0x00: 2, 0x01: 3, 0x02: 5}.get(head[0] & 0x03, 0)
        if offset == 0 or len(head) < offset:
            return
        body = _openpgp_body_length(head, offset)
        if body is None or body <= 0:
            return
        if body > _MAX_OPENPGP_BODY_BYTES:
            return
        if offset + body > len(head):
            # The packet declares more body than is here. On a whole file that means a truncated or
            # malformed keyring and the walk simply ends -- but this runs on the FIRST CHUNK of a
            # streamed scan, so it is equally what a well-formed sequence looks like when a large
            # early packet crosses the chunk boundary. Measured: `gpg --export` followed by
            # `--export-secret-keys`, with a packet padding the public block past 1 MiB, imported
            # its secret key while the scan reported clean, because the walk stopped here and the
            # sequence test only ever sees chunk one.
            #
            # Yielding the remainder before stopping is what makes the difference visible to the
            # caller: a packet that outruns the buffer is undecided, not absent.
            yield _TRUNCATED_PACKET
            return
        head = head[offset + body :]
    # The loop bound, reached with bytes still in hand: the remainder is unwalked, not absent.
    # A distinct sentinel from `_TRUNCATED_PACKET`, because the two are undecided for different
    # reasons. Running past the buffer is only undecided when more bytes follow -- at end of file
    # it is an ordinary corrupt keyring. Exhausting the cap leaves bytes unexamined that are RIGHT
    # HERE, so it is undecided whatever the caller says about what follows.
    if head:
        yield _UNWALKED_REMAINDER


def _openpgp_secret_key_in_sequence(head: bytes, *, truncated: bool = False) -> bool | None:
    """Whether any packet in the OpenPGP sequence `head` begins with is a secret key.

    The FIRST packet decides the undecided case, and it is evaluated before the walk. The marker
    bound is a property of where the sequence starts: a walk that steps forward one marker at a
    time sees fewer remaining markers at every later position, so asking each in turn turned "still
    on a marker at the bound" into a confident answer from a position further along -- which is
    exactly the refusal `_after_openpgp_markers` exists to raise. So the leading packet is asked
    once, and only a decided `False` continues into the sequence.

    A packet that outruns the buffer is undecided ONLY when more bytes follow, which is what
    `truncated` states. On a streamed scan the walk runs on the first chunk alone, so a large public
    block padded past that boundary left the secret packet behind it unread and the file published
    -- while `gpg --import` installed the key from those same bytes. At true end of file the same
    shape is an ordinary corrupt or partial keyring with nothing unread behind it, and refusing
    those would fail a publish over a file that demonstrably holds no key.
    """
    leading = _is_openpgp_secret_key(head)
    if leading is not False:
        return leading
    for packet in _openpgp_packet_starts(head):
        if packet is _UNWALKED_REMAINDER:
            return None
        if packet is _TRUNCATED_PACKET:
            return None if truncated else False
        if _is_openpgp_secret_key(packet):
            return True
    return False


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


def _past_partial_chunks(head: bytes, first: int) -> int | None:
    """The offset just past a partial-length packet's chunks, or None if they run past `head`.

    RFC 4880 section 4.2.2.4: a first length octet in 224-254 declares a chunk of `1 << (octet &
    0x1f)` bytes, and further chunks follow, each with its own length octet, until one is written in
    a definite form. Walking them is what lets the packet AFTER this one be read; the alternative is
    to call the rest of the file undecided, which refuses ordinary multi-chunk messages.

    Bounded by `_MAX_OPENPGP_PACKETS` chunks, for the reason the packet walk itself is bounded: the
    lengths come from the file, and a stream of tiny partial chunks would otherwise be an unbounded
    loop over attacker-chosen input. Running out of chunks is `None`, the same undecided answer as
    running past the buffer, so the cap cannot become its own bypass.
    """
    at = 2
    for _ in range(_MAX_OPENPGP_PACKETS):
        at += 1 << (first & 0x1F)
        if at >= len(head):
            return None
        first = head[at]
        at += 1
        if first < 192:
            return _within(head, at + first)
        if first < 224:
            if at + 1 > len(head) - 1:
                return None
            length = ((first - 192) << 8) + head[at] + 192
            return _within(head, at + 1 + length)
        if first == 255:
            if at + 4 > len(head):
                return None
            return _within(head, at + 4 + int.from_bytes(head[at : at + 4], "big"))
        # another partial chunk: loop, with `first` naming its size
    return None


def _within(head: bytes, offset: int) -> int | None:
    """`offset` when it lands inside `head`, otherwise None.

    A final chunk's DECLARED length may reach past the bytes read so far -- the scan hands over one
    buffer at a time, and a legal five-octet length can name a body larger than it. Returning the
    offset regardless meant slicing at it produced an EMPTY remainder, and an empty remainder walks
    to the end without finding a key: a secret-key packet behind such a chunk was reported clean.
    The packet is not absent, it is unread, so the answer is the undecided one.
    """
    return offset if offset <= len(head) else None


def _openpgp_body_length(head: bytes, offset: int) -> int | None:
    """The body length an OpenPGP packet header declares, or None if it is not stated.

    Which ENCODING applies is decided by bit 6 of the tag byte, not by the tag itself. Naming the
    two secret-key tags meant every other new-format packet fell through to the old-format branch,
    which reads `head[1:offset]` as a big-endian integer -- so a two- or five-byte length was read
    as though its first byte were the whole length, and the five-byte form returned a nonsense
    trillion-byte body built from four bytes of the packet's own payload.
    """
    if head[0] & 0xC0 == 0xC0:
        first = head[1] if len(head) > 1 else 0
        if first < 192:
            return first
        if first < 224:
            return ((first - 192) << 8) + head[2] + 192 if len(head) > 2 else None
        return int.from_bytes(head[2:6], "big") if len(head) > 5 else None
    return int.from_bytes(head[1:offset], "big")
