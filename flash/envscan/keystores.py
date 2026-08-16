"""Whether a key store or an OpenPGP sequence has been read far enough to answer.

Two formats whose verdict is not "is a credential here" but "have enough bytes arrived to say".
Both are walked entry by entry from the head of a stream, so a store or a keyring larger than one
scan chunk is undecided rather than clean -- and undecided is what the caller turns into a refusal.

Split out of `flash.envscan.secrets` to keep that module under the file-size limit. The dependency runs
one way: this knows about the two formats' structure, nothing about files, packages or the scan.
"""

from __future__ import annotations

from flash.envscan.formats import _KEYSTORE_MAGIC, _jks_private_key_entries
from flash.envscan.openpgp import (
    _is_openpgp_encrypted,
    _is_openpgp_secret_key,
    _openpgp_secret_key_in_sequence,
    _packet_tag,
)

# How much of a stream is accumulated to walk a key store to its end. The walk is head-anchored, so
# a store larger than one chunk had its remaining entries unread -- and a private key BEHIND a
# certificate whose body crossed the boundary published intact.
#
# Buffered rather than refused because refusing is a false alarm on exactly the ordinary case: a
# truststore is mostly certificates, holds no private key at all, and grows past a chunk simply by
# holding enough of them. The walk itself is cheap whatever the size -- it steps entry to entry by
# arithmetic and never reads a certificate body -- so the cap only has to sit above any real store.
_MAX_KEYSTORE_BYTES = 16 << 20


class _Unscannable(Exception):
    """A member could not be scanned to the end, so the publish cannot be vouched for.

    Every limit the scan imposes -- time, nesting depth, buffer size -- raises this rather than
    returning None. A limit that returns "no credential found" is indistinguishable from a clean
    scan, so the cheapest way past the whole check is to make it expensive: bury the key deeper
    than the depth cap, or behind a member too large to buffer. Refusing keeps the limits honest
    about what they mean, which is "not verified", not "verified clean".

    Defined HERE rather than in the scanner because the two walks below raise it and the scanner
    imports them, so the other direction would be a cycle. Re-exported from `flash.envscan.secrets`,
    which is where every other raise site and every test reads it from.
    """


def _keystore_undecided(head: bytes) -> bool | None:
    """Whether more bytes are needed to settle if `head` is a keystore holding a key.

    None means it IS one holding a key entry; False means it is not a keystore at all; True means
    the walk ran out of bytes and the answer is still open.

    ACCUMULATED across chunks by the caller rather than tested on the first one. A store whose walk
    ran off the end of a chunk reported "not a keystore" -- indistinguishable from bytes that never
    were one -- and since the overlap carry is non-empty from the second chunk on, nothing
    re-entered the parser. A single trusted certificate larger than a chunk was enough to hide the
    private key stored behind it.

    Re-walking the growing head each time is cheap: the walk steps from entry to entry by
    arithmetic and never reads a certificate body, so its cost is per ENTRY, not per byte.
    """
    if not head.startswith(_KEYSTORE_MAGIC):
        return False  # settled in four bytes, so nothing needs accumulating
    if (store := _jks_private_key_entries(head)) is None:
        if len(head) >= _MAX_KEYSTORE_BYTES:
            raise _Unscannable("contains a key store this check cannot finish walking")
        return True
    return None if store else False


def _decoded_key_kind(data: bytes) -> str | None:
    """The key kind in one exact base64 decode, using anchored binary format checks."""
    store = _keystore_undecided(data)
    if store is None:
        return "a key store"
    if store:
        raise _Unscannable("contains a key store this check cannot finish walking")
    secret = _is_openpgp_secret_key(data)
    if secret is not False or (data and _packet_tag(data[0]) in (6, 10, 14)):
        return _openpgp_kind(data, truncated=False)
    return None


def _openpgp_kind(chunk: bytes, *, truncated: bool) -> str | None:
    """The kind of OpenPGP key a stream BEGINS with, or None if it holds no key.

    Anchored at offset 0, where every packet format is decisive. Every file and every archive
    member reaches this, so a binary export is covered wherever it sits.

    Raises rather than returning None for the undecided cases: more marker packets than the walk
    allows, a packet whose body runs past this chunk while `truncated` says more follows, and an
    encrypted message whose body cannot be read at all. The WHOLE chunk goes to the encrypted test,
    since a public-key session packet carries the encrypted session key inline and runs to a few
    hundred bytes -- a fixed head could not reach the data packet behind it.

    The whole chunk goes to the SEQUENCE walk too, rather than a fixed head. A keyring holding both
    halves leads with the public block, which is thousands of bytes of key material, user IDs and
    signatures, so any fixed prefix stops short of the secret packet behind it -- the walk needs to
    reach whatever the earlier packets declare. It steps only between boundaries the packets
    themselves state, so this stays anchored rather than becoming a search.
    """
    secret_key = _openpgp_secret_key_in_sequence(chunk, truncated=truncated)
    if secret_key is None:
        # Two undecided shapes share this verdict: a sequence still on a marker packet at the walk
        # bound, and one whose packet declares a body running past the bytes in hand. The message
        # names the sequence rather than either cause, because the author's remedy is the same and
        # claiming the wrong one is worse than naming neither.
        raise _Unscannable("contains an OpenPGP packet sequence this check cannot walk to the end")
    if secret_key:
        return "a private key"
    if _is_openpgp_encrypted(chunk) is not False:
        raise _Unscannable("contains an encrypted OpenPGP message this check cannot read")
    return None
