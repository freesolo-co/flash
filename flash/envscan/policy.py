"""Deciding which recognised format a stream carries, and what to say when it cannot be read.

The tables in `flash.envscan.formats` say what a signature IS. This says how hard to look for one and
how to phrase the refusal -- two questions of policy rather than of format, and the ones that carry
the false-positive budget: a signature searched at an arbitrary offset can refuse an ordinary model
shard, so the rule for which ones are searched lives here beside the reasoning for it.

Split out to keep both neighbours under the file-size limit. The dependency runs one way: this
imports the tables, and nothing here knows about files, packages or the scan.
"""

from __future__ import annotations

from flash.envscan.formats import _ANCHORED_ONLY_MAGIC, _UNEXPANDABLE_MAGIC, ENCRYPTED_FORMATS

# How long a signature must be to be searched for at an ARBITRARY offset rather than only at the
# start of a stream. Six bytes is where the two real self-extracting formats sit (7-Zip at six, RAR
# at seven and eight) and where a chance occurrence stops being plausible: measured 0 hits across
# 256 MiB of random bytes for every signature, but a 4-byte magic is only 1 in 4 billion per
# position, which a large enough model shard reaches. The short zstd and LZ4 magics stay decisive
# at offset zero, where they mean what they say.
_SFX_MAGIC_BYTES = 6


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

    `_ANCHORED_ONLY_MAGIC` answers the anchored question alone. Those signatures are long enough to
    pass the length test but are printable words rather than binary magic, so searching for them
    would refuse a file that merely mentions the format.
    """
    magics = (
        (*_UNEXPANDABLE_MAGIC, *_ANCHORED_ONLY_MAGIC)
        if anchored
        else [pair for pair in _UNEXPANDABLE_MAGIC if len(pair[0]) >= _SFX_MAGIC_BYTES]
    )
    if anchored:
        return next((fmt for magic, fmt in magics if data.startswith(magic)), None)
    return next((fmt for magic, fmt in magics if magic in data), None)


def _uninspectable_reason(fmt: str) -> str:
    """Why `fmt` cannot be vouched for, phrased as the thing the author has to act on.

    An encrypted file and an unexpandable archive are the same verdict and different advice. Telling
    someone holding an `openssl enc` envelope that it is "an archive this check cannot expand" points
    them at a decompressor that does not exist; what they have to do is keep the ciphertext out of
    the package.
    """
    if fmt in ENCRYPTED_FORMATS:
        return f"contains a {fmt} file this check cannot read"
    return f"contains a {fmt} archive this check cannot expand"
