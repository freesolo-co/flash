"""Compressed textual metadata extraction from PNG images.

PNG stores zTXt and iTXt text plus iCCP colour profiles in zlib streams inside typed chunks.
Those streams start after the PNG and chunk framing, so the head-anchored compressed-stream scanner
cannot see them. This module only recovers the declared metadata. Pixel data remains image content.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_CHUNK_HEADER_BYTES = 8
_PNG_CHUNK_CRC_BYTES = 4
_PNG_MAX_KEYWORD_BYTES = 79


def _png_keyword_end(payload: bytes, unreadable: type[Exception]) -> int:
    """The terminating nul of a valid PNG text keyword."""
    end = payload.find(b"\0")
    if not 1 <= end <= _PNG_MAX_KEYWORD_BYTES:
        raise unreadable("invalid PNG text keyword")
    return end


def _inflate_png_text(payload: bytes, budget: int, unreadable: type[Exception]) -> bytes | None:
    """The complete zlib text payload, or None when its expanded form exceeds the budget."""
    inflate = zlib.decompressobj()
    try:
        plain = inflate.decompress(payload, budget + 1)
    except zlib.error:
        raise unreadable("invalid PNG compressed text") from None
    if len(plain) > budget or inflate.unconsumed_tail:
        return None
    if not inflate.eof or inflate.unused_data:
        raise unreadable("incomplete PNG compressed text")
    return plain


def _ztxt_payload(payload: bytes, budget: int, unreadable: type[Exception]) -> bytes | None:
    """The decoded value in one zTXt or iCCP chunk."""
    end = _png_keyword_end(payload, unreadable)
    if len(payload) <= end + 1 or payload[end + 1] != 0:
        raise unreadable("unsupported PNG text compression method")
    return _inflate_png_text(payload[end + 2 :], budget, unreadable)


def _itxt_payload(payload: bytes, budget: int, unreadable: type[Exception]) -> bytes | None:
    """The text in one iTXt chunk, decoded only when its compression flag requires it."""
    end = _png_keyword_end(payload, unreadable)
    if len(payload) < end + 3:
        raise unreadable("incomplete PNG international text header")
    compressed, method = payload[end + 1 : end + 3]
    if compressed not in (0, 1) or method != 0:
        raise unreadable("unsupported PNG international text compression")
    at = end + 3
    for _ in range(2):
        at = payload.find(b"\0", at)
        if at < 0:
            raise unreadable("incomplete PNG international text fields")
        at += 1
    text = payload[at:]
    return _inflate_png_text(text, budget, unreadable) if compressed else text


def _png_text_payloads(
    data: bytes, budget: int, unreadable: type[Exception]
) -> Iterator[bytes | None]:
    """Every decoded zTXt, iTXt, and iCCP value in a structurally complete PNG."""
    if not data.startswith(_PNG_SIGNATURE):
        return
    at = len(_PNG_SIGNATURE)
    first = True
    while at < len(data):
        if at + _PNG_CHUNK_HEADER_BYTES + _PNG_CHUNK_CRC_BYTES > len(data):
            raise unreadable("incomplete PNG chunk header")
        size = int.from_bytes(data[at : at + 4], "big")
        kind = data[at + 4 : at + _PNG_CHUNK_HEADER_BYTES]
        payload_at = at + _PNG_CHUNK_HEADER_BYTES
        end = payload_at + size + _PNG_CHUNK_CRC_BYTES
        if end > len(data):
            raise unreadable("incomplete PNG chunk payload")
        if first and kind != b"IHDR":
            raise unreadable("PNG does not begin with IHDR")
        first = False
        payload = data[payload_at : payload_at + size]
        if kind in (b"zTXt", b"iCCP"):
            yield _ztxt_payload(payload, budget, unreadable)
        elif kind == b"iTXt":
            yield _itxt_payload(payload, budget, unreadable)
        at = end
        if kind == b"IEND":
            if size or at != len(data):
                raise unreadable("invalid PNG end chunk")
            return
    raise unreadable("PNG has no end chunk")
