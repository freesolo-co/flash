"""Bounded extraction of PNG metadata and reconstructed image samples."""

from __future__ import annotations

import zlib
from collections.abc import Callable, Iterator

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_CHUNK_HEADER_BYTES = 8
_PNG_CHUNK_CRC_BYTES = 4
_PNG_MAX_KEYWORD_BYTES = 79
_MAX_PNG_CHUNKS = 100_000


class _PngState:
    def __init__(self, ihdr: tuple[int, int, int, int]):
        self.width, self.height, self.bit_depth, self.color_type = ihdr
        self.idat = bytearray()
        self.idat_dimensions = (self.width, self.height)
        self.idat_started = False
        self.idat_ended = False
        self.plte = False
        self.animation_frames: list[tuple[int, int, bytearray]] = []
        self.active_frame: tuple[int, int, bytearray] | None = None
        self.animation = False
        self.declared_frames = 0
        self.frame_controls = 0
        self.next_sequence = 0
        self.default_frame_control = False


def _png_keyword_end(payload: bytes, unreadable: type[Exception]) -> int:
    end = payload.find(b"\0")
    if not 1 <= end <= _PNG_MAX_KEYWORD_BYTES:
        raise unreadable("invalid PNG text keyword")
    return end


def _inflate_png_text(payload: bytes, budget: int, unreadable: type[Exception]) -> bytes | None:
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
    end = _png_keyword_end(payload, unreadable)
    if len(payload) <= end + 1 or payload[end + 1] != 0:
        raise unreadable("unsupported PNG text compression method")
    return _inflate_png_text(payload[end + 2 :], budget, unreadable)


def _itxt_payload(payload: bytes, budget: int, unreadable: type[Exception]) -> bytes | None:
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
    if not compressed:
        return None if len(text) > budget else text
    return _inflate_png_text(text, budget, unreadable)


def _png_ihdr(payload: bytes, unreadable: type[Exception]) -> tuple[int, int, int, int]:
    if len(payload) != 13:
        raise unreadable("invalid PNG image header")
    width = int.from_bytes(payload[:4], "big")
    height = int.from_bytes(payload[4:8], "big")
    bit_depth, color_type, compression, filtering, interlace = payload[8:13]
    valid_depths = {
        0: (1, 2, 4, 8, 16),
        2: (8, 16),
        3: (1, 2, 4, 8),
        4: (8, 16),
        6: (8, 16),
    }
    if (
        width == 0
        or height == 0
        or bit_depth not in valid_depths.get(color_type, ())
        or compression != 0
        or filtering != 0
        or interlace != 0
    ):
        raise unreadable("unsupported PNG sample layout")
    return width, height, bit_depth, color_type


def _frame_control(
    state: _PngState, payload: bytes, unreadable: type[Exception]
) -> tuple[int, int]:
    if len(payload) != 26 or not state.animation:
        raise unreadable("invalid PNG animation frame control")
    sequence = int.from_bytes(payload[:4], "big")
    width = int.from_bytes(payload[4:8], "big")
    height = int.from_bytes(payload[8:12], "big")
    x = int.from_bytes(payload[12:16], "big")
    y = int.from_bytes(payload[16:20], "big")
    if (
        sequence != state.next_sequence
        or width == 0
        or height == 0
        or x + width > state.width
        or y + height > state.height
        or payload[24] > 2
        or payload[25] > 1
    ):
        raise unreadable("unsupported PNG animation sequencing")
    state.next_sequence += 1
    state.frame_controls += 1
    return width, height


def _consume_png_chunk(
    state: _PngState, kind: bytes, payload: bytes, unreadable: type[Exception]
) -> None:
    if state.idat_started and kind != b"IDAT":
        state.idat_ended = True
    if kind == b"PLTE":
        if (
            state.color_type in (0, 4)
            or state.plte
            or state.idat_started
            or not payload
            or len(payload) % 3
            or len(payload) > 768
            or (state.color_type == 3 and len(payload) // 3 > 1 << state.bit_depth)
        ):
            raise unreadable("invalid PNG palette")
        state.plte = True
    elif kind == b"acTL":
        if state.animation or state.idat_started or len(payload) != 8:
            raise unreadable("invalid PNG animation control")
        state.animation = True
        state.declared_frames = int.from_bytes(payload[:4], "big")
        if state.declared_frames == 0:
            raise unreadable("invalid PNG animation control")
    elif kind == b"fcTL":
        dimensions = _frame_control(state, payload, unreadable)
        if not state.idat_started:
            if (
                state.default_frame_control
                or dimensions != (state.width, state.height)
                or payload[12:20] != bytes(8)
            ):
                raise unreadable("unsupported PNG animation sequencing")
            state.default_frame_control = True
            state.idat_dimensions = dimensions
        else:
            if state.active_frame is not None:
                state.animation_frames.append(state.active_frame)
            state.active_frame = (*dimensions, bytearray())
    elif kind == b"IDAT":
        if state.idat_ended:
            raise unreadable("nonconsecutive PNG image data")
        state.idat_started = True
        state.idat.extend(payload)
    elif kind == b"fdAT":
        if len(payload) < 4 or state.active_frame is None:
            raise unreadable("unsupported PNG animation sequencing")
        sequence = int.from_bytes(payload[:4], "big")
        if sequence != state.next_sequence:
            raise unreadable("unsupported PNG animation sequencing")
        state.next_sequence += 1
        state.active_frame[2].extend(payload[4:])
    elif kind not in (b"IHDR", b"IEND", b"zTXt", b"iTXt", b"iCCP") and kind[:1].isupper():
        raise unreadable("unsupported critical PNG chunk")


def _png_chunks(
    data: bytes, unreadable: type[Exception], check: Callable[[], None]
) -> Iterator[tuple[bytes, bytes]]:
    if not data.startswith(_PNG_SIGNATURE):
        return
    at = len(_PNG_SIGNATURE)
    for _count in range(_MAX_PNG_CHUNKS):
        check()
        if at + _PNG_CHUNK_HEADER_BYTES + _PNG_CHUNK_CRC_BYTES > len(data):
            raise unreadable("incomplete PNG chunk header")
        size = int.from_bytes(data[at : at + 4], "big")
        kind = data[at + 4 : at + _PNG_CHUNK_HEADER_BYTES]
        if not all(
            byte in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" for byte in kind
        ):
            raise unreadable("invalid PNG chunk type")
        if kind[2:3].islower():
            raise unreadable("invalid PNG reserved chunk bit")
        payload_at = at + _PNG_CHUNK_HEADER_BYTES
        end = payload_at + size
        if end + _PNG_CHUNK_CRC_BYTES > len(data):
            raise unreadable("incomplete PNG chunk payload")
        payload = data[payload_at:end]
        expected = int.from_bytes(data[end : end + _PNG_CHUNK_CRC_BYTES], "big")
        if zlib.crc32(kind + payload) != expected:
            raise unreadable("invalid PNG chunk checksum")
        yield kind, payload
        at = end + _PNG_CHUNK_CRC_BYTES
        if kind == b"IEND":
            if size or at != len(data):
                raise unreadable("invalid PNG end chunk")
            return
    raise unreadable("PNG has too many chunks to inspect")


def _png_streams(
    data: bytes, unreadable: type[Exception], check: Callable[[], None]
) -> tuple[_PngState, list[tuple[bytes, bytes]]]:
    state: _PngState | None = None
    metadata: list[tuple[bytes, bytes]] = []
    for index, (kind, payload) in enumerate(_png_chunks(data, unreadable, check)):
        if index == 0:
            if kind != b"IHDR":
                raise unreadable("PNG does not begin with IHDR")
            state = _PngState(_png_ihdr(payload, unreadable))
            continue
        if state is None or kind == b"IHDR":
            raise unreadable("invalid PNG image header placement")
        _consume_png_chunk(state, kind, payload, unreadable)
        if kind in (b"zTXt", b"iCCP", b"iTXt"):
            metadata.append((kind, payload))
        if kind == b"IEND":
            break
    if state is None or not state.idat_started:
        raise unreadable("PNG has no image data")
    if state.color_type == 3 and not state.plte:
        raise unreadable("indexed PNG has no palette")
    if state.active_frame is not None:
        state.animation_frames.append(state.active_frame)
    if state.animation and (
        state.frame_controls == 0 or state.frame_controls != state.declared_frames
    ):
        raise unreadable("unsupported PNG animation sequencing")
    return state, metadata


def _paeth(left: int, above: int, upper_left: int) -> int:
    prediction = left + above - upper_left
    left_distance = abs(prediction - left)
    above_distance = abs(prediction - above)
    upper_left_distance = abs(prediction - upper_left)
    if left_distance <= above_distance and left_distance <= upper_left_distance:
        return left
    return above if above_distance <= upper_left_distance else upper_left


def _unfilter_row(filtered: bytes, previous: bytes, stride: int, kind: int) -> bytes | None:
    row = bytearray(len(filtered))
    for at, value in enumerate(filtered):
        left = row[at - stride] if at >= stride else 0
        above = previous[at] if previous else 0
        upper_left = previous[at - stride] if previous and at >= stride else 0
        if kind == 0:
            predictor = 0
        elif kind == 1:
            predictor = left
        elif kind == 2:
            predictor = above
        elif kind == 3:
            predictor = (left + above) // 2
        elif kind == 4:
            predictor = _paeth(left, above, upper_left)
        else:
            return None
        row[at] = (value + predictor) & 0xFF
    return bytes(row)


def _png_samples(
    packed: bytes,
    dimensions: tuple[int, int],
    bit_depth: int,
    color_type: int,
    budget: int,
    unreadable: type[Exception],
    check: Callable[[], None],
) -> bytes | None:
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    width, height = dimensions
    row_bytes = (width * channels * bit_depth + 7) // 8
    sample_bytes = row_bytes * height
    filtered_bytes = sample_bytes + height
    if sample_bytes > budget:
        return None
    inflate = zlib.decompressobj()
    try:
        filtered = inflate.decompress(packed, filtered_bytes + 1)
    except zlib.error:
        raise unreadable("invalid PNG image data") from None
    if (
        len(filtered) != filtered_bytes
        or inflate.unconsumed_tail
        or not inflate.eof
        or inflate.unused_data
    ):
        raise unreadable("incomplete PNG image data")
    stride = max(1, (channels * bit_depth + 7) // 8)
    output = bytearray()
    previous = b""
    at = 0
    for _ in range(height):
        check()
        filter_kind = filtered[at]
        row = _unfilter_row(filtered[at + 1 : at + 1 + row_bytes], previous, stride, filter_kind)
        if row is None:
            raise unreadable("unsupported PNG row filter")
        output.extend(row)
        previous = row
        at += row_bytes + 1
    return bytes(output)


def _png_text_payloads(
    data: bytes,
    budget: int,
    unreadable: type[Exception],
    check: Callable[[], None] | None = None,
) -> Iterator[bytes | None]:
    """Decoded metadata and reconstructed samples from one complete PNG."""
    check = check or (lambda: None)
    state, metadata = _png_streams(data, unreadable, check)
    remaining = budget
    for kind, payload in metadata:
        check()
        decoded = (
            _itxt_payload(payload, remaining, unreadable)
            if kind == b"iTXt"
            else _ztxt_payload(payload, remaining, unreadable)
        )
        yield decoded
        if decoded is None:
            return
        remaining -= len(decoded)
    streams = [(state.idat_dimensions, bytes(state.idat))]
    streams.extend(
        ((width, height), bytes(packed)) for width, height, packed in state.animation_frames
    )
    for dimensions, packed in streams:
        check()
        samples = _png_samples(
            packed,
            dimensions,
            state.bit_depth,
            state.color_type,
            remaining,
            unreadable,
            check,
        )
        yield samples
        if samples is None:
            return
        remaining -= len(samples)
