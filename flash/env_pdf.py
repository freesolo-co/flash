"""Bounded lexical helpers for PDF credential scanning."""

from __future__ import annotations

from collections.abc import Callable, Iterator

_PDF_WHITESPACE = frozenset(b"\x00\t\n\x0c\r ")
_PDF_DELIMITERS = frozenset(b"()<>[]{}/%")
_PDF_TOKEN_END = _PDF_WHITESPACE | _PDF_DELIMITERS
_PDF_CHECK_BYTES = 4096

_Check = Callable[[], None]
_PdfToken = tuple[bytes, int, int]


def _pdf_name(raw: bytes) -> bytes:
    """Resolve valid `#xx` escapes in one PDF name."""
    out = bytearray()
    at = 0
    while at < len(raw):
        if at + 2 < len(raw) and raw[at] == 35:
            try:
                out.append(int(raw[at + 1 : at + 3], 16))
            except ValueError:
                out.append(raw[at])
                at += 1
                continue
            at += 3
            continue
        out.append(raw[at])
        at += 1
    return bytes(out)


def pdf_tokens(data: bytes, check: _Check) -> Iterator[_PdfToken]:
    """Yield lexical PDF tokens while omitting strings and comments."""
    at = 0
    checkpoint = 0
    check()
    while at < len(data):
        if at >= checkpoint:
            check()
            checkpoint = at + _PDF_CHECK_BYTES
        byte = data[at]
        if byte in _PDF_WHITESPACE:
            at += 1
            continue
        if byte == 37:
            at += 1
            while at < len(data) and data[at] not in (10, 13):
                if at >= checkpoint:
                    check()
                    checkpoint = at + _PDF_CHECK_BYTES
                at += 1
            continue
        if byte == 40:
            depth = 1
            at += 1
            while at < len(data) and depth:
                if at >= checkpoint:
                    check()
                    checkpoint = at + _PDF_CHECK_BYTES
                if data[at] == 92:
                    at = min(len(data), at + 2)
                    continue
                depth += data[at] == 40
                depth -= data[at] == 41
                at += 1
            continue
        if byte == 60 and data[at : at + 2] != b"<<":
            at += 1
            while at < len(data) and data[at] != 62:
                if at >= checkpoint:
                    check()
                    checkpoint = at + _PDF_CHECK_BYTES
                at += 1
            at = min(len(data), at + 1)
            continue
        if data[at : at + 2] in (b"<<", b">>"):
            yield data[at : at + 2], at, at + 2
            at += 2
            continue
        if byte == 47:
            end = at + 1
            while end < len(data) and data[end] not in _PDF_TOKEN_END:
                if end >= checkpoint:
                    check()
                    checkpoint = end + _PDF_CHECK_BYTES
                end += 1
            raw = data[at + 1 : end]
            yield b"/" + (_pdf_name(raw) if len(raw) <= 64 else b""), at, end
            at = end
            continue
        if byte in _PDF_DELIMITERS:
            yield bytes((byte,)), at, at + 1
            at += 1
            continue
        end = at + 1
        while end < len(data) and data[end] not in _PDF_TOKEN_END:
            if end >= checkpoint:
                check()
                checkpoint = end + _PDF_CHECK_BYTES
            end += 1
        yield data[at:end] if end - at <= 64 else b"", at, end
        at = end
    check()


def pdf_dictionary_spans(data: bytes, check: _Check) -> list[tuple[int, int]]:
    """Return every balanced dictionary span outside strings and comments."""
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    for token, start, end in pdf_tokens(data, check):
        if token == b"<<":
            stack.append(start)
        elif token == b">>" and stack:
            spans.append((stack.pop(), end))
    check()
    return spans


def pdf_inline_images(data: bytes, check: _Check) -> Iterator[tuple[bytes, int]]:
    """Yield inline-image headers and payload starts from lexical `BI ... ID` pairs."""
    header_at: int | None = None
    for token, start, end in pdf_tokens(data, check):
        if header_at is None:
            if token == b"BI":
                header_at = end
            continue
        if token != b"ID":
            continue
        payload_at = end
        if data[payload_at : payload_at + 2] == b"\r\n":
            payload_at += 2
        elif data[payload_at : payload_at + 1] in (b"\x00", b"\t", b"\n", b"\x0c", b"\r", b" "):
            payload_at += 1
        else:
            continue
        yield data[header_at:start], payload_at
        header_at = None
    check()
