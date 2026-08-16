"""Context-bound joining for narrowly wrapped base64 values."""

from __future__ import annotations

import re

# short lines are common in prose and source, so the lower-width join is admitted only after a
# complete assignment or yaml block header identifies one value. measured consequence: applying
# the same width globally makes ordinary files pay speculative joins and can exhaust the scan's
# 60-second deadline on a 300 mib expansion.
_BLOCK_HINT = re.compile(
    rb"(?m)^[ \t]*[A-Za-z_][A-Za-z0-9_.-]*[ \t]*:[ \t]*[|>][^\r\n]*\r?\n"
    rb"[ \t]+[A-Za-z0-9+/\-_]+=*[ \t]*(?:\r?\n|\Z)"
)
_ASSIGNMENT_HINT = re.compile(
    rb"(?m)^[ \t]*[A-Za-z_][A-Za-z0-9_.-]*[ \t]*[=:][ \t]*"
    rb"[A-Za-z0-9+/\-_]+=*[ \t]*\r?\n[ \t]+[A-Za-z0-9+/\-_]+"
)
_BLOCK_HEADER = re.compile(rb"^[ \t]*[A-Za-z_][A-Za-z0-9_.-]*[ \t]*:[ \t]*[|>][^\r\n]*$")
_ASSIGNED_LINE = re.compile(
    rb"^[ \t]*[A-Za-z_][A-Za-z0-9_.-]*[ \t]*[=:][ \t]*"
    rb"(?P<value>[A-Za-z0-9+/\-_]+={0,2})[ \t]*$"
)
_INDENTED_VALUE = re.compile(rb"^(?P<indent>[ \t]+)(?P<value>[A-Za-z0-9+/\-_]+={0,2})[ \t]*$")


def _line_body(line: bytes) -> bytes:
    """One line without its ending."""
    return line.removesuffix(b"\n").removesuffix(b"\r")


def _fixed_width(chunks: list[bytes], minimum_run: int) -> bool:
    """Whether chunks are one fixed-width wrapping with an optional short tail."""
    if len(chunks) < 2 or sum(map(len, chunks)) < minimum_run:
        return False
    width = len(chunks[0])
    return all(len(chunk) == width for chunk in chunks[:-1]) and len(chunks[-1]) <= width


def _body_run(lines: list[bytes], start: int, minimum_run: int) -> tuple[int, bytes] | None:
    """The equally-indented base64 run beginning at start."""
    first = _INDENTED_VALUE.fullmatch(_line_body(lines[start]))
    if first is None:
        return None
    indent = first.group("indent")
    chunks = [first.group("value")]
    at = start + 1
    while at < len(lines):
        match = _INDENTED_VALUE.fullmatch(_line_body(lines[at]))
        if match is None or match.group("indent") != indent:
            break
        chunks.append(match.group("value"))
        at += 1
    if not _fixed_width(chunks, minimum_run):
        return None
    ending = lines[at - 1][len(_line_body(lines[at - 1])) :]
    return at, indent + b"".join(chunks) + ending


def _assignment_run(lines: list[bytes], start: int, minimum_run: int) -> tuple[int, bytes] | None:
    """A base64 assignment continued by equally-indented value lines."""
    first_line = _line_body(lines[start])
    first = _ASSIGNED_LINE.fullmatch(first_line)
    if first is None or start + 1 >= len(lines):
        return None
    continuation = _INDENTED_VALUE.fullmatch(_line_body(lines[start + 1]))
    if continuation is None:
        return None
    indent = continuation.group("indent")
    chunks = [first.group("value"), continuation.group("value")]
    at = start + 2
    while at < len(lines):
        match = _INDENTED_VALUE.fullmatch(_line_body(lines[at]))
        if match is None or match.group("indent") != indent:
            break
        chunks.append(match.group("value"))
        at += 1
    if not _fixed_width(chunks, minimum_run):
        return None
    value_at = first.start("value")
    ending = lines[at - 1][len(_line_body(lines[at - 1])) :]
    return at, first_line[:value_at] + b"".join(chunks) + ending


def context_unwrapped(data: bytes, minimum_run: int) -> bytes:
    """Join narrow base64 only where syntax proves consecutive lines are one value."""
    if b"\n" not in data:
        return data
    block = (b"|" in data or b">" in data) and _BLOCK_HINT.search(data) is not None
    assigned = (b"=" in data or b":" in data) and _ASSIGNMENT_HINT.search(data) is not None
    if not block and not assigned:
        return data
    lines = data.splitlines(keepends=True)
    out: list[bytes] = []
    changed = False
    at = 0
    while at < len(lines):
        body = _line_body(lines[at])
        run = None
        if _BLOCK_HEADER.fullmatch(body) and at + 1 < len(lines):
            run = _body_run(lines, at + 1, minimum_run)
            if run is not None:
                out.append(lines[at])
        else:
            run = _assignment_run(lines, at, minimum_run)
        if run is None:
            out.append(lines[at])
            at += 1
            continue
        at, joined = run
        out.append(joined)
        changed = True
    return b"".join(out) if changed else data
