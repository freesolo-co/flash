"""SSE framing helpers for the recording proxy."""

from __future__ import annotations


class SseDoneGate:
    def __init__(self) -> None:
        self._buffer = b""
        self._done = bytearray()
        self.done_event: bytes | None = None

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buffer += chunk
        if self._done:
            self._consume_done_suffix()
            return []

        cursor = 0
        while True:
            newline = self._buffer.find(b"\n", cursor)
            if newline < 0:
                break
            content = self._buffer[cursor:newline].rstrip(b"\r")
            if content.startswith(b"data:") and content[len(b"data:") :].strip() == b"[DONE]":
                forwarded = self._buffer[:cursor]
                self._done.extend(self._buffer[cursor : newline + 1])
                self._buffer = self._buffer[newline + 1 :]
                self._consume_done_suffix()
                return [forwarded] if forwarded else []
            cursor = newline + 1

        trailing = self._buffer[cursor:]
        if _could_be_done_line(trailing):
            forwarded = self._buffer[:cursor]
            self._buffer = trailing
        else:
            forwarded = self._buffer
            self._buffer = b""
        return [forwarded] if forwarded else []

    def finish(self) -> list[bytes]:
        if self._done:
            self._done.extend(self._buffer)
            self._buffer = b""
            self.done_event = bytes(self._done)
            self._done.clear()
            return []
        forwarded = self._buffer
        self._buffer = b""
        return [forwarded] if forwarded else []

    def _consume_done_suffix(self) -> None:
        newline = self._buffer.find(b"\n")
        if newline < 0:
            return
        if self._buffer[:newline].rstrip(b"\r"):
            return
        self._done.extend(self._buffer[: newline + 1])
        self._buffer = self._buffer[newline + 1 :]
        self.done_event = bytes(self._done)
        self._done.clear()


def _could_be_done_line(line: bytes) -> bool:
    prefix = b"data:"
    if len(line) < len(prefix):
        return prefix.startswith(line)
    if not line.startswith(prefix):
        return False
    data = line[len(prefix) :].lstrip()
    target = b"[DONE]"
    return target.startswith(data) or (
        data.startswith(target) and not data[len(target) :].strip(b" \t\r")
    )
