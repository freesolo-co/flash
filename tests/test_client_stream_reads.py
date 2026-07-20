from __future__ import annotations

import codecs
import io

import pytest

from flash.client.http import ApiClient


class _ReadResponse:
    def __init__(self, payload: bytes) -> None:
        self.headers = {"Content-Type": "text/plain; charset=utf-8"}
        self._stream = io.BytesIO(payload)
        self.calls: list[tuple[str, int]] = []

    def __enter__(self) -> _ReadResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        self.calls.append(("read", size))
        return self._stream.read(size)


class _Read1Response(_ReadResponse):
    def read1(self, size: int = -1) -> bytes:
        self.calls.append(("read1", size))
        return self._stream.read(size)


def _decode_byte_by_byte(payload: bytes) -> list[str]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    chunks: list[str] = []
    for byte in payload:
        chunk = decoder.decode(bytes([byte]))
        if chunk:
            chunks.append(chunk)
    tail = decoder.decode(b"", final=True)
    if tail:
        chunks.append(tail)
    return chunks


@pytest.mark.parametrize(
    ("response_type", "reader_name"),
    [(_Read1Response, "read1"), (_ReadResponse, "read")],
)
def test_chat_stream_buffered_reads_match_bytewise_utf8_chunks(
    monkeypatch: pytest.MonkeyPatch,
    response_type: type[_ReadResponse],
    reader_name: str,
) -> None:
    text = "a" * 4095 + "€\nfirst line\n第二行\n"
    payload = text.encode("utf-8")
    response = response_type(payload)
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)

    chunks = list(ApiClient("http://test").chat_stream("run-a", []))

    assert chunks == _decode_byte_by_byte(payload)
    assert "".join(chunks) == text
    assert response.calls
    assert all(call == (reader_name, 4096) for call in response.calls)
