from __future__ import annotations

import codecs
import io
from collections.abc import Iterator

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


class _BlockingLargeReadResponse(_ReadResponse):
    def read(self, size: int = -1) -> bytes:
        self.calls.append(("read", size))
        if size != 1:
            raise TimeoutError("large reads would block")
        return self._stream.read(size)


def _decode_byte_by_byte(payload: bytes) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    for byte in payload:
        yield from decoder.decode(bytes([byte]))
    yield from decoder.decode(b"", final=True)


def _collect_until_unicode_error(chunks: Iterator[str]) -> list[str]:
    output: list[str] = []
    while True:
        try:
            output.append(next(chunks))
        except UnicodeDecodeError:
            return output
        except StopIteration:
            pytest.fail("stream ended without a UnicodeDecodeError")


@pytest.mark.parametrize(
    ("response_type", "reader_name", "read_size"),
    [(_Read1Response, "read1", 4096), (_ReadResponse, "read", 1)],
)
def test_chat_stream_reads_match_bytewise_utf8_chunks(
    monkeypatch: pytest.MonkeyPatch,
    response_type: type[_ReadResponse],
    reader_name: str,
    read_size: int,
) -> None:
    text = "a" * 4095 + "€\nfirst line\n第二行\n"
    payload = text.encode("utf-8")
    response = response_type(payload)
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)

    chunks = list(ApiClient("http://test").chat_stream("run-a", []))

    assert chunks == list(_decode_byte_by_byte(payload))
    assert "".join(chunks) == text
    assert response.calls
    assert all(call == (reader_name, read_size) for call in response.calls)


def test_chat_stream_without_read1_yields_before_full_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _BlockingLargeReadResponse(b"first chunk")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)
    stream = ApiClient("http://test").chat_stream("run-a", [])

    try:
        assert next(stream) == "f"
    finally:
        stream.close()

    assert response.calls == [("read", 1)]


def test_chat_stream_yields_valid_prefix_before_unicode_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"visible prefix\xffhidden"
    response = _Read1Response(payload)
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)

    expected = _collect_until_unicode_error(_decode_byte_by_byte(payload))
    actual = _collect_until_unicode_error(ApiClient("http://test").chat_stream("run-a", []))

    assert actual == expected == list("visible prefix")
