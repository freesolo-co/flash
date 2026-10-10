from __future__ import annotations

import codecs
import http.client
import io
from collections.abc import Iterator

import pytest

from flash.client.http import ApiClient, ClientError


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

    chunks = list(ApiClient("http://test").chat_stream("run-a/final", []))

    assert chunks == list(_decode_byte_by_byte(payload))
    assert "".join(chunks) == text
    assert response.calls
    assert all(call == (reader_name, read_size) for call in response.calls)


def test_chat_stream_decodes_raw_openai_sse_for_cli_callers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (
        b'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"reasoning_content":"why"},"finish_reason":null}]}\n\n'
        b'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"answer"},"finish_reason":null}]}\n\n'
        b'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        b'data: {"id":"chatcmpl-1","choices":[],"usage":{"total_tokens":4}}\n\n'
        b"data: [DONE]\n\n"
    )
    response = _Read1Response(payload)
    response.headers = {"Content-Type": "Text/Event-Stream; Charset=UTF-8"}
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)

    chunks = list(ApiClient("http://test").chat_stream("run-a/final", []))

    assert "".join(chunks) == "<think>why</think>answer"


def test_chat_stream_closes_reasoning_before_raising_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (
        b'data: {"choices":[{"delta":{"reasoning_content":"why"}}]}\n\n'
        b'data: {"error":{"message":"engine failed"}}\n\n'
        b"trailing bytes are not consumed"
    )
    response = _Read1Response(payload)
    response.headers = {"Content-Type": "text/event-stream"}
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)
    stream = ApiClient("http://test").chat_stream("run-a/final", [])
    chunks = []

    with pytest.raises(ClientError, match="engine failed"):
        chunks.extend(stream)

    assert "".join(chunks) == "<think>why</think>"


def test_chat_stream_empty_reasoning_after_answer_does_not_reopen_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (
        b'data: {"choices":[{"delta":{"reasoning_content":"why"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n'
        b'data: {"choices":[{"delta":{"reasoning_content":""}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    response = _Read1Response(payload)
    response.headers = {"Content-Type": "text/event-stream"}
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)

    chunks = list(ApiClient("http://test").chat_stream("run-a/final", []))

    assert "".join(chunks) == "<think>why</think>answer"


@pytest.mark.parametrize("payload", [b"null", b"[]"])
def test_chat_stream_rejects_non_object_sse_json(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    response = _Read1Response(b"data: " + payload + b"\n\ndata: [DONE]\n\n")
    response.headers = {"Content-Type": "text/event-stream"}
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)

    with pytest.raises(ClientError, match="non-object openai sse payload"):
        list(ApiClient("http://test").chat_stream("run-a/final", []))


def test_chat_stream_rejects_complete_sse_eof_without_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
    response = _Read1Response(payload)
    response.headers = {"Content-Type": "text/event-stream"}
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)
    stream = ApiClient("http://test").chat_stream("run-a/final", [])

    assert next(stream) == "partial"
    with pytest.raises(ClientError, match=r"terminal \[DONE\]"):
        next(stream)


def test_chat_stream_json_fallback_rejects_a_wrong_service_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 application/json without choices must error, not read as an empty answer."""
    response = _ReadResponse(b'{"hello": "world"}')
    response.headers = {"Content-Type": "application/json"}
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)

    with pytest.raises(ClientError) as caught:
        list(ApiClient("http://test").chat_stream("run-a/final", []))
    assert "'choices'" in str(caught.value)


def test_chat_stream_without_read1_yields_before_full_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _BlockingLargeReadResponse(b"first chunk")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)
    stream = ApiClient("http://test").chat_stream("run-a/final", [])

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
    actual = _collect_until_unicode_error(ApiClient("http://test").chat_stream("run-a/final", []))

    assert actual == expected == list("visible prefix")


class _TruncatedChunkedResponse(_Read1Response):
    """A chunked body whose connection drops before the terminating chunk arrives.

    http.client raises IncompleteRead when a chunked stream ends without the zero-length
    terminator, which is what the server produces by aborting the response on a mid-stream
    upstream failure.
    """

    def read1(self, size: int = -1) -> bytes:
        self.calls.append(("read1", size))
        data = self._stream.read(size)
        if not data:
            raise http.client.IncompleteRead(b"")
        return data


def test_chat_stream_truncated_body_raises_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An aborted chunked response surfaces as ClientError, never as a clean end of stream.

    the text received before the abort is still yielded (the user has already seen it), but
    the stream must end by raising so a mid-generation serving failure cannot present as a
    short, finished answer."""
    response = _TruncatedChunkedResponse(b"partial answer")
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: response)

    stream = ApiClient("http://test").chat_stream("run-a/final", [])
    collected: list[str] = []

    def _drain() -> None:
        collected.extend(stream)

    with pytest.raises(ClientError, match="ended unexpectedly"):
        _drain()
    assert "".join(collected) == "partial answer"
