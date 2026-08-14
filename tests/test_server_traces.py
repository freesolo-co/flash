from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from collections.abc import AsyncIterator
from typing import ClassVar

import anyio
import httpx
import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from flash.server.platform import db
from flash.server.platform import traces as platform_traces
from flash.server.platform.traces import (
    TraceSpan,
    export_traces,
    list_projects,
    sanitize_json_value,
    store_trace,
)
from flash.server.routes import trace_redaction, trace_sse, traces

_PROJECT_ID = "11111111-1111-4111-8111-111111111111"
_OTHER_PROJECT_ID = "22222222-2222-4222-8222-222222222222"
_KEY = "operator-secret-long"
_PROVIDER_KEY = "provider-secret-long"
_HEADERS = {
    "Authorization": f"Bearer {_KEY}",
    "X-Freesolo-Provider": "openai",
    "X-Freesolo-Provider-Key": _PROVIDER_KEY,
    "X-Freesolo-Project-Id": _PROJECT_ID,
}
_REQUEST = {
    "model": "gpt-test",
    "messages": [{"role": "user", "content": "hello"}],
    "metadata": {"source": "test"},
}
_RESPONSE = {
    "id": "chatcmpl-test",
    "choices": [{"message": {"role": "assistant", "content": "world"}}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 4},
}


def _reply_envelope(content: str) -> dict:
    """A chat-completions response carrying `content`, the shape the proxy actually stores."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@pytest.mark.parametrize("standalone", [False, True], ids=["managed", "standalone"])
def test_app_registers_trace_routes_only_in_standalone(monkeypatch, standalone: bool) -> None:
    from flash.server import app as app_mod

    monkeypatch.setenv("FLASH_STANDALONE", "1" if standalone else "0")

    schema = app_mod.create_app().openapi()
    paths = set(schema["paths"])

    assert ("/v1/chat/completions" in paths) is standalone
    assert ("/api/traces/export" in paths) is standalone


@pytest.fixture
def trace_api(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    monkeypatch.setenv("FLASH_STANDALONE", "1")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", _KEY)
    app = FastAPI()
    app.include_router(traces.router)
    with TestClient(app) as client:
        yield client


def _raw(trace_api: TestClient, project_id: str = _PROJECT_ID) -> dict:
    response = trace_api.get(
        "/api/traces/export",
        headers={"Authorization": f"Bearer {_KEY}"},
        params={"project_id": project_id, "format": "raw"},
    )
    assert response.status_code == 200
    return response.json()


class _StaticAsyncClient:
    response = httpx.Response(200, json=_RESPONSE)
    requests: ClassVar[list[dict]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    class _StreamContext:
        def __init__(self, response: httpx.Response) -> None:
            self.response = response

        async def __aenter__(self) -> httpx.Response:
            return self.response

        async def __aexit__(self, *args) -> None:
            await self.response.aclose()

    def stream(self, method, url, *, headers, json):
        assert method == "POST"
        type(self).requests.append({"url": url, "headers": headers, "json": json})
        return self._StreamContext(type(self).response)

    async def aclose(self) -> None:
        self.closed = True


class _StreamingBody(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], error: Exception | None = None) -> None:
        self.chunks = chunks
        self.error = error
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error

    async def aclose(self) -> None:
        self.closed = True


class _BlockingStreamingBody(httpx.AsyncByteStream):
    def __init__(self, first: bytes) -> None:
        self.first = first
        self.blocked = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.first
        self.blocked.set()
        await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


class _StreamingAsyncClient(_StaticAsyncClient):
    body = _StreamingBody([])
    status_code = 200

    def build_request(self, method, url, *, headers, json) -> httpx.Request:
        type(self).requests.append({"url": url, "headers": headers, "json": json})
        return httpx.Request(method, url, headers=headers, json=json)

    async def send(self, request, *, stream) -> httpx.Response:
        assert stream is True
        return httpx.Response(
            type(self).status_code,
            headers={"content-type": "text/event-stream", "set-cookie": "not-relayed=1"},
            stream=type(self).body,
            request=request,
        )


def test_non_streaming_records_response_and_filters_headers(trace_api, monkeypatch) -> None:
    _StaticAsyncClient.requests = []
    _StaticAsyncClient.response = httpx.Response(
        200,
        json=_RESPONSE,
        headers={
            "x-request-id": "request-1",
            "set-cookie": "provider-cookie=1",
            "x-ratelimit-remaining": "9",
        },
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers={
            **_HEADERS,
            "OpenAI-Organization": "org-1",
            "Cookie": "plane-cookie=1",
        },
        json={
            **_REQUEST,
            "credential": _PROVIDER_KEY,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"api_key": {"type": "string"}},
                        },
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == _RESPONSE
    assert response.headers["x-request-id"] == "request-1"
    assert response.headers["x-ratelimit-remaining"] == "9"
    assert "set-cookie" not in response.headers
    sent = _StaticAsyncClient.requests[0]
    assert sent["url"] == "https://api.openai.com/v1/chat/completions"
    assert sent["headers"]["Authorization"] == f"Bearer {_PROVIDER_KEY}"
    normalized_headers = {name.casefold(): value for name, value in sent["headers"].items()}
    assert normalized_headers["openai-organization"] == "org-1"
    assert "cookie" not in normalized_headers
    # the PROVIDER gets the caller's request verbatim. redaction applies to the stored copy only:
    # rewriting the forwarded body would bill the caller for inference on a request they never
    # wrote, with no way to tell from the response that it had been altered.
    assert sent["json"]["credential"] == _PROVIDER_KEY
    assert sent["json"]["tools"][0]["function"]["parameters"]["properties"]["api_key"] == {
        "type": "string"
    }

    raw = _raw(trace_api)
    assert raw["traces"] == 1
    span = raw["records"][0]["spans"][0]
    assert span["name"] == "chat.completions"
    assert span["provider"] == "openai"
    assert span["model"] == "gpt-test"
    assert span["input_tokens"] == 3
    assert span["output_tokens"] == 4
    assert span["status_code"] == "OK"
    assert span["input_payload"]["messages"] == _REQUEST["messages"]
    # ...and the STORED copy is the redacted one, so both halves of the split are pinned here
    assert span["input_payload"]["credential"] == "[redacted]"
    assert span["output_payload"] == _RESPONSE
    assert _PROVIDER_KEY not in json.dumps(raw)


def test_out_of_range_usage_counters_do_not_drop_the_trace(trace_api, monkeypatch) -> None:
    too_large = 2**63
    _StaticAsyncClient.response = httpx.Response(
        200,
        json={
            **_RESPONSE,
            "usage": {"prompt_tokens": too_large, "completion_tokens": -(2**63) - 1},
        },
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["input_tokens"] is None
    assert span["output_tokens"] is None
    assert response.headers.get(traces._RECORD_FAILED_HEADER) is None


def test_a_non_streaming_response_under_the_limit_is_relayed_byte_identically(
    trace_api, monkeypatch
) -> None:
    """Bounding the provider body must not re-encode ordinary replies. A caller may verify signatures
    or depend on whitespace, so fitting bodies retain the provider's exact bytes, status, and headers."""
    body = b'{ "choices" : [ { "message" : { "role" : "assistant", "content" : "world" } } ] }\n'
    _StaticAsyncClient.requests = []
    _StaticAsyncClient.response = httpx.Response(
        201,
        content=body,
        headers={"content-type": "application/json", "x-request-id": "exact-1"},
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 201
    assert response.content == body
    assert response.headers["x-request-id"] == "exact-1"


def test_json_value_conversion_failure_relays_body_and_records_text(trace_api, monkeypatch) -> None:
    body = b'{"number":' + b"9" * 4_301 + b"}"
    _StaticAsyncClient.response = httpx.Response(
        200,
        content=body,
        headers={"content-type": "application/json"},
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 200
    assert response.content == body
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["output_payload"] == body.decode()
    assert span["status_code"] == "OK"


def test_recursive_json_response_relays_body_and_records_text(trace_api, monkeypatch) -> None:
    body = b"[" * 100_000 + b"]" * 100_000
    _StaticAsyncClient.response = httpx.Response(
        200,
        content=body,
        headers={"content-type": "application/json"},
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 200
    assert response.content == body
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["output_payload"] == body.decode()
    assert span["status_code"] == "OK"


def test_huge_integer_request_body_returns_invalid_json(trace_api) -> None:
    body = b'{"number":' + b"9" * 4_301 + b"}"

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, content=body)

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid JSON body"}


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_request_body_returns_invalid_json_without_creating_a_client(
    trace_api, monkeypatch, constant: str
) -> None:
    created = 0

    class _CountingClient(_StreamingAsyncClient):
        def __init__(self, *args, **kwargs) -> None:
            nonlocal created
            created += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(traces.httpx, "AsyncClient", _CountingClient)
    body = f'{{"model":"gpt-test","stream":true,"temperature":{constant}}}'.encode()

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, content=body)

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid JSON body"}
    assert created == 0


def test_streaming_request_encoding_failure_closes_the_client(trace_api, monkeypatch) -> None:
    clients = []

    class _EncodingFailureClient(_StreamingAsyncClient):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            clients.append(self)

        def build_request(self, method, url, *, headers, json) -> httpx.Request:
            raise ValueError("encoding failed")

    monkeypatch.setattr(traces.httpx, "AsyncClient", _EncodingFailureClient)

    with pytest.raises(ValueError, match="encoding failed"):
        trace_api.post("/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True})

    assert len(clients) == 1
    assert clients[0].closed is True


def test_streamed_json_value_conversion_failure_records_text(trace_api, monkeypatch) -> None:
    body = b'{"number":' + b"9" * 4_301 + b"}"

    class _HugeIntegerStreamingClient(_StreamingAsyncClient):
        requests: ClassVar[list[dict]] = []

        async def send(self, request, *, stream) -> httpx.Response:
            assert stream is True
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=type(self).body,
                request=request,
            )

    _HugeIntegerStreamingClient.body = _StreamingBody([body])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _HugeIntegerStreamingClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    assert response.content == body
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["output_payload"] == body.decode()
    assert span["status_code"] == "OK"


def test_an_oversized_non_streaming_response_returns_502_and_records_error(
    trace_api, monkeypatch
) -> None:
    """A bounded relay must never hand the caller a JSON prefix that looks like a provider reply.
    Reject the oversized body and record only an ERROR diagnosis so converted records skip it."""
    payload = (
        b'{"choices":[{"message":{"role":"assistant","content":"'
        + b"x" * (platform_traces.MAX_PAYLOAD_TOTAL_BYTES)
        + b'"}}]}'
    )
    _StaticAsyncClient.response = httpx.Response(
        200,
        content=payload,
        headers={"content-type": "application/json", "x-request-id": "large-1"},
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 502
    assert response.json() == {"detail": "Upstream response was too large to relay"}
    assert response.content != payload[: platform_traces.MAX_PAYLOAD_TOTAL_BYTES]
    assert response.headers["x-request-id"] == "large-1"
    raw = _raw(trace_api)
    span = raw["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == traces._UPSTREAM_TOO_LARGE_ERROR
    assert span["output_payload"] is None
    records = trace_api.get(
        "/api/traces/export",
        headers={"Authorization": f"Bearer {_KEY}"},
        params={"project_id": _PROJECT_ID, "format": "records"},
    ).json()
    assert records["records"] == []
    assert records["skipped"] == 1


def test_a_short_secret_does_not_corrupt_payload_substrings(trace_api, monkeypatch, caplog) -> None:
    """A short self-hosted key such as `test` is ordinary language, not a safe global substring.
    Replacing it turns `testing` into corrupted training text, so short credentials use field-only
    redaction and raise an operator warning instead of silently mutating every stored payload."""
    short_headers = {**_HEADERS, "X-Freesolo-Provider-Key": "test"}
    body = {**_REQUEST, "messages": [{"role": "user", "content": "testing"}]}
    _StaticAsyncClient.response = httpx.Response(200, json=_reply_envelope("testing reply"))
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    with caplog.at_level("WARNING"):
        response = trace_api.post("/v1/chat/completions", headers=short_headers, json=body)

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["input_payload"]["messages"][0]["content"] == "testing"
    assert span["output_payload"]["choices"][0]["message"]["content"] == "testing reply"
    assert any("substring redaction is disabled" in record.message for record in caplog.records)


def test_a_full_length_secret_is_still_redacted(trace_api, monkeypatch) -> None:
    """The short-secret guard must not weaken credential protection. A real-length bearer secret
    quoted in prompt and reply text is still removed from both stored payload sides."""
    long_secret = "credential-0123456789abcdef"
    headers = {
        **_HEADERS,
        "X-Freesolo-Provider-Key": long_secret,
    }
    body = {
        **_REQUEST,
        "messages": [{"role": "user", "content": f"inspect {long_secret} now"}],
    }
    _StaticAsyncClient.response = httpx.Response(200, json=_reply_envelope(f"saw {long_secret}"))
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=headers, json=body)

    assert response.status_code == 200
    raw = _raw(trace_api)
    assert long_secret not in json.dumps(raw)
    span = raw["records"][0]["spans"][0]
    assert span["input_payload"]["messages"][0]["content"] == "inspect [redacted] now"
    assert span["output_payload"]["choices"][0]["message"]["content"] == "saw [redacted]"


def test_a_credential_used_as_an_object_key_is_not_stored(trace_api, monkeypatch) -> None:
    """A credential is a credential wherever it sits. Value-only redaction walks dict VALUES, so a
    key-shaped secret -- `{"<key>": "seen"}` -- would round-trip into the span and back out through
    `format=raw`, which is the one export that never skips anything."""
    _StaticAsyncClient.requests = []
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={**_REQUEST, "metadata": {_PROVIDER_KEY: "marker"}},
    )

    assert response.status_code == 200
    # forwarded verbatim, stored redacted
    assert _StaticAsyncClient.requests[0]["json"]["metadata"] == {_PROVIDER_KEY: "marker"}
    assert _PROVIDER_KEY not in json.dumps(_raw(trace_api))


def test_an_oversized_sse_stream_forwards_every_byte_and_marks_output_truncated(
    trace_api, monkeypatch
) -> None:
    """The storage budget governs only the trace accumulator. A runaway stream still reaches the
    caller byte-for-byte, while its bounded stored prefix is marked output-truncated so `records`
    skips it and `prompts` can still recover the unaffected input side."""
    monkeypatch.setattr(traces, "MAX_PAYLOAD_TOTAL_BYTES", 256)
    chunks = [
        b'data: {"choices":[{"index":0,"delta":{"content":"' + b"x" * 180 + b'"}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"'
        + b"y" * 180
        + b'"},"finish_reason":"stop"}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(chunks)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    assert response.content == b"".join(chunks)
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["attributes"] == {"payload_truncated": ["output"]}
    stored_content = span["output_payload"]["choices"][0]["message"]["content"]
    assert len(stored_content) < 360
    records = trace_api.get(
        "/api/traces/export",
        headers={"Authorization": f"Bearer {_KEY}"},
        params={"project_id": _PROJECT_ID, "format": "records"},
    ).json()
    prompts = trace_api.get(
        "/api/traces/export",
        headers={"Authorization": f"Bearer {_KEY}"},
        params={"project_id": _PROJECT_ID, "format": "prompts"},
    ).json()
    assert records["records"] == []
    assert records["skipped"] == 1
    assert prompts["records"] == [{"input": "hello"}]
    assert prompts["skipped"] == 0


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"], ids=["lf", "crlf"])
def test_done_gate_waits_for_split_event_terminator(line_ending: bytes) -> None:
    completion = b'data: {"a":1}' + line_ending * 2
    terminator = b"data: [DONE]" + line_ending * 2
    gate = trace_sse.SseDoneGate()
    forwarded: list[bytes] = []

    for chunk in [completion, terminator[: -len(line_ending)], terminator[-len(line_ending) :]]:
        forwarded.extend(gate.feed(chunk))
        if gate.terminated:
            break
    forwarded.extend(gate.finish())
    if gate.done_event is not None:
        forwarded.append(gate.done_event)

    assert b"".join(forwarded) == completion + terminator


@pytest.mark.parametrize(
    ("chunks", "terminated"),
    [
        ([b"data: [DONE]\n", b"\n"], True),
        ([b"data: [DONE]\r\n", b"\r\n"], True),
        ([b"data: [DONE]\r", b"\n\r\n"], True),
        ([b"data: [DONE]\r", b"data: x\r", b"\r"], False),
    ],
    ids=["lf", "crlf", "split-crlf-at-cr", "bare-cr-multiline"],
)
def test_done_gate_resolves_an_ambiguous_trailing_cr_from_the_next_byte(
    chunks: list[bytes], terminated: bool
) -> None:
    gate = trace_sse.SseDoneGate()

    relayed = [part for chunk in chunks for part in gate.feed(chunk)]
    relayed.extend(gate.finish())
    parked = gate.done_event or b""

    assert gate.terminated is terminated
    assert len(b"".join(relayed)) + len(parked) == len(b"".join(chunks))
    if terminated:
        assert relayed == []
        assert parked == b"".join(chunks)
    else:
        assert b"".join(relayed) == b"".join(chunks)
        assert parked == b""


@pytest.mark.parametrize(
    "chunks",
    [
        [b"\xef\xbb\xbfdata: [DONE]\n\n"],
        [b"\xef", b"\xbb\xbfdata: [DONE]\n\n"],
    ],
    ids=["whole-bom", "split-bom"],
)
def test_done_gate_parses_a_leading_bom_without_dropping_its_bytes(chunks: list[bytes]) -> None:
    stream = b"".join(chunks)
    gate = trace_sse.SseDoneGate()

    relayed = [part for chunk in chunks for part in gate.feed(chunk)]
    relayed.extend(gate.finish())
    parked = gate.done_event or b""

    assert gate.terminated is True
    assert relayed == []
    assert parked == stream
    assert len(b"".join(relayed)) + len(parked) == len(stream)


def test_done_gate_parses_a_done_after_a_bom_prefixed_first_event() -> None:
    stream = b'\xef\xbb\xbfdata: {"choices":[]}\n\ndata: [DONE]\n\n'
    gate = trace_sse.SseDoneGate()

    relayed = gate.feed(stream)
    relayed.extend(gate.finish())
    parked = gate.done_event or b""

    assert gate.terminated is True
    assert b"".join(relayed) == b'\xef\xbb\xbfdata: {"choices":[]}\n\n'
    assert parked == b"data: [DONE]\n\n"
    assert len(b"".join(relayed)) + len(parked) == len(stream)


def test_done_gate_accepts_bare_cr_and_preserves_split_crlf() -> None:
    bare_cr = b'data: {"choices":[]}\r\rdata: [DONE]\r\r'
    gate = trace_sse.SseDoneGate()
    forwarded = gate.feed(bare_cr + b": after-done")
    if gate.done_event is not None:
        forwarded.append(gate.done_event)

    assert b"".join(forwarded) == bare_cr
    assert gate.terminated is True

    split_gate = trace_sse.SseDoneGate()
    split_forwarded = split_gate.feed(b'data: {"choices":[]}\r')
    split_forwarded.extend(split_gate.feed(b"\n\r"))
    split_forwarded.extend(split_gate.feed(b"\n"))
    split_forwarded.extend(split_gate.finish())
    assert b"".join(split_forwarded) == b'data: {"choices":[]}\r\n\r\n'


def test_sse_accumulator_discards_unterminated_data_event_at_eof() -> None:
    event = b'data: {"choices":[{"index":0,"delta":{"content":"GHOST"},"finish_reason":"stop"}]}\n'
    accumulator = trace_sse.SseAccumulator()

    accumulator.feed(event)
    accumulator.finish()

    assert accumulator.received is False
    assert accumulator.terminal is False
    assert accumulator.output()["choices"] == []
    assert accumulator.defect == "stream ended with an unterminated data event"


def test_sse_accumulator_discards_unterminated_data_line_at_eof() -> None:
    event = b'data: {"choices":[{"index":0,"delta":{"content":"GHOST"}}]}'
    accumulator = trace_sse.SseAccumulator()

    accumulator.feed(event)
    accumulator.finish()

    assert accumulator.received is False
    assert accumulator.output()["choices"] == []
    assert accumulator.defect == "stream ended with an unterminated data event"


def test_sse_accumulator_dispatches_terminal_cr_event_delimiter() -> None:
    payload = b'data: {"choices":[{"index":0,"delta":{"content":"OK"}}]}'
    for delimiter in (b"\r\r", b"\r\r\n", b"\n\n"):
        accumulator = trace_sse.SseAccumulator()
        accumulator.feed(payload + delimiter)
        accumulator.finish()

        assert accumulator.output()["choices"][0]["message"]["content"] == "OK"
        assert accumulator.received is True
        assert accumulator.defect is None

    gate = trace_sse.SseDoneGate()
    assert gate.feed(b"data: [DONE]\r\r") == []
    assert gate.finish() == []
    assert gate.terminated is True
    assert gate.done_event == b"data: [DONE]\r\r"


def test_sse_accumulator_dispatches_blank_line_terminated_control() -> None:
    event = (
        b'data: {"choices":[{"index":0,"delta":{"content":"GHOST"},"finish_reason":"stop"}]}\n\n'
    )
    accumulator = trace_sse.SseAccumulator()

    accumulator.feed(event)
    accumulator.finish()

    assert accumulator.output()["choices"][0]["message"]["content"] == "GHOST"
    assert accumulator.terminal is True
    assert accumulator.defect is None


def test_sse_accumulator_accepts_bare_cr_and_preserves_split_crlf() -> None:
    event = (
        b'data: {"choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}]}\r\r'
    )
    accumulator = trace_sse.SseAccumulator()
    accumulator.feed(event + b": next-event")

    assert accumulator.defect is None
    assert accumulator.output()["choices"][0]["message"]["content"] == "world"
    assert accumulator.terminal is True

    split_accumulator = trace_sse.SseAccumulator()
    split_accumulator.feed(b'data: {"choices":[{"index":0,"delta":{"content":"split"},')
    split_accumulator.feed(b'"finish_reason":"stop"}]}\r')
    assert split_accumulator.received is False
    split_accumulator.feed(b"\n\r")
    split_accumulator.feed(b"\n")
    assert split_accumulator.defect is None
    assert split_accumulator.output()["choices"][0]["message"]["content"] == "split"


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n", b"\r"], ids=["lf", "crlf", "cr"])
def test_done_gate_does_not_terminate_on_done_inside_multiline_data_event(
    line_ending: bytes,
) -> None:
    first = line_ending.join([b'data: {"notice":1}', b"data: [DONE]", b""])
    later = b'data: {"choices":[{"delta":{"content":"LATER"}}]}' + line_ending * 2
    gate = trace_sse.SseDoneGate()

    forwarded = gate.feed(first)
    forwarded.extend(gate.feed(later))
    forwarded.extend(gate.finish())

    assert b"".join(forwarded) == first + later
    assert gate.terminated is False


@pytest.mark.parametrize(
    "chunks",
    [
        [b'data: {"notice":1}', b"\ndata: [DONE]\n\n"],
        [b'data: {"notice":1}\r', b"\ndata: [DONE]\r\n\r\n"],
    ],
    ids=["lf", "split-crlf"],
)
def test_done_gate_preserves_a_split_first_data_line_in_a_multiline_event(
    chunks: list[bytes],
) -> None:
    gate = trace_sse.SseDoneGate()

    forwarded = [part for chunk in chunks for part in gate.feed(chunk)]
    forwarded.extend(gate.finish())

    assert b"".join(forwarded) == b"".join(chunks)
    assert gate.terminated is False
    assert gate.done_event is None


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n", b"\r"], ids=["lf", "crlf", "cr"])
def test_done_gate_does_not_terminate_when_done_precedes_multiline_data(
    line_ending: bytes,
) -> None:
    first = line_ending.join([b"data: [DONE]", b'data: {"notice":1}']) + line_ending * 2
    later = b'data: {"choices":[{"delta":{"content":"LATER"}}]}' + line_ending * 2
    gate = trace_sse.SseDoneGate()

    forwarded = gate.feed(first)
    forwarded.extend(gate.feed(later))
    forwarded.extend(gate.finish())

    assert b"".join(forwarded) == first + later
    assert gate.terminated is False


def test_done_gate_treats_colonless_data_as_an_empty_data_field() -> None:
    stream = (
        b"data: [DONE]\ndata\n\n"
        b'data: {"choices":[{"index":0,"delta":{"content":"AFTER"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    gate = trace_sse.SseDoneGate()

    relayed = gate.feed(stream)
    relayed.extend(gate.finish())

    assert len(stream) == 94
    assert b"".join(relayed) + (gate.done_event or b"") == stream
    assert len(b"".join(relayed)) == 80
    assert b"AFTER" in b"".join(relayed)
    assert gate.done_event == b"data: [DONE]\n\n"


@pytest.mark.parametrize("field", [b"event", b"database", b"datax"])
def test_done_gate_does_not_treat_other_colonless_fields_as_data(field: bytes) -> None:
    stream = b"data: [DONE]\n" + field + b"\n\n"
    gate = trace_sse.SseDoneGate()

    relayed = gate.feed(stream)
    relayed.extend(gate.finish())

    assert relayed == []
    assert gate.terminated is True
    assert gate.done_event == stream

    accumulator = trace_sse.SseAccumulator()
    accumulator.feed(stream)
    accumulator.finish()

    assert accumulator._done is True
    assert accumulator.defect is None


@pytest.mark.parametrize(
    ("event", "terminated"),
    [
        (b"data: [DONE]   \n\n", False),
        (b"data:  [DONE]\n\n", False),
        (b"data: [DONE]\t\n\n", False),
        (b"data: [DONE]\n\n", True),
        (b"data:[DONE]\n\n", True),
    ],
    ids=["trailing-spaces", "two-leading-spaces", "trailing-tab", "space", "no-space"],
)
def test_done_gate_matches_only_the_exact_sse_done_value(event: bytes, terminated: bool) -> None:
    line = event[: -len(b"\n\n")]
    gate = trace_sse.SseDoneGate()

    relayed = gate.feed(event)
    relayed.extend(gate.finish())

    assert gate.terminated is terminated
    assert trace_sse._could_be_done_line(line) is terminated
    if terminated:
        assert relayed == []
        assert gate.done_event == event
    else:
        assert b"".join(relayed) == event
        assert gate.done_event is None


def test_done_gate_relays_events_after_a_whitespace_suffixed_done_value() -> None:
    stream = (
        b'data: [DONE]   \n\ndata: {"choices":[{"delta":{"content":"REAL"}}]}\n\ndata: [DONE]\n\n'
    )
    gate = trace_sse.SseDoneGate()

    relayed = gate.feed(stream)
    relayed.extend(gate.finish())

    assert b"".join(relayed) == stream[: -len(b"data: [DONE]\n\n")]
    assert b"REAL" in b"".join(relayed)
    assert gate.done_event == b"data: [DONE]\n\n"


@pytest.mark.parametrize(
    ("event", "terminated"),
    [
        (b"data: [DONE]\n\n", True),
        (b"data: [DONE]\ndata\n\n", False),
        (b"data\ndata: [DONE]\n\n", False),
        (b"data: [DONE]   \n\n", False),
        (b"data:[DONE]\n\n", True),
        (b"data:  [DONE]\n\n", False),
        (b"data: [DONE]\t\n\n", False),
        (b"data: [DONE]x\n\n", False),
    ],
    ids=[
        "canonical",
        "colonless-after",
        "colonless-before",
        "trailing-spaces",
        "no-space",
        "two-leading-spaces",
        "tab",
        "junk",
    ],
)
def test_done_gate_and_accumulator_agree_on_sentinel_values(event: bytes, terminated: bool) -> None:
    gate = trace_sse.SseDoneGate()
    accumulator = trace_sse.SseAccumulator()

    relayed = gate.feed(event)
    relayed.extend(gate.finish())
    accumulator.feed(event)
    accumulator.finish()

    assert gate.terminated is accumulator._done is terminated
    if terminated:
        assert relayed == []
        assert gate.done_event == event
    else:
        assert b"".join(relayed) == event
        assert gate.done_event is None


def test_padded_done_does_not_hide_later_stream_content() -> None:
    stream = (
        b'data: [DONE]   \n\ndata: {"choices":[{"index":0,"delta":{"content":"REAL"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    gate = trace_sse.SseDoneGate()
    accumulator = trace_sse.SseAccumulator()

    relayed = gate.feed(stream)
    relayed.extend(gate.finish())
    accumulator.feed(stream)
    accumulator.finish()

    assert b"REAL" in b"".join(relayed)
    assert accumulator.output()["choices"][0]["message"]["content"] == "REAL"
    assert gate.terminated is accumulator._done is True


def test_padded_done_preserves_a_complete_reply_without_a_defect() -> None:
    stream = (
        b'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n'
        b"data: [DONE]   \n\n"
        b'data: {"choices":[{"index":0,"delta":{"content":" world"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    accumulator = trace_sse.SseAccumulator()

    accumulator.feed(stream)
    accumulator.finish()

    assert accumulator.output()["choices"][0]["message"]["content"] == "Hello world"
    assert accumulator.defect is None


@pytest.mark.parametrize(
    "event",
    [
        b'data: {"choices":[{"index":0,"delta":\n\ndata: [DONE]\n\n',
        b"data: <html>oops</html>\n\ndata: [DONE]\n\n",
        b"data: [DONE] and then some\n\ndata: [DONE]\n\n",
        b"data: [DONEISH]\n\ndata: [DONE]\n\n",
    ],
    ids=["truncated-json", "junk-payload", "done-with-junk-suffix", "doneish"],
)
def test_corrupt_data_events_remain_defects(event: bytes) -> None:
    accumulator = trace_sse.SseAccumulator()

    accumulator.feed(event)
    accumulator.finish()

    assert accumulator.defect == "stream contained an unparseable data event"
    assert accumulator._done is True


def test_done_gate_bounds_an_unterminated_done_candidate() -> None:
    gate = trace_sse.SseDoneGate()

    assert gate.feed(b"data: [DONE]") == []
    forwarded = []
    for _ in range(5):
        forwarded.extend(gate.feed(b" " * 100_000))

    assert gate.terminated is False
    assert gate.done_event is None
    assert len(gate._buffer) <= trace_sse._POST_DONE_SUFFIX_LIMIT
    assert b"".join(forwarded).startswith(b"data: [DONE]")


def test_done_gate_releases_an_unterminated_post_done_suffix() -> None:
    gate = trace_sse.SseDoneGate()
    stream = b"data: [DONE]\n" + b"x" * 500_000

    relayed = gate.feed(b"data: [DONE]\n")
    for _ in range(5):
        relayed.extend(gate.feed(b"x" * 100_000))
    relayed.extend(gate.finish())

    assert b"".join(relayed) == stream
    assert gate.terminated is False
    assert gate.done_event is None
    assert gate._buffer == b""


def test_done_gate_parks_an_eof_terminated_done_candidate() -> None:
    gate = trace_sse.SseDoneGate()

    assert gate.feed(b"data: [DONE]\n") == []
    assert gate.finish() == []

    assert gate.done_event == b"data: [DONE]\n"
    assert gate.terminated is True


def test_done_gate_discards_bytes_after_the_terminator() -> None:
    gate = trace_sse.SseDoneGate()
    chunk = b'data: [DONE]\n\ndata: {"late":1}\n\n'

    assert gate.feed(chunk) == []
    assert gate.finish() == []
    assert gate.done_event == b"data: [DONE]\n\n"
    assert gate._buffer == b""


@pytest.mark.parametrize(
    "chunks",
    [
        [b'data: [DONE]\ndata: {"x":1}\n\n'],
        [b"data: [DONE]\n", b'data: {"x":1}\n', b"\n"],
    ],
    ids=["one-chunk", "split"],
)
def test_done_gate_relays_a_multiline_done_event_whole(chunks: list[bytes]) -> None:
    """`[DONE]` on its own line does not end the event -- in SSE only a blank line does. The
    continuation lines belong to the terminator, so bounding the post-`[DONE]` suffix must not
    drop them: the relay is byte-exact or it is not a relay."""

    gate = trace_sse.SseDoneGate()
    forwarded: list[bytes] = []

    for chunk in chunks:
        forwarded.extend(gate.feed(chunk))
        if gate.terminated:
            break
    forwarded.extend(gate.finish())
    if gate.done_event is not None:
        forwarded.append(gate.done_event)

    assert b"".join(forwarded) == b'data: [DONE]\ndata: {"x":1}\n\n'
    assert gate._buffer == b""


def test_done_gate_releases_an_undelimited_multiline_post_done_suffix() -> None:
    """The suffix bound counts the retained terminator too. Once an undelimited event exceeds it,
    the candidate must be released rather than settled because a later data line can still change it."""

    gate = trace_sse.SseDoneGate()

    assert gate.feed(b"data: [DONE]\n") == []
    forwarded = []
    for _ in range(5_000):
        forwarded.extend(gate.feed(b": keepalive\n"))

    assert gate.terminated is False
    assert gate._buffer == b""
    assert gate.done_event is None
    assert b"".join(forwarded).startswith(b"data: [DONE]\n")


def test_done_gate_releases_a_large_second_data_line_and_later_events() -> None:
    """A second data line changes the SSE event's combined payload, so it was never a terminator.
    The genuine-terminator suffix bound must not truncate that legal event or anything after it."""

    first = b"data: [DONE]\ndata: " + b"x" * 2_000 + b"\n"
    completion = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
    suffix = b"\n" + completion + b":\n\n"
    assert len(first + suffix) == 2_072
    gate = trace_sse.SseDoneGate()

    forwarded = gate.feed(first)
    assert b"".join(forwarded) == first
    forwarded.extend(gate.feed(suffix))
    forwarded.extend(gate.finish())

    assert b"".join(forwarded) == first + suffix
    assert gate.terminated is False
    assert gate.done_event is None


def test_done_gate_releases_a_large_partial_second_data_line() -> None:
    gate = trace_sse.SseDoneGate()
    partial = b"data: " + b"y" * 2_000

    assert len(partial) == 2_006
    assert gate.feed(b"data: [DONE]\n") == []
    assert b"".join(gate.feed(partial)) == b"data: [DONE]\n" + partial
    assert gate.terminated is False
    assert gate.done_event is None
    assert gate._buffer == b""


def test_done_gate_releases_oversized_closed_comment_before_later_data() -> None:
    stream = (
        b"data: [DONE]\n"
        b":"
        + b"x" * 1_200
        + b'\ndata: {"choices":[{"index":0,"delta":{"content":"AFTER"}}]}\n\n'
        + b"data: [DONE]\n\n"
    )
    assert len(stream) == 1_290
    gate = trace_sse.SseDoneGate()
    accumulator = trace_sse.SseAccumulator()

    relayed = gate.feed(stream)
    relayed.extend(gate.finish())
    for chunk in relayed:
        accumulator.feed(chunk)
    if gate.done_event is not None:
        accumulator.feed(gate.done_event)
    accumulator.finish()

    assert b"".join(relayed) + (gate.done_event or b"") == stream
    assert len(b"".join(relayed)) == 1_276
    assert len(gate.done_event or b"") == 14
    assert b"AFTER" in b"".join(relayed)
    assert accumulator.received is False
    assert accumulator.output()["choices"] == []
    assert accumulator.defect == "stream contained an unparseable data event"


def test_done_gate_releases_oversized_closed_comment_across_chunks() -> None:
    chunk1 = b"data: [DONE]\n:" + b"x" * 1_200 + b"\n"
    chunk2 = b'data: {"choices":[{"index":0,"delta":{"content":"AFTER"}}]}\n\ndata: [DONE]\n\n'
    gate = trace_sse.SseDoneGate()
    accumulator = trace_sse.SseAccumulator()

    relayed = gate.feed(chunk1)
    assert gate.terminated is False
    relayed.extend(gate.feed(chunk2))
    relayed.extend(gate.finish())
    accumulator.feed(chunk1)
    accumulator.feed(chunk2)
    accumulator.finish()

    assert b"".join(relayed) + (gate.done_event or b"") == chunk1 + chunk2
    assert b"AFTER" in b"".join(relayed)
    assert gate.terminated is accumulator._done


def test_done_gate_releases_oversized_partial_comment_before_later_data() -> None:
    first = b"data: [DONE]\n:" + b"x" * 1_224
    suffix = b'\ndata: {"choices":[{"index":0,"delta":{"content":"AFTER"}}]}\n\ndata: [DONE]\n\n'
    stream = first + suffix
    gate = trace_sse.SseDoneGate()

    relayed = gate.feed(first)
    relayed.extend(gate.feed(suffix))
    relayed.extend(gate.finish())

    assert b"".join(relayed) + (gate.done_event or b"") == stream
    assert b"AFTER" in b"".join(relayed)
    assert gate.done_event == b"data: [DONE]\n\n"
    assert gate.terminated is True


def test_accumulator_marks_a_multiline_leading_done_event_unparseable() -> None:
    event = b'data: [DONE]\ndata: {"choices":[{"index":0,"delta":{"content":"X"}}]}\n\n'
    accumulator = trace_sse.SseAccumulator()

    accumulator.feed(event)
    accumulator.finish()

    assert accumulator.received is False
    assert accumulator.output()["choices"] == []
    assert accumulator.defect == "stream contained an unparseable data event"


def test_accumulator_handles_two_done_lines_in_one_event() -> None:
    accumulator = trace_sse.SseAccumulator()

    accumulator.feed(b"data: [DONE]\ndata: [DONE]\n\n")
    accumulator.finish()

    assert accumulator.received is False
    assert accumulator.output()["choices"] == []
    assert accumulator.defect == "stream contained an unparseable data event"


def test_done_gate_releases_a_newline_free_non_data_suffix() -> None:
    gate = trace_sse.SseDoneGate()
    stream = b"data: [DONE]\n:" + b"k" * 5_000

    relayed = gate.feed(b"data: [DONE]\n")
    relayed.extend(gate.feed(b":" + b"k" * 5_000))
    relayed.extend(gate.finish())

    assert b"".join(relayed) == stream
    assert gate.terminated is False
    assert gate.done_event is None
    assert gate._buffer == b""


def test_done_gate_partial_second_data_line_keeps_event_nonterminal() -> None:
    chunks = [b"data: [DONE]\n", b"data: a", b"\n", b"data: [DONE]\n", b"\n"]
    gate = trace_sse.SseDoneGate()

    forwarded = [part for chunk in chunks for part in gate.feed(chunk)]
    forwarded.extend(gate.finish())

    assert b"".join(forwarded) == b"".join(chunks)
    assert gate.terminated is False
    assert gate.done_event is None


def test_done_gate_drops_forwarded_data_from_terminator_state() -> None:
    gate = trace_sse.SseDoneGate()

    gate.feed(b"data: x\n")
    for _ in range(20_000):
        gate.feed(b"data: yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy\n")

    assert not hasattr(gate, "_event_data")
    assert gate._event_in_progress is True
    assert gate.terminated is False


def test_empty_string_deltas_do_not_accumulate_fragment_entries() -> None:
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=256)
    empty_event = b'data: {"choices":[{"index":0,"delta":{"content":""}}]}\n\n'

    for _ in range(10_000):
        accumulator.feed(empty_event)
    accumulator.feed(
        b'data: {"choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}]}\n\n'
    )

    content = accumulator._choices[0]["message"]["content"]
    assert content.parts == ["world"]
    assert accumulator.output()["choices"][0]["message"]["content"] == "world"
    assert accumulator.truncated is False


def test_repeated_sse_envelope_fields_do_not_consume_the_stream_budget() -> None:
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=700)
    fragment = "x"
    event = (
        b'data: {"id":"chatcmpl-x","model":"gpt-test","object":"chat.completion.chunk",'
        b'"created":123,"choices":[{"index":0,"delta":{"content":"x"}}]}\n\n'
    )

    for _ in range(512):
        accumulator.feed(event)
    accumulator.feed(
        b'data: {"id":"chatcmpl-x","model":"gpt-test","object":"chat.completion.chunk",'
        b'"created":123,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    )

    assert accumulator.truncated is False
    assert accumulator.output()["choices"][0]["message"]["content"] == fragment * 512


def test_tool_call_fragments_charge_only_retained_output_bytes() -> None:
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=100_000)
    event = (
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":'
        b'[{"index":0,"function":{"arguments":"x"}}]}}]}\n\n'
    )

    for _ in range(30_000):
        accumulator.feed(event)
    accumulator.feed(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n')

    output = accumulator.output()
    assert accumulator.truncated is False
    assert output["choices"][0]["finish_reason"] == "tool_calls"
    assert output["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] == (
        "x" * 30_000
    )
    assert accumulator._accumulated_bytes < 31_000


def test_function_call_fragments_charge_only_retained_output_bytes() -> None:
    accumulator = trace_sse.SseAccumulator(
        max_accumulated_bytes=platform_traces.MAX_PAYLOAD_TOTAL_BYTES
    )
    event = b'data: {"choices":[{"index":0,"delta":{"function_call":{"arguments":"x"}}}]}\n\n'

    for _ in range(20_000):
        accumulator.feed(event)
    accumulator.feed(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n')

    output = accumulator.output()
    assert accumulator.truncated is False
    assert output["choices"][0]["message"]["function_call"]["arguments"] == "x" * 20_000
    assert accumulator._accumulated_bytes < 21_000


def test_large_function_call_still_truncates_at_a_small_stream_budget() -> None:
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=4_096)
    event = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"function_call": {"arguments": "x" * 5_000}},
                }
            ]
        }
    ).encode()

    accumulator.feed(b"data: " + event + b"\n\n")

    assert accumulator.truncated is True


def test_fragment_accounting_does_not_materialize_arguments_per_delta(monkeypatch) -> None:
    text_calls = 0
    original_text = trace_sse._StringFragments.text

    def counted_text(fragments) -> str:
        nonlocal text_calls
        text_calls += 1
        return original_text(fragments)

    monkeypatch.setattr(trace_sse._StringFragments, "text", counted_text)
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=100_000)
    event = (
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":'
        b'[{"index":0,"function":{"arguments":"x"}}]}}]}\n\n'
    )

    for _ in range(8_000):
        accumulator.feed(event)

    assert text_calls == 0
    assert (
        accumulator.output()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        == "x" * 8_000
    )


def test_tool_call_id_and_type_duplicate_suppression_is_preserved() -> None:
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=100_000)
    event = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_1", "type": "function"},
                        ]
                    },
                }
            ]
        },
        separators=(",", ":"),
    ).encode()

    for _ in range(100):
        accumulator.feed(b"data: " + event + b"\n\n")

    tool_call = accumulator.output()["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["id"] == "call_1"
    assert tool_call["type"] == "function"


def test_tool_call_slots_count_toward_the_stream_budget() -> None:
    budget = 5_000
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=budget)

    for index in range(2_000):
        event = json.dumps(
            {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": index}]}}]}
        ).encode()
        accumulator.feed(b"data: " + event + b"\n\n")
        if accumulator.truncated:
            break

    assert accumulator.truncated is True
    assert accumulator.defect is None
    assert len(accumulator._choices[0]["tool_calls"]) < 2_000
    assert accumulator._accumulated_bytes <= budget


def test_materialization_depth_limit_records_a_defect_without_truncating() -> None:
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=100_000)
    nested: object = "leaf"
    for _ in range(2_000):
        nested = {"x": nested}
    accumulator._choices[0] = {
        "message": nested,
        "tool_calls": {},
        "logprobs": {},
        "extensions": {},
        "finish_reason": "stop",
    }

    output = accumulator.output()

    assert accumulator.defect == "stream output exceeded the maximum nesting depth"
    assert accumulator.truncated is False
    cursor = output["choices"][0]["message"]
    traversed = 0
    while isinstance(cursor, dict):
        cursor = cursor["x"]
        traversed += 1
    assert traversed <= platform_traces._MAX_PAYLOAD_DEPTH
    assert cursor == "[redacted]"


def test_distinct_sse_delta_field_names_count_toward_the_stream_budget() -> None:
    budget = 5_000
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=budget)

    for index in range(2_000):
        field = f"extension_{index}_" + "x" * 200
        event = json.dumps({"choices": [{"index": 0, "delta": {field: ""}}]}).encode()
        accumulator.feed(b"data: " + event + b"\n\n")
        if accumulator.truncated:
            break

    assert accumulator.truncated is True
    assert accumulator.defect is None
    assert len(json.dumps(accumulator.output()).encode()) <= budget


def test_distinct_sse_envelope_field_names_count_toward_the_stream_budget() -> None:
    budget = 5_000
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=budget)

    for index in range(2_000):
        field = f"extension_{index}_" + "x" * 200
        event = json.dumps({field: "", "choices": []}).encode()
        accumulator.feed(b"data: " + event + b"\n\n")
        if accumulator.truncated:
            break

    assert accumulator.truncated is True
    assert accumulator.defect is None
    assert len(accumulator._envelope) < 2_000
    assert len(json.dumps(accumulator.output()).encode()) <= budget + 200


def test_large_choice_indices_count_toward_the_stream_budget() -> None:
    budget = 5_000
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=budget)
    index = int("9" * 4_001)

    for offset in range(50):
        accumulator.feed(
            b"data: "
            + json.dumps({"choices": [{"index": index + offset, "delta": {}}]}).encode()
            + b"\n\n"
        )
        if accumulator.truncated:
            break

    assert accumulator.truncated is True
    assert accumulator.defect is None
    assert len(json.dumps(accumulator.output()).encode()) <= budget


def test_choice_level_extension_fields_survive_stream_accumulation() -> None:
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=1_000)
    accumulator.feed(
        b'data: {"choices":[{"index":0,"delta":{"content":"hello"},'
        b'"native_finish_reason":"STOP","safety":{"a":1},"finish_reason":"stop"}]}\n\n'
    )

    choice = accumulator.output()["choices"][0]
    assert choice["native_finish_reason"] == "STOP"
    assert choice["safety"] == {"a": 1}
    assert choice["message"]["content"] == "hello"
    assert choice["finish_reason"] == "stop"


def test_choice_level_extension_fields_count_toward_the_stream_budget() -> None:
    budget = 1_000
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=budget)

    for index in range(2_000):
        field = f"choice_extension_{index}_" + "x" * 100
        event = json.dumps({"choices": [{"index": 0, "delta": {}, field: "x" * 100}]}).encode()
        accumulator.feed(b"data: " + event + b"\n\n")
        if accumulator.truncated:
            break

    assert accumulator.truncated is True
    assert accumulator.defect is None
    assert len(json.dumps(accumulator.output()).encode()) <= budget + 200


def test_repeated_choice_extensions_do_not_recharge_the_stream_budget() -> None:
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=1_000)
    safety = "s" * 100

    for _ in range(20):
        event = json.dumps(
            {"choices": [{"index": 0, "delta": {"content": "x"}, "safety": safety}]}
        ).encode()
        accumulator.feed(b"data: " + event + b"\n\n")

    assert accumulator.truncated is False
    assert accumulator.defect is None
    assert accumulator.output()["choices"][0]["safety"] == safety
    assert accumulator.output()["choices"][0]["message"]["content"] == "x" * 20
    assert accumulator._accumulated_bytes < 300


def test_finish_reasons_count_toward_the_stream_budget() -> None:
    budget = 100_000
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=budget)

    for index in range(50):
        event = json.dumps(
            {"choices": [{"index": index, "delta": {}, "finish_reason": "x" * 4_000}]}
        ).encode()
        accumulator.feed(b"data: " + event + b"\n\n")

    assert accumulator.truncated is True
    assert len(json.dumps(accumulator.output()).encode()) <= budget


def test_unterminated_sse_line_is_bounded_by_the_accumulation_budget() -> None:
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=128)

    for _ in range(20):
        accumulator.feed(b"x" * 16)

    assert accumulator.truncated is True
    assert accumulator._buffer == b""


@pytest.mark.wallclock
def test_sse_accumulator_bytearray_feed_cost_stays_near_linear() -> None:
    def elapsed(feeds: int) -> float:
        best = float("inf")
        for _ in range(3):
            accumulator = trace_sse.SseAccumulator()
            started = time.perf_counter()
            for _ in range(feeds):
                accumulator.feed(b"x")
            best = min(best, time.perf_counter() - started)
        return best

    accumulator = trace_sse.SseAccumulator()
    assert isinstance(accumulator._buffer, bytearray)

    small = elapsed(40_000)
    large = elapsed(160_000)

    assert large / 160_000 < (small / 40_000) * 4


def test_unterminated_sse_line_scans_each_byte_once(monkeypatch) -> None:
    scanned_bytes = 0
    original_line_end = trace_sse._line_end

    def counted_line_end(data: bytes | bytearray, start: int = 0):
        nonlocal scanned_bytes
        result = original_line_end(data, start)
        scanned_bytes += (result[0] + 1 if result is not None else len(data)) - start
        return result

    monkeypatch.setattr(trace_sse, "_line_end", counted_line_end)
    accumulator = trace_sse.SseAccumulator()
    chunk = b"x" * (16 * 1024)

    for _ in range(64):
        accumulator.feed(chunk)

    assert scanned_bytes == 1024 * 1024


@pytest.mark.parametrize(
    ("line_ending", "terminated"),
    [(b"\n", True), (b"\r\n", True), (b"\n", False), (b"\r\n", False)],
    ids=["lf-delimited", "crlf-delimited", "lf-eof", "crlf-eof"],
)
def test_sse_accumulator_assembles_multiline_data_events(
    line_ending: bytes, terminated: bool
) -> None:
    accumulator = trace_sse.SseAccumulator()
    lines = [
        b"data: {",
        b'data: "choices": [{"index": 0, "delta": {"content": "world"},',
        b'data: "finish_reason": "stop"}]',
        b"data: }",
    ]
    event = line_ending.join(lines) + line_ending
    if terminated:
        event += line_ending

    accumulator.feed(event)
    accumulator.finish()

    if terminated:
        assert accumulator.defect is None
        assert accumulator.output()["choices"][0]["message"]["content"] == "world"
        assert accumulator.terminal is True
    else:
        assert accumulator.defect == "stream ended with an unterminated data event"
        assert accumulator.output()["choices"] == []
        assert accumulator.terminal is False

    bounded = trace_sse.SseAccumulator(max_accumulated_bytes=10)
    bounded.feed(b"data: 12345\ndata: 67890")
    assert bounded.truncated is True


def test_streaming_client_disconnect_before_any_event_records_no_output(
    trace_api, monkeypatch
) -> None:
    """An empty stream has no output, which is not an output with zero choices. Recording the
    synthesized `{"choices": []}` envelope would make `format=records` emit a training pair whose
    response half is empty -- the exact row that format promises to skip."""

    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody([])  # provider closed before its first event
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    with trace_api.stream(
        "POST", "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    ) as resp:
        assert resp.status_code == 200
        resp.read()

    raw = _raw(trace_api)
    assert raw["traces"] == 1
    assert raw["records"][0]["spans"][0]["output_payload"] is None
    # and the records export skips it rather than emitting an empty response half
    records = trace_api.get(
        "/api/traces/export",
        headers={"Authorization": f"Bearer {_KEY}"},
        params={"project_id": _PROJECT_ID, "format": "records"},
    ).json()
    assert records["records"] == []
    assert records["skipped"] == 1


@pytest.mark.parametrize(
    "chunks",
    [[], [b'data: {"choices":[],"usage":{"prompt_tokens":3}}\n\n']],
    ids=["empty", "usage-only"],
)
def test_unterminated_stream_without_choices_is_recorded_as_incomplete(
    trace_api, monkeypatch, chunks: list[bytes]
) -> None:
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(chunks)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "upstream stream ended before completion"
    assert span["output_payload"] is None


def test_a_completed_empty_stream_remains_ok(trace_api, monkeypatch) -> None:
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody([b"data: [DONE]\n\n"])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "OK"
    assert span["error"] is None
    assert span["output_payload"] is None


def test_streamed_trace_stops_accumulating_at_done(trace_api, monkeypatch) -> None:
    event = (
        b'data: {"choices":[{"index":0,"delta":{"content":"real"},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
        b'data: {"choices":[{"index":0,"delta":{"content":"LATE"}}]}\n\n'
    )
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody([event])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    expected = event[: event.index(b'data: {"choices"', len(b"data:"))]
    assert response.content == expected
    stored = _raw(trace_api)["records"][0]["spans"][0]["output_payload"]
    assert stored["choices"][0]["message"]["content"] == "real"


def test_a_cleanly_truncated_stream_is_not_a_training_target(trace_api, monkeypatch) -> None:
    """A provider can close a 200 stream mid-word without raising. Treating that partial text as a
    successful reply silently trains the model toward an answer the provider never completed."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [b'data: {"choices":[{"index":0,"delta":{"content":"cut-of"}}]}\n\n']
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    raw = _raw(trace_api)
    span = raw["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "upstream stream ended before completion"
    assert span["output_payload"]["choices"][0]["message"]["content"] == "cut-of"
    records = trace_api.get(
        "/api/traces/export",
        headers={"Authorization": f"Bearer {_KEY}"},
        params={"project_id": _PROJECT_ID, "format": "records"},
    ).json()
    assert records["records"] == []
    assert records["skipped"] == 1


def test_a_streamed_trace_keeps_response_envelope_fields(trace_api, monkeypatch) -> None:
    """A streamed reply carries the same response identity and provider extensions as a non-streamed
    reply. Dropping every top-level field except choices and usage makes raw traces unable to identify
    the paid response, its model revision, or provider-specific metadata."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"id":"chatcmpl-x","model":"gpt-old","created":123,"system_fingerprint":"fp_1","provider_extension":{"region":"west"},"choices":[{"index":0,"delta":{"content":"wor"}}]}\n\n',
            b'data: {"model":"gpt-new","choices":[{"index":0,"delta":{"content":"ld"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":4}}\n\ndata: [DONE]\n\n',
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    output = _raw(trace_api)["records"][0]["spans"][0]["output_payload"]
    assert output["id"] == "chatcmpl-x"
    assert output["model"] == "gpt-new"
    assert output["created"] == 123
    assert output["system_fingerprint"] == "fp_1"
    assert output["provider_extension"] == {"region": "west"}
    assert output["choices"][0]["message"]["content"] == "world"


def test_streamed_full_message_is_accumulated_within_the_budget() -> None:
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=100_000)
    event = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ]
        },
        separators=(",", ":"),
    ).encode()

    accumulator.feed(b"data: " + event + b"\n\n")
    accumulator.feed(b"data: [DONE]\n\n")

    assert accumulator.defect is None
    assert accumulator.truncated is False
    assert accumulator.output()["choices"][0]["message"] == {
        "role": "assistant",
        "content": "hello",
    }

    bounded = trace_sse.SseAccumulator(max_accumulated_bytes=64)
    bounded.feed(b"data: " + event + b"\n\n")
    assert bounded.truncated is True


def test_streaming_records_accumulated_response(trace_api, monkeypatch) -> None:
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"wor"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"ld"},"finish_reason":"stop"}],',
            b'"usage":{"prompt_tokens":3,"completion_tokens":4}}\n\ndata: [DONE]\n\n',
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={**_REQUEST, "stream": True},
    )

    assert response.status_code == 200
    assert b"data:" in response.content
    assert "set-cookie" not in response.headers
    raw = _raw(trace_api)
    span = raw["records"][0]["spans"][0]
    assert span["output_payload"]["choices"][0]["message"]["content"] == "world"
    assert span["input_tokens"] == 3
    assert span["output_tokens"] == 4
    assert span["error"] is None


@pytest.mark.anyio
async def test_streaming_client_disconnect_stores_partial_trace(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    owner = db.ensure_internal_key(_KEY)
    context = traces._UpstreamRequestContext(
        url="https://api.openai.com/v1/chat/completions",
        headers={},
        body={**_REQUEST, "stream": True},
        provider="openai",
        model="gpt-test",
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        metadata=None,
        secrets=(_KEY, _PROVIDER_KEY),
        started_at=traces.time.perf_counter(),
        record_trace=True,
    )
    body = _BlockingStreamingBody(
        b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
    )
    request = httpx.Request("POST", context.url)
    response = httpx.Response(
        200, headers={"content-type": "text/event-stream"}, stream=body, request=request
    )
    client = _StaticAsyncClient()
    stream = traces._stream_response(client=client, upstream_response=response, context=context)
    chunks: list[bytes] = []

    async def consume() -> None:
        async for chunk in stream:
            chunks.append(chunk)  # noqa: PERF401 - expose each chunk before cancellation

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(consume)
        await body.blocked.wait()
        task_group.cancel_scope.cancel()

    assert chunks == [body.first]
    exported = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000
    )
    assert exported["traces"] == 1
    span = exported["records"][0]["spans"][0]
    assert span["error"] == "client disconnected"
    assert span["status_code"] == "ERROR"
    assert span["output_payload"]["choices"][0]["message"]["content"] == "partial"
    assert body.closed is True
    assert client.closed is True


@pytest.mark.anyio
async def test_unrecorded_stream_stops_reading_after_done_event() -> None:
    context = traces._UpstreamRequestContext(
        url="https://api.openai.com/v1/chat/completions",
        headers={},
        body={**_REQUEST, "stream": True},
        provider="openai",
        model="gpt-test",
        key_id=1,
        project_id=None,
        metadata=None,
        secrets=(),
        started_at=traces.time.perf_counter(),
        record_trace=False,
    )

    class _PostDoneBlockingBody(_BlockingStreamingBody):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield self.first
            self.blocked.set()
            await asyncio.Event().wait()

    body = _PostDoneBlockingBody(b"data: [DONE]\n\n")
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=body,
        request=httpx.Request("POST", context.url),
    )
    client = _StaticAsyncClient()

    async def consume() -> list[bytes]:
        return [
            chunk
            async for chunk in traces._stream_response(
                client=client, upstream_response=response, context=context
            )
        ]

    consume_task = asyncio.create_task(consume())
    blocked_task = asyncio.create_task(body.blocked.wait())
    completed, pending = await asyncio.wait(
        {consume_task, blocked_task}, return_when=asyncio.FIRST_COMPLETED
    )
    try:
        assert consume_task in completed
        chunks = await consume_task
    finally:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    assert chunks == [body.first]
    assert body.blocked.is_set() is False
    assert body.closed is True
    assert client.closed is True


@pytest.mark.anyio
async def test_unrecorded_stream_does_not_accumulate_events(monkeypatch) -> None:
    """Recording off must not pay to parse events into a span that is then discarded.

    The stored row cannot detect this: nothing is stored either way. Only the work done is
    observable, so this counts `SseAccumulator.feed` calls. The done gate still has to run,
    because it decides stream termination, not recording.
    """
    context = traces._UpstreamRequestContext(
        url="https://api.openai.com/v1/chat/completions",
        headers={},
        body={**_REQUEST, "stream": True},
        provider="openai",
        model="gpt-test",
        key_id=1,
        project_id=None,
        metadata=None,
        secrets=(),
        started_at=traces.time.perf_counter(),
        record_trace=False,
    )
    events = (
        b"".join(
            b'data: {"choices":[{"index":0,"delta":{"content":"tok"}}]}\n\n' for _ in range(50)
        )
        + b"data: [DONE]\n\n"
    )
    fed: list[bytes] = []
    original_feed = trace_sse.SseAccumulator.feed

    def counting_feed(self, chunk: bytes) -> None:
        fed.append(chunk)
        original_feed(self, chunk)

    monkeypatch.setattr(trace_sse.SseAccumulator, "feed", counting_feed)
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=events,
        request=httpx.Request("POST", context.url),
    )
    client = _StaticAsyncClient()

    chunks = [
        chunk
        async for chunk in traces._stream_response(
            client=client, upstream_response=response, context=context
        )
    ]

    assert fed == []
    # the caller still receives the provider's bytes verbatim, terminator included
    assert b"".join(chunks) == events
    assert client.closed is True


@pytest.mark.anyio
async def test_recorded_stream_still_accumulates_events(tmp_path, monkeypatch) -> None:
    """Control for the gate above: with recording ON the same stream is still accumulated."""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    owner = db.ensure_internal_key(_KEY)
    context = traces._UpstreamRequestContext(
        url="https://api.openai.com/v1/chat/completions",
        headers={},
        body={**_REQUEST, "stream": True},
        provider="openai",
        model="gpt-test",
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        metadata=None,
        secrets=(_KEY, _PROVIDER_KEY),
        started_at=traces.time.perf_counter(),
        record_trace=True,
    )
    events = b'data: {"choices":[{"index":0,"delta":{"content":"kept"}}]}\n\ndata: [DONE]\n\n'
    fed: list[bytes] = []
    original_feed = trace_sse.SseAccumulator.feed

    def counting_feed(self, chunk: bytes) -> None:
        fed.append(chunk)
        original_feed(self, chunk)

    monkeypatch.setattr(trace_sse.SseAccumulator, "feed", counting_feed)
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=events,
        request=httpx.Request("POST", context.url),
    )

    chunks = [
        chunk
        async for chunk in traces._stream_response(
            client=_StaticAsyncClient(), upstream_response=response, context=context
        )
    ]

    assert fed != []
    assert b"".join(chunks) == events
    exported = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000
    )
    assert (
        exported["records"][0]["spans"][0]["output_payload"]["choices"][0]["message"]["content"]
        == "kept"
    )


@pytest.mark.anyio
async def test_stream_stops_reading_after_done_event(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    owner = db.ensure_internal_key(_KEY)
    context = traces._UpstreamRequestContext(
        url="https://api.openai.com/v1/chat/completions",
        headers={},
        body={**_REQUEST, "stream": True},
        provider="openai",
        model="gpt-test",
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        metadata=None,
        secrets=(_KEY, _PROVIDER_KEY),
        started_at=traces.time.perf_counter(),
        record_trace=True,
    )

    class _PostDoneBlockingBody(_BlockingStreamingBody):
        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield self.first
            self.blocked.set()
            await asyncio.wait_for(asyncio.Event().wait(), timeout=0.1)

    body = _PostDoneBlockingBody(
        b'data: {"choices":[{"index":0,"delta":{"content":"world"},'
        b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    )
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=body,
        request=httpx.Request("POST", context.url),
    )
    client = _StaticAsyncClient()

    chunks = [
        chunk
        async for chunk in traces._stream_response(
            client=client, upstream_response=response, context=context
        )
    ]

    assert b"".join(chunks) == body.first
    assert body.blocked.is_set() is False
    assert body.closed is True
    assert client.closed is True
    exported = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )
    assert exported["records"] == [{"input": "hello", "output": "world"}]


@pytest.mark.anyio
async def test_streaming_upstream_interruption_records_partial_trace(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    owner = db.ensure_internal_key(_KEY)
    context = traces._UpstreamRequestContext(
        url="https://api.openai.com/v1/chat/completions",
        headers={},
        body={**_REQUEST, "stream": True},
        provider="openai",
        model="gpt-test",
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        metadata=None,
        secrets=(_KEY, _PROVIDER_KEY),
        started_at=traces.time.perf_counter(),
        record_trace=True,
    )
    body = _StreamingBody(
        [b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'],
        error=httpx.ReadError("broken stream"),
    )
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=body,
        request=httpx.Request("POST", context.url),
    )
    client = _StaticAsyncClient()
    stream = traces._stream_response(client=client, upstream_response=response, context=context)

    assert b"partial" in await anext(stream)
    with pytest.raises(httpx.ReadError):
        await anext(stream)

    exported = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000
    )
    span = exported["records"][0]["spans"][0]
    assert span["error"] == "upstream stream interrupted"
    assert span["output_payload"]["choices"][0]["message"]["content"] == "partial"


@pytest.mark.anyio
async def test_streaming_upstream_error_records_raw_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    owner = db.ensure_internal_key(_KEY)
    context = traces._UpstreamRequestContext(
        url="https://api.openai.com/v1/chat/completions",
        headers={},
        body={**_REQUEST, "stream": True},
        provider="openai",
        model="gpt-test",
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        metadata=None,
        secrets=(_KEY, _PROVIDER_KEY),
        started_at=traces.time.perf_counter(),
        record_trace=True,
    )
    body = _StreamingBody([b'{"error":{"message":"rate limited"}}'])
    response = httpx.Response(429, stream=body, request=httpx.Request("POST", context.url))
    client = _StaticAsyncClient()

    chunks = [
        chunk
        async for chunk in traces._stream_response(
            client=client, upstream_response=response, context=context
        )
    ]

    assert chunks == [b'{"error":{"message":"rate limited"}}']
    exported = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000
    )
    span = exported["records"][0]["spans"][0]
    assert span["error"] == "upstream returned status 429"
    assert span["output_payload"] == {"error": {"message": "rate limited"}}


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("/v1/retry", "https://api.openai.com/v1/retry"),
        ("https://provider.example/retry", "https://provider.example/retry"),
    ],
    ids=["relative", "absolute"],
)
def test_provider_redirect_locations_resolve_against_the_upstream_url(
    location: str, expected: str
) -> None:
    headers = traces._safe_provider_response_headers(
        {"Location": location},
        status_code=302,
        upstream_url="https://api.openai.com/v1/chat/completions",
    )

    assert headers == {"Location": expected}
    assert (
        traces._safe_provider_response_headers(
            {"Location": location},
            status_code=200,
            upstream_url="https://api.openai.com/v1/chat/completions",
        )
        == {}
    )


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streamed"])
def test_relative_provider_redirects_target_the_upstream_origin(
    trace_api, monkeypatch, stream: bool
) -> None:
    redirect_body = b"retry upstream"
    if stream:

        class _RelativeRedirectClient(_StreamingAsyncClient):
            body = _StreamingBody([redirect_body])

            async def send(self, request, *, stream) -> httpx.Response:
                assert stream is True
                return httpx.Response(
                    302,
                    headers={"content-type": "text/plain", "location": "/v1/retry"},
                    stream=type(self).body,
                    request=request,
                )

        monkeypatch.setattr(traces.httpx, "AsyncClient", _RelativeRedirectClient)
    else:
        _StaticAsyncClient.response = httpx.Response(
            302,
            content=redirect_body,
            headers={"content-type": "text/plain", "location": "/v1/retry"},
        )
        monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={**_REQUEST, "stream": stream},
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["location"] == "https://api.openai.com/v1/retry"
    assert response.content == redirect_body


def test_a_redirect_is_recorded_but_not_exported_as_a_training_target(
    trace_api, monkeypatch
) -> None:
    """The proxy does not follow redirects, so a 3xx body is an interstitial rather than the model's
    reply. Marking it successful exports HTML or redirect metadata as the assistant target, even though
    raw export should still preserve the provider response for diagnosis."""
    redirect_body = b'<html><a href="https://provider.example/login">continue</a></html>'
    _StaticAsyncClient.response = httpx.Response(
        307,
        content=redirect_body,
        headers={
            "content-type": "text/html",
            "location": "https://provider.example/login",
            "set-cookie": "unsafe=1",
        },
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json=_REQUEST, follow_redirects=False
    )

    assert response.status_code == 307
    assert response.content == redirect_body
    assert response.headers["location"] == "https://provider.example/login"
    assert "set-cookie" not in response.headers
    raw = _raw(trace_api)
    span = raw["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "upstream returned status 307"
    assert span["output_payload"] == redirect_body.decode()
    records = trace_api.get(
        "/api/traces/export",
        headers={"Authorization": f"Bearer {_KEY}"},
        params={"project_id": _PROJECT_ID, "format": "records"},
    ).json()
    assert records["records"] == []
    assert records["skipped"] == 1


@pytest.mark.anyio
async def test_streaming_header_wait_cancellation_closes_client_and_propagates(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    monkeypatch.setenv("FLASH_STANDALONE", "1")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", _KEY)
    owner = db.ensure_internal_key(_KEY)
    body = json.dumps({**_REQUEST, "stream": True}).encode()
    sent_body = False

    async def receive() -> dict:
        nonlocal sent_body
        if sent_body:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent_body = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = traces.Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [
                (name.lower().encode(), value.encode()) for name, value in _HEADERS.items()
            ],
        },
        receive,
    )

    class _HeaderBlockingClient(_StaticAsyncClient):
        instance: ClassVar[_HeaderBlockingClient | None] = None
        send_started = asyncio.Event()

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            type(self).instance = self

        def build_request(self, method, url, *, headers, json) -> httpx.Request:
            return httpx.Request(method, url, headers=headers, json=json)

        async def send(self, request, *, stream) -> httpx.Response:
            assert stream is True
            type(self).send_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    monkeypatch.setattr(traces.httpx, "AsyncClient", _HeaderBlockingClient)
    task = asyncio.create_task(traces.chat_completions(request, owner))
    await _HeaderBlockingClient.send_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _HeaderBlockingClient.instance is not None
    assert _HeaderBlockingClient.instance.closed is True
    exported = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000
    )
    span = exported["records"][0]["spans"][0]
    assert span["error"] == "client disconnected"
    assert span["output_payload"] is None


@pytest.mark.anyio
async def test_non_streaming_cancellation_records_trace_and_propagates(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    monkeypatch.setenv("FLASH_STANDALONE", "1")
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", _KEY)
    owner = db.ensure_internal_key(_KEY)
    body = json.dumps(_REQUEST).encode()
    sent_body = False

    async def receive() -> dict:
        nonlocal sent_body
        if sent_body:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent_body = True
        return {"type": "http.request", "body": body, "more_body": False}

    request = traces.Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [
                (name.lower().encode(), value.encode()) for name, value in _HEADERS.items()
            ],
        },
        receive,
    )

    class _BodyBlockingClient(_StaticAsyncClient):
        read_started = asyncio.Event()

        class _StreamContext(_StaticAsyncClient._StreamContext):
            async def __aenter__(self) -> httpx.Response:
                _BodyBlockingClient.read_started.set()
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

    monkeypatch.setattr(traces.httpx, "AsyncClient", _BodyBlockingClient)
    task = asyncio.create_task(traces.chat_completions(request, owner))
    await _BodyBlockingClient.read_started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    exported = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000
    )
    span = exported["records"][0]["spans"][0]
    assert span["error"] == "client disconnected"
    assert span["output_payload"] is None


def test_upstream_transport_failure_returns_502_and_records(trace_api, monkeypatch) -> None:
    class _FailingClient(_StaticAsyncClient):
        def stream(self, method, url, *, headers, json):
            raise httpx.ConnectError("offline", request=httpx.Request(method, url))

    monkeypatch.setattr(traces.httpx, "AsyncClient", _FailingClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 502
    raw = _raw(trace_api)
    span = raw["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "upstream request failed"
    assert span["output_payload"] is None


def test_a_transport_failure_502_reports_when_persistence_failed(trace_api, monkeypatch) -> None:
    """The provider transport already failed, but raw export must still reveal that failure. If the
    diagnostic write also fails, the 502 must say so instead of looking recorded when no row exists."""

    class _FailingClient(_StaticAsyncClient):
        def stream(self, method, url, *, headers, json):
            raise httpx.ConnectError("offline", request=httpx.Request(method, url))

    monkeypatch.setattr(traces.httpx, "AsyncClient", _FailingClient)
    monkeypatch.setattr(
        traces,
        "store_trace",
        lambda **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 502
    assert response.headers["x-freesolo-record-failed"] == "true"
    assert _raw(trace_api)["traces"] == 0


def test_a_transport_failure_502_omits_record_failed_when_persistence_succeeded(
    trace_api, monkeypatch
) -> None:
    """The failure header means a missing trace, not merely an upstream failure. A successfully
    persisted transport error keeps the header absent so callers do not retry or flag a false gap."""

    class _FailingClient(_StaticAsyncClient):
        def stream(self, method, url, *, headers, json):
            raise httpx.ConnectError("offline", request=httpx.Request(method, url))

    monkeypatch.setattr(traces.httpx, "AsyncClient", _FailingClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 502
    assert "x-freesolo-record-failed" not in response.headers
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["error"] == "upstream request failed"


def test_record_false_proxies_without_project_or_storage(trace_api, monkeypatch) -> None:
    """Recording off still has to PROXY. Asserting only "200 and nothing stored" would be equally
    satisfied by returning an empty response and never calling the provider at all, which is the
    one outcome a caller would never accept: a successful-looking completion that is fabricated."""
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    _StaticAsyncClient.requests.clear()
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)
    headers = {key: value for key, value in _HEADERS.items() if key != "X-Freesolo-Project-Id"}
    headers["X-Freesolo-Record"] = "false"

    response = trace_api.post("/v1/chat/completions", headers=headers, json=_REQUEST)

    assert response.status_code == 200
    # the provider was really called, and the caller got the provider's own body back
    assert len(_StaticAsyncClient.requests) == 1
    assert _StaticAsyncClient.requests[0]["json"]["model"] == _REQUEST["model"]
    assert response.json() == _RESPONSE
    assert trace_api.get(
        "/api/traces/projects", headers={"Authorization": f"Bearer {_KEY}"}
    ).json() == {"projects": []}


@pytest.mark.parametrize(
    ("body", "expected_detail"),
    [
        ({"messages": []}, "model is required"),
        ({**_REQUEST, "metadata": "not-an-object"}, "metadata must be an object"),
    ],
)
def test_recording_validates_model_and_metadata(
    trace_api, monkeypatch, body, expected_detail
) -> None:
    _StaticAsyncClient.requests.clear()
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=body)

    assert response.status_code == 400
    assert response.json()["detail"] == expected_detail
    assert _raw(trace_api)["traces"] == 0
    # rejected locally BEFORE the provider is billed: forwarding first and refusing afterwards
    # would spend the caller's quota (and disclose the payload) on a request we already know is bad
    assert _StaticAsyncClient.requests == []


@pytest.mark.parametrize("model", [123, {"name": "gpt-test"}, ["gpt-test"]])
def test_recording_rejects_non_string_models(trace_api, monkeypatch, model: object) -> None:
    _StaticAsyncClient.requests.clear()
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "model": model}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "model must be a string"
    assert _StaticAsyncClient.requests == []

    for blank in ("", "  \t\n"):
        blank_response = trace_api.post(
            "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "model": blank}
        )
        assert blank_response.status_code == 400
        assert blank_response.json()["detail"] == "model is required"
        assert _StaticAsyncClient.requests == []

    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    body = {**_REQUEST, "model": " gpt-test "}
    valid_response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=body)
    assert valid_response.status_code == 200
    assert _StaticAsyncClient.requests[0]["json"] == body


@pytest.mark.parametrize(
    ("headers", "expected_status"),
    [
        ({key: value for key, value in _HEADERS.items() if key != "X-Freesolo-Project-Id"}, 400),
        ({**_HEADERS, "X-Freesolo-Project-Id": "not-a-uuid"}, 400),
        ({key: value for key, value in _HEADERS.items() if key != "Authorization"}, 401),
    ],
)
def test_recording_rejects_missing_invalid_project_or_auth(
    trace_api, monkeypatch, headers, expected_status
) -> None:
    _StaticAsyncClient.requests.clear()
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=headers, json=_REQUEST)

    assert response.status_code == expected_status
    assert trace_api.get(
        "/api/traces/projects", headers={"Authorization": f"Bearer {_KEY}"}
    ).json() == {"projects": []}
    # an unrecordable or unauthenticated request must not reach the provider first
    assert _StaticAsyncClient.requests == []


def test_an_oversized_declared_request_is_rejected_before_json_decoding(
    trace_api, monkeypatch
) -> None:
    """The persistence cap is too late to protect ingress. A declared oversized body must be refused
    before decoding or forwarding, even when its bytes are not valid JSON."""
    _StaticAsyncClient.requests.clear()
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers={**_HEADERS, "Content-Length": str(platform_traces.MAX_PAYLOAD_TOTAL_BYTES + 1)},
        content=b"not json",
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body exceeds the 8 MiB limit"
    assert _StaticAsyncClient.requests == []


def test_an_oversized_chunked_request_is_rejected_while_streaming(trace_api, monkeypatch) -> None:
    """Chunked requests have no trustworthy declared size. The reader must stop on the first chunk
    that crosses the cap rather than materializing the full attacker-controlled body."""
    _StaticAsyncClient.requests.clear()
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    def body_chunks():
        yield b"{" + b" " * (platform_traces.MAX_PAYLOAD_TOTAL_BYTES - 1)
        yield b"x"

    response = trace_api.post(
        "/v1/chat/completions",
        headers={**_HEADERS, "Transfer-Encoding": "chunked"},
        content=body_chunks(),
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body exceeds the 8 MiB limit"
    assert _StaticAsyncClient.requests == []


def test_a_small_request_still_reaches_the_provider_after_ingress_bounding(
    trace_api, monkeypatch
) -> None:
    """The streaming bound must remain transparent for ordinary JSON requests: the provider receives
    the same object and the caller receives the provider's response."""
    _StaticAsyncClient.requests.clear()
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 200
    assert response.json() == _RESPONSE
    assert len(_StaticAsyncClient.requests) == 1
    assert _StaticAsyncClient.requests[0]["json"] == _REQUEST


def test_export_formats_convert_and_count_skips(trace_api) -> None:
    owner = db.ensure_standalone_owner()
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="complete",
        metadata={"source": "test"},
        spans=[
            TraceSpan(
                input_payload={"messages": [{"role": "user", "content": "hello"}]},
                output_payload=_reply_envelope("world"),
            )
        ],
    )
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="prompt-only",
        metadata=None,
        spans=[TraceSpan(input_payload={"messages": [{"role": "user", "content": "sample me"}]})],
    )
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="empty",
        metadata=None,
        spans=[TraceSpan()],
    )

    auth = {"Authorization": f"Bearer {_KEY}"}
    records = trace_api.get(
        "/api/traces/export",
        headers=auth,
        params={"project_id": _PROJECT_ID, "format": "records"},
    ).json()
    prompts = trace_api.get(
        "/api/traces/export",
        headers=auth,
        params={"project_id": _PROJECT_ID, "format": "prompts"},
    ).json()
    raw = trace_api.get(
        "/api/traces/export",
        headers=auth,
        params={"project_id": _PROJECT_ID, "format": "raw"},
    ).json()

    assert records == {
        "records": [{"input": "hello", "output": "world"}],
        "traces": 3,
        "skipped": 2,
        "format": "records",
        "truncated": False,
    }
    assert prompts == {
        "records": [{"input": "sample me"}, {"input": "hello"}],
        "traces": 3,
        "skipped": 1,
        "format": "prompts",
        "truncated": False,
    }
    assert raw["traces"] == 3
    assert raw["skipped"] == 0
    assert raw["format"] == "raw"
    assert len(raw["records"]) == 3


def test_a_contextual_chat_envelope_is_preserved_only_in_raw_export(trace_api) -> None:
    """A converted row contains only one prompt string, so exporting the final user turn from a full
    transcript drops instructions and prior answers the target depends on. Raw remains the lossless
    escape hatch for consumers that can represent the complete conversation."""
    owner = db.ensure_standalone_owner()
    request = {
        "model": "gpt-test",
        "messages": [
            {"role": "system", "content": "be concise"},
            {"role": "user", "content": "older question"},
            {"role": "assistant", "content": "older answer"},
            {
                "role": "user",
                "content": [{"type": "text", "text": "latest "}, {"text": "question"}],
            },
        ],
    }
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "final "}, {"text": "answer"}],
                }
            }
        ]
    }
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="chat",
        metadata=None,
        spans=[TraceSpan(input_payload=request, output_payload=response, status_code="OK")],
    )

    records = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )
    prompts = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="prompts", limit=1000
    )
    raw = export_traces(key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000)

    assert records["records"] == []
    assert records["skipped"] == 1
    assert prompts["records"] == []
    assert prompts["skipped"] == 1
    span = raw["records"][0]["spans"][0]
    assert span["input_payload"] == request
    assert span["output_payload"] == response


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ([{"type": "image_url", "image_url": {"url": "u"}, "text": "caption"}], None),
        (
            [
                {
                    "type": "input_audio",
                    "input_audio": {"data": "d"},
                    "text": "transcript",
                }
            ],
            None,
        ),
        ([{"type": "file", "file": {"id": "f"}, "text": "filename"}], None),
        (
            [
                {"type": "text", "text": "look: "},
                {"type": "image_url", "image_url": {"url": "u"}, "text": "cap"},
            ],
            None,
        ),
        ([{"type": "image_url", "image_url": {"url": "u"}}], None),
        ([{"type": "text", "text": "plain"}], "plain"),
        ([{"text": "bare"}], "bare"),
        ("hello", "hello"),
    ],
    ids=[
        "image-with-text",
        "audio-with-text",
        "file-with-text",
        "mixed-text-image",
        "image-without-text",
        "typed-text",
        "untyped-text",
        "string",
    ],
)
def test_message_text_rejects_explicit_non_text_parts(content, expected) -> None:
    assert platform_traces._message_text(content) == expected


def test_a_non_chat_response_body_is_skipped_not_exported_as_the_target(trace_api) -> None:
    """A body that is not a chat envelope is malformed, not an alternative encoding of a reply.

    Every stored span comes from the proxy's chat.completions route, so the only way an output is
    not an envelope is that it never was one -- here a gateway's HTTP 200 login interstitial, which
    records as OK because the status said so. Falling back to the raw payload exported that JSON
    error object whole, as the exact text `records` trains the model to produce."""
    owner = db.ensure_standalone_owner()
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="interstitial",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload={"messages": [{"role": "user", "content": "q"}]},
                output_payload={"error": {"message": "upstream gateway: login required"}},
                status_code="OK",
            )
        ],
    )

    records = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert records["records"] == []
    assert records["skipped"] == 1


def test_a_non_chat_request_body_is_skipped_on_the_prompt_half_too(trace_api) -> None:
    """The same rule on the input half. A decode failure stores the provider's raw text, and
    exporting it as the prompt would train the model to answer an error page."""
    owner = db.ensure_standalone_owner()
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="scalar-request",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload="<html>gateway timeout</html>",
                output_payload={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
            )
        ],
    )

    prompts = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="prompts", limit=1000
    )

    assert prompts["records"] == []
    assert prompts["skipped"] == 1


def test_a_length_limited_completion_is_not_exported_as_a_training_target(trace_api) -> None:
    """A `length` stop cuts the paid completion at the token cap, often mid-word. Exporting that
    partial text as the desired output teaches the model to reproduce an answer that never finished."""
    owner = db.ensure_standalone_owner()
    response = _reply_envelope("cut of")
    response["choices"][0]["finish_reason"] = "length"
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="length limited",
        metadata=None,
        spans=[TraceSpan(input_payload=_REQUEST, output_payload=response)],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == []
    assert export["skipped"] == 1


def test_a_content_filtered_completion_is_not_exported_as_a_training_target(trace_api) -> None:
    """A content-filter stop can carry censored partial text. Treating it as a successful terminal
    reply turns provider safety intervention into the exact behavior the training row rewards."""
    owner = db.ensure_standalone_owner()
    response = _reply_envelope("partial")
    response["choices"][0]["finish_reason"] = "content_filter"
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="content filtered",
        metadata=None,
        spans=[TraceSpan(input_payload=_REQUEST, output_payload=response)],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == []
    assert export["skipped"] == 1


@pytest.mark.parametrize(
    ("action_key", "action"),
    [
        (
            "tool_calls",
            [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"id": 7}'},
                }
            ],
        ),
        ("function_call", {"name": "lookup", "arguments": '{"id": 7}'}),
    ],
)
def test_a_reply_with_text_and_tool_calls_is_not_exported_as_text_only(
    trace_api, action_key, action
) -> None:
    """One assistant action can contain both visible text and a tool invocation. Exporting only the
    text silently changes what happened and trains an incomplete target, so the whole row must skip.

    Deliberately reported as `finish_reason: "stop"`, which is what a provider sends when the model
    both answered and invoked. That isolates the message-payload guard: the finish-reason set does
    not reject this row, so if the payload check regresses the row exports and this test fails.
    """
    owner = db.ensure_standalone_owner()
    response = _reply_envelope("I will look that up.")
    response["choices"][0]["finish_reason"] = "stop"
    response["choices"][0]["message"][action_key] = action
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="tool call",
        metadata=None,
        spans=[TraceSpan(input_payload=_REQUEST, output_payload=response)],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == []
    assert export["skipped"] == 1


@pytest.mark.parametrize(
    ("action_key", "action"),
    [("tool_calls", {}), ("function_call", "")],
    ids=["tool-calls-dict", "function-call-string"],
)
def test_a_wrong_typed_action_container_is_not_exported_as_text(
    action_key: str, action: object
) -> None:
    response = _reply_envelope("partial")
    response["choices"][0]["finish_reason"] = "stop"
    response["choices"][0]["message"][action_key] = action

    assert platform_traces._chat_reply(response) is None


@pytest.mark.parametrize(
    ("tool_calls", "expected"),
    [([], "ok"), ([{"id": "call-1", "type": "function"}], None)],
    ids=["empty-list", "real-tool-call"],
)
def test_tool_call_action_controls_remain_distinct(
    tool_calls: list[dict], expected: str | None
) -> None:
    response = _reply_envelope("ok")
    response["choices"][0]["finish_reason"] = "stop"
    response["choices"][0]["message"]["tool_calls"] = tool_calls

    assert platform_traces._chat_reply(response) == expected


@pytest.mark.parametrize("audio", [{"id": "audio-1"}, ""], ids=["object", "wrong-type"])
def test_a_reply_with_audio_is_not_exported_as_text_only(audio: object) -> None:
    response = _reply_envelope("spoken and written")
    response["choices"][0]["finish_reason"] = "stop"
    response["choices"][0]["message"]["audio"] = audio

    assert platform_traces._chat_reply(response) is None

    text = _reply_envelope("ok")
    tool_call = _reply_envelope("partial")
    tool_call["choices"][0]["message"]["tool_calls"] = [{"id": "call-1"}]
    function_call = _reply_envelope("partial")
    function_call["choices"][0]["message"]["function_call"] = {"name": "lookup"}
    assert platform_traces._chat_reply(text) == "ok"
    assert platform_traces._chat_reply(tool_call) is None
    assert platform_traces._chat_reply(function_call) is None


@pytest.mark.parametrize("tool_calls", [None, [], {}])
def test_a_tool_call_finish_is_skipped_even_with_no_tool_calls_payload(
    trace_api, tool_calls
) -> None:
    """`finish_reason: "tool_calls"` disqualifies a row on its own, without a tool-call payload.

    Gating this on the message payload alone left a hole. A provider can report that the model
    stopped to invoke a tool while sending an empty or absent `tool_calls` array -- OpenAI-compatible
    backends behind OpenRouter do exactly this -- and both guards then pass: the finish reason was
    accepted, and an empty list is falsy. The model's narration ("I will look that up.") exported as
    the training target for an action that was never in the row.

    The finish reason is the provider stating what the assistant did. When it says the turn ended in
    an invocation, converted text cannot represent that turn, whatever the payload happens to carry.
    """
    owner = db.ensure_standalone_owner()
    response = _reply_envelope("I will look that up.")
    response["choices"][0]["finish_reason"] = "tool_calls"
    if tool_calls is not None:
        response["choices"][0]["message"]["tool_calls"] = tool_calls
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="empty tool call",
        metadata=None,
        spans=[TraceSpan(input_payload=_REQUEST, output_payload=response)],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == []
    assert export["skipped"] == 1


@pytest.mark.parametrize("role", ["tool", "user"])
def test_an_explicit_non_assistant_reply_role_is_skipped(trace_api, role) -> None:
    """A gateway can return the chat message shape with text owned by a tool or user. Treating that
    explicit role as the desired assistant target corrupts converted training records."""
    owner = db.ensure_standalone_owner()
    response = _reply_envelope("not an assistant reply")
    response["choices"][0]["message"]["role"] = role
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="wrong reply role",
        metadata=None,
        spans=[TraceSpan(input_payload=_REQUEST, output_payload=response)],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == []
    assert export["skipped"] == 1


@pytest.mark.parametrize("role", [[], {}, 1, True])
def test_an_unhashable_or_non_string_reply_role_is_skipped(role) -> None:
    response = _reply_envelope("not an assistant reply")
    response["choices"][0]["message"]["role"] = role

    assert platform_traces._chat_reply(response) is None


@pytest.mark.parametrize("role", [None, "assistant"])
def test_an_absent_or_assistant_reply_role_still_exports(trace_api, role) -> None:
    """Older providers may omit the response role, while current providers send `assistant`.
    Tightening explicit bad roles must preserve both accepted success shapes."""
    owner = db.ensure_standalone_owner()
    response = _reply_envelope("complete")
    if role is None:
        response["choices"][0]["message"].pop("role")
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="accepted reply role",
        metadata=None,
        spans=[TraceSpan(input_payload=_REQUEST, output_payload=response)],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == [{"input": "hello", "output": "complete"}]
    assert export["skipped"] == 0


@pytest.mark.parametrize(
    ("error", "exported"),
    [
        ({"message": "upstream model failure", "type": "server_error"}, False),
        ("upstream model failure", False),
        (None, True),
        ({}, False),
        (False, False),
        ([], False),
        (0, False),
        ("", False),
    ],
    ids=[
        "object-error",
        "string-error",
        "null",
        "empty-object",
        "false",
        "empty-list",
        "zero",
        "empty-string",
    ],
)
def test_records_skip_only_meaningful_top_level_error_envelopes(
    trace_api, error: object, exported: bool
) -> None:
    owner = db.ensure_standalone_owner()
    response = _reply_envelope("good")
    response["choices"][0]["finish_reason"] = "stop"
    response["error"] = error
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="response error envelope",
        metadata=None,
        spans=[TraceSpan(input_payload=_REQUEST, output_payload=response)],
    )

    raw = export_traces(key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000)
    records = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert raw["records"][0]["spans"][0]["output_payload"] == response
    assert records["records"] == ([{"input": "hello", "output": "good"}] if exported else [])
    assert records["skipped"] == (0 if exported else 1)


def test_records_accept_absent_and_per_choice_error_fields() -> None:
    absent = _reply_envelope("good")
    nested = _reply_envelope("good")
    nested["choices"][0]["error"] = {"message": "choice metadata"}

    assert platform_traces._chat_reply(absent) == "good"
    assert platform_traces._chat_reply(nested) == "good"


def test_a_text_only_reply_still_exports_after_tool_call_filtering(trace_api) -> None:
    """The tool-call guard must not reduce ordinary completed assistant text, which remains a complete
    action and is the primary records export path.
    """
    owner = db.ensure_standalone_owner()
    response = _reply_envelope("complete")
    response["choices"][0]["finish_reason"] = "stop"
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="text only",
        metadata=None,
        spans=[TraceSpan(input_payload=_REQUEST, output_payload=response)],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == [{"input": "hello", "output": "complete"}]
    assert export["skipped"] == 0


def test_a_completion_without_a_finish_reason_still_exports(trace_api) -> None:
    """Older and non-OpenAI-shaped providers may omit `finish_reason` entirely. Absence is not an
    explicit failure signal, so rejecting it would silently empty otherwise valid provider exports."""
    owner = db.ensure_standalone_owner()
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="no finish reason",
        metadata=None,
        spans=[TraceSpan(input_payload=_REQUEST, output_payload=_reply_envelope("complete"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == [{"input": "hello", "output": "complete"}]
    assert export["skipped"] == 0


@pytest.mark.parametrize("finish_reason", [[], {}, 1, True])
def test_an_unhashable_or_non_string_finish_reason_is_skipped(finish_reason) -> None:
    response = _reply_envelope("not a clean reply")
    response["choices"][0]["finish_reason"] = finish_reason

    assert platform_traces._chat_reply(response) is None


def test_an_unknown_finish_reason_is_not_treated_as_a_clean_stop(trace_api) -> None:
    """An unknown explicit reason is not evidence that generation completed successfully. Accepting
    every new string would silently ship partial targets when a provider adds another failure reason."""
    owner = db.ensure_standalone_owner()
    response = _reply_envelope("uncertain")
    response["choices"][0]["finish_reason"] = "provider_abort"
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="unknown finish reason",
        metadata=None,
        spans=[TraceSpan(input_payload=_REQUEST, output_payload=response)],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == []
    assert export["skipped"] == 1


def test_a_string_truncated_payload_is_skipped_by_converted_exports(trace_api) -> None:
    """The stored ellipsis is indistinguishable from real authored text. Once any payload value was
    shortened, neither records nor prompts may turn that mutated trace into training data."""
    owner = db.ensure_standalone_owner()
    oversized_prompt = "x" * (platform_traces.MAX_PAYLOAD_VALUE_LENGTH + 1)
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="truncated prompt",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload={"messages": [{"role": "user", "content": oversized_prompt}]},
                output_payload=_reply_envelope("reply"),
            )
        ],
    )

    records = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )
    prompts = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="prompts", limit=1000
    )

    assert records["records"] == []
    assert records["skipped"] == 1
    assert prompts["records"] == []
    assert prompts["skipped"] == 1


@pytest.mark.anyio
async def test_a_redaction_depth_clipped_payload_is_marked_and_skipped(
    trace_api, monkeypatch
) -> None:
    owner = db.ensure_standalone_owner()
    nested: object = "leaf"
    for _ in range(platform_traces._MAX_PAYLOAD_DEPTH + 10):
        nested = {"level": nested}
    context = traces._UpstreamRequestContext(
        url="https://api.openai.com/v1/chat/completions",
        headers={},
        body={"model": "gpt-test", "messages": [{"role": "user", "content": nested}]},
        provider="openai",
        model="gpt-test",
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        metadata=None,
        secrets=(),
        started_at=traces.time.perf_counter(),
        record_trace=True,
    )

    await traces._record_trace(
        context,
        output_payload=_reply_envelope("reply"),
        error=None,
    )

    raw = _raw(trace_api)
    span = raw["records"][0]["spans"][0]
    records = trace_api.get(
        "/api/traces/export",
        headers={"Authorization": f"Bearer {_KEY}"},
        params={"project_id": _PROJECT_ID, "format": "records"},
    ).json()

    assert span["attributes"] == {"payload_truncated": ["input"]}
    assert records["records"] == []
    assert records["skipped"] == 1


def test_a_collection_clipped_payload_is_marked_and_skipped(trace_api, monkeypatch) -> None:
    """Dropping a collection tail changes the request just as surely as shortening a string. Without
    the marker, the bounded prefix looked intact and converted into a prompt the model never received.
    """
    monkeypatch.setattr(platform_traces, "_MAX_PAYLOAD_COLLECTION", 2)
    owner = db.ensure_standalone_owner()
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="clipped collection",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"text": "first "},
                                {"text": "second"},
                                {"text": " dropped"},
                            ],
                        }
                    ]
                },
                output_payload=_reply_envelope("reply"),
            )
        ],
    )

    raw = export_traces(key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000)
    records = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    span = raw["records"][0]["spans"][0]
    assert span["input_payload"]["messages"] == [
        {"role": "user", "content": [{"text": "first "}, {"text": "second"}]}
    ]
    assert span["attributes"] == {"payload_truncated": ["input"]}
    assert records["records"] == []
    assert records["skipped"] == 1


def test_an_output_only_truncation_keeps_the_intact_prompt(trace_api) -> None:
    """Prompts deliberately need no usable reply. An oversized completion must still disqualify a
    records pair, but sharing its marker with the intact request silently discarded valid GRPO input.
    """
    owner = db.ensure_standalone_owner()
    oversized_reply = "y" * (platform_traces.MAX_PAYLOAD_VALUE_LENGTH + 1)
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="output truncated",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload=_REQUEST,
                output_payload=_reply_envelope(oversized_reply),
            )
        ],
    )

    prompts = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="prompts", limit=1000
    )
    records = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )
    raw = export_traces(key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000)

    assert prompts["records"] == [{"input": "hello"}]
    assert prompts["skipped"] == 0
    assert records["records"] == []
    assert records["skipped"] == 1
    assert raw["records"][0]["spans"][0]["attributes"] == {"payload_truncated": ["output"]}


def test_caller_output_marker_unions_with_detected_input_truncation(trace_api) -> None:
    owner = db.ensure_standalone_owner()
    oversized_prompt = "x" * (platform_traces.MAX_PAYLOAD_VALUE_LENGTH + 1)
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="both sides truncated",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload={"messages": [{"role": "user", "content": oversized_prompt}]},
                output_payload=_reply_envelope("partial reply"),
                attributes={"payload_truncated": ["output"]},
            )
        ],
    )

    raw = export_traces(key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000)
    records = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )
    prompts = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="prompts", limit=1000
    )

    assert raw["records"][0]["spans"][0]["attributes"] == {"payload_truncated": ["input", "output"]}
    assert records["records"] == []
    assert records["skipped"] == 1
    assert prompts["records"] == []
    assert prompts["skipped"] == 1


@pytest.mark.parametrize("caller_value", ["output", ["bogus"]])
def test_bogus_caller_truncation_marker_is_discarded(trace_api, caller_value: object) -> None:
    owner = db.ensure_standalone_owner()
    oversized_prompt = "x" * (platform_traces.MAX_PAYLOAD_VALUE_LENGTH + 1)
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="invalid marker",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload={"messages": [{"role": "user", "content": oversized_prompt}]},
                output_payload=_reply_envelope("reply"),
                attributes={"payload_truncated": caller_value},
            )
        ],
    )

    raw = export_traces(key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000)

    assert raw["records"][0]["spans"][0]["attributes"] == {"payload_truncated": ["input"]}


def test_an_input_truncation_disqualifies_records_and_prompts(trace_api) -> None:
    """Every converted shape requires the request. If its stored text was shortened, neither a pair
    nor a prompt-only row represents what the provider saw, even when the reply remains intact.
    """
    owner = db.ensure_standalone_owner()
    oversized_prompt = "x" * (platform_traces.MAX_PAYLOAD_VALUE_LENGTH + 1)
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="input truncated",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload={"messages": [{"role": "user", "content": oversized_prompt}]},
                output_payload=_reply_envelope("reply"),
            )
        ],
    )

    records = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )
    prompts = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="prompts", limit=1000
    )

    assert records["records"] == []
    assert records["skipped"] == 1
    assert prompts["records"] == []
    assert prompts["skipped"] == 1


def test_a_string_truncated_payload_remains_available_in_raw_export(trace_api) -> None:
    """Raw is the diagnostic escape hatch and must never hide a trace because converted training
    formats reject it. It preserves both the bounded payload and the marker explaining the mutation."""
    owner = db.ensure_standalone_owner()
    oversized_reply = "y" * (platform_traces.MAX_PAYLOAD_VALUE_LENGTH + 1)
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="truncated reply",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload=_REQUEST,
                output_payload=_reply_envelope(oversized_reply),
            )
        ],
    )

    raw = export_traces(key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000)

    span = raw["records"][0]["spans"][0]
    assert raw["skipped"] == 0
    assert span["output_payload"]["choices"][0]["message"]["content"].endswith("...")
    assert span["attributes"] == {"payload_truncated": ["output"]}


def test_unexpected_truncation_marker_shapes_do_not_hide_converted_rows() -> None:
    """Stored attributes are caller-facing raw data and may be absent, malformed, or from another
    producer. Only the exact side-list marker written by this recorder may suppress converted output.
    """
    base_span = {
        "input_payload": _REQUEST,
        "output_payload": _reply_envelope("reply"),
        "status_code": "OK",
    }
    for attributes in (None, "not-a-dict", {"payload_truncated": True}, {"payload_truncated": 7}):
        raw = {"spans": [{**base_span, "attributes": attributes}]}
        assert platform_traces._export_record(raw, "records") == {
            "input": "hello",
            "output": "reply",
        }


def test_an_untruncated_trace_keeps_null_attributes(trace_api) -> None:
    """The truncation marker is exceptional state, not default metadata. Writing an empty attributes
    object on every happy-path span changes existing raw records and needlessly grows the database."""
    owner = db.ensure_standalone_owner()
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="ordinary",
        metadata=None,
        spans=[TraceSpan(input_payload=_REQUEST, output_payload=_reply_envelope("reply"))],
    )

    raw = export_traces(key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000)

    assert raw["records"][0]["spans"][0]["attributes"] is None


def test_a_system_instruction_makes_the_single_turn_export_unreachable(trace_api) -> None:
    """A target conditioned on a system instruction cannot be learned from the user text alone.
    Dropping that instruction creates a context-free row whose output is unreachable from its input."""
    owner = db.ensure_standalone_owner()
    request = {
        "messages": [
            {"role": "system", "content": "answer in rhyme"},
            {"role": "user", "content": "describe rain"},
        ]
    }
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="system context",
        metadata=None,
        spans=[TraceSpan(input_payload=request, output_payload=_reply_envelope("reply"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == []
    assert export["skipped"] == 1


@pytest.mark.parametrize("field", ["prediction", "web_search_options"])
def test_top_level_content_context_makes_the_exported_prompt_unreachable(
    trace_api, field: str
) -> None:
    owner = db.ensure_standalone_owner()
    request = {
        **_REQUEST,
        field: {"content": "Alice"} if field == "prediction" else {"search_context_size": "high"},
    }
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="top level context",
        metadata=None,
        spans=[TraceSpan(input_payload=request, output_payload=_reply_envelope("Alice"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == []
    assert export["skipped"] == 1


@pytest.mark.parametrize(
    "field",
    [
        "tool_choice",
        "function_call",
        "response_schema",
        "prediction",
        "web_search_options",
    ],
)
def test_present_empty_instruction_context_makes_the_exported_prompt_unreachable(
    trace_api, field: str
) -> None:
    owner = db.ensure_standalone_owner()
    request = {**_REQUEST, field: {}}
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="empty top level context",
        metadata=None,
        spans=[TraceSpan(input_payload=request, output_payload=_reply_envelope("reply"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == []
    assert export["skipped"] == 1


@pytest.mark.parametrize("field", ["tools", "functions"])
def test_empty_instruction_collections_are_unset(trace_api, field: str) -> None:
    owner = db.ensure_standalone_owner()
    request = {**_REQUEST, field: []}
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="empty instruction collection",
        metadata=None,
        spans=[TraceSpan(input_payload=request, output_payload=_reply_envelope("reply"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == [{"input": "hello", "output": "reply"}]


def test_explicit_null_instruction_context_is_unset(trace_api) -> None:
    owner = db.ensure_standalone_owner()
    request = {**_REQUEST, "web_search_options": None}
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="null top level context",
        metadata=None,
        spans=[TraceSpan(input_payload=request, output_payload=_reply_envelope("reply"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == [{"input": "hello", "output": "reply"}]


def test_plain_text_response_format_remains_exportable(trace_api) -> None:
    owner = db.ensure_standalone_owner()
    request = {**_REQUEST, "response_format": {"type": "text"}}
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="plain text response format",
        metadata=None,
        spans=[TraceSpan(input_payload=request, output_payload=_reply_envelope("reply"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == [{"input": "hello", "output": "reply"}]


def test_a_response_schema_makes_the_exported_prompt_unreachable(trace_api) -> None:
    """A JSON schema can determine the entire reply shape while the user text stays vague. Omitting
    that top-level instruction makes both converted formats claim an input that cannot reach the row."""
    owner = db.ensure_standalone_owner()
    request = {
        **_REQUEST,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        },
    }
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="schema context",
        metadata=None,
        spans=[
            TraceSpan(input_payload=request, output_payload=_reply_envelope('{"answer":"yes"}'))
        ],
    )

    records = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )
    prompts = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="prompts", limit=1000
    )

    assert records["records"] == []
    assert records["skipped"] == 1
    assert prompts["records"] == []
    assert prompts["skipped"] == 1


def test_a_non_empty_tool_list_makes_the_exported_prompt_unreachable(trace_api) -> None:
    """Available tools instruct the model which actions and arguments it may produce. User text alone
    does not contain that contract, so a reply conditioned on it is not a faithful converted row."""
    owner = db.ensure_standalone_owner()
    request = {
        **_REQUEST,
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
    }
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="tool context",
        metadata=None,
        spans=[TraceSpan(input_payload=request, output_payload=_reply_envelope("use lookup"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == []
    assert export["skipped"] == 1


def test_an_empty_tool_list_does_not_disqualify_a_single_user_message(trace_api) -> None:
    """An empty tool list carries no instruction. Treating mere field presence as context would drop
    ordinary rows emitted by clients that serialize optional arrays unconditionally."""
    owner = db.ensure_standalone_owner()
    request = {**_REQUEST, "tools": []}
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="empty tools",
        metadata=None,
        spans=[TraceSpan(input_payload=request, output_payload=_reply_envelope("world"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == [{"input": "hello", "output": "world"}]
    assert export["skipped"] == 0


def test_ordinary_sampling_controls_do_not_disqualify_a_single_user_message(trace_api) -> None:
    """Sampling knobs shape decoding but add no missing instruction to the exported prompt. Rejecting
    them would discard the normal proxy path, including streamed collection requests."""
    owner = db.ensure_standalone_owner()
    request = {**_REQUEST, "temperature": 0.2, "max_tokens": 64, "stream": True}
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="sampling controls",
        metadata=None,
        spans=[TraceSpan(input_payload=request, output_payload=_reply_envelope("world"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == [{"input": "hello", "output": "world"}]
    assert export["skipped"] == 0


@pytest.mark.parametrize("field", ["name", "tool_call_id", "provider_context"])
def test_per_message_context_makes_the_single_turn_export_unreachable(
    trace_api, field: str
) -> None:
    owner = db.ensure_standalone_owner()
    request = {"messages": [{"role": "user", "content": "What is my name?", field: "alice"}]}
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="message context",
        metadata=None,
        spans=[TraceSpan(input_payload=request, output_payload=_reply_envelope("Alice"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == []
    assert export["skipped"] == 1


@pytest.mark.parametrize("value", [None, "", [], {}])
def test_empty_per_message_extensions_do_not_disqualify_a_single_user_message(
    trace_api, value
) -> None:
    owner = db.ensure_standalone_owner()
    request = {"messages": [{"role": "user", "content": "hello", "provider_context": value}]}
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="empty message context",
        metadata=None,
        spans=[TraceSpan(input_payload=request, output_payload=_reply_envelope("world"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == [{"input": "hello", "output": "world"}]


def test_a_single_user_message_still_exports_as_training_data(trace_api) -> None:
    """Correctness filtering must retain the ordinary case: a sole user turn contains all available
    request context, so its completed reply remains a reachable and useful training target."""
    owner = db.ensure_standalone_owner()
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="single turn",
        metadata=None,
        spans=[TraceSpan(input_payload=_REQUEST, output_payload=_reply_envelope("world"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == [{"input": "hello", "output": "world"}]


def test_a_trailing_assistant_prefill_invalidates_the_single_turn_export(trace_api) -> None:
    """A trailing assistant message can be a prefill that constrains the generated continuation.
    Exporting only the preceding user text drops that context just as surely as dropping prior turns."""
    owner = db.ensure_standalone_owner()
    request = {
        "messages": [
            {"role": "user", "content": "complete this"},
            {"role": "assistant", "content": "Once upon a"},
        ]
    }
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="assistant prefill",
        metadata=None,
        spans=[TraceSpan(input_payload=request, output_payload=_reply_envelope(" time"))],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert export["records"] == []
    assert export["skipped"] == 1


def test_an_aggregate_oversized_payload_is_dropped_marked_and_skipped(
    trace_api, monkeypatch
) -> None:
    """Many individually valid strings can still form one attacker-sized sqlite value. Persistence
    must replace that aggregate with a small diagnostic placeholder and keep it out of training data."""
    monkeypatch.setattr(platform_traces, "MAX_PAYLOAD_TOTAL_BYTES", 300)
    owner = db.ensure_standalone_owner()
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="aggregate oversized",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload={
                    "messages": [{"role": "user", "content": "x" * 400}],
                },
                output_payload=_reply_envelope("reply"),
            )
        ],
    )

    raw = export_traces(key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000)
    records = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    span = raw["records"][0]["spans"][0]
    dropped = span["input_payload"]["flash_payload_dropped"]
    assert dropped["reason"] == "payload exceeded the stored size limit"
    assert dropped["bytes"] > platform_traces.MAX_PAYLOAD_TOTAL_BYTES
    assert span["attributes"] == {"payload_truncated": ["input"]}
    assert records["records"] == []
    assert records["skipped"] == 1


def test_the_aggregate_payload_cap_counts_encoded_bytes_not_characters(
    trace_api, monkeypatch
) -> None:
    """Non-ASCII text occupies multiple UTF-8 bytes. A character-count cap would let the stored blob
    exceed its nominal budget several times over even though the same payload passes a code-point count."""
    monkeypatch.setattr(platform_traces, "MAX_PAYLOAD_TOTAL_BYTES", 500)
    owner = db.ensure_standalone_owner()
    content = "\U0001f600" * 150
    payload = {"messages": [{"role": "user", "content": content}]}
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    assert len(serialized) < platform_traces.MAX_PAYLOAD_TOTAL_BYTES
    assert len(serialized.encode("utf-8")) > platform_traces.MAX_PAYLOAD_TOTAL_BYTES
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="encoded bytes",
        metadata=None,
        spans=[TraceSpan(input_payload=payload, output_payload=_reply_envelope("reply"))],
    )

    raw = export_traces(key_id=owner["id"], project_id=_PROJECT_ID, export_format="raw", limit=1000)

    span = raw["records"][0]["spans"][0]
    assert "flash_payload_dropped" in span["input_payload"]
    assert span["attributes"] == {"payload_truncated": ["input"]}


def test_a_capped_export_reports_that_it_is_incomplete(trace_api) -> None:
    """The read is a capped window of the newest traces. Reporting the truncated count on its own
    reads as "that is all of them", so a project past the cap would look exported whole while its
    older traces were silently absent from the dataset."""
    owner = db.ensure_standalone_owner()
    for index in range(3):
        store_trace(
            key_id=owner["id"],
            project_id=_PROJECT_ID,
            trace_title=f"trace-{index}",
            metadata=None,
            spans=[
                TraceSpan(
                    input_payload={"messages": [{"role": "user", "content": str(index)}]},
                    output_payload=_reply_envelope("reply"),
                )
            ],
        )

    auth = {"Authorization": f"Bearer {_KEY}"}
    capped = trace_api.get(
        "/api/traces/export",
        headers=auth,
        params={"project_id": _PROJECT_ID, "format": "records", "limit": 2},
    ).json()
    whole = trace_api.get(
        "/api/traces/export",
        headers=auth,
        params={"project_id": _PROJECT_ID, "format": "records", "limit": 50},
    ).json()

    assert capped["truncated"] is True
    assert capped["traces"] == 2
    assert whole["truncated"] is False
    assert whole["traces"] == 3


def test_an_export_stops_at_the_byte_budget_and_says_it_is_truncated(
    trace_api, monkeypatch
) -> None:
    """The trace cap bounds rows, not bytes. Payload strings may legitimately reach 1 MB, so the
    row cap alone permits a multi-gigabyte body -- built in memory on the plane, serialized again
    by the framework, then held whole by the CLI. The byte budget has to stop it, and stopping
    early has to report `truncated` or the export looks like it read the project whole."""
    monkeypatch.setattr(platform_traces, "MAX_EXPORT_BYTES", 4_000)
    owner = db.ensure_standalone_owner()
    for index in range(4):
        store_trace(
            key_id=owner["id"],
            project_id=_PROJECT_ID,
            trace_title=f"bulky-{index}",
            metadata=None,
            spans=[
                TraceSpan(
                    input_payload={"messages": [{"role": "user", "content": "q"}]},
                    output_payload=_reply_envelope("y" * 2_000),
                )
            ],
        )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    # one row overshoots the budget by design: the first record is always emitted whole, so an
    # export can never come back empty merely because its newest trace is large.
    assert len(export["records"]) == 2
    assert export["truncated"] is True
    # `traces` counts what was examined, so `skipped` stays the count of unusable rows rather than
    # silently absorbing the ones the budget cut.
    assert export["traces"] == 2
    assert export["skipped"] == 0


def test_errored_spans_are_not_exported_as_training_targets(trace_api) -> None:
    """A failed call's output is the provider's rejection -- a 429 body, an interrupted partial --
    not a reply to train toward. `raw` keeps it; `records` must not present it as the target."""
    owner = db.ensure_standalone_owner()
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="rate limited",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload={"prompt": "hello"},
                output_payload={"error": {"message": "rate limit exceeded"}},
                status_code="ERROR",
                error="upstream returned status 429",
            )
        ],
    )

    auth = {"Authorization": f"Bearer {_KEY}"}
    records = trace_api.get(
        "/api/traces/export", headers=auth, params={"project_id": _PROJECT_ID, "format": "records"}
    ).json()
    raw = trace_api.get(
        "/api/traces/export", headers=auth, params={"project_id": _PROJECT_ID, "format": "raw"}
    ).json()

    assert records["records"] == []
    assert records["skipped"] == 1
    # the prompt is still a usable prompt, and raw never drops anything
    assert raw["traces"] == 1
    assert raw["records"][0]["spans"][0]["status_code"] == "ERROR"


def test_a_long_completion_is_stored_whole(trace_api) -> None:
    """Payloads are the product: a recorded completion becomes a training row, so cutting it at the
    8 KiB attribute bound would export a truncated target with an ellipsis and no sign it was cut."""
    owner = db.ensure_standalone_owner()
    long_reply = "x" * 40_000
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="long",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload={"messages": [{"role": "user", "content": "hi"}]},
                output_payload=_reply_envelope(long_reply),
            )
        ],
    )

    records = trace_api.get(
        "/api/traces/export",
        headers={"Authorization": f"Bearer {_KEY}"},
        params={"project_id": _PROJECT_ID, "format": "records"},
    ).json()

    assert records["records"] == [{"input": "hi", "output": long_reply}]


def test_conventional_cloud_credential_fields_are_redacted(trace_api, monkeypatch) -> None:
    """Cloud SDKs conventionally name credentials `access_key_id`, `secret_key`, and
    `secret_access_key`. Suffix matching missed all three normalized forms, so unrelated credentials
    in metadata or tool arguments survived into raw exports even though ordinary tokens did not.
    """
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)
    body = {
        **_REQUEST,
        "metadata": {
            "access_key_id": "AKIAEXAMPLE",
            "secret_key": "short-secret",
            "secret_access_key": "long-secret",
        },
        "tool_arguments": {"accesskey": "tool-secret"},
    }

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=body)

    assert response.status_code == 200
    stored = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]
    assert stored["metadata"] == {
        "access_key_id": "[redacted]",
        "secret_key": "[redacted]",
        "secret_access_key": "[redacted]",
    }
    assert stored["tool_arguments"]["accesskey"] == "[redacted]"


def test_a_bare_key_field_is_not_mistaken_for_a_credential(trace_api, monkeypatch) -> None:
    """`key` is ordinary data in schemas and tool arguments. Redacting it to catch access keys would
    mutate common requests and make the stored raw trace differ from the call the provider received.
    """
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)
    body = {
        **_REQUEST,
        "tool_arguments": {"key": "customer-id"},
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                    },
                },
            }
        ],
    }

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=body)

    assert response.status_code == 200
    stored = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]
    assert stored["tool_arguments"]["key"] == "customer-id"
    assert stored["tools"][0]["function"]["parameters"]["properties"]["key"] == {"type": "string"}


def test_custom_vocabulary_keeps_confirmed_schema_property_shaped() -> None:
    payload = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "login",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "password": {
                                "type": "string",
                                "vendorKeyword": True,
                                "default": "SECRET",
                            }
                        },
                    },
                },
            }
        ]
    }

    stored = traces._redact_secret_fields(payload)

    assert stored["tools"][0]["function"]["parameters"]["properties"]["password"] == {
        "type": "string",
        "vendorKeyword": True,
        "default": "[redacted]",
    }


def test_bare_properties_with_custom_vocabulary_remains_instance_data() -> None:
    payload = {"properties": {"password": {"type": "text", "value": "SECRET"}}}

    assert traces._redact_secret_fields(payload)["properties"]["password"] == "[redacted]"


@pytest.mark.parametrize(
    "wrapper",
    ["schema", "parameters", "input_schema", "output_schema"],
)
def test_schema_wrapper_names_require_real_schema_evidence(wrapper: str) -> None:
    instance = {wrapper: {"properties": {"password": {"type": "text", "value": "SECRET"}}}}
    schema = {
        wrapper: {
            "type": "object",
            "properties": {
                "password": {
                    "type": "string",
                    "vendorKeyword": True,
                    "default": "SECRET",
                }
            },
        }
    }

    assert traces._redact_secret_fields(instance)[wrapper]["properties"]["password"] == (
        "[redacted]"
    )
    assert traces._redact_secret_fields(schema)[wrapper]["properties"]["password"] == {
        "type": "string",
        "vendorKeyword": True,
        "default": "[redacted]",
    }


def test_wrapper_key_with_type_contradicting_its_applicators_is_instance_data() -> None:
    """`{"type": "string", "properties": {...}}` is not a schema: a string has no properties.

    A wrapper-named key plus a syntactically valid `type` was enough to claim the schema exemption,
    so ordinary metadata shaped this way kept its literals verbatim in raw exports.
    """
    payload = {
        "parameters": {
            "type": "string",
            "properties": {"password": {"type": "string", "value": "SECRET"}},
        }
    }

    stored = traces._redact_secret_fields(payload)

    assert "SECRET" not in json.dumps(stored)
    assert stored["parameters"]["properties"]["password"] == "[redacted]"


def test_wrapper_key_with_coherent_type_keeps_schema_shape() -> None:
    """Control for the check above: `type: "object"` agrees with `properties`, so it is a schema."""
    payload = {
        "parameters": {
            "type": "object",
            "properties": {"password": {"type": "string", "default": "SECRET"}},
        }
    }

    stored = traces._redact_secret_fields(payload)

    assert stored["parameters"]["properties"]["password"] == {
        "type": "string",
        "default": "[redacted]",
    }


def test_legacy_dependencies_map_preserves_subschema_entries() -> None:
    """Draft-07 `dependencies` is polymorphic: a schema value is a subschema, an array is not.

    Replacing a secret-named schema entry with the string "[redacted]" made the stored JSON Schema
    invalid, the same defect already fixed for `dependentSchemas`.
    """
    schema = {
        "type": "object",
        "dependencies": {"password": {"type": "object", "default": "SECRET-DEPENDENCY"}},
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["dependencies"]["password"] == {"type": "object", "default": "[redacted]"}
    # the ARRAY form lists required property names: instance data, redacted as before
    array_form = traces._redact_secret_fields(
        {"type": "object", "dependencies": {"password": ["a", "b"]}}
    )
    assert array_form["dependencies"]["password"] == "[redacted]"


def test_recursive_ref_reaches_an_outer_recursive_anchor() -> None:
    """`$recursiveRef` resolves against the dynamic scope, so an ENCLOSING anchor can be the target.

    A sibling embedded resource is different: it is not on the reference's evaluation path, so its
    anchor is never in scope and its literals must survive.
    """
    nested = {
        "$id": "https://example.com/root",
        "$recursiveAnchor": True,
        "default": "SECRET-OUTER",
        "$defs": {
            "Inner": {
                "$id": "https://example.com/inner",
                "$recursiveAnchor": True,
                "default": "SECRET-INNER",
                "properties": {"password": {"$recursiveRef": "#"}},
            }
        },
    }

    stored = traces._redact_secret_fields(nested)

    assert stored["default"] == "[redacted]"
    assert stored["$defs"]["Inner"]["default"] == "[redacted]"


def test_scalar_auth_key_is_redacted() -> None:
    """A bare `auth` key carries third-party credentials but matched neither exact nor suffix set."""
    stored = traces._redact_secret_fields(
        {"auth": "Bearer third-party-secret-value", "author": "amy"}
    )

    assert stored["auth"] == "[redacted]"
    # `author` ends in "auth" backwards but is an ordinary word: exact-matched, never a suffix
    assert stored["author"] == "amy"


def test_malformed_redirect_location_does_not_abandon_the_trace() -> None:
    """`urljoin` raises on a malformed IPv6 authority, and this helper runs after the paid call.

    Letting it propagate cost the caller its trace and, on the streaming path, skipped the generator
    that closes the upstream response and client.
    """
    safe = traces._safe_provider_response_headers(
        {"location": "http://[broken"},
        status_code=302,
        upstream_url="https://api.openai.com/v1/chat/completions",
    )

    assert safe["location"] == "http://[broken"
    resolved = traces._safe_provider_response_headers(
        {"location": "/v2/chat"},
        status_code=302,
        upstream_url="https://api.openai.com/v1/chat/completions",
    )
    assert resolved["location"] == "https://api.openai.com/v2/chat"


def test_duplicate_choice_index_within_one_event_is_malformed() -> None:
    """Two entries sharing an index in ONE event would merge into a reply the provider never sent.

    The same index across SUCCESSIVE events is ordinary streaming and must keep concatenating.
    """
    duplicated = trace_sse.SseAccumulator()
    duplicated.feed(
        b'data: {"choices":[{"index":0,"delta":{"content":"A"}},'
        b'{"index":0,"delta":{"content":"B"}}]}\n\ndata: [DONE]\n\n'
    )
    duplicated.finish()

    assert duplicated.defect == "stream event repeated a choice index"

    successive = trace_sse.SseAccumulator()
    successive.feed(
        b'data: {"choices":[{"index":0,"delta":{"content":"A"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":"B"}}]}\n\ndata: [DONE]\n\n'
    )
    successive.finish()

    assert successive.defect is None
    assert successive.output()["choices"][0]["message"]["content"] == "AB"


def test_refusal_only_reply_is_exported() -> None:
    """A refusal IS the assistant's complete reply; skipping it drops every safety row silently."""
    refusal_only = {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "refusal": "I cannot help"},
            }
        ]
    }

    assert platform_traces._chat_reply(refusal_only) == "I cannot help"
    # both present is a malformed combination with no single faithful target
    both = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi", "refusal": "no"}}
        ]
    }
    assert platform_traces._chat_reply(both) is None
    non_string = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": None, "refusal": {"x": 1}}}
        ]
    }
    assert platform_traces._chat_reply(non_string) is None


def test_an_empty_schema_under_a_secret_name_survives_redaction(trace_api, monkeypatch) -> None:
    """`{}` is the permissive JSON Schema, so `{"password": {}}` is a declaration, not a secret.
    Replacing it with the string "[redacted]" turns a valid schema into an invalid one."""
    _StaticAsyncClient.requests = []
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "login",
                "parameters": {"type": "object", "properties": {"password": {}}},
            },
        }
    ]

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "tools": tools}
    )

    assert response.status_code == 200
    stored = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]
    assert stored["tools"][0]["function"]["parameters"]["properties"]["password"] == {}


def test_schema_map_keywords_preserve_subschemas_and_redact_their_literals() -> None:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "dependentSchemas": {
            "password": {
                "type": "object",
                "properties": {"token": {"type": "string", "default": "SECRET-DEPENDENT"}},
            }
        },
        "patternProperties": {"^secret_": {"type": "string", "default": "SECRET-PATTERN"}},
        "metadata": {"password": "SECRET-INSTANCE"},
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["dependentSchemas"]["password"] == {
        "type": "object",
        "properties": {
            "token": {"type": "string", "default": "[redacted]"},
        },
    }
    assert stored["patternProperties"]["^secret_"] == {
        "type": "string",
        "default": "[redacted]",
    }
    assert stored["metadata"]["password"] == "[redacted]"


@pytest.mark.parametrize("container", ["properties", "$defs", "definitions"])
def test_schema_container_names_do_not_exempt_instance_secrets(container: str) -> None:
    schema = {
        "type": "object",
        "properties": {
            "password": {"type": "string", "default": "SECRET"},
        },
    }
    instance = {container: {"password": {"type": "text", "value": "SECRET"}}}
    nested_instance = {"metadata": instance}
    typed_extension = {"type": "object", **instance}
    direct = {"password": {"type": "text", "value": "SECRET"}}

    stored_schema = traces._redact_secret_fields(schema)

    assert stored_schema["properties"]["password"] == {
        "type": "string",
        "default": "[redacted]",
    }
    assert traces._redact_secret_fields(instance)[container]["password"] == "[redacted]"
    assert traces._redact_secret_fields(nested_instance)["metadata"][container]["password"] == (
        "[redacted]"
    )
    assert traces._redact_secret_fields(typed_extension)[container]["password"] == "[redacted]"
    assert traces._redact_secret_fields(direct)["password"] == "[redacted]"


@pytest.mark.parametrize("keyword", ["default", "examples", "example"])
def test_annotation_only_secret_named_schema_stays_schema_shaped(keyword: str) -> None:
    literal = ["SECRET"] if keyword == "examples" else "SECRET"
    schema = {
        "properties": {
            "password": {keyword: literal},
        },
        "metadata": {"password": {keyword: literal}},
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["properties"]["password"] == {
        keyword: ["[redacted]"] if keyword == "examples" else "[redacted]"
    }
    assert stored["metadata"]["password"] == "[redacted]"


def test_secret_named_schema_literals_cross_embedded_resource_boundaries() -> None:
    literals = {
        "default": "SECRET-DEFAULT",
        "const": "SECRET-CONST",
        "enum": ["SECRET-ENUM"],
        "examples": ["SECRET-EX"],
    }
    identified = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "password": {
                "$id": "https://example.com/pw",
                "type": "string",
                **literals,
            }
        },
    }
    nested = {
        "$schema": "x",
        "type": "object",
        "properties": {
            "password": {
                "$id": "https://example.com/pw",
                "type": "object",
                "properties": {
                    "inner": {
                        "$id": "https://example.com/pw/i",
                        "default": "SECRET-INNER",
                    }
                },
            }
        },
    }
    control = {
        "$schema": identified["$schema"],
        "type": "object",
        "properties": {"password": {"type": "string", **literals}},
    }
    redacted_literals = {
        "default": "[redacted]",
        "const": "[redacted]",
        "enum": ["[redacted]"],
        "examples": ["[redacted]"],
    }

    stored_identified = traces._redact_secret_fields(identified)
    stored_nested = traces._redact_secret_fields(nested)
    stored_control = traces._redact_secret_fields(control)

    assert stored_identified["properties"]["password"] == {
        "$id": "https://example.com/pw",
        "type": "string",
        **redacted_literals,
    }
    assert stored_nested["properties"]["password"]["properties"]["inner"] == {
        "$id": "https://example.com/pw/i",
        "default": "[redacted]",
    }
    assert stored_control["properties"]["password"] == {
        "type": "string",
        **redacted_literals,
    }


def test_secret_named_schema_literals_are_redacted_without_losing_structure(
    trace_api, monkeypatch
) -> None:
    third_party_secret = "third-party-secret-abc123"
    schema = {
        "type": "object",
        "properties": {
            "password": {
                "type": "object",
                "description": "credential payload",
                "default": {"value": third_party_secret},
                "const": third_party_secret,
                "enum": [third_party_secret],
                "examples": [{"value": third_party_secret}],
                "example": {"value": third_party_secret},
                "properties": {
                    "value": {"type": "string", "default": third_party_secret},
                },
                "required": ["value"],
            },
            "label": {"type": "string", "default": "ordinary-default"},
        },
    }
    tools = [
        {
            "type": "function",
            "function": {"name": "login", "parameters": schema},
        }
    ]
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "tools": tools}
    )

    assert response.status_code == 200
    properties = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]["tools"][0]["function"][
        "parameters"
    ]["properties"]
    password = properties["password"]
    assert password["type"] == "object"
    assert password["description"] == "credential payload"
    assert password["required"] == ["value"]
    assert password["default"] == {"value": "[redacted]"}
    assert password["const"] == "[redacted]"
    assert password["enum"] == ["[redacted]"]
    assert password["examples"] == [{"value": "[redacted]"}]
    assert password["example"] == {"value": "[redacted]"}
    assert password["properties"]["value"] == {
        "type": "string",
        "default": "[redacted]",
    }
    assert properties["label"]["default"] == "ordinary-default"
    assert third_party_secret not in json.dumps(_raw(trace_api))


def test_secret_schema_literal_object_keys_are_redacted(trace_api, monkeypatch) -> None:
    literal_secret = "third_party_password"
    schema = {
        "type": "object",
        "properties": {
            "password": {
                "type": "object",
                "default": {literal_secret: True},
            }
        },
    }
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={
            **_REQUEST,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        },
    )

    assert response.status_code == 200
    stored_default = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]["response_format"][
        "json_schema"
    ]["schema"]["properties"]["password"]["default"]
    assert stored_default == {"[redacted]": "[redacted]"}
    assert literal_secret not in json.dumps(_raw(trace_api))


def test_secret_named_schema_ref_literals_are_redacted(trace_api, monkeypatch) -> None:
    secret = "third-party-secret-abc123"
    schema = {
        "type": "object",
        "properties": {"password": {"$ref": "#/$defs/Benign"}},
        "$defs": {
            "Benign": {"type": "string", "default": secret, "enum": [secret]},
        },
    }
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={
            **_REQUEST,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        },
    )

    assert response.status_code == 200
    stored_schema = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]["response_format"][
        "json_schema"
    ]["schema"]
    assert stored_schema["properties"] == schema["properties"]
    assert stored_schema["$defs"]["Benign"] == {
        "type": "string",
        "default": "[redacted]",
        "enum": ["[redacted]"],
    }


@pytest.mark.parametrize(
    "ref",
    ["#/$defs/Alpha", "#/%24defs/Alpha", "#Alpha", "#Alph%61"],
    ids=["pointer", "encoded-pointer", "anchor", "encoded-anchor"],
)
def test_percent_encoded_secret_schema_refs_are_redacted(trace_api, monkeypatch, ref: str) -> None:
    schema = {
        "type": "object",
        "properties": {"api_key": {"$ref": ref}},
        "$defs": {
            "Alpha": {
                "$anchor": "Alpha",
                "default": "LEAKED-DEFAULT",
                "const": "LEAKED-CONST",
                "enum": ["LEAKED-ENUM"],
            }
        },
    }

    stored = _recorded_response_schema(trace_api, monkeypatch, schema)

    assert stored["$defs"]["Alpha"]["default"] == "[redacted]"
    assert stored["$defs"]["Alpha"]["const"] == "[redacted]"
    assert stored["$defs"]["Alpha"]["enum"] == ["[redacted]"]


def test_percent_encoded_unreserved_schema_resource_uri_matches() -> None:
    schema = {
        "$id": "https://example.com/schema",
        "type": "object",
        "properties": {
            "api_key": {"$ref": "https://example.com/%73chema#/$defs/Cred"},
        },
        "$defs": {
            "Cred": {
                "default": "SECRET-PCT",
                "const": "SECRET-C",
                "enum": ["SECRET-E"],
            }
        },
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["$defs"]["Cred"] == {
        "default": "[redacted]",
        "const": "[redacted]",
        "enum": ["[redacted]"],
    }
    assert trace_redaction._canonical_resource_uri("https://example.com/%73chema/%7e") == (
        "https://example.com/schema/~"
    )


def test_http_root_resource_ids_match_explicit_slash_refs() -> None:
    schema = {
        "$id": "https://example.com",
        "type": "object",
        "properties": {
            "password": {"$ref": "https://example.com/#/$defs/Cred"},
        },
        "$defs": {"Cred": {"type": "string", "default": "SECRET"}},
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["$defs"]["Cred"]["default"] == "[redacted]"
    assert trace_redaction._canonical_resource_uri("https://example.com") == (
        "https://example.com/"
    )
    assert trace_redaction._canonical_resource_uri("https://other.example.com") != (
        trace_redaction._canonical_resource_uri("https://example.com/")
    )
    assert trace_redaction._canonical_resource_uri("https://example.com/path") != (
        trace_redaction._canonical_resource_uri("https://example.com/")
    )
    assert trace_redaction._canonical_resource_uri("https://example.com/%2F") != (
        trace_redaction._canonical_resource_uri("https://example.com/")
    )
    assert trace_redaction._canonical_resource_uri("https://example.com/a/../b") == (
        "https://example.com/b"
    )
    assert trace_redaction._canonical_resource_uri("urn:example:") == "urn:example:"


def test_percent_encoded_reserved_schema_resource_uri_stays_distinct() -> None:
    schema = {
        "$id": "https://example.com/root",
        "type": "object",
        "properties": {
            "api_key": {"$ref": "https://example.com/a%2fb#/$defs/Cred"},
        },
        "$defs": {
            "Encoded": {
                "$id": "https://example.com/a%2Fb",
                "$defs": {"Cred": {"default": "SECRET-ENCODED"}},
            },
            "Slash": {
                "$id": "https://example.com/a/b",
                "$defs": {"Cred": {"default": "PUBLIC-SLASH"}},
            },
        },
    }

    stored = traces._redact_secret_fields(schema)["$defs"]

    assert stored["Encoded"]["$defs"]["Cred"]["default"] == "[redacted]"
    assert stored["Slash"]["$defs"]["Cred"]["default"] == "PUBLIC-SLASH"
    assert trace_redaction._canonical_resource_uri("https://example.com/a%2fb") == (
        "https://example.com/a%2Fb"
    )


@pytest.mark.parametrize(
    ("ref", "redacted"),
    [
        ("https://example.com/a/./schema#/$defs/Cred", True),
        ("https://example.com/a/b/../schema#/$defs/Cred", True),
        ("https://example.com/%61/schema#/$defs/Cred", True),
        ("https://example.com/b/schema#/$defs/Cred", False),
        ("https://example.com/a%2Fschema#/$defs/Cred", False),
    ],
    ids=["dot", "parent", "unreserved", "distinct-path", "reserved-slash"],
)
def test_dot_segment_schema_resource_refs_match_only_the_same_resource(
    ref: str, redacted: bool
) -> None:
    schema = {
        "$id": "https://example.com/a/schema",
        "type": "object",
        "properties": {"api_key": {"$ref": ref}},
        "$defs": {
            "Cred": {
                "default": "SECRET",
                "const": "SECRET-C",
                "enum": ["SECRET-E"],
            }
        },
    }

    stored = traces._redact_secret_fields(schema)["$defs"]["Cred"]

    expected = (
        {"default": "[redacted]", "const": "[redacted]", "enum": ["[redacted]"]}
        if redacted
        else {"default": "SECRET", "const": "SECRET-C", "enum": ["SECRET-E"]}
    )
    assert stored == expected


@pytest.mark.parametrize(
    ("uri", "canonical"),
    [
        ("https://example.com/a/./schema", "https://example.com/a/schema"),
        ("https://example.com/a/b/../schema", "https://example.com/a/schema"),
        ("https://example.com/../../a/schema", "https://example.com/a/schema"),
        ("https://example.com/a/.", "https://example.com/a/"),
        ("https://example.com/a/b/..", "https://example.com/a/"),
        ("https://example.com/a/", "https://example.com/a/"),
    ],
    ids=["dot", "parent", "leading-parent", "trailing-dot", "trailing-parent", "trailing-slash"],
)
def test_canonical_resource_uri_removes_dot_segments(uri: str, canonical: str) -> None:
    assert trace_redaction._canonical_resource_uri(uri) == canonical


def test_secret_schema_ref_with_extension_keyword_redacts_neutral_target() -> None:
    assert traces._is_secret_key("Widget") is False
    schema = {
        "type": "object",
        "properties": {
            "api_key": {"$ref": "#/$defs/Widget", "vendorKeyword": True},
        },
        "$defs": {
            "Widget": {
                "type": "string",
                "default": "SECRET-M1-DEFAULT",
                "enum": ["SECRET-M1-ENUM"],
            }
        },
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["$defs"]["Widget"] == {
        "type": "string",
        "default": "[redacted]",
        "enum": ["[redacted]"],
    }


@pytest.mark.parametrize(
    ("schema_id", "ref"),
    [
        ("https://example.com/schema", "https://example.com:443/schema#/$defs/Widget"),
        ("http://example.com/schema", "http://example.com:80/schema#/$defs/Widget"),
    ],
    ids=["https-443", "http-80"],
)
def test_schema_resource_uris_strip_scheme_default_ports(schema_id: str, ref: str) -> None:
    assert traces._is_secret_key("Widget") is False
    schema = {
        "$id": schema_id,
        "type": "object",
        "properties": {"api_key": {"$ref": ref}},
        "$defs": {
            "Widget": {
                "type": "string",
                "default": "SECRET-N1",
                "enum": ["SECRET-N1-ENUM"],
            }
        },
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["$defs"]["Widget"] == {
        "type": "string",
        "default": "[redacted]",
        "enum": ["[redacted]"],
    }


def test_schema_resource_uris_preserve_nondefault_ports() -> None:
    assert traces._is_secret_key("Widget") is False
    schema = {
        "$id": "https://example.com/schema",
        "type": "object",
        "properties": {
            "api_key": {"$ref": "https://example.com:8443/schema#/$defs/Widget"},
        },
        "$defs": {"Widget": {"type": "string", "default": "PUBLIC-8443"}},
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["$defs"]["Widget"]["default"] == "PUBLIC-8443"
    assert trace_redaction._canonical_resource_uri("https://example.com:/schema") == (
        "https://example.com/schema"
    )
    assert trace_redaction._canonical_resource_uri("https://example.com:8443/schema") == (
        "https://example.com:8443/schema"
    )


def test_schema_resource_uris_normalize_scheme_and_host_only() -> None:
    assert traces._is_secret_key("Widget") is False
    schema = {
        "$id": "https://Example.com/schema",
        "type": "object",
        "properties": {
            "api_key": {"$ref": "https://example.com/schema#/$defs/Widget"},
        },
        "$defs": {
            "Widget": {
                "type": "string",
                "default": "SECRET-M2",
                "enum": ["SECRET-M2-ENUM"],
            }
        },
    }
    distinct_paths = {
        "$id": "https://example.com/root",
        "type": "object",
        "properties": {
            "api_key": {"$ref": "https://EXAMPLE.com/A#/$defs/Widget"},
        },
        "$defs": {
            "Upper": {
                "$id": "https://example.com/A",
                "$defs": {"Widget": {"default": "UPPER"}},
            },
            "Lower": {
                "$id": "https://example.com/a",
                "$defs": {"Widget": {"default": "lower"}},
            },
        },
    }

    userinfo = "https://User:Pass@Example.com/schema"
    stored = traces._redact_secret_fields(schema)
    control = traces._redact_secret_fields(distinct_paths)

    assert (
        trace_redaction._canonical_resource_uri(userinfo) == "https://User:Pass@example.com/schema"
    )
    assert stored["$defs"]["Widget"] == {
        "type": "string",
        "default": "[redacted]",
        "enum": ["[redacted]"],
    }
    assert control["$defs"]["Upper"]["$defs"]["Widget"]["default"] == "[redacted]"
    assert control["$defs"]["Lower"]["$defs"]["Widget"]["default"] == "lower"


def test_overlapping_secret_values_are_redacted_longest_first() -> None:
    short = "sk-plane-abcdefghijklmnop"
    longer = "sk-plane-abcdefghijklmnop-PROVIDER-TAIL"
    text = f"authorization: Bearer {longer}"

    assert traces._redact_secret_string(text, (short, longer)) == "authorization: Bearer [redacted]"
    assert traces._redact_secret_string(text, (longer, short)) == "authorization: Bearer [redacted]"
    assert traces._redact_secret_string(text, (short, longer, longer)) == (
        "authorization: Bearer [redacted]"
    )


def test_embedded_schema_resource_id_refs_redact_the_local_target() -> None:
    assert traces._is_secret_key("Widget") is False
    schema = {
        "$id": "https://example/root",
        "properties": {"api_key": {"$ref": "https://example/cred"}},
        "$defs": {
            "Widget": {
                "$id": "https://example/cred",
                "default": "SECRET-EMBEDDED",
                "enum": ["SECRET-E2"],
            }
        },
    }
    control = {
        **schema,
        "properties": {"api_key": {"$ref": "#/$defs/Widget"}},
    }

    stored = traces._redact_secret_fields(schema)["$defs"]["Widget"]
    control_stored = traces._redact_secret_fields(control)["$defs"]["Widget"]

    nested = {
        "$id": "https://example/root",
        "$defs": {
            "Scope": {
                "$id": "https://example/resources/",
                "properties": {"api_key": {"$ref": "cred"}},
                "$defs": {
                    "Widget": {
                        "$id": "cred",
                        "default": "SECRET-RELATIVE",
                    }
                },
            }
        },
    }
    nested_stored = traces._redact_secret_fields(nested)["$defs"]["Scope"]["$defs"]["Widget"]

    assert stored["default"] == "[redacted]"
    assert stored["enum"] == ["[redacted]"]
    assert control_stored["default"] == "[redacted]"
    assert nested_stored["default"] == "[redacted]"


def test_document_id_schema_refs_redact_only_local_targets() -> None:
    assert traces._is_secret_key("Widget") is False
    expected = {
        "#/$defs/Widget": "[redacted]",
        "https://example/schema#/$defs/Widget": "[redacted]",
        "schema#/$defs/Widget": "[redacted]",
        "https://other/x#/$defs/Widget": "SECRET-DEFAULT",
    }

    for ref, expected_default in expected.items():
        schema = {
            "$id": "https://example/schema",
            "properties": {"api_key": {"$ref": ref}},
            "$defs": {"Widget": {"default": "SECRET-DEFAULT"}},
        }

        stored = traces._redact_secret_fields(schema)

        assert stored["$defs"]["Widget"]["default"] == expected_default


def test_percent_decoding_precedes_schema_pointer_unescaping() -> None:
    anchors: dict[str, frozenset[tuple[str, ...]]] = {}

    assert traces._local_schema_pointer("#/%24defs/Alpha", anchors) == frozenset(
        {("$defs", "Alpha")}
    )
    assert traces._local_schema_pointer("#/a%2Fb", anchors) == frozenset({("a", "b")})
    assert traces._local_schema_pointer("#/a~1b", anchors) == frozenset({("a/b",)})
    assert traces._local_schema_pointer("#/x/%zz", anchors) == frozenset({("x", "%zz")})
    assert traces._local_schema_pointer("http://e/x#Alpha", anchors) == frozenset()


def test_bare_root_schema_ref_redacts_only_through_the_ref_edge() -> None:
    literal = "ROOT-LITERAL"
    linked = {
        "enum": [literal],
        "properties": {"password": {"$ref": "#"}},
    }
    control = {
        "enum": [literal],
        "properties": {"password": {"$ref": "#/nonexistent"}},
    }

    assert traces._local_schema_pointer("#", {}) == frozenset({()})
    assert traces._redact_secret_fields(linked)["enum"] == ["[redacted]"]
    assert traces._redact_secret_fields(control)["enum"] == [literal]


def test_root_schema_anchor_ref_redacts_only_through_the_ref_edge() -> None:
    literal = "ROOT-ANCHOR-LITERAL"
    target = {"$anchor": "Alpha", "default": literal}
    linked = {
        **target,
        "properties": {"password": {"$ref": "#Alpha"}},
    }
    control = {
        **target,
        "properties": {"label": {"type": "string"}},
    }

    assert traces._redact_secret_fields(linked)["default"] == "[redacted]"
    assert traces._redact_secret_fields(control)["default"] == literal


@pytest.mark.parametrize("nesting", [0, 9], ids=["shallow", "deep"])
def test_secret_schema_anchor_ref_literals_are_redacted_transitively(
    trace_api, monkeypatch, nesting: int
) -> None:
    secret_property = {"properties": {"password": {"$ref": "#Credential"}}}
    for _ in range(nesting):
        secret_property = {"type": "object", "properties": {"level": secret_property}}
    schema = {
        **secret_property,
        "properties": {
            **secret_property["properties"],
            "access_token": {"$ref": "https://example.com/schema#Sibling"},
            "recovery_token": {"$ref": "#Dynamic"},
        },
        "$defs": {
            "Holder": {"$anchor": "Credential", "$ref": "#Payload"},
            "Target": {
                "$anchor": "Payload",
                "type": "string",
                "default": "SECRET-ANCHOR",
            },
            "Sibling": {
                "$anchor": "Sibling",
                "type": "string",
                "default": "ordinary-default",
            },
            "Remote": {
                "$anchor": "Remote",
                "$ref": "https://example.com/schema#Sibling",
            },
            "Dynamic": {
                "$dynamicAnchor": "Dynamic",
                "type": "string",
                "default": "dynamic-default",
            },
        },
    }
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={
            **_REQUEST,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        },
    )

    assert response.status_code == 200
    definitions = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]["response_format"][
        "json_schema"
    ]["schema"]["$defs"]
    assert definitions["Target"]["default"] == "[redacted]"
    assert definitions["Sibling"]["default"] == "ordinary-default"
    # `recovery_token` is a secret-named property, and in 2020-12 `$dynamicAnchor` also declares an
    # ordinary plain-name fragment, so its `$ref: "#Dynamic"` genuinely reaches this definition.
    assert definitions["Dynamic"]["default"] == "[redacted]"


def test_secret_schema_recursive_ref_redacts_recursive_anchor_root() -> None:
    schema = {
        "$recursiveAnchor": True,
        "type": "object",
        "default": "SECRET-ROOT",
        "properties": {"password": {"$recursiveRef": "#"}},
    }

    stored = traces._redact_secret_fields(schema)

    assert "$recursiveRef" in trace_redaction._JSON_SCHEMA_STRUCTURAL_KEYWORDS
    assert "$recursiveAnchor" in trace_redaction._JSON_SCHEMA_STRUCTURAL_KEYWORDS
    assert stored["default"] == "[redacted]"
    assert stored["properties"]["password"] == {"$recursiveRef": "#"}


def test_recursive_ref_without_local_anchor_falls_back_to_resource_root() -> None:
    schema = {
        "type": "object",
        "default": "ROOT-SECRET",
        "properties": {"password": {"$recursiveRef": "#"}},
        "$defs": {
            "Sibling": {
                "$id": "https://example.com/sibling",
                "$recursiveAnchor": True,
                "default": "SIBLING-PUBLIC",
            }
        },
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["default"] == "[redacted]"
    assert stored["$defs"]["Sibling"]["default"] == "SIBLING-PUBLIC"


def test_secret_schema_dynamic_ref_literals_are_redacted(trace_api, monkeypatch) -> None:
    schema = {
        "type": "object",
        "properties": {"password": {"$dynamicRef": "#Credential"}},
        "$defs": {
            "Dynamic": {
                "$dynamicAnchor": "Credential",
                "type": "string",
                "default": "SECRET-DYNAMIC",
            },
            "Static": {
                "$anchor": "Credential",
                "type": "string",
                "default": "SECRET-STATIC",
            },
            "Ordinary": {"default": "KEEP"},
        },
    }
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={
            **_REQUEST,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        },
    )

    assert response.status_code == 200
    definitions = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]["response_format"][
        "json_schema"
    ]["schema"]["$defs"]
    assert definitions["Dynamic"]["default"] == "[redacted]"
    assert definitions["Static"]["default"] == "[redacted]"
    assert definitions["Ordinary"]["default"] == "KEEP"


def test_percent_encoded_unreserved_authority_matches_schema_resource() -> None:
    """`%65xample.com` and `example.com` are the same host, so the `$ref` reaches the local target.

    Normalizing only the path left an authority escape looking like an external resource, and an
    unresolved reference means the target's literals are never reached and stay in the stored copy.
    """
    schema = {
        "$id": "https://example.com/a/schema",
        "type": "object",
        "properties": {"api_key": {"$ref": "https://%65xample.com/a/schema#/$defs/Cred"}},
        "$defs": {"Cred": {"default": "SECRET", "const": "SECRET-C", "enum": ["SECRET-E"]}},
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["$defs"]["Cred"]["default"] == "[redacted]"
    assert stored["$defs"]["Cred"]["const"] == "[redacted]"
    assert stored["$defs"]["Cred"]["enum"] == ["[redacted]"]
    # a reserved escape is NOT a decoded delimiter, so a genuinely different host stays distinct
    assert trace_redaction._canonical_resource_uri("https://exa%2Fmple.com/s") != (
        trace_redaction._canonical_resource_uri("https://exa/mple.com/s")
    )


def test_dynamic_ref_redacts_outer_dynamic_anchor_target() -> None:
    """`$dynamicRef` resolves against the dynamic scope, so it can leave its own resource.

    Which resource wins depends on the evaluation entry point, which a redactor recording a payload
    cannot know. Restricting the lookup to the reference's own resource left the outer target's
    literals exposed, so every same-named `$dynamicAnchor` counts as a possible target.
    """
    schema = {
        "$id": "https://example.com/root",
        "type": "object",
        "$defs": {
            "Outer": {"$dynamicAnchor": "T", "default": "SECRET-OUTER"},
            "Inner": {
                "$id": "https://example.com/inner",
                "type": "object",
                "properties": {"password": {"$dynamicRef": "#T"}},
                "$defs": {"Local": {"$dynamicAnchor": "T", "default": "SECRET-INNER"}},
            },
            "Ordinary": {"default": "KEEP"},
        },
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["$defs"]["Outer"]["default"] == "[redacted]"
    assert stored["$defs"]["Inner"]["$defs"]["Local"]["default"] == "[redacted]"
    assert stored["$defs"]["Ordinary"]["default"] == "KEEP"


def test_static_ref_does_not_cross_into_an_outer_resource() -> None:
    """Control for the dynamic widening above: a plain `$ref` stays inside its own resource."""
    schema = {
        "$id": "https://example.com/root",
        "$defs": {
            "Outer": {"$anchor": "S", "default": "OUTER-PUBLIC"},
            "Inner": {
                "$id": "https://example.com/inner",
                "properties": {"password": {"$ref": "#S"}},
                "$defs": {"Local": {"$anchor": "S", "default": "SECRET-INNER"}},
            },
        },
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["$defs"]["Inner"]["$defs"]["Local"]["default"] == "[redacted]"
    assert stored["$defs"]["Outer"]["default"] == "OUTER-PUBLIC"


def test_static_ref_resolves_a_dynamic_anchor(trace_api, monkeypatch) -> None:
    # `$dynamicAnchor` also declares an ordinary plain-name fragment, so a plain `$ref` reaches it.
    # resolving the two keywords against separate anchor maps left this pairing unredacted.
    schema = {
        "type": "object",
        "properties": {"password": {"$ref": "#Credential"}},
        "$defs": {
            "Dynamic": {"$dynamicAnchor": "Credential", "default": "SECRET-DYNAMIC"},
            "Ordinary": {"default": "KEEP"},
        },
    }
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={
            **_REQUEST,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        },
    )

    assert response.status_code == 200
    definitions = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]["response_format"][
        "json_schema"
    ]["schema"]["$defs"]
    assert definitions["Dynamic"]["default"] == "[redacted]"
    assert definitions["Ordinary"]["default"] == "KEEP"


@pytest.mark.parametrize(
    ("ref", "definitions", "path"),
    [
        (
            "#/$defs/Outer/$defs/Benign",
            {"Outer": {"$defs": {"Benign": {"type": "string", "default": "nested-secret"}}}},
            ("Outer", "$defs", "Benign"),
        ),
        (
            "#/$defs/we~1ird",
            {"we/ird": {"type": "string", "default": "escaped-secret"}},
            ("we/ird",),
        ),
    ],
    ids=["nested-pointer", "escaped-segment"],
)
def test_secret_schema_local_pointer_literals_are_redacted(
    trace_api, monkeypatch, ref: str, definitions: dict, path: tuple[str, ...]
) -> None:
    schema = {
        "type": "object",
        "properties": {"password": {"$ref": ref}},
        "$defs": definitions,
    }
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={
            **_REQUEST,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        },
    )

    assert response.status_code == 200
    stored = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]["response_format"][
        "json_schema"
    ]["schema"]["$defs"]
    for segment in path:
        stored = stored[segment]
    assert stored["default"] == "[redacted]"


@pytest.mark.parametrize(
    ("nesting", "transitive"),
    [
        (0, False),
        (8, False),
        (9, False),
        (10, False),
        (11, False),
        (12, False),
        (0, True),
        (9, True),
        (11, True),
    ],
    ids=lambda value: str(value).casefold(),
)
def test_deep_secret_schema_refs_redact_only_reachable_definitions(
    trace_api, monkeypatch, nesting: int, transitive: bool
) -> None:
    secret = "third-party-secret-abc123"
    property_ref = "#/$defs/Outer" if transitive else "#/$defs/Leaf"
    nested_schema = {"properties": {"password": {"$ref": property_ref}}}
    for _ in range(nesting):
        nested_schema = {"type": "object", "properties": {"lvl": nested_schema}}
    definitions = {
        "Leaf": {
            "type": "string",
            "default": secret,
            "const": secret,
            "enum": [secret],
            "examples": [secret],
            "example": secret,
        },
        "Sibling": {"type": "string", "default": "ordinary-default"},
    }
    if transitive:
        definitions["Outer"] = {"$ref": "#/$defs/Leaf"}
    schema = {**nested_schema, "$defs": definitions}
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={
            **_REQUEST,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        },
    )

    assert response.status_code == 200
    stored_definitions = _raw(trace_api)["records"][0]["spans"][0]["input_payload"][
        "response_format"
    ]["json_schema"]["schema"]["$defs"]
    leaf = stored_definitions["Leaf"]
    assert leaf["type"] == "string"
    assert leaf["default"] == "[redacted]"
    assert leaf["const"] == "[redacted]"
    assert leaf["enum"] == ["[redacted]"]
    assert leaf["examples"] == ["[redacted]"]
    assert leaf["example"] == "[redacted]"
    assert stored_definitions["Sibling"]["default"] == "ordinary-default"


def test_nested_secret_named_schema_ref_literals_are_redacted(trace_api, monkeypatch) -> None:
    secret = "third-party-secret-abc123"
    schema = {
        "type": "object",
        "properties": {
            "recovery_token": {"anyOf": [{"$ref": "#/definitions/Recovery"}]},
        },
        "definitions": {
            "Recovery": {"type": "string", "const": secret, "example": secret},
        },
    }
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={
            **_REQUEST,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        },
    )

    assert response.status_code == 200
    stored_schema = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]["response_format"][
        "json_schema"
    ]["schema"]
    assert stored_schema["properties"] == schema["properties"]
    assert stored_schema["definitions"]["Recovery"] == {
        "type": "string",
        "const": "[redacted]",
        "example": "[redacted]",
    }


def test_unreferenced_schema_definition_literals_are_not_redacted(trace_api, monkeypatch) -> None:
    schema = {
        "type": "object",
        "properties": {
            "recovery_token": {"anyOf": [{"$ref": "#/definitions/Recovery"}]},
        },
        "definitions": {
            "Recovery": {"type": "string", "default": "hunter2"},
            "Benign": {"type": "string", "default": "ordinary-default"},
        },
    }
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={
            **_REQUEST,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        },
    )

    assert response.status_code == 200
    definitions = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]["response_format"][
        "json_schema"
    ]["schema"]["definitions"]
    assert definitions["Recovery"]["default"] == "[redacted]"
    assert definitions["Benign"]["default"] == "ordinary-default"


@pytest.mark.parametrize("literal_keyword", ["default", "examples"])
def test_schema_refs_inside_instance_data_do_not_link_unrelated_definitions(
    trace_api, monkeypatch, literal_keyword: str
) -> None:
    instance_data = {"$ref": "#/$defs/Unrelated"}
    schema = {
        "type": "object",
        "properties": {"password": {"$ref": "#/$defs/Outer"}},
        "$defs": {
            "Outer": {
                "type": "object",
                literal_keyword: [instance_data]
                if literal_keyword == "examples"
                else instance_data,
            },
            "Unrelated": {
                "type": "string",
                "default": "ORDINARY-VALUE",
            },
        },
    }
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={
            **_REQUEST,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        },
    )

    assert response.status_code == 200
    definitions = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]["response_format"][
        "json_schema"
    ]["schema"]["$defs"]
    assert definitions["Unrelated"]["default"] == "ORDINARY-VALUE"


def test_cyclic_schema_refs_from_a_secret_property_terminate_and_redact(
    trace_api, monkeypatch
) -> None:
    schema = {
        "type": "object",
        "properties": {
            "recovery_token": {"anyOf": [{"$ref": "#/definitions/Recovery"}]},
        },
        "definitions": {
            "Recovery": {
                "type": "object",
                "default": "hunter2",
                "properties": {"next": {"$ref": "#/definitions/Recovery"}},
            }
        },
    }
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={
            **_REQUEST,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        },
    )

    assert response.status_code == 200
    stored_definition = _raw(trace_api)["records"][0]["spans"][0]["input_payload"][
        "response_format"
    ]["json_schema"]["schema"]["definitions"]["Recovery"]
    assert stored_definition["default"] == "[redacted]"
    assert stored_definition["properties"]["next"] == {"$ref": "#/definitions/Recovery"}


def _recorded_response_schema(trace_api, monkeypatch, schema: dict) -> dict:
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={
            **_REQUEST,
            "response_format": {"type": "json_schema", "json_schema": {"schema": schema}},
        },
    )

    assert response.status_code == 200
    return _raw(trace_api)["records"][0]["spans"][0]["input_payload"]["response_format"][
        "json_schema"
    ]["schema"]


def test_secret_schema_pointer_outside_definition_containers_is_redacted(
    trace_api, monkeypatch
) -> None:
    schema = {
        "properties": {"password": {"$ref": "#/components/Cred"}},
        "components": {"Cred": {"default": "S1"}},
    }

    stored = _recorded_response_schema(trace_api, monkeypatch, schema)

    assert stored["components"]["Cred"]["default"] == "[redacted]"


def test_nested_definition_name_collision_preserves_unreferenced_literal(
    trace_api, monkeypatch
) -> None:
    schema = {
        "type": "object",
        "properties": {
            "password": {"$ref": "#/$defs/Cred"},
            "profile": {
                "type": "object",
                "$defs": {"Cred": {"default": "UNRELATED"}},
            },
        },
        "$defs": {"Cred": {"default": "SROOT"}},
    }

    stored = _recorded_response_schema(trace_api, monkeypatch, schema)

    assert stored["$defs"]["Cred"]["default"] == "[redacted]"
    assert stored["properties"]["profile"]["$defs"]["Cred"]["default"] == "UNRELATED"


def test_embedded_resource_anchor_refs_redact_only_the_selected_resource() -> None:
    assert traces._is_secret_key("Widget") is False
    schema = {
        "$id": "https://example.com/root",
        "type": "object",
        "properties": {"api_key": {"$ref": "https://example.com/A#Widget"}},
        "$defs": {
            "A": {
                "$id": "https://example.com/A",
                "$defs": {"W": {"$anchor": "Widget", "type": "string", "default": "SECRET-IN-A"}},
            },
            "B": {
                "$id": "https://example.com/B",
                "$defs": {"W": {"$anchor": "Widget", "type": "string", "default": "PUBLIC-IN-B"}},
            },
        },
    }

    definitions = traces._redact_secret_fields(schema)["$defs"]

    assert definitions["A"]["$defs"]["W"]["default"] == "[redacted]"
    assert definitions["B"]["$defs"]["W"]["default"] == "PUBLIC-IN-B"


def test_resource_anchor_does_not_match_anchor_in_embedded_child_resource() -> None:
    schema = {
        "$id": "https://example.com/root",
        "properties": {"password": {"$ref": "#Cred"}},
        "$defs": {
            "Child": {
                "$id": "https://example.com/child",
                "$defs": {
                    "C": {"$anchor": "Cred", "default": "CHILD-PUBLIC"},
                },
            }
        },
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["$defs"]["Child"]["$defs"]["C"]["default"] == "CHILD-PUBLIC"


def test_document_local_anchor_ref_still_redacts() -> None:
    assert traces._is_secret_key("Widget") is False
    schema = {
        "type": "object",
        "properties": {"api_key": {"$ref": "#Widget"}},
        "$defs": {
            "W": {"$anchor": "Widget", "type": "string", "default": "SECRET-LOCAL"},
        },
    }

    stored = traces._redact_secret_fields(schema)

    assert stored["$defs"]["W"]["default"] == "[redacted]"


def test_duplicate_schema_anchor_declarations_are_all_redacted(trace_api, monkeypatch) -> None:
    schema = {
        "properties": {"password": {"$ref": "#Dup"}},
        "$defs": {
            "A": {"$anchor": "Dup", "default": "S-FIRST"},
            "B": {"$anchor": "Dup", "default": "S-SECOND"},
        },
    }

    stored = _recorded_response_schema(trace_api, monkeypatch, schema)

    assert stored["$defs"]["A"]["default"] == "[redacted]"
    assert stored["$defs"]["B"]["default"] == "[redacted]"


def test_schema_anchor_outside_definition_containers_is_redacted(trace_api, monkeypatch) -> None:
    schema = {
        "properties": {"password": {"$ref": "#External"}},
        "components": {
            "Cred": {"$anchor": "External", "default": "S-ANCHOR"},
        },
    }

    stored = _recorded_response_schema(trace_api, monkeypatch, schema)

    assert stored["components"]["Cred"]["default"] == "[redacted]"


def test_schema_reference_resolution_regressions(trace_api, monkeypatch) -> None:
    schema = {
        "properties": {
            "password": {"$ref": "#/$defs/C"},
            "recovery_token": {"$ref": "#/$defs/A"},
            "access_token": {"$ref": "https://x#Remote"},
            "credential": {"$ref": "#/$defs/LiteralHolder"},
        },
        "$defs": {
            "C": {"default": "DIRECT"},
            "A": {"$ref": "#/$defs/B"},
            "B": {"default": "TRANSITIVE"},
            "Sibling": {"default": "KEEP"},
            "Remote": {"default": "REMOTE"},
            "LiteralHolder": {
                "default": {"$ref": "#/$defs/DefaultTarget"},
                "examples": [{"$ref": "#/$defs/ExamplesTarget"}],
            },
            "DefaultTarget": {"default": "DEFAULT-KEEP"},
            "ExamplesTarget": {"default": "EXAMPLES-KEEP"},
        },
    }

    definitions = _recorded_response_schema(trace_api, monkeypatch, schema)["$defs"]

    assert definitions["C"]["default"] == "[redacted]"
    assert definitions["B"]["default"] == "[redacted]"
    assert definitions["Sibling"]["default"] == "KEEP"
    assert definitions["Remote"]["default"] == "REMOTE"
    assert definitions["DefaultTarget"]["default"] == "DEFAULT-KEEP"
    assert definitions["ExamplesTarget"]["default"] == "EXAMPLES-KEEP"


@pytest.mark.parametrize("applicator", ["allOf", "anyOf", "oneOf"])
def test_applicator_only_secret_schema_refs_are_redacted(
    trace_api, monkeypatch, applicator: str
) -> None:
    schema = {
        "type": "object",
        applicator: [{"properties": {"password": {"$ref": "#/$defs/Cred"}}}],
        "$defs": {
            "Cred": {"default": "S-APP"},
            "Other": {"default": "KEEP"},
        },
    }

    definitions = _recorded_response_schema(trace_api, monkeypatch, schema)["$defs"]

    assert definitions["Cred"]["default"] == "[redacted]"
    assert definitions["Other"]["default"] == "KEEP"


def test_secret_schema_pointer_into_top_level_array_is_redacted(trace_api, monkeypatch) -> None:
    schema = {
        "properties": {"password": {"$ref": "#/allOf/0"}},
        "allOf": [{"default": "S-ARR", "$ref": "#/$defs/Linked"}],
        "$defs": {"Linked": {"default": "S-LINKED"}},
    }

    stored = _recorded_response_schema(trace_api, monkeypatch, schema)

    assert stored["allOf"][0]["default"] == "[redacted]"
    assert stored["$defs"]["Linked"]["default"] == "[redacted]"


def test_secret_schema_pointer_with_deep_array_segment_is_redacted(trace_api, monkeypatch) -> None:
    schema = {
        "properties": {"password": {"$ref": "#/$defs/Wrapper/anyOf/0"}},
        "$defs": {
            "Wrapper": {"anyOf": [{"default": "S-DEEP", "$ref": "#/$defs/Linked"}]},
            "Linked": {"default": "S-DEEP-LINKED"},
        },
    }

    stored = _recorded_response_schema(trace_api, monkeypatch, schema)

    assert stored["$defs"]["Wrapper"]["anyOf"][0]["default"] == "[redacted]"
    assert stored["$defs"]["Linked"]["default"] == "[redacted]"


def test_secret_schema_pointer_into_definition_array_is_redacted(trace_api, monkeypatch) -> None:
    schema = {
        "properties": {"password": {"$ref": "#/$defs/List/0"}},
        "$defs": {
            "List": [{"default": "S-DL", "$ref": "#/$defs/Linked"}],
            "Linked": {"default": "S-DL-LINKED"},
        },
    }

    stored = _recorded_response_schema(trace_api, monkeypatch, schema)

    assert stored["$defs"]["List"][0]["default"] == "[redacted]"
    assert stored["$defs"]["Linked"]["default"] == "[redacted]"


def test_out_of_range_secret_schema_array_pointer_does_not_over_redact(
    trace_api, monkeypatch
) -> None:
    schema = {
        "properties": {"password": {"$ref": "#/allOf/9"}},
        "allOf": [{"default": "KEEP"}],
    }

    stored = _recorded_response_schema(trace_api, monkeypatch, schema)

    assert stored["allOf"][0]["default"] == "KEEP"


def test_secret_schema_anchor_inside_array_element_is_redacted(trace_api, monkeypatch) -> None:
    schema = {
        "properties": {"password": {"$ref": "#Credential"}},
        "allOf": [
            {"default": "KEEP"},
            {"$anchor": "Credential", "default": "S-ANCHOR"},
        ],
    }

    stored = _recorded_response_schema(trace_api, monkeypatch, schema)

    assert stored["allOf"][0]["default"] == "KEEP"
    assert stored["allOf"][1]["default"] == "[redacted]"


def test_non_schema_choice_list_content_survives_list_path_tracking() -> None:
    payload = {"choices": [{"message": {"role": "assistant", "content": "hello"}}]}

    assert traces._redact_secret_fields(payload, response_root=True) == payload


def test_non_schema_secret_fields_in_lists_remain_redacted() -> None:
    payload = {"a": [{"password": "pw"}]}

    assert traces._redact_secret_fields(payload) == {"a": [{"password": "[redacted]"}]}


@pytest.mark.parametrize(
    ("message_field", "action"),
    [
        (
            "tool_calls",
            [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "login",
                        "arguments": '{"password":"HUNTER2","user":"bob"}',
                    },
                }
            ],
        ),
        (
            "function_call",
            {"name": "login", "arguments": '{"api_key":"AKIA-SECRET","user":"bob"}'},
        ),
    ],
    ids=["tool-calls", "function-call"],
)
def test_json_encoded_function_arguments_are_redacted(message_field: str, action: object) -> None:
    payload = {"messages": [{"role": "assistant", message_field: action}]}

    stored = traces._redact_secret_fields(payload)
    stored_action = stored["messages"][0][message_field]
    function = stored_action[0]["function"] if message_field == "tool_calls" else stored_action
    arguments = json.loads(function["arguments"])

    assert arguments == {
        "password" if message_field == "tool_calls" else "api_key": "[redacted]",
        "user": "bob",
    }


def test_unparseable_function_arguments_are_redacted_as_a_string() -> None:
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "login",
                            "arguments": '{"password":"HUNTER2"',
                        }
                    }
                ],
            }
        ]
    }

    stored = traces._redact_secret_fields(payload)
    arguments = stored["messages"][0]["tool_calls"][0]["function"]["arguments"]

    assert arguments == "[redacted]"
    assert isinstance(arguments, str)

    valid = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "login",
                            "arguments": '{"password":"HUNTER2","user":"amy"}',
                        }
                    }
                ],
            }
        ]
    }
    valid_stored = traces._redact_secret_fields(valid)
    valid_arguments = valid_stored["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(valid_arguments, str)
    assert json.loads(valid_arguments) == {"password": "[redacted]", "user": "amy"}


def test_a_request_token_field_is_redacted(trace_api, monkeypatch) -> None:
    """A request-side `token` can hold an unrelated third-party credential. Exempting that key
    globally because response logprobs also call generated text `token` persists the credential and
    returns it through `format=raw`, even though it is not one of the proxy's known header secrets."""
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)
    third_party_secret = "third-party-secret-abc123"

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={**_REQUEST, "metadata": {"token": third_party_secret}},
    )

    assert response.status_code == 200
    assert _StaticAsyncClient.requests[-1]["json"]["metadata"]["token"] == third_party_secret
    stored = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]
    assert stored["metadata"]["token"] == "[redacted]"
    assert third_party_secret not in json.dumps(_raw(trace_api))


def test_logprob_token_text_survives_redaction(trace_api, monkeypatch) -> None:
    """A bare `token` in Chat Completions logprobs is generated text, not a credential. Redacting it
    destroys the token-to-logprob pairing that raw trace consumers need for scoring and analysis."""
    response_payload = {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "hello",
                            "logprob": -0.1,
                            "top_logprobs": [{"token": "hi", "logprob": -0.2}],
                        }
                    ]
                }
            }
        ]
    }
    _StaticAsyncClient.response = httpx.Response(200, json=response_payload)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 200
    stored = _raw(trace_api)["records"][0]["spans"][0]["output_payload"]
    assert stored["choices"][0]["logprobs"]["content"][0] == {
        "token": "hello",
        "logprob": -0.1,
        "top_logprobs": [{"token": "hi", "logprob": -0.2}],
    }


def test_streamed_logprobs_are_accumulated_without_redacting_token_text(
    trace_api, monkeypatch
) -> None:
    """Streaming puts logprobs beside `delta`, not inside it. Dropping that choice-level field makes
    streaming traces poorer than identical non-streaming traces, while redacting its `token` entries
    destroys the token-to-score pairing raw consumers need for analysis."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"hel"},"logprobs":{"content":[{"token":"hel","logprob":-0.1}]}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"lo"},"logprobs":{"content":[{"token":"lo","logprob":-0.2}],"refusal":[{"token":"no","logprob":-3.0}]},"finish_reason":"stop"}]}\n\n',
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    choice = _raw(trace_api)["records"][0]["spans"][0]["output_payload"]["choices"][0]
    assert choice["message"]["content"] == "hello"
    assert choice["logprobs"] == {
        "content": [
            {"token": "hel", "logprob": -0.1},
            {"token": "lo", "logprob": -0.2},
        ],
        "refusal": [{"token": "no", "logprob": -3.0}],
    }


def test_a_usage_only_stream_records_no_output(trace_api, monkeypatch) -> None:
    """Providers close some streams with a usage-only chunk and no choice. "An event arrived" is not
    "a reply arrived": treating the bookkeeping chunk as output stores a choiceless envelope, which
    `records` would then have to skip while `raw` shows a completion that never existed."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [b'data: {"choices":[],"usage":{"prompt_tokens":3}}\n\ndata: [DONE]\n\n']
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    assert _raw(trace_api)["records"][0]["spans"][0]["output_payload"] is None


def test_a_usage_only_stream_keeps_token_counters_without_output(trace_api, monkeypatch) -> None:
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            (
                b'data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":0}}\n\n'
                b"data: [DONE]\n\n"
            )
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["output_payload"] is None
    assert span["input_tokens"] == 11
    assert span["output_tokens"] == 0


def test_a_provider_specific_delta_field_is_accumulated(trace_api, monkeypatch) -> None:
    """Reasoning traces arrive as `delta.reasoning` fragments exactly like `content` does. Keeping
    only the fields this proxy happens to know about stores the last fragment of everything else --
    a reasoning field reading "ing" instead of the thought the caller paid for."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"reasoning":"think ","content":"h"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"reasoning":"more","content":"i"}}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    message = _raw(trace_api)["records"][0]["spans"][0]["output_payload"]["choices"][0]["message"]
    assert message["content"] == "hi"
    assert message["reasoning"] == "think more"


def test_a_list_delta_keeps_every_streamed_fragment(trace_api, monkeypatch) -> None:
    """Providers can send one `reasoning_details` list per chunk. Keeping only the first list makes
    the recorded reasoning shorter than the response the caller received, so raw traces silently lose
    paid output whenever a list-valued extension is streamed incrementally."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"reasoning_details":[{"a":1}]}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"reasoning_details":[{"b":2}]},"finish_reason":"stop"}]}\n\n',
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    message = _raw(trace_api)["records"][0]["spans"][0]["output_payload"]["choices"][0]["message"]
    assert message["reasoning_details"] == [{"a": 1}, {"b": 2}]


def test_many_stream_fragments_stay_unjoined_until_output() -> None:
    """Each streamed fragment runs on the async response iterator. Rebuilding the whole reply per
    chunk makes total work quadratic, while retaining one part per chunk keeps ingestion linear
    regardless of scheduler timing."""
    accumulator = trace_sse.SseAccumulator()
    fragment = "x" * 40

    for _ in range(32_000):
        accumulator.feed(
            b'data: {"choices":[{"index":0,"delta":{"content":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}}]}\n\n'
        )

    content = accumulator._choices[0]["message"]["content"]
    assert isinstance(content, trace_sse._StringFragments)
    assert content.parts == [fragment] * 32_000
    assert accumulator.output()["choices"][0]["message"]["content"] == fragment * 32_000


def test_a_mapping_delta_keeps_every_nested_fragment(trace_api, monkeypatch) -> None:
    """Providers split audio transcripts across mapping-valued deltas. Dropping every mapping after
    the first stores "hel" instead of "hello", so the trace no longer matches the reply delivered."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"audio":{"transcript":"hel"}}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"audio":{"transcript":"lo"}},"finish_reason":"stop"}]}\n\n',
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    message = _raw(trace_api)["records"][0]["spans"][0]["output_payload"]["choices"][0]["message"]
    assert message["audio"] == {"transcript": "hello"}


def test_an_oversized_collection_is_bounded_before_it_is_copied() -> None:
    """The payload width cap protects the trace writer only if work stops at the cap. Copying an
    entire multi-million-item request before slicing it lets one oversized call allocate and iterate
    in proportion to attacker-controlled input despite storing only the bounded prefix."""

    class _BoundedList(list):
        def __iter__(self):
            for index, item in enumerate(super().__iter__()):
                if index >= 3:
                    raise AssertionError("read beyond the collection bound")
                yield item

    class _BoundedDict(dict):
        def items(self):
            for index, item in enumerate(super().items()):
                if index >= 3:
                    raise AssertionError("read beyond the collection bound")
                yield item

    assert sanitize_json_value(_BoundedList(range(10)), max_collection=3) == [0, 1, 2]
    assert sanitize_json_value(
        _BoundedDict({str(index): index for index in range(10)}), max_collection=3
    ) == {"0": 0, "1": 1, "2": 2}


def test_a_long_conversation_is_stored_whole(trace_api) -> None:
    """A payload is a training row, not an attribute. Capping sequences at the attribute bound drops
    the NEWEST turns -- the reply and the turn that prompted it -- leaving a row that reads complete
    and trains toward an answer to a question no longer in it."""
    owner = db.ensure_standalone_owner()
    messages = [{"role": "user", "content": f"m{index}"} for index in range(200)]
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="long conversation",
        metadata=None,
        spans=[TraceSpan(input_payload={"messages": messages}, output_payload="reply")],
    )

    stored = _raw(trace_api)["records"][0]["spans"][0]["input_payload"]

    assert stored["messages"] == messages


def test_an_unrecorded_call_says_so_in_its_response(trace_api, monkeypatch) -> None:
    """The provider call succeeded and the caller was billed, so the request must not fail. But a
    silent miss is discovered at export, after the collection run is over and unrepeatable -- say it
    on the response the caller is already reading."""
    _StaticAsyncClient.requests = []
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    def _explode(**kwargs) -> str:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(traces, "store_trace", _explode)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 200
    assert response.json() == _RESPONSE
    assert response.headers["x-freesolo-record-failed"] == "true"
    assert _raw(trace_api)["traces"] == 0


@pytest.mark.anyio
async def test_trace_redaction_runs_in_the_worker_thread(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    owner = db.ensure_internal_key(_KEY)
    context = traces._UpstreamRequestContext(
        url="https://api.openai.com/v1/chat/completions",
        headers={},
        body=_REQUEST,
        provider="openai",
        model="gpt-test",
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        metadata=None,
        secrets=(),
        started_at=traces.time.perf_counter(),
        record_trace=True,
    )
    caller_thread = threading.get_ident()
    redaction_threads: list[int] = []
    sanitize_for_trace = traces._sanitize_for_trace

    def _capture_thread(value, *args, **kwargs):
        redaction_threads.append(threading.get_ident())
        return sanitize_for_trace(value, *args, **kwargs)

    monkeypatch.setattr(traces, "_sanitize_for_trace", _capture_thread)

    await traces._record_trace(context, output_payload=_RESPONSE, error=None)

    assert redaction_threads
    assert all(thread_id != caller_thread for thread_id in redaction_threads)


@pytest.mark.anyio
async def test_trace_duration_excludes_threadpool_queue_delay(monkeypatch) -> None:
    context = traces._UpstreamRequestContext(
        url="https://api.openai.com/v1/chat/completions",
        headers={},
        body=_REQUEST,
        provider="openai",
        model="gpt-test",
        key_id=1,
        project_id=_PROJECT_ID,
        metadata=None,
        secrets=(),
        started_at=100.0,
        record_trace=True,
    )
    now = 100.125
    captured: dict[str, object] = {}

    monkeypatch.setattr(traces.time, "perf_counter", lambda: now)
    monkeypatch.setattr(traces, "store_trace", lambda **kwargs: captured.update(kwargs))

    async def _delayed_dispatch(func, *args, **kwargs):
        nonlocal now
        now = 105.0
        return func(*args, **kwargs)

    monkeypatch.setattr(traces, "run_in_threadpool", _delayed_dispatch)

    await traces._record_trace(context, output_payload=_RESPONSE, error=None)

    span = captured["spans"][0]
    assert span.duration_ms == 125


def test_redaction_failure_does_not_break_the_relay(trace_api, monkeypatch) -> None:
    class _Untouched(dict):
        def items(self):
            raise AssertionError("redaction traversed beyond the payload depth bound")

    bounded = _Untouched({"password": "must-not-be-read"})
    nested = bounded
    for _ in range(platform_traces._MAX_PAYLOAD_DEPTH):
        nested = {"nested": nested}
    sanitized = traces._sanitize_for_trace(nested, ())
    for _ in range(platform_traces._MAX_PAYLOAD_DEPTH):
        sanitized = sanitized["nested"]
    assert sanitized == "[redacted]"

    _StaticAsyncClient.requests = []
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)
    sanitize_for_trace = traces._sanitize_for_trace

    def _explode_on_request(value, *args, **kwargs):
        if value is _StaticAsyncClient.requests[-1]["json"]:
            raise RecursionError("redaction depth exceeded")
        return sanitize_for_trace(value, *args, **kwargs)

    monkeypatch.setattr(traces, "_sanitize_for_trace", _explode_on_request)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 200
    assert response.json() == _RESPONSE
    assert response.headers["x-freesolo-record-failed"] == "true"
    assert _raw(trace_api)["traces"] == 0

    completion = (
        b'data: {"choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}]}\n\n'
    )
    terminator = b"data: [DONE]\n\n"
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody([completion, terminator])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)
    request_context = traces._request_context

    def _stream_request_context(**kwargs):
        context = request_context(**kwargs)
        _StaticAsyncClient.requests.append({"json": context.body})
        return context

    monkeypatch.setattr(traces, "_request_context", _stream_request_context)

    streamed = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert streamed.status_code == 200
    assert streamed.content == completion + b": freesolo-record-failed\n\n" + terminator
    assert _raw(trace_api)["traces"] == 0


def test_a_streamed_redirect_stays_raw_even_when_recording_fails(trace_api, monkeypatch) -> None:
    """A redirect is a non-success provider body, not an SSE completion. Treating only 4xx and 5xx as
    errors relabels its HTML as an event stream, drops the body from the trace, and can append an SSE
    comment that corrupts what the caller needs to inspect."""
    redirect_body = b'<html><a href="https://provider.example/login">continue</a></html>'

    class _RedirectStreamingClient(_StreamingAsyncClient):
        requests: ClassVar[list[dict]] = []
        body = _StreamingBody([redirect_body])

        async def send(self, request, *, stream) -> httpx.Response:
            assert stream is True
            return httpx.Response(
                307,
                headers={
                    "content-type": "text/html; charset=utf-8",
                    "location": "https://provider.example/login",
                    "set-cookie": "unsafe=1",
                },
                stream=type(self).body,
                request=request,
            )

    monkeypatch.setattr(traces.httpx, "AsyncClient", _RedirectStreamingClient)

    stored: list[dict] = []

    def _capture(**kwargs) -> str:
        stored.append(kwargs)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(traces, "store_trace", _capture)

    response = trace_api.post(
        "/v1/chat/completions",
        headers=_HEADERS,
        json={**_REQUEST, "stream": True},
        follow_redirects=False,
    )

    assert response.status_code == 307
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["location"] == "https://provider.example/login"
    assert "set-cookie" not in response.headers
    assert response.content == redirect_body
    assert b"freesolo-record-failed" not in response.content
    span = stored[0]["spans"][0]
    assert span.output_payload == redirect_body.decode()
    assert span.status_code == "ERROR"
    assert span.error == "upstream returned status 307"
    assert _raw(trace_api)["traces"] == 0


def test_a_failed_recording_does_not_corrupt_a_streamed_json_error(trace_api, monkeypatch) -> None:
    """A caller may request streaming and still receive a provider's ordinary JSON error body.
    Appending an SSE comment after persistence fails makes that 4xx body invalid JSON, hiding the
    provider's actual rejection behind a recording failure that should only be logged."""
    error_body = b'{"error":{"message":"rate limited"}}'

    class _JsonErrorStreamingClient(_StreamingAsyncClient):
        requests: ClassVar[list[dict]] = []
        body = _StreamingBody([error_body])

        async def send(self, request, *, stream) -> httpx.Response:
            assert stream is True
            return httpx.Response(
                429,
                headers={"content-type": "application/json"},
                stream=type(self).body,
                request=request,
            )

    monkeypatch.setattr(traces.httpx, "AsyncClient", _JsonErrorStreamingClient)

    def _explode(**kwargs) -> str:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(traces, "store_trace", _explode)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 429
    assert response.headers["content-type"].startswith("application/json")
    assert response.content == error_body
    assert response.json() == {"error": {"message": "rate limited"}}
    assert _raw(trace_api)["traces"] == 0


@pytest.mark.parametrize(
    "terminator",
    [b"data: [DONE]\n\n", b"event: end\ndata: [DONE]\n\n"],
    ids=["plain", "metadata-bearing"],
)
def test_a_streamed_recording_failure_preserves_the_done_event_structure(
    trace_api, monkeypatch, terminator: bytes
) -> None:
    completion = (
        b'data: {"choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}]}\n\n'
    )
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody([completion + terminator])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)
    monkeypatch.setattr(
        traces,
        "store_trace",
        lambda **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    if terminator.startswith(b"event:"):
        assert response.content == (
            completion + b"event: end\n" + b": freesolo-record-failed\n" + b"data: [DONE]\n\n"
        )
        assert b"event: end\n\n" not in response.content
    else:
        assert response.content == completion + b": freesolo-record-failed\n\n" + terminator


def test_a_streamed_recording_failure_preserves_a_split_event_prefix(
    trace_api, monkeypatch
) -> None:
    completion = (
        b'data: {"choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}]}\n\n'
    )
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody([completion, b"event: end", b"\ndata: [DONE]\n\n"])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)
    monkeypatch.setattr(
        traces,
        "store_trace",
        lambda **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    expected = completion + b"event: end\n: freesolo-record-failed\ndata: [DONE]\n\n"
    assert response.content == expected
    assert b"event: end\n\n" not in response.content


def test_a_streamed_recording_failure_arrives_before_the_split_terminator(
    trace_api, monkeypatch
) -> None:
    """OpenAI stream consumers stop reading at `[DONE]`, so a recording-failure signal after it is
    invisible and may never run once the client closes. Hold back only that terminator even when its
    line is split across chunks, then put the signal before the one byte-exact terminator."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    completion = (
        b'data: {"choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}]}\n\n'
    )
    terminator = b"data: [DONE]\n\n"
    _StreamingAsyncClient.body = _StreamingBody([completion + terminator[:9], terminator[9:]])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    def _explode(**kwargs) -> str:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(traces, "store_trace", _explode)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    assert response.content == completion + b": freesolo-record-failed\n\n" + terminator
    assert response.content.count(terminator) == 1
    assert "x-freesolo-record-failed" not in response.headers
    assert _raw(trace_api)["traces"] == 0


def test_projects_and_exports_are_scoped_to_authenticated_key(trace_api, monkeypatch) -> None:
    owner = db.ensure_external_key("owner-key")
    external = db.ensure_external_key("external-key")
    assert owner is not None
    assert external is not None
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="owner",
        metadata=None,
        spans=[TraceSpan(input_payload="owner", output_payload="reply")],
    )
    store_trace(
        key_id=external["id"],
        project_id=_OTHER_PROJECT_ID,
        trace_title="external",
        metadata=None,
        spans=[TraceSpan(input_payload="external", output_payload="reply")],
    )

    projects = list_projects(key_id=owner["id"])
    hidden = export_traces(
        key_id=owner["id"], project_id=_OTHER_PROJECT_ID, export_format="raw", limit=1000
    )

    assert projects == [{"id": _PROJECT_ID, "name": _PROJECT_ID}]
    assert hidden == {
        "records": [],
        "traces": 0,
        "skipped": 0,
        "format": "raw",
        "truncated": False,
    }


def test_standalone_transition_adopts_existing_traces(tmp_path, monkeypatch) -> None:
    """Switching a managed database to standalone deliberately preserves run ownership. Leaving trace
    rows on the old key makes every recorded project disappear from listing and export after cutover."""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "server.db"))
    external = db.ensure_external_key("managed-key")
    assert external is not None
    store_trace(
        key_id=external["id"],
        project_id=_PROJECT_ID,
        trace_title="before transition",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload={"messages": [{"role": "user", "content": "hello"}]},
                output_payload=_reply_envelope("world"),
            )
        ],
    )

    owner = db.ensure_standalone_owner()

    assert export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )["records"] == [{"input": "hello", "output": "world"}]


def test_existing_database_is_upgraded_with_trace_tables(tmp_path, monkeypatch) -> None:
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE api_keys ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, key_hash TEXT NOT NULL UNIQUE, "
        "key_prefix TEXT NOT NULL, email TEXT, created_at REAL NOT NULL, "
        "last_used_at REAL, disabled INTEGER NOT NULL DEFAULT 0)"
    )
    connection.commit()
    connection.close()
    monkeypatch.setattr(db, "DB_PATH", str(path))

    db.ensure_internal_key("legacy-owner")

    connection = sqlite3.connect(path)
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    span_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(llm_trace_spans)").fetchall()
    }
    connection.close()
    assert {"llm_traces", "llm_trace_spans"} <= tables
    assert "input_payload" in span_columns
    assert "output_payload" in span_columns
    assert "otel_span_id" not in span_columns


def test_a_successful_non_sse_stream_keeps_its_body_and_content_type(
    trace_api, monkeypatch
) -> None:
    """A 2xx body that is not an event stream is still not an event stream. Classifying by STATUS
    called a 200 gateway envelope a stream: the caller got `text/event-stream` for JSON their SSE
    reader discards as malformed, and the accumulator parsed it for deltas that never arrive, so
    the trace stored None where the response bytes belong."""
    envelope = b'{"error": {"message": "upstream gateway: quota exhausted"}}'

    class _JsonStreamingClient(_StreamingAsyncClient):
        requests: ClassVar[list[dict]] = []
        body = _StreamingBody([envelope])

        async def send(self, request, *, stream) -> httpx.Response:
            assert stream is True
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=type(self).body,
                request=request,
            )

    monkeypatch.setattr(traces.httpx, "AsyncClient", _JsonStreamingClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.content == envelope
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["output_payload"] == {"error": {"message": "upstream gateway: quota exhausted"}}


def test_a_streamed_non_sse_body_uses_its_declared_charset(trace_api, monkeypatch) -> None:
    body = "caf\N{LATIN SMALL LETTER E WITH ACUTE}".encode("iso-8859-1")

    class _Latin1StreamingClient(_StreamingAsyncClient):
        requests: ClassVar[list[dict]] = []

        async def send(self, request, *, stream) -> httpx.Response:
            assert stream is True
            return httpx.Response(
                500,
                headers={"content-type": "text/plain; charset=iso-8859-1"},
                stream=type(self).body,
                request=request,
            )

    _Latin1StreamingClient.body = _StreamingBody([body])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _Latin1StreamingClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 500
    assert response.content == body
    assert (
        _raw(trace_api)["records"][0]["spans"][0]["output_payload"]
        == "caf\N{LATIN SMALL LETTER E WITH ACUTE}"
    )

    invalid = httpx.Response(
        500,
        headers={"content-type": "text/plain; charset=not-a-real-charset"},
        content=b"caf\xe9",
    )
    assert traces._decode_response_bytes(invalid, invalid.content) == "caf\N{REPLACEMENT CHARACTER}"


def test_the_export_budget_counts_encoded_bytes_not_characters() -> None:
    """`ensure_ascii=False` keeps non-ASCII text as itself, so one character can be up to four
    UTF-8 bytes on the wire. Counting characters let an emoji-heavy export ship several times the
    nominal budget -- the exact exhaustion the budget exists to prevent."""
    record = {"input": "q", "output": "\U0001f600" * 1000}
    encoded = json.dumps(record, ensure_ascii=False)

    assert len(encoded.encode("utf-8")) > len(encoded)


def test_an_oversized_model_name_is_bounded_before_it_is_stored(trace_api) -> None:
    """`model` lands in two columns outside the payload bounds, and a span is stored even for a
    failed upstream call, so an unbounded value is database growth an authenticated caller controls
    directly without ever making a successful request."""
    owner = db.ensure_standalone_owner()
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="huge model",
        metadata=None,
        spans=[TraceSpan(model="m" * 50_000, input_payload={"prompt": "q"})],
    )

    span = _raw(trace_api)["records"][0]["spans"][0]
    assert len(span["model"]) <= 500
    assert _raw(trace_api)["records"][0]["model"] is not None
    assert len(_raw(trace_api)["records"][0]["model"]) <= 500


def test_a_multimodal_prompt_is_skipped_rather_than_stripped_to_its_text(trace_api) -> None:
    """`records` rows are text in and text out. Joining only the text parts of an image prompt
    pairs an answer that may depend entirely on the image with an input that no longer contains
    it -- a row whose target is unreachable from its prompt, and nothing downstream can detect it.
    Losing the example is recoverable; a silently corrupted one is not."""
    owner = db.ensure_standalone_owner()
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="vision",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "what is in this photo?"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://example.test/cat.png"},
                                },
                            ],
                        }
                    ]
                },
                output_payload=_reply_envelope("a cat"),
            )
        ],
    )

    records = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert records["records"] == []
    assert records["skipped"] == 1


def test_a_huge_integer_data_event_marks_the_stream_errored(trace_api, monkeypatch) -> None:
    huge_integer = b"9" * 4_301
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"hel"}}]}\n\n',
            b'data: {"number":' + huge_integer + b"}\n\n",
            b'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "stream contained an unparseable data event"


def test_a_leading_utf8_bom_is_stripped_only_at_stream_start(trace_api, monkeypatch) -> None:
    first = b'\xef\xbb\xbfdata: {"choices":[{"index":0,"delta":{"content":"hello "}}]}\n\n'
    second = (
        b'data: {"choices":[{"index":0,"delta":{"content":"world"},'
        b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    )
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody([first, second])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "OK"
    assert span["output_payload"]["choices"][0]["message"]["content"] == "hello world"

    later_bom = trace_sse.SseAccumulator()
    later_bom.feed(
        b'data: {"choices":[{"index":0,"delta":{"content":"kept"}}]}\n\n'
        b'\xef\xbb\xbfdata: {"choices":[{"index":0,"delta":{"content":"dropped"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    )
    assert later_bom.output()["choices"][0]["message"]["content"] == "kept"


def test_a_malformed_data_event_marks_the_stream_errored(trace_api, monkeypatch) -> None:
    """A 200 SSE stream can still be broken. An unparseable `data:` event between valid deltas
    drops a fragment out of the MIDDLE of the reply, and the stream can still deliver a finish
    reason and `[DONE]` afterwards -- so the span looked OK and `records` exported text with a
    hole in it as a complete training target."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"hel"}}]}\n\n',
            b"data: {not json at all\n\n",
            b'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert "unparseable" in span["error"]
    # and the holed text must not become a training row
    records = export_traces(
        key_id=db.ensure_standalone_owner()["id"],
        project_id=_PROJECT_ID,
        export_format="records",
        limit=1000,
    )
    assert records["records"] == []


@pytest.mark.parametrize("role", [[], {}, 1, True, ""])
def test_a_present_malformed_stream_role_marks_the_stream_errored(
    trace_api, monkeypatch, role
) -> None:
    event = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": role, "content": "hi"},
                    "finish_reason": "stop",
                }
            ]
        }
    ).encode()
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody([b"data: " + event + b"\n\n", b"data: [DONE]\n\n"])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "stream choice contained a non-string role"


@pytest.mark.parametrize("content", [123, {}, True], ids=["integer", "object", "boolean"])
def test_present_malformed_stream_content_marks_the_stream_errored(content) -> None:
    accumulator = trace_sse.SseAccumulator()
    malformed = json.dumps({"choices": [{"index": 0, "delta": {"content": content}}]}).encode()
    accumulator.feed(b"data: " + malformed + b"\n\n")
    accumulator.feed(
        b'data: {"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\n'
    )

    assert accumulator.defect == "stream choice contained malformed content"
    assert accumulator.output()["choices"][0]["message"]["content"] == "hello"


@pytest.mark.parametrize("content", [{}, {"content": None}])
def test_absent_or_null_stream_content_is_not_a_defect(content: dict) -> None:
    accumulator = trace_sse.SseAccumulator()
    delta = {**content, "role": "assistant"}
    event = json.dumps(
        {"choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}]}
    ).encode()

    accumulator.feed(b"data: " + event + b"\n\n")

    assert accumulator.defect is None


def test_stream_content_list_is_preserved() -> None:
    accumulator = trace_sse.SseAccumulator()
    content = [{"type": "text", "text": "hello"}]
    event = json.dumps(
        {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}]}
    ).encode()

    accumulator.feed(b"data: " + event + b"\n\n")

    assert accumulator.defect is None
    assert accumulator.output()["choices"][0]["message"]["content"] == content


@pytest.mark.parametrize(
    "payload",
    [
        {"marker": "\ud800", "choices": []},
        {"choices": [{"index": 0, "delta": {}, "marker": "\ud800"}]},
        {"choices": [{"index": 0, "delta": {"content": "\ud800"}}]},
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\ud800"}}]},
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "\ud800"}]},
    ],
    ids=["envelope", "choice-extension", "content", "tool-call", "finish-reason"],
)
def test_surrogates_mark_recording_defective_without_truncating(payload: dict) -> None:
    accumulator = trace_sse.SseAccumulator(max_accumulated_bytes=10_000)
    event = json.dumps(payload, ensure_ascii=True).encode()

    accumulator.feed(b"data: " + event + b"\n\n")

    assert accumulator.defect == "stream contained text that is not valid utf-8"
    assert accumulator.truncated is False
    assert "\ud800" not in repr(accumulator.output())


def test_ordinary_content_with_a_production_budget_is_not_defective() -> None:
    accumulator = trace_sse.SseAccumulator(
        max_accumulated_bytes=platform_traces.MAX_PAYLOAD_TOTAL_BYTES
    )

    accumulator.feed(b'data: {"choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n')

    assert accumulator.defect is None
    assert accumulator.truncated is False


def test_json_dump_replaces_lone_surrogates_and_preserves_unicode() -> None:
    ordinary = "héllo 日本語 🎉"
    payload = {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "\ud800"}},
        ],
        "ordinary": ordinary,
    }

    serialized = platform_traces._json_dump(payload)

    assert serialized is not None
    assert "\ud800" not in serialized
    assert "�" in serialized
    assert ordinary in serialized
    assert "\\u00e9" not in serialized
    assert "\\u65e5" not in serialized
    assert "\\ud83c" not in serialized
    restored = json.loads(serialized)
    assert restored["choices"][0]["message"]["content"] == "�"
    assert restored["ordinary"] == ordinary


def test_a_surrogate_response_is_stored_instead_of_dropped(trace_api, monkeypatch) -> None:
    body = b'{"choices":[{"index":0,"message":{"role":"assistant","content":"\\ud800"}}]}'
    _StaticAsyncClient.response = httpx.Response(
        200,
        content=body,
        headers={"content-type": "application/json"},
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 200
    assert response.content == body
    assert response.headers.get(traces._RECORD_FAILED_HEADER) is None
    raw = _raw(trace_api)
    assert raw["traces"] == 1
    assert (
        raw["records"][0]["spans"][0]["output_payload"]["choices"][0]["message"]["content"] == "�"
    )


def test_a_surrogate_in_a_stream_does_not_interrupt_the_relay(trace_api, monkeypatch) -> None:
    event = b'data: {"choices":[{"index":0,"delta":{"content":"\\ud800"}}]}\n\n'
    terminator = b"data: [DONE]\n\n"
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody([event, terminator])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    assert response.content == event + terminator
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["error"] == "stream contained text that is not valid utf-8"
    assert span["attributes"] is None


def test_a_non_object_delta_marks_the_stream_errored(trace_api, monkeypatch) -> None:
    """A present malformed delta can stand between valid fragments and silently remove paid output.
    A later finish and terminator must not relabel the surviving partial text as a complete target."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"hel"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":"missing fragment"}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert "non-object delta" in span["error"]
    records = export_traces(
        key_id=db.ensure_standalone_owner()["id"],
        project_id=_PROJECT_ID,
        export_format="records",
        limit=1000,
    )
    assert records["records"] == []


def test_a_non_object_choice_entry_marks_the_stream_errored_without_blocking_valid_choices(
    trace_api, monkeypatch
) -> None:
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[null,{"index":0,"delta":{"content":"right"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "stream choices contained a non-object entry"
    assert span["output_payload"]["choices"][0]["message"]["content"] == "right"


def test_a_present_malformed_choices_container_marks_the_stream_errored(
    trace_api, monkeypatch
) -> None:
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n',
            b'data: {"choices":{"missing":"fragment"}}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "stream contained non-list choices"
    records = export_traces(
        key_id=db.ensure_standalone_owner()["id"],
        project_id=_PROJECT_ID,
        export_format="records",
        limit=1000,
    )
    assert records["records"] == []


@pytest.mark.parametrize("bad_index", ["a", True])
def test_a_present_malformed_choice_index_marks_the_stream_errored_without_merging(
    trace_api, monkeypatch, bad_index
) -> None:
    first = json.dumps({"choices": [{"index": bad_index, "delta": {"content": "wrong"}}]}).encode()
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b"data: " + first + b"\n\n",
            b'data: {"choices":[{"index":0,"delta":{"content":"right"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "stream choice contained a non-integer index"
    assert span["output_payload"]["choices"][0]["message"]["content"] == "right"


@pytest.mark.parametrize("index_field", [{}, {"index": None}])
def test_absent_or_null_choice_index_uses_list_position(
    trace_api, monkeypatch, index_field
) -> None:
    choice = {**index_field, "delta": {"content": "world"}, "finish_reason": "stop"}
    event = b"data: " + json.dumps({"choices": [choice]}).encode() + b"\n\n"
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody([event, b"data: [DONE]\n\n"])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "OK"
    assert span["output_payload"]["choices"][0]["message"]["content"] == "world"


def test_null_logprobs_on_every_stream_chunk_is_not_a_defect(trace_api, monkeypatch) -> None:
    """OpenAI routinely sends `logprobs: null` when logprobs were not requested. Null means no scored
    fragment, not corruption, so an ordinary successful stream must remain exportable by `records`."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"wor"},"logprobs":null}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"ld"},"logprobs":null,"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "OK"
    assert span["error"] is None
    records = export_traces(
        key_id=db.ensure_standalone_owner()["id"],
        project_id=_PROJECT_ID,
        export_format="records",
        limit=1000,
    )
    assert records["records"] == [{"input": "hello", "output": "world"}]


def test_a_null_delta_on_a_terminal_chunk_is_not_a_defect(trace_api, monkeypatch) -> None:
    """Compatible providers may send `delta: null` on the terminal bookkeeping chunk. It carries no
    content to lose, so treating it as malformed incorrectly rejects an otherwise complete reply."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"world"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":null,"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "OK"
    assert span["error"] is None
    records = export_traces(
        key_id=db.ensure_standalone_owner()["id"],
        project_id=_PROJECT_ID,
        export_format="records",
        limit=1000,
    )
    assert records["records"] == [{"input": "hello", "output": "world"}]


def test_a_present_malformed_logprobs_still_marks_the_stream_errored(
    trace_api, monkeypatch
) -> None:
    """Ignoring null must not reopen the original hole. A non-null string cannot carry the expected
    scored fragments, so the surviving text is incomplete evidence and must not enter `records`."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"world"},"logprobs":"bad","finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "stream choice contained non-object logprobs"
    records = export_traces(
        key_id=db.ensure_standalone_owner()["id"],
        project_id=_PROJECT_ID,
        export_format="records",
        limit=1000,
    )
    assert records["records"] == []


def test_a_present_malformed_delta_list_still_marks_the_stream_errored(
    trace_api, monkeypatch
) -> None:
    """Null is empty bookkeeping, but a present list is the wrong representation for a delta and may
    hide paid content. It must retain the defect guard rather than being treated like absence."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"hel"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":["missing fragment"]}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "stream choice contained a non-object delta"
    records = export_traces(
        key_id=db.ensure_standalone_owner()["id"],
        project_id=_PROJECT_ID,
        export_format="records",
        limit=1000,
    )
    assert records["records"] == []


def test_a_wrong_typed_function_call_container_marks_the_stream_errored(
    trace_api, monkeypatch
) -> None:
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"partial","function_call":"missing action"},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "stream function_call was not an object"


@pytest.mark.parametrize("function_call", [{}, {"function_call": None}])
def test_absent_or_null_function_call_is_not_a_stream_defect(
    trace_api, monkeypatch, function_call
) -> None:
    delta = {"content": "world", **function_call}
    event = json.dumps(
        {"choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}]}
    ).encode()
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody([b"data: " + event + b"\n\n", b"data: [DONE]\n\n"])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "OK"
    assert span["error"] is None


def test_a_wrong_typed_tool_calls_container_marks_the_stream_errored(
    trace_api, monkeypatch
) -> None:
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"partial","tool_calls":{"id":"call-1"}},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "stream tool_calls was not a list"
    records = export_traces(
        key_id=db.ensure_standalone_owner()["id"],
        project_id=_PROJECT_ID,
        export_format="records",
        limit=1000,
    )
    assert records["records"] == []


@pytest.mark.parametrize("index", ["0", 0.5, True, [], {}])
def test_a_non_integer_tool_call_index_marks_the_stream_errored(
    trace_api, monkeypatch, index
) -> None:
    malformed = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "partial",
                        "tool_calls": [
                            {
                                "index": index,
                                "id": "call-1",
                                "function": {"name": "lookup"},
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ]
        }
    ).encode()
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [b"data: " + malformed + b"\n\n", b"data: [DONE]\n\n"]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "stream tool_call contained a non-integer index"
    assert span["output_payload"]["choices"][0]["message"].get("tool_calls") is None


@pytest.mark.parametrize("tool_call", [{"id": "call-1"}, {"index": None, "id": "call-1"}])
def test_absent_or_null_tool_call_index_uses_its_list_position(
    trace_api, monkeypatch, tool_call
) -> None:
    event = json.dumps(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [tool_call]},
                    "finish_reason": "tool_calls",
                }
            ]
        }
    ).encode()
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody([b"data: " + event + b"\n\n", b"data: [DONE]\n\n"])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "OK"
    assert span["error"] is None
    assert span["output_payload"]["choices"][0]["message"]["tool_calls"] == [{"id": "call-1"}]


def test_a_malformed_tool_call_entry_marks_the_stream_errored(trace_api, monkeypatch) -> None:
    """A `tool_calls` list advertises assistant actions. A scalar or null slot is therefore a lost
    invocation, not the same as an absent or null field, and the remaining text cannot be a complete
    training target even when the stream later finishes cleanly."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"partial","tool_calls":[null,"bad"]},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "stream tool_calls contained a non-object entry"
    records = export_traces(
        key_id=db.ensure_standalone_owner()["id"],
        project_id=_PROJECT_ID,
        export_format="records",
        limit=1000,
    )
    assert records["records"] == []


def test_an_error_only_sse_envelope_survives_in_raw_but_not_records(trace_api, monkeypatch) -> None:
    """An SSE error event may carry the provider's only useful diagnosis without any choices. Raw
    export must preserve that exact envelope, while the ERROR span remains excluded from training
    records just like every other failed upstream response."""
    error_event = b'data: {"error":{"message":"quota exceeded","code":"quota"}}\n\ndata: [DONE]\n\n'
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody([error_event])
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    assert response.content == error_event
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["output_payload"]["error"] == {
        "message": "quota exceeded",
        "code": "quota",
    }
    records = export_traces(
        key_id=db.ensure_standalone_owner()["id"],
        project_id=_PROJECT_ID,
        export_format="records",
        limit=1000,
    )
    assert records["records"] == []
    assert records["skipped"] == 1


def test_a_choice_without_delta_is_not_a_stream_defect(trace_api, monkeypatch) -> None:
    """Usage and finish bookkeeping may arrive without `delta`. Absence loses no fragment, so marking
    it defective would reject clean provider streams that separate content from termination metadata."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"world"}}]}\n\n',
            b'data: {"choices":[{"index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "OK"
    assert span["error"] is None
    records = export_traces(
        key_id=db.ensure_standalone_owner()["id"],
        project_id=_PROJECT_ID,
        export_format="records",
        limit=1000,
    )
    assert records["records"] == [{"input": "hello", "output": "world"}]


def test_a_mid_stream_error_envelope_marks_the_stream_errored(trace_api, monkeypatch) -> None:
    """Providers report a mid-stream failure in-band: a `data: {"error": ...}` event on a 200
    response, sometimes after real deltas already arrived. A trailing `[DONE]` then made the
    accumulator terminal, so the partial text was stored OK and exported as a finished reply."""
    _StreamingAsyncClient.requests = []
    _StreamingAsyncClient.status_code = 200
    _StreamingAsyncClient.body = _StreamingBody(
        [
            b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n',
            b'data: {"error":{"message":"provider overloaded"}}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StreamingAsyncClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert "error mid-stream" in span["error"]


def test_a_large_successful_non_sse_stream_body_is_kept_whole(trace_api, monkeypatch) -> None:
    """A successful body may exceed the per-string cap while remaining below the aggregate payload
    cap. Cutting it at 1 MB makes valid multi-field JSON undecodable before persistence can accept it.
    """
    reply = "y" * 600_000
    payload = json.dumps(
        {
            "choices": [{"message": {"role": "assistant", "content": reply}}],
            "provider_details": reply,
        }
    ).encode()
    assert len(payload) > platform_traces.MAX_PAYLOAD_VALUE_LENGTH
    assert len(payload) < platform_traces.MAX_PAYLOAD_TOTAL_BYTES

    class _BigJsonStreamingClient(_StreamingAsyncClient):
        requests: ClassVar[list[dict]] = []
        body = _StreamingBody([payload])

        async def send(self, request, *, stream) -> httpx.Response:
            assert stream is True
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=type(self).body,
                request=request,
            )

    monkeypatch.setattr(traces.httpx, "AsyncClient", _BigJsonStreamingClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 200
    span = _raw(trace_api)["records"][0]["spans"][0]
    # decoded as JSON, not stored as a truncated text prefix
    assert span["output_payload"]["choices"][0]["message"]["content"] == reply
    assert span["output_payload"]["provider_details"] == reply


def test_an_error_non_sse_stream_body_remains_capped_at_64_kib(trace_api, monkeypatch) -> None:
    """Provider failures are diagnostics, not training payloads. Keeping only 64 KiB prevents an
    attacker-sized error response from growing proxy memory even though successful bodies get 8 MiB.
    """
    payload = b"e" * (traces._MAX_RECORDED_ERROR_BYTES + 10_000)

    class _BigErrorStreamingClient(_StreamingAsyncClient):
        requests: ClassVar[list[dict]] = []
        body = _StreamingBody([payload])

        async def send(self, request, *, stream) -> httpx.Response:
            assert stream is True
            return httpx.Response(
                500,
                headers={"content-type": "text/plain"},
                stream=type(self).body,
                request=request,
            )

    monkeypatch.setattr(traces.httpx, "AsyncClient", _BigErrorStreamingClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 500
    assert response.content == payload
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["output_payload"] == "e" * traces._MAX_RECORDED_ERROR_BYTES
    assert span["status_code"] == "ERROR"
    assert span["attributes"] == {"payload_truncated": ["output"]}


def test_a_fitting_non_sse_stream_body_is_not_marked_truncated(trace_api, monkeypatch) -> None:
    payload = b"ordinary provider error"

    class _SmallErrorStreamingClient(_StreamingAsyncClient):
        requests: ClassVar[list[dict]] = []
        body = _StreamingBody([payload])

        async def send(self, request, *, stream) -> httpx.Response:
            assert stream is True
            return httpx.Response(
                500,
                headers={"content-type": "text/plain"},
                stream=type(self).body,
                request=request,
            )

    monkeypatch.setattr(traces.httpx, "AsyncClient", _SmallErrorStreamingClient)

    response = trace_api.post(
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 500
    span = _raw(trace_api)["records"][0]["spans"][0]
    assert span["output_payload"] == payload.decode()
    assert span["attributes"] is None


def test_an_export_that_ends_on_the_last_row_is_not_reported_truncated(
    trace_api, monkeypatch
) -> None:
    """Exhausting the byte budget ON the final row is not truncation -- everything was returned.
    Reporting it anyway sends the CLI and the environment scaffold to warn about older traces that
    do not exist."""
    monkeypatch.setattr(platform_traces, "MAX_EXPORT_BYTES", 200)
    owner = db.ensure_standalone_owner()
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="only one",
        metadata=None,
        spans=[
            TraceSpan(
                input_payload={"messages": [{"role": "user", "content": "q"}]},
                output_payload=_reply_envelope("y" * 400),
            )
        ],
    )

    export = export_traces(
        key_id=owner["id"], project_id=_PROJECT_ID, export_format="records", limit=1000
    )

    assert len(export["records"]) == 1
    assert export["truncated"] is False


def test_a_failed_recording_on_a_non_sse_stream_is_reported(trace_api, monkeypatch, caplog) -> None:
    """A non-SSE streamed body cannot carry the SSE failure comment (appending one to JSON is the
    corruption an earlier fix removed) and its headers left before persistence ran, so neither
    caller-facing signal is available. It must at least not fail silently: a silent drop lets a
    collection run finish believing every paid call was recorded."""
    envelope = b'{"choices":[{"message":{"role":"assistant","content":"hi"}}]}'

    class _JsonStreamClient(_StreamingAsyncClient):
        requests: ClassVar[list[dict]] = []
        body = _StreamingBody([envelope])

        async def send(self, request, *, stream) -> httpx.Response:
            assert stream is True
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=type(self).body,
                request=request,
            )

    monkeypatch.setattr(traces.httpx, "AsyncClient", _JsonStreamClient)
    monkeypatch.setattr(
        traces,
        "store_trace",
        lambda **kwargs: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")),
    )

    with caplog.at_level("WARNING"):
        response = trace_api.post(
            "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
        )

    assert response.status_code == 200
    # the caller still receives the provider's body unaltered -- no SSE comment spliced into JSON
    assert response.content == envelope
    assert any("trace not recorded" in record.message for record in caplog.records)
