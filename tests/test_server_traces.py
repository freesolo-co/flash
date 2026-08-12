from __future__ import annotations

import asyncio
import json
import sqlite3
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
from flash.server.routes import traces

_PROJECT_ID = "11111111-1111-4111-8111-111111111111"
_OTHER_PROJECT_ID = "22222222-2222-4222-8222-222222222222"
_KEY = "operator-secret"
_PROVIDER_KEY = "provider-secret"
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

    async def post(self, url, *, headers, json) -> httpx.Response:
        type(self).requests.append({"url": url, "headers": headers, "json": json})
        return type(self).response

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
    response = httpx.Response(200, stream=body, request=request)
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
    response = httpx.Response(200, stream=body, request=httpx.Request("POST", context.url))
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
        headers={"content-type": "text/html", "location": "https://provider.example/login"},
    )
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 307
    assert response.content == redirect_body
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


def test_upstream_transport_failure_returns_502_and_records(trace_api, monkeypatch) -> None:
    class _FailingClient(_StaticAsyncClient):
        async def post(self, url, *, headers, json):
            raise httpx.ConnectError("offline", request=httpx.Request("POST", url))

    monkeypatch.setattr(traces.httpx, "AsyncClient", _FailingClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=_REQUEST)

    assert response.status_code == 502
    raw = _raw(trace_api)
    span = raw["records"][0]["spans"][0]
    assert span["status_code"] == "ERROR"
    assert span["error"] == "upstream request failed"
    assert span["output_payload"] is None


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
        ({"model": "  ", "messages": []}, "model is required"),
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


def test_chat_envelopes_export_as_trainable_text(trace_api) -> None:
    """The scaffold trains on `record.input` as prompt text. Exporting the whole request and choices
    envelopes JSON-stringifies protocol metadata into both halves instead of the user's turn and reply."""
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

    assert records["records"] == [{"input": "latest question", "output": "final answer"}]
    assert prompts["records"] == [{"input": "latest question"}]
    span = raw["records"][0]["spans"][0]
    assert span["input_payload"] == request
    assert span["output_payload"] == response


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
    accumulator = traces._SseAccumulator()
    fragment = "x" * 40

    for _ in range(32_000):
        accumulator.feed(
            b'data: {"choices":[{"index":0,"delta":{"content":"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}}]}\n\n'
        )

    content = accumulator._choices[0]["message"]["content"]
    assert isinstance(content, traces._StringFragments)
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
        "/v1/chat/completions", headers=_HEADERS, json={**_REQUEST, "stream": True}
    )

    assert response.status_code == 307
    assert response.headers["content-type"].startswith("text/html")
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
