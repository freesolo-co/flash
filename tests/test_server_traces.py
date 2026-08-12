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
from flash.server.platform.traces import TraceSpan, export_traces, store_trace
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
    assert sent["json"]["credential"] == "[redacted]"
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
    assert span["output_payload"] == _RESPONSE
    assert _PROVIDER_KEY not in json.dumps(raw)


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
    _StaticAsyncClient.response = httpx.Response(200, json=_RESPONSE)
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)
    headers = {key: value for key, value in _HEADERS.items() if key != "X-Freesolo-Project-Id"}
    headers["X-Freesolo-Record"] = "false"

    response = trace_api.post("/v1/chat/completions", headers=headers, json=_REQUEST)

    assert response.status_code == 200
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
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=_HEADERS, json=body)

    assert response.status_code == 400
    assert response.json()["detail"] == expected_detail
    assert _raw(trace_api)["traces"] == 0


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
    monkeypatch.setattr(traces.httpx, "AsyncClient", _StaticAsyncClient)

    response = trace_api.post("/v1/chat/completions", headers=headers, json=_REQUEST)

    assert response.status_code == expected_status
    assert trace_api.get(
        "/api/traces/projects", headers={"Authorization": f"Bearer {_KEY}"}
    ).json() == {"projects": []}


def test_export_formats_convert_and_count_skips(trace_api) -> None:
    owner = db.ensure_standalone_owner()
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="complete",
        metadata={"source": "test"},
        spans=[TraceSpan(input_payload={"prompt": "hello"}, output_payload="world")],
    )
    store_trace(
        key_id=owner["id"],
        project_id=_PROJECT_ID,
        trace_title="prompt-only",
        metadata=None,
        spans=[TraceSpan(input_payload="sample me")],
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
        "records": [{"input": {"prompt": "hello"}, "output": "world"}],
        "traces": 3,
        "skipped": 2,
        "format": "records",
    }
    assert prompts == {
        "records": [{"input": "sample me"}, {"input": {"prompt": "hello"}}],
        "traces": 3,
        "skipped": 1,
        "format": "prompts",
    }
    assert raw["traces"] == 3
    assert raw["skipped"] == 0
    assert raw["format"] == "raw"
    assert len(raw["records"]) == 3


def test_projects_and_exports_are_scoped_to_authenticated_key(trace_api, monkeypatch) -> None:
    owner = db.ensure_standalone_owner()
    external = db.ensure_external_key("external-key")
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

    projects = trace_api.get(
        "/api/traces/projects", headers={"Authorization": f"Bearer {_KEY}"}
    ).json()
    hidden = trace_api.get(
        "/api/traces/export",
        headers={"Authorization": f"Bearer {_KEY}"},
        params={"project_id": _OTHER_PROJECT_ID, "format": "raw"},
    ).json()

    assert projects == {"projects": [{"id": _PROJECT_ID, "name": _PROJECT_ID}]}
    assert hidden == {"records": [], "traces": 0, "skipped": 0, "format": "raw"}


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
