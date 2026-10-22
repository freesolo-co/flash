"""real ASGI response cancellation must close the hosted generation and its receipt."""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from starlette.requests import ClientDisconnect

from flash.serving.src.http.routing import AdapterRouter
from flash.serving.src.http.stream_response import ClosingStreamingResponse
from flash.serving.src.io.openai_stream import openai_chat_stream
from tests.serving.test_usage_outbox import _revision, _StreamSession


@pytest.mark.parametrize("disconnect_at", ["next_event", "send", "send_error"])
def test_asgi_disconnect_settles_usage_and_closes_engine(disconnect_at):
    async def scenario():
        record = _revision()
        session = _StreamSession()
        disconnected = asyncio.Event()
        closed = asyncio.Event()

        async def events():
            try:
                yield {
                    "type": "delta",
                    "text": "hello",
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "prompt_token_ids": [1, 2],
                    "completion_token_ids": [3],
                    "reasoning_tokens": 0,
                    "thinking": False,
                }
                await asyncio.Event().wait()
            finally:
                closed.set()

        stream = openai_chat_stream(
            AdapterRouter([record]),
            record=record,
            events=events(),
            model_id=record.model_id,
            completion_id="chatcmpl-disconnect",
            created=123,
            include_usage=True,
            usage_session=session,
        )
        response = ClosingStreamingResponse(stream)

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if b"hello" in message.get("body", b""):
                disconnected.set()
                if disconnect_at == "send":
                    await asyncio.Event().wait()
                if disconnect_at == "send_error":
                    raise OSError("client disconnected")

        scope = {
            "type": "http",
            "asgi": {"spec_version": "2.4" if disconnect_at == "send_error" else "2.3"},
        }
        with contextlib.suppress(ClientDisconnect):
            await asyncio.wait_for(response(scope, receive, send), timeout=1)
        # assert before asyncio.run tears down outstanding tasks/generators.
        assert closed.is_set(), "response returned with the engine stream still open"
        assert len(session.failed) == 1, "disconnect left the usage receipt active"
        assert session.failed[0][1] == "client_disconnected"
        assert session.failed[0][0]["completion_tokens"] == 1

    asyncio.run(scenario())


def test_disconnect_before_response_headers_releases_prepared_generation():
    async def scenario():
        from datetime import UTC, datetime

        from flash.serving.src.http.inference_routes import _stream_chat_completion
        from flash.serving.src.io.schemas import GenerateRequest
        from tests.serving.conftest import RecordingUsageStore
        from tests.serving.test_pre_first_disconnect import (
            _Context,
            _identity,
            _ready,
            _record,
            _request,
            _traffic,
        )

        record = _record()
        store = RecordingUsageStore()
        closed = asyncio.Event()

        async def events():
            try:
                yield _ready(record, _identity().request_id)
                await asyncio.Event().wait()
            finally:
                closed.set()

        class Pool:
            def stream_generate(self, *_args, **_kwargs):
                return events()

        async def receive():
            await asyncio.Event().wait()

        async def send(_message):
            raise OSError("client disappeared before headers")

        request = _request(receive)
        request.scope["asgi"]["spec_version"] = "2.4"
        response = await _stream_chat_completion(
            _Context(Pool(), record, store),
            request,
            GenerateRequest(model_id=record.model_id, prompt="hi"),
            record,
            record,
            model_id=record.model_id,
            completion_id="chatcmpl-before-headers",
            created=123,
            include_usage=True,
            identity=_identity(),
            traffic=_traffic(),
            admitted_at=datetime.now(UTC),
        )
        with pytest.raises(ClientDisconnect):
            await response(request.scope, receive, send)
        assert closed.is_set()
        assert len(store.failed) == 1
        assert store.failed[0][1] == "client_disconnected"

    asyncio.run(scenario())


def test_disconnect_relinquishes_receipt_when_settlement_times_out(monkeypatch):
    from flash.serving.src.accounting.usage_outbox import UsageOutboxError
    from flash.serving.src.io import openai_stream

    monkeypatch.setattr(openai_stream, "_CLEANUP_TIMEOUT_SECONDS", 0.02)

    async def scenario():
        closed = asyncio.Event()
        record = _revision()

        class StalledSession(_StreamSession):
            async def fail(self, result, code):
                await asyncio.Event().wait()

        session = StalledSession()

        async def events():
            try:
                yield {
                    "type": "delta",
                    "text": "partial",
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                }
                await asyncio.Event().wait()
            finally:
                closed.set()

        stream = openai_chat_stream(
            AdapterRouter([record]),
            record=record,
            events=events(),
            model_id=record.model_id,
            completion_id="chatcmpl-stalled",
            created=123,
            include_usage=True,
            usage_session=session,
        )
        await anext(stream)
        await anext(stream)
        with pytest.raises(UsageOutboxError, match="stream_producer_shutdown_timed_out"):
            await asyncio.wait_for(stream.aclose(), timeout=1)
        assert closed.is_set()
        assert session.relinquished == 1
        assert session.finalized == []

    asyncio.run(scenario())
