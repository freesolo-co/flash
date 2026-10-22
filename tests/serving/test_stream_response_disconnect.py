"""real ASGI response cancellation must close the hosted generation and its receipt."""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from starlette.requests import ClientDisconnect

from flash.serving.src.accounting.usage_outbox import DurableUsageOutbox, UsageOutboxError
from flash.serving.src.http.routing import AdapterRouter
from flash.serving.src.http.stream_response import ClosingStreamingResponse
from flash.serving.src.io.openai_stream import openai_chat_stream
from tests.serving.test_usage_outbox import (
    _outbox_settings,
    _QueuedClient,
    _revision,
    _StreamSession,
)


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


@pytest.mark.parametrize("disconnect_at", ["headers", "immediate", "before_stream_start"])
@pytest.mark.parametrize("settlement_fails", [False, True])
def test_disconnect_before_stream_releases_prepared_generation(
    monkeypatch, disconnect_at, settlement_fails
):
    async def scenario():
        from datetime import UTC, datetime

        from flash.serving.src.http.inference_routes import _stream_chat_completion
        from flash.serving.src.io.schemas import GenerateRequest
        from tests.serving.test_pre_first_disconnect import (
            _Context,
            _identity,
            _ready,
            _record,
            _request,
            _traffic,
        )

        record = _record()
        client = _QueuedClient(
            [
                (200, [{"state": "in_progress", "lease_seconds": 120, "heartbeat_seconds": 20}]),
                (503, {}) if settlement_fails else (200, [{"state": "failed"}]),
            ]
        )
        store = DurableUsageOutbox(_outbox_settings(), client=client, worker_id="worker-1")
        closed = asyncio.Event()
        response_prepared = False

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
            if not response_prepared:
                await asyncio.Event().wait()
            return {"type": "http.disconnect"}

        async def send(_message):
            if disconnect_at == "headers":
                raise OSError("client disappeared before headers")
            await asyncio.Event().wait()

        request = _request(receive)
        request.scope["asgi"]["spec_version"] = "2.4" if disconnect_at == "headers" else "2.3"
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
        assert _identity().request_id in store._active_generations
        response_prepared = True
        expected_error = contextlib.nullcontext()
        if settlement_fails:
            expected_error = pytest.raises(UsageOutboxError, match="supabase_rpc_503")
        elif disconnect_at == "headers":
            expected_error = pytest.raises(ClientDisconnect)
        stream_started = False
        stream_response = response.stream_response

        async def deferred_stream_response(send):
            nonlocal stream_started
            # make the pending disconnect win before the streaming implementation runs.
            await asyncio.sleep(0)
            stream_started = True
            await stream_response(send)

        if disconnect_at == "before_stream_start":
            monkeypatch.setattr(response, "stream_response", deferred_stream_response)
        with expected_error:
            await asyncio.wait_for(response(request.scope, receive, send), timeout=1)
        if disconnect_at == "before_stream_start":
            assert not stream_started
        assert closed.is_set()
        assert len(client.calls) == 2
        assert client.calls[1][0].endswith("/rpc/fail_serving_generation")
        assert client.calls[1][1]["p_failure_code"] == "client_disconnected"
        assert _identity().request_id not in store._active_generations
        assert _identity().request_id not in store._generation_lease_deadlines

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
