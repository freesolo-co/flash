from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import pytest
from fastapi import Request

from flash.serving.src import inference_routes
from flash.serving.src.inference_routes import _discard_prepared_stream, _stream_chat_completion
from flash.serving.src.routing import AdapterRouter
from flash.serving.src.schemas import AdapterRecord, GenerateRequest
from flash.serving.src.serving_io import _sse
from flash.serving.src.streaming import openai_chat_stream, prepare_stream

QWEN = "Qwen/Qwen3.5-9B"


def _record() -> AdapterRecord:
    run_id = "run-a"
    revision = hashlib.sha1(run_id.encode()).hexdigest()
    return AdapterRecord.model_validate(
        {
            "adapter_id": f"{run_id}@final.{revision}",
            "repo_id": "org/run-a",
            "org_id": "org-1",
            "base_model": QWEN,
            "checkpoint": run_id,
            "status": "ready",
            "thinking": False,
            "metadata": {
                "record_type": "revision",
                "run_id": run_id,
                "checkpoint_step": None,
                "hf_revision": revision,
            },
        }
    )


def _request(receive) -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"spec_version": "2.3"},
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
        },
        receive,
    )


@pytest.mark.parametrize("route", ["generate", "generate_for_adapter", "chat_completions"])
def test_non_streaming_disconnect_cancels_generation(monkeypatch, route: str) -> None:
    async def scenario() -> tuple[bool, bool]:
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        record = _record()

        class Lookup:
            async def resolve(self, _adapter_id: str):
                return record, record

        class Context:
            lookup = Lookup()

            async def authorize_inference(self, *_args):
                return "org-1"

            async def generate(self, *_args, **_kwargs):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

        async def receive():
            return await messages.get()

        context = Context()
        monkeypatch.setattr(
            inference_routes.ServingContext,
            "of",
            staticmethod(lambda _request: context),
        )
        request = _request(receive)
        if route == "generate":
            awaitable = inference_routes.generate(
                GenerateRequest(adapter_id=record.adapter_id, prompt="hi"), request
            )
        elif route == "generate_for_adapter":
            awaitable = inference_routes.generate_for_adapter(
                record.adapter_id, {"prompt": "hi"}, request
            )
        else:
            awaitable = inference_routes.chat_completions(
                {
                    "model": record.adapter_id,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                request,
            )
        task = asyncio.create_task(awaitable)
        await entered.wait()
        await messages.put({"type": "http.disconnect"})
        done, _ = await asyncio.wait({task}, timeout=0.2)
        cancelled_before_cleanup = cancelled.is_set()
        if not done:
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return bool(done), cancelled_before_cleanup

    completed, generation_cancelled = asyncio.run(scenario())
    assert completed, "the handler kept generating after the peer disconnected"
    assert generation_cancelled, "the handler returned without cancelling generation"


class _Context:
    def __init__(self, pool: Any, record: AdapterRecord) -> None:
        self.pool = pool
        self.router = AdapterRouter([record])
        self.reports: list[dict[str, Any]] = []
        self.chat_stream_calls = 0

    async def prepare_stream(
        self,
        payload: GenerateRequest,
        requested: AdapterRecord,
        target: AdapterRecord,
        *,
        expected_checkpoint: str | None,
    ):
        return await prepare_stream(
            self.pool,
            self.router,
            payload,
            requested,
            target,
            expected_checkpoint=expected_checkpoint,
        )

    def schedule_usage(
        self, _record: AdapterRecord, usage: dict[str, Any], _caller_org: str | None
    ) -> None:
        self.reports.append(usage.copy())

    def chat_stream(self, **kwargs):
        self.chat_stream_calls += 1
        return openai_chat_stream(self.router, self.schedule_usage, **kwargs)


def test_disconnect_before_first_event_closes_engine_without_starting_response_body() -> None:
    async def scenario() -> tuple[bool, bool, int, list[dict[str, Any]]]:
        entered = asyncio.Event()
        closed = asyncio.Event()
        messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        record = _record()

        class Events:
            def __aiter__(self):
                return self

            async def __anext__(self):
                entered.set()
                await asyncio.Event().wait()
                raise StopAsyncIteration

            async def aclose(self) -> None:
                closed.set()

        class Pool:
            def stream_generate(self, *_args, **_kwargs):
                return Events()

        async def receive():
            return await messages.get()

        context = _Context(Pool(), record)
        task = asyncio.create_task(
            _stream_chat_completion(
                context,
                _request(receive),
                GenerateRequest(adapter_id=record.adapter_id, prompt="hi"),
                record,
                record,
                adapter_id=record.adapter_id,
                completion_id="chatcmpl-disconnect",
                created=123,
                include_usage=True,
                caller_org="org-1",
            )
        )
        await entered.wait()
        await messages.put({"type": "http.disconnect"})
        done, _ = await asyncio.wait({task}, timeout=0.2)
        if not done:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return False, closed.is_set(), context.chat_stream_calls, context.reports
        try:
            await task
        except asyncio.CancelledError:
            disconnected = True
        except Exception:
            disconnected = False
        else:
            disconnected = False
        return disconnected, closed.is_set(), context.chat_stream_calls, context.reports

    disconnected, closed, chat_stream_calls, reports = asyncio.run(scenario())
    assert disconnected, "the handler did not cancel when the peer disconnected"
    assert closed, "the cancelled preparation left the engine iterator suspended"
    assert chat_stream_calls == 0, "a response body was built after the peer disconnected"
    assert reports == [], "a request cancelled before any usage event must not be billed"


def test_disconnect_after_first_event_closes_engine_and_preserves_partial_usage() -> None:
    async def scenario() -> tuple[bool, list[dict[str, Any]]]:
        closed = asyncio.Event()
        record = _record()

        async def events():
            try:
                yield {
                    "type": "ready",
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "request_id": "req-raced-disconnect",
                }
                await asyncio.Event().wait()
            finally:
                closed.set()

        context = _Context(object(), record)
        prepared = events()
        await _discard_prepared_stream(context, record, "org-1", prepared)
        return closed.is_set(), context.reports

    closed, reports = asyncio.run(scenario())
    assert closed, "a prepared stream abandoned by a raced disconnect was not closed"
    assert reports == [
        {
            "type": "ready",
            "prompt_tokens": 4,
            "completion_tokens": 1,
            "request_id": "req-raced-disconnect",
        }
    ]


def test_first_event_failure_closes_engine_iterator() -> None:
    async def scenario() -> bool:
        closed = asyncio.Event()
        receive_block = asyncio.Event()
        record = _record()

        class Events:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise RuntimeError("first event failed")

            async def aclose(self) -> None:
                closed.set()

        class Pool:
            def stream_generate(self, *_args, **_kwargs):
                return Events()

        async def receive():
            await receive_block.wait()
            return {"type": "http.disconnect"}

        context = _Context(Pool(), record)
        with pytest.raises(RuntimeError, match="first event failed"):
            await _stream_chat_completion(
                context,
                _request(receive),
                GenerateRequest(adapter_id=record.adapter_id, prompt="hi"),
                record,
                record,
                adapter_id=record.adapter_id,
                completion_id="chatcmpl-failed",
                created=123,
                include_usage=True,
                caller_org="org-1",
            )
        return closed.is_set()

    assert asyncio.run(scenario()), "a failed first advance leaked the engine iterator"


def test_connected_preparation_preserves_headers_events_and_usage() -> None:
    async def scenario():
        closed = asyncio.Event()
        receive_block = asyncio.Event()
        record = _record()

        class Pool:
            async def stream_generate(self, *_args, **_kwargs):
                try:
                    yield {
                        "type": "ready",
                        "checkpoint": "run-a",
                        "thinking": False,
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "request_id": "req-connected",
                        "lora_request_adapter": record.adapter_id,
                    }
                    yield {
                        "type": "delta",
                        "text": "answer",
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "request_id": "req-connected",
                    }
                    yield {
                        "type": "final",
                        "finish_reason": "stop",
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "request_id": "req-connected",
                    }
                finally:
                    closed.set()

        async def receive():
            await receive_block.wait()
            return {"type": "http.disconnect"}

        context = _Context(Pool(), record)
        response = await _stream_chat_completion(
            context,
            _request(receive),
            GenerateRequest(adapter_id=record.adapter_id, prompt="hi"),
            record,
            record,
            adapter_id=record.adapter_id,
            completion_id="chatcmpl-connected",
            created=123,
            include_usage=True,
            caller_org="org-1",
        )
        chunks = [chunk async for chunk in response.body_iterator]
        return response, chunks, context, closed.is_set()

    response, chunks, context, closed = asyncio.run(scenario())
    revision = _record().metadata["hf_revision"]
    assert response.headers["x-freesolo-adapter-revision"] == _record().adapter_id
    assert response.headers["x-freesolo-checkpoint"] == "run-a"
    assert response.headers["x-freesolo-hf-revision"] == revision
    assert chunks == [
        _sse(
            {
                "id": "chatcmpl-connected",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": _record().adapter_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        ),
        _sse(
            {
                "id": "chatcmpl-connected",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": _record().adapter_id,
                "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}],
            }
        ),
        _sse(
            {
                "id": "chatcmpl-connected",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": _record().adapter_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        ),
        _sse("[DONE]"),
    ]
    assert context.reports == [
        {
            "type": "final",
            "finish_reason": "stop",
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "request_id": "req-connected",
        }
    ]
    assert context.chat_stream_calls == 1
    assert closed


def test_completed_generation_wins_a_same_tick_disconnect_race() -> None:
    """a generation that finished must not be discarded because the peer left in the same tick.

    `asyncio.wait(FIRST_COMPLETED)` returns a SET: when generation and disconnect both resolve
    before the loop wakes, BOTH are in `done`. deciding on the disconnect first throws away a
    result whose `schedule_usage` has already run, so the caller is billed for a response nobody
    returns. the packaged helper in flash/serve/app/http.py resolves the tie toward the operation,
    and this pins the hosted copy to the same rule.
    """

    async def scenario() -> Any:
        async def already_done() -> str:
            return "generated"

        # both futures are resolved before the await, which forces the two-member `done` set.
        request = _request(lambda: asyncio.sleep(0, {"type": "http.disconnect"}))
        return await inference_routes._await_until_disconnect(request, already_done())

    assert asyncio.run(scenario()) == "generated"
