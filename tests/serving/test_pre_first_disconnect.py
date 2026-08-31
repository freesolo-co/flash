from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import Request

from flash.serving.src.accounting.usage import (
    AuthorizedTraffic,
    build_usage_session,
    principal_for_external_org,
)
from flash.serving.src.accounting.usage_outbox import RequestIdentity
from flash.serving.src.http import inference_routes
from flash.serving.src.http.context import ServingContext
from flash.serving.src.http.inference_routes import (
    _discard_prepared_stream,
    _stream_chat_completion,
)
from flash.serving.src.http.routing import AdapterRouter
from flash.serving.src.io.schemas import AdapterRecord, GenerateRequest
from flash.serving.src.io.streaming import openai_chat_stream, prepare_stream
from tests.serving.conftest import RecordingUsageStore

QWEN = "Qwen/Qwen3.5-9B"


def _record() -> AdapterRecord:
    run_id = "run-a"
    checkpoint_id = f"{run_id}/final"
    return AdapterRecord.model_validate(
        {
            "adapter_id": checkpoint_id,
            "repo_id": "org/run-a",
            "org_id": "org-1",
            "base_model": QWEN,
            "checkpoint": checkpoint_id,
            "status": "ready",
            "thinking": False,
            "run_id": run_id,
            "checkpoint_step": None,
            "artifact_revision": hashlib.sha1(run_id.encode()).hexdigest(),
            "artifact_digest": hashlib.sha256(b"run-a-artifact").hexdigest(),
            "artifact_fingerprint": hashlib.sha256(b"run-a-binding").hexdigest(),
            "lora_rank": 16,
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


async def _wait_event_or_task(event: asyncio.Event, task: asyncio.Task[Any]) -> None:
    event_wait = asyncio.create_task(event.wait())
    done, _ = await asyncio.wait({event_wait, task}, return_when=asyncio.FIRST_COMPLETED)
    if task in done:
        event_wait.cancel()
        await asyncio.gather(event_wait, return_exceptions=True)
        task.result()
        raise AssertionError("route completed before the expected event")
    await event_wait


def _ready(record: AdapterRecord, generation_id: str) -> dict[str, Any]:
    return {
        "type": "ready",
        "checkpoint": record.checkpoint,
        "thinking": False,
        "prompt_token_ids": [1, 2],
        "completion_token_ids": [],
        "prompt_tokens": 2,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cached_tokens_reported": True,
        "reasoning_tokens": 0,
        "request_id": generation_id,
        "engine_replica_id": "replica-1",
        "lora_request_adapter": record.adapter_id,
    }


@pytest.mark.parametrize("route", ["generate", "generate_for_adapter", "chat_completions"])
def test_non_streaming_disconnect_cancels_generation(monkeypatch, route: str) -> None:
    async def scenario() -> tuple[bool, bool]:
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        record = _record()

        class Lookup:
            async def resolve(self, _adapter_id: str, *, org_id: str | None = None):
                del org_id
                return record, record

        class Context:
            lookup = Lookup()

            async def authorize_inference(self, *_args):
                return AuthorizedTraffic(principal=principal_for_external_org("org-1"))

            def reject_unsettleable_thinking(self, *_args) -> None:
                return None

            async def generate(self, *_args, **_kwargs):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

        async def receive():
            return await messages.get()

        monkeypatch.setattr(
            inference_routes.ServingContext,
            "of",
            staticmethod(lambda _request: Context()),
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
        await _wait_event_or_task(entered, task)
        await messages.put({"type": "http.disconnect"})
        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)
        return task.done(), cancelled.is_set()

    completed, generation_cancelled = asyncio.run(scenario())
    assert completed
    assert generation_cancelled


@pytest.mark.parametrize("route", ["generate", "generate_for_adapter", "chat_completions"])
def test_non_streaming_disconnect_after_engine_completion_finishes_finalization_once(
    monkeypatch, route: str
) -> None:
    async def scenario() -> tuple[Any, RecordingUsageStore]:
        finalization_entered = asyncio.Event()
        release_finalization = asyncio.Event()
        messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        record = _record()
        store = RecordingUsageStore()

        async def blocking_finalize(event) -> None:
            finalization_entered.set()
            await release_finalization.wait()
            store.finalized.append(event)

        store.finalize = blocking_finalize  # type: ignore[method-assign]

        class Lookup:
            async def resolve(self, _adapter_id: str, *, org_id: str | None = None):
                del org_id
                return record, record

        class Context(ServingContext):
            async def authorize_inference(self, *_args):
                return AuthorizedTraffic(principal=principal_for_external_org("org-1"))

        context = Context(
            object(),  # type: ignore[arg-type]
            AdapterRouter([record]),
            Lookup(),  # type: ignore[arg-type]
            store,
            internal_key=None,
            deployment_id="deployment-1",
            serving_release="release-1",
            reload_records=None,
            lookup_record=None,
            chat_authorizer=None,
        )

        async def completed_generation(*_args, generation_id: str, **_kwargs):
            return _ready(record, generation_id)

        async def receive():
            return await messages.get()

        monkeypatch.setattr("flash.serving.src.http.context.generate_once", completed_generation)
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
        await _wait_event_or_task(finalization_entered, task)
        await messages.put({"type": "http.disconnect"})
        await asyncio.sleep(0)
        assert not task.done()
        release_finalization.set()
        result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)
        return result[0], store

    result, store = asyncio.run(scenario())

    assert isinstance(result, asyncio.CancelledError)
    assert len(store.finalized) == 1
    assert store.failed == []


class _Context:
    def __init__(self, pool: Any, record: AdapterRecord, store: RecordingUsageStore) -> None:
        self.pool = pool
        self.router = AdapterRouter([record])
        self.store = store
        self.chat_stream_calls = 0

    async def prepare_stream(
        self,
        payload: GenerateRequest,
        requested: AdapterRecord,
        target: AdapterRecord,
        *,
        generation_id: str,
        expected_checkpoint: str | None,
    ):
        return await prepare_stream(
            self.pool,
            self.router,
            payload,
            requested,
            target,
            generation_id=generation_id,
            require_generation_id=True,
            expected_checkpoint=expected_checkpoint,
        )

    def usage_session(self, identity, traffic, requested, target, first, admitted_at):
        return build_usage_session(
            self.store,
            identity,
            traffic.principal,
            requested,
            target,
            first,
            deployment_id="deployment-1",
            serving_release="release-1",
            captured_at=admitted_at,
        )

    def chat_stream(self, **kwargs):
        self.chat_stream_calls += 1
        return openai_chat_stream(self.router, **kwargs)


def _identity() -> RequestIdentity:
    return RequestIdentity(
        request_id="fsgen-00000000000000000000000000000001",
        correlation_id="correlation-1",
    )


def _traffic() -> AuthorizedTraffic:
    return AuthorizedTraffic(principal=principal_for_external_org("org-1"))


def test_disconnect_before_first_event_closes_engine_without_starting_response_body() -> None:
    async def scenario() -> tuple[bool, bool, int]:
        entered = asyncio.Event()
        closed = asyncio.Event()
        messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        record = _record()
        store = RecordingUsageStore()

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

        context = _Context(Pool(), record, store)
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
                identity=_identity(),
                traffic=_traffic(),
                admitted_at=datetime.now(UTC),
            )
        )
        await _wait_event_or_task(entered, task)
        await messages.put({"type": "http.disconnect"})
        result = await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), timeout=1)
        return (
            isinstance(result[0], asyncio.CancelledError),
            closed.is_set(),
            context.chat_stream_calls,
        )

    disconnected, closed, chat_stream_calls = asyncio.run(scenario())
    assert disconnected
    assert closed
    assert chat_stream_calls == 0


def test_discard_prepared_stream_persists_ready_failure_and_closes_iterator() -> None:
    async def scenario() -> tuple[RecordingUsageStore, bool]:
        closed = asyncio.Event()
        record = _record()
        store = RecordingUsageStore()
        identity = _identity()
        first = _ready(record, identity.request_id)
        session = build_usage_session(
            store,
            identity,
            _traffic().principal,
            record,
            record,
            first,
            deployment_id="deployment-1",
            serving_release="release-1",
            captured_at=datetime.now(UTC),
        )

        async def events():
            try:
                yield first
                await asyncio.Event().wait()
            finally:
                closed.set()

        await _discard_prepared_stream(session, events())
        return store, closed.is_set()

    store, closed = asyncio.run(scenario())
    assert closed
    assert len(store.failed) == 1
    assert store.failed[0][1] == "client_disconnected"


@pytest.mark.parametrize("failure", ["generation_id", "attestation"])
def test_post_first_validation_failure_closes_engine_iterator(failure: str) -> None:
    async def scenario() -> bool:
        closed = asyncio.Event()
        record = _record()
        generation_id = _identity().request_id
        first = _ready(record, generation_id)
        if failure == "generation_id":
            first["request_id"] = "fsgen-00000000000000000000000000000002"
        else:
            first["lora_request_adapter"] = "wrong-adapter"

        async def events():
            try:
                yield first
            finally:
                closed.set()

        class Pool:
            def stream_generate(self, *_args, **_kwargs):
                return events()

        with pytest.raises((RuntimeError, Exception)):
            await prepare_stream(
                Pool(),
                AdapterRouter([record]),
                GenerateRequest(adapter_id=record.adapter_id, prompt="hi"),
                record,
                record,
                generation_id=generation_id,
                require_generation_id=True,
                expected_checkpoint=None,
            )
        return closed.is_set()

    assert asyncio.run(scenario())


def test_completed_preparation_wins_same_tick_disconnect() -> None:
    async def scenario() -> bool:
        record = _record()
        identity = _identity()
        store = RecordingUsageStore()
        context = _Context(object(), record, store)

        async def prepared_events():
            if False:
                yield {}

        async def prepared(*_args, **_kwargs):
            return prepared_events(), {}, False, _ready(record, identity.request_id)

        context.prepare_stream = prepared  # type: ignore[method-assign]
        request = _request(lambda: asyncio.sleep(0, {"type": "http.disconnect"}))
        response = await _stream_chat_completion(
            context,
            request,
            GenerateRequest(adapter_id=record.adapter_id, prompt="hi"),
            record,
            record,
            adapter_id=record.adapter_id,
            completion_id="chatcmpl-race",
            created=123,
            include_usage=True,
            identity=identity,
            traffic=_traffic(),
            admitted_at=datetime.now(UTC),
        )
        return response.status_code == 200

    assert asyncio.run(scenario())


def test_completed_generation_wins_a_same_tick_disconnect_race() -> None:
    async def scenario() -> Any:
        async def already_done() -> str:
            return "generated"

        request = _request(lambda: asyncio.sleep(0, {"type": "http.disconnect"}))
        return await inference_routes._await_until_disconnect(request, already_done())

    assert asyncio.run(scenario()) == "generated"
