"""OpenAI-shaped SSE rendering for streamed chat completions.

Split out of router.py's app builder. The engine pool, the adapter router and the usage reporter
are passed in rather than captured, so the stream can be rendered against a fake pool without
building the app.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Callable
from typing import Any

from flash.serving.src.engine_errors import raise_if_engine_error, terminating_on_engine_error
from flash.serving.src.responses import _ReasoningStreamSplitter, _usage_block
from flash.serving.src.routing import AdapterRouter, EnginePool
from flash.serving.src.schemas import AdapterRecord
from flash.serving.src.serving_io import (
    _active_checkpoint_ref,
    _provenance_headers,
    _revision_provenance,
    _sse,
    require_attested_revision,
)


async def _next_event_or_disconnect(
    events: AsyncIterator[dict[str, Any]], disconnect_wait: asyncio.Task[bool]
) -> dict[str, Any] | None:
    if disconnect_wait.done():
        return None
    next_event = asyncio.create_task(anext(events))
    done, _ = await asyncio.wait({next_event, disconnect_wait}, return_when=asyncio.FIRST_COMPLETED)
    if next_event in done:
        return next_event.result()
    next_event.cancel()
    await asyncio.gather(next_event, return_exceptions=True)
    return None


async def _close_async_iterator(events: AsyncIterator[dict[str, Any]]) -> None:
    close = getattr(events, "aclose", None)
    if close is not None:
        await close()


async def _replay_first_event(
    first: dict[str, Any], events: AsyncIterator[dict[str, Any]]
) -> AsyncIterator[dict[str, Any]]:
    try:
        yield first
        async for event in events:
            yield event
    finally:
        await _close_async_iterator(events)


async def _await_producer_shutdown(producer: asyncio.Task[None]) -> None:
    cancelled = False
    while not producer.done():
        try:
            await asyncio.shield(producer)
        except asyncio.CancelledError:
            cancelled = True
    producer.result()
    if cancelled:
        raise asyncio.CancelledError


def _assistant_role_chunk(completion_id: str, created: int, adapter_id: str) -> bytes:
    return _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": adapter_id,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )


async def _produce_openai_chat_stream(
    router: AdapterRouter,
    schedule_usage: Callable[[AdapterRecord, dict[str, Any], str | None], None],
    output: asyncio.Queue[tuple[bytes | None, Exception | None]],
    disconnected: asyncio.Event,
    *,
    record: AdapterRecord,
    events: AsyncIterator[dict[str, Any]],
    adapter_id: str,
    completion_id: str,
    created: int,
    include_usage: bool,
    caller_org: str | None,
    thinking: bool,
) -> None:
    async def emit(chunk: bytes | None = None, error: Exception | None = None) -> None:
        if not disconnected.is_set():
            await output.put((chunk, error))

    try:

        def _delta_chunk(delta: dict[str, Any]) -> bytes:
            return _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": adapter_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
            )

        def _error_chunk(message: str, code: int) -> bytes:
            return _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": adapter_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                    "error": {"message": message, "type": "engine_error", "code": code},
                }
            )

        splitter = _ReasoningStreamSplitter(thinking)
        final: dict[str, Any] | None = None
        latest_usage: dict[str, Any] = {}
        guarded_events = terminating_on_engine_error(router, events, adapter_id)
        disconnect_wait = asyncio.create_task(disconnected.wait())
        try:
            try:
                first = await _next_event_or_disconnect(guarded_events, disconnect_wait)
            except StopAsyncIteration:
                first = None
            if first is not None:
                if (
                    first.get("prompt_tokens") is not None
                    and first.get("completion_tokens") is not None
                ):
                    latest_usage = first
                guarded_events = _replay_first_event(first, guarded_events)
            await emit(_assistant_role_chunk(completion_id, created, adapter_id))
            while True:
                try:
                    event = await _next_event_or_disconnect(guarded_events, disconnect_wait)
                except StopAsyncIteration:
                    break
                if event is None:
                    break
                kind = event.get("type")
                if (
                    event.get("prompt_tokens") is not None
                    and event.get("completion_tokens") is not None
                ):
                    latest_usage = event
                if kind == "delta":
                    text = event.get("text") or ""
                    if not text:
                        continue
                    reasoning_delta, content_delta = splitter.feed(text)
                    if reasoning_delta:
                        await emit(_delta_chunk({"reasoning_content": reasoning_delta}))
                    if content_delta:
                        await emit(_delta_chunk({"content": content_delta}))
                elif kind == "final":
                    final = event
                elif kind == "error":
                    # the 200 and its headers went out with the first chunk, so the status can no
                    # longer carry the failure. without this the failure would propagate and starlette
                    # would drop the connection: the caller receives a well-formed but silently
                    # truncated stream with no error and no [done], indistinguishable from a short
                    # completion. emit the error into the stream and then close the protocol normally
                    # so the failure is detectable by an unmodified openai client.
                    await emit(_error_chunk(event["message"], event["code"]))
                    await emit(_sse("[DONE]"))
                    return
        finally:
            disconnect_wait.cancel()
            await asyncio.gather(disconnect_wait, return_exceptions=True)
            # one billing point prevents normal completion, an engine error, or a simultaneous
            # disconnect plus error from double billing. errors carry only message and code, while
            # cumulative events update latest_usage; the empty guard skips unobserved usage.
            if latest_usage:
                schedule_usage(record, latest_usage, caller_org)
            try:
                await _close_async_iterator(guarded_events)
            finally:
                await _close_async_iterator(events)

        if disconnected.is_set():
            return

        trailing = splitter.flush()
        if trailing:
            await emit(_delta_chunk({"reasoning_content": trailing}))

        if final is None or final.get("finish_reason") is None:
            await emit(_error_chunk("The serving engine ended without a terminal event.", 502))
            await emit(_sse("[DONE]"))
            return

        done_chunk: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": adapter_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": final.get("finish_reason"),
                }
            ],
        }
        if include_usage:
            prompt_tokens = final.get("prompt_tokens")
            completion_tokens = final.get("completion_tokens")
            if prompt_tokens is not None and completion_tokens is not None:
                done_chunk["usage"] = _usage_block(
                    int(prompt_tokens), int(completion_tokens), final.get("cached_tokens")
                )
        await emit(_sse(done_chunk))
        await emit(_sse("[DONE]"))
    except Exception as exc:
        await emit(error=exc)
    finally:
        await emit()


async def openai_chat_stream(
    router: AdapterRouter,
    schedule_usage: Callable[[AdapterRecord, dict[str, Any], str | None], None],
    *,
    record: AdapterRecord,
    events: AsyncIterator[dict[str, Any]],
    adapter_id: str,
    completion_id: str,
    created: int,
    include_usage: bool,
    caller_org: str | None,
    thinking: bool = False,
) -> AsyncIterator[bytes]:
    output: asyncio.Queue[tuple[bytes | None, Exception | None]] = asyncio.Queue(maxsize=1)
    disconnected = asyncio.Event()
    producer = asyncio.create_task(
        _produce_openai_chat_stream(
            router,
            schedule_usage,
            output,
            disconnected,
            record=record,
            events=events,
            adapter_id=adapter_id,
            completion_id=completion_id,
            created=created,
            include_usage=include_usage,
            caller_org=caller_org,
            thinking=thinking,
        )
    )
    try:
        while True:
            chunk, error = await output.get()
            if error is not None:
                raise error
            if chunk is None:
                return
            yield chunk
    finally:
        disconnected.set()
        with contextlib.suppress(asyncio.QueueEmpty):
            output.get_nowait()
        # keep cancellation out of the producer, but do not let it detach from the request before
        # it has closed the engine iterator and scheduled the last cumulative usage snapshot.
        await _await_producer_shutdown(producer)


async def prepare_stream(
    pool: EnginePool,
    router: AdapterRouter,
    payload: Any,
    requested: AdapterRecord,
    target: AdapterRecord,
    *,
    expected_checkpoint: str | None,
) -> tuple[AsyncIterator[dict[str, Any]], dict[str, str], bool]:
    engine_payload = payload.model_copy(update={"adapter_id": target.adapter_id})
    events: AsyncIterator[dict[str, Any]] | None = None
    try:
        # construction is inside the try with the first advance: ``EnginePool.stream_generate``
        # is declared as an ordinary method returning an AsyncIterator, so a conforming pool may
        # raise while building the iterator rather than on first advance. The current Modal pool
        # is an async generator (whose body is deferred to ``anext``), but the protocol does not
        # require that, and a dispatch failure must map identically either way.
        events = pool.stream_generate(
            target.base_model,
            engine_payload,
            target,
            expected_checkpoint=expected_checkpoint,
        )
        first = await anext(events)
        if target.is_revision:
            require_attested_revision(first, target)
    except BaseException as exc:
        # cancellation while waiting for the first engine event must still enter the pool iterator's
        # finally block, which aborts the remote generation. ordinary dispatch failures need the same
        # release before their existing http error mapping runs.
        if events is not None:
            with contextlib.suppress(Exception):
                await _close_async_iterator(events)
        if isinstance(exc, Exception):
            raise_if_engine_error(router, requested.adapter_id, exc)
        raise
    if first.get("type") == "ready":
        active_checkpoint = first.get("checkpoint")
        provenance = _revision_provenance(target, active_checkpoint)
        headers = _provenance_headers(provenance, active_checkpoint)
        if target.is_revision:
            headers["X-Freesolo-LoRA-Request-Adapter"] = target.adapter_id
        # replay ready internally so its first-output usage remains available if the client
        # disconnects before a text delta, while keeping it out of the client-facing sse protocol.
        return (
            _replay_first_event(first, events),
            headers,
            bool(first.get("thinking")),
        )

    active_checkpoint = _active_checkpoint_ref(target)
    provenance = _revision_provenance(target, active_checkpoint)
    # no ready event, so the rendered mode is unknown. report it as non-thinking rather than
    # guessing from ``target.thinking``: a base-model serve honors a caller enable_thinking
    # override, so the record can disagree with what was actually rendered, and splitting a
    # non-thinking completion that merely quotes </think> would tear the answer in half.
    return (
        _replay_first_event(first, events),
        _provenance_headers(provenance, active_checkpoint),
        False,
    )


async def generate_once(
    pool: EnginePool,
    router: AdapterRouter,
    schedule_usage: Callable[[AdapterRecord, dict[str, Any], str | None], None],
    payload: Any,
    requested: AdapterRecord,
    target: AdapterRecord,
    *,
    expected_checkpoint: str | None = None,
    caller_org: str | None = None,
) -> dict[str, Any]:
    """Dispatch one non-streaming generation and meter it.

    The result echoes the REQUESTED adapter id rather than the resolved target's, so an alias
    caller sees the id it asked for instead of the revision behind it.
    """
    engine_payload = payload.model_copy(update={"adapter_id": target.adapter_id})
    try:
        result = await pool.generate(
            target.base_model,
            engine_payload,
            target,
            expected_checkpoint=expected_checkpoint,
        )
    except Exception as exc:
        raise_if_engine_error(router, requested.adapter_id, exc)
    # attest before metering, not after: `schedule_usage` is what bills the caller, so a
    # generation the engine never attested to the resolved immutable adapter must not reach it.
    # this also covers every non-streaming route at once -- the plain `/generate` paths would
    # otherwise serve an unattested adapter with no check at all.
    require_attested_revision(result, target)
    if "adapter_id" in result:
        result = {**result, "adapter_id": requested.adapter_id}
    schedule_usage(requested, result, caller_org)
    return result
