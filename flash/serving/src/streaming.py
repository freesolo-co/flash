"""OpenAI-shaped SSE rendering for streamed chat completions.

Split out of router.py's app builder. The engine pool, adapter router, and durable usage session
are passed in rather than captured, so the stream can be rendered against fakes without building
the app.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any, cast

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
from flash.serving.src.usage import UsageSession
from flash.serving.src.usage_outbox import UsageOutboxError

_CLEANUP_TIMEOUT_SECONDS = 10.0


async def _next_event_or_disconnect(
    events: AsyncIterator[dict[str, Any]], disconnect_wait: asyncio.Task[bool]
) -> dict[str, Any] | None:
    next_event: asyncio.Future[dict[str, Any] | bool] = asyncio.ensure_future(anext(events))
    waiters: set[asyncio.Future[Any]] = {next_event, disconnect_wait}
    await asyncio.sleep(0)
    if next_event.done():
        return cast("dict[str, Any]", next_event.result())
    done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
    if next_event in done:
        return cast("dict[str, Any]", next_event.result())
    next_event.cancel()
    await asyncio.gather(next_event, return_exceptions=True)
    return None


async def _close_async_iterator(events: AsyncIterator[dict[str, Any]]) -> None:
    close = getattr(events, "aclose", None)
    if close is not None:
        await asyncio.wait_for(close(), timeout=_CLEANUP_TIMEOUT_SECONDS)


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
    try:
        await asyncio.wait_for(producer, timeout=_CLEANUP_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        raise UsageOutboxError("stream_producer_shutdown_timed_out") from exc


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


class _StreamOutput:
    def __init__(
        self,
        output: asyncio.Queue[tuple[bytes | None, Exception | None]],
        disconnected: asyncio.Event,
        *,
        completion_id: str,
        created: int,
        adapter_id: str,
    ) -> None:
        self._output = output
        self._disconnected = disconnected
        self._completion_id = completion_id
        self._created = created
        self._adapter_id = adapter_id
        self._terminal_sent = False

    async def emit(
        self,
        chunk: bytes | None = None,
        error: Exception | None = None,
        *,
        ignore_disconnect: bool = False,
    ) -> None:
        if ignore_disconnect or not self._disconnected.is_set():
            await self._output.put((chunk, error))

    def delta(self, delta: dict[str, Any]) -> bytes:
        return _sse(
            {
                "id": self._completion_id,
                "object": "chat.completion.chunk",
                "created": self._created,
                "model": self._adapter_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )

    def error(self, message: str, code: int, error_type: str = "engine_error") -> bytes:
        return _sse(
            {
                "id": self._completion_id,
                "object": "chat.completion.chunk",
                "created": self._created,
                "model": self._adapter_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                "error": {"message": message, "type": error_type, "code": code},
            }
        )

    async def terminal(self, chunk: bytes, *, ignore_disconnect: bool = False) -> None:
        if self._terminal_sent:
            return
        self._terminal_sent = True
        if ignore_disconnect:
            with contextlib.suppress(asyncio.QueueEmpty):
                while True:
                    self._output.get_nowait()
        await self.emit(chunk, ignore_disconnect=ignore_disconnect)
        await self.emit(_sse("[DONE]"), ignore_disconnect=ignore_disconnect)

    async def finish(self) -> None:
        if self._disconnected.is_set():
            if not self._terminal_sent:
                with contextlib.suppress(asyncio.QueueEmpty):
                    while True:
                        self._output.get_nowait()
            elif self._output.full():
                self._output.get_nowait()
            self._output.put_nowait((None, None))
            return
        await self._output.put((None, None))


async def _close_stream_sources(
    disconnect_wait: asyncio.Task[bool],
    guarded_events: AsyncIterator[dict[str, Any]],
    events: AsyncIterator[dict[str, Any]],
    stream_output: _StreamOutput,
) -> None:
    disconnect_wait.cancel()
    await asyncio.gather(disconnect_wait, return_exceptions=True)
    close_error: BaseException | None = None
    for source in (guarded_events, events):
        try:
            await _close_async_iterator(source)
        except BaseException as exc:
            close_error = close_error or exc
    await stream_output.finish()
    if close_error is not None:
        raise close_error


async def _produce_openai_chat_stream(
    router: AdapterRouter,
    output: asyncio.Queue[tuple[bytes | None, Exception | None]],
    disconnected: asyncio.Event,
    *,
    record: AdapterRecord,
    events: AsyncIterator[dict[str, Any]],
    adapter_id: str,
    completion_id: str,
    created: int,
    include_usage: bool,
    usage_session: UsageSession,
    thinking: bool,
) -> None:
    del record
    stream_output = _StreamOutput(
        output,
        disconnected,
        completion_id=completion_id,
        created=created,
        adapter_id=adapter_id,
    )

    splitter = _ReasoningStreamSplitter(thinking)
    final: dict[str, Any] | None = None
    latest_usage: dict[str, Any] = {}
    terminal_persisted = False
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
        await stream_output.emit(_assistant_role_chunk(completion_id, created, adapter_id))
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
                    await stream_output.emit(
                        stream_output.delta({"reasoning_content": reasoning_delta})
                    )
                if content_delta:
                    await stream_output.emit(stream_output.delta({"content": content_delta}))
            elif kind == "final":
                final = event
            elif kind == "error":
                if latest_usage:
                    await usage_session.fail(latest_usage, "engine_failed")
                    terminal_persisted = True
                await stream_output.terminal(stream_output.error(event["message"], event["code"]))
                return

        if disconnected.is_set() and final is None:
            if latest_usage:
                await usage_session.fail(latest_usage, "client_disconnected")
                terminal_persisted = True
            return
        trailing = splitter.flush()
        if trailing:
            await stream_output.emit(stream_output.delta({"reasoning_content": trailing}))
        if final is None or final.get("finish_reason") is None:
            if latest_usage:
                await usage_session.fail(latest_usage, "engine_terminal_missing")
                terminal_persisted = True
            await stream_output.terminal(
                stream_output.error("The serving engine ended without a terminal event.", 502)
            )
            return

        try:
            await usage_session.finalize(final)
        except UsageOutboxError:
            with contextlib.suppress(UsageOutboxError):
                await usage_session.capture(final)
            usage_session.relinquish()
            await stream_output.terminal(
                stream_output.error(
                    "Durable serving accounting finalization failed.",
                    503,
                    "accounting_error",
                )
            )
            return
        terminal_persisted = True
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
        await stream_output.terminal(
            _sse(done_chunk),
            ignore_disconnect=disconnected.is_set(),
        )
    except Exception as exc:
        if latest_usage and not terminal_persisted:
            with contextlib.suppress(UsageOutboxError):
                await usage_session.fail(latest_usage, "stream_failed")
        if isinstance(exc, UsageOutboxError):
            if disconnected.is_set():
                raise
            await stream_output.terminal(
                stream_output.error(
                    "Durable serving accounting persistence failed.",
                    503,
                    "accounting_error",
                )
            )
        else:
            await stream_output.emit(error=exc)
    finally:
        await _close_stream_sources(disconnect_wait, guarded_events, events, stream_output)


async def openai_chat_stream(
    router: AdapterRouter,
    *,
    record: AdapterRecord,
    events: AsyncIterator[dict[str, Any]],
    adapter_id: str,
    completion_id: str,
    created: int,
    include_usage: bool,
    usage_session: UsageSession,
    thinking: bool = False,
) -> AsyncIterator[bytes]:
    output: asyncio.Queue[tuple[bytes | None, Exception | None]] = asyncio.Queue(maxsize=3)
    disconnected = asyncio.Event()
    producer = asyncio.create_task(
        _produce_openai_chat_stream(
            router,
            output,
            disconnected,
            record=record,
            events=events,
            adapter_id=adapter_id,
            completion_id=completion_id,
            created=created,
            include_usage=include_usage,
            usage_session=usage_session,
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
        await _await_producer_shutdown(producer)


async def prepare_stream(
    pool: EnginePool,
    router: AdapterRouter,
    payload: Any,
    requested: AdapterRecord,
    target: AdapterRecord,
    *,
    generation_id: str,
    require_generation_id: bool,
    expected_checkpoint: str | None,
) -> tuple[AsyncIterator[dict[str, Any]], dict[str, str], bool, dict[str, Any]]:
    engine_payload = payload.model_copy(
        update={"adapter_id": target.adapter_id, "generation_id": generation_id}
    )
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
    try:
        if require_generation_id and first.get("request_id") != generation_id:
            raise RuntimeError("serving engine returned a mismatched generation id")
        if first.get("type") == "ready":
            active_checkpoint = first.get("checkpoint")
            provenance = _revision_provenance(target, active_checkpoint)
            headers = _provenance_headers(provenance, active_checkpoint)
            if target.is_revision:
                headers["X-Freesolo-LoRA-Request-Adapter"] = first["lora_request_adapter"]
            return (
                _replay_first_event(first, events),
                headers,
                bool(first.get("thinking")),
                first,
            )

        active_checkpoint = _active_checkpoint_ref(target)
        provenance = _revision_provenance(target, active_checkpoint)
        headers = _provenance_headers(provenance, active_checkpoint)
        if target.is_revision:
            headers["X-Freesolo-LoRA-Request-Adapter"] = first["lora_request_adapter"]
        return (
            _replay_first_event(first, events),
            headers,
            False,
            first,
        )
    except BaseException:
        await _close_async_iterator(events)
        raise


async def generate_once(
    pool: EnginePool,
    router: AdapterRouter,
    payload: Any,
    requested: AdapterRecord,
    target: AdapterRecord,
    *,
    generation_id: str,
    require_generation_id: bool,
    expected_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Dispatch one non-streaming generation and meter it.

    The result echoes the REQUESTED adapter id rather than the resolved target's, so an alias
    caller sees the id it asked for instead of the revision behind it.
    """
    engine_payload = payload.model_copy(
        update={"adapter_id": target.adapter_id, "generation_id": generation_id}
    )
    try:
        result = await pool.generate(
            target.base_model,
            engine_payload,
            target,
            expected_checkpoint=expected_checkpoint,
        )
    except Exception as exc:
        raise_if_engine_error(router, requested.adapter_id, exc)
    require_attested_revision(result, target)
    if require_generation_id and result.get("request_id") != generation_id:
        raise RuntimeError("serving engine returned a mismatched generation id")
    if "adapter_id" in result:
        result = {**result, "adapter_id": requested.adapter_id}
    return result
