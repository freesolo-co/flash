"""choice-aware hosted OpenAI stream producer with durable settlement."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any, cast

import orjson

from flash.serve.app.openai import ReasoningDeltaSplitter
from flash.serving.src.accounting.usage import UsageSession
from flash.serving.src.accounting.usage_outbox import UsageOutboxError
from flash.serving.src.engine.errors import terminating_on_engine_error
from flash.serving.src.http.routing import AdapterRouter
from flash.serving.src.io.responses import _usage_block
from flash.serving.src.io.schemas import AdapterRecord

_CLEANUP_TIMEOUT_SECONDS = 10.0


def _sse(data: dict[str, Any] | str) -> bytes:
    encoded = data.encode("utf-8") if isinstance(data, str) else orjson.dumps(data)
    return b"data: " + encoded + b"\n\n"


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


def _assistant_role_chunk(
    completion_id: str, created: int, adapter_id: str, choice_count: int = 1
) -> bytes:
    return _sse(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": adapter_id,
            "choices": [
                {
                    "index": index,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
                for index in range(choice_count)
            ],
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

    def delta(
        self,
        delta: dict[str, Any],
        *,
        index: int = 0,
        logprobs: list[dict[str, Any]] | None = None,
    ) -> bytes:
        choice: dict[str, Any] = {
            "index": index,
            "delta": delta,
            "finish_reason": None,
        }
        if logprobs is not None:
            choice["logprobs"] = {"content": logprobs}
        return _sse(
            {
                "id": self._completion_id,
                "object": "chat.completion.chunk",
                "created": self._created,
                "model": self._adapter_id,
                "choices": [choice],
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
    choice_count: int = 1,
) -> None:
    del record
    stream_output = _StreamOutput(
        output,
        disconnected,
        completion_id=completion_id,
        created=created,
        adapter_id=adapter_id,
    )
    splitters = {index: ReasoningDeltaSplitter(thinking=thinking) for index in range(choice_count)}
    terminals: dict[int, dict[str, Any]] = {}
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
            if _has_usage(first):
                latest_usage = first
            guarded_events = _replay_first_event(first, guarded_events)
        await stream_output.emit(
            _assistant_role_chunk(completion_id, created, adapter_id, choice_count)
        )
        while final is None:
            try:
                event = await _next_event_or_disconnect(guarded_events, disconnect_wait)
            except StopAsyncIteration:
                break
            if event is None:
                break
            if _has_usage(event):
                latest_usage = event
            kind = event.get("type")
            if kind == "delta":
                await _emit_delta(stream_output, splitters, terminals, event)
            elif kind == "choice_finished":
                index = _event_index(event, choice_count)
                if index in terminals:
                    raise RuntimeError("duplicate choice terminal")
                terminals[index] = event
            elif kind == "final":
                if final is not None:
                    raise RuntimeError("duplicate request terminal")
                if choice_count == 1 and not terminals and event.get("finish_reason") is not None:
                    terminals[0] = {
                        "index": 0,
                        "finish_reason": event["finish_reason"],
                    }
                final = event
            elif kind == "error":
                if latest_usage:
                    await usage_session.fail(latest_usage, "engine_failed")
                    terminal_persisted = True
                await stream_output.terminal(stream_output.error(event["message"], event["code"]))
                return
            elif kind not in {"ready", "usage_progress"}:
                raise RuntimeError("invalid engine stream event")
        if disconnected.is_set() and final is None:
            if latest_usage:
                await usage_session.fail(latest_usage, "client_disconnected")
                terminal_persisted = True
            return
        if final is None or set(terminals) != set(range(choice_count)):
            if latest_usage:
                await usage_session.fail(latest_usage, "engine_terminal_missing")
                terminal_persisted = True
            await stream_output.terminal(
                stream_output.error(
                    "The serving engine ended without a terminal event for every choice.",
                    502,
                )
            )
            return
        if not await _finalize_usage(stream_output, usage_session, final):
            terminal_persisted = True
            return
        terminal_persisted = True
        for index in range(choice_count):
            for key, value in splitters[index].finish():
                await stream_output.emit(stream_output.delta({key: value}, index=index))
        terminal_chunk = _terminal_chunk(
            completion_id,
            created,
            adapter_id,
            terminals,
            final,
            include_usage,
        )
        await stream_output.terminal(
            _sse(terminal_chunk),
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


async def _emit_delta(
    stream_output: _StreamOutput,
    splitters: dict[int, ReasoningDeltaSplitter],
    terminals: dict[int, dict[str, Any]],
    event: dict[str, Any],
) -> None:
    index = _event_index(event, len(splitters))
    if index in terminals:
        raise RuntimeError("choice delta followed its terminal")
    tool_calls = event.get("tool_calls")
    if tool_calls:
        deltas = []
        for call_index, call in enumerate(tool_calls):
            delta = dict(call)
            delta["index"] = call_index
            deltas.append(delta)
        await stream_output.emit(stream_output.delta({"tool_calls": deltas}, index=index))
        return
    rendered = splitters[index].feed(str(event.get("text") or ""))
    logprobs = event.get("logprobs")
    if logprobs is not None and len(rendered) > 1:
        raise RuntimeError("a logprob delta crossed the reasoning boundary")
    if not rendered and logprobs is not None:
        rendered = [("content", "")]
    for position, (key, value) in enumerate(rendered):
        await stream_output.emit(
            stream_output.delta(
                {key: value},
                index=index,
                logprobs=logprobs if position == 0 else None,
            )
        )


def _event_index(event: dict[str, Any], choice_count: int) -> int:
    index = event.get("index", 0 if choice_count == 1 else None)
    if type(index) is not int or not 0 <= index < choice_count:
        raise RuntimeError("invalid stream choice index")
    return index


def _has_usage(event: dict[str, Any]) -> bool:
    return event.get("prompt_tokens") is not None and event.get("completion_tokens") is not None


async def _finalize_usage(
    stream_output: _StreamOutput, usage_session: UsageSession, final: dict[str, Any]
) -> bool:
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
        return False
    return True


def _terminal_chunk(
    completion_id: str,
    created: int,
    adapter_id: str,
    terminals: dict[int, dict[str, Any]],
    final: dict[str, Any],
    include_usage: bool,
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": adapter_id,
        "choices": [
            {
                "index": index,
                "delta": {},
                "finish_reason": terminals[index]["finish_reason"],
            }
            for index in sorted(terminals)
        ],
    }
    if include_usage and _has_usage(final):
        chunk["usage"] = _usage_block(
            int(final["prompt_tokens"]),
            int(final["completion_tokens"]),
            final.get("cached_tokens"),
        )
    return chunk


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
    choice_count: int = 1,
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
            choice_count=choice_count,
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
