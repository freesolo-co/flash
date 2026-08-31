"""choice-aware packaged OpenAI SSE orchestration."""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any

from flash.serve.runtime import (
    StreamChoiceFinished,
    StreamDelta,
    StreamFinished,
    StreamReady,
)

from .bootstrap import PublishedAdapter
from .openai import (
    ReasoningDeltaSplitter,
    sse_data,
    stream_chunk,
    usage_stream_chunk,
)

_HTTP_CANCELLATION_GRACE_SECONDS = 3.0
_DETACHED_TASKS: set[asyncio.Future[Any]] = set()


def _consume_terminal(task: asyncio.Future[Any]) -> None:
    with suppress(asyncio.CancelledError):
        task.exception()


def _cancellation_count() -> int:
    caller = asyncio.current_task()
    return 0 if caller is None else caller.cancelling()


def _new_caller_cancellation(previous_count: int) -> bool:
    return _cancellation_count() > previous_count


def _raise_caller_cancellation(
    caught: asyncio.CancelledError,
    interrupted: BaseException | None,
) -> None:
    """re-raise caller cancellation without discarding the exception it interrupted.

    these helpers run from `finally`, so an operation error is often already in flight. plain
    asyncio preserves it as the cancellation's `__context__`, but that only holds while the
    original is the frame's current exception: awaiting inside the handler makes `CancelledError`
    current, so by the time we re-raise, `sys.exc_info()` is the cancellation itself and the
    original is unreachable. `interrupted` is therefore captured on entry, before the first await.
    """

    if interrupted is not None and interrupted is not caught:
        caught.__context__ = interrupted
    raise caught


def detach_task(task: asyncio.Future[Any]) -> None:
    """retain a detached task until its terminal exception has been consumed."""

    _DETACHED_TASKS.add(task)

    def consume(done: asyncio.Future[Any]) -> None:
        _DETACHED_TASKS.discard(done)
        _consume_terminal(done)

    task.add_done_callback(consume)


async def cancel_task(task: asyncio.Future[Any]) -> bool:
    """cancel and await one task within the http cancellation grace."""

    interrupted = sys.exc_info()[1]
    cancellation_count = _cancellation_count()
    if task.done():
        _consume_terminal(task)
        return True
    task.cancel()
    try:
        await asyncio.wait_for(
            asyncio.shield(task),
            timeout=_HTTP_CANCELLATION_GRACE_SECONDS,
        )
    except asyncio.CancelledError as exc:
        if task.done():
            _consume_terminal(task)
        if _new_caller_cancellation(cancellation_count):
            if not task.done():
                detach_task(task)
            _raise_caller_cancellation(exc, interrupted)
        return True
    except TimeoutError:
        if task.done():
            _consume_terminal(task)
            return True
        detach_task(task)
        return False
    except Exception:
        _consume_terminal(task)
        return True
    return True


async def close_iterator(iterator: AsyncIterator[Any]) -> bool:
    """close one iterator within the same bounded http cancellation grace."""

    interrupted = sys.exc_info()[1]
    cancellation_count = _cancellation_count()
    close = getattr(iterator, "aclose", None)
    if close is None:
        return True
    try:
        result = close()
    except asyncio.CancelledError:
        return True
    except Exception:
        return True
    if not inspect.isawaitable(result):
        return True
    task = asyncio.ensure_future(result)
    try:
        await asyncio.wait_for(
            asyncio.shield(task),
            timeout=_HTTP_CANCELLATION_GRACE_SECONDS,
        )
    except asyncio.CancelledError as exc:
        if task.done():
            _consume_terminal(task)
        if _new_caller_cancellation(cancellation_count):
            if not task.done():
                detach_task(task)
            _raise_caller_cancellation(exc, interrupted)
        return True
    except TimeoutError:
        if task.done():
            _consume_terminal(task)
            return True
        detach_task(task)
        return False
    except Exception:
        _consume_terminal(task)
        return True
    return True


async def stream_chat_body(
    event_stream: AsyncIterator[Any],
    ready: StreamReady,
    resolved: PublishedAdapter,
    provenance: dict[str, Any],
    *,
    choice_count: int,
    include_usage: bool,
) -> AsyncIterator[bytes]:
    """render indexed choice events with one aggregate usage and done terminal."""

    splitters = {
        index: ReasoningDeltaSplitter(thinking=bool(ready.thinking))
        for index in range(choice_count)
    }
    terminals: dict[int, StreamChoiceFinished] = {}
    finished: StreamFinished | None = None
    succeeded = False
    try:
        for index in range(choice_count):
            yield sse_data(
                stream_chunk(
                    request_id=ready.request_id,
                    model=resolved.requested_model,
                    index=index,
                    delta={"role": "assistant", "content": ""},
                    provenance=provenance,
                )
            )
        async for event in event_stream:
            if type(event) is StreamDelta and finished is None:
                if event.index in terminals or event.index not in splitters:
                    raise RuntimeError("invalid stream choice delta")
                if event.tool_calls:
                    yield sse_data(
                        stream_chunk(
                            request_id=ready.request_id,
                            model=resolved.requested_model,
                            index=event.index,
                            delta={
                                "tool_calls": [
                                    call.wire(index=call_index)
                                    for call_index, call in enumerate(event.tool_calls)
                                ]
                            },
                        )
                    )
                    continue
                rendered = splitters[event.index].feed(event.text)
                if event.logprobs is not None and len(rendered) > 1:
                    raise RuntimeError("a logprob delta crossed the reasoning boundary")
                if not rendered and event.logprobs is not None:
                    rendered = [("content", "")]
                for position, (key, value) in enumerate(rendered):
                    yield sse_data(
                        stream_chunk(
                            request_id=ready.request_id,
                            model=resolved.requested_model,
                            index=event.index,
                            delta={key: value},
                            logprobs=event.logprobs if position == 0 else None,
                        )
                    )
                continue
            if type(event) is StreamChoiceFinished and finished is None:
                if event.index in terminals or event.index not in splitters:
                    raise RuntimeError("invalid stream choice terminal")
                terminals[event.index] = event
                continue
            if type(event) is StreamFinished and finished is None:
                finished = event
                continue
            raise RuntimeError("invalid stream event order")
        if finished is None or set(terminals) != set(range(choice_count)):
            raise RuntimeError("stream ended without every choice terminal")
        if (
            finished.request_id != ready.request_id
            or finished.runtime_id != ready.runtime_id
            or finished.adapter_id != resolved.adapter.checkpoint_id
            or finished.incarnation != resolved.adapter.aggregate_sha256
            or finished.thinking != ready.thinking
        ):
            raise RuntimeError("stream terminal identity mismatch")
        for index in range(choice_count):
            for key, value in splitters[index].finish():
                yield sse_data(
                    stream_chunk(
                        request_id=ready.request_id,
                        model=resolved.requested_model,
                        index=index,
                        delta={key: value},
                    )
                )
            yield sse_data(
                stream_chunk(
                    request_id=ready.request_id,
                    model=resolved.requested_model,
                    index=index,
                    delta={},
                    finish_reason=terminals[index].finish_reason,
                    provenance=provenance,
                )
            )
        if include_usage:
            yield sse_data(usage_stream_chunk(finished, resolved.requested_model, provenance))
        succeeded = True
    except asyncio.CancelledError:
        raise
    except Exception:
        yield sse_data(
            {
                "error": {
                    "message": "generation stream terminated",
                    "type": "server_error",
                    "code": "stream_terminated",
                }
            }
        )
    finally:
        await close_iterator(event_stream)
    if succeeded:
        yield sse_data("[DONE]")
