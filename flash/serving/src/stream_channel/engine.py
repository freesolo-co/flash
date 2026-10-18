"""Engine-side producer for the internal cancellable stream channel."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import queue as stdlib_queue
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol, TypeVar

from flash.serving.src.stream_channel import _tasks
from flash.serving.src.stream_channel.protocol import (
    CLEANUP_SECONDS,
    CONTROL_PARTITION,
    CONTROL_POLL_SECONDS,
    DATA_PARTITION,
    PARTITION_TTL_SECONDS,
    ChannelErrorCode,
    ControlEnvelope,
    ControlSequenceValidator,
    DataEnvelope,
    StreamChannelError,
    TerminalManifest,
)

_T = TypeVar("_T")


class StreamGenerateOwner(Protocol):
    """The engine surface `stream_generate_call` drives on the replica."""

    engine: Any

    def _replica_identifier(self) -> str: ...

    def _stream_generate(
        self,
        payload_dict: dict[str, Any],
        record_dict: dict[str, Any] | None = None,
        expected_checkpoint: str | None = None,
        generation_id: str | None = None,
        *,
        pre_generate_check: Callable[[], Awaitable[None]] | None = None,
    ) -> AsyncIterator[dict[str, Any]]: ...


class _UnsettledGuardedOperation(Exception):
    def __init__(self, cause: BaseException) -> None:
        self.cause = cause
        super().__init__(str(cause))


# one process-wide retention set, shared with the router side
_BACKGROUND_TASKS = _tasks.BACKGROUND_TASKS


async def _join_task(task: asyncio.Task[Any]) -> None:
    await _tasks.join_task(task, CLEANUP_SECONDS)


async def _stop_task(task: asyncio.Task[Any]) -> None:
    await _tasks.stop_task(task, CLEANUP_SECONDS)


async def _bounded_operation(operation: Awaitable[_T], timeout_seconds: float) -> _T:
    return await _tasks.bounded(operation, timeout_seconds, CLEANUP_SECONDS)


async def _finish_cleanup(operation: Awaitable[Any]) -> asyncio.CancelledError | None:
    return await _tasks.finish_cleanup(operation, CLEANUP_SECONDS)


class _LeaseWatch:
    def __init__(
        self,
        queue: Any,
        *,
        generation_id: str,
        invocation_nonce: str,
        function_call_id: str,
    ) -> None:
        self._queue = queue
        self._validator = ControlSequenceValidator(
            generation_id=generation_id,
            invocation_nonce=invocation_nonce,
            function_call_id=function_call_id,
        )
        self._latest: ControlEnvelope | None = None
        self._task: asyncio.Task[None] | None = None
        self._closed_error: BaseException | None = None
        self._consume_lock = asyncio.Lock()

    async def _consume(
        self,
        *,
        block: bool = True,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> bool:
        async with self._consume_lock:
            options: dict[str, Any] = {"partition": CONTROL_PARTITION}
            if not block:
                options["block"] = False
            elif timeout is not None:
                options["timeout"] = timeout
            local_timeout = (timeout or CONTROL_POLL_SECONDS) + CLEANUP_SECONDS
            try:
                raw = await _bounded_operation(
                    self._queue.get.aio(**options),
                    local_timeout,
                )
            except stdlib_queue.Empty:
                return False
            except TimeoutError as exc:
                raise StreamChannelError(
                    ChannelErrorCode.CHANNEL_FAULT,
                    "control channel read timed out",
                ) from exc
            if raw is None:
                return False
            self._latest = self._validator.accept(raw)
            return True

    async def admit(self) -> None:
        consumed = await self._consume(timeout=CONTROL_POLL_SECONDS)
        if not consumed:
            raise StreamChannelError(ChannelErrorCode.CHANNEL_FAULT, "initial lease is missing")
        while self._latest is not None and self._latest.function_call_id is None:
            if await self._consume(block=False):
                continue
            if not await self._consume(timeout=CONTROL_POLL_SECONDS):
                self.check()
        while await self._consume(block=False):
            pass
        self.check()
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            await self.refresh_and_check()
            await self._consume(timeout=CONTROL_POLL_SECONDS)

    async def refresh_and_check(self) -> None:
        while await self._consume(block=False):
            pass
        self.check()

    def check(self) -> None:
        if self._closed_error is not None:
            raise self._closed_error
        task = self._task
        if task is not None and task.done() and not task.cancelled():
            exception = task.exception()
            if exception is not None:
                raise exception
            raise StreamChannelError(ChannelErrorCode.CHANNEL_FAULT, "lease watcher stopped")
        latest = self._latest
        if latest is None:
            raise StreamChannelError(ChannelErrorCode.CHANNEL_FAULT, "lease is missing")
        if latest.kind == "cancel":
            raise StreamChannelError(ChannelErrorCode.CANCELLED, "stream cancelled")
        if time.time() >= latest.lease_deadline_unix:
            raise StreamChannelError(ChannelErrorCode.LEASE_EXPIRED, "stream lease expired")

    async def guarded(
        self,
        operation: Awaitable[_T],
        *,
        abort: Callable[[], Awaitable[None]] | None = None,
        require_settled: bool = False,
    ) -> _T:
        operation_task = asyncio.ensure_future(operation)

        async def stop_operation() -> bool:
            operation_task.cancel()
            if abort is not None:
                with contextlib.suppress(Exception):
                    await _bounded_operation(abort(), CLEANUP_SECONDS)
            await _join_task(operation_task)
            return operation_task.done()

        try:
            self.check()
            if self._task is None:
                raise StreamChannelError(
                    ChannelErrorCode.CHANNEL_FAULT, "lease watcher is not running"
                )
        except Exception as exc:
            settled = await stop_operation()
            if require_settled and not settled:
                raise _UnsettledGuardedOperation(exc) from exc
            raise
        try:
            done, _ = await asyncio.wait(
                {operation_task, self._task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        except asyncio.CancelledError:
            await stop_operation()
            raise
        if operation_task in done:
            return await operation_task
        await stop_operation()
        exception = self._task.exception()
        if exception is None:
            exception = StreamChannelError(ChannelErrorCode.CHANNEL_FAULT, "lease watcher stopped")
        if require_settled:
            raise _UnsettledGuardedOperation(exception) from exception
        raise exception

    async def close(self) -> None:
        if self._task is None:
            if self._closed_error is None:
                self._closed_error = StreamChannelError(
                    ChannelErrorCode.CHANNEL_FAULT, "lease watcher is closed"
                )
            return
        task = self._task
        if task.done() and not task.cancelled():
            self._closed_error = task.exception()
        if self._closed_error is None:
            self._closed_error = StreamChannelError(
                ChannelErrorCode.CHANNEL_FAULT, "lease watcher is closed"
            )
        self._task = None
        await _stop_task(task)


async def _close_stream(stream: Any) -> None:
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def _abort_generation(owner: StreamGenerateOwner, generation_id: str) -> None:
    result = owner.engine.abort(generation_id)
    if inspect.isawaitable(result):
        await result


async def _put_data(queue: Any, envelope: DataEnvelope) -> None:
    await queue.put.aio(
        envelope.to_dict(),
        partition=DATA_PARTITION,
        partition_ttl=PARTITION_TTL_SECONDS,
    )


async def stream_generate_call(
    owner: StreamGenerateOwner,
    payload_dict: dict[str, Any],
    record_dict: dict[str, Any] | None,
    expected_checkpoint: str | None,
    generation_id: str,
    dispatch_deadline_unix: float,
    queue_id: str,
    invocation_nonce: str,
) -> dict[str, Any]:
    """Run one exact generation behind a leased queue channel."""
    import modal

    queue = modal.Queue.from_id(queue_id)
    function_call_id = modal.current_function_call_id()
    if not function_call_id:
        raise StreamChannelError(ChannelErrorCode.CHANNEL_FAULT, "function call id is unavailable")
    engine_replica_id = owner._replica_identifier()
    lease = _LeaseWatch(
        queue,
        generation_id=generation_id,
        invocation_nonce=invocation_nonce,
        function_call_id=function_call_id,
    )
    stream = None
    generation_may_have_started = False
    abort_attempted = False
    sequence = 0
    terminal_kind = "error"
    completed = False

    async def pre_generate_check() -> None:
        nonlocal generation_may_have_started
        await lease.refresh_and_check()
        if time.time() >= dispatch_deadline_unix:
            raise StreamChannelError(
                ChannelErrorCode.DISPATCH_DEADLINE,
                "dispatch deadline expired before generation",
            )
        generation_may_have_started = True

    async def abort_if_started() -> None:
        nonlocal abort_attempted
        if not generation_may_have_started or completed or abort_attempted:
            return
        abort_attempted = True
        with contextlib.suppress(Exception):
            await _abort_generation(owner, generation_id)

    try:
        await lease.admit()
        if time.time() >= dispatch_deadline_unix:
            raise StreamChannelError(
                ChannelErrorCode.DISPATCH_DEADLINE,
                "dispatch deadline expired before hydration",
            )
        lease.check()
        stream = owner._stream_generate(
            payload_dict,
            record_dict,
            expected_checkpoint,
            generation_id,
            pre_generate_check=pre_generate_check,
        )
        while True:
            try:
                event = await lease.guarded(anext(stream), abort=abort_if_started)
            except StopAsyncIteration:
                break
            terminal = event.get("type") == "final"
            envelope = DataEnvelope(
                kind="event",
                generation_id=generation_id,
                invocation_nonce=invocation_nonce,
                function_call_id=function_call_id,
                engine_replica_id=engine_replica_id,
                sequence=sequence,
                terminal=terminal,
                event=event,
            )
            await lease.guarded(
                _put_data(queue, envelope),
                abort=abort_if_started,
                require_settled=True,
            )
            sequence += 1
            if terminal:
                terminal_kind = "event"
                break
        if terminal_kind != "event":
            raise StreamChannelError(
                ChannelErrorCode.ENGINE_ERROR, "stream ended without final event"
            )
        completed = True
    except asyncio.CancelledError:
        raise
    except _UnsettledGuardedOperation as exc:
        raise exc.cause from None
    except Exception as exc:
        code = exc.code if isinstance(exc, StreamChannelError) else ChannelErrorCode.ENGINE_ERROR
        envelope = DataEnvelope(
            kind="error",
            generation_id=generation_id,
            invocation_nonce=invocation_nonce,
            function_call_id=function_call_id,
            engine_replica_id=engine_replica_id,
            sequence=sequence,
            terminal=True,
            error_code=code,
        )
        try:
            await _bounded_operation(
                _put_data(queue, envelope),
                CONTROL_POLL_SECONDS,
            )
        except Exception:
            raise StreamChannelError(
                ChannelErrorCode.CHANNEL_FAULT,
                "failed to publish terminal channel error",
            ) from exc
        sequence += 1
        if not isinstance(exc, StreamChannelError):
            raise
    finally:
        cleanup_cancellation: asyncio.CancelledError | None = None
        if stream is not None:
            cleanup_cancellation = await _finish_cleanup(_close_stream(stream))
        abort_cancellation = await _finish_cleanup(abort_if_started())
        cleanup_cancellation = cleanup_cancellation or abort_cancellation
        lease_cancellation = await _finish_cleanup(lease.close())
        cleanup_cancellation = cleanup_cancellation or lease_cancellation
        if cleanup_cancellation is not None:
            raise cleanup_cancellation

    manifest = TerminalManifest(
        generation_id=generation_id,
        invocation_nonce=invocation_nonce,
        function_call_id=function_call_id,
        engine_replica_id=engine_replica_id,
        final_sequence=sequence - 1,
        terminal_kind=terminal_kind,
        event_count=sequence,
    )
    return manifest.to_dict()
