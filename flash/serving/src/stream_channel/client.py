"""Router-side owner for one internal cancellable stream channel."""

from __future__ import annotations

import asyncio
import contextlib
import queue as stdlib_queue
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from flash.serving.src.stream_channel.protocol import (
    CALL_RESULT_SECONDS,
    CLEANUP_SECONDS,
    CONTROL_PARTITION,
    DATA_GET_SECONDS,
    DATA_PARTITION,
    HEARTBEAT_INTERVAL_SECONDS,
    LEASE_SECONDS,
    PARTITION_TTL_SECONDS,
    TERMINAL_DRAIN_SECONDS,
    ChannelErrorCode,
    ControlEnvelope,
    DataSequenceValidator,
    StreamChannelError,
)

_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.exception()


def _retain_background_task(task: asyncio.Task[Any]) -> None:
    _BACKGROUND_TASKS.add(task)

    def finish(completed: asyncio.Task[Any]) -> None:
        _consume_task_result(completed)
        _BACKGROUND_TASKS.discard(completed)

    task.add_done_callback(finish)


async def _stop_task(task: asyncio.Task[Any]) -> None:
    if not task.done():
        task.cancel()
        done, _ = await asyncio.wait({task}, timeout=CLEANUP_SECONDS)
        if not done:
            _retain_background_task(task)
            return
    _consume_task_result(task)


async def _cancel_spawn_task_result(
    task: asyncio.Task[Any],
    cancel: Callable[[Any], Awaitable[None]],
) -> None:
    try:
        call = task.result()
    except (asyncio.CancelledError, Exception):
        return
    with contextlib.suppress(Exception):
        await cancel(call)


def _schedule_spawn_result_cancel(
    task: asyncio.Task[Any],
    cancel: Callable[[Any], Awaitable[None]],
) -> None:
    cleanup = asyncio.create_task(_cancel_spawn_task_result(task, cancel))
    _retain_background_task(cleanup)


async def _bounded_shield(operation: Awaitable[Any], deadline_seconds: float) -> None:
    task = asyncio.ensure_future(operation)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=deadline_seconds)
    except (TimeoutError, asyncio.CancelledError):
        await _stop_task(task)
        raise


async def _cancel_or_retain_spawn(
    task: asyncio.Task[Any],
    cancel: Callable[[Any], Awaitable[None]],
) -> None:
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=CLEANUP_SECONDS)
    if done:
        await _cancel_spawn_task_result(task, cancel)
        return
    task.add_done_callback(lambda completed: _schedule_spawn_result_cancel(completed, cancel))
    _retain_background_task(task)


async def _spawn_cancellation_safe(
    spawn: Callable[[], Awaitable[Any]],
    cancel: Callable[[Any], Awaitable[None]],
    *,
    dispatch_deadline_unix: float,
) -> Any:
    task = asyncio.create_task(spawn())
    remaining = max(0.0, dispatch_deadline_unix - time.time())
    try:
        done, _ = await asyncio.wait({task}, timeout=remaining)
        if done:
            return task.result()
        await _cancel_or_retain_spawn(task, cancel)
        raise StreamChannelError(
            ChannelErrorCode.DISPATCH_DEADLINE,
            "dispatch deadline expired while awaiting spawn",
        )
    except asyncio.CancelledError as cancellation:
        remaining = max(0.0, dispatch_deadline_unix - time.time())
        handle_wait = min(LEASE_SECONDS, remaining)
        try:
            call = await asyncio.wait_for(asyncio.shield(task), timeout=handle_wait)
        except Exception:
            await _cancel_or_retain_spawn(task, cancel)
            raise cancellation from None
        with contextlib.suppress(Exception):
            await cancel(call)
        raise cancellation


class _ControlWriter:
    def __init__(self, queue: Any, generation_id: str, invocation_nonce: str) -> None:
        self._queue = queue
        self._generation_id = generation_id
        self._invocation_nonce = invocation_nonce
        self._function_call_id: str | None = None
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._stopped = asyncio.Event()
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._cancelled = False

    async def initial(self) -> None:
        await self._write("active")

    def bind(self, function_call_id: str) -> None:
        self._function_call_id = function_call_id

    async def bind_and_start(self, function_call_id: str) -> None:
        self.bind(function_call_id)
        await self._write("active")
        self._heartbeat_task = asyncio.create_task(self._heartbeat())

    async def _heartbeat(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._stopped.wait(),
                    timeout=HEARTBEAT_INTERVAL_SECONDS,
                )
                return
            except TimeoutError:
                await self._write("active")

    def check(self) -> None:
        task = self._heartbeat_task
        if task is None or not task.done():
            return
        exception = task.exception()
        if exception is not None:
            raise StreamChannelError(
                ChannelErrorCode.CHANNEL_FAULT, "heartbeat channel failed"
            ) from exception
        raise StreamChannelError(ChannelErrorCode.CHANNEL_FAULT, "heartbeat task stopped")

    async def cancel(self) -> None:
        await self.stop_heartbeats()
        if self._cancelled:
            return
        self._cancelled = True
        with contextlib.suppress(Exception):
            await _bounded_shield(self._write("cancel"), CLEANUP_SECONDS)

    async def stop_heartbeats(self) -> None:
        self._stopped.set()
        task = self._heartbeat_task
        if task is None:
            return
        self._heartbeat_task = None
        await _stop_task(task)

    async def _write(self, kind: str) -> None:
        async with self._lock:
            envelope = ControlEnvelope(
                kind=kind,
                generation_id=self._generation_id,
                invocation_nonce=self._invocation_nonce,
                function_call_id=self._function_call_id,
                sequence=self._sequence,
                lease_deadline_unix=time.time() + LEASE_SECONDS,
            )
            await self._queue.put.aio(
                envelope.to_dict(),
                partition=CONTROL_PARTITION,
                partition_ttl=PARTITION_TTL_SECONDS,
            )
            self._sequence += 1


async def _cancel_call(call: Any) -> None:
    await _bounded_shield(call.cancel.aio(terminate_containers=False), CALL_RESULT_SECONDS)


async def _clear_partition(queue: Any, partition: str) -> None:
    with contextlib.suppress(Exception):
        await asyncio.wait_for(queue.clear.aio(partition=partition), timeout=CLEANUP_SECONDS)


class CancellableStreamChannel:
    """Own the queue, exact FunctionCall, heartbeat lease, validation, and cleanup."""

    def __init__(
        self,
        *,
        spawn_method: Any,
        payload_dict: dict[str, Any],
        record_dict: dict[str, Any] | None,
        expected_checkpoint: str | None,
        generation_id: str,
        dispatch_deadline_unix: float,
        invocation_nonce: str,
        queue_context: Callable[[], Any] | None = None,
    ) -> None:
        self._spawn_method = spawn_method
        self._payload_dict = payload_dict
        self._record_dict = record_dict
        self._expected_checkpoint = expected_checkpoint
        self._generation_id = generation_id
        self._dispatch_deadline_unix = dispatch_deadline_unix
        self._invocation_nonce = invocation_nonce
        self._queue_context = queue_context
        self._iterator: AsyncIterator[dict[str, Any]] | None = None
        self._closed = False

    def __aiter__(self) -> CancellableStreamChannel:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._closed:
            raise StopAsyncIteration
        if self._iterator is None:
            self._iterator = self._run()
        try:
            return await anext(self._iterator)
        except (StopAsyncIteration, asyncio.CancelledError):
            self._closed = True
            self._iterator = None
            raise
        except Exception:
            self._closed = True
            self._iterator = None
            raise

    async def aclose(self) -> None:
        if self._closed and self._iterator is None:
            return
        self._closed = True
        iterator = self._iterator
        self._iterator = None
        if iterator is not None:
            await iterator.aclose()

    async def _run(self) -> AsyncIterator[dict[str, Any]]:
        import modal

        context = (
            self._queue_context()
            if self._queue_context is not None
            else modal.Queue.ephemeral.aio()
        )
        call = None
        call_result_task: asyncio.Task[Any] | None = None
        completed = False
        async with context as queue:
            control = _ControlWriter(queue, self._generation_id, self._invocation_nonce)

            async def cancel_spawned(spawned: Any) -> None:
                function_call_id = getattr(spawned, "object_id", None)
                if isinstance(function_call_id, str) and function_call_id:
                    control.bind(function_call_id)
                try:
                    await _cancel_call(spawned)
                finally:
                    await control.cancel()

            try:
                await control.initial()
                call = await _spawn_cancellation_safe(
                    lambda: self._spawn_method.spawn.aio(
                        self._payload_dict,
                        self._record_dict,
                        self._expected_checkpoint,
                        self._generation_id,
                        self._dispatch_deadline_unix,
                        queue.object_id,
                        self._invocation_nonce,
                    ),
                    cancel_spawned,
                    dispatch_deadline_unix=self._dispatch_deadline_unix,
                )
                function_call_id = call.object_id
                if not isinstance(function_call_id, str) or not function_call_id:
                    raise StreamChannelError(
                        ChannelErrorCode.CHANNEL_FAULT,
                        "spawn returned no function call id",
                    )
                await control.bind_and_start(function_call_id)
                validator = DataSequenceValidator(
                    generation_id=self._generation_id,
                    invocation_nonce=self._invocation_nonce,
                    function_call_id=function_call_id,
                )
                call_result_task = asyncio.create_task(call.get.aio())
                drain_deadline: float | None = None
                while True:
                    raw = None
                    with contextlib.suppress(stdlib_queue.Empty):
                        raw = await queue.get.aio(
                            timeout=DATA_GET_SECONDS,
                            partition=DATA_PARTITION,
                        )
                    control.check()
                    if raw is not None:
                        envelope = validator.accept(raw)
                        if envelope.terminal:
                            if envelope.kind == "error":
                                try:
                                    manifest = await self._call_result(call_result_task)
                                except StreamChannelError:
                                    raise
                                except Exception:
                                    completed = True
                                    raise
                                validator.reconcile(manifest)
                                completed = True
                                raise StreamChannelError(
                                    envelope.error_code or ChannelErrorCode.CHANNEL_FAULT,
                                    "remote stream failed without a function exception",
                                )
                            manifest = await self._call_result(call_result_task)
                            validator.reconcile(manifest)
                            extra = None
                            with contextlib.suppress(stdlib_queue.Empty):
                                extra = await queue.get.aio(
                                    block=False,
                                    partition=DATA_PARTITION,
                                )
                            if extra is not None:
                                validator.accept(extra)
                            completed = True
                            if envelope.event is None:
                                raise StreamChannelError(
                                    ChannelErrorCode.PROTOCOL_ERROR,
                                    "terminal event is missing",
                                )
                            yield envelope.event
                            return
                        if envelope.event is None:
                            raise StreamChannelError(
                                ChannelErrorCode.PROTOCOL_ERROR,
                                "stream event is missing",
                            )
                        yield envelope.event
                    if call_result_task.done() and validator.terminal is None:
                        if drain_deadline is None:
                            drain_deadline = time.monotonic() + TERMINAL_DRAIN_SECONDS
                        elif time.monotonic() >= drain_deadline:
                            call_result_task.result()
                            raise StreamChannelError(
                                ChannelErrorCode.CHANNEL_FAULT,
                                "function completed before terminal data became visible",
                            )
            finally:
                if not completed and call is not None:
                    with contextlib.suppress(Exception):
                        await _cancel_call(call)
                if not completed:
                    await control.cancel()
                else:
                    await control.stop_heartbeats()
                if call_result_task is not None:
                    await _stop_task(call_result_task)
                await _clear_partition(queue, DATA_PARTITION)
                await _clear_partition(queue, CONTROL_PARTITION)

    @staticmethod
    async def _call_result(call_result_task: asyncio.Task[Any]) -> Any:
        try:
            return await asyncio.wait_for(
                asyncio.shield(call_result_task),
                timeout=CALL_RESULT_SECONDS,
            )
        except TimeoutError as exc:
            raise StreamChannelError(
                ChannelErrorCode.CHANNEL_FAULT,
                "remote function result timed out",
            ) from exc
