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


async def _bounded_shield(operation: Awaitable[Any], deadline_seconds: float) -> None:
    task = asyncio.ensure_future(operation)
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=deadline_seconds)
    except (TimeoutError, asyncio.CancelledError):
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise


async def _spawn_cancellation_safe(
    spawn: Callable[[], Awaitable[Any]],
    cancel: Callable[[Any], Awaitable[None]],
) -> Any:
    task = asyncio.create_task(spawn())
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        try:
            call = await asyncio.shield(task)
        except Exception:
            raise
        with contextlib.suppress(Exception):
            await asyncio.shield(cancel(call))
        raise


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
            await asyncio.wait_for(self._write("cancel"), timeout=CLEANUP_SECONDS)

    async def stop_heartbeats(self) -> None:
        self._stopped.set()
        task = self._heartbeat_task
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._heartbeat_task = None

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

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._run()

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
                await control.cancel()
                await _cancel_call(spawned)

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
                            manifest = await self._terminal_manifest(call_result_task)
                            validator.reconcile(manifest)
                            extra = await queue.get.aio(block=False, partition=DATA_PARTITION)
                            if extra is not None:
                                validator.accept(extra)
                            completed = True
                            if envelope.kind == "error":
                                raise StreamChannelError(
                                    envelope.error_code or ChannelErrorCode.CHANNEL_FAULT,
                                    "remote stream failed",
                                )
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
                            with contextlib.suppress(Exception):
                                call_result_task.result()
                            raise StreamChannelError(
                                ChannelErrorCode.CHANNEL_FAULT,
                                "function completed before terminal data became visible",
                            )
            finally:
                await control.stop_heartbeats()
                if not completed:
                    await control.cancel()
                    if call is not None:
                        with contextlib.suppress(Exception):
                            await _cancel_call(call)
                if call_result_task is not None and not call_result_task.done():
                    call_result_task.cancel()
                    await asyncio.gather(call_result_task, return_exceptions=True)
                await _clear_partition(queue, DATA_PARTITION)
                await _clear_partition(queue, CONTROL_PARTITION)

    @staticmethod
    async def _terminal_manifest(call_result_task: asyncio.Task[Any]) -> Any:
        try:
            return await asyncio.wait_for(
                asyncio.shield(call_result_task),
                timeout=CALL_RESULT_SECONDS,
            )
        except TimeoutError as exc:
            raise StreamChannelError(
                ChannelErrorCode.CHANNEL_FAULT,
                "terminal manifest timed out",
            ) from exc
        except Exception as exc:
            raise StreamChannelError(
                ChannelErrorCode.CHANNEL_FAULT,
                "remote function failed before manifest",
            ) from exc
