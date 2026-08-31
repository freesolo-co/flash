"""Router-side owner for one internal cancellable stream channel."""

from __future__ import annotations

import asyncio
import contextlib
import queue as stdlib_queue
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from flash.serving.src.stream_channel import _tasks
from flash.serving.src.stream_channel._tasks import OrphanHandler
from flash.serving.src.stream_channel._tasks import (
    retain_background_task as _retain_background_task,
)
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

# one process-wide retention set, shared with the engine side
_BACKGROUND_TASKS = _tasks.BACKGROUND_TASKS


async def _stop_task(
    task: asyncio.Task[Any],
    *,
    on_orphan: OrphanHandler | None = None,
) -> None:
    await _tasks.stop_task(task, CLEANUP_SECONDS, on_orphan=on_orphan)


async def _bounded_shield(
    operation: Awaitable[Any],
    deadline_seconds: float,
    *,
    on_orphan: OrphanHandler | None = None,
) -> Any:
    return await _tasks.bounded(
        operation,
        deadline_seconds,
        CLEANUP_SECONDS,
        on_orphan=on_orphan,
    )


async def _cancel_spawn_task_result(
    task: asyncio.Task[Any],
    cancel: Callable[[Any], Awaitable[None]],
) -> None:
    """Dispose of a spawn that landed after we stopped waiting for it."""
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
    _retain_background_task(asyncio.create_task(_cancel_spawn_task_result(task, cancel)))


def _spawn_orphan_handler(cancel: Callable[[Any], Awaitable[None]]) -> OrphanHandler:
    async def dispose(task: asyncio.Task[Any]) -> None:
        await _cancel_spawn_task_result(task, cancel)

    return dispose


async def _dispatch_bounded(
    operation: Awaitable[Any],
    dispatch_deadline_unix: float,
    phase: str,
) -> Any:
    """Await one dispatch step within what is left of the dispatch deadline."""
    remaining = max(0.0, dispatch_deadline_unix - time.time())
    try:
        return await _bounded_shield(operation, remaining)
    except TimeoutError as exc:
        raise StreamChannelError(
            ChannelErrorCode.DISPATCH_DEADLINE,
            f"dispatch deadline expired during {phase}",
        ) from exc


class _DispatchDeadlineContext:
    def __init__(self, context: Any, dispatch_deadline_unix: float) -> None:
        self._context = context
        self._dispatch_deadline_unix = dispatch_deadline_unix
        self._entered = False

    async def _exit_late_entry(self, task: asyncio.Task[Any]) -> None:
        """Close a context that finished opening after we stopped waiting for it."""
        try:
            task.result()
        except (asyncio.CancelledError, Exception):
            return
        with contextlib.suppress(Exception):
            await _bounded_shield(
                self._context.__aexit__(None, None, None),
                CLEANUP_SECONDS,
            )

    async def __aenter__(self) -> Any:
        remaining = max(0.0, self._dispatch_deadline_unix - time.time())
        try:
            value = await _bounded_shield(
                self._context.__aenter__(),
                remaining,
                on_orphan=self._exit_late_entry,
            )
        except TimeoutError as exc:
            raise StreamChannelError(
                ChannelErrorCode.DISPATCH_DEADLINE,
                "dispatch deadline expired during channel setup",
            ) from exc
        self._entered = True
        return value

    async def __aexit__(self, *exc_info: object) -> Any:
        if not self._entered:
            return None
        self._entered = False
        try:
            return await _bounded_shield(
                self._context.__aexit__(*exc_info),
                CLEANUP_SECONDS,
            )
        except Exception:
            return None


async def _cancel_or_retain_spawn(
    task: asyncio.Task[Any],
    cancel: Callable[[Any], Awaitable[None]],
) -> None:
    await _stop_task(task, on_orphan=_spawn_orphan_handler(cancel))


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
        await _bounded_shield(
            queue.clear.aio(partition=partition),
            CLEANUP_SECONDS,
        )


async def _get_data(queue: Any, *, block: bool = True) -> Any:
    options: dict[str, Any] = {"partition": DATA_PARTITION}
    local_timeout = CLEANUP_SECONDS
    if block:
        options["timeout"] = DATA_GET_SECONDS
        local_timeout += DATA_GET_SECONDS
    else:
        options["block"] = False
    try:
        return await _bounded_shield(
            queue.get.aio(**options),
            local_timeout,
        )
    except TimeoutError as exc:
        raise StreamChannelError(
            ChannelErrorCode.CHANNEL_FAULT,
            "data channel read timed out",
        ) from exc


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
        except BaseException:
            # any exit from the generator ends this channel, including exhaustion
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

        raw_context = (
            self._queue_context()
            if self._queue_context is not None
            else modal.Queue.ephemeral.aio()
        )
        context = _DispatchDeadlineContext(raw_context, self._dispatch_deadline_unix)
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
                await _dispatch_bounded(
                    control.initial(),
                    self._dispatch_deadline_unix,
                    "initial lease publication",
                )
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
                await _dispatch_bounded(
                    control.bind_and_start(function_call_id),
                    self._dispatch_deadline_unix,
                    "bound lease publication",
                )
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
                        raw = await _get_data(queue)
                    control.check()
                    if raw is not None:
                        envelope = validator.accept(raw)
                        drain_deadline = None
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
                                extra = await _get_data(queue, block=False)
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
