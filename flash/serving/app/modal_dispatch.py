"""Cancellable Modal call dispatch bounded by an absolute pre-header deadline.

Split out of ``modal_app`` so the app module stays a registration surface. These helpers own the
one behavior that must hold for every hosted request: a request that cannot reach a gpu before its
deadline is refused as retryable capacity pressure, and anything it already started is cancelled
rather than left billing.

``modal`` is imported here at module scope because this module is only imported from the app and
from the engine pool, both of which already require it.
"""

import asyncio
import contextlib
import math
import time
from typing import Any

import modal

from flash.serving.src.engine.dispatch import PreHeaderDispatchExpired


def _remaining_pre_header_dispatch_time(deadline: float) -> float:
    remaining = deadline - time.time()
    if remaining <= 0:
        raise PreHeaderDispatchExpired("request expired before gpu generation began")
    return remaining


async def _await_task_before_deadline(task: asyncio.Task[Any], deadline: float) -> Any:
    done, _ = await asyncio.wait(
        {task},
        timeout=_remaining_pre_header_dispatch_time(deadline),
    )
    if task not in done:
        raise PreHeaderDispatchExpired("request expired before gpu generation began")
    return task.result()


async def _cancel_modal_call(call: Any) -> None:
    with contextlib.suppress(BaseException):
        await call.cancel.aio()


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    with contextlib.suppress(BaseException):
        task.result()


# cleanup runs after the request is already unwinding, so nothing awaits these tasks. holding a
# strong reference keeps the event loop from collecting them mid-flight.
_MODAL_CLEANUP_TASKS: set[asyncio.Task[Any]] = set()
_MODAL_CLEANUP_SECONDS = 1.0


def _retain_modal_task(task: asyncio.Task[Any]) -> None:
    _MODAL_CLEANUP_TASKS.add(task)

    def finish(completed: asyncio.Task[Any]) -> None:
        _consume_task_result(completed)
        _MODAL_CLEANUP_TASKS.discard(completed)

    task.add_done_callback(finish)


def _start_modal_call_cleanup(call: Any) -> None:
    _retain_modal_task(asyncio.create_task(_cancel_modal_call(call)))


def _stop_local_task(task: asyncio.Task[Any]) -> None:
    if not task.done():
        task.cancel()
    _retain_modal_task(task)


def _cancel_modal_call_when_spawned(spawn: asyncio.Task[Any]) -> None:
    # the spawn may still land after we stopped waiting for it. cancelling only what actually got
    # created is what keeps an abandoned dispatch from billing a full generation.
    _MODAL_CLEANUP_TASKS.add(spawn)

    def cancel_if_created(task: asyncio.Task[Any]) -> None:
        _MODAL_CLEANUP_TASKS.discard(task)
        if task.cancelled():
            return
        try:
            call = task.result()
        except BaseException:
            return
        _start_modal_call_cleanup(call)

    spawn.add_done_callback(cancel_if_created)


async def _spawn_modal_call(method: Any, deadline: float, *args: Any) -> Any:
    """Acquire a cancellable handle within the remaining pre-header dispatch budget."""
    spawn = asyncio.create_task(method.spawn.aio(*args))
    try:
        return await _await_task_before_deadline(spawn, deadline)
    except (PreHeaderDispatchExpired, asyncio.CancelledError):
        _cancel_modal_call_when_spawned(spawn)
        raise


async def _finish_admission_context(context: Any) -> None:
    cleanup = asyncio.create_task(context.__aexit__(None, None, None))
    done, _ = await asyncio.wait({cleanup}, timeout=_MODAL_CLEANUP_SECONDS)
    if cleanup not in done:
        _retain_modal_task(cleanup)
        return
    _consume_task_result(cleanup)


def _finish_late_admission_entry(context: Any, task: asyncio.Task[Any]) -> None:
    async def finish() -> None:
        try:
            task.result()
        except BaseException:
            return
        await _finish_admission_context(context)

    _retain_modal_task(asyncio.create_task(finish()))


@contextlib.asynccontextmanager
async def _admission_queue(deadline: float):
    """An ephemeral queue the engine acknowledges admission on, torn down with the request."""
    context = modal.Queue.ephemeral.aio()
    entry = asyncio.create_task(context.__aenter__())
    try:
        queue = await _await_task_before_deadline(entry, deadline)
    except BaseException:
        # the queue may still be created after the deadline passed. close it when it lands so an
        # abandoned dispatch does not leak an ephemeral queue.
        _retain_modal_task(entry)
        entry.add_done_callback(lambda task: _finish_late_admission_entry(context, task))
        raise
    try:
        yield queue
    finally:
        await _finish_admission_context(context)


async def _await_modal_call(
    call: Any,
    queue: Any,
    deadline: float,
    *,
    generation_id: str,
    invocation_nonce: str,
) -> Any:
    """Race exact remote admission, completion, and the absolute pre-header deadline.

    admission means the engine has this exact request on a gpu. once acknowledged the deadline no
    longer applies, because refusing then would abandon work already being paid for.
    """
    from flash.serving.src.engine.dispatch import validate_admission_acknowledgement

    function_call_id = getattr(call, "object_id", None)
    if not isinstance(function_call_id, str) or not function_call_id:
        _start_modal_call_cleanup(call)
        raise RuntimeError("spawn returned no function call id")
    result = asyncio.create_task(call.get.aio())
    acknowledgement = asyncio.create_task(queue.get.aio())
    try:
        done, _ = await asyncio.wait(
            {result, acknowledgement},
            timeout=_remaining_pre_header_dispatch_time(deadline),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if result in done:
            _stop_local_task(acknowledgement)
            return result.result()
        if acknowledgement in done:
            value = acknowledgement.result()
            # deliberately not re-checked against the deadline here. the acknowledgement proves the
            # engine already holds this exact request on a gpu, so expiring on a boundary tick would
            # refuse the caller with a retryable 503 and cancel work that is already being paid for.
            # an acknowledgement that arrives too late loses the race above and never reaches here.
            validate_admission_acknowledgement(
                value,
                generation_id=generation_id,
                invocation_nonce=invocation_nonce,
                function_call_id=function_call_id,
            )
            return await result
        raise PreHeaderDispatchExpired("request expired before gpu generation began")
    except BaseException:
        _stop_local_task(result)
        _stop_local_task(acknowledgement)
        _start_modal_call_cleanup(call)
        raise


def _required_stream_identity(payload: Any) -> tuple[str, float]:
    generation_id = getattr(payload, "generation_id", None)
    if (
        not isinstance(generation_id, str)
        or not generation_id
        or len(generation_id) > 512
        or generation_id != generation_id.strip()
    ):
        raise RuntimeError("valid generation id is required before modal dispatch")
    deadline = getattr(payload, "_pre_header_dispatch_deadline", None)
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(deadline)
    ):
        raise RuntimeError("valid pre-header dispatch deadline is required before modal dispatch")
    return generation_id, deadline
