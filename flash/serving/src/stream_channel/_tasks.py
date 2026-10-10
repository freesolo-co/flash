"""Bounded task lifecycle shared by both ends of the stream channel.

Both the router and the engine drive remote calls that may outlive the deadline
they were given. The rules are the same on both sides, so they live here once:

* an operation gets a bounded window, and is cancelled when the window closes;
* a task that refuses to stop within the cleanup window is retained, never
  dropped, so its exception is always consumed;
* an operation that lands after we stopped waiting is an *orphan*, and the
  caller may pass ``on_orphan`` to dispose of whatever it produced.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

_T = TypeVar("_T")

OrphanHandler = Callable[["asyncio.Task[Any]"], Awaitable[None]]

BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


def consume_task_result(task: asyncio.Task[Any]) -> None:
    with contextlib.suppress(asyncio.CancelledError, Exception):
        task.exception()


def retain_background_task(
    task: asyncio.Task[Any],
    *,
    on_orphan: OrphanHandler | None = None,
) -> None:
    """Hold a task that would not stop, and dispose of a late result if asked."""
    BACKGROUND_TASKS.add(task)

    def finish(completed: asyncio.Task[Any]) -> None:
        consume_task_result(completed)
        BACKGROUND_TASKS.discard(completed)
        if on_orphan is not None:
            retain_background_task(asyncio.ensure_future(on_orphan(completed)))

    task.add_done_callback(finish)


async def join_task(
    task: asyncio.Task[Any],
    cleanup_seconds: float,
    *,
    on_orphan: OrphanHandler | None = None,
) -> None:
    """Wait out the cleanup window, then either dispose of the result or retain the task."""
    if not task.done():
        done, _ = await asyncio.wait({task}, timeout=cleanup_seconds)
        if not done:
            retain_background_task(task, on_orphan=on_orphan)
            return
    consume_task_result(task)
    if on_orphan is not None:
        await on_orphan(task)


async def stop_task(
    task: asyncio.Task[Any],
    cleanup_seconds: float,
    *,
    on_orphan: OrphanHandler | None = None,
) -> None:
    if not task.done():
        task.cancel()
    await join_task(task, cleanup_seconds, on_orphan=on_orphan)


async def bounded(
    operation: Awaitable[_T],
    timeout_seconds: float,
    cleanup_seconds: float,
    *,
    on_orphan: OrphanHandler | None = None,
) -> _T:
    """Await one operation under a deadline, stopping it if the deadline or a cancel arrives."""
    task = asyncio.ensure_future(operation)
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout_seconds)
    except (TimeoutError, asyncio.CancelledError):
        await stop_task(task, cleanup_seconds, on_orphan=on_orphan)
        raise


async def _settle(
    task: asyncio.Task[Any],
    cleanup_seconds: float,
) -> asyncio.CancelledError | None:
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=cleanup_seconds)
    except asyncio.CancelledError as exc:
        return exc
    except Exception:
        return None
    return None


async def finish_cleanup(
    operation: Awaitable[Any],
    cleanup_seconds: float,
) -> asyncio.CancelledError | None:
    """Run one cleanup step to completion, returning a cancellation instead of raising it.

    Cleanup must not be abandoned halfway because the caller was cancelled, so a
    cancellation buys the step one more bounded attempt and is handed back to the
    caller to re-raise once every step has run.
    """
    task = asyncio.ensure_future(operation)
    cancellation = await _settle(task, cleanup_seconds)
    if cancellation is not None:
        await _settle(task, cleanup_seconds)
    await stop_task(task, cleanup_seconds)
    return cancellation
