"""bounded ownership and cancellation for one pinned vllm request id."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

_T = TypeVar("_T")
_ABORT_TIMEOUT_SECONDS = 1.0
_DRAIN_TIMEOUT_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class CancellationResult:
    """whether cleanup settled and whether the explicit abort was confirmed."""

    settled: bool
    abort_confirmed: bool


class GenerationOwner:
    """own one request id from dispatch through confirmed completion or unhealthy detachment."""

    def __init__(
        self,
        engine: Any,
        request_id: str,
        *,
        mark_unhealthy: Callable[[str, str], None],
        detach: Callable[[str, asyncio.Future[Any]], None],
    ) -> None:
        self.request_id = request_id
        self._engine = engine
        self._mark_unhealthy = mark_unhealthy
        self._detach = detach
        self._active: set[asyncio.Future[Any]] = set()
        self._cancel_lock = asyncio.Lock()
        self._cancel_task: asyncio.Task[CancellationResult] | None = None

    async def wait(self, awaitable: Awaitable[_T]) -> _T:
        """shield engine work so caller cancellation reaches the explicit abort path only."""

        task = asyncio.ensure_future(awaitable)
        self._active.add(task)
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._active.discard(task)
            raise
        self._active.discard(task)
        return result

    async def cancel(self, stream: AsyncIterator[Any] | None = None) -> CancellationResult:
        """abort exactly once, then bound settlement without cancelling vllm's generator."""

        async with self._cancel_lock:
            if self._cancel_task is None:
                self._cancel_task = asyncio.create_task(self._cancel(stream))
            task = self._cancel_task
        return await asyncio.shield(task)

    async def _cancel(self, stream: AsyncIterator[Any] | None) -> CancellationResult:
        abort_confirmed = await self._abort_once()
        settlement = asyncio.create_task(self._settle(stream))
        settled = True
        try:
            await asyncio.wait_for(asyncio.shield(settlement), timeout=_DRAIN_TIMEOUT_SECONDS)
        except TimeoutError:
            settled = False
            self._mark_unhealthy(self.request_id, "generation cancellation did not settle")
            self._detach(self.request_id, settlement)
        if not abort_confirmed:
            self._mark_unhealthy(self.request_id, "vllm abort could not be confirmed")
        return CancellationResult(settled=settled, abort_confirmed=abort_confirmed)

    async def _abort_once(self) -> bool:
        abort = getattr(self._engine, "abort", None)
        if abort is None:
            return False
        try:
            result = abort(self.request_id)
        except Exception:
            return False
        if not inspect.isawaitable(result):
            return result is not False
        task = asyncio.ensure_future(result)
        try:
            result = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=_ABORT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            self._detach(self.request_id, task)
            return False
        except Exception:
            return False
        return result is not False

    async def _settle(self, stream: AsyncIterator[Any] | None) -> None:
        active = tuple(self._active)
        if active:
            await asyncio.gather(*active, return_exceptions=True)
            self._active.difference_update(active)
        if stream is None:
            return
        try:
            async for _ in stream:
                pass
        except BaseException:
            return
