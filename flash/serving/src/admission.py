"""Per-model FIFO admission control for hosted serving requests."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Protocol, Self

from flash.serving.src.model_config import HostedTrafficPolicy


class DispatchLimit(Protocol):
    """Return the safe absolute local active limit for one model."""

    def __call__(self, model: str) -> int: ...


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    model: str
    active: int
    queued: int
    current_dispatch_limit: int
    hard_limit: int
    captured_at: float
    oldest_wait_seconds: float


class ServingOverloaded(RuntimeError):
    """The model's bounded admission queue is full."""

    code = "serving_overloaded"

    def __init__(self, model: str, retry_after_seconds: int) -> None:
        self.model = model
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"serving overloaded for model {model!r}")


class ServingCapacityUnavailable(RuntimeError):
    """The model has no fresh positive dispatch capacity."""

    code = "serving_capacity_unavailable"

    def __init__(self, model: str, retry_after_seconds: int) -> None:
        self.model = model
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"serving capacity unavailable for model {model!r}")


@dataclass(slots=True)
class _Waiter:
    future: asyncio.Future[None]
    enqueued_at: float
    admitted: bool = False


@dataclass(slots=True)
class _ModelState:
    policy: HostedTrafficPolicy
    active: int = 0
    waiters: deque[_Waiter] = field(default_factory=deque)


class AdmissionLease:
    """One admitted request slot whose release is exactly-once and idempotent."""

    __slots__ = ("_controller", "_model", "_queue_duration_seconds", "_released")

    def __init__(
        self,
        controller: AdmissionController,
        model: str,
        *,
        queue_duration_seconds: float,
    ) -> None:
        self._controller = controller
        self._model = model
        self._queue_duration_seconds = queue_duration_seconds
        self._released = False

    @property
    def queue_duration_seconds(self) -> float:
        return self._queue_duration_seconds

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._controller._release(self._model)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.release()


class AdmissionController:
    """Bounded FIFO request admission with isolated counters for each model."""

    def __init__(
        self,
        policy_for: Callable[[str], HostedTrafficPolicy],
        current_dispatch_limit: DispatchLimit,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._policy_for = policy_for
        self._current_dispatch_limit = current_dispatch_limit
        self._clock = clock
        self._states: dict[str, _ModelState] = {}

    async def acquire(self, model: str) -> AdmissionLease:
        state = self._state_for(model)
        self._promote(model, state)
        dispatch_limit = self._dispatch_limit(model, state.policy)
        if dispatch_limit <= 0:
            raise ServingCapacityUnavailable(model, state.policy.retry_after_seconds)
        if not state.waiters and state.active < dispatch_limit:
            state.active += 1
            return AdmissionLease(self, model, queue_duration_seconds=0.0)
        if len(state.waiters) >= state.policy.queue_capacity:
            raise ServingOverloaded(model, state.policy.retry_after_seconds)

        waiter = _Waiter(asyncio.get_running_loop().create_future(), self._clock())
        state.waiters.append(waiter)
        try:
            await waiter.future
            queue_duration_seconds = max(0.0, self._clock() - waiter.enqueued_at)
            return AdmissionLease(
                self,
                model,
                queue_duration_seconds=queue_duration_seconds,
            )
        except asyncio.CancelledError:
            if waiter.admitted:
                state.active -= 1
                self._promote(model, state)
            else:
                self._remove_waiter(state, waiter)
            raise

    def capacity_changed(self, model: str) -> None:
        state = self._states.get(model)
        if state is not None:
            self._promote(model, state)

    def active_count(self, model: str) -> int:
        state = self._states.get(model)
        return 0 if state is None else state.active

    def snapshot(self, model: str) -> AdmissionSnapshot:
        state = self._state_for(model)
        self._discard_cancelled(state)
        now = self._clock()
        oldest_wait_seconds = 0.0
        if state.waiters:
            oldest_wait_seconds = max(0.0, now - state.waiters[0].enqueued_at)
        return AdmissionSnapshot(
            model=model,
            active=state.active,
            queued=len(state.waiters),
            current_dispatch_limit=self._dispatch_limit(model, state.policy),
            hard_limit=state.policy.max_inputs * state.policy.max_containers,
            captured_at=now,
            oldest_wait_seconds=oldest_wait_seconds,
        )

    def _state_for(self, model: str) -> _ModelState:
        state = self._states.get(model)
        if state is None:
            state = _ModelState(self._policy_for(model))
            self._states[model] = state
        return state

    def _dispatch_limit(self, model: str, policy: HostedTrafficPolicy) -> int:
        limit = self._current_dispatch_limit(model)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            return 0
        return min(limit, policy.max_inputs * policy.max_containers)

    def _promote(self, model: str, state: _ModelState) -> None:
        self._discard_cancelled(state)
        limit = self._dispatch_limit(model, state.policy)
        while state.active < limit and state.waiters:
            waiter = state.waiters.popleft()
            if waiter.future.cancelled():
                continue
            waiter.admitted = True
            state.active += 1
            waiter.future.set_result(None)

    def _release(self, model: str) -> None:
        state = self._states[model]
        if state.active <= 0:
            raise RuntimeError(f"admission active count underflow for model {model!r}")
        state.active -= 1
        self._promote(model, state)

    @staticmethod
    def _remove_waiter(state: _ModelState, waiter: _Waiter) -> None:
        with suppress(ValueError):
            state.waiters.remove(waiter)

    @staticmethod
    def _discard_cancelled(state: _ModelState) -> None:
        if state.waiters:
            state.waiters = deque(
                waiter for waiter in state.waiters if not waiter.future.cancelled()
            )
