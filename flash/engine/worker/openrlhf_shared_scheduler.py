"""fair event-driven scheduling for shared OpenRLHF logical runs."""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from flash.engine.worker.openrlhf_shared_scoring import (
    ScoringBatchIdentity,
    ScoringCapacityError,
    ScoringFuture,
    ScoringIdentityError,
    ScoringKind,
    ScoringResult,
    SharedScoringPool,
)


class SharedSchedulerError(RuntimeError):
    """base error for invalid shared scheduler operations."""


class RunStateTransitionError(SharedSchedulerError):
    """raised when a run attempts an invalid state transition."""


class RunPhase(StrEnum):
    """explicit lifecycle states for one logical training run."""

    ADMITTED = "admitted"
    READY_ROLLOUT = "ready_rollout"
    ROLLOUT_IN_FLIGHT = "rollout_in_flight"
    SCORE_PENDING = "score_pending"
    READY_UPDATE = "ready_update"
    UPDATE_IN_FLIGHT = "update_in_flight"
    SYNCING = "syncing"
    FINISHING = "finishing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SchedulerEventKind(StrEnum):
    """events that advance one run state machine."""

    RUN_ADMITTED = "run_admitted"
    ROLLOUT_STARTED = "rollout_started"
    ROLLOUT_COMPLETED = "rollout_completed"
    SCORING_SUBMITTED = "scoring_submitted"
    SCORING_COMPLETED = "scoring_completed"
    UPDATE_STARTED = "update_started"
    UPDATE_COMPLETED = "update_completed"
    SYNC_COMPLETED = "sync_completed"
    RUN_DONE = "run_done"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"


@dataclass(frozen=True, slots=True)
class GpuWorkResult:
    """callback result with optional deterministic GPU time for simulation tests."""

    value: Any
    gpu_ms: float | None = None


@dataclass(frozen=True, slots=True)
class SchedulerEvent:
    """immutable event emitted by the controller."""

    kind: SchedulerEventKind
    run_id: str
    step: int
    timestamp: float


@dataclass(frozen=True, slots=True)
class StepWorldResult:
    """summary of one non-blocking controller advance."""

    progressed: bool
    gpu_run_id: str | None = None
    gpu_phase: RunPhase | None = None


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """public immutable snapshot of one scheduler-owned run."""

    run_id: str
    phase: RunPhase
    completed_steps: int
    total_steps: int
    weight: float
    deficit_ms: float
    gpu_time_ms: float
    outstanding_identity: ScoringBatchIdentity | None
    outstanding_rollout: bool
    failure: BaseException | None


RolloutCallback = Callable[[str, int], Awaitable[Any] | Any]
ScoringPayloadCallback = Callable[[str, int, Any], Mapping[str, Any]]
UpdateCallback = Callable[[str, int, Any, ScoringResult], Awaitable[Any] | Any]
CleanupCallback = Callable[[str], Awaitable[None] | None]
ScoringBridge = Callable[[dict[str, Any]], Any]

_MISSING_ROLLOUT = object()


@dataclass(frozen=True, slots=True)
class SchedulerRunHooks:
    """algorithm-neutral callbacks supplied by the later GRPO or OPD driver."""

    rollout: RolloutCallback
    scoring_payload: ScoringPayloadCallback
    update_and_publish: UpdateCallback
    cleanup: CleanupCallback | None = None


@dataclass(slots=True)
class _RunState:
    run_id: str
    hooks: SchedulerRunHooks
    weight: float
    total_steps: int
    registration_order: int
    phase: RunPhase = RunPhase.ADMITTED
    completed_steps: int = 0
    deficit_ms: float = 0.0
    gpu_time_ms: float = 0.0
    ready_since: float | None = None
    rollout: Any = _MISSING_ROLLOUT
    scoring_payload: Mapping[str, Any] | None = None
    scoring_identity: ScoringBatchIdentity | None = None
    scoring_future: ScoringFuture | None = None
    scoring_result: ScoringResult | None = None
    failure: BaseException | None = None
    cleaned_up: bool = False


_TERMINAL_PHASES = frozenset({RunPhase.DONE, RunPhase.FAILED, RunPhase.CANCELLED})
_ALLOWED_TRANSITIONS: dict[RunPhase, frozenset[RunPhase]] = {
    RunPhase.ADMITTED: frozenset({RunPhase.READY_ROLLOUT, RunPhase.FAILED, RunPhase.CANCELLED}),
    RunPhase.READY_ROLLOUT: frozenset(
        {RunPhase.ROLLOUT_IN_FLIGHT, RunPhase.FAILED, RunPhase.CANCELLED}
    ),
    RunPhase.ROLLOUT_IN_FLIGHT: frozenset(
        {RunPhase.SCORE_PENDING, RunPhase.FAILED, RunPhase.CANCELLED}
    ),
    RunPhase.SCORE_PENDING: frozenset({RunPhase.READY_UPDATE, RunPhase.FAILED, RunPhase.CANCELLED}),
    RunPhase.READY_UPDATE: frozenset(
        {RunPhase.UPDATE_IN_FLIGHT, RunPhase.FAILED, RunPhase.CANCELLED}
    ),
    RunPhase.UPDATE_IN_FLIGHT: frozenset({RunPhase.SYNCING, RunPhase.FAILED, RunPhase.CANCELLED}),
    RunPhase.SYNCING: frozenset(
        {RunPhase.READY_ROLLOUT, RunPhase.FINISHING, RunPhase.FAILED, RunPhase.CANCELLED}
    ),
    RunPhase.FINISHING: frozenset({RunPhase.DONE, RunPhase.FAILED, RunPhase.CANCELLED}),
    RunPhase.DONE: frozenset(),
    RunPhase.FAILED: frozenset(),
    RunPhase.CANCELLED: frozenset(),
}


async def _resolve_callback(value: Awaitable[Any] | Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class SharedEngineRunController:
    """advance independent runs over one serialized GPU lease.

    scoring remains in the bounded PR3 pool. ``step_the_world`` only polls completed
    futures, retries capacity-limited submissions, and executes at most one ready GPU
    quantum. weighted deficit accounting uses measured GPU milliseconds. aged updates
    close rollout admission until the oldest fair update runs.
    """

    def __init__(
        self,
        scoring_pool: SharedScoringPool,
        *,
        deficit_quantum_ms: float = 10.0,
        update_priority_ms: float = 2.5,
        update_starvation_s: float = 30.0,
        age_boost_ms_per_s: float = 1.0,
        max_consecutive_quanta: int = 4,
        scoring_poll_interval_s: float = 0.001,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if deficit_quantum_ms <= 0 or not math.isfinite(deficit_quantum_ms):
            raise ValueError("deficit_quantum_ms must be positive and finite")
        if update_priority_ms < 0 or not math.isfinite(update_priority_ms):
            raise ValueError("update_priority_ms must be non-negative and finite")
        if update_starvation_s <= 0 or not math.isfinite(update_starvation_s):
            raise ValueError("update_starvation_s must be positive and finite")
        if age_boost_ms_per_s < 0 or not math.isfinite(age_boost_ms_per_s):
            raise ValueError("age_boost_ms_per_s must be non-negative and finite")
        if isinstance(max_consecutive_quanta, bool) or max_consecutive_quanta < 1:
            raise ValueError("max_consecutive_quanta must be positive")
        if scoring_poll_interval_s <= 0 or not math.isfinite(scoring_poll_interval_s):
            raise ValueError("scoring_poll_interval_s must be positive and finite")

        self._scoring_pool = scoring_pool
        self._deficit_quantum_ms = float(deficit_quantum_ms)
        self._update_priority_ms = float(update_priority_ms)
        self._update_starvation_s = float(update_starvation_s)
        self._age_boost_ms_per_s = float(age_boost_ms_per_s)
        self._max_consecutive_quanta = int(max_consecutive_quanta)
        self._scoring_poll_interval_s = float(scoring_poll_interval_s)
        self._clock = clock
        self._runs: dict[str, _RunState] = {}
        self._event_queue: deque[tuple[SchedulerEvent, Any]] = deque()
        self._event_history: list[SchedulerEvent] = []
        self._cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        self._gpu_lease = asyncio.Lock()
        self._registration_counter = 0
        self._last_gpu_run_id: str | None = None
        self._consecutive_gpu_quanta = 0

    @property
    def run_ids(self) -> tuple[str, ...]:
        """return all run ids in admission order, including terminal runs."""

        return tuple(self._runs)

    @property
    def event_history(self) -> tuple[SchedulerEvent, ...]:
        """return the immutable event history in controller order."""

        return tuple(self._event_history)

    def run_snapshot(self, run_id: str) -> RunSnapshot:
        """return one run's current scheduler state."""

        state = self._require_run(run_id)
        return RunSnapshot(
            run_id=state.run_id,
            phase=state.phase,
            completed_steps=state.completed_steps,
            total_steps=state.total_steps,
            weight=state.weight,
            deficit_ms=state.deficit_ms,
            gpu_time_ms=state.gpu_time_ms,
            outstanding_identity=state.scoring_identity,
            outstanding_rollout=state.rollout is not _MISSING_ROLLOUT,
            failure=state.failure,
        )

    def add_run(
        self,
        run_id: str,
        *,
        hooks: SchedulerRunHooks,
        scoring_kind: ScoringKind,
        scoring_bridge: ScoringBridge,
        total_steps: int,
        weight: float = 1.0,
    ) -> RunSnapshot:
        """admit one run and make its first rollout ready."""

        normalized_run_id = str(run_id).strip()
        if not normalized_run_id:
            raise ValueError("run_id must not be empty")
        if normalized_run_id in self._runs:
            raise SharedSchedulerError(f"scheduler run was already admitted: {normalized_run_id}")
        if not isinstance(hooks, SchedulerRunHooks):
            raise TypeError("hooks must be SchedulerRunHooks")
        if isinstance(total_steps, bool) or total_steps < 1:
            raise ValueError("total_steps must be positive")
        if weight <= 0 or not math.isfinite(weight):
            raise ValueError("weight must be positive and finite")

        self._scoring_pool.register_run(
            normalized_run_id,
            kind=ScoringKind(scoring_kind),
            bridge=scoring_bridge,
        )
        state = _RunState(
            run_id=normalized_run_id,
            hooks=hooks,
            weight=float(weight),
            total_steps=int(total_steps),
            registration_order=self._registration_counter,
        )
        self._registration_counter += 1
        self._runs[normalized_run_id] = state
        self._enqueue(SchedulerEventKind.RUN_ADMITTED, state)
        self._drain_events_without_io()
        self._assert_outstanding_bound(state)
        return self.run_snapshot(normalized_run_id)

    async def step_the_world(self) -> StepWorldResult:
        """poll external events and execute at most one ready GPU quantum."""

        async with self._gpu_lease:
            progressed = await self._collect_scoring_completions()
            progressed = await self._submit_waiting_scores() or progressed
            progressed = await self._collect_scoring_completions() or progressed
            candidate = self._select_gpu_candidate()
            if candidate is None:
                return StepWorldResult(progressed=progressed)

            scheduled_phase = candidate.phase
            if scheduled_phase is RunPhase.READY_ROLLOUT:
                await self._execute_rollout(candidate)
            elif scheduled_phase is RunPhase.READY_UPDATE:
                await self._execute_update(candidate)
            else:
                raise RunStateTransitionError(
                    f"selected run {candidate.run_id} in non-ready phase {scheduled_phase}"
                )
            return StepWorldResult(
                progressed=True,
                gpu_run_id=candidate.run_id,
                gpu_phase=scheduled_phase,
            )

    async def drain(self, *, timeout_s: float | None = None) -> tuple[RunSnapshot, ...]:
        """advance until every admitted run reaches a terminal state."""

        if timeout_s is not None and (timeout_s <= 0 or not math.isfinite(timeout_s)):
            raise ValueError("timeout_s must be positive and finite")
        if not self._runs:
            return ()
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while not self._is_drained():
            result = await self.step_the_world()
            if self._is_drained():
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("shared scheduler drain timed out")
            if not result.progressed:
                await asyncio.sleep(self._scoring_poll_interval_s)
            else:
                await asyncio.sleep(0)
        return tuple(self.run_snapshot(run_id) for run_id in self._runs)

    async def remove_run(self, run_id: str) -> RunSnapshot:
        """cancel one run, reject its score future, and leave peers schedulable."""

        async with self._gpu_lease:
            state = self._require_run(run_id)
            if state.phase not in _TERMINAL_PHASES:
                self._transition(state, RunPhase.CANCELLED)
                self._enqueue(SchedulerEventKind.RUN_CANCELLED, state)
            if state.run_id in self._scoring_pool.registered_run_ids:
                self._scoring_pool.cancel_run(state.run_id)
            self._clear_outstanding(state)
            cleanup_task = self._start_cleanup(state)
            self._drain_events_without_io()
        if cleanup_task is not None:
            await cleanup_task
        return self.run_snapshot(state.run_id)

    async def _execute_rollout(self, state: _RunState) -> None:
        self._enqueue(SchedulerEventKind.ROLLOUT_STARTED, state)
        self._drain_events_without_io()
        try:
            result, gpu_ms = await self._run_gpu_callback(
                state.hooks.rollout,
                state.run_id,
                state.completed_steps,
            )
            payload = state.hooks.scoring_payload(
                state.run_id,
                state.completed_steps,
                result,
            )
            if not isinstance(payload, Mapping):
                raise TypeError("scoring payload callback must return a mapping")
            self._charge_gpu_time(state, gpu_ms)
            self._enqueue(
                SchedulerEventKind.ROLLOUT_COMPLETED,
                state,
                (result, payload),
            )
            self._drain_events_without_io()
            await self._submit_waiting_scores()
        except asyncio.CancelledError as exc:
            await asyncio.shield(self._fail_run(state, exc))
            raise
        except Exception as exc:
            await self._fail_run(state, exc)

    async def _execute_update(self, state: _RunState) -> None:
        self._enqueue(SchedulerEventKind.UPDATE_STARTED, state)
        self._drain_events_without_io()
        try:
            if state.rollout is _MISSING_ROLLOUT or state.scoring_result is None:
                raise RunStateTransitionError(
                    f"run {state.run_id} has no complete rollout and score for update"
                )
            _value, gpu_ms = await self._run_gpu_callback(
                state.hooks.update_and_publish,
                state.run_id,
                state.completed_steps,
                state.rollout,
                state.scoring_result,
            )
            self._charge_gpu_time(state, gpu_ms)
            self._enqueue(SchedulerEventKind.UPDATE_COMPLETED, state)
            self._drain_events_without_io()
        except asyncio.CancelledError as exc:
            await asyncio.shield(self._fail_run(state, exc))
            raise
        except Exception as exc:
            await self._fail_run(state, exc)

    async def _submit_waiting_scores(self) -> bool:
        progressed = False
        waiting = sorted(
            (
                state
                for state in self._runs.values()
                if state.phase is RunPhase.SCORE_PENDING
                and state.scoring_future is None
                and state.scoring_payload is not None
            ),
            key=lambda state: state.registration_order,
        )
        for state in waiting:
            identity = ScoringBatchIdentity(
                state.run_id,
                state.completed_steps,
                f"{state.run_id}-step-{state.completed_steps}",
            )
            try:
                future = self._scoring_pool.submit(identity, state.scoring_payload)
            except ScoringCapacityError:
                break
            except Exception as exc:
                await self._fail_run(state, exc)
                progressed = True
                continue
            state.scoring_identity = identity
            state.scoring_future = future
            self._enqueue(SchedulerEventKind.SCORING_SUBMITTED, state)
            progressed = True
        self._drain_events_without_io()
        return progressed

    async def _collect_scoring_completions(self) -> bool:
        progressed = False
        pending_identities = frozenset(self._scoring_pool.pending_identities)
        for state in tuple(self._runs.values()):
            identity = state.scoring_identity
            future = state.scoring_future
            if state.phase is not RunPhase.SCORE_PENDING or identity is None or future is None:
                continue
            if identity not in pending_identities:
                await self._fail_run(
                    state,
                    ScoringIdentityError(
                        "scheduler scoring identity disappeared before exact consumption"
                    ),
                )
                progressed = True
                continue
            if not future.done():
                continue
            try:
                result = self._scoring_pool.consume(identity, future, timeout=0)
            except Exception as exc:
                await self._fail_run(state, exc)
            else:
                self._enqueue(SchedulerEventKind.SCORING_COMPLETED, state, result)
                self._drain_events_without_io()
            progressed = True
        return progressed

    def _select_gpu_candidate(self) -> _RunState | None:
        ready = [
            state
            for state in self._runs.values()
            if state.phase in {RunPhase.READY_ROLLOUT, RunPhase.READY_UPDATE}
        ]
        if not ready:
            return None

        total_weight = sum(state.weight for state in ready)
        for state in ready:
            state.deficit_ms += self._deficit_quantum_ms * state.weight / total_weight

        now = self._clock()
        aged_updates = [
            state
            for state in ready
            if state.phase is RunPhase.READY_UPDATE
            and state.ready_since is not None
            and now - state.ready_since >= self._update_starvation_s
        ]
        candidates = aged_updates or ready
        if (
            self._last_gpu_run_id is not None
            and self._consecutive_gpu_quanta >= self._max_consecutive_quanta
        ):
            alternatives = [state for state in candidates if state.run_id != self._last_gpu_run_id]
            if alternatives:
                candidates = alternatives

        def priority(state: _RunState) -> tuple[float, float, int]:
            ready_since = state.ready_since if state.ready_since is not None else now
            age_s = max(0.0, now - ready_since)
            phase_bonus = self._update_priority_ms if state.phase is RunPhase.READY_UPDATE else 0.0
            return (
                state.deficit_ms + phase_bonus + age_s * self._age_boost_ms_per_s,
                age_s,
                -state.registration_order,
            )

        return max(candidates, key=priority)

    async def _run_gpu_callback(
        self,
        callback: Callable[..., Awaitable[Any] | Any],
        *args: Any,
    ) -> tuple[Any, float]:
        started = self._clock()
        result = await _resolve_callback(callback(*args))
        elapsed_ms = max(0.001, (self._clock() - started) * 1000.0)
        if isinstance(result, GpuWorkResult):
            if result.gpu_ms is None:
                return result.value, elapsed_ms
            if result.gpu_ms <= 0 or not math.isfinite(result.gpu_ms):
                raise ValueError("gpu_ms must be positive and finite")
            return result.value, float(result.gpu_ms)
        return result, elapsed_ms

    def _charge_gpu_time(self, state: _RunState, gpu_ms: float) -> None:
        state.deficit_ms -= gpu_ms
        state.gpu_time_ms += gpu_ms
        if self._last_gpu_run_id == state.run_id:
            self._consecutive_gpu_quanta += 1
        else:
            self._last_gpu_run_id = state.run_id
            self._consecutive_gpu_quanta = 1

    async def _fail_run(self, state: _RunState, error: BaseException) -> None:
        if state.phase in _TERMINAL_PHASES:
            return
        state.failure = error
        self._transition(state, RunPhase.FAILED)
        self._enqueue(SchedulerEventKind.RUN_FAILED, state)
        if state.run_id in self._scoring_pool.registered_run_ids:
            self._scoring_pool.cancel_run(state.run_id)
        self._clear_outstanding(state)
        self._start_cleanup(state)
        self._drain_events_without_io()

    def _start_cleanup(self, state: _RunState) -> asyncio.Task[None] | None:
        if state.cleaned_up:
            return None
        existing = self._cleanup_tasks.get(state.run_id)
        if existing is not None:
            return existing
        if state.hooks.cleanup is None:
            state.cleaned_up = True
            return None
        task = asyncio.create_task(self._cleanup_run(state))
        self._cleanup_tasks[state.run_id] = task
        task.add_done_callback(
            lambda completed, run_id=state.run_id: self._finish_cleanup(run_id, completed)
        )
        return task

    def _finish_cleanup(self, run_id: str, task: asyncio.Task[None]) -> None:
        if self._cleanup_tasks.get(run_id) is task:
            self._cleanup_tasks.pop(run_id)
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        state = self._runs[run_id]
        if state.failure is None:
            state.failure = error
        else:
            state.failure = BaseExceptionGroup(
                f"run {run_id} failed and cleanup also failed",
                [state.failure, error],
            )

    async def _cleanup_run(self, state: _RunState) -> None:
        if state.cleaned_up or state.hooks.cleanup is None:
            state.cleaned_up = True
            return
        await _resolve_callback(state.hooks.cleanup(state.run_id))
        state.cleaned_up = True

    def _enqueue(
        self,
        kind: SchedulerEventKind,
        state: _RunState,
        payload: Any = None,
    ) -> None:
        event = SchedulerEvent(
            kind=kind,
            run_id=state.run_id,
            step=state.completed_steps,
            timestamp=self._clock(),
        )
        self._event_queue.append((event, payload))
        self._event_history.append(event)

    def _drain_events_without_io(self) -> None:
        while self._event_queue:
            event, payload = self._event_queue.popleft()
            state = self._runs[event.run_id]
            if event.kind is SchedulerEventKind.RUN_ADMITTED:
                self._transition(state, RunPhase.READY_ROLLOUT)
            elif event.kind is SchedulerEventKind.ROLLOUT_STARTED:
                self._transition(state, RunPhase.ROLLOUT_IN_FLIGHT)
            elif event.kind is SchedulerEventKind.ROLLOUT_COMPLETED:
                rollout, scoring_payload = payload
                state.rollout = rollout
                state.scoring_payload = scoring_payload
                self._transition(state, RunPhase.SCORE_PENDING)
            elif event.kind is SchedulerEventKind.SCORING_COMPLETED:
                result = payload
                if (
                    state.scoring_identity is None
                    or result.identity != state.scoring_identity
                    or result.identity.run_id != state.run_id
                    or result.identity.step != state.completed_steps
                ):
                    raise ScoringIdentityError(
                        "scoring completion does not match the scheduler-owned run step"
                    )
                state.scoring_result = result
                state.scoring_future = None
                state.scoring_identity = None
                state.scoring_payload = None
                self._transition(state, RunPhase.READY_UPDATE)
            elif event.kind is SchedulerEventKind.UPDATE_STARTED:
                self._transition(state, RunPhase.UPDATE_IN_FLIGHT)
            elif event.kind is SchedulerEventKind.UPDATE_COMPLETED:
                self._transition(state, RunPhase.SYNCING)
                state.completed_steps += 1
                self._clear_outstanding(state)
                self._enqueue(SchedulerEventKind.SYNC_COMPLETED, state)
            elif event.kind is SchedulerEventKind.SYNC_COMPLETED:
                if state.completed_steps >= state.total_steps:
                    self._transition(state, RunPhase.FINISHING)
                    self._enqueue(SchedulerEventKind.RUN_DONE, state)
                else:
                    self._transition(state, RunPhase.READY_ROLLOUT)
            elif event.kind is SchedulerEventKind.RUN_DONE:
                self._transition(state, RunPhase.DONE)
            elif event.kind in {
                SchedulerEventKind.SCORING_SUBMITTED,
                SchedulerEventKind.RUN_FAILED,
                SchedulerEventKind.RUN_CANCELLED,
            }:
                pass
            self._assert_outstanding_bound(state)

    def _transition(self, state: _RunState, target: RunPhase) -> None:
        if target not in _ALLOWED_TRANSITIONS[state.phase]:
            raise RunStateTransitionError(
                f"invalid scheduler transition for {state.run_id}: {state.phase} -> {target}"
            )
        state.phase = target
        state.ready_since = (
            self._clock() if target in {RunPhase.READY_ROLLOUT, RunPhase.READY_UPDATE} else None
        )

    @staticmethod
    def _clear_outstanding(state: _RunState) -> None:
        state.rollout = _MISSING_ROLLOUT
        state.scoring_payload = None
        state.scoring_identity = None
        state.scoring_future = None
        state.scoring_result = None

    @staticmethod
    def _assert_outstanding_bound(state: _RunState) -> None:
        if state.phase is RunPhase.READY_ROLLOUT and (
            state.rollout is not _MISSING_ROLLOUT
            or any(
                value is not None
                for value in (
                    state.scoring_payload,
                    state.scoring_identity,
                    state.scoring_future,
                    state.scoring_result,
                )
            )
        ):
            raise RunStateTransitionError(
                f"run {state.run_id} retained an outstanding rollout while ready to generate"
            )
        if state.scoring_future is not None and state.rollout is _MISSING_ROLLOUT:
            raise RunStateTransitionError(
                f"run {state.run_id} has a scoring future without its originating rollout"
            )
        if state.scoring_result is not None and state.rollout is _MISSING_ROLLOUT:
            raise RunStateTransitionError(
                f"run {state.run_id} has a scoring result without its originating rollout"
            )

    def _require_run(self, run_id: str) -> _RunState:
        normalized_run_id = str(run_id).strip()
        try:
            return self._runs[normalized_run_id]
        except KeyError as exc:
            raise SharedSchedulerError(f"unknown scheduler run: {normalized_run_id}") from exc

    def _all_terminal(self) -> bool:
        return bool(self._runs) and all(
            state.phase in _TERMINAL_PHASES for state in self._runs.values()
        )

    def _is_drained(self) -> bool:
        return self._all_terminal() and not self._cleanup_tasks
