"""cpu-only tests for the shared OpenRLHF event-driven scheduler."""

from __future__ import annotations

import asyncio
import threading
from collections import Counter
from typing import Any

import pytest

from flash.engine.worker.openrlhf_shared_scheduler import (
    GpuWorkResult,
    RunPhase,
    SchedulerEventKind,
    SchedulerRunHooks,
    SharedEngineRunController,
)
from flash.engine.worker.openrlhf_shared_scoring import (
    ScoringKind,
    SharedScoringPool,
)


async def _advance_gpu_quanta(controller: SharedEngineRunController, count: int) -> None:
    completed = 0
    attempts = 0
    while completed < count:
        result = await controller.step_the_world()
        if result.gpu_run_id is not None:
            completed += 1
        await asyncio.sleep(0.001)
        attempts += 1
        if attempts > count * 100:
            raise AssertionError("scheduler did not produce the expected GPU quanta")


def _hooks(
    events: list[str],
    *,
    rollout_ms: float = 1.0,
    update_ms: float = 1.0,
    update_error: BaseException | None = None,
) -> SchedulerRunHooks:
    def rollout(run_id: str, step: int) -> GpuWorkResult:
        events.append(f"{run_id}:rollout:{step}")
        return GpuWorkResult({"run_id": run_id, "step": step}, rollout_ms)

    def scoring_payload(run_id: str, step: int, rollout_value: Any) -> dict[str, Any]:
        assert rollout_value == {"run_id": run_id, "step": step}
        return {"run_id": run_id, "step": step}

    def update(run_id: str, step: int, rollout_value: Any, scoring_result) -> GpuWorkResult:
        events.append(f"{run_id}:update:{step}")
        assert rollout_value == {"run_id": run_id, "step": step}
        assert scoring_result.identity.run_id == run_id
        assert scoring_result.identity.step == step
        assert scoring_result.value == {"scored_run": run_id, "step": step}
        if update_error is not None:
            raise update_error
        return GpuWorkResult(None, update_ms)

    return SchedulerRunHooks(
        rollout=rollout,
        scoring_payload=scoring_payload,
        update_and_publish=update,
    )


def _immediate_score(payload: dict[str, Any]) -> dict[str, Any]:
    return {"scored_run": payload["run_id"], "step": payload["step"]}


def test_slow_scoring_run_yields_gpu_to_another_runs_rollout_and_update():
    async def scenario():
        score_a_started = threading.Event()
        release_score_a = threading.Event()
        events: list[str] = []

        def score_a(payload):
            events.append("run-a:score:start")
            score_a_started.set()
            assert release_score_a.wait(timeout=2)
            events.append("run-a:score:done")
            return {"scored_run": "run-a", "step": payload["step"]}

        with SharedScoringPool(pool_size=2) as scoring_pool:
            controller = SharedEngineRunController(
                scoring_pool,
                deficit_quantum_ms=1,
                update_priority_ms=0.25,
            )
            controller.add_run(
                "run-a",
                hooks=_hooks(events),
                scoring_kind=ScoringKind.REWARD,
                scoring_bridge=score_a,
                total_steps=1,
            )
            controller.add_run(
                "run-b",
                hooks=_hooks(events),
                scoring_kind=ScoringKind.REWARD,
                scoring_bridge=_immediate_score,
                total_steps=1,
            )

            first = await controller.step_the_world()
            assert first.gpu_run_id == "run-a"
            assert first.gpu_phase is RunPhase.READY_ROLLOUT
            assert score_a_started.wait(timeout=1)

            second = await controller.step_the_world()
            assert second.gpu_run_id == "run-b"
            assert second.gpu_phase is RunPhase.READY_ROLLOUT
            await asyncio.sleep(0.01)
            third = await controller.step_the_world()
            assert third.gpu_run_id == "run-b"
            assert third.gpu_phase is RunPhase.READY_UPDATE
            assert release_score_a.is_set() is False

            release_score_a.set()
            snapshots = await controller.drain(timeout_s=2)

        assert [snapshot.phase for snapshot in snapshots] == [RunPhase.DONE, RunPhase.DONE]
        assert events.index("run-a:score:start") < events.index("run-b:rollout:0")
        assert events.index("run-b:rollout:0") < events.index("run-a:score:done")
        assert events.index("run-b:update:0") < events.index("run-a:update:0")

    asyncio.run(scenario())


def test_weighted_deficit_round_robin_shares_measured_gpu_time_without_starvation():
    async def scenario():
        events: list[str] = []
        with SharedScoringPool(pool_size=2) as scoring_pool:
            controller = SharedEngineRunController(
                scoring_pool,
                deficit_quantum_ms=1,
                update_priority_ms=0.1,
                max_consecutive_quanta=4,
            )
            controller.add_run(
                "run-one",
                hooks=_hooks(events),
                scoring_kind=ScoringKind.REWARD,
                scoring_bridge=_immediate_score,
                total_steps=1000,
                weight=1,
            )
            controller.add_run(
                "run-two",
                hooks=_hooks(events),
                scoring_kind=ScoringKind.REWARD,
                scoring_bridge=_immediate_score,
                total_steps=1000,
                weight=2,
            )

            await _advance_gpu_quanta(controller, 600)
            one = controller.run_snapshot("run-one")
            two = controller.run_snapshot("run-two")

        assert one.gpu_time_ms > 0
        assert two.gpu_time_ms > 0
        assert two.gpu_time_ms / one.gpu_time_ms == pytest.approx(2.0, rel=0.12)
        gpu_events = Counter(event.split(":", 1)[0] for event in events)
        assert gpu_events["run-one"] > 0
        assert gpu_events["run-two"] > 0

    asyncio.run(scenario())


def test_aged_update_closes_rollout_admission_until_it_runs():
    class _Clock:
        def __init__(self) -> None:
            self.value = 0.0

        def __call__(self) -> float:
            return self.value

        def advance_ms(self, milliseconds: float) -> None:
            self.value += milliseconds / 1000.0

    async def scenario():
        clock = _Clock()
        events: list[str] = []

        def timed_hooks(run_id: str, rollout_ms: float, update_ms: float) -> SchedulerRunHooks:
            def rollout(_run_id: str, step: int) -> GpuWorkResult:
                events.append(f"{run_id}:rollout:{step}")
                clock.advance_ms(rollout_ms)
                return GpuWorkResult({"run_id": run_id, "step": step}, rollout_ms)

            def payload(_run_id: str, step: int, _rollout: Any) -> dict[str, Any]:
                return {"run_id": run_id, "step": step}

            def update(_run_id: str, step: int, _rollout: Any, score) -> GpuWorkResult:
                assert score.identity.run_id == run_id
                events.append(f"{run_id}:update:{step}")
                clock.advance_ms(update_ms)
                return GpuWorkResult(None, update_ms)

            return SchedulerRunHooks(rollout, payload, update)

        with SharedScoringPool(pool_size=2) as scoring_pool:
            controller = SharedEngineRunController(
                scoring_pool,
                clock=clock,
                deficit_quantum_ms=1,
                update_priority_ms=0.1,
                update_starvation_s=0.005,
                age_boost_ms_per_s=0,
                max_consecutive_quanta=100,
            )
            controller.add_run(
                "aged",
                hooks=timed_hooks("aged", 100, 1),
                scoring_kind=ScoringKind.REWARD,
                scoring_bridge=_immediate_score,
                total_steps=2,
                weight=0.01,
            )
            await controller.step_the_world()
            controller.add_run(
                "stream",
                hooks=timed_hooks("stream", 1, 1),
                scoring_kind=ScoringKind.REWARD,
                scoring_bridge=_immediate_score,
                total_steps=100,
                weight=100,
            )

            await _advance_gpu_quanta(controller, 20)

        aged_update_index = events.index("aged:update:0")
        assert any(event.startswith("stream:") for event in events[1:aged_update_index])
        assert clock.value >= 0.105
        assert controller.run_snapshot("aged").completed_steps >= 1

    asyncio.run(scenario())


def test_results_remain_identity_bound_and_one_run_failure_does_not_block_peer():
    async def scenario():
        events: list[str] = []
        expected_failure = RuntimeError("run-a update crashed")
        cleaned: list[str] = []
        run_a_hooks = _hooks(events, update_error=expected_failure)
        run_a_hooks = SchedulerRunHooks(
            rollout=run_a_hooks.rollout,
            scoring_payload=run_a_hooks.scoring_payload,
            update_and_publish=run_a_hooks.update_and_publish,
            cleanup=lambda run_id: cleaned.append(run_id),
        )

        with SharedScoringPool(pool_size=2) as scoring_pool:
            controller = SharedEngineRunController(scoring_pool, deficit_quantum_ms=1)
            controller.add_run(
                "run-a",
                hooks=run_a_hooks,
                scoring_kind=ScoringKind.REWARD,
                scoring_bridge=_immediate_score,
                total_steps=1,
            )
            controller.add_run(
                "run-b",
                hooks=_hooks(events),
                scoring_kind=ScoringKind.TEACHER,
                scoring_bridge=_immediate_score,
                total_steps=2,
            )

            snapshots = await controller.drain(timeout_s=2)

        run_a, run_b = snapshots
        assert run_a.phase is RunPhase.FAILED
        assert run_a.failure is expected_failure
        assert run_b.phase is RunPhase.DONE
        assert run_b.completed_steps == 2
        assert cleaned == ["run-a"]
        assert "run-b:update:1" in events

    asyncio.run(scenario())


def test_removed_run_and_scoring_capacity_wait_do_not_block_other_runs():
    async def scenario():
        started = threading.Event()
        release = threading.Event()
        events: list[str] = []

        def blocked_score(payload):
            started.set()
            assert release.wait(timeout=2)
            return {"scored_run": payload["run_id"], "step": payload["step"]}

        scoring_pool = SharedScoringPool(pool_size=1)
        controller = SharedEngineRunController(scoring_pool, deficit_quantum_ms=1)
        controller.add_run(
            "remove-me",
            hooks=_hooks(events),
            scoring_kind=ScoringKind.REWARD,
            scoring_bridge=blocked_score,
            total_steps=2,
        )
        controller.add_run(
            "survivor",
            hooks=_hooks(events),
            scoring_kind=ScoringKind.REWARD,
            scoring_bridge=_immediate_score,
            total_steps=1,
        )

        try:
            await controller.step_the_world()
            assert started.wait(timeout=1)
            await controller.step_the_world()
            survivor = controller.run_snapshot("survivor")
            assert survivor.phase is RunPhase.SCORE_PENDING
            assert survivor.outstanding_identity is None
            assert survivor.outstanding_rollout is True

            removed = await controller.remove_run("remove-me")
            assert removed.phase is RunPhase.CANCELLED
            release.set()
            snapshots = await controller.drain(timeout_s=2)
        finally:
            release.set()
            scoring_pool.shutdown()

        assert snapshots[0].phase is RunPhase.CANCELLED
        assert snapshots[1].phase is RunPhase.DONE
        assert "survivor:update:0" in events

    asyncio.run(scenario())


def test_one_outstanding_rollout_per_run_and_explicit_state_events():
    async def scenario():
        started = threading.Event()
        release = threading.Event()
        events: list[str] = []

        def score(payload):
            started.set()
            assert release.wait(timeout=2)
            return {"scored_run": payload["run_id"], "step": payload["step"]}

        with SharedScoringPool(pool_size=1) as scoring_pool:
            controller = SharedEngineRunController(scoring_pool, deficit_quantum_ms=1)
            controller.add_run(
                "run-a",
                hooks=_hooks(events),
                scoring_kind=ScoringKind.REWARD,
                scoring_bridge=score,
                total_steps=1,
            )
            await controller.step_the_world()
            assert started.wait(timeout=1)

            for _ in range(20):
                result = await controller.step_the_world()
                assert result.gpu_run_id is None
            assert events == ["run-a:rollout:0"]
            snapshot = controller.run_snapshot("run-a")
            assert snapshot.phase is RunPhase.SCORE_PENDING
            assert snapshot.outstanding_rollout is True

            release.set()
            final = (await controller.drain(timeout_s=2))[0]

        assert final.phase is RunPhase.DONE
        kinds = [event.kind for event in controller.event_history]
        assert kinds == [
            SchedulerEventKind.RUN_ADMITTED,
            SchedulerEventKind.ROLLOUT_STARTED,
            SchedulerEventKind.ROLLOUT_COMPLETED,
            SchedulerEventKind.SCORING_SUBMITTED,
            SchedulerEventKind.SCORING_COMPLETED,
            SchedulerEventKind.UPDATE_STARTED,
            SchedulerEventKind.UPDATE_COMPLETED,
            SchedulerEventKind.SYNC_COMPLETED,
            SchedulerEventKind.RUN_DONE,
        ]

    asyncio.run(scenario())
