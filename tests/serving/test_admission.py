"""Per-model hosted serving admission control."""

from __future__ import annotations

import asyncio

import pytest

from flash.serving.src.engine.model_config import HostedTrafficPolicy
from flash.serving.src.traffic.admission import (
    AdmissionController,
    ServingCapacityUnavailable,
    ServingOverloaded,
)

_MODEL_A = "model-a"
_MODEL_B = "model-b"


def _policy(_model: str) -> HostedTrafficPolicy:
    return HostedTrafficPolicy.from_engine({"max_num_seqs": 8})


class _Limits:
    def __init__(self, **limits: int) -> None:
        self.values = limits

    def __call__(self, model: str) -> int:
        return self.values.get(model, 0)


class _Clock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now


def test_immediate_acquire_has_zero_queue_and_hard_limit() -> None:
    async def run() -> None:
        controller = AdmissionController(_policy, _Limits(**{_MODEL_A: 1}))
        lease = await controller.acquire(_MODEL_A)

        assert lease.queue_duration_seconds == 0.0
        snapshot = controller.snapshot(_MODEL_A)
        assert snapshot.active == 1
        assert snapshot.queued == 0
        assert snapshot.current_dispatch_limit == 1
        assert snapshot.hard_limit == 16

        lease.release()
        assert controller.snapshot(_MODEL_A).active == 0
        assert controller.active_count(_MODEL_A) == 0

    asyncio.run(run())


def test_fifo_queue_is_bounded_to_two_waiters() -> None:
    async def run() -> None:
        controller = AdmissionController(_policy, _Limits(**{_MODEL_A: 1}))
        active = await controller.acquire(_MODEL_A)
        first = asyncio.create_task(controller.acquire(_MODEL_A))
        second = asyncio.create_task(controller.acquire(_MODEL_A))
        await asyncio.sleep(0)

        assert controller.snapshot(_MODEL_A).queued == 2
        with pytest.raises(ServingOverloaded) as exc_info:
            await controller.acquire(_MODEL_A)
        assert exc_info.value.code == "serving_overloaded"
        assert exc_info.value.retry_after_seconds == 1
        assert exc_info.value.model == _MODEL_A

        active.release()
        await asyncio.sleep(0)
        assert first.done()
        assert not second.done()

        first_lease = await first
        first_lease.release()
        await asyncio.sleep(0)
        assert second.done()
        second_lease = await second
        second_lease.release()
        assert controller.snapshot(_MODEL_A).active == 0

    asyncio.run(run())


def test_models_have_isolated_active_and_queue_counts() -> None:
    async def run() -> None:
        limits = _Limits(**{_MODEL_A: 1, _MODEL_B: 1})
        controller = AdmissionController(_policy, limits)
        active_a = await controller.acquire(_MODEL_A)
        waiting_a = asyncio.create_task(controller.acquire(_MODEL_A))
        active_b = await controller.acquire(_MODEL_B)
        await asyncio.sleep(0)

        assert controller.snapshot(_MODEL_A).active == 1
        assert controller.snapshot(_MODEL_A).queued == 1
        assert controller.snapshot(_MODEL_B).active == 1
        assert controller.snapshot(_MODEL_B).queued == 0

        active_b.release()
        assert controller.snapshot(_MODEL_A).queued == 1
        active_a.release()
        waiting_lease = await waiting_a
        waiting_lease.release()

    asyncio.run(run())


def test_cancelled_waiter_removes_itself_without_changing_active_count() -> None:
    async def run() -> None:
        clock = _Clock()
        controller = AdmissionController(_policy, _Limits(**{_MODEL_A: 1}), clock=clock)
        active = await controller.acquire(_MODEL_A)
        waiting = asyncio.create_task(controller.acquire(_MODEL_A))
        await asyncio.sleep(0)
        clock.now = 13.5

        snapshot = controller.snapshot(_MODEL_A)
        assert snapshot.queued == 1
        assert snapshot.oldest_wait_seconds == 3.5

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        snapshot = controller.snapshot(_MODEL_A)
        assert snapshot.active == 1
        assert snapshot.queued == 0

        active.release()
        assert controller.snapshot(_MODEL_A).active == 0

    asyncio.run(run())


def test_release_is_idempotent_and_counts_never_go_negative() -> None:
    async def run() -> None:
        controller = AdmissionController(_policy, _Limits(**{_MODEL_A: 1}))
        lease = await controller.acquire(_MODEL_A)

        lease.release()
        lease.release()

        assert lease.released is True
        assert controller.snapshot(_MODEL_A).active == 0
        assert controller.snapshot(_MODEL_A).queued == 0

    asyncio.run(run())


def test_cancellation_after_promotion_returns_reserved_active_slot() -> None:
    async def run() -> None:
        controller = AdmissionController(_policy, _Limits(**{_MODEL_A: 1}))
        active = await controller.acquire(_MODEL_A)
        waiting = asyncio.create_task(controller.acquire(_MODEL_A))
        await asyncio.sleep(0)

        active.release()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting

        assert controller.snapshot(_MODEL_A).active == 0
        assert controller.snapshot(_MODEL_A).queued == 0

    asyncio.run(run())


def test_capacity_changed_allows_future_acquire_when_dispatch_limit_increases() -> None:
    async def run() -> None:
        limits = _Limits(**{_MODEL_A: 0})
        controller = AdmissionController(_policy, limits)

        with pytest.raises(ServingCapacityUnavailable):
            await controller.acquire(_MODEL_A)

        limits.values[_MODEL_A] = 1
        controller.capacity_changed(_MODEL_A)
        lease = await controller.acquire(_MODEL_A)

        assert lease.queue_duration_seconds == 0.0
        assert controller.snapshot(_MODEL_A).active == 1
        assert controller.snapshot(_MODEL_A).queued == 0
        lease.release()

    asyncio.run(run())


def test_zero_capacity_fails_closed_immediately_without_queueing() -> None:
    async def run() -> None:
        controller = AdmissionController(_policy, _Limits(**{_MODEL_A: 0}))

        with pytest.raises(ServingCapacityUnavailable) as exc_info:
            await controller.acquire(_MODEL_A)
        assert exc_info.value.code == "serving_capacity_unavailable"
        assert exc_info.value.retry_after_seconds == 1
        assert exc_info.value.model == _MODEL_A

        snapshot = controller.snapshot(_MODEL_A)
        assert snapshot.active == 0
        assert snapshot.queued == 0
        assert snapshot.current_dispatch_limit == 0

    asyncio.run(run())


def test_zero_capacity_transition_fails_waiters_preserves_active_and_recovers() -> None:
    async def run() -> None:
        limits = _Limits(**{_MODEL_A: 1})
        controller = AdmissionController(_policy, limits)
        active = await controller.acquire(_MODEL_A)
        waiters = [asyncio.create_task(controller.acquire(_MODEL_A)) for _ in range(2)]
        await asyncio.sleep(0)
        assert controller.snapshot(_MODEL_A).queued == 2

        limits.values[_MODEL_A] = 0
        controller.capacity_changed(_MODEL_A)
        results = await asyncio.gather(*waiters, return_exceptions=True)

        assert all(isinstance(result, ServingCapacityUnavailable) for result in results)
        assert controller.snapshot(_MODEL_A).active == 1
        assert controller.snapshot(_MODEL_A).queued == 0

        limits.values[_MODEL_A] = 1
        controller.capacity_changed(_MODEL_A)
        active.release()
        recovered = await controller.acquire(_MODEL_A)
        recovered.release()
        assert controller.snapshot(_MODEL_A).active == 0

    asyncio.run(run())


def test_dispatch_limit_is_capped_by_policy_hard_limit() -> None:
    async def run() -> None:
        controller = AdmissionController(_policy, _Limits(**{_MODEL_A: 1000}))
        assert controller.snapshot(_MODEL_A).current_dispatch_limit == 16
        assert controller.snapshot(_MODEL_A).hard_limit == 16

    asyncio.run(run())
