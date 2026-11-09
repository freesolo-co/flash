"""Capacity shortages wait without retrying ambiguous VM allocations."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from flash.serving.lambda_cloud import capacity
from flash.serving.lambda_cloud.config import NoCapacityError


class Clock:
    def __init__(self):
        self.now = 0
        self.waits = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def test_capacity_returning_after_two_polls_launches_once():
    clock = Clock()
    launch = AsyncMock(side_effect=[NoCapacityError(), NoCapacityError(), {"id": "vm"}])
    result = asyncio.run(
        capacity.launch_with_capacity_wait(launch, "owner", clock=clock, sleep=clock.sleep)
    )
    assert result == {"id": "vm"}
    assert clock.waits == [300, 300]
    assert all(call.args == ("owner",) for call in launch.call_args_list)


def test_no_capacity_stops_at_one_hour_without_a_late_launch():
    clock = Clock()
    attempts = []

    async def launch():
        attempts.append(clock.now)
        raise NoCapacityError

    with pytest.raises(TimeoutError, match="one hour"):
        asyncio.run(capacity.launch_with_capacity_wait(launch, clock=clock, sleep=clock.sleep))
    assert attempts == list(range(0, 3600, 300))
    assert clock.now == 3600


@pytest.mark.parametrize("error", [httpx.ReadTimeout("ambiguous launch"), RuntimeError("auth")])
def test_non_capacity_errors_are_not_retried(error):
    clock = Clock()
    launch = AsyncMock(side_effect=error)
    with pytest.raises(type(error)):
        asyncio.run(capacity.launch_with_capacity_wait(launch, clock=clock, sleep=clock.sleep))
    assert launch.await_count == 1
    assert clock.waits == []


def test_cancellation_during_capacity_wait_stops_retrying():
    clock = Clock()
    launch = AsyncMock(side_effect=NoCapacityError())
    sleep = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(capacity.launch_with_capacity_wait(launch, clock=clock, sleep=sleep))
    assert launch.await_count == 1


def test_deadline_also_bounds_a_stalled_remote_call(monkeypatch):
    monkeypatch.setattr(capacity, "CAPACITY_WAIT_SECONDS", 0.01)

    async def launch():
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        asyncio.run(capacity.launch_with_capacity_wait(launch))
