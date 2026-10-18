from __future__ import annotations

import asyncio

from flash.serving.loadtest.schedule import FakeClock, build_schedule
from flash.serving.loadtest.schema import BaseTarget, RequestProfile, Scenario
from tests.serving.test_loadtest_schema import scenario_payload


def _scenario() -> Scenario:
    payload = scenario_payload()
    payload["targets"] = [
        {"name": "target-a", "kind": "base_model", "model": "model-a", "weight": 1.0},
        {"name": "target-b", "kind": "base_model", "model": "model-b", "weight": 3.0},
    ]
    payload["discovery"] = {"enabled": False}
    payload["profiles"].append(
        {
            "name": "long",
            "weight": 2.0,
            "messages": [{"role": "user", "content": "long"}],
            "max_tokens": 16,
        }
    )
    return Scenario.model_validate(payload)


def test_schedule_is_deterministic_for_seed() -> None:
    scenario = _scenario()
    first = build_schedule(scenario.phases, scenario.targets, scenario.profiles, seed=11)
    second = build_schedule(scenario.phases, scenario.targets, scenario.profiles, seed=11)

    def identity(phases):
        return [
            (item.request_id, item.scheduled_offset_ns, item.target.name, item.profile.name)
            for phase in phases
            for item in phase
        ]

    assert identity(first) == identity(second)


def test_open_loop_offsets_are_precomputed_and_warm_is_closed_loop() -> None:
    scenario = _scenario()
    schedules = build_schedule(scenario.phases, scenario.targets, scenario.profiles, seed=1)
    cold, warm, sustained, mixed, overload = schedules
    assert [item.scheduled_offset_ns for item in cold] == [0, 100_000_000]
    assert all(item.open_loop for item in cold + sustained + mixed + overload)
    assert all(not item.open_loop for item in warm)
    assert [item.scheduled_offset_ns for item in sustained] == [0, 500_000_000]
    assert [item.stage_index for item in overload] == [0, 0, 0]


def test_fake_clock_uses_monotonic_nanoseconds() -> None:
    clock = FakeClock(start_ns=10)
    asyncio.run(clock.sleep_until_ns(100))
    assert clock.monotonic_ns() == 100
    clock.advance_ns(25)
    assert clock.monotonic_ns() == 125


def test_weighted_schedule_respects_phase_selectors() -> None:
    phase = (
        _scenario()
        .phases[1]
        .model_copy(update={"target_names": ["target-a"], "profile_names": ["short"]})
    )
    schedules = build_schedule(
        [phase],
        [
            BaseTarget(name="target-a", model="model-a"),
            BaseTarget(name="target-b", model="model-b"),
        ],
        [
            RequestProfile(
                name="short",
                messages=[{"role": "user", "content": "hi"}],
            ),
            RequestProfile(
                name="long",
                messages=[{"role": "user", "content": "hello"}],
            ),
        ],
        seed=3,
    )[0]
    assert {item.target.name for item in schedules} == {"target-a"}
    assert {item.profile.name for item in schedules} == {"short"}
