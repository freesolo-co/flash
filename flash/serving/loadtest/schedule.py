"""deterministic request scheduling for authored load-test phases."""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass
from typing import Protocol

from flash.serving.loadtest.schema import (
    ColdBurstPhase,
    MixedPhase,
    OverloadPhase,
    Phase,
    RequestProfile,
    SustainedPhase,
    Target,
    WarmPhase,
)

_NANOSECONDS_PER_SECOND = 1_000_000_000


class Clock(Protocol):
    def monotonic_ns(self) -> int: ...

    async def sleep_until_ns(self, deadline_ns: int) -> None: ...


class SystemClock:
    def monotonic_ns(self) -> int:
        return time.monotonic_ns()

    async def sleep_until_ns(self, deadline_ns: int) -> None:
        remaining = (deadline_ns - self.monotonic_ns()) / _NANOSECONDS_PER_SECOND
        if remaining > 0:
            await asyncio.sleep(remaining)


class FakeClock:
    """manually advancing monotonic clock for deterministic unit tests."""

    def __init__(self, start_ns: int = 0) -> None:
        self._now_ns = start_ns

    def monotonic_ns(self) -> int:
        return self._now_ns

    async def sleep_until_ns(self, deadline_ns: int) -> None:
        self._now_ns = max(self._now_ns, deadline_ns)
        await asyncio.sleep(0)

    def advance_ns(self, value: int) -> None:
        if value < 0:
            raise ValueError("clock advance must be non-negative")
        self._now_ns += value


@dataclass(frozen=True)
class ScheduledRequest:
    request_id: str
    phase_index: int
    phase_name: str
    phase_kind: str
    phase_request_index: int
    scheduled_offset_ns: int
    target: Target
    profile: RequestProfile
    open_loop: bool
    stage_index: int | None = None


def phase_duration_seconds(phase: Phase) -> float | None:
    if isinstance(phase, SustainedPhase | MixedPhase):
        return float(phase.duration_seconds)
    if isinstance(phase, OverloadPhase):
        return sum(float(stage.duration_seconds) for stage in phase.stages)
    if isinstance(phase, ColdBurstPhase) and phase.burst_window_seconds > 0:
        return float(phase.burst_window_seconds)
    return None


def build_schedule(
    phases: list[Phase],
    targets: list[Target],
    profiles: list[RequestProfile],
    *,
    seed: int,
) -> list[list[ScheduledRequest]]:
    rng = random.Random(seed)
    request_number = 0
    result: list[list[ScheduledRequest]] = []
    for phase_index, phase in enumerate(phases):
        phase_targets = _select_named(targets, phase.target_names, "target")
        phase_profiles = _select_named(profiles, phase.profile_names, "profile")
        offsets = _phase_offsets(phase)
        stage_indexes = _stage_indexes(phase)
        scheduled = []
        for phase_request_index, offset_ns in enumerate(offsets):
            scheduled.append(
                ScheduledRequest(
                    request_id=f"request-{request_number:08d}",
                    phase_index=phase_index,
                    phase_name=phase.name,
                    phase_kind=phase.kind,
                    phase_request_index=phase_request_index,
                    scheduled_offset_ns=offset_ns,
                    target=_weighted_choice(rng, phase_targets),
                    profile=_weighted_choice(rng, phase_profiles),
                    open_loop=not isinstance(phase, WarmPhase),
                    stage_index=stage_indexes[phase_request_index],
                )
            )
            request_number += 1
        result.append(scheduled)
    return result


def _select_named(values: list, names: list[str], label: str) -> list:
    if not names:
        selected = list(values)
    else:
        by_name = {value.name: value for value in values}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"unknown {label} selectors: {', '.join(missing)}")
        selected = [by_name[name] for name in names]
    if not selected:
        raise ValueError(f"phase has no selected {label}s")
    return selected


def _weighted_choice(rng: random.Random, values: list):
    total = sum(float(value.weight) for value in values)
    point = rng.random() * total
    cumulative = 0.0
    for value in values:
        cumulative += float(value.weight)
        if point < cumulative:
            return value
    return values[-1]


def _phase_offsets(phase: Phase) -> list[int]:
    if isinstance(phase, WarmPhase):
        return [0] * phase.requests
    if isinstance(phase, ColdBurstPhase):
        if phase.requests == 1 or phase.burst_window_seconds == 0:
            return [0] * phase.requests
        spacing = phase.burst_window_seconds / (phase.requests - 1)
        return [_seconds_to_ns(index * spacing) for index in range(phase.requests)]
    if isinstance(phase, SustainedPhase | MixedPhase):
        return _rate_offsets(float(phase.duration_seconds), float(phase.rate_rps))
    offsets: list[int] = []
    stage_start = 0.0
    for stage in phase.stages:
        offsets.extend(
            _seconds_to_ns(stage_start + offset)
            for offset in _rate_offset_seconds(float(stage.duration_seconds), float(stage.rate_rps))
        )
        stage_start += float(stage.duration_seconds)
    return offsets


def _stage_indexes(phase: Phase) -> list[int | None]:
    if not isinstance(phase, OverloadPhase):
        return [None] * len(_phase_offsets(phase))
    indexes: list[int | None] = []
    for stage_index, stage in enumerate(phase.stages):
        indexes.extend(
            [stage_index]
            * len(_rate_offset_seconds(float(stage.duration_seconds), float(stage.rate_rps)))
        )
    return indexes


def _rate_offsets(duration_seconds: float, rate_rps: float) -> list[int]:
    return [_seconds_to_ns(value) for value in _rate_offset_seconds(duration_seconds, rate_rps)]


def _rate_offset_seconds(duration_seconds: float, rate_rps: float) -> list[float]:
    request_count = max(1, math.ceil(duration_seconds * rate_rps))
    return [
        index / rate_rps for index in range(request_count) if index / rate_rps < duration_seconds
    ]


def _seconds_to_ns(value: float) -> int:
    return round(value * _NANOSECONDS_PER_SECOND)
