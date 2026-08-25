"""Live Modal capacity snapshots for hosted request admission."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from flash.serving.src.model_config import hosted_traffic_policy_for

CAPACITY_SNAPSHOT_MAX_AGE_SECONDS = 2.0
CAPACITY_POLL_INTERVAL_SECONDS = 0.5
CAPACITY_REFRESH_TIMEOUT_SECONDS = 1.0


def fixed_local_active_limit(
    observed_local_active: int,
    running_inputs: int,
    input_headroom: int,
    hard_limit: int,
) -> int:
    """Bind one FunctionStats observation to an absolute local admission limit."""
    values = (observed_local_active, running_inputs, input_headroom)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("capacity observation counts must be nonnegative integers")
    if isinstance(hard_limit, bool) or not isinstance(hard_limit, int) or hard_limit <= 0:
        raise ValueError("hard_limit must be a positive integer")
    unreflected_local = max(0, observed_local_active - running_inputs)
    incremental_slots = max(0, input_headroom - unreflected_local)
    return min(hard_limit, observed_local_active + incremental_slots)


@dataclass(frozen=True, slots=True)
class CapacitySnapshot:
    """One immutable observation for one exact deployed model engine."""

    model: str
    deployment_identity: str
    observed_at: float
    total_runners: int
    running_inputs: int
    input_headroom: int
    backlog: int
    observed_local_active: int
    local_active_limit: int
    unavailable: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.model or self.model != self.model.strip():
            raise ValueError("capacity snapshot model must be nonempty and unpadded")
        if (
            not self.deployment_identity
            or self.deployment_identity != self.deployment_identity.strip()
            or len(self.deployment_identity) > 256
        ):
            raise ValueError(
                "capacity snapshot deployment_identity must be nonempty, unpadded, and at most 256 characters"
            )
        if self.observed_at < 0:
            raise ValueError("capacity snapshot observed_at must be nonnegative")
        counts = (
            self.total_runners,
            self.running_inputs,
            self.input_headroom,
            self.backlog,
            self.observed_local_active,
            self.local_active_limit,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in counts):
            raise ValueError("capacity snapshot counts must be integers")
        if any(value < 0 for value in counts):
            raise ValueError("capacity snapshot counts must be nonnegative")
        if self.unavailable and not self.error:
            raise ValueError("unavailable capacity snapshot requires an error indication")
        if not self.unavailable and self.error is not None:
            raise ValueError("available capacity snapshot cannot carry an error")
        if not self.unavailable and self.total_runners <= 0:
            raise ValueError("available capacity snapshot requires a warm runner")

    def is_fresh(
        self, now: float, *, max_age_seconds: float = CAPACITY_SNAPSHOT_MAX_AGE_SECONDS
    ) -> bool:
        return 0.0 <= now - self.observed_at <= max_age_seconds

    def is_dispatchable(
        self,
        now: float,
        *,
        model: str,
        deployment_identity: str,
        max_age_seconds: float = CAPACITY_SNAPSHOT_MAX_AGE_SECONDS,
    ) -> bool:
        return (
            self.model == model
            and self.deployment_identity == deployment_identity
            and not self.unavailable
            and self.total_runners > 0
            and self.is_fresh(now, max_age_seconds=max_age_seconds)
        )

    @classmethod
    def unavailable_snapshot(
        cls,
        model: str,
        deployment_identity: str,
        observed_at: float,
        error: str,
        *,
        total_runners: int = 0,
        running_inputs: int = 0,
        input_headroom: int = 0,
        backlog: int = 0,
        observed_local_active: int = 0,
    ) -> CapacitySnapshot:
        return cls(
            model=model,
            deployment_identity=deployment_identity,
            observed_at=observed_at,
            total_runners=total_runners,
            running_inputs=running_inputs,
            input_headroom=input_headroom,
            backlog=backlog,
            observed_local_active=observed_local_active,
            local_active_limit=0,
            unavailable=True,
            error=error,
        )


class CapacityProvider(Protocol):
    async def capacity_snapshot(
        self, model: str, observed_local_active: int
    ) -> CapacitySnapshot: ...

    def current_dispatch_capacity(self, model: str) -> int: ...


class ConfiguredCapacityProvider:
    """Offline/test-only provider that exposes the catalog hard limit without remote polling."""

    deployment_identity = "offline/test-only:configured-capacity"

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock

    async def capacity_snapshot(self, model: str, observed_local_active: int) -> CapacitySnapshot:
        policy = hosted_traffic_policy_for(model)
        hard_limit = policy.max_inputs * policy.max_containers
        return CapacitySnapshot(
            model=model,
            deployment_identity=self.deployment_identity,
            observed_at=self._clock(),
            total_runners=policy.max_containers,
            running_inputs=observed_local_active,
            input_headroom=max(0, hard_limit - observed_local_active),
            backlog=0,
            observed_local_active=observed_local_active,
            local_active_limit=hard_limit,
        )

    def current_dispatch_capacity(self, model: str) -> int:
        policy = hosted_traffic_policy_for(model)
        return policy.max_inputs * policy.max_containers
