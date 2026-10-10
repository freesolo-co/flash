"""Atomic persisted retry authorization for supervised training attempts."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from flash.core.spec import JobSpec

INFRA_RETRY_FLOOR = 5
INFRA_RETRY_FAILURES = frozenset({"stalled", "no_capacity", "poll_error", "job_preempted"})
CANDIDATE_RETRY_FAILURES = INFRA_RETRY_FAILURES | {"oom"}
CACHE_FALLBACK_FAILURES = frozenset({"no_capacity", "poll_error"})
_STATE_KEYS = frozenset(
    {
        "version",
        "usable_vram_floor",
        "infra_used",
        "oom_used",
        "drop_weight_cache",
        "cache_retry_shape",
        "last_decision",
    }
)
_DECISION_KEYS = frozenset({"attempt", "failure", "plan"})
_PLAN_KEYS = frozenset({"retry", "action", "infra_retry_ordinal"})


@dataclass(frozen=True)
class _PersistedCandidate:
    provider: str
    gpu: str
    gpu_count: int
    usable_vram_gb: float


@dataclass(frozen=True)
class RetryPlan:
    retry: bool
    action: str
    infra_retry_ordinal: int | None = None


@dataclass(frozen=True)
class PersistedRetryDecision:
    attempt: int
    failure: str
    plan: RetryPlan


@dataclass(frozen=True)
class FailureObservation:
    failure: str
    chosen: object | None = None
    candidates: tuple | None = None
    managed_cache_mounted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "failure", self.failure or "unknown")
        if self.candidates is not None:
            object.__setattr__(self, "candidates", tuple(self.candidates))


@dataclass(frozen=True)
class RetryState:
    infra_retries: int
    oom_retries: int
    cache_retries: int
    infra_used: int = 0
    oom_used: int = 0
    usable_vram_floor: float = 0.0
    drop_weight_cache: bool = False
    cache_retry_shape: tuple[str, str, int] | None = None
    last_decision: PersistedRetryDecision | None = None

    @classmethod
    def initial_for_spec(cls, spec: JobSpec) -> RetryState:
        retries = int(spec.gpu.max_retries)
        return cls(max(retries, INFRA_RETRY_FLOOR) if retries else 0, retries, int(retries > 0))

    @classmethod
    def from_snapshot(cls, spec: JobSpec, raw: object) -> RetryState:
        if not isinstance(raw, dict) or set(raw) != _STATE_KEYS or raw.get("version") != 1:
            raise ValueError("persisted retry state has an invalid shape or version")
        maxima = cls.initial_for_spec(spec)
        floor = raw["usable_vram_floor"]
        if (
            isinstance(floor, bool)
            or not isinstance(floor, (int, float))
            or not math.isfinite(float(floor))
            or floor < 0
        ):
            raise ValueError("persisted retry state has invalid usable vram floor")
        counters = {
            key: _bounded_count(raw[key], label, maximum)
            for key, label, maximum in (
                ("infra_used", "infra retry count", maxima.infra_retries),
                ("oom_used", "oom retry count", maxima.oom_retries),
            )
        }
        drop = raw["drop_weight_cache"]
        shape = _restore_cache_shape(raw["cache_retry_shape"])
        if type(drop) is not bool:
            raise ValueError("persisted retry state has invalid cache-drop state")
        if drop and maxima.cache_retries == 0:
            raise ValueError("persisted retry state exceeds cache retry budget")
        if shape is not None and not drop:
            raise ValueError("persisted retry state has an unarmed cache retry shape")
        decision = _restore_last_decision(raw["last_decision"], counters)
        return cls(
            maxima.infra_retries,
            maxima.oom_retries,
            maxima.cache_retries,
            **counters,
            usable_vram_floor=float(floor),
            drop_weight_cache=drop,
            cache_retry_shape=shape,
            last_decision=decision,
        )

    def to_snapshot(self) -> dict:
        return {
            "version": 1,
            "usable_vram_floor": float(self.usable_vram_floor),
            "infra_used": self.infra_used,
            "oom_used": self.oom_used,
            "drop_weight_cache": self.drop_weight_cache,
            "cache_retry_shape": list(self.cache_retry_shape) if self.cache_retry_shape else None,
            "last_decision": _decision_snapshot(self.last_decision),
        }

    def persisted_plan(self, attempt: int) -> RetryPlan | None:
        decision = self.last_decision
        return decision.plan if decision is not None and decision.attempt == attempt else None

    def select_candidate(self, candidates):
        eligible = _strictly_larger_candidates(candidates, self.usable_vram_floor)
        if not eligible or self.cache_retry_shape is None:
            return eligible, eligible[0] if eligible else None
        return eligible, next(
            (
                candidate
                for candidate in eligible
                if _candidate_shape(candidate) == self.cache_retry_shape
            ),
            None,
        )

    def on_last_gpu(self, chosen, candidates, *, cache_fallback_available: bool) -> bool:
        if cache_fallback_available:
            return False
        if chosen is None or self.infra_used >= self.infra_retries:
            return True
        floor = _candidate_usable_vram_gb(chosen)
        return not any(_candidate_usable_vram_gb(candidate) > floor for candidate in candidates)


def _bounded_count(value: object, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"persisted retry state has invalid {label}")
    return value


def _restore_plan(raw: object, counters: dict) -> RetryPlan:
    if not isinstance(raw, dict) or set(raw) != _PLAN_KEYS:
        raise ValueError("persisted retry state has an invalid last decision plan")
    retry, action, ordinal = raw["retry"], raw["action"], raw["infra_retry_ordinal"]
    if type(retry) is not bool or not isinstance(action, str) or not action:
        raise ValueError("persisted retry state has an invalid last decision plan")
    if ordinal is not None and (
        isinstance(ordinal, bool)
        or not isinstance(ordinal, int)
        or not 1 <= ordinal <= counters["infra_used"]
        or retry is not True
    ):
        raise ValueError("persisted retry state has invalid infrastructure retry ordinal")
    return RetryPlan(retry, action, ordinal)


def _restore_last_decision(raw: object, counters: dict) -> PersistedRetryDecision | None:
    if raw is None:
        if any(counters.values()):
            raise ValueError("persisted retry state has an incomplete last decision")
        return None
    if not isinstance(raw, dict) or set(raw) != _DECISION_KEYS:
        raise ValueError("persisted retry state has an invalid last decision")
    attempt, failure = raw["attempt"], raw["failure"]
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 0
        or not isinstance(failure, str)
        or not failure
    ):
        raise ValueError("persisted retry state has an invalid last decision")
    return PersistedRetryDecision(attempt, failure, _restore_plan(raw["plan"], counters))


def _decision_snapshot(decision: PersistedRetryDecision | None) -> dict | None:
    if decision is None:
        return None
    return {
        "attempt": decision.attempt,
        "failure": decision.failure,
        "plan": {
            "retry": decision.plan.retry,
            "action": decision.plan.action,
            "infra_retry_ordinal": decision.plan.infra_retry_ordinal,
        },
    }


def _restore_cache_shape(value: object) -> tuple[str, str, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("persisted retry state has invalid cache retry shape")
    provider, gpu, count = value
    if (
        not isinstance(provider, str)
        or not provider.strip()
        or not isinstance(gpu, str)
        or not gpu.strip()
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
    ):
        raise ValueError("persisted retry state has invalid cache retry shape")
    return provider, gpu, count


def _candidate_shape(candidate) -> tuple[str, str, int]:
    return candidate.provider, candidate.gpu, int(getattr(candidate, "gpu_count", 1) or 1)


def _candidate_usable_vram_gb(candidate) -> float:
    persisted = getattr(candidate, "usable_vram_gb", None)
    if persisted is not None:
        return float(persisted)
    from flash.providers.core.sharding import combined_vram_gb

    rented = int(getattr(candidate, "gpu_count", 1) or 1)
    executed = getattr(candidate, "executed_gpu_count", None)
    return combined_vram_gb(
        candidate.vram_gb, executed if type(executed) is int and executed > 0 else rented
    )


def _strictly_larger_candidates(candidates, usable_vram_floor: float):
    return tuple(
        candidate
        for candidate in candidates
        if _candidate_usable_vram_gb(candidate) > usable_vram_floor
    )


def _drop_weight_cache(spec: JobSpec) -> JobSpec:
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    if spec.gpu.network_volume != WEIGHT_CACHE_VOLUME_NAME:
        return spec
    data = spec.to_internal_dict()
    data["gpu"] = {**data["gpu"], "network_volume": None}
    return JobSpec.from_dict(data)


def _managed_cache_mounted(spec: JobSpec, chosen) -> bool:
    if chosen is None:
        return False
    from flash.providers.core.registry import get_provider
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    return bool(
        spec.gpu.network_volume == WEIGHT_CACHE_VOLUME_NAME
        and getattr(get_provider(chosen.provider), "supports_weight_cache", False)
    )


def _finish(
    state: RetryState,
    plan: RetryPlan,
    observation: FailureObservation,
    attempt: int,
) -> tuple[RetryState, RetryPlan]:
    return replace(
        state,
        last_decision=PersistedRetryDecision(attempt, observation.failure, plan),
    ), plan


def transition_failure(
    state: RetryState,
    observation: FailureObservation,
    *,
    attempt: int,
) -> tuple[RetryState, RetryPlan]:
    """Return the pure retry transition for one failure observation."""
    failure = observation.failure
    chosen = observation.chosen
    candidates = observation.candidates
    if failure not in CANDIDATE_RETRY_FAILURES:
        return _finish(state, RetryPlan(False, "not retrying"), observation, attempt)
    if chosen is None:
        if failure == "no_capacity":
            plan = RetryPlan(False, "not retrying: allocation reported no capacity")
        elif failure == "poll_error" and state.infra_used < state.infra_retries:
            ordinal = state.infra_used + 1
            state = replace(state, infra_used=ordinal)
            plan = RetryPlan(True, "retrying allocation (resume from last checkpoint)", ordinal)
        else:
            plan = RetryPlan(False, "not retrying")
        return _finish(state, plan, observation, attempt)
    if (
        failure in CACHE_FALLBACK_FAILURES
        and observation.managed_cache_mounted
        and not state.drop_weight_cache
        and state.cache_retries > 0
    ):
        state = replace(
            state,
            drop_weight_cache=True,
            cache_retry_shape=_candidate_shape(chosen),
        )
        return _finish(
            state,
            RetryPlan(
                True,
                f"retrying cacheless on the same {chosen.gpu} @ {chosen.provider} (resume from last checkpoint)",
            ),
            observation,
            attempt,
        )
    floor = max(state.usable_vram_floor, _candidate_usable_vram_gb(chosen))
    state = replace(state, usable_vram_floor=floor, cache_retry_shape=None)
    if candidates is not None and not any(
        _candidate_usable_vram_gb(candidate) > floor for candidate in candidates
    ):
        plan = RetryPlan(
            False, f"not retrying: no candidate has more than {floor:g} GB usable vram"
        )
    elif failure == "oom":
        if state.oom_used >= state.oom_retries:
            plan = RetryPlan(False, "not retrying: oom retry budget exhausted")
        else:
            state = replace(state, oom_used=state.oom_used + 1)
            plan = RetryPlan(
                True,
                f"retrying on a strictly larger GPU (> {floor:g} GB usable vram; resume from last checkpoint)",
            )
    elif state.infra_used >= state.infra_retries:
        plan = RetryPlan(False, "not retrying: infrastructure retry budget exhausted")
    else:
        ordinal = state.infra_used + 1
        state = replace(state, infra_used=ordinal)
        plan = RetryPlan(
            True,
            f"retrying on a strictly larger GPU (> {floor:g} GB usable vram; resume from last checkpoint)",
            ordinal,
        )
    return _finish(state, plan, observation, attempt)


def retry_candidate_from_remote(remote: object) -> _PersistedCandidate:
    if not isinstance(remote, dict):
        raise ValueError("persisted retry candidate is missing")
    provider, gpu = remote.get("provider"), remote.get("allocated_gpu")
    count, usable = remote.get("allocated_gpu_count"), remote.get("allocated_usable_vram_gb")
    if (
        not isinstance(provider, str)
        or not provider.strip()
        or not isinstance(gpu, str)
        or not gpu.strip()
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or isinstance(usable, bool)
        or not isinstance(usable, (int, float))
        or not math.isfinite(float(usable))
        or usable <= 0
    ):
        raise ValueError("persisted retry candidate is missing or invalid")
    return _PersistedCandidate(provider, gpu, count, float(usable))


def require_retry_authorization_from_raw(spec: JobSpec, raw: dict, attempt: int) -> RetryState:
    from flash.runner.lifecycle import state

    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        raise RuntimeError("attempt identity is invalid")
    try:
        retry_state = RetryState.from_snapshot(spec, raw[state._RETRY_STATE_KEY])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("persisted retry state is missing or invalid") from exc
    decision = retry_state.last_decision
    if attempt > 0 and (
        decision is None or decision.attempt != attempt - 1 or decision.plan.retry is not True
    ):
        raise RuntimeError(
            f"attempt {attempt} lacks exact persisted retry authorization for attempt {attempt - 1}"
        )
    return retry_state
