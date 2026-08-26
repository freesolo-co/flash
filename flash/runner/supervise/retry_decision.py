"""Central retry policy for supervised training attempts."""

from __future__ import annotations

import math
from dataclasses import dataclass

from flash.core.spec import JobSpec

INFRA_RETRY_FLOOR = 5
INFRA_RETRY_FAILURES = frozenset({"stalled", "no_capacity", "poll_error", "job_preempted"})
CANDIDATE_RETRY_FAILURES = INFRA_RETRY_FAILURES | {"oom"}
CACHE_FALLBACK_FAILURES = frozenset({"no_capacity", "poll_error"})
_RETRY_STATE_VERSION = 1
_RETRY_STATE_FIELDS = frozenset(
    {
        "version",
        "started_with_shared_cache",
        "usable_vram_floor",
        "infra_used",
        "oom_used",
        "cache_used",
        "drop_weight_cache",
        "cache_retry_shape",
        "last_decision_attempt",
        "last_decision_failure",
        "last_decision_retry",
        "last_infra_retry_ordinal",
    }
)
_EXPECTED_REMOTE_UNSET = object()


@dataclass(frozen=True)
class _PersistedCandidate:
    provider: str
    gpu: str
    gpu_count: int
    usable_vram_gb: float


def _candidate_shape(candidate) -> tuple[str, str, int]:
    return (
        candidate.provider,
        candidate.gpu,
        int(getattr(candidate, "gpu_count", 1) or 1),
    )


def _candidate_usable_vram_gb(candidate) -> float:
    """Return usable vram on the allocator's executed-width-aware fit scale."""
    persisted = getattr(candidate, "usable_vram_gb", None)
    if persisted is not None:
        return float(persisted)
    from flash.providers.core.sharding import combined_vram_gb

    rented = int(getattr(candidate, "gpu_count", 1) or 1)
    executed = getattr(candidate, "executed_gpu_count", None)
    width = int(executed) if type(executed) is int and executed > 0 else rented
    return combined_vram_gb(candidate.vram_gb, width)


def _strictly_larger_candidates(candidates, usable_vram_floor: float):
    """Preserve allocator order while removing shapes at or below the floor."""
    return tuple(
        candidate
        for candidate in candidates
        if _candidate_usable_vram_gb(candidate) > usable_vram_floor
    )


def _drop_weight_cache(spec: JobSpec) -> JobSpec:
    """Remove only the managed shared cache for the cacheless all-region retry."""
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    if getattr(spec.gpu, "network_volume", None) != WEIGHT_CACHE_VOLUME_NAME:
        return spec
    data = spec.to_internal_dict()
    data["gpu"] = {**data["gpu"], "network_volume": None}
    return JobSpec.from_dict(data)


def _started_with_shared_cache(spec: JobSpec) -> bool:
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    return getattr(spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME


def _require_counter(value: object, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ValueError(f"persisted retry state has invalid {name}")
    return value


def _require_optional_attempt(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("persisted retry state has invalid last decision attempt")
    return value


def _require_cache_shape(value: object) -> tuple[str, str, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("persisted retry state has invalid cache retry shape")
    provider, gpu, count = value
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("persisted retry state has invalid cache retry provider")
    if not isinstance(gpu, str) or not gpu.strip():
        raise ValueError("persisted retry state has invalid cache retry gpu")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("persisted retry state has invalid cache retry count")
    return provider, gpu, count


@dataclass(frozen=True)
class RetryPlan:
    retry: bool
    action: str
    infra_retry_ordinal: int | None = None


@dataclass
class RetryState:
    """Monotonic retry state shared by allocation and failure handling."""

    infra_retries: int
    oom_retries: int
    cache_retries: int
    infra_used: int = 0
    oom_used: int = 0
    cache_used: int = 0
    usable_vram_floor: float = 0.0
    drop_weight_cache: bool = False
    cache_retry_shape: tuple[str, str, int] | None = None
    started_with_shared_cache: bool = False
    last_decision_attempt: int | None = None
    last_decision_failure: str | None = None
    last_decision_retry: bool | None = None
    last_infra_retry_ordinal: int | None = None

    @classmethod
    def for_spec(cls, spec: JobSpec, *, started_with_shared_cache: bool) -> RetryState:
        max_retries = int(spec.gpu.max_retries)
        infra_retries = max(max_retries, INFRA_RETRY_FLOOR) if max_retries else 0
        cache_retries = 1 if started_with_shared_cache and max_retries > 0 else 0
        return cls(
            infra_retries,
            max_retries,
            cache_retries,
            started_with_shared_cache=started_with_shared_cache,
        )

    @classmethod
    def initial_for_spec(cls, spec: JobSpec) -> RetryState:
        return cls.for_spec(spec, started_with_shared_cache=_started_with_shared_cache(spec))

    @classmethod
    def from_snapshot(cls, spec: JobSpec, raw: object) -> RetryState:
        if not isinstance(raw, dict) or set(raw) != _RETRY_STATE_FIELDS:
            raise ValueError("persisted retry state has an invalid shape")
        if raw.get("version") != _RETRY_STATE_VERSION:
            raise ValueError("persisted retry state has an unsupported version")
        started_with_shared_cache = raw.get("started_with_shared_cache")
        if type(started_with_shared_cache) is not bool:
            raise ValueError("persisted retry state has invalid cache provenance")
        maxima = cls.for_spec(spec, started_with_shared_cache=started_with_shared_cache)
        floor = raw.get("usable_vram_floor")
        if (
            isinstance(floor, bool)
            or not isinstance(floor, (int, float))
            or not math.isfinite(float(floor))
            or float(floor) < 0
        ):
            raise ValueError("persisted retry state has invalid usable vram floor")
        infra_used = _require_counter(
            raw.get("infra_used"), "infra retry count", maxima.infra_retries
        )
        oom_used = _require_counter(raw.get("oom_used"), "oom retry count", maxima.oom_retries)
        cache_used = _require_counter(
            raw.get("cache_used"), "cache retry count", maxima.cache_retries
        )
        drop_weight_cache = raw.get("drop_weight_cache")
        if type(drop_weight_cache) is not bool:
            raise ValueError("persisted retry state has invalid cache-drop state")
        cache_retry_shape = _require_cache_shape(raw.get("cache_retry_shape"))
        if cache_used > 0 and not drop_weight_cache:
            raise ValueError("persisted retry state lost its consumed cache fallback")
        if drop_weight_cache and (not started_with_shared_cache or cache_used != 1):
            raise ValueError("persisted retry state has inconsistent cache-drop state")
        if cache_retry_shape is not None and not drop_weight_cache:
            raise ValueError("persisted retry state has an unarmed cache retry shape")

        last_attempt = _require_optional_attempt(raw.get("last_decision_attempt"))
        last_failure = raw.get("last_decision_failure")
        last_retry = raw.get("last_decision_retry")
        last_ordinal = raw.get("last_infra_retry_ordinal")
        if last_attempt is None:
            if any(value is not None for value in (last_failure, last_retry, last_ordinal)) or any(
                (infra_used, oom_used, cache_used)
            ):
                raise ValueError("persisted retry state has an incomplete last decision")
        else:
            if not isinstance(last_failure, str) or not last_failure:
                raise ValueError("persisted retry state has invalid last decision failure")
            if type(last_retry) is not bool:
                raise ValueError("persisted retry state has invalid last decision result")
            if last_ordinal is not None and (
                isinstance(last_ordinal, bool)
                or not isinstance(last_ordinal, int)
                or not 1 <= last_ordinal <= infra_used
                or not last_retry
            ):
                raise ValueError("persisted retry state has invalid infrastructure retry ordinal")

        return cls(
            infra_retries=maxima.infra_retries,
            oom_retries=maxima.oom_retries,
            cache_retries=maxima.cache_retries,
            infra_used=infra_used,
            oom_used=oom_used,
            cache_used=cache_used,
            usable_vram_floor=float(floor),
            drop_weight_cache=drop_weight_cache,
            cache_retry_shape=cache_retry_shape,
            started_with_shared_cache=started_with_shared_cache,
            last_decision_attempt=last_attempt,
            last_decision_failure=last_failure,
            last_decision_retry=last_retry,
            last_infra_retry_ordinal=last_ordinal,
        )

    def to_snapshot(self) -> dict:
        return {
            "version": _RETRY_STATE_VERSION,
            "started_with_shared_cache": self.started_with_shared_cache,
            "usable_vram_floor": float(self.usable_vram_floor),
            "infra_used": self.infra_used,
            "oom_used": self.oom_used,
            "cache_used": self.cache_used,
            "drop_weight_cache": self.drop_weight_cache,
            "cache_retry_shape": list(self.cache_retry_shape) if self.cache_retry_shape else None,
            "last_decision_attempt": self.last_decision_attempt,
            "last_decision_failure": self.last_decision_failure,
            "last_decision_retry": self.last_decision_retry,
            "last_infra_retry_ordinal": self.last_infra_retry_ordinal,
        }

    @property
    def max_attempts(self) -> int:
        return 1 + (
            self.infra_retries
            - self.infra_used
            + self.oom_retries
            - self.oom_used
            + self.cache_retries
            - self.cache_used
        )

    def select_candidate(self, candidates):
        """Return eligible candidates and the next candidate in allocator order."""
        eligible = _strictly_larger_candidates(candidates, self.usable_vram_floor)
        if not eligible:
            return eligible, None
        if self.cache_retry_shape is None:
            return eligible, eligible[0]
        chosen = next(
            (
                candidate
                for candidate in eligible
                if _candidate_shape(candidate) == self.cache_retry_shape
            ),
            None,
        )
        return eligible, chosen

    def on_last_gpu(self, chosen, candidates, *, cache_fallback_available: bool) -> bool:
        """Return whether queue grace should treat this as the final usable class."""
        if chosen is None:
            return True
        if self.infra_used >= self.infra_retries and not cache_fallback_available:
            return True
        floor = _candidate_usable_vram_gb(chosen)
        return not any(_candidate_usable_vram_gb(candidate) > floor for candidate in candidates)

    def _record_decision(
        self,
        plan: RetryPlan,
        *,
        failure: str,
        attempt: int | None,
    ) -> RetryPlan:
        if attempt is not None:
            if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
                raise ValueError("retry decision attempt is invalid")
            self.last_decision_attempt = attempt
            self.last_decision_failure = failure
            self.last_decision_retry = plan.retry
            self.last_infra_retry_ordinal = plan.infra_retry_ordinal
        return plan

    def _cached_decision(self, attempt: int | None) -> RetryPlan | None:
        if attempt is None or attempt != self.last_decision_attempt:
            return None
        return RetryPlan(
            bool(self.last_decision_retry),
            "retry decision already persisted",
            self.last_infra_retry_ordinal,
        )

    def decide_failure(
        self,
        failure: str | None,
        *,
        chosen,
        candidates,
        managed_cache_mounted: bool,
        attempt: int | None = None,
    ) -> RetryPlan:
        """Record one failure and return the only permitted replacement plan."""
        normalized_failure = failure or "unknown"
        cached = self._cached_decision(attempt)
        if cached is not None:
            return cached

        def finish(plan: RetryPlan) -> RetryPlan:
            return self._record_decision(plan, failure=normalized_failure, attempt=attempt)

        if failure not in CANDIDATE_RETRY_FAILURES:
            return finish(RetryPlan(False, "not retrying"))

        if chosen is None:
            if failure == "no_capacity":
                return finish(RetryPlan(False, "not retrying: allocation reported no capacity"))
            if failure == "poll_error" and self.infra_used < self.infra_retries:
                self.infra_used += 1
                return finish(
                    RetryPlan(
                        True,
                        "retrying allocation (resume from last checkpoint)",
                        self.infra_used,
                    )
                )
            return finish(RetryPlan(False, "not retrying"))

        if (
            failure in CACHE_FALLBACK_FAILURES
            and managed_cache_mounted
            and not self.drop_weight_cache
            and self.cache_used < self.cache_retries
        ):
            self.cache_used += 1
            self.drop_weight_cache = True
            self.cache_retry_shape = _candidate_shape(chosen)
            return finish(
                RetryPlan(
                    True,
                    f"retrying cacheless on the same {chosen.gpu} @ {chosen.provider} "
                    "(resume from last checkpoint)",
                )
            )

        self.cache_retry_shape = None
        failed_vram = _candidate_usable_vram_gb(chosen)
        self.usable_vram_floor = max(self.usable_vram_floor, failed_vram)
        if candidates is not None and not any(
            _candidate_usable_vram_gb(candidate) > self.usable_vram_floor
            for candidate in candidates
        ):
            return finish(
                RetryPlan(
                    False,
                    f"not retrying: no candidate has more than "
                    f"{self.usable_vram_floor:g} GB usable vram",
                )
            )

        if failure == "oom":
            if self.oom_used >= self.oom_retries:
                return finish(RetryPlan(False, "not retrying: oom retry budget exhausted"))
            self.oom_used += 1
            ordinal = None
        else:
            if self.infra_used >= self.infra_retries:
                return finish(
                    RetryPlan(False, "not retrying: infrastructure retry budget exhausted")
                )
            self.infra_used += 1
            ordinal = self.infra_used

        return finish(
            RetryPlan(
                True,
                f"retrying on a strictly larger GPU (> {self.usable_vram_floor:g} GB usable vram; "
                "resume from last checkpoint)",
                ordinal,
            )
        )


def retry_candidate_from_remote(remote: object) -> _PersistedCandidate:
    if not isinstance(remote, dict):
        raise ValueError("persisted retry candidate is missing")
    provider = remote.get("provider")
    gpu = remote.get("allocated_gpu")
    count = remote.get("allocated_gpu_count")
    usable = remote.get("allocated_usable_vram_gb")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("persisted retry provider is missing or invalid")
    if not isinstance(gpu, str) or not gpu.strip():
        raise ValueError("persisted allocated gpu is missing or invalid")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("persisted allocated gpu count is missing or invalid")
    if (
        isinstance(usable, bool)
        or not isinstance(usable, (int, float))
        or not math.isfinite(float(usable))
        or float(usable) <= 0
    ):
        raise ValueError("persisted usable vram is missing or invalid")
    return _PersistedCandidate(provider, gpu, count, float(usable))


def load_retry_state(run_id: str, spec: JobSpec) -> RetryState:
    from flash.runner.lifecycle import state as lifecycle_state
    from flash.runner.lifecycle.status import _load_status_json

    raw = _load_status_json(run_id)
    if lifecycle_state._RETRY_STATE_KEY not in raw:
        raise RuntimeError("persisted retry state is missing")
    try:
        return RetryState.from_snapshot(spec, raw[lifecycle_state._RETRY_STATE_KEY])
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc


def decide_attached_failure_atomically(
    run_id: str,
    spec: JobSpec,
    *,
    expected_remote: dict,
    expected_retry_snapshot: dict,
    failure: str | None,
    chosen,
    managed_cache_mounted: bool,
    attempt: int,
) -> tuple[RetryState, RetryPlan] | None:
    """Return the immutable attempt decision when the captured state still owns the run."""
    from flash.runner.lifecycle import state as lifecycle_state
    from flash.runner.lifecycle.status import _load_status_json, _runstatus_from_json

    with lifecycle_state._status_guard(run_id):
        raw = _load_status_json(run_id)
        if raw.get("state") in lifecycle_state.TERMINAL_STATES:
            return None
        if raw.get("remote") != expected_remote:
            return None
        if raw.get(lifecycle_state._RETRY_STATE_KEY) != expected_retry_snapshot:
            return None
        try:
            retry_state = RetryState.from_snapshot(
                spec,
                raw[lifecycle_state._RETRY_STATE_KEY],
            )
        except (KeyError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc
        plan = retry_state.decide_failure(
            failure,
            chosen=chosen,
            candidates=None,
            managed_cache_mounted=managed_cache_mounted,
            attempt=attempt,
        )
        snapshot = retry_state.to_snapshot()
        if snapshot != raw[lifecycle_state._RETRY_STATE_KEY]:
            status = _runstatus_from_json(raw)
            lifecycle_state._save_status_unlocked(
                status,
                _retry_state=snapshot,
            )
    return retry_state, plan


def persist_retry_state(
    run_id: str,
    retry_state: RetryState,
    *,
    expected_remote: object = _EXPECTED_REMOTE_UNSET,
) -> bool:
    from flash.runner.lifecycle import state as lifecycle_state
    from flash.runner.lifecycle.status import _load_status_json, _runstatus_from_json

    snapshot = retry_state.to_snapshot()
    with lifecycle_state._status_guard(run_id):
        raw = _load_status_json(run_id)
        if expected_remote is not _EXPECTED_REMOTE_UNSET and raw.get("remote") != expected_remote:
            return False
        status = _runstatus_from_json(raw)
        lifecycle_state._save_status_unlocked(status, _retry_state=snapshot)
    return True
