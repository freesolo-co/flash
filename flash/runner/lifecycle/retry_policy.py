"""Private durable retry counters and placement exclusions for one run."""

from __future__ import annotations

from dataclasses import dataclass

from flash.core.spec import JobSpec

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RetryPolicyState:
    infra_used: int = 0
    oom_used: int = 0
    cache_used: int = 0
    failed_providers: frozenset[str] = frozenset()
    tried_classes: frozenset[tuple[str, str, int]] = frozenset()
    oom_vram_floor: float = 0.0
    consumed_attempt: tuple[int, int] | None = None
    consumed_failure: str | None = None
    consumed_cache_drop: bool = False
    consumed_retry_allowed: bool = False

    def to_dict(self) -> dict:
        consumed = None
        if self.consumed_attempt is not None:
            consumed = {
                "attempt_id": self.consumed_attempt[0],
                "fence": self.consumed_attempt[1],
                "failure": self.consumed_failure,
                "cache_drop": self.consumed_cache_drop,
                "retry_allowed": self.consumed_retry_allowed,
            }
        return {
            "schema_version": _SCHEMA_VERSION,
            "infra_used": self.infra_used,
            "oom_used": self.oom_used,
            "cache_used": self.cache_used,
            "failed_providers": sorted(self.failed_providers),
            "tried_classes": [
                {"provider": provider, "gpu": gpu, "gpu_count": gpu_count}
                for provider, gpu, gpu_count in sorted(self.tried_classes)
            ],
            "oom_vram_floor": self.oom_vram_floor,
            "consumed_attempt": consumed,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> RetryPolicyState:
        if not isinstance(raw, dict) or raw.get("schema_version") != _SCHEMA_VERSION:
            raise RuntimeError("stored retry policy is missing or invalid")
        counters = []
        for key in ("infra_used", "oom_used", "cache_used"):
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError("stored retry policy counter is invalid")
            counters.append(value)
        providers_raw = raw.get("failed_providers")
        if not isinstance(providers_raw, list) or any(
            not isinstance(provider, str) or not provider for provider in providers_raw
        ):
            raise RuntimeError("stored retry provider exclusions are invalid")
        tried_raw = raw.get("tried_classes")
        if not isinstance(tried_raw, list):
            raise RuntimeError("stored retry shape exclusions are invalid")
        tried = set()
        for item in tried_raw:
            if not isinstance(item, dict) or set(item) != {"provider", "gpu", "gpu_count"}:
                raise RuntimeError("stored retry shape exclusion is invalid")
            provider = item["provider"]
            gpu = item["gpu"]
            gpu_count = item["gpu_count"]
            if (
                not isinstance(provider, str)
                or not provider
                or not isinstance(gpu, str)
                or not gpu
                or isinstance(gpu_count, bool)
                or not isinstance(gpu_count, int)
                or gpu_count < 1
            ):
                raise RuntimeError("stored retry shape exclusion is invalid")
            tried.add((provider, gpu, gpu_count))
        floor = raw.get("oom_vram_floor")
        if isinstance(floor, bool) or not isinstance(floor, int | float) or floor < 0:
            raise RuntimeError("stored retry oom floor is invalid")
        consumed_raw = raw.get("consumed_attempt")
        consumed_attempt = None
        consumed_failure = None
        consumed_cache_drop = False
        consumed_retry_allowed = False
        if consumed_raw is not None:
            if not isinstance(consumed_raw, dict) or set(consumed_raw) != {
                "attempt_id",
                "fence",
                "failure",
                "cache_drop",
                "retry_allowed",
            }:
                raise RuntimeError("stored consumed retry attempt is invalid")
            attempt_id = consumed_raw["attempt_id"]
            fence = consumed_raw["fence"]
            failure = consumed_raw["failure"]
            cache_drop = consumed_raw["cache_drop"]
            retry_allowed = consumed_raw["retry_allowed"]
            if (
                isinstance(attempt_id, bool)
                or not isinstance(attempt_id, int)
                or attempt_id < 0
                or isinstance(fence, bool)
                or not isinstance(fence, int)
                or fence < 1
                or not isinstance(failure, str)
                or not failure
                or not isinstance(cache_drop, bool)
                or not isinstance(retry_allowed, bool)
            ):
                raise RuntimeError("stored consumed retry attempt is invalid")
            consumed_attempt = (attempt_id, fence)
            consumed_failure = failure
            consumed_cache_drop = cache_drop
            consumed_retry_allowed = retry_allowed
        return cls(
            infra_used=counters[0],
            oom_used=counters[1],
            cache_used=counters[2],
            failed_providers=frozenset(providers_raw),
            tried_classes=frozenset(tried),
            oom_vram_floor=float(floor),
            consumed_attempt=consumed_attempt,
            consumed_failure=consumed_failure,
            consumed_cache_drop=consumed_cache_drop,
            consumed_retry_allowed=consumed_retry_allowed,
        )


def initial_retry_policy() -> dict:
    return RetryPolicyState().to_dict()


def retry_limits(spec: JobSpec) -> tuple[int, int, int]:
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME
    from flash.runner.supervise.lifecycle import INFRA_RETRY_FLOOR

    max_retries = int(spec.gpu.max_retries)
    infra_retries = max(max_retries, INFRA_RETRY_FLOOR) if max_retries else 0
    cache_fallbacks = (
        1
        if max_retries > 0 and getattr(spec.gpu, "network_volume", None) == WEIGHT_CACHE_VOLUME_NAME
        else 0
    )
    return infra_retries, max_retries, cache_fallbacks


def load_retry_policy(run_id: str) -> RetryPolicyState:
    from flash.runner.lifecycle import state
    from flash.runner.lifecycle import status as status_ops

    raw = status_ops._load_status_json(run_id)
    return RetryPolicyState.from_dict(raw.get(state._RETRY_POLICY_KEY))


def consume_retry(
    run_id: str,
    spec: JobSpec,
    *,
    expected_attempt: tuple[int, int],
    failure: str | None,
    cache_drop: bool,
    allow_retry: bool = True,
    failed_providers: frozenset[str] = frozenset(),
    tried_classes: frozenset[tuple[str, str, int]] = frozenset(),
    oom_vram_floor: float = 0.0,
) -> RetryPolicyState | None:
    """CAS-consume one fenced terminal failure and persist cumulative retry policy."""
    from flash.runner.lifecycle import state
    from flash.runner.lifecycle import status as status_ops
    from flash.runner.lifecycle.protocol import AttemptRecord

    if not isinstance(failure, str) or not failure:
        return None
    with state._status_guard(run_id):
        raw = status_ops._load_status_json(run_id)
        status = status_ops._runstatus_from_json(raw)
        if status.state in state.TERMINAL_STATES or not status.attempt:
            return None
        attempt = AttemptRecord.from_dict(status.attempt)
        if (attempt.attempt_id, attempt.fence) != expected_attempt:
            raise RuntimeError("current attempt changed before retry policy update")
        policy = RetryPolicyState.from_dict(raw.get(state._RETRY_POLICY_KEY))
        if policy.consumed_attempt == expected_attempt:
            if policy.consumed_failure != failure or policy.consumed_cache_drop != cache_drop:
                raise RuntimeError(
                    "current attempt retry policy was consumed by different evidence"
                )
            return policy if policy.consumed_retry_allowed else None
        infra_limit, oom_limit, cache_limit = retry_limits(spec)
        retry_allowed = False
        infra_used = policy.infra_used
        oom_used = policy.oom_used
        cache_used = policy.cache_used
        if cache_drop:
            retry_allowed = allow_retry and policy.cache_used < cache_limit
            if retry_allowed:
                cache_used += 1
        elif failure == "oom":
            retry_allowed = allow_retry and policy.oom_used < oom_limit
            if retry_allowed:
                oom_used += 1
        elif failure in {"no_capacity", "poll_error", "job_preempted"}:
            retry_allowed = allow_retry and policy.infra_used < infra_limit
            if retry_allowed:
                infra_used += 1
        updated = RetryPolicyState(
            infra_used=infra_used,
            oom_used=oom_used,
            cache_used=cache_used,
            failed_providers=(
                policy.failed_providers | failed_providers
                if retry_allowed
                else policy.failed_providers
            ),
            tried_classes=(
                policy.tried_classes | tried_classes if retry_allowed else policy.tried_classes
            ),
            oom_vram_floor=(
                max(policy.oom_vram_floor, float(oom_vram_floor))
                if retry_allowed
                else policy.oom_vram_floor
            ),
            consumed_attempt=expected_attempt,
            consumed_failure=failure,
            consumed_cache_drop=cache_drop,
            consumed_retry_allowed=retry_allowed,
        )
        state._save_status_unlocked(status, _retry_policy=updated.to_dict())
        return updated if retry_allowed else None
