"""Exact native usage facts for durable hosted-serving settlement."""

import math
from collections.abc import Mapping
from typing import Any

from flash.serving.src.accounting.usage_outbox import (
    ReasoningSettlementUnavailable,
    UsageFacts,
    UsageOutboxError,
)


def exact_reasoning_tokens(result: Mapping[str, Any]) -> int:
    """Return an exact count or refuse to guess from rendered text or delimiters."""
    exact = result.get("reasoning_tokens")
    if isinstance(exact, int) and not isinstance(exact, bool) and exact >= 0:
        return exact
    if result.get("thinking") is False:
        return 0
    raise ReasoningSettlementUnavailable("exact_reasoning_tokens_unavailable")


def _optional_nonnegative_float(value: Any) -> float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return normalized if math.isfinite(normalized) and normalized >= 0 else None


def _optional_nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _optional_bool(value: Any) -> bool | None:
    return value if type(value) is bool else None


def usage_facts(result: Mapping[str, Any]) -> UsageFacts:
    prompt_ids = result.get("prompt_token_ids")
    completion_ids = result.get("completion_token_ids", result.get("token_ids"))
    prompt_tokens = result.get("prompt_tokens")
    completion_tokens = result.get("completion_tokens")
    if isinstance(prompt_ids, list) and prompt_ids:
        prompt_tokens = len(prompt_ids)
    if isinstance(completion_ids, list) and completion_ids and completion_tokens is None:
        completion_tokens = len(completion_ids)
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        raise UsageOutboxError("native_token_ids_unavailable")
    cached = result.get("cached_tokens")
    cached_tokens = int(cached) if isinstance(cached, int) and not isinstance(cached, bool) else 0
    cached_reported = result.get("cached_tokens_reported")
    cached_tokens_reported = type(cached_reported) is bool and cached_reported
    if min(prompt_tokens, completion_tokens, cached_tokens) < 0 or cached_tokens > prompt_tokens:
        raise UsageOutboxError("usage_counters_invalid")
    duration = result.get("inference_time_seconds")
    if duration is not None:
        duration = float(duration)
        if duration < 0:
            raise UsageOutboxError("usage_duration_invalid")
    replica = result.get("engine_replica_id")
    return UsageFacts(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cached_tokens_reported=cached_tokens_reported,
        reasoning_tokens=exact_reasoning_tokens(result),
        generation_duration_seconds=duration,
        time_to_first_token_seconds=_optional_nonnegative_float(
            result.get("time_to_first_token_seconds")
        ),
        queue_wait_seconds=_optional_nonnegative_float(result.get("queue_wait_seconds")),
        replica_in_flight_requests_at_admission=_optional_nonnegative_int(
            result.get("replica_in_flight_requests_at_admission")
        ),
        replica_boot_duration_seconds=_optional_nonnegative_float(
            result.get("replica_boot_duration_seconds")
        ),
        replica_freshly_booted=_optional_bool(result.get("replica_freshly_booted")),
        engine_replica_id=replica if isinstance(replica, str) and replica else None,
    )
