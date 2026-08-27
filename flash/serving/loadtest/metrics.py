"""deterministic summaries for hosted inference request terminal events."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from typing import Any

from flash.serving.loadtest.schema import claim_limitations

_LATENCIES = {
    "scheduling_ms": ("scheduled_ns", "dispatch_ns"),
    "header_ms": ("dispatch_ns", "headers_ns"),
    "ttft_ms": ("dispatch_ns", "first_generated_ns"),
    "visible_ms": ("dispatch_ns", "first_visible_ns"),
    "total_ms": ("dispatch_ns", "completed_ns"),
}
_PERCENTILES = (50, 90, 95, 99)


def summarize_events(
    events: list[dict[str, Any]],
    *,
    fake: bool,
    capacity_expectations: dict[str, bool] | None = None,
) -> dict[str, Any]:
    terminals = [event for event in events if event.get("type") == "request_terminal"]
    if not terminals:
        raise ValueError("events contain no request terminal records")
    capacity = capacity_expectations or {}
    return {
        "schema_version": 1,
        "fake": fake,
        "production_claims_allowed": False if fake else "subject_to_claim_limitations",
        "claim_limitations": claim_limitations(capacity, fake=fake),
        "overall": _group_metrics(terminals),
        "per_phase": _grouped(terminals, lambda event: event["phase_name"]),
        "per_target": _grouped(terminals, lambda event: event["target_name"]),
        "per_profile": _grouped(terminals, lambda event: event["profile_name"]),
        "sustained_mixed_windows": _duration_windows(terminals),
        "overload": _overload_verdicts(terminals, capacity),
    }


def _overload_verdicts(
    events: list[dict[str, Any]], expectations: dict[str, bool]
) -> dict[str, Any]:
    """report per-overload-phase whether capacity rejection was actually demonstrated.

    an overload phase that produced no exact capacity rejection is ``overload_not_demonstrated``,
    never a success. that verdict is the same whether the deployment declares a capacity contract
    or not: what differs is only the reason, because a deployment with no such contract cannot
    express overload over public http at all. an absence is never read as headroom.
    """
    phases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("phase_kind") == "overload":
            phases[event["phase_name"]].append(event)
    result: dict[str, Any] = {}
    for name in sorted(phases):
        rows = phases[name]
        expected = bool(expectations.get(name, False))
        capacity = sum(1 for event in rows if event.get("error_class") == "exact_capacity_503")
        if capacity:
            verdict = "capacity_rejection_observed"
        elif expected:
            verdict = "overload_not_demonstrated_despite_declared_contract"
        else:
            verdict = "overload_not_demonstrated"
        result[name] = {
            "expects_capacity_contract": expected,
            "exact_capacity_503": capacity,
            "other_503": sum(1 for event in rows if event.get("error_class") == "other_503"),
            "http_429": sum(1 for event in rows if event.get("error_class") == "http_429"),
            "verdict": verdict,
        }
    return result


def _grouped(events: list[dict[str, Any]], key: Callable[[dict[str, Any]], str]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        groups[key(event)].append(event)
    return {name: _group_metrics(groups[name]) for name in sorted(groups)}


def _group_metrics(events: list[dict[str, Any]]) -> dict[str, Any]:
    dispatched = [event for event in events if event.get("dispatch_ns") is not None]
    completed = [event for event in dispatched if event.get("completed_ns") is not None]
    successes = [event for event in events if event.get("outcome") == "success"]
    usage = [
        event
        for event in successes
        if event.get("prompt_tokens") is not None and event.get("completion_tokens") is not None
    ]
    offered_seconds, offered_basis = _offered_seconds(events)
    dispatch_seconds = _span_seconds(dispatched, "dispatch_ns")
    completion_seconds = _span_seconds(completed, "completed_ns")
    output_tokens = sum(int(event["completion_tokens"]) for event in usage)
    decode_seconds = sum(_decode_seconds(event) for event in usage)
    statuses = Counter(
        str(event.get("http_status")) for event in events if event.get("http_status")
    )
    errors = Counter(event["error_class"] for event in events if event.get("error_class"))
    target_authored = Counter(event["target_name"] for event in events)
    target_achieved = Counter(event["target_name"] for event in dispatched)
    profile_authored = Counter(event["profile_name"] for event in events)
    profile_achieved = Counter(event["profile_name"] for event in dispatched)
    return {
        "counts": {
            "scheduled": len(events),
            "dispatched": len(dispatched),
            "completed": len(completed),
            "success": len(successes),
            "client_admission_missed": errors.get("client_admission_missed", 0),
        },
        "rates": {
            "offered": _rate(len(events), offered_seconds, offered_basis),
            "dispatch": _rate(len(dispatched), dispatch_seconds, "dispatch_span"),
            "completion": _rate(len(completed), completion_seconds, "completion_span"),
        },
        "latency": {
            name: _latency_metrics(events, start, end) for name, (start, end) in _LATENCIES.items()
        },
        "usage": {
            "covered_requests": len(usage),
            "successful_requests": len(successes),
            "coverage_ratio": len(usage) / len(successes) if successes else None,
            "prompt_tokens": sum(int(event["prompt_tokens"]) for event in usage),
            "completion_tokens": output_tokens,
            "cached_tokens": sum(int(event.get("cached_tokens") or 0) for event in usage),
            "aggregate_output_tokens_per_second": _ratio(output_tokens, _sum_total_seconds(usage)),
            "decode_tokens_per_second": _ratio(output_tokens, decode_seconds),
            "throughput_available": bool(usage),
        },
        "peak_client_in_flight": max(
            (int(event.get("client_in_flight_at_dispatch") or 0) for event in events),
            default=0,
        ),
        "http_status_counts": dict(sorted(statuses.items())),
        "error_counts": dict(sorted(errors.items())),
        "retry_count": sum(int(event.get("retry_count") or 0) for event in events),
        "exact_capacity_ratio": errors.get("exact_capacity_503", 0) / len(dispatched)
        if dispatched
        else None,
        "http_429_count": errors.get("http_429", 0),
        "mix": {
            "target": _mix(target_authored, target_achieved),
            "profile": _mix(profile_authored, profile_achieved),
        },
    }


def _latency_metrics(events: list[dict[str, Any]], start_key: str, end_key: str) -> dict[str, Any]:
    values = sorted(
        (event[end_key] - event[start_key]) / 1_000_000
        for event in events
        if event.get(start_key) is not None and event.get(end_key) is not None
    )
    result: dict[str, Any] = {"count": len(values)}
    for percentile in _PERCENTILES:
        result[f"p{percentile}"] = _percentile(values, percentile)
    result["max"] = values[-1] if values else None
    return result


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    index = max(0, math.ceil((percentile / 100) * len(values)) - 1)
    return values[index]


def _offered_seconds(events: list[dict[str, Any]]) -> tuple[float | None, str]:
    """how long this group of events was offering load, with the basis that produced it.

    a group can span several phases (``overall``, ``per_target``, ``per_profile`` all do), so the
    authored denominator is the sum of the distinct phase windows rather than the longest one.
    dividing a combined count by a single phase's window would inflate offered rps by exactly the
    factor of the phases it ignored.

    a phase without an authored window (warm, an unbounded cold burst) has no authored duration to
    contribute. a group made only of those falls back to the scheduled span; a group that mixes
    both has no defensible denominator and reports none rather than a number built from half the
    traffic.
    """
    windows = {event.get("phase_name"): event.get("authored_window_seconds") for event in events}
    authored = [float(value) for value in windows.values() if value is not None]
    if len(authored) == len(windows):
        return sum(authored), "authored_windows"
    if not authored:
        return _span_seconds(events, "scheduled_ns"), "scheduled_span"
    return None, "undefined_mixed_phase_windows"


def _span_seconds(events: list[dict[str, Any]], key: str) -> float | None:
    values = [int(event[key]) for event in events if event.get(key) is not None]
    if len(values) < 2:
        return None
    return max(0.0, (max(values) - min(values)) / 1_000_000_000)


def _rate(count: int, seconds: float | None, basis: str) -> dict[str, Any]:
    return {
        "count": count,
        "denominator_seconds": seconds,
        "denominator_basis": basis,
        "requests_per_second": _ratio(count, seconds),
    }


def _ratio(numerator: float, denominator: float | None) -> float | None:
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _decode_seconds(event: dict[str, Any]) -> float:
    first = event.get("first_generated_ns")
    completed = event.get("completed_ns")
    if first is None or completed is None or completed <= first:
        return 0.0
    return (completed - first) / 1_000_000_000


def _sum_total_seconds(events: list[dict[str, Any]]) -> float:
    total = 0.0
    for event in events:
        dispatch = event.get("dispatch_ns")
        completed = event.get("completed_ns")
        if dispatch is not None and completed is not None and completed > dispatch:
            total += (completed - dispatch) / 1_000_000_000
    return total


def _mix(authored: Counter, achieved: Counter) -> dict[str, Any]:
    authored_total = sum(authored.values())
    achieved_total = sum(achieved.values())
    names = sorted(set(authored) | set(achieved))
    return {
        name: {
            "authored_count": authored[name],
            "authored_ratio": authored[name] / authored_total if authored_total else None,
            "achieved_count": achieved[name],
            "achieved_ratio": achieved[name] / achieved_total if achieved_total else None,
        }
        for name in names
    }


def _duration_windows(events: list[dict[str, Any]]) -> dict[str, Any]:
    phases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("phase_kind") in {"sustained", "mixed"}:
            phases[event["phase_name"]].append(event)
    result = {}
    for name in sorted(phases):
        values = phases[name]
        result[name] = {
            "authored_seconds": max(float(event["authored_window_seconds"]) for event in values),
            "scheduled_start_ns": min(int(event["scheduled_ns"]) for event in values),
            "scheduled_end_ns": max(int(event["scheduled_ns"]) for event in values),
            "dispatch_start_ns": _minimum(values, "dispatch_ns"),
            "dispatch_end_ns": _maximum(values, "dispatch_ns"),
            "completion_start_ns": _minimum(values, "completed_ns"),
            "completion_end_ns": _maximum(values, "completed_ns"),
        }
    return result


def _minimum(events: list[dict[str, Any]], key: str) -> int | None:
    values = [int(event[key]) for event in events if event.get(key) is not None]
    return min(values) if values else None


def _maximum(events: list[dict[str, Any]], key: str) -> int | None:
    values = [int(event[key]) for event in events if event.get(key) is not None]
    return max(values) if values else None
