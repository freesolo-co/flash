"""Quality targets, overlap-aware time accounting, and independent money ledgers."""

from __future__ import annotations

import math
from collections import defaultdict

from flash.benchmarks.evidence import digest
from flash.benchmarks.schema import (
    LEDGERS,
    PHASES,
    Cache,
    Case,
    Complete,
    Money,
    Provenance,
    Quality,
    Request,
    Span,
)


def percentile(values: list[float], quantile: float) -> float | None:
    """Nearest-rank quantiles retain actual observed latencies for small samples."""
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(quantile * len(values)) - 1)]


def timeline(spans: list[Span], end_s: float) -> list[tuple[float, float, set[str]]]:
    """Sweep phase boundaries; nested spans of the same phase count only once."""
    changes: dict[float, list[tuple[str, int]]] = defaultdict(list)
    changes[0.0]
    changes[end_s]
    for span in spans:
        if span.end_s > span.start_s:
            changes[span.start_s].append((span.phase, 1))
            changes[span.end_s].append((span.phase, -1))
    active: dict[str, int] = defaultdict(int)
    result = []
    previous = 0.0
    for at_s, deltas in sorted(changes.items()):
        if at_s > previous:
            result.append((previous, at_s, {phase for phase, n in active.items() if n}))
        for phase, delta in deltas:
            active[phase] += delta
        previous = at_s
    return result


def timing_summary(spans: list[Span], end_s: float) -> dict:
    inclusive = dict.fromkeys(PHASES, 0.0)
    exclusive = dict.fromkeys(PHASES, 0.0)
    overlap = unattributed = 0.0
    for start, end, phases in timeline(spans, end_s):
        duration = end - start
        for phase in phases:
            inclusive[phase] += duration
        if len(phases) == 1:
            exclusive[next(iter(phases))] += duration
        elif phases:
            overlap += duration
        else:
            unattributed += duration
    measured = {span.phase for span in spans}
    return {
        "wall_s": end_s,
        "phase_busy_s": {
            phase: inclusive[phase] if phase in measured else None for phase in PHASES
        },
        "phase_exclusive_s": {
            phase: exclusive[phase] if phase in measured else None for phase in PHASES
        },
        "overlap_s": overlap,
        "unattributed_s": unattributed,
        "missing_phases": sorted(set(PHASES) - measured),
    }


def _money_at_cutoff(rows: list[Money], cutoff: float) -> dict:
    lower = upper = allocated = 0.0
    for row in rows:
        if row.end_s <= cutoff:
            lower += row.amount_usd
            upper += row.amount_usd
            allocated += row.amount_usd
        elif row.start_s < cutoff:
            upper += row.amount_usd
            allocated += row.amount_usd * (cutoff - row.start_s) / (row.end_s - row.start_s)
    return {
        "lower_usd": lower,
        "upper_usd": upper,
        "uniform_interval_allocation_usd": allocated,
    }


def _phase_cost(rows: list[Money], spans: list[Span], end_s: float) -> dict:
    """An allocation of receipt totals, not measured phase-specific vendor charges.

    Uniform accrual inside each receipt interval; simultaneous phases share equally. Point
    charges remain unattributed. The allocations always reconcile to the ledger's known total.
    """
    totals = dict.fromkeys((*PHASES, "unattributed"), 0.0)
    segments = timeline(spans, end_s)
    for row in rows:
        duration = row.end_s - row.start_s
        if not duration:
            totals["unattributed"] += row.amount_usd
            continue
        for start, end, phases in segments:
            overlap = max(0.0, min(end, row.end_s) - max(start, row.start_s))
            owners = phases or {"unattributed"}
            for phase in owners:
                totals[phase] += row.amount_usd * overlap / duration / len(owners)
    return totals


def money_summary(events, spans, complete, end_s: float, target_s: float | None) -> dict:
    result = {}
    for ledger in LEDGERS:
        rows = [event for event in events if isinstance(event, Money) and event.ledger == ledger]
        settled = complete is not None and ledger in complete.complete_ledgers
        total = sum(row.amount_usd for row in rows)
        result[ledger] = {
            "complete": settled,
            "known_usd": total,
            "total_usd": total if settled else None,
            "to_target": _money_at_cutoff(rows, target_s)
            if settled and target_s is not None
            else None,
            "phase_allocation_usd": _phase_cost(rows, spans, end_s) if rows else None,
        }
    return result


def request_summary(requests: list[Request]) -> dict:
    groups = defaultdict(list)
    for request in requests:
        groups[(request.customer, request.adapter)].append(request)
    rows = []
    for (customer, adapter), group in sorted(groups.items()):
        successful = [item for item in group if item.success]
        latencies = [item.end_s - item.start_s for item in successful]
        ttft = [item.ttft_s for item in successful if item.ttft_s is not None]
        rows.append(
            {
                "customer": customer,
                "adapter": adapter,
                "requests": len(group),
                "successes": len(successful),
                "failures": len(group) - len(successful),
                "latency_s": {f"p{q}": percentile(latencies, q / 100) for q in (50, 95, 99)},
                "ttft_s": {f"p{q}": percentile(ttft, q / 100) for q in (50, 95, 99)},
                "ttft_samples": len(ttft),
                "failed_latency_s": [
                    item.end_s - item.start_s for item in group if not item.success
                ],
            }
        )
    # Demonstrate overlapping requests across different customers AND adapters, not just a
    # declared concurrency parameter. A sweep avoids an O(n^2) scan on large load tests.
    boundaries = []
    for request in requests:
        if request.success and request.end_s > request.start_s:
            key = (request.customer, request.adapter)
            boundaries.extend([(request.start_s, 1, key), (request.end_s, -1, key)])
    active: dict[tuple[str, str], int] = defaultdict(int)
    concurrent = False
    for _, delta, key in sorted(boundaries):
        active[key] += delta
        if not active[key]:
            del active[key]
        if len({item[0] for item in active}) > 1 and len({item[1] for item in active}) > 1:
            concurrent = True
    return {"by_customer_adapter": rows, "concurrent_customers_and_adapters_observed": concurrent}


def summarize_case(
    case: Case, request: dict, events: list, *, process_ok: bool, observed_end_s: float = 0.0
) -> dict:
    completions = [event for event in events if isinstance(event, Complete)]
    if len(completions) > 1 or (completions and events[-1] is not completions[0]):
        raise ValueError("complete must occur once, as the final event")
    complete = completions[0] if completions else None
    end_s = (
        complete.at_s
        if complete
        else max(
            (getattr(event, "end_s", getattr(event, "at_s", 0.0)) for event in events), default=0.0
        )
    )
    if complete is None:
        end_s = max(end_s, observed_end_s)
    if any(getattr(event, "end_s", getattr(event, "at_s", 0.0)) > end_s for event in events):
        raise ValueError("evidence extends beyond completion")
    receipts = [(e.ledger, e.receipt, e.line_item) for e in events if isinstance(e, Money)]
    if len(set(receipts)) != len(receipts):
        raise ValueError("duplicate receipt line item in a ledger")
    provenance = [event for event in events if isinstance(event, Provenance)]
    caches = [event for event in events if isinstance(event, Cache)]
    if len(provenance) > 1 or len(caches) > 1:
        raise ValueError("one provenance and cache observation are allowed per trial")
    provenance_ok = bool(
        provenance
        and provenance[0].revisions == case.revisions
        and provenance[0].configuration_digest == digest(request["configuration"])
        and provenance[0].resolved_configuration.seed == request["seed"]
        and provenance[0].resolved_configuration.cache == request["cache"]
    )
    cache_ok = bool(caches and caches[0].observed == request["cache"])
    qualities = sorted(
        (e for e in events if isinstance(e, Quality) and e.metric == case.target.metric),
        key=lambda e: e.at_s,
    )
    reached = next(
        (
            e
            for e in qualities
            if e.samples >= case.target.min_samples
            and (
                e.value >= case.target.value
                if case.target.direction == "at_least"
                else e.value <= case.target.value
            )
        ),
        None,
    )
    spans = [event for event in events if isinstance(event, Span)]
    requests = request_summary([event for event in events if isinstance(event, Request)])
    issues = []
    if not process_ok or complete is None or complete.outcome != "succeeded":
        issues.append("execution_incomplete_or_failed")
    if not provenance_ok:
        issues.append("missing_or_mismatched_runtime_provenance")
    if not cache_ok:
        issues.append("cache_state_unattested_or_mismatched")
    if not any(e.samples >= case.target.min_samples for e in qualities):
        issues.append("insufficient_held_out_evaluation")
    if (
        case.workload == "concurrent_adapters"
        and not requests["concurrent_customers_and_adapters_observed"]
    ):
        issues.append("concurrent_customers_and_adapters_not_observed")
    timing = timing_summary(spans, end_s)
    money = money_summary(events, spans, complete, end_s, reached.at_s if reached else None)
    gaps: list[str] = sorted(set(case.required_phases) & set(timing["missing_phases"]))
    gaps.extend(
        ledger
        for ledger in ("infrastructure_actual", "customer_charge")
        if not money[ledger]["complete"]
    )
    return {
        "case": case.name,
        "workload": case.workload,
        "seed": request["seed"],
        "cache_requested": request["cache"],
        "cache_observed": caches[0].observed if caches else "unknown",
        "valid": not issues,
        "issues": issues,
        "outcome": complete.outcome if complete else "incomplete",
        "target": case.target.model_dump(),
        "target_reached": reached is not None,
        "time_to_target_s": reached.at_s if reached else None,
        "target_checkpoint": reached.checkpoint if reached else None,
        "quality_curve": [event.model_dump() for event in qualities],
        "resolved_configuration_digest": digest(
            provenance[0].resolved_configuration.model_dump(mode="json")
        )
        if provenance
        else None,
        "timing": timing,
        "money": money,
        "accounting_complete": not gaps,
        "measurement_gaps": gaps,
        "requests": requests,
    }
