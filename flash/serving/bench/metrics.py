"""Per-request records and the capacity arithmetic computed from them.

Metric definitions are fixed here so every cell is reduced the same way, and so the definitions can
be unit-tested against hand-computed values with no GPU.

The rules that make the numbers honest:

* **Failures are never replaced.** The error rate's denominator is every attempt. A benchmark that
  retries a failure and reports the retry has measured a system that does not exist.
* **Throughput uses engine-reported completion tokens** from successful requests only, over the full
  measurement wall time, including time spent on requests that later failed. Dividing by successful
  time only would credit the system for capacity it did not deliver.
* **A request is successful only if it is completely and verifiably successful**: terminal event
  seen, a real finish reason, authoritative usage present, and no hidden prefix-cache hit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from typing import Any

# Normalized failure taxonomy. Everything that is not OK is an error; there is no "partial" bucket,
# because a request that returned some tokens and then broke did not serve the caller.
ERROR_TIMEOUT = "timeout"
ERROR_ENGINE = "engine_error"
ERROR_MALFORMED_STREAM = "malformed_stream"
ERROR_MISSING_USAGE = "missing_usage"
ERROR_NO_FINISH_REASON = "no_finish_reason"
ERROR_CACHE_CONTAMINATED = "cache_contaminated"
ERROR_CACHE_UNVERIFIED = "cache_unverified"
ERROR_PROVENANCE = "provenance_mismatch"
ERROR_TOKEN_MISMATCH = "token_mismatch"
ERROR_CONTEXT_OVERFLOW = "context_overflow"
# The engine received a prompt whose length disagrees with what the fitter measured offline. Such a
# request belongs to a different input-size bucket and must not be averaged into this one.
ERROR_PROMPT_LENGTH = "prompt_length_mismatch"


@dataclass
class RequestRecord:
    """One attempted request. Serialized verbatim into the per-model JSONL evidence file.

    Deliberately holds no generated text: the evidence files are published, and model output is both
    unnecessary for a capacity measurement and a disclosure risk.
    """

    uid: str
    base_model: str
    bucket: str
    concurrency: int
    block: int
    # Monotonic seconds relative to the cell's start.
    started_at: float
    # The assembled length the fitter measured for this prompt, checked against what the engine
    # reports so a mis-sized prompt cannot be counted into the wrong bucket.
    expected_prompt_tokens: int | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    cached_tokens_reported: bool | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    engine_replica_id: str | None = None
    ok: bool = False
    error: str | None = None
    error_detail: str | None = None

    @property
    def latency(self) -> float | None:
        if self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    @property
    def ttft(self) -> float | None:
        """Time to first token: first reasoning-or-content delta."""
        if self.first_token_at is None:
            return None
        return self.first_token_at - self.started_at

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["latency_seconds"] = self.latency
        data["ttft_seconds"] = self.ttft
        return data


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolated percentile; ``q`` in [0, 1]. None for an empty sample."""
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be within [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: list[float], sample_floor_p99: int) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": (sum(values) / len(values)) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        # p99 from a small sample is a single unlucky request, not a tail estimate. Labeling it here
        # keeps the number readable without letting it be quoted as if it were stable.
        "p99_descriptive_only": len(values) < sample_floor_p99,
    }


def wilson_upper_bound(failures: int, attempts: int, z: float = 1.645) -> float:
    """One-sided upper confidence bound on the failure rate (default 95%).

    A cell with 0/20 failures has an observed error rate of 0%, which is not evidence that the true
    rate is under 1%. The feasibility gate uses this bound so "no errors observed" cannot be read as
    "no errors occur" on a small sample.
    """
    if attempts <= 0:
        return 1.0
    phat = failures / attempts
    denominator = 1 + z**2 / attempts
    center = phat + z**2 / (2 * attempts)
    margin = z * math.sqrt(phat * (1 - phat) / attempts + z**2 / (4 * attempts**2))
    return min(1.0, (center + margin) / denominator)


@dataclass
class CellResult:
    """Reduced metrics for one (model, bucket, concurrency, block) cell."""

    base_model: str
    bucket: str
    concurrency: int
    block: int
    # The steady-state measurement window: the span over which `concurrency` requests were held in
    # flight. Every rate below is per THIS second count, not per total elapsed time.
    wall_seconds: float
    attempted: int
    succeeded: int
    failed: int
    # Teardown time after the window closed, at falling concurrency. Reported so a reader can see
    # how much tail was excluded rather than having to trust that some was.
    drain_seconds: float = 0.0
    # Successes that FINISHED inside the window, i.e. the ones the rates below are built from.
    # `succeeded` counts every success including drain completions; this counts the numerator. The
    # two differ by exactly the work that landed in the tail, so a reader can see how much of the
    # cell's output was excluded from the rates rather than inferring it.
    succeeded_in_window: int = 0
    error_breakdown: dict[str, int] = field(default_factory=dict)
    attempted_rps: float = 0.0
    successful_rps: float = 0.0
    output_tokens_per_second: float = 0.0
    total_tokens_per_second: float = 0.0
    prompt_tokens_total: int = 0
    completion_tokens_total: int = 0
    ttft_seconds: dict[str, Any] = field(default_factory=dict)
    latency_seconds: dict[str, Any] = field(default_factory=dict)
    error_rate: float = 0.0
    error_rate_upper_bound: float = 1.0
    # Whether the sample was deep enough for the bound to resolve below max_error_rate AT ALL. A
    # zero-failure cell needs ~268 attempts to bound the rate under 1%; below that the bound is wide
    # for want of data, not because the cell erred. Separating the two keeps "we did not look long
    # enough" from being published as "this cell failed".
    error_bound_resolved: bool = False
    feasible: bool = False
    max_error_rate: float = 0.01
    replica_ids: list[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        """Whether this cell DEMONSTRATED an unacceptable error rate.

        Distinct from ``not feasible``, which is also true for a cell that merely ran too few
        requests. A shallow clean cell is undecided; only an observed rate above the bar, a resolved
        bound that still misses it, or zero successes is evidence of failure.
        """
        if not self.succeeded:
            return True
        if self.error_rate > self.max_error_rate:
            return True
        return self.error_bound_resolved and not self.feasible

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def reduce_cell(
    records: list[RequestRecord],
    *,
    base_model: str,
    bucket: str,
    concurrency: int,
    block: int,
    wall_seconds: float,
    drain_seconds: float = 0.0,
    window_seconds: float | None = None,
    sample_floor_p99: int = 400,
    max_error_rate: float = 0.01,
) -> CellResult:
    """Reduce one cell's request records into its capacity metrics."""
    if wall_seconds <= 0:
        raise ValueError("wall_seconds must be positive")
    attempted = len(records)
    successes = [record for record in records if record.ok]
    failures = [record for record in records if not record.ok]

    # Rates divide by the steady-state window, so their numerators must contain only work that
    # happened INSIDE it. A cell that closes its window with `concurrency` requests still in flight
    # then completes them during the drain: counting those completions against the shorter window
    # credits tail work to steady-state time and inflates RPS and token throughput by roughly the
    # in-flight fraction -- largest exactly where the window is shortest relative to a request, i.e.
    # the near-32k cells whose numbers matter most.
    #
    # `window_seconds=None` means the caller is not distinguishing (a single window with no drain),
    # so every success counts. Records with no `finished_at` cannot be placed and are excluded from
    # the numerator rather than assumed to be in-window.
    in_window = (
        successes
        if window_seconds is None
        else [
            record
            for record in successes
            if record.finished_at is not None and record.finished_at <= window_seconds
        ]
    )

    error_breakdown: dict[str, int] = {}
    for record in failures:
        key = record.error or ERROR_ENGINE
        error_breakdown[key] = error_breakdown.get(key, 0) + 1

    completion_total = sum(record.completion_tokens or 0 for record in in_window)
    prompt_total = sum(record.prompt_tokens or 0 for record in in_window)

    # Latency and TTFT are also window-scoped. A drain completion ran at FALLING concurrency, on a
    # progressively idler engine, so its latency describes a load the cell was not measuring.
    ttft_values = [value for record in in_window if (value := record.ttft) is not None]
    latency_values = [value for record in in_window if (value := record.latency) is not None]

    error_rate = (len(failures) / attempted) if attempted else 0.0
    upper_bound = wilson_upper_bound(len(failures), attempted)
    # The best bound this sample size could produce, i.e. if the cell had zero failures. If even
    # that does not clear the threshold, the sample is too shallow to decide either way.
    bound_resolved = wilson_upper_bound(0, attempted) < max_error_rate

    replica_ids = sorted(
        {record.engine_replica_id for record in records if record.engine_replica_id}
    )

    return CellResult(
        base_model=base_model,
        bucket=bucket,
        concurrency=concurrency,
        block=block,
        wall_seconds=wall_seconds,
        drain_seconds=drain_seconds,
        attempted=attempted,
        succeeded=len(successes),
        failed=len(failures),
        error_breakdown=error_breakdown,
        succeeded_in_window=len(in_window),
        attempted_rps=attempted / wall_seconds,
        successful_rps=len(in_window) / wall_seconds,
        output_tokens_per_second=completion_total / wall_seconds,
        total_tokens_per_second=(completion_total + prompt_total) / wall_seconds,
        prompt_tokens_total=prompt_total,
        completion_tokens_total=completion_total,
        ttft_seconds=_distribution(ttft_values, sample_floor_p99),
        latency_seconds=_distribution(latency_values, sample_floor_p99),
        error_rate=error_rate,
        error_rate_upper_bound=upper_bound,
        error_bound_resolved=bound_resolved,
        max_error_rate=max_error_rate,
        # Feasibility needs real successes AND statistical support, so a cell that ran two requests
        # cleanly cannot be called feasible. Read it with error_bound_resolved: feasible=False with
        # error_bound_resolved=False means undecided, not failed.
        feasible=bool(successes) and upper_bound < max_error_rate,
        replica_ids=replica_ids,
    )


def summarize_curve(
    cells: list[CellResult],
    *,
    knee_fraction: float = 0.95,
    saturation_throughput_gain: float = 0.10,
    saturation_latency_rise: float = 0.25,
) -> dict[str, Any]:
    """Throughput ceiling, knee concurrency, and saturation point for one model+bucket curve.

    * ceiling: highest output tokens/sec among points that did not demonstrate failure.
    * knee: SMALLEST such concurrency reaching ``knee_fraction`` of the ceiling, i.e. the cheapest
      concurrency that buys essentially all the throughput.
    * saturation: the first point that adds less than ``saturation_throughput_gain`` throughput while
      p95 latency rises at least ``saturation_latency_rise``, or that is degraded.

    The curve is built from non-degraded points rather than from strictly feasible ones. A near-32k
    cell cannot afford the ~268 attempts the 1% bound needs, so gating the curve on that bound would
    erase the whole long-context envelope and, worse, report the first shallow cell as the saturation
    point. ``error_bound_resolved`` travels with each cell so a reader can see which throughput
    numbers rest on a resolved error bound and which do not.
    """
    ordered = sorted(cells, key=lambda cell: cell.concurrency)
    usable = [cell for cell in ordered if not cell.degraded]
    feasible = [cell for cell in ordered if cell.feasible]
    if not usable:
        return {
            "throughput_ceiling_tokens_per_second": None,
            "knee_concurrency": None,
            "saturation_concurrency": None,
            "feasible_points": 0,
            "bound_resolved_points": 0,
            "usable_points": 0,
            "measured_points": len(ordered),
        }

    ceiling_cell = max(usable, key=lambda cell: cell.output_tokens_per_second)
    ceiling = ceiling_cell.output_tokens_per_second
    knee = next(
        (
            cell.concurrency
            for cell in usable
            if ceiling > 0 and cell.output_tokens_per_second >= knee_fraction * ceiling
        ),
        None,
    )

    # `pairwise` yields (0,1), (1,2), ... so it only ever TESTS the second element of each pair.
    # The first cell's own degradation would therefore never be seen: a concurrency-1 cell that is
    # already degraded, followed by a usable cell, reported saturation at some later concurrency or
    # None, contradicting the documented definition of "the first degraded point".
    saturation: int | None = None
    if ordered and ordered[0].degraded:
        saturation = ordered[0].concurrency
    pairs = [] if saturation is not None else list(pairwise(ordered))
    for previous, current in pairs:
        if current.degraded:
            saturation = current.concurrency
            break
        previous_tps = previous.output_tokens_per_second
        gain = (
            (current.output_tokens_per_second - previous_tps) / previous_tps
            if previous_tps > 0
            else 0.0
        )
        previous_p95 = previous.latency_seconds.get("p95")
        current_p95 = current.latency_seconds.get("p95")
        rise = (
            (current_p95 - previous_p95) / previous_p95
            if previous_p95 and current_p95 and previous_p95 > 0
            else 0.0
        )
        if gain < saturation_throughput_gain and rise >= saturation_latency_rise:
            saturation = current.concurrency
            break

    return {
        "throughput_ceiling_tokens_per_second": ceiling,
        "throughput_ceiling_concurrency": ceiling_cell.concurrency,
        "knee_concurrency": knee,
        "saturation_concurrency": saturation,
        "feasible_points": len(feasible),
        "bound_resolved_points": sum(1 for cell in ordered if cell.error_bound_resolved),
        "usable_points": len(usable),
        "measured_points": len(ordered),
    }


__all__ = [
    "ERROR_CACHE_CONTAMINATED",
    "ERROR_CONTEXT_OVERFLOW",
    "ERROR_ENGINE",
    "ERROR_MALFORMED_STREAM",
    "ERROR_MISSING_USAGE",
    "ERROR_NO_FINISH_REASON",
    "ERROR_PROMPT_LENGTH",
    "ERROR_PROVENANCE",
    "ERROR_TIMEOUT",
    "ERROR_TOKEN_MISMATCH",
    "CellResult",
    "RequestRecord",
    "percentile",
    "reduce_cell",
    "summarize_curve",
    "wilson_upper_bound",
]
