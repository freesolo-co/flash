"""Normalize measured metrics into an agent-skill achievement score.

The headline number is **improvement-normalized**:

    achievement = clamp((agent - base) / (paper - base), 0, 1)

It credits the *delta the agent's training produced* relative to the paper's delta over the
same untrained base. That is robust to the harness using a different (Flash-supported) base
model than the paper: an absolute ratio would punish a smaller base unfairly, but the
improvement ratio asks "did the agent close the same fraction of the gap?" For
lower-is-better metrics the deltas are flipped so improvement is always positive-good.

A noise band (``1.96*sqrt(p(1-p)/N)`` for a rate metric over ``N`` eval rows) is reported
alongside; an agent-vs-base gap inside the band is "no signal", not a win.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Score:
    """A scored result: the headline achievement plus the components behind it."""

    achievement: float | None
    ratio: float | None
    base_metric: float
    agent_metric: float
    paper_metric: float
    noise_band: float | None = None
    note: str = ""


def _orient(value: float, higher_is_better: bool) -> float:
    return value if higher_is_better else -value


def improvement_normalized(
    agent: float, base: float, paper: float, *, higher_is_better: bool = True
) -> float | None:
    """``(agent - base) / (paper - base)`` clamped to [0, 1]; None when the denominator is ~0.

    A near-zero denominator means the paper reported no improvement over the base (or a base
    measurement collided with the paper number) — the ratio is undefined, so return None and
    let the caller fall back to the raw ratio / raw numbers.
    """
    a, b, p = (_orient(x, higher_is_better) for x in (agent, base, paper))
    denom = p - b
    if abs(denom) < 1e-9:
        return None
    return max(0.0, min(1.0, (a - b) / denom))


def ratio(agent: float, paper: float, *, higher_is_better: bool = True) -> float | None:
    """``agent / paper`` (clamped to [0, 1]); None when the paper number is ~0.

    For lower-is-better metrics this inverts to ``paper / agent`` so 1.0 still means "matched
    the paper". A lower-is-better ``agent`` of ~0 is the *best possible* result (e.g. zero error
    rate), so it returns 1.0 (matched/exceeded), not ``None`` — division by ~0 there would be a
    perfect score, not an undefined one.
    """
    if higher_is_better:
        if abs(paper) < 1e-9:
            return None
        return max(0.0, min(1.0, agent / paper))
    if abs(agent) < 1e-9:
        return 1.0
    return max(0.0, min(1.0, paper / agent))


def noise_band(p: float, n: int) -> float | None:
    """95% sampling half-width of a rate metric ``p`` over ``n`` examples; None if n<=0."""
    if n <= 0:
        return None
    p = max(0.0, min(1.0, p))
    return 1.96 * math.sqrt(p * (1 - p) / n)


def score(
    *,
    agent_metric: float,
    base_metric: float,
    paper_metric: float,
    higher_is_better: bool = True,
    eval_n: int | None = None,
) -> Score:
    """Build the full ``Score`` from measured numbers."""
    ach = improvement_normalized(
        agent_metric, base_metric, paper_metric, higher_is_better=higher_is_better
    )
    rat = ratio(agent_metric, paper_metric, higher_is_better=higher_is_better)
    band = noise_band(agent_metric, eval_n) if eval_n else None
    note = ""
    if ach is None:
        note = "improvement-normalized score undefined (paper ~= base); see raw ratio"
    elif band is not None and abs(agent_metric - base_metric) <= band:
        note = "agent-vs-base gap within the noise band — treat as no signal"
    return Score(
        achievement=ach,
        ratio=rat,
        base_metric=base_metric,
        agent_metric=agent_metric,
        paper_metric=paper_metric,
        noise_band=band,
        note=note,
    )
