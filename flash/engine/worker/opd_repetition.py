"""shared token-id repetition detection and local unlikelihood primitives for opd."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

MIN_REPEAT_COUNT = 3
MIN_REPEAT_TOKENS = 6
MAX_CYCLE_PERIOD = 16
MIN_NGRAM = 3
MAX_NGRAM = 12
REPETITION_WEIGHT_MIN = 0.5
REPETITION_WEIGHT_MAX = 1.5
LOOP_UNLIKELIHOOD_COEF = 0.1


@dataclass(frozen=True)
class RepetitionMatch:
    """one conservative suffix repetition match over token ids."""

    kind: str
    start: int
    end: int
    unit_length: int
    repeats: int

    @property
    def repeated_tokens(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class RepetitionAnalysis:
    """detached repetition metadata and loop-closing token mask."""

    matches: tuple[RepetitionMatch, ...] = ()
    closing_mask: tuple[bool, ...] = ()
    severity: float = 0.0

    @property
    def has_loop(self) -> bool:
        return any(self.closing_mask)


def _token_ids(token_ids: Sequence[int] | None) -> tuple[int, ...]:
    if not token_ids:
        return ()
    return tuple(int(token_id) for token_id in token_ids)


def detect_token_cycle(
    token_ids: Sequence[int] | None,
    *,
    min_repeats: int = MIN_REPEAT_COUNT,
    min_repeat_tokens: int = MIN_REPEAT_TOKENS,
    max_period: int = MAX_CYCLE_PERIOD,
) -> RepetitionMatch | None:
    """detect the shortest repeated suffix cycle without decoding text."""

    ids = _token_ids(token_ids)
    repeats = max(MIN_REPEAT_COUNT, int(min_repeats))
    min_tokens = max(1, int(min_repeat_tokens))
    max_unit = min(max(1, int(max_period)), len(ids) // repeats)
    for period in range(1, max_unit + 1):
        span = period * repeats
        unit = ids[-period:]
        if unit * repeats != ids[-span:]:
            continue
        actual_repeats = repeats
        while (actual_repeats + 1) * period <= len(ids):
            start = len(ids) - (actual_repeats + 1) * period
            if ids[start : start + period] != unit:
                break
            actual_repeats += 1
        if actual_repeats * period < min_tokens:
            continue
        return RepetitionMatch(
            kind="cycle",
            start=len(ids) - actual_repeats * period,
            end=len(ids),
            unit_length=period,
            repeats=actual_repeats,
        )
    return None


def detect_repeated_ngram(
    token_ids: Sequence[int] | None,
    *,
    min_repeats: int = MIN_REPEAT_COUNT,
    min_ngram: int = MIN_NGRAM,
    max_ngram: int = MAX_NGRAM,
) -> RepetitionMatch | None:
    """detect a repeated suffix ngram, requiring three non-overlapping occurrences."""

    ids = _token_ids(token_ids)
    repeats = max(MIN_REPEAT_COUNT, int(min_repeats))
    lo = max(2, int(min_ngram))
    hi = min(max(lo, int(max_ngram)), len(ids) // repeats)
    for ngram_size in range(lo, hi + 1):
        needle = ids[-ngram_size:]
        starts = [
            start
            for start in range(0, len(ids) - ngram_size + 1)
            if ids[start : start + ngram_size] == needle
        ]
        non_overlapping: list[int] = []
        for start in starts:
            if not non_overlapping or start >= non_overlapping[-1] + ngram_size:
                non_overlapping.append(start)
        if len(non_overlapping) < repeats or non_overlapping[-1] != len(ids) - ngram_size:
            continue
        selected = non_overlapping[-repeats:]
        return RepetitionMatch(
            kind="ngram",
            start=selected[0],
            end=len(ids),
            unit_length=ngram_size,
            repeats=len(selected),
        )
    return None


def analyze_repetition(token_ids: Sequence[int] | None) -> RepetitionAnalysis:
    """return suffix-loop matches and positions whose sampled tokens close the loop."""

    ids = _token_ids(token_ids)
    if len(ids) < MIN_REPEAT_TOKENS:
        return RepetitionAnalysis(closing_mask=(False,) * len(ids))
    cycle = detect_token_cycle(ids)
    ngram = None if cycle is not None else detect_repeated_ngram(ids)
    matches = tuple(match for match in (cycle, ngram) if match is not None)
    mask = [False] * len(ids)
    for match in matches:
        final_start = max(match.start, match.end - match.unit_length)
        for index in range(final_start, match.end):
            mask[index] = True
    repeated = max((match.repeated_tokens for match in matches), default=0)
    severity = min(1.0, repeated / max(1, len(ids)))
    return RepetitionAnalysis(matches=matches, closing_mask=tuple(mask), severity=severity)


def normalize_repetition_weights(
    severities: Sequence[float],
    *,
    lower: float = REPETITION_WEIGHT_MIN,
    upper: float = REPETITION_WEIGHT_MAX,
) -> tuple[float, ...]:
    """downweight repetitive sequences while preserving bounded mean-one weights."""

    if not severities:
        return ()
    lo = float(lower)
    hi = float(upper)
    if not 0.0 < lo <= 1.0 <= hi:
        raise ValueError("repetition weight bounds must satisfy 0 < lower <= 1 <= upper")
    raw = [1.0 - 0.5 * min(1.0, max(0.0, float(value))) for value in severities]
    weights = [min(hi, max(lo, value)) for value in raw]
    target = float(len(weights))
    for _ in range(8):
        delta = target - sum(weights)
        if abs(delta) <= 1e-12:
            break
        adjustable = [
            index
            for index, value in enumerate(weights)
            if (delta > 0 and value < hi) or (delta < 0 and value > lo)
        ]
        if not adjustable:
            break
        share = delta / len(adjustable)
        for index in adjustable:
            weights[index] = min(hi, max(lo, weights[index] + share))
    return tuple(weights)


def loop_closing_unlikelihood(
    rows: Any,
    token_ids: Sequence[int] | None,
    *,
    forced: Sequence[bool] = (),
    coef: float = LOOP_UNLIKELIHOOD_COEF,
    analysis: RepetitionAnalysis | None = None,
):
    """penalize sampled tokens that close a confirmed local repetition loop."""

    import torch
    import torch.nn.functional as F

    ids = _token_ids(token_ids)
    if rows is None or not ids or float(coef) <= 0:
        return None
    analysis = analysis or analyze_repetition(ids)
    selected = [
        index
        for index, closes in enumerate(analysis.closing_mask)
        if closes and index < rows.shape[0] and not (index < len(forced) and bool(forced[index]))
    ]
    if not selected:
        return None
    index_t = torch.tensor(selected, dtype=torch.long, device=rows.device)
    target_t = torch.tensor(
        [ids[index] for index in selected], dtype=torch.long, device=rows.device
    )
    probabilities = F.softmax(rows.index_select(0, index_t).float(), dim=-1)
    repeated_probability = probabilities.gather(1, target_t.unsqueeze(1)).squeeze(1)
    loss = -torch.log1p(-repeated_probability.clamp(max=1.0 - 1e-6)).mean()
    return float(coef) * loss
