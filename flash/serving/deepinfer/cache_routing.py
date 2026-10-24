"""Experimental cache-local routing policy; not enabled by the hosted router.

Work is expressed in prefill-token equivalents. Callers supply measured decode
costs and verified, tenant-scoped prefix lengths; this module never infers hits.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RankCandidate:
    rank: int
    cached_tokens: int
    queued_work: float
    active_requests: int
    capacity: int


def choose_rank(candidates: list[RankCandidate], prompt_tokens: int, decode_work: float) -> int:
    if prompt_tokens < 1 or not math.isfinite(decode_work) or decode_work < 0:
        raise ValueError("Invalid request work estimate")
    available = []
    for rank in candidates:
        if (
            not 0 <= rank.cached_tokens <= prompt_tokens
            or not math.isfinite(rank.queued_work)
            or rank.queued_work < 0
        ):
            raise ValueError("Invalid cache claim or queued work")
        if not 0 <= rank.active_requests <= rank.capacity:
            raise ValueError("Invalid rank capacity")
        if rank.active_requests < rank.capacity:
            available.append(rank)
    if not available:
        raise OverflowError("All ranks are at capacity")
    return min(
        available,
        key=lambda rank: (
            rank.queued_work + prompt_tokens - rank.cached_tokens + decode_work,
            rank.active_requests,
            rank.rank,
        ),
    ).rank
