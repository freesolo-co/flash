"""Shared bounded OPD teacher-request limits."""

from __future__ import annotations

from typing import Any

OPD_DEFAULT_EPISODE_TURNS = 24
OPD_MIN_EPISODE_TURNS = 8
OPD_MAX_EPISODE_TURNS = 64
OPD_NO_SIGNAL_ATTEMPTS = 3
# measured against the managed teacher endpoint with the supplied-token scoring contract this path
# actually issues (echo + logprobs, max_tokens=0), 160 requests per level, two interleaved passes:
#
#     concurrency   throughput     rejected (HTTP 429)
#      8 (previous)  ~15 req/s      0
#     32             ~35-44 req/s   0        <- highest level clean in BOTH passes
#     40             ~53-55 req/s   0 / 6
#     48             ~61 req/s      1 / 10
#     64             ~70-74 req/s   32 / 32
#
# 32 is the ceiling that is rejection-free, not the one with the highest raw throughput. above it
# the provider sheds requests, and a shed request is a LOST TEACHER SCORE -- the OPD path drops
# training signal rather than going faster. raw requests-per-second keeps climbing past 32 only
# because rejections return quickly, so throughput alone would pick a value that trains on less data.
OPD_TEACHER_SCORING_CONCURRENCY = 32


def configured_opd_turn_limit(environment: Any) -> tuple[bool, int | None]:
    params = getattr(environment, "params", None)
    params = params if isinstance(params, dict) else {}
    raw_turns = params.get("max_turns")
    if params.get("multi_turn") is True or raw_turns is not None:
        if isinstance(raw_turns, bool) or not isinstance(raw_turns, int):
            return True, None
        return True, max(OPD_MIN_EPISODE_TURNS, min(OPD_MAX_EPISODE_TURNS, raw_turns))
    if params.get("multi_turn") is False:
        return False, None
    environment_id = getattr(environment, "id", "")
    if isinstance(environment_id, str) and environment_id.strip():
        return True, None
    return False, None


def opd_teacher_request_multiplier(*, multi_turn: bool, max_turns: int | None) -> int:
    turns = 1
    if multi_turn:
        turns = OPD_MAX_EPISODE_TURNS if max_turns is None else max_turns
        turns = max(OPD_MIN_EPISODE_TURNS, min(OPD_MAX_EPISODE_TURNS, int(turns)))
    return turns * OPD_NO_SIGNAL_ATTEMPTS
