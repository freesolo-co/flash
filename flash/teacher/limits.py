"""Shared bounded OPD teacher-request limits."""

from __future__ import annotations

from typing import Any

OPD_DEFAULT_EPISODE_TURNS = 24
OPD_MIN_EPISODE_TURNS = 8
OPD_MAX_EPISODE_TURNS = 64
OPD_NO_SIGNAL_ATTEMPTS = 3
# measured by sweep against supplied-token scoring: 32 is the highest concurrency that was
# rejection-free in BOTH passes, not the one with the highest raw throughput.
# higher throughput includes fast HTTP 429 responses that drop teacher scores and training signal.
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
