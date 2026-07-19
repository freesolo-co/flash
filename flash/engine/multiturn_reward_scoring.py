"""Typed terminal reward scoring for multi-turn GRPO rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass

from flash.envs.base import RolloutReward


@dataclass(frozen=True)
class RolloutScoreRequest:
    example: dict
    state: dict
    turn_count: int


def _validated_reward(reward: RolloutReward, request: RolloutScoreRequest) -> RolloutReward:
    episode = float(reward.episode)
    reason: str | None = None
    turns: tuple[float, ...] | None = None

    if not math.isfinite(episode):
        reason = "episode reward is non-finite"
    elif reward.turns is not None:
        try:
            turns = tuple(float(value) for value in reward.turns)
        except (TypeError, ValueError):
            reason = "per-turn rewards contain a non-number"
        else:
            if len(turns) != request.turn_count:
                reason = (
                    f"received {len(turns)} reward(s) for {request.turn_count} assistant turn(s)"
                )
            elif not all(math.isfinite(value) for value in turns):
                reason = "per-turn rewards contain a non-finite value"

    if reason is not None:
        print(f"[grpo][warn] per-turn rewards unavailable ({reason}); using episode reward")
        turns = None
    return RolloutReward(episode=episode, turns=turns)


def score_rollouts(active_env, requests: list[RolloutScoreRequest]) -> list[RolloutReward]:
    """Score terminal rollout states once and return normalized typed rewards."""
    items = [(request.example, request.state) for request in requests]
    rewards = active_env.rollout_rewards_many(items)
    if len(rewards) != len(requests):
        raise RuntimeError("env.rollout_rewards_many returned the wrong number of rewards")
    return [
        _validated_reward(reward, request)
        for reward, request in zip(rewards, requests, strict=True)
    ]
