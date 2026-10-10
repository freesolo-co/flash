"""Typed terminal reward scoring for multi-turn GRPO rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass

from flash.envs.loading.base import BaseEnvironment, RolloutReward


@dataclass(frozen=True)
class RolloutScoreRequest:
    example: dict
    state: dict
    turn_count: int


def _validated_reward(reward: RolloutReward, request: RolloutScoreRequest) -> RolloutReward:
    episode = float(reward.episode)
    # a non-finite episode reward is unscorable and has no valid scalar fallback. canonicalize it to
    # nan, the ONLY unscorable marker: torch.isnan excludes the row from the group baseline and
    # nan_to_num then zeros its advantage, matching stock grpo. forwarding a raw
    # inf instead would NOT be recognized as unscorable and would contaminate the whole group with
    # huge advantages. per-turn credit is disabled so the unscorable row is never revived.
    if not math.isfinite(episode):
        print(
            "[grpo][warn] episode reward non-finite; rollout unscorable, per-turn credit disabled"
        )
        return RolloutReward(episode=float("nan"), turns=None)
    if reward.turns is None:
        return RolloutReward(episode=episode, turns=None)
    if not isinstance(reward.turns, (list, tuple)):
        # only an ordered list/tuple carries a well-defined per-turn order. a str, bytes,
        # bytearray, mapping, or unordered set would still iterate into floats and could pass
        # the count check while assigning rewards to the wrong turns -- reject and fall back.
        print(
            "[grpo][warn] per-turn rewards unavailable (turns is not an ordered list/tuple); "
            "using episode reward"
        )
        return RolloutReward(episode=episode, turns=None)

    reason: str | None = None
    coerced: tuple[float, ...] | None = None
    try:
        coerced = tuple(float(value) for value in reward.turns)
    except (TypeError, ValueError):
        reason = "per-turn rewards contain a non-number"
    else:
        if len(coerced) != request.turn_count:
            reason = f"received {len(coerced)} reward(s) for {request.turn_count} assistant turn(s)"
        elif not all(math.isfinite(value) for value in coerced):
            reason = "per-turn rewards contain a non-finite value"

    if reason is not None:
        print(f"[grpo][warn] per-turn rewards unavailable ({reason}); using episode reward")
        return RolloutReward(episode=episode, turns=None)
    return RolloutReward(episode=episode, turns=coerced)


def score_rollouts(active_env, requests: list[RolloutScoreRequest]) -> list[RolloutReward]:
    """Score terminal rollout states once and return normalized typed rewards."""
    items = [(request.example, request.state) for request in requests]
    rollout_rewards_many = getattr(active_env, "rollout_rewards_many", None)
    if callable(rollout_rewards_many):
        rewards = rollout_rewards_many(items)
    else:
        rewards = BaseEnvironment.rollout_rewards_many(active_env, items)
    if len(rewards) != len(requests):
        raise RuntimeError("env.rollout_rewards_many returned the wrong number of rewards")
    return [
        _validated_reward(reward, request)
        for reward, request in zip(rewards, requests, strict=True)
    ]
