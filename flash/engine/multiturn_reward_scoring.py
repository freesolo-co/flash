"""Typed terminal reward scoring for multi-turn GRPO rollouts."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    rollout_rewards_many = getattr(active_env, "rollout_rewards_many", None)
    if callable(rollout_rewards_many):
        rewards = rollout_rewards_many(items)
        if len(rewards) != len(requests):
            raise RuntimeError("env.rollout_rewards_many returned the wrong number of rewards")
        return [
            _validated_reward(reward, request)
            for reward, request in zip(rewards, requests, strict=True)
        ]

    reward_many = getattr(active_env, "reward_many", None)
    if callable(reward_many):
        episode_rewards = reward_many(items)
        if len(episode_rewards) != len(requests):
            raise RuntimeError("env.reward_many returned the wrong number of rewards")
    else:

        def score_one(request: RolloutScoreRequest) -> float:
            return float(active_env.reward("", request.example, request.state))

        if len(requests) <= 1 or not getattr(active_env, "reward_thread_safe", True):
            episode_rewards = [score_one(request) for request in requests]
        else:
            pool = ThreadPoolExecutor(max_workers=min(16, len(requests)))
            try:
                futures = {
                    pool.submit(score_one, request): index for index, request in enumerate(requests)
                }
                episode_rewards = [0.0] * len(requests)
                for future in as_completed(futures):
                    episode_rewards[futures[future]] = future.result()
            finally:
                pool.shutdown(wait=True, cancel_futures=True)

    return [RolloutReward(episode=float(episode_reward)) for episode_reward in episode_rewards]
