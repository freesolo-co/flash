"""GRPO trainer support for group-relative per-turn credit assignment."""

from __future__ import annotations

import math
from typing import Any

import torch
from trl import GRPOTrainer


def build_per_turn_advantages(
    turn_spans_per_completion: list[list[tuple[int, int]]],
    turn_rewards_per_completion: list[list[float] | None],
    num_generations: int,
    completion_len: int,
    *,
    episode_advantages: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build token-aligned advantages from per-turn rewards in consecutive GRPO groups."""
    batch_size = len(turn_spans_per_completion)
    if len(turn_rewards_per_completion) != batch_size:
        raise ValueError("turn span and reward row counts must match")
    if num_generations <= 0 or batch_size % num_generations != 0:
        raise ValueError("batch size must be divisible by num_generations")
    if completion_len < 0:
        raise ValueError("completion_len must be non-negative")

    if episode_advantages is None:
        advantages = torch.zeros((batch_size, completion_len), dtype=torch.float32)
    else:
        if episode_advantages.dim() != 1 or episode_advantages.numel() != batch_size:
            raise ValueError("episode_advantages must have shape [B]")
        advantages = episode_advantages.new_zeros((batch_size, completion_len))

    normalized_spans: list[list[tuple[int, int]]] = []
    normalized_rewards: list[list[float] | None] = []
    for row, (spans, rewards) in enumerate(
        zip(turn_spans_per_completion, turn_rewards_per_completion, strict=True)
    ):
        normalized_row = [(int(start), int(end)) for start, end in spans]
        for start, end in normalized_row:
            if not 0 <= start <= end <= completion_len:
                raise ValueError(
                    f"turn span [{start}, {end}) for row {row} exceeds completion width "
                    f"{completion_len}"
                )
        normalized_spans.append(normalized_row)

        normalized_reward_row: list[float] | None = None
        if rewards is not None:
            try:
                candidate = [float(reward) for reward in rewards]
            except (TypeError, ValueError):
                candidate = []
            if len(candidate) == len(normalized_row) and all(
                math.isfinite(reward) for reward in candidate
            ):
                normalized_reward_row = candidate
        if normalized_reward_row is None and episode_advantages is None:
            raise ValueError("episode_advantages are required for unusable turn rewards")
        normalized_rewards.append(normalized_reward_row)

    for group_start in range(0, batch_size, num_generations):
        group_end = group_start + num_generations
        group_rewards = normalized_rewards[group_start:group_end]
        if any(rewards is None for rewards in group_rewards):
            if episode_advantages is None:
                raise ValueError("episode_advantages are required for group fallback")
            for row in range(group_start, group_end):
                completion_end = normalized_spans[row][-1][1] if normalized_spans[row] else 0
                advantages[row, :completion_end] = episode_advantages[row]
            continue

        max_turns = max(
            (len(rewards) for rewards in group_rewards if rewards is not None), default=0
        )
        for turn_index in range(max_turns):
            members = [
                row
                for row in range(group_start, group_end)
                if turn_index < len(normalized_rewards[row])
            ]
            mean_reward = sum(normalized_rewards[row][turn_index] for row in members) / len(members)
            for row in members:
                reward = normalized_rewards[row][turn_index]
                start, end = normalized_spans[row][turn_index]
                advantages[row, start:end] = reward - mean_reward

    if not bool(torch.isfinite(advantages).all()):
        raise ValueError("per-turn advantages must be finite")
    return advantages


class GRPOPerTurnTrainer(GRPOTrainer):
    """Replace scalar GRPO advantages with aligned per-turn advantages when supplied."""

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        output = super()._generate_and_score_completions(inputs)
        turn_rewards = [item.get("turn_rewards") for item in inputs]
        if not any(rewards is not None for rewards in turn_rewards):
            return output
        if self.accelerator.num_processes > 1:
            raise NotImplementedError(
                "per-turn GRPO advantages currently support single-process training only; "
                "distributed group centering requires gather-aligned turn metadata"
            )

        turn_spans = [item.get("turn_spans") for item in inputs]
        if any(spans is None for spans in turn_spans):
            raise ValueError("per-turn rollout rows must all include turn_spans")
        scalar_advantages = output["advantages"]
        if scalar_advantages.dim() != 1:
            raise ValueError("expected TRL scalar advantages with shape [B]")
        batch_size, completion_len = output["completion_ids"].shape
        if len(inputs) != batch_size:
            raise ValueError(
                f"per-turn metadata has {len(inputs)} row(s) for output batch size {batch_size}"
            )

        num_generations = self.num_generations if self.model.training else self.num_generations_eval
        output["advantages"] = build_per_turn_advantages(
            turn_spans,
            turn_rewards,
            num_generations,
            completion_len,
            episode_advantages=scalar_advantages,
        ).to(device=scalar_advantages.device, dtype=scalar_advantages.dtype)
        return output
