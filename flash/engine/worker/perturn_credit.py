"""Backend-neutral group-relative per-turn credit assignment for multi-turn GRPO.

The per-turn advantage math lives here, independent of any training backend, so the TRL per-turn
trainer (:mod:`flash.engine.worker.grpo_perturn_trainer`) and the OpenRLHF multi-turn path
(:mod:`flash.engine.worker.openrlhf_multiturn`) share a single source of truth instead of two
copies that can drift. Given the per-turn action spans and per-turn rewards of the completions in
consecutive GRPO groups, it centers each turn's reward against that turn's group members and writes
the group-relative advantage onto exactly that turn's tokens. It holds no backend state beyond the
tensors it is handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch


@dataclass(frozen=True)
class TurnCreditRow:
    spans: tuple[tuple[int, int], ...]
    turns: tuple[float, ...] | None


def build_per_turn_advantages(
    turn_spans_per_completion: list[list[tuple[int, int]]],
    turn_rewards_per_completion: list[list[float] | None],
    num_generations: int,
    completion_len: int,
    *,
    episode_advantages: torch.Tensor,
) -> torch.Tensor:
    """Build token-aligned advantages from per-turn rewards in consecutive GRPO groups."""
    batch_size = len(turn_spans_per_completion)
    if len(turn_rewards_per_completion) != batch_size:
        raise ValueError("turn span and reward row counts must match")
    if num_generations <= 0 or batch_size % num_generations != 0:
        raise ValueError("batch size must be divisible by num_generations")
    if completion_len < 0:
        raise ValueError("completion_len must be non-negative")
    if episode_advantages.dim() != 1 or episode_advantages.numel() != batch_size:
        raise ValueError("episode_advantages must have shape [B]")

    rows: list[TurnCreditRow] = []
    for row_index, (spans, turns) in enumerate(
        zip(turn_spans_per_completion, turn_rewards_per_completion, strict=True)
    ):
        normalized_spans = tuple((int(start), int(end)) for start, end in spans)
        for start, end in normalized_spans:
            if not 0 <= start <= end <= completion_len:
                raise ValueError(
                    f"turn span [{start}, {end}) for row {row_index} exceeds completion width "
                    f"{completion_len}"
                )
        rows.append(
            TurnCreditRow(
                spans=normalized_spans,
                turns=None if turns is None else tuple(turns),
            )
        )

    advantages = episode_advantages.new_zeros((batch_size, completion_len))
    for group_start in range(0, batch_size, num_generations):
        group_end = group_start + num_generations
        group = rows[group_start:group_end]
        if any(row.turns is None for row in group):
            for row_index in range(group_start, group_end):
                completion_end = rows[row_index].spans[-1][1] if rows[row_index].spans else 0
                advantages[row_index, :completion_end] = episode_advantages[row_index]
            continue

        max_turns = max(len(row.turns or ()) for row in group)
        for turn_index in range(max_turns):
            member_indexes = [
                row_index
                for row_index in range(group_start, group_end)
                if turn_index < len(rows[row_index].turns or ())
                and rows[row_index].spans[turn_index][1] > rows[row_index].spans[turn_index][0]
            ]
            if not member_indexes:
                # every member's span for this turn is zero-width (no emitted tokens); an
                # empty turn contributes no advantage and must not skew the group baseline.
                continue
            mean_reward = sum(
                cast(tuple[float, ...], rows[row_index].turns)[turn_index]
                for row_index in member_indexes
            ) / len(member_indexes)
            for row_index in member_indexes:
                row = rows[row_index]
                reward = cast(tuple[float, ...], row.turns)[turn_index]
                start, end = row.spans[turn_index]
                advantages[row_index, start:end] = reward - mean_reward

    if not bool(torch.isfinite(advantages).all()):
        raise ValueError("per-turn advantages must be finite")
    return advantages
