"""GRPO trainer support for group-relative per-turn credit assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from trl import GRPOTrainer


@dataclass(frozen=True)
class TurnCreditRow:
    spans: tuple[tuple[int, int], ...]
    turns: tuple[float, ...] | None


# one line per run, not one per group per step: the condition is a property of the environment, so it
# is either true for most groups or none of them, and a per-group warning would bury the training log.
_WARNED_EPISODE_FALLBACK = False


def _warn_episode_fallback() -> None:
    """Say that the requested per-turn credit is not actually being applied.

    ``credit_assignment = "per_turn"`` is accepted by the spec and echoed back in run status, so a
    user who sets it believes it took effect. When the environment returns no ``per_turn_rewards``
    this silently trains on episode credit instead -- the configured setting is inert, the run looks
    healthy, and nothing distinguishes it from a run that never asked for per-turn credit.
    """
    global _WARNED_EPISODE_FALLBACK
    if _WARNED_EPISODE_FALLBACK:
        return
    _WARNED_EPISODE_FALLBACK = True
    print(
        '[grpo][warn] credit_assignment="per_turn" was requested, but the environment returned no '
        "per_turn_rewards for at least one group; those groups train on episode credit. have "
        "score_episodes() put a per-turn reward list in result.metadata['per_turn_rewards'] to "
        "apply per-turn credit."
    )


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
            _warn_episode_fallback()
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
                cast("tuple[float, ...]", rows[row_index].turns)[turn_index]
                for row_index in member_indexes
            ) / len(member_indexes)
            for row_index in member_indexes:
                row = rows[row_index]
                reward = cast("tuple[float, ...]", row.turns)[turn_index]
                start, end = row.spans[turn_index]
                advantages[row_index, start:end] = reward - mean_reward

    if not bool(torch.isfinite(advantages).all()):
        raise ValueError("per-turn advantages must be finite")
    return advantages


class GRPOPerTurnTrainer(GRPOTrainer):
    """Replace scalar GRPO advantages with aligned per-turn advantages when supplied."""

    def _generate_and_score_completions(self, inputs: list[dict[str, object]]) -> dict[str, object]:
        output = cast("dict[str, object]", super()._generate_and_score_completions(inputs))
        turn_rewards = cast(
            "list[list[float] | None]",
            [item.get("turn_rewards") for item in inputs],
        )
        if not any(rewards is not None for rewards in turn_rewards):
            # the whole-batch miss is the common case: the environment never emits per_turn_rewards
            # at all. it returns before build_per_turn_advantages(), so warn here too or the very
            # configuration this warning exists for is the one that stays silent.
            _warn_episode_fallback()
            return output
        if self.accelerator.num_processes > 1:
            raise NotImplementedError(
                "per-turn GRPO advantages currently support single-process training only; "
                "distributed group centering requires gather-aligned turn metadata"
            )

        turn_spans = cast(
            "list[list[tuple[int, int]] | None]",
            [item.get("turn_spans") for item in inputs],
        )
        if any(spans is None for spans in turn_spans):
            raise ValueError("per-turn rollout rows must all include turn_spans")
        aligned_turn_spans = cast("list[list[tuple[int, int]]]", turn_spans)
        scalar_advantages = cast("torch.Tensor", output["advantages"])
        if scalar_advantages.dim() != 1:
            raise ValueError("expected TRL scalar advantages with shape [B]")
        completion_ids = cast("torch.Tensor", output["completion_ids"])
        batch_size, completion_len = completion_ids.shape
        if len(inputs) != batch_size:
            raise ValueError(
                f"per-turn metadata has {len(inputs)} row(s) for output batch size {batch_size}"
            )

        output["advantages"] = build_per_turn_advantages(
            aligned_turn_spans,
            turn_rewards,
            num_generations=(
                self.num_generations if self.model.training else self.num_generations_eval
            ),
            completion_len=completion_len,
            episode_advantages=scalar_advantages,
        ).to(device=scalar_advantages.device, dtype=scalar_advantages.dtype)
        if not getattr(self, "_per_turn_credit_logged", False):
            print("[rl] multi-turn per-turn group-relative credit is active")
            self._per_turn_credit_logged = True
        return output
