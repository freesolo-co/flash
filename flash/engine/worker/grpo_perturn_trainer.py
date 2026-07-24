"""TRL GRPO trainer that applies backend-neutral group-relative per-turn credit.

The per-turn advantage math lives in :mod:`flash.engine.worker.perturn_credit` so this TRL trainer
and the OpenRLHF multi-turn path share one implementation instead of two copies that can drift; this
module only adapts that math into TRL's ``_generate_and_score_completions`` output.
"""

from __future__ import annotations

from typing import cast

import torch
from trl import GRPOTrainer

from flash.engine.worker.perturn_credit import TurnCreditRow, build_per_turn_advantages

__all__ = ["GRPOPerTurnTrainer", "TurnCreditRow", "build_per_turn_advantages"]


class GRPOPerTurnTrainer(GRPOTrainer):
    """Replace scalar GRPO advantages with aligned per-turn advantages when supplied."""

    def _generate_and_score_completions(self, inputs: list[dict[str, object]]) -> dict[str, object]:
        output = cast(dict[str, object], super()._generate_and_score_completions(inputs))
        turn_rewards = cast(
            list[list[float] | None],
            [item.get("turn_rewards") for item in inputs],
        )
        if not any(rewards is not None for rewards in turn_rewards):
            return output
        if self.accelerator.num_processes > 1:
            raise NotImplementedError(
                "per-turn GRPO advantages currently support single-process training only; "
                "distributed group centering requires gather-aligned turn metadata"
            )

        turn_spans = cast(
            list[list[tuple[int, int]] | None],
            [item.get("turn_spans") for item in inputs],
        )
        if any(spans is None for spans in turn_spans):
            raise ValueError("per-turn rollout rows must all include turn_spans")
        aligned_turn_spans = cast(list[list[tuple[int, int]]], turn_spans)
        scalar_advantages = cast(torch.Tensor, output["advantages"])
        if scalar_advantages.dim() != 1:
            raise ValueError("expected TRL scalar advantages with shape [B]")
        completion_ids = cast(torch.Tensor, output["completion_ids"])
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
