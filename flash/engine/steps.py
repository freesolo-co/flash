"""Training-step derivation shared by worker and cost-estimate paths."""

from __future__ import annotations

import math


def on_policy_steps(
    *,
    epochs: int,
    prompt_count: int,
    prompts_per_step: int,
) -> int:
    """Resolve GRPO/OPD optimizer steps from full passes over the retained prompt pool."""
    prompt_count = int(prompt_count)
    if prompt_count <= 0:
        raise ValueError("cannot derive epoch-based steps without at least one retained prompt")
    return max(1, math.ceil(prompt_count * int(epochs) / max(1, int(prompts_per_step))))
