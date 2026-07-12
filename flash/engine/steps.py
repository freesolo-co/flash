"""Training-step derivation shared by worker and cost-estimate paths."""

from __future__ import annotations

import math


def on_policy_steps(
    *,
    epochs: int,
    prompt_count: int,
    prompts_per_step: int,
    max_steps: int | None = None,
) -> int:
    """Resolve on-policy optimizer steps from retained-prompt passes and an optional ceiling."""
    prompt_count = int(prompt_count)
    if prompt_count <= 0:
        raise ValueError("cannot derive epoch-based steps without at least one retained prompt")
    derived = max(1, math.ceil(prompt_count * int(epochs) / max(1, int(prompts_per_step))))
    ceiling = int(max_steps or 0)
    if ceiling < 0:
        raise ValueError("max_steps must be non-negative")
    return min(derived, ceiling) if ceiling > 0 else derived
