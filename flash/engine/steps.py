"""Training-step derivation shared by worker and cost-estimate paths."""

from __future__ import annotations

import math


def resolve_update_horizon(derived_steps: int, max_steps: int | None) -> int:
    """Return the authoritative optimizer-update horizon for one run."""
    configured = int(max_steps or 0)
    return configured if configured > 0 else int(derived_steps)


def sft_update_steps(
    *,
    epochs: int,
    example_count: int,
    examples_per_update: int,
    packed_block_count: int | None = None,
) -> int:
    """Derive SFT updates from the rows the trainer actually iterates."""
    training_rows = (
        int(packed_block_count) if packed_block_count is not None else int(example_count)
    )
    return max(1, math.ceil(training_rows / max(1, int(examples_per_update))) * int(epochs))


def save_step_due(
    step: int,
    save_at_steps: tuple[int, ...] | list[int],
    save_every: int | None,
) -> bool:
    """Select exact save steps when configured, otherwise preserve periodic saves."""
    step = int(step)
    if step <= 0:
        return False
    required_steps = tuple(int(item) for item in save_at_steps)
    if required_steps:
        return step in required_steps
    interval = int(save_every or 0)
    return interval > 0 and step % interval == 0


def configure_trainer_save_schedule(
    config: dict, save_at_steps: tuple[int, ...] | list[int]
) -> None:
    """Disable periodic trainer saves when exact save steps are configured."""
    if save_at_steps:
        config["save_strategy"] = "no"


def final_save_due(step: int, save_at_steps: tuple[int, ...] | list[int]) -> bool:
    """Preserve the final checkpoint unless exact save steps exclude it."""
    step = int(step)
    if step <= 0:
        return False
    return not tuple(save_at_steps)


def validate_save_steps(save_at_steps: tuple[int, ...] | list[int], horizon: int) -> None:
    """Reject an exact save step that the worker cannot reach."""
    required_steps = tuple(int(item) for item in save_at_steps)
    if required_steps and required_steps[-1] > int(horizon):
        raise ValueError(
            f"save_at_steps entry {required_steps[-1]} exceeds the {int(horizon)}-update horizon"
        )


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
