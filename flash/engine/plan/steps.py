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


def sft_data_parallel_cards(gpu_count: int, train_batch_size: int) -> int:
    """Cards SFT can actually train on, given verl splits the batch across them.

    SFT runs data-parallel (see ``sft_train_runner._prepare_sft_child``), so verl derives
    ``train_batch_size_per_dp = train_batch_size // dp_size`` and hands that straight to a
    DataLoader. A count ABOVE the batch floors the per-rank batch to 0, and
    ``DataLoader(batch_size=0)`` raises ``ValueError`` -- measured, not inferred. A count that does
    not DIVIDE the batch leaves a remainder that cannot be dealt to every rank equally.

    So take the largest divisor of the batch that is <= the allocated cards. That keeps the
    realized global batch exactly ``train_batch_size`` on every width this returns, which is what
    makes the card count a pure throughput choice rather than a hyperparameter change.

    Returns 1 for an unpacked run (``examples_per_update`` is 1 there), which is correct: one
    example cannot be split, so extra cards would have nothing to hold.

    Lives here rather than beside its caller because the cost path must quote the width that will
    execute, and ``sft_train_runner`` is not importable from it (it cycles through ``sft_train``).
    """
    cards = max(1, int(gpu_count))
    batch = max(1, int(train_batch_size))
    for count in range(min(cards, batch), 0, -1):
        if batch % count == 0:
            return count
    return 1


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
