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


def sft_data_parallel_cards(gpu_count: int, train_batch_size: int, row_count: int = 0) -> int:
    """Cards SFT can actually train on, given verl splits the batch AND the dataset across them.

    SFT runs data-parallel (see ``sft_train_runner._prepare_sft_child``), so verl derives
    ``train_batch_size_per_dp = train_batch_size // dp_size`` and hands that straight to a
    DataLoader. A count ABOVE the batch floors the per-rank batch to 0, and
    ``DataLoader(batch_size=0)`` raises ``ValueError`` -- measured, not inferred.

    The width must divide ``row_count`` too, and that one silently corrupts the run rather than
    raising: verl builds ``DistributedSampler(..., drop_last=True)`` (``sft_trainer.py:237``) and
    Flash's exact-dataloader shim overrides ``drop_last`` on the LOADER only, so a remainder is
    dropped from every epoch -- MEASURED at 11 rows, 2 ranks trains 10 and 4 ranks trains 8, while
    the frozen quote still bills all 11. ``drop_last=False`` is not the fix: it pads by DUPLICATING
    rows, trading silent row loss for silent row repetition.

    So take the largest count <= the allocated cards that divides both, keeping the realized global
    batch exactly ``train_batch_size`` and every profiled row trained exactly once. That is what
    makes the card count a pure throughput choice rather than a hyperparameter change. Returns 1 for
    an unpacked run, which is correct: one example cannot be split.

    ``row_count`` defaults to 0, meaning "unknown, do not constrain" -- the cost path quotes before
    the dataset is materialized. Quote-side only; the worker always passes the real count.

    Lives here rather than beside its caller because the cost path must quote the width that will
    execute, and ``sft_train_runner`` is not importable from it (it cycles through ``sft_train``).
    """
    cards = max(1, int(gpu_count))
    batch = max(1, int(train_batch_size))
    return widest_usable_dp_width(range(min(cards, batch), 0, -1), batch, row_count)


def widest_usable_dp_width(candidates, train_batch_size: int, row_count: int) -> int:
    """First candidate that divides the batch and the rows, or 1 when none does.

    ``candidates`` must be ordered widest-first; the caller owns which shapes are eligible.
    ``sft_data_parallel_cards`` searches every count up to the allocation because it answers "what
    will verl actually run", while the worker's idle-card warning searches only the power-of-two
    shapes providers rent because it answers "what should you allocate instead". Same predicate,
    different candidate sets -- the divisibility rule itself lives here once.
    """
    batch = max(1, int(train_batch_size))
    rows = max(0, int(row_count))
    for count in candidates:
        if batch % count == 0 and (rows == 0 or rows % count == 0):
            return int(count)
    return 1


def rl_data_parallel_cards(gpu_count: int, sequences_per_step: int) -> int:
    """Cards grpo/opd can actually train on, because verl splits the batch evenly across dp ranks.

    With ulysses pinned off (see ``verl.parallelism.ULYSSES_SEQUENCE_PARALLEL_SIZE``) every rank is a
    DATA-parallel rank, so verl's dp width is the card count rather than 1. Two places then require
    the sequence count to divide that width exactly, and both RAISE rather than degrade:
    ``DataProto.chunk`` asserts ``len(self) % chunks == 0`` on every dp dispatch (``protocol.py``),
    and ``_balance_batch`` partitions with ``equal_size=True``, which asserts the same
    (``seqlen_balancing.py``). Neither is reachable at width 1, which is why sequence parallelism hid
    this: ``n % 1 == 0`` always holds.

    ``sequences_per_step`` is what verl holds after ``batch.repeat(rollout.n)`` -- prompts times
    group size -- not the prompt count. verl pads to a divisor only for VALIDATION generation, and
    its ``VERL_AUTO_PADDING`` path is off by default and unset here; enabling it would pad by
    DUPLICATING rows, which changes the gradient, so narrowing the width is the correct trade (the
    same one ``sft_data_parallel_cards`` makes for its own loader).

    This costs no capacity at real knobs: batch times group is 16/32/64/128 in practice, and every
    one of those divides 1/2/4/8. It bites only on tiny batches, which is exactly where verl would
    otherwise abort at step 0 on a box already paid for.

    Only POWERS OF TWO are candidates, which is where this parts company with
    ``sft_data_parallel_cards``: the same count becomes vLLM's ``tensor_model_parallel_size`` for the
    rollout engine (sft has no rollout engine), and vLLM requires the attention heads to divide it.
    Searching every count would return 3 for 6 sequences on 4 cards -- correct for the dp chunk, and
    a head-divisibility failure at engine init on five of the six catalog rows. Trading verl's abort
    for vLLM's is not a fix.
    """
    # local import: `providers.base` reaches back into `engine.plan` (recipe, vram), so binding it at
    # module level here would close that cycle.
    from flash.providers.core.base import rentable_gpu_counts

    cards = max(1, int(gpu_count))
    sequences = max(1, int(sequences_per_step))
    widths = [n for n in rentable_gpu_counts(cards) if n <= sequences]
    return widest_usable_dp_width(widths, sequences, 0)


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
