"""Generic, task-agnostic data helpers.

Task-specific data loading/prompting (e.g. GSM8K) lives with its example in
``examples/<task>/data.py``; only reusable, environment-independent helpers
belong here.
"""

from __future__ import annotations

import random


def eval_subset_indices(num_examples: int, subset_seed: int, total: int) -> list[int]:
    """Deterministic eval-subset indices for a given seed (no data loaded).

    Reusable across environments: pick ``num_examples`` indices out of ``total``
    so every training arm evaluates on the same rows.
    """
    rng = random.Random(subset_seed)
    n = min(num_examples, total)
    return sorted(rng.sample(range(total), n))
