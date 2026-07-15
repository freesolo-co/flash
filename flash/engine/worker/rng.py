"""Deterministic RNG initialization shared by training workers."""

from __future__ import annotations

import random


def backend_seed(seed: int) -> int:
    """Map a validated run seed to APIs restricted to unsigned 32-bit values."""
    return int(seed) % (2**32)


def seed_training_rngs(seed: int) -> None:
    """Seed Python, NumPy, torch CPU, and available CUDA generators."""
    seed = int(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(backend_seed(seed))
    except ImportError:
        pass

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
