"""Deterministic RNG state shared by training workers."""

from __future__ import annotations

import random


def backend_seed(seed: int) -> int:
    """Map a validated run seed to APIs restricted to unsigned 32-bit values."""
    return int(seed) % (2**32)


def seed_host_rngs(seed: int) -> None:
    """seed the generators that run outside the model: python's and numpy's.

    environment code may consume either generator while building its training rows. neither has
    anything to do with model initialization, so this helper stays independent of torch.
    """
    seed = int(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(backend_seed(seed))
    except ImportError:
        pass


def seed_training_rngs(seed: int) -> None:
    """Seed Python, NumPy, torch CPU, and available CUDA generators."""
    seed_host_rngs(seed)

    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
