"""Deterministic RNG state shared by training workers."""

from __future__ import annotations

import random


def backend_seed(seed: int) -> int:
    """Map a validated run seed to APIs restricted to unsigned 32-bit values."""
    return int(seed) % (2**32)


def seed_host_rngs(seed: int) -> None:
    """Seed the generators that run outside the model: Python's and NumPy's.

    Split out for the workload-profile run, which must reach the same rows the training run will and
    must not import torch. Environment code is user code and may consume either of these while
    building its dataset, so a profile that skipped them could select a different sample than
    training and fail the parity check. Neither generator has anything to do with model init, which
    is the part a profile deliberately never reaches.
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
