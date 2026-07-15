"""Deterministic RNG state shared by training workers."""

from __future__ import annotations

import contextlib
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


def capture_training_rng_state(torch) -> dict:
    """Capture Python, NumPy, torch CPU, and available CUDA generator state."""
    state = {
        "python": random.getstate(),
        "numpy": None,
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    with contextlib.suppress(ImportError):
        import numpy as np

        state["numpy"] = np.random.get_state()
    return state


def restore_training_rng_state(torch, state: object) -> bool:
    """Restore a captured training RNG state, returning false on any incompatibility."""
    if not isinstance(state, dict):
        return False
    if set(state) != {"python", "numpy", "torch", "cuda"}:
        return False
    try:
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            if not torch.cuda.is_available():
                return False
            torch.cuda.set_rng_state_all(state["cuda"])
        random.setstate(state["python"])
        if state["numpy"] is not None:
            import numpy as np

            np.random.set_state(state["numpy"])
    except Exception:
        return False
    return True
