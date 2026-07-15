"""Deterministic RNG state shared by training workers."""

from __future__ import annotations

import contextlib
import random


def backend_seed(seed: int) -> int:
    """Map a validated run seed to APIs restricted to unsigned 32-bit values."""
    return int(seed) % (2**32)


def rollout_request_seed(run_seed: int, ordinal: int) -> int:
    """derive a stable nonnegative 63-bit seed for one ordered rollout request."""
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("rollout seed ordinal must be a nonnegative integer")
    mask = (1 << 64) - 1
    value = (int(run_seed) + 0x9E3779B97F4A7C15 * (ordinal + 1)) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & ((1 << 63) - 1)


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
    except MemoryError:
        raise
    except Exception as exc:
        cuda_oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if isinstance(cuda_oom_type, type) and isinstance(exc, cuda_oom_type):
            raise
        return False
    return True
