"""Checkpoint phase timings usable in both the trainer and its parent process."""

import time
from contextlib import contextmanager


@contextmanager
def checkpoint_phase(step: int, phase: str):
    started = time.monotonic()
    try:
        yield
    finally:
        print(f"[ckpt] step={step} {phase}_s={time.monotonic() - started:.3f}", flush=True)
