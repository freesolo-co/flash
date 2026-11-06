"""Time-based checkpoint decisions; copied unchanged into isolated trainer children."""

from __future__ import annotations

import os
import time

SPOT_RECOVERY_SECONDS = 600


def enabled() -> bool:
    return os.environ.get("FLASH_SPOT_CHECKPOINT_SECONDS") == str(SPOT_RECOVERY_SECONDS)


class CheckpointClock:
    """Observe completed steps and save before another typical step exceeds the target.

    Save the first completed step after startup/resume to establish a safe baseline.
    Greedily allow another step only when its latest measured duration fits the budget.
    Steps at or above the target save after every completed step.
    Local checkpoint-writing time is excluded from the step-duration samples.
    """

    def __init__(self, *, clock=time.monotonic, target_seconds=SPOT_RECOVERY_SECONDS):
        self.clock = clock
        self.target_seconds = target_seconds
        self.last_step = None
        self.last_end = None
        self.last_save = None

    def due(self, step: int, *, required=(), final_step: int) -> bool:
        now = self.clock()
        if step <= 0 or (self.last_step is not None and step <= self.last_step):
            return False
        step_seconds = (
            max(0.001, (now - self.last_end) / (step - self.last_step))
            if self.last_end is not None
            else 0
        )
        self.last_step, self.last_end = step, now
        if self.last_save is None or step in required or step == final_step:
            return True
        elapsed = now - self.last_save
        return elapsed >= self.target_seconds or elapsed + step_seconds > self.target_seconds

    def saved(self) -> None:
        self.last_save = self.last_end = self.clock()


def should_save(owner, step: int, required, final_step: int, *, distributed=False) -> bool:
    timer = getattr(owner, "_flash_checkpoint_clock", None)
    if timer is None:
        timer = owner._flash_checkpoint_clock = CheckpointClock()
    decision = timer.due(step, required=required, final_step=final_step)
    if distributed:
        # SFT calls this on every FSDP rank. One rank's clock must decide for all
        # ranks or some will skip a collective save while others enter it.
        import torch
        import torch.distributed as dist

        if dist.is_initialized():
            device = torch.cuda.current_device() if dist.get_backend() == "nccl" else "cpu"
            flag = torch.tensor([int(decision)], dtype=torch.int32, device=device)
            dist.broadcast(flag, src=0)
            decision = bool(flag.item())
    return decision


def saved(owner) -> None:
    owner._flash_checkpoint_clock.saved()
