"""Coordinate large prefills across driver ranks without delaying small suffixes."""

import time

from flash.serving.deepinfer.contract.config import (
    PREFILL_COALESCING_MIN_TOKENS,
    PREFILL_COALESCING_SECONDS,
    PREFILL_DECODE_SECONDS,
    PREFILL_MAX_WAIT_SECONDS,
)


class PrefillScheduler:
    def __init__(self, engine, seconds=PREFILL_COALESCING_SECONDS):
        import torch
        from deepinfer.model_executor.parallel_utils.parallel_state import PG

        self.engine = engine
        self.seconds = seconds
        self.pending = {}
        self.decode_remaining = 0
        self.protected_until = 0.0
        self.last_prefill = 0.0
        self.decoding = False
        self.group = PG.cpu_data_parallel_group()
        self.age = torch.zeros(4, dtype=torch.float64, device="cpu")

    def command(self, command, active, cached_tokens=0):
        request_id = command["id"]
        if command["type"] == "cancel":
            self.pending.pop(request_id, None)
        elif request_id in active:
            remaining = max(0, len(command["tokens"]) - cached_tokens)
            wait = self.seconds if remaining >= PREFILL_COALESCING_MIN_TOKENS else 0.0
            self.pending[request_id] = (time.monotonic(), wait)

    def before_step(self):
        import torch.distributed as dist

        now = time.monotonic()
        self.age[0] = max(
            (now - start - wait for start, wait in self.pending.values()), default=-1.0
        )
        self.age[1] = float(self.decoding)
        self.age[2] = max(0.0, self.protected_until - now) if self.decode_remaining else 0.0
        # Bound deferral of another chunk, not completion of an entire prompt.
        # Reset after real prefill progress so an old long prompt cannot bypass
        # decode protection on every subsequent chunk.
        self.age[3] = max(
            (now - max(start, self.last_prefill) for start, _ in self.pending.values()),
            default=-1.0,
        )
        # Every DP driver performs this collective once per scheduling iteration.
        # Only age durations cross hosts. All drivers then choose the same vote.
        if self.group.size() > 1:
            dist.all_reduce(self.age, op=dist.ReduceOp.MAX, group=self.group)
        # Reduce durations and flags together so no rank chooses a different
        # native collective path. Eight turns is still the iteration ceiling;
        # slow decode iterations exhaust the wall-clock budget earlier.
        overdue = self.age[3].item() >= PREFILL_MAX_WAIT_SECONDS
        if self.age[1].item() and self.age[2].item() > 0 and not overdue:
            self.engine.scheduler.decode_step = 1
        else:
            self.engine.scheduler.decode_step = (
                self.engine.scheduler_config.prefill_any_attempt_interval
                if self.age[0].item() >= 0 or overdue
                else 0
            )

    def after_step(self, active):
        self.pending = {key: value for key, value in self.pending.items() if key in active}
        self.decoding = bool(active - self.pending.keys())
        if self.engine.scheduler.decode_step == 0:
            self.decode_remaining = 8
            self.last_prefill = time.monotonic()
            self.protected_until = self.last_prefill + PREFILL_DECODE_SECONDS
        else:
            self.decode_remaining = max(0, self.decode_remaining - 1)
