"""Coordinate large prefills across driver ranks without delaying small suffixes."""

import time

from flash.serving.deepinfer.contract.config import (
    PREFILL_COALESCING_MIN_TOKENS,
    PREFILL_COALESCING_SECONDS,
)


class PrefillScheduler:
    def __init__(self, engine, seconds=PREFILL_COALESCING_SECONDS):
        import torch
        from deepinfer.model_executor.parallel_utils.parallel_state import PG

        self.engine = engine
        self.seconds = seconds
        self.pending = {}
        self.decode_remaining = 0
        self.decoding = False
        self.group = PG.cpu_data_parallel_group()
        self.age = torch.zeros(2, dtype=torch.float64, device="cpu")

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
        # Every DP driver performs this collective once per scheduling iteration.
        # Only age durations cross hosts. All drivers then choose the same vote.
        if self.group.size() > 1:
            dist.all_reduce(self.age, op=dist.ReduceOp.MAX, group=self.group)
        # Keep a bounded decode opportunity between long prefill chunks. Every
        # driver sees the same global decode flag and advances the same step budget.
        if self.age[1].item() and self.decode_remaining:
            self.engine.scheduler.decode_step = 1
        else:
            self.engine.scheduler.decode_step = (
                self.engine.scheduler_config.prefill_any_attempt_interval
                if self.age[0].item() >= 0
                else 0
            )

    def after_step(self, active):
        self.pending = {key: value for key, value in self.pending.items() if key in active}
        self.decoding = bool(active - self.pending.keys())
        if self.engine.scheduler.decode_step == 0:
            self.decode_remaining = 8
        else:
            self.decode_remaining = max(0, self.decode_remaining - 1)
