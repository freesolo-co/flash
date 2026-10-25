"""Coordinate large prefills across driver ranks without delaying small suffixes."""

import time

from flash.serving.deepinfer.contract.config import (
    PREFILL_COALESCING_MIN_TOKENS,
    PREFILL_COALESCING_SECONDS,
)


class PrefillCoalescing:
    def __init__(self, engine, seconds=PREFILL_COALESCING_SECONDS):
        import torch
        from deepinfer.model_executor.parallel_utils.parallel_state import PG

        self.engine = engine
        self.seconds = seconds
        self.pending = {}
        self.group = PG.cpu_data_parallel_group()
        self.age = torch.zeros(1, dtype=torch.float64, device="cpu")

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
        # Every DP driver performs this collective once per scheduling iteration.
        # Only age durations cross hosts. All drivers then choose the same vote.
        if self.group.size() > 1:
            dist.all_reduce(self.age, op=dist.ReduceOp.MAX, group=self.group)
        self.engine.scheduler.decode_step = (
            self.engine.scheduler_config.prefill_any_attempt_interval if self.age.item() >= 0 else 0
        )

    def after_step(self, active):
        self.pending = {key: value for key, value in self.pending.items() if key in active}
