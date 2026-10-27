"""Coordinate adaptive prefill readiness and bounded decode turns across drivers."""

import time
from typing import NamedTuple

from flash.serving.deepinfer.contract.config import (
    PREFILL_COALESCING_MIN_TOKENS,
    PREFILL_COALESCING_SECONDS,
    PREFILL_ISOLATED_SECONDS,
    PREFILL_QUIET_SECONDS,
    PREFILL_TOKENS,
)


class _RankState(NamedTuple):
    pending: float = 0
    work: float = 0
    oldest: float = 0
    small_suffix: float = 0
    quiet: float = 0
    active: float = 0
    decoding: float = 0


def _prefill_ready(states, seconds):
    pending = [state for state in states if state.pending]
    if not pending:
        return False
    if any(state.small_suffix for state in pending):
        return True
    oldest = max(state.oldest for state in pending)
    if oldest >= seconds:
        return True
    if len(pending) == len(states) and all(state.work >= PREFILL_TOKENS for state in states):
        return min(state.quiet for state in states) >= PREFILL_QUIET_SECONDS
    return sum(state.active for state in states) <= 1 and oldest >= PREFILL_ISOLATED_SECONDS


class PrefillScheduler:
    def __init__(self, engine, seconds=PREFILL_COALESCING_SECONDS):
        import torch
        from deepinfer.model_executor.parallel_utils.parallel_state import PG

        self.engine = engine
        self.seconds = seconds
        self.pending = {}
        self.decode_remaining = 0
        self.decoding = False
        self.active_count = 0
        self.group = PG.cpu_data_parallel_group()
        self.rank = self.group.rank()
        self.state = torch.zeros(
            (self.group.size(), len(_RankState._fields)), dtype=torch.float64, device="cpu"
        )

    def command(self, command, active, cached_tokens=0):
        request_id = command["id"]
        if command["type"] == "cancel":
            self.pending.pop(request_id, None)
        elif request_id in active:
            remaining = max(0, len(command["tokens"]) - cached_tokens)
            self.pending[request_id] = (time.monotonic(), remaining)
        self.active_count = len(active)

    def before_step(self):
        import torch.distributed as dist

        now = time.monotonic()
        pending = list(self.pending.values())
        local = _RankState(
            pending=len(pending),
            # Admission-time estimate, capped at one chunk per pending request.
            # Pending entries persist until first output; this is not exact remaining work.
            work=sum(min(tokens, PREFILL_TOKENS) for _, tokens in pending),
            oldest=max((now - start for start, _ in pending), default=0.0),
            small_suffix=any(tokens < PREFILL_COALESCING_MIN_TOKENS for _, tokens in pending),
            quiet=min((now - start for start, _ in pending), default=0.0),
            active=self.active_count,
            decoding=self.decoding,
        )
        self.state.zero_()
        row = self.state[self.rank]
        for index, value in enumerate(local):
            row[index] = value
        # Each driver owns one row. One collective publishes the same queue view
        # to every driver; only durations, never host timestamps, cross ranks.
        if self.group.size() > 1:
            dist.all_reduce(self.state, op=dist.ReduceOp.MAX, group=self.group)
        states = [_RankState(*row) for row in self.state.tolist()]
        if any(state.decoding for state in states) and self.decode_remaining:
            self.engine.scheduler.decode_step = 1
        else:
            self.engine.scheduler.decode_step = (
                self.engine.scheduler_config.prefill_any_attempt_interval
                if _prefill_ready(states, self.seconds)
                else 0
            )

    def after_step(self, active):
        self.pending = {key: value for key, value in self.pending.items() if key in active}
        self.active_count = len(active)
        self.decoding = bool(active - self.pending.keys())
        if self.engine.scheduler.decode_step == 0:
            self.decode_remaining = 8
        else:
            self.decode_remaining = max(0, self.decode_remaining - 1)
