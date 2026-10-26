"""Collective scheduling obeys elapsed-time bounds without starving decode."""

from types import SimpleNamespace

import pytest

from flash.serving.deepinfer.runtime.prefill_scheduler import PrefillScheduler
from tests.serving.test_deepinfer_parallel import fake_dist


class Number(float):
    def item(self):
        return float(self)


class Vector(list):
    def __getitem__(self, index):
        return Number(super().__getitem__(index))


@pytest.fixture
def scheduling(monkeypatch):
    clock = [10.0]
    remote = [-1.0, 0.0, 0.0, -1.0]

    def reduce(values, **kwargs):
        for index, value in enumerate(remote):
            values[index] = max(values[index], value)

    fake_dist(monkeypatch, all_reduce=reduce, ReduceOp=SimpleNamespace(MAX="max"))
    monkeypatch.setattr(
        "flash.serving.deepinfer.runtime.prefill_scheduler.time.monotonic", lambda: clock[0]
    )
    policy = PrefillScheduler.__new__(PrefillScheduler)
    policy.engine = SimpleNamespace(
        scheduler=SimpleNamespace(decode_step=0),
        scheduler_config=SimpleNamespace(prefill_any_attempt_interval=8),
    )
    policy.seconds = 0.42
    policy.pending = {}
    policy.decode_remaining = 0
    policy.protected_until = 0.0
    policy.last_prefill = 0.0
    policy.decoding = False
    policy.group = SimpleNamespace(size=lambda: 2)
    policy.age = Vector([0.0] * 4)
    return policy, clock, remote


def test_slow_decode_exhausts_time_budget_before_eight_turns(scheduling):
    policy, clock, _remote = scheduling
    policy.pending = {"prompt": (9.0, 0.42)}
    policy.after_step({"prompt", "decode"})
    clock[0] += 0.079
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 1
    policy.engine.scheduler.decode_step = 2
    policy.after_step({"prompt", "decode"})
    assert policy.decode_remaining == 7
    clock[0] += 0.002
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8


def test_remote_protection_prevents_ranks_from_diverging(scheduling):
    policy, clock, remote = scheduling
    policy.pending = {"prompt": (9.0, 0.42)}
    remote[1:3] = [1.0, 0.04]
    policy.last_prefill = clock[0]
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 1
    remote[2] = 0.0
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8


def test_remote_deadline_forces_attempt_even_with_local_protection(scheduling):
    policy, _clock, remote = scheduling
    policy.after_step({"decode"})
    remote[3] = 0.5
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8


def test_prefill_progress_resets_deadline_for_old_long_prompt(scheduling):
    policy, clock, _remote = scheduling
    policy.pending = {"long": (1.0, 0.42)}
    policy.after_step({"long", "decode"})
    policy.before_step()
    assert policy.age[3] == 0.0
    assert policy.engine.scheduler.decode_step == 1
    policy.engine.scheduler.decode_step = 0
    clock[0] += 0.2
    policy.after_step({"long", "decode"})
    policy.before_step()
    assert policy.age[3] == 0.0
    assert policy.engine.scheduler.decode_step == 1


def test_no_decoders_skips_protection_and_cancellation_removes_deadline(scheduling):
    policy, _clock, _remote = scheduling
    policy.pending = {"prompt": (9.0, 0.42)}
    policy.after_step({"prompt"})
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8
    policy.command({"id": "prompt", "type": "cancel"}, set())
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 0
    assert policy.age[3] == -1.0


def test_single_rank_uses_same_timing_policy_without_collective(scheduling):
    policy, _clock, _remote = scheduling
    policy.group = SimpleNamespace(size=lambda: 1)
    policy.pending = {"prompt": (9.0, 0.42)}
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8
