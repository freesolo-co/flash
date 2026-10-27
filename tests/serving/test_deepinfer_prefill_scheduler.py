"""Queue transitions, rank agreement and decode fairness for adaptive prefill."""

import random
import sys
from types import ModuleType, SimpleNamespace

import pytest

from flash.serving.deepinfer.runtime import prefill_scheduler as scheduling
from flash.serving.deepinfer.runtime.prefill_scheduler import _QueueState as QueueState


class Matrix:
    def __init__(self, shape):
        self.shape = shape
        self.zero_()

    def zero_(self):
        self.values = [[0.0] * self.shape[1] for _ in range(self.shape[0])]

    def __getitem__(self, index):
        return self.values[index]

    def tolist(self):
        return self.values


@pytest.fixture
def scheduler(monkeypatch):
    clock, remote, votes = [100.0], [QueueState()], []
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    torch = ModuleType("torch")
    torch.float64 = "float64"
    torch.zeros = lambda shape, **kw: Matrix(shape)
    dist = ModuleType("torch.distributed")
    dist.ReduceOp = SimpleNamespace(MAX="max")

    def reduce(matrix, **kwargs):
        assert kwargs == {"op": "max", "group": group}
        votes.append(tuple(matrix[0]))
        assert matrix[1] == [0] * 7
        matrix.values[1] = list(remote[0])

    dist.all_reduce = reduce
    torch.distributed = dist
    parallel = ModuleType("deepinfer.model_executor.parallel_utils.parallel_state")
    parallel.PG = SimpleNamespace(cpu_data_parallel_group=lambda: group)
    for name, module in [
        ("torch", torch),
        ("torch.distributed", dist),
        (parallel.__name__, parallel),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(scheduling.time, "monotonic", lambda: clock[0])
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(decode_step=0),
        scheduler_config=SimpleNamespace(prefill_any_attempt_interval=8),
    )
    return scheduling.PrefillScheduler(engine), clock, remote, votes


def admit(policy, request_id, active, tokens=32768, cached=0):
    policy.command({"id": request_id, "type": "generate", "tokens": [1] * tokens}, active, cached)


def test_isolated_request_releases_at_50ms_on_either_driver():
    for age, expected in [(0, False), (0.049999, False), (0.05, True), (0.419, True)]:
        state = QueueState(
            pending_prefill_count=1,
            estimated_prefill_tokens=16384,
            oldest_prefill_age_seconds=age,
            newest_prefill_age_seconds=age,
            active_request_count=1,
        )
        for states in [[state, QueueState()], [QueueState(), state]]:
            assert scheduling._prefill_ready(states, 0.42) is expected


def test_remote_active_decode_prevents_isolated_shortcut():
    state = QueueState(
        pending_prefill_count=1,
        estimated_prefill_tokens=16384,
        oldest_prefill_age_seconds=0.1,
        newest_prefill_age_seconds=0.1,
        active_request_count=1,
    )
    assert not scheduling._prefill_ready(
        [state, QueueState(active_request_count=1, has_decoding_requests=1)], 0.42
    )
    assert scheduling._prefill_ready(
        [state._replace(oldest_prefill_age_seconds=0.42), QueueState(active_request_count=1)], 0.42
    )


def test_balanced_work_requires_quiet_window_and_full_chunk_on_every_driver():
    left = QueueState(
        pending_prefill_count=1,
        estimated_prefill_tokens=16384,
        oldest_prefill_age_seconds=0.02,
        newest_prefill_age_seconds=0.02,
        active_request_count=1,
    )
    right = left._replace(newest_prefill_age_seconds=0.009)
    assert not scheduling._prefill_ready([left, right], 0.42)
    assert scheduling._prefill_ready([left, right._replace(newest_prefill_age_seconds=0.01)], 0.42)
    assert not scheduling._prefill_ready(
        [left, right._replace(estimated_prefill_tokens=8192, newest_prefill_age_seconds=0.02)], 0.42
    )
    assert not scheduling._prefill_ready(
        [left._replace(pending_prefill_count=2, active_request_count=2), QueueState()], 0.42
    )


def test_small_suffix_bypasses_wait_but_empty_queue_does_not():
    assert scheduling._prefill_ready(
        [
            QueueState(
                pending_prefill_count=1, has_small_uncached_suffix=1, active_request_count=1
            ),
            QueueState(),
        ],
        0.42,
    )
    assert not scheduling._prefill_ready(
        [QueueState(active_request_count=2, has_decoding_requests=1), QueueState()], 0.42
    )


def test_bounded_wait_and_rank_order_for_random_queue_geometries():
    rng = random.Random(812)
    for _ in range(5000):
        states = [
            QueueState(
                pending_prefill_count=rng.randrange(4),
                estimated_prefill_tokens=rng.randrange(65536),
                oldest_prefill_age_seconds=rng.random(),
                newest_prefill_age_seconds=rng.random() / 100,
                active_request_count=rng.randrange(1, 6),
            )
            for _ in range(2)
        ]
        ready = scheduling._prefill_ready(states, 0.42)
        assert ready == scheduling._prefill_ready(states[::-1], 0.42)
        if any(
            state.pending_prefill_count and state.oldest_prefill_age_seconds >= 0.42
            for state in states
        ):
            assert ready


def test_admission_cancellation_and_cached_suffix(scheduler):
    policy, _, remote, votes = scheduler
    admit(policy, "large", {"large"})
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 0
    remote[0] = QueueState(
        pending_prefill_count=1,
        estimated_prefill_tokens=16384,
        oldest_prefill_age_seconds=0.43,
        newest_prefill_age_seconds=0.43,
        active_request_count=1,
    )
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8
    remote[0] = QueueState()
    policy.command({"id": "large", "type": "cancel"}, set())
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 0
    admit(policy, "cached", {"cached"}, cached=30147)
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8
    assert votes[-1][3] == 1
    policy.after_step(set())
    assert not policy.pending
    assert policy.active_count == 0
    assert len(votes) == 4


def test_new_admission_resets_quiet_window_and_work_is_capped(scheduler):
    policy, clock, remote, _ = scheduler
    admit(policy, "one", {"one"}, tokens=131072)
    remote[0] = QueueState(
        pending_prefill_count=1,
        estimated_prefill_tokens=16384,
        oldest_prefill_age_seconds=0.02,
        newest_prefill_age_seconds=0.02,
        active_request_count=1,
    )
    clock[0] += 0.02
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8
    admit(policy, "two", {"one", "two"})
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 0
    assert policy.state[0][1] == 32768
    clock[0] += 0.011
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8
    policy.command({"id": "two", "type": "cancel"}, {"one"})
    policy.before_step()
    assert policy.state[0][1] == 16384


def test_native_prefill_reset_preserves_eight_decode_turns_for_remote_work(scheduler):
    policy, _, remote, _ = scheduler
    remote[0] = QueueState(
        pending_prefill_count=1,
        estimated_prefill_tokens=16384,
        oldest_prefill_age_seconds=0.5,
        newest_prefill_age_seconds=0.5,
        active_request_count=2,
        has_decoding_requests=1,
    )
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8
    policy.engine.scheduler.decode_step = 0
    policy.after_step(set())
    for _ in range(8):
        policy.before_step()
        assert policy.engine.scheduler.decode_step == 1
        policy.engine.scheduler.decode_step = 2
        policy.after_step(set())
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8
    remote[0] = remote[0]._replace(has_decoding_requests=0)
    policy.decode_remaining = 8
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8


def test_first_output_keeps_admission_age_and_tracks_decode_activity(scheduler):
    policy, clock, _, _ = scheduler
    admit(policy, "prefill", {"prefill"})
    start = policy.pending["prefill"][0]
    clock[0] += 0.1
    policy.after_step({"prefill", "decode"})
    assert policy.decoding
    assert policy.active_count == 2
    assert policy.pending["prefill"][0] == start
    policy.after_step({"prefill"})
    assert not policy.decoding
    assert policy.active_count == 1
    policy.after_step(set())
    assert not policy.pending
