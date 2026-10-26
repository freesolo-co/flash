"""Distributed prefill decisions and single-host transport must fail safely."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from flash.serving.deepinfer.native.local_collectives import (
    local_bandwidth,
    patch_bucket,
    validate_local_domains,
)
from flash.serving.deepinfer.runtime.prefill_coalescing import PrefillCoalescing


def fake_dist(monkeypatch, **methods):
    torch = ModuleType("torch")
    dist = ModuleType("torch.distributed")
    vars(dist).update(methods)
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)


@pytest.mark.parametrize(("hosts", "allowed"), [(["one", "one"], True), (["one", "two"], False)])
def test_missing_rdma_requires_all_participants_on_one_host(monkeypatch, hosts, allowed):
    def gather(output, hostname, *, group):
        output[:] = hosts

    fake_dist(monkeypatch, all_gather_object=gather)
    groups = [SimpleNamespace(size=lambda: 2)]
    if allowed:
        assert local_bandwidth(groups, 0) == (1.0, True)
    else:
        with pytest.raises(RuntimeError, match="multi-host"):
            local_bandwidth(groups, 0)
    assert local_bandwidth(groups, 50) == (50, False)


def test_local_fallback_rechecks_physical_topology():
    local = SimpleNamespace(get_physical_domain_size=lambda: (1, 4))
    remote = SimpleNamespace(get_physical_domain_size=lambda: (2, 2))
    validate_local_domains([local], True)
    validate_local_domains([remote], False)
    with pytest.raises(RuntimeError, match="NVLink-only"):
        validate_local_domains([local, remote], True)


def test_unknown_deep_ep_source_is_never_modified(tmp_path):
    path = tmp_path / "bucket.py"
    source = "unreviewed upstream code"
    path.write_text(source)
    with pytest.raises(RuntimeError, match="Unrecognized"):
        patch_bucket(path)
    assert path.read_text() == source


def test_prefill_vote_honors_other_rank_and_skips_wait_for_small_cached_suffix(monkeypatch):
    class Number(float):
        def item(self):
            return float(self)

    class Scalar(list):
        def __getitem__(self, index):
            return Number(super().__getitem__(index))

    remote_age = [-1.0]
    votes = []

    def reduce(age, **kwargs):
        votes.append(age[0])
        age[0] = max(age[0], remote_age[0])

    fake_dist(monkeypatch, all_reduce=reduce, ReduceOp=SimpleNamespace(MAX="max"))
    now = [10.0]
    monkeypatch.setattr(
        "flash.serving.deepinfer.runtime.prefill_coalescing.time.monotonic", lambda: now[0]
    )
    policy = PrefillCoalescing.__new__(PrefillCoalescing)
    policy.engine = SimpleNamespace(
        scheduler=SimpleNamespace(decode_step=0),
        scheduler_config=SimpleNamespace(prefill_any_attempt_interval=8),
    )
    policy.seconds = 0.42
    policy.pending = {}
    policy.decode_remaining = 0
    policy.decoding = False
    policy.group = SimpleNamespace(size=lambda: 2)
    policy.age = Scalar([0, 0])
    policy.command({"id": "large", "type": "generate", "tokens": [1] * 32768}, {"large"})
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 0
    remote_age[0] = 0.1
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8
    remote_age[0] = -1
    policy.command({"id": "large", "type": "cancel"}, set())
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 0
    policy.command({"id": "cached", "type": "generate", "tokens": [1] * 32768}, {"cached"}, 30147)
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8
    policy.after_step(set())
    assert not policy.pending
    assert len(votes) == 4


def test_prefill_chunks_leave_bounded_decode_turns_on_every_rank(monkeypatch):
    class Number(float):
        def item(self):
            return float(self)

    class Vector(list):
        def __getitem__(self, index):
            return Number(super().__getitem__(index))

    remote_decoding = [1]

    def reduce(values, **kwargs):
        values[0] = 1.0
        values[1] = max(values[1], remote_decoding[0])

    fake_dist(monkeypatch, all_reduce=reduce, ReduceOp=SimpleNamespace(MAX="max"))
    policy = PrefillCoalescing.__new__(PrefillCoalescing)
    policy.engine = SimpleNamespace(
        scheduler=SimpleNamespace(decode_step=0),
        scheduler_config=SimpleNamespace(prefill_any_attempt_interval=8),
    )
    policy.pending = {}
    policy.decode_remaining = 0
    policy.decoding = False
    policy.group = SimpleNamespace(size=lambda: 2)
    policy.age = Vector([0, 0])
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8
    # Native scheduling resets this counter to zero after a prefill step.
    policy.engine.scheduler.decode_step = 0
    policy.after_step(set())
    for _ in range(8):
        policy.before_step()
        assert policy.engine.scheduler.decode_step == 1
        policy.engine.scheduler.decode_step = 2
        policy.after_step(set())
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8
    # No decoding anywhere: do not stall a fresh prompt with empty decode turns.
    policy.decode_remaining = 8
    remote_decoding[0] = 0
    policy.before_step()
    assert policy.engine.scheduler.decode_step == 8
    policy.pending = {"prefill": (0, 0)}
    policy.after_step({"prefill", "decode"})
    assert policy.decoding
    policy.after_step({"prefill"})
    assert not policy.decoding
