"""Distributed prefill decisions and single-host transport must fail safely."""

import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from flash.serving.deepinfer.native import transport
from flash.serving.deepinfer.native.transport import patch_bucket
from flash.serving.deepinfer.runtime.prefill_scheduler import PrefillScheduler


def fake_dist(monkeypatch, **methods):
    torch = ModuleType("torch")
    dist = ModuleType("torch.distributed")
    vars(dist).update(methods)
    torch.distributed = dist
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)


@pytest.mark.parametrize(
    ("nvl_gbs", "rdma_gbs", "expected"),
    [(900, 0, (900, 1.0)), (900, 50, (900, 50)), (0, 0, None), (-1, 50, None), (900, -1, None)],
)
def test_single_host_patch_preserves_bandwidth_and_requires_nvlink(
    tmp_path, monkeypatch, nvl_gbs, rdma_gbs, expected
):
    source = (
        "class Bucket:\n"
        "    def __init__(self, nvl_gbs, rdma_gbs):\n"
        "        assert nvl_gbs > 0 and rdma_gbs > 0, 'Failed to detect the NVLink or RDMA bandwidth'\n"
        "        self.bandwidth = (nvl_gbs, rdma_gbs)\n"
    )
    monkeypatch.setattr(transport, "BUCKET_SHA256", hashlib.sha256(source.encode()).hexdigest())
    path = tmp_path / "bucket.py"
    path.write_text(source)
    patch_bucket(path)
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    if expected is None:
        with pytest.raises(AssertionError, match="communication bandwidth"):
            namespace["Bucket"](nvl_gbs, rdma_gbs)
    else:
        assert namespace["Bucket"](nvl_gbs, rdma_gbs).bandwidth == expected


@pytest.mark.parametrize("matches", [0, 2])
def test_ambiguous_bandwidth_patch_does_not_modify_source(tmp_path, monkeypatch, matches):
    source = (
        "        assert nvl_gbs > 0 and rdma_gbs > 0, 'Failed to detect the NVLink or RDMA bandwidth'\n"
    ) * matches
    monkeypatch.setattr(transport, "BUCKET_SHA256", hashlib.sha256(source.encode()).hexdigest())
    path = tmp_path / "bucket.py"
    path.write_text(source)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        patch_bucket(path)
    assert path.read_text() == source


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
        "flash.serving.deepinfer.runtime.prefill_scheduler.time.monotonic", lambda: now[0]
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
    policy.age = Scalar([0, 0, 0, 0])
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

    monkeypatch.setattr(
        "flash.serving.deepinfer.runtime.prefill_scheduler.time.monotonic", lambda: 10.0
    )
    remote_decoding = [1]

    def reduce(values, **kwargs):
        values[0] = 1.0
        values[1] = max(values[1], remote_decoding[0])

    fake_dist(monkeypatch, all_reduce=reduce, ReduceOp=SimpleNamespace(MAX="max"))
    policy = PrefillScheduler.__new__(PrefillScheduler)
    policy.engine = SimpleNamespace(
        scheduler=SimpleNamespace(decode_step=0),
        scheduler_config=SimpleNamespace(prefill_any_attempt_interval=8),
    )
    policy.pending = {}
    policy.decode_remaining = 0
    policy.protected_until = 0.0
    policy.last_prefill = 0.0
    policy.decoding = False
    policy.group = SimpleNamespace(size=lambda: 2)
    policy.age = Vector([0, 0, 0, 0])
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
