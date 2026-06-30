"""CUDA-OOM escalation coverage (CPU-only; no GPU/network)."""

from __future__ import annotations

import types

import pytest


def test_is_cuda_oom_is_structured(monkeypatch):
    # torch-free (the offline CI image has no torch): is_cuda_oom imports torch internally under a
    # try/except, so the counter + MemoryError paths classify without it.
    from flash.engine.worker.perf import lifecycle as lc

    # NO string matching: a RuntimeError that SAYS "out of memory" is not an OOM without a real signal
    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 0)
    assert lc.is_cuda_oom(RuntimeError("Triton Error [CUDA]: out of memory")) is False
    assert lc.is_cuda_oom(ValueError("bad config")) is False
    # torch's allocator counter advancing is the structured "error code"
    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 2)
    assert lc.is_cuda_oom(RuntimeError("whatever")) is True
    # a host-RAM OOM is never a GPU OOM (a bigger card can't fix it), even with the counter pinned >0
    assert lc.is_cuda_oom(MemoryError("host ram: out of memory")) is False


def test_is_cuda_oom_typed_torch_error():
    torch = pytest.importorskip("torch")  # skipped on the torch-less offline CI image
    from flash.engine.worker.perf.lifecycle import is_cuda_oom

    assert is_cuda_oom(torch.cuda.OutOfMemoryError("x")) is True  # typed allocator signal


def test_is_cuda_oom_none_is_never_oom(monkeypatch):
    from flash.engine.worker.perf import lifecycle as lc

    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 5)
    assert lc.is_cuda_oom(None) is False


def _card(gpu, vram):
    return types.SimpleNamespace(gpu=gpu, vram_gb=vram, provider="runpod", hourly_usd=1.0)


def test_oom_escalated_keeps_only_strictly_larger_cards():
    from flash.runner.lifecycle import _oom_escalated

    cands = [_card("A100", 80), _card("A100b", 80), _card("Pro6000", 96), _card("B200", 180)]
    assert [c.gpu for c in _oom_escalated(cands, 0)] == [c.gpu for c in cands]  # no OOM -> unchanged
    assert {c.gpu for c in _oom_escalated(cands, 80)} == {"Pro6000", "B200"}  # >80 only
    assert _oom_escalated(cands, 180) == []  # OOM'd the biggest -> nowhere larger


def test_surfaced_worker_flags_reads_both_flags_in_one_pass():
    from flash.providers.runpod.jobs import surfaced_worker_flags

    say = lambda _m: None  # noqa: E731
    reads = {"n": 0}

    def reader(force=False):
        reads["n"] += 1
        return {"oom": True, "attempt": "0", "retriable": False, "stage": "rl_train"}

    _key, retriable, oom = surfaced_worker_flags(reader, None, say, 0)
    assert (retriable, oom) == (False, True)
    assert reads["n"] == 1  # surfacing + both flags share ONE forced read
    assert surfaced_worker_flags(lambda force=False: {"retriable": True}, None, say)[1:] == (True, False)
    assert surfaced_worker_flags(lambda force=False: {"retriable": True}, None, say, 0)[1:] == (True, False)
    assert surfaced_worker_flags(
        lambda force=False: {"retriable": True, "ts": 9_000.0}, None, say, 0, launch_ts=10_000.0
    )[1:] == (False, False)
    assert surfaced_worker_flags(
        lambda force=False: {"retriable": True, "ts": 10_500.0}, None, say, 0, launch_ts=10_000.0
    )[1:] == (True, False)
    assert surfaced_worker_flags(None, None, say)[1:] == (False, False)
    stale = lambda force=False: {"oom": True, "attempt": "0", "retriable": False}  # noqa: E731
    assert surfaced_worker_flags(stale, None, say, 0)[1:] == (False, True)
    assert surfaced_worker_flags(stale, None, say, 1)[1:] == (False, False)


def test_heartbeat_oom_for_attempt_gates_stale_flag():
    from flash.providers._poll import heartbeat_oom_for_attempt

    assert heartbeat_oom_for_attempt({"oom": True, "attempt": "0"}, 0) is True
    assert heartbeat_oom_for_attempt({"oom": True, "attempt": "0"}, 1) is False
    assert heartbeat_oom_for_attempt({"oom": True, "attempt": "1"}, 1) is True
    assert heartbeat_oom_for_attempt({"oom": True}, 1) is False
    assert heartbeat_oom_for_attempt({"oom": True, "attempt": "0"}, None) is False
    assert heartbeat_oom_for_attempt(None, 0) is False
    assert heartbeat_oom_for_attempt({"retriable": True}, 0) is False


def test_poll_job_maps_only_matching_oom_attempt(monkeypatch):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(jobs.time, "sleep", lambda _s: None)
    monkeypatch.setattr(runpod_api, "job_status", lambda _eid, _jid: {"status": "FAILED", "error": "x"})
    handle = jobs.JobHandle("ep", "name", "job")

    res = jobs.poll_job(
        handle,
        interval_s=0,
        heartbeat_reader=lambda force=False: {"oom": True, "attempt": "2"},
        current_attempt=2,
    )
    assert res.failure == "oom"

    res = jobs.poll_job(
        handle,
        interval_s=0,
        heartbeat_reader=lambda force=False: {"oom": True, "attempt": "1"},
        current_attempt=2,
    )
    assert res.failure == "job_failed"


def test_worker_failure_flags_prioritize_retriable_over_oom(monkeypatch):
    import flash.engine.worker as worker

    monkeypatch.setattr(worker, "is_cuda_oom", lambda _exc: True)
    assert worker._worker_failure_flags(RuntimeError("cuda oom")) == {
        "retriable": False,
        "oom": True,
    }
    assert worker._worker_failure_flags(worker.RetriableInfraError("bad host")) == {
        "retriable": True,
        "oom": False,
    }
