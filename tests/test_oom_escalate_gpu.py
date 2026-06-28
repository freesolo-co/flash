"""CUDA-OOM -> retry on a strictly LARGER GPU (CPU-only; no GPU/network).

A training CUDA OOM means the card was too small. The worker classifies it STRUCTURALLY (torch's
typed OutOfMemoryError + its num_ooms allocator counter — never the message text) and stamps an
``oom`` heartbeat flag; the poller maps it to ``failure="oom"``; the runner retries on a card with
more VRAM (bounded by the available tiers). These cover the pieces + the wiring.
"""

from __future__ import annotations

import ast
import pathlib
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
        return {"oom": True, "retriable": False, "stage": "rl_train"}

    _key, retriable, oom = surfaced_worker_flags(reader, None, say)
    assert (retriable, oom) == (False, True)
    assert reads["n"] == 1  # surfacing + both flags share ONE forced read
    assert surfaced_worker_flags(lambda force=False: {"retriable": True}, None, say)[1:] == (True, False)
    assert surfaced_worker_flags(None, None, say)[1:] == (False, False)


def _read(mod):
    return pathlib.Path(__import__(mod, fromlist=["x"]).__file__).read_text()


def test_worker_stamps_oom_flag():
    src = _read("flash.engine.worker")
    assert "oom = not retriable and is_cuda_oom(e)" in src  # short-circuits on retriable infra
    assert '{"retriable": retriable, "oom": oom}' in src


def test_poller_maps_oom_flag_to_oom_failure():
    src = _read("flash.providers.runpod.jobs")
    assert src.count('"oom" if oom else') == 2  # oom wins in both worker-fail paths
    assert "def surfaced_worker_flags" in src  # single-read helper feeds both paths


def test_runner_escalates_on_oom():
    src = _read("flash.runner.lifecycle")
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_submit_seed_supervised"
    )
    body = ast.get_source_segment(src, fn)
    assert 'oom_shaped = res.failure == "oom"' in body
    assert "retry_shaped = infra_shaped or oom_shaped" in body
    assert "if not retry_shaped:" in body  # oom retries, not fail-fast
    assert "oom_vram_floor = max(oom_vram_floor, int(chosen.vram_gb))" in body
    assert "_oom_escalated(alloc.candidates, oom_vram_floor)" in body
    assert "if not oom_shaped:" in body  # oom grows the card, doesn't escape the provider
    # OOM escalation is cost -> bounded by the user's RAW max_retries, not the floored infra budget.
    assert "retry_budget = max_retries if oom_shaped else infra_budget" in body
