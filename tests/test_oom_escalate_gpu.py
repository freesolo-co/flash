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


def test_is_cuda_oom_is_structured(monkeypatch):
    import torch

    from flash.engine.worker.perf import lifecycle as lc

    assert lc.is_cuda_oom(torch.cuda.OutOfMemoryError("x")) is True  # typed signal
    # NO string matching: a RuntimeError that SAYS "out of memory" is not an OOM without a real signal
    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 0)
    assert lc.is_cuda_oom(RuntimeError("Triton Error [CUDA]: out of memory")) is False
    assert lc.is_cuda_oom(ValueError("bad config")) is False
    # torch's allocator counter advancing is the structured "error code"
    monkeypatch.setattr(lc, "cuda_oom_count", lambda: 2)
    assert lc.is_cuda_oom(RuntimeError("whatever")) is True
    # a host-RAM OOM is never a GPU OOM (a bigger card can't fix it), even with the counter pinned >0
    assert lc.is_cuda_oom(MemoryError("host ram: out of memory")) is False


def _card(gpu, vram):
    return types.SimpleNamespace(gpu=gpu, vram_gb=vram, provider="runpod", hourly_usd=1.0)


def test_oom_escalated_keeps_only_strictly_larger_cards():
    from flash.runner.lifecycle import _oom_escalated

    cands = [_card("A100", 80), _card("A100b", 80), _card("Pro6000", 96), _card("B200", 180)]
    assert [c.gpu for c in _oom_escalated(cands, 0)] == [c.gpu for c in cands]  # no OOM -> unchanged
    assert {c.gpu for c in _oom_escalated(cands, 80)} == {"Pro6000", "B200"}  # >80 only
    assert _oom_escalated(cands, 180) == []  # OOM'd the biggest -> nowhere larger


def test_worker_flagged_oom_reads_the_flag():
    from flash.providers.runpod.jobs import worker_flagged_oom

    assert worker_flagged_oom(lambda force=False: {"oom": True}) is True
    assert worker_flagged_oom(lambda force=False: {"oom": False}) is False
    assert worker_flagged_oom(lambda force=False: {"retriable": True}) is False
    assert worker_flagged_oom(None) is False


def _read(mod):
    return pathlib.Path(__import__(mod, fromlist=["x"]).__file__).read_text()


def test_worker_stamps_oom_flag():
    src = _read("flash.engine.worker")
    assert "oom = is_cuda_oom(e) and not retriable" in src
    assert '{"retriable": retriable, "oom": oom}' in src


def test_poller_maps_oom_flag_to_oom_failure():
    src = _read("flash.providers.runpod.jobs")
    assert src.count('failure="oom" if oom else ("job_preempted" if retriable else "job_failed")') == 2
    assert "def worker_flagged_oom" in src


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
