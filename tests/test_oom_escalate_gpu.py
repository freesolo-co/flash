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


def test_is_cuda_oom_none_is_never_oom(monkeypatch):
    # No exception object must NEVER classify as OOM — not even when the allocator counter is pinned >0
    # (e.g. an OOM the run already recovered from). A missing exception => no spurious larger-GPU escalation.
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
        return {"oom": True, "retriable": False, "stage": "rl_train"}

    _key, retriable, oom = surfaced_worker_flags(reader, None, say)
    assert (retriable, oom) == (False, True)
    assert reads["n"] == 1  # surfacing + both flags share ONE forced read
    assert surfaced_worker_flags(lambda force=False: {"retriable": True}, None, say)[1:] == (True, False)
    assert surfaced_worker_flags(None, None, say)[1:] == (False, False)
    # A prior attempt's lingering {"oom": true} must NOT be honored for a DIFFERENT attempt: when the
    # caller passes the attempt it is polling, trust the oom flag only when the heartbeat's own
    # worker-stamped `attempt` matches (retriable is NOT attempt-gated — see surfaced_worker_flags).
    stale = lambda force=False: {"oom": True, "attempt": "0", "retriable": False}  # noqa: E731
    assert surfaced_worker_flags(stale, None, say, 0)[1:] == (False, True)  # this attempt's own flag
    assert surfaced_worker_flags(stale, None, say, 1)[1:] == (False, False)  # STALE prior-attempt flag


def test_oom_from_hb_attempt_gates_stale_flag():
    from flash.providers.runpod.jobs import _oom_from_hb

    # The seed heartbeat path is SHARED across attempts, so a prior attempt's lingering {"oom": true}
    # must not escalate a fresh non-OOM failure — gate on the heartbeat's own worker-stamped `attempt`
    # (a str env var) matching the attempt being polled.
    assert _oom_from_hb({"oom": True, "attempt": "0"}, 0) is True  # this attempt's own flag
    assert _oom_from_hb({"oom": True, "attempt": "0"}, 1) is False  # STALE prior-attempt flag
    assert _oom_from_hb({"oom": True, "attempt": "1"}, 1) is True
    assert _oom_from_hb({"oom": True}, 1) is False  # no attempt stamp -> can't confirm it's ours
    assert _oom_from_hb({"oom": True, "attempt": "0"}, None) is True  # no expected attempt -> trust
    assert _oom_from_hb(None, 0) is False
    assert _oom_from_hb({"retriable": True}, 0) is False


def _read(mod):
    return pathlib.Path(__import__(mod, fromlist=["x"]).__file__).read_text()


def test_worker_stamps_oom_flag():
    # AST-based (not an exact source string) so it survives ruff/black reflow while still proving the
    # SHORT-CIRCUIT STRUCTURE: `oom = not retriable and is_cuda_oom(e)` — the `not retriable` guard is the
    # FIRST `and` operand, so is_cuda_oom (which can touch torch/CUDA) runs only for non-retriable fails.
    tree = ast.parse(_read("flash.engine.worker"))
    oom_assigns = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "oom" for t in n.targets)
    ]
    assert oom_assigns, "worker never assigns an `oom` flag"
    val = oom_assigns[0].value
    assert isinstance(val, ast.BoolOp), "`oom` must be a boolean (`and`) expression"
    assert isinstance(val.op, ast.And), "`oom` must short-circuit via `and`"
    first, second = val.values[0], val.values[1]
    # First `and` operand must be `not retriable` (short-circuit BEFORE is_cuda_oom).
    assert isinstance(first, ast.UnaryOp), "first operand must be a `not ...` guard"
    assert isinstance(first.op, ast.Not), "first operand must be `not retriable`"
    assert isinstance(first.operand, ast.Name), "first operand must negate a bare name"
    assert first.operand.id == "retriable", "first operand must be `not retriable`"
    # Second `and` operand must be the is_cuda_oom(...) probe (only reached for non-retriable fails).
    assert isinstance(second, ast.Call), "second operand must be a call"
    assert isinstance(second.func, ast.Name), "second operand must call a bare name"
    assert second.func.id == "is_cuda_oom", "second operand must be is_cuda_oom(...)"
    # The flag ships to the poller in the error heartbeat alongside ``retriable`` (order-independent).
    src = _read("flash.engine.worker")
    assert '"oom": oom' in src
    assert '"retriable": retriable' in src


def test_poller_maps_oom_flag_to_oom_failure():
    src = _read("flash.providers.runpod.jobs")
    assert src.count('"oom" if oom else') == 2  # oom wins in both worker-fail paths
    assert "def surfaced_worker_flags" in src  # single-read helper feeds both paths
    assert "def _oom_from_hb" in src  # the attempt-gating helper exists
    assert "_oom_from_hb(hb, current_attempt)" in src  # surfaced_worker_flags gates the oom flag
    assert "current_attempt=int(attempt)" in src  # submit_run forwards the polled attempt
    # Both worker-fail poll paths thread the attempt being polled into the single-read helper so a
    # stale prior-attempt heartbeat can't escalate (AST: surfaced_worker_flags is called twice and each
    # call forwards a ``current_attempt`` positional arg — robust to formatting/reflow).
    calls = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "surfaced_worker_flags"
    ]
    assert len(calls) == 2
    for c in calls:
        assert any(isinstance(a, ast.Name) and a.id == "current_attempt" for a in c.args)


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
    assert "retry_shaped = infra_shaped or oom_shaped" in body  # oom folds into the retry decision
    assert "if not will_retry:" in body  # single exit check (oom retries, not fail-fast)
    assert "oom_vram_floor = max(oom_vram_floor, chosen.vram_gb)" in body
    assert "_oom_escalated(alloc.candidates, oom_vram_floor)" in body
    assert "if not oom_shaped:" in body  # oom grows the card, doesn't escape the provider
    # OOM escalation is cost -> bounded by the user's RAW max_retries, not the floored infra budget.
    assert "retry_budget = max_retries if oom_shaped else infra_budget" in body
