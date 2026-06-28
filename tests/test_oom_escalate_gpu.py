"""CUDA-OOM -> escalate the retry to a strictly LARGER GPU (CPU-only; no GPU/network).

A training CUDA OOM means the card was too small for THIS run's peak. Before this change it was a
``job_failed`` and failed fast (no walk). Now the worker stamps a structured ``oom`` heartbeat flag,
the poller maps it to ``failure="oom"``, and the runner retries on a strictly larger class (a
same-size walk would just OOM again) — failing terminally only once the largest available card OOMs.
These cover the pure pieces + the wiring; the end-to-end escalation needs a live-GPU run.
"""

from __future__ import annotations

import ast
import pathlib


def test_is_cuda_oom_matches_the_real_traceback_not_generic_errors():
    from flash.engine.worker.perf.lifecycle import is_cuda_oom

    # the actual prod crash (fla/Triton raises a plain RuntimeError, not torch's typed OOM)
    assert is_cuda_oom(RuntimeError("Triton Error [CUDA]: out of memory")) is True
    assert is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")) is True
    # detection also works off the traceback text when the top exception is a wrapper
    tb = "...\n  File fla/modules/l2norm.py ...\nRuntimeError: Triton Error [CUDA]: out of memory\n"
    assert is_cuda_oom(None, tb) is True
    # genuine NON-oom training/config errors must NOT escalate (they'd waste a bigger GPU)
    assert is_cuda_oom(ValueError("bad config")) is False
    assert is_cuda_oom(RuntimeError("loss became nan")) is False
    # an OOM with NO CUDA/Triton context is a host-RAM / other-library OOM — must NOT trigger GPU
    # escalation just because the substring "out of memory" appears.
    assert is_cuda_oom(RuntimeError("DataLoader worker (pid 1) killed: out of memory")) is False
    assert is_cuda_oom(MemoryError("Unable to allocate array: out of memory")) is False


def _card(gpu, vram, provider="runpod", price=1.0):
    import types

    return types.SimpleNamespace(gpu=gpu, vram_gb=vram, provider=provider, hourly_usd=price)


def test_oom_escalated_keeps_only_strictly_larger_cards():
    from flash.runner.lifecycle import _oom_escalated

    cands = [
        _card("A100 PCIe", 80),
        _card("A100 SXM", 80),
        _card("RTX Pro 6000", 96),
        _card("H200", 141),
        _card("B200", 180),
    ]
    # no prior OOM -> list unchanged
    assert [c.gpu for c in _oom_escalated(cands, 0)] == [c.gpu for c in cands]
    # after an 80GB OOM -> only the >80GB classes (NOT the other 80GB A100)
    assert {c.gpu for c in _oom_escalated(cands, 80)} == {"RTX Pro 6000", "H200", "B200"}
    # OOM'd the biggest -> nowhere larger to escalate to (caller fails terminally)
    assert _oom_escalated(cands, 180) == []


def test_worker_flagged_oom_reads_the_structured_flag():
    from flash.providers.runpod.jobs import worker_flagged_oom

    assert worker_flagged_oom(lambda force=False: {"oom": True}) is True
    assert worker_flagged_oom(lambda force=False: {"oom": False}) is False
    assert worker_flagged_oom(lambda force=False: {"retriable": True}) is False  # not an OOM
    assert worker_flagged_oom(lambda force=False: "not a dict") is False
    assert worker_flagged_oom(None) is False


def test_once_forced_reads_underlying_heartbeat_only_once():
    # A terminal path surfaces the heartbeat AND extracts retriable+oom from it; _once_forced makes
    # those share ONE underlying force-read (fewer HF round-trips, no cross-snapshot race).
    from flash.providers.runpod.jobs import (
        _once_forced,
        worker_flagged_oom,
        worker_flagged_retriable,
    )

    calls = {"n": 0}

    def underlying(force=False):
        calls["n"] += 1
        return {"oom": True, "retriable": False}

    reader = _once_forced(underlying)
    # three logical consumers (surface + 2 flags) -> still one underlying read
    reader(force=True)
    assert worker_flagged_oom(reader) is True
    assert worker_flagged_retriable(reader) is False
    assert calls["n"] == 1
    assert _once_forced(None) is None


def test_lambda_poller_escalates_oom_for_cross_provider_parity():
    # The Lambda poller must also map a worker OOM flag to failure="oom" so a Lambda candidate's OOM
    # escalates to a larger GPU instead of failing fast as job_failed (parity with RunPod's poll_job).
    src, _ = _src("flash.providers.lambdalabs.jobs", "")
    # the oom flag is read in BOTH Lambda worker-fail paths (marker + dead-host error_*.txt crash)
    assert src.count("worker_flagged_oom(heartbeat_reader)") >= 2
    # marker path: oom wins over the retriable/job_failed split
    assert 'failure="oom" if oom else ("job_preempted" if retriable else "job_failed")' in src
    # dead-host crash path: oom takes precedence before the crash/preempt split
    assert '"job_failed" if worker_crashed else "job_preempted"' in src


def _src(modfile_attr, name):
    import importlib

    mod = importlib.import_module(modfile_attr)
    return pathlib.Path(mod.__file__).read_text(), name


def test_worker_stamps_oom_flag_in_its_failure_heartbeat():
    # The worker classifies OOM off its LIVE exception and stamps it alongside `retriable`, so the
    # poller never has to parse the failure detail string.
    src, _ = _src("flash.engine.worker", "")
    assert "oom = is_cuda_oom(e, tb)" in src
    assert '{"retriable": retriable, "oom": oom}' in src


def test_poller_maps_oom_flag_to_the_oom_failure_category():
    src, _ = _src("flash.providers.runpod.jobs", "")
    # the oom flag wins over retriable/job_failed in BOTH worker-fail paths
    assert src.count('failure="oom" if oom else ("job_preempted" if retriable else "job_failed")') == 2
    assert "def worker_flagged_oom" in src


def test_runner_classifies_oom_as_retriable_and_escalates_vram():
    src = pathlib.Path(
        __import__("flash.runner.lifecycle", fromlist=["x"]).__file__
    ).read_text()
    tree = ast.parse(src)
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_submit_seed_supervised"
    )
    body = ast.get_source_segment(src, fn)
    # OOM is a retry category (NOT fail-fast) AND it grows the escalation floor
    assert 'oom_shaped = res.failure == "oom"' in body
    assert "infra_shaped or oom_shaped" in body  # OOM joins the retry-shaped categories
    assert "if not retry_shaped:" in body  # the fail-fast break now honors oom too
    assert "oom_vram_floor = max(oom_vram_floor, int(chosen.vram_gb))" in body
    # the next attempt restricts to strictly-larger cards, and an OOM does NOT escape the provider
    assert "_oom_escalated(alloc.candidates, oom_vram_floor)" in body
    assert "if not oom_shaped:" in body  # guards failed_providers.add
    # an OOM on the largest class terminates immediately (no extra "retrying on a larger GPU" spin)
    assert "oom_no_larger" in body
    assert "retry_shaped = (infra_shaped or oom_shaped) and not oom_no_larger" in body
