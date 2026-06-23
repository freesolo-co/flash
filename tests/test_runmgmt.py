"""Tests for run-management helpers (runs/cost/cancel) — no GPU/network."""

from __future__ import annotations

import importlib
import tempfile


def test_list_and_cancel(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import flash.runner as runner

        importlib.reload(runner)
        # fixed constant; redirect to tmp via monkeypatch so it's restored after the test.
        monkeypatch.setattr(runner, "RUNS_DIR", tmp)
        from flash.spec import JobSpec

        # two dry-run records
        for rid in ("a", "b"):
            runner.submit_job(
                JobSpec(run_id=rid, model="Qwen/Qwen3.5-4B", algorithm="grpo"),
                dry_run=True,
            )
        runs = {r.run_id for r in runner.list_runs()}
        assert {"a", "b"} <= runs

        # cancel a non-terminal run (force it to a running-ish state first; go through
        # _save_status, not _update, since _update now refuses to resurrect a terminal
        # state like the submitted dry_run).
        running = runner.get_status("a")
        running.state = "running"
        runner._save_status(running)
        status = runner.cancel_run("a")
        assert status.state == "cancelled"

        # cancelling a terminal run is a no-op
        same = runner.cancel_run("b")  # b is dry_run (terminal-ish)
        assert same.state in {"dry_run", "cancelled"}


def test_persist_metrics_keeps_stamped_zero_vast(monkeypatch):
    import json
    import os

    with tempfile.TemporaryDirectory() as tmp:
        import flash.runner as runner

        importlib.reload(runner)
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp)
        from flash.spec import JobSpec

        spec = JobSpec(run_id="v1", model="Qwen/Qwen3.5-4B", algorithm="grpo")
        # A near-zero-duration Vast run stamps cost_usd=0.0 + notes.provider="vast".
        # This 0.0 is legitimate and must NOT trigger the RunPod wall-pricing fallback
        # (which would recompute cost and overwrite provider notes with 'runpod') —
        # the fallback keys off the absence of a non-runpod provider note, not the value.
        metrics = {
            "cost_usd": 0.0,
            "wall_seconds": 1.0,
            "notes": {"provider": "vast", "vast_rate_usd_hr": 0.5},
        }
        out = runner._persist_metrics(spec, 0, metrics)
        assert out == 0.0
        with open(os.path.join(runner.artifacts_dir(spec), "seed0", "metrics.json")) as f:
            on_disk = json.load(f)
        assert on_disk["cost_usd"] == 0.0
        assert on_disk["notes"]["provider"] == "vast"
        assert "runpod_rate_usd_hr" not in on_disk["notes"]


def test_persist_metrics_falls_back_when_cost_absent(monkeypatch):
    import json
    import os

    with tempfile.TemporaryDirectory() as tmp:
        import flash.runner as runner

        importlib.reload(runner)
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp)
        monkeypatch.setattr(runner, "_gpu_rate", lambda gpu: 3600.0)
        from flash.spec import JobSpec

        spec = JobSpec(run_id="r1", model="Qwen/Qwen3.5-4B", algorithm="grpo")
        # No cost_usd stamped (RunPod path): fall back to wall * rate and attribute runpod.
        out = runner._persist_metrics(spec, 0, {"wall_seconds": 1.0, "allocated_gpu": "RTX 5090"})
        assert out == 1.0  # 1s / 3600 * 3600/hr
        with open(os.path.join(runner.artifacts_dir(spec), "seed0", "metrics.json")) as f:
            on_disk = json.load(f)
        assert on_disk["notes"]["provider"] == "runpod"
