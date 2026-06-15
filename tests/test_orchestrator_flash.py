"""Orchestrator end-to-end path with the Flash submission mocked (no RunPod calls)."""

from __future__ import annotations

import importlib
import json
import os
import tempfile


def test_run_job_persists_flash_metrics(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import autoslm.providers.runpod.train as flash_train
        import autoslm.runner as runner

        importlib.reload(flash_train)
        importlib.reload(runner)
        # Storage roots are fixed constants now; redirect via monkeypatch (auto-restored).
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        monkeypatch.setattr(runner, "RESULTS_DIR", os.path.join(tmp, "results"))
        from autoslm.spec import GpuSpec, JobSpec, TrainSpec

        captured = {}

        def fake_submit(spec, seed, log=None):
            captured["gpu"] = spec.gpu.type
            captured["seed"] = seed
            return {
                "arm": "runpod",
                "phase": spec.phase,
                "seed": seed,
                "wall_seconds": 3600.0,
                "trained_eval_acc": 0.7,
                "base_eval_acc": 0.3,
                "cost_usd": 0.0,
                "notes": {},
            }

        # _run_job does `from autoslm.providers.runpod.train import submit_train, upload_code` at call time.
        monkeypatch.setattr(flash_train, "submit_train", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "mock/repo")

        spec = JobSpec(
            run_id="flash-run",
            model="Qwen/Qwen3.5-4B",
            algorithm="grpo",
            train=TrainSpec(steps=2, seeds=(0,)),
            gpu=GpuSpec(type="RTX 4090"),
        )
        status = runner.submit_job(spec, dry_run=False, background=False)

        assert status.state == "done", status.error
        # 1h on a 4090 at the projected rate (static fallback under AUTOSLM_SKIP_NET)
        from autoslm.providers.runpod.pricing import hourly_rate

        assert abs(status.cost_usd - hourly_rate("RTX 4090")) < 1e-6, status.cost_usd
        assert captured["gpu"] == "RTX 4090"

        # Metrics are namespaced by run id so same-phase runs cannot collide.
        metrics_path = os.path.join(
            tmp, "results", "runpod", "rl", status.run_id, "seed0", "metrics.json"
        )
        assert os.path.exists(metrics_path)
        with open(metrics_path) as f:
            m = json.load(f)
        assert m["trained_eval_acc"] == 0.7
        assert m["notes"]["runpod_gpu"] == "RTX 4090"
