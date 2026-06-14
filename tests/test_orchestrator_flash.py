"""Orchestrator end-to-end path with the Flash submission mocked (no RunPod calls)."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_run_job_persists_flash_metrics(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import autoslm.flash.train as flash_train
        import autoslm.orchestrator as orchestrator

        importlib.reload(flash_train)
        importlib.reload(orchestrator)
        # Storage roots are fixed constants now; redirect via monkeypatch (auto-restored).
        monkeypatch.setattr(orchestrator, "RUNS_DIR", os.path.join(tmp, "runs"))
        monkeypatch.setattr(orchestrator, "RESULTS_DIR", os.path.join(tmp, "results"))
        from autoslm.worker_spec import GpuSpec, JobSpec, TrainSpec

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

        # _run_job does `from autoslm.flash.train import submit_train, upload_code` at call time.
        monkeypatch.setattr(flash_train, "submit_train", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda: "mock/repo")

        spec = JobSpec(
            run_id="flash-run",
            model="Qwen/Qwen3-4B-Instruct-2507",
            algorithm="grpo",
            train=TrainSpec(steps=2, seeds=(0,)),
            gpu=GpuSpec(type="RTX 4090"),
        )
        status = orchestrator.submit_job(spec, dry_run=False, background=False)

        assert status.state == "done", status.error
        # 1h on a 4090 at the projected rate (static fallback under AUTOSLM_SKIP_NET)
        from autoslm.flash.pricing import hourly_rate

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

        os.environ.pop("AUTOSLM_RUNS_DIR", None)
        os.environ.pop("RESULTS_DIR", None)
