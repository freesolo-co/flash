"""Orchestrator end-to-end path with the Flash submission mocked (no RunPod calls)."""

from __future__ import annotations

import importlib
import json
import os
import tempfile


def test_run_job_persists_flash_metrics(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import flash.providers.runpod.train as flash_train
        import flash.runner as runner

        importlib.reload(flash_train)
        importlib.reload(runner)
        # Storage roots are fixed constants now; redirect via monkeypatch (auto-restored).
        monkeypatch.setattr(runner, "RUNS_DIR", os.path.join(tmp, "runs"))
        monkeypatch.setattr(runner, "RESULTS_DIR", os.path.join(tmp, "results"))
        from flash.spec import GpuSpec, JobSpec, TrainSpec

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

        # _run_job does `from flash.providers.runpod.train import submit_train, upload_code` at call time.
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
        # 1h on a 4090 at the projected rate (static fallback under FLASH_SKIP_NET)
        from flash.providers.runpod.pricing import hourly_rate

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


def test_upload_code_forces_private_on_reused_repo(monkeypatch):
    """Run artifact repos are ALWAYS private. create_repo(exist_ok=True) is a no-op on an existing
    repo, so a repo previously created public must still be flipped private via
    update_repo_settings — otherwise reused/public repos leak run code, adapters, and metrics."""
    import sys
    import types

    calls = {"create": [], "settings": [], "upload": []}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo, **kw):
            calls["create"].append((repo, kw))

        def update_repo_settings(self, **kw):
            calls["settings"].append(kw)

        def upload_folder(self, **kw):
            calls["upload"].append(kw)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    import flash.providers.runpod.train as flash_train

    assert flash_train.upload_code("owner/run-artifacts") == "owner/run-artifacts"

    # created private...
    assert calls["create"], "create_repo was not called"
    assert calls["create"][0][1].get("private") is True
    # ...AND visibility forced private on the (possibly pre-existing public) repo
    assert calls["settings"], "update_repo_settings was not called — reused public repo can leak"
    assert calls["settings"][0].get("private") is True
    assert calls["settings"][0].get("repo_id") == "owner/run-artifacts"
