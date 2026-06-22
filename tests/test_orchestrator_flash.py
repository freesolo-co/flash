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
        # _run_job_inner uploads the run code before the seed loop; stub it (no HF).
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "mock/repo")
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        captured = {}

        def fake_submit(spec, seed, log=None, **kwargs):
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

        # Stub the per-seed submit/poll path (the seam that used to be the in-process
        # offline shortcut) so the run completes without provisioning a GPU.
        # _run_seed_loop resolves it via `from flash.runner import _submit_seed_supervised`.
        monkeypatch.setattr(runner, "_submit_seed_supervised", fake_submit)

        spec = JobSpec(
            run_id="flash-run",
            model="Qwen/Qwen3.5-4B",
            algorithm="grpo",
            train=TrainSpec(steps=2, seeds=(0,)),
            gpu=GpuSpec(type="RTX 4090"),
        )
        status = runner.submit_job(spec, dry_run=False, background=False)

        assert status.state == "done", status.error
        # 1h on a 4090 at the projected rate (static fallback, no live pricing)
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


def test_upload_code_mirrors_package_purging_stale_remote(monkeypatch):
    """The upload must MIRROR the local flash package: delete_patterns=['**'] (relative to
    code/flash) so any orphaned/renamed remote module from a prior commit is purged, not left for
    the worker to re-import. This is the deployment-robustness guard against a run picking up OLD
    code in code/flash after a redeploy (the "missing recent fixes on submit" symptom)."""
    import os
    import sys
    import types

    import flash

    calls = {"upload": []}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo, **kw):
            pass

        def update_repo_settings(self, **kw):
            pass

        def upload_folder(self, **kw):
            calls["upload"].append(kw)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    import flash.providers.runpod.train as flash_train

    flash_train.upload_code("owner/run-artifacts")
    assert calls["upload"], "upload_folder was not called"
    up = calls["upload"][0]
    assert up["path_in_repo"] == "code/flash"
    # the exact-mirror guard: delete everything under code/flash not in this upload
    assert up.get("delete_patterns") == ["**"]
    # still uploads from the real (symlink-collapsed) package dir, and skips bytecode
    assert up["folder_path"] == os.path.realpath(os.path.dirname(os.path.abspath(flash.__file__)))
    assert "*.pyc" in up.get("ignore_patterns", [])


def test_upload_code_stages_optional_chalk_wheel(monkeypatch, tmp_path):
    """An unpublished chalk wheel can ride in the run-private code artifact for live validation."""
    import sys
    import types

    wheel = tmp_path / "freesolo_chalk-0.4.12-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    calls = {"wheel": []}

    class _FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo, **kw):
            pass

        def update_repo_settings(self, **kw):
            pass

        def upload_folder(self, **kw):
            pass

        def upload_file(self, **kw):
            calls["wheel"].append(kw)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setenv("FLASH_CHALK_WHEEL", str(wheel))

    import flash.providers.runpod.train as flash_train

    flash_train.upload_code("owner/run-artifacts")
    assert calls["wheel"] == [
        {
            "path_or_fileobj": str(wheel),
            "path_in_repo": "code/wheels/freesolo_chalk-0.4.12-py3-none-any.whl",
            "repo_id": "owner/run-artifacts",
            "repo_type": "dataset",
        }
    ]
