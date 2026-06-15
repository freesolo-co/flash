"""CPU tests for the AutoSLM MVP package."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_catalog_validation():
    from autoslm.catalog import get_model, validate_model_for_algorithm

    info = get_model("Qwen/Qwen3.5-4B")
    assert "grpo" in info.algos
    # Qwen3.6-35B-A3B is GRPO-capable (QLoRA on an 80 GB A100, transformers generation).
    assert (
        validate_model_for_algorithm("Qwen/Qwen3.6-35B-A3B", "grpo").id == "Qwen/Qwen3.6-35B-A3B"
    )
    # An sft-only model still rejects grpo (inject one — no catalog entry is sft-only now).
    from autoslm.catalog import MODELS, ModelInfo

    MODELS["test/sft-only"] = ModelInfo(
        id="test/sft-only", display_name="x", params="1B", algos=("sft",), min_vram_gb=12
    )
    try:
        with pytest.raises(ValueError, match="not grpo"):
            validate_model_for_algorithm("test/sft-only", "grpo")
    finally:
        MODELS.pop("test/sft-only", None)


def test_config_to_job_spec():
    from autoslm.schema import spec_from_file

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "run.toml")
        with open(path, "w") as f:
            f.write(
                'model = "Qwen/Qwen3.5-4B"\n'
                'algorithm = "grpo"\n'
                "[environment]\n"
                'id = "primeintellect/gsm8k"\n'
                "[train]\n"
                "steps = 10\n"
                "seeds = [0, 1]\n"
                'hf_repo = "owner/runs"\n'
                "[gpu]\n"
                'type = "RTX 5090"\n'
            )
        spec = spec_from_file(path, run_id="test-run")
        assert spec.run_id == "test-run"
        assert spec.phase == "rl"
        assert spec.train.seeds == (0, 1)


def test_environment_registry():
    from autoslm.envs.registry import load_environment

    # Verifiers-only: there are no builtin envs and no default — an empty id is a hard
    # error (env loading itself is covered in test_envs_coverage).
    with pytest.raises(ValueError, match="no environment specified"):
        load_environment("")


def test_orchestrator_dry_run(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import autoslm.runner as runner

        importlib.reload(runner)
        # fixed constant; redirect to tmp via monkeypatch so it's restored after the test.
        monkeypatch.setattr(runner, "RUNS_DIR", tmp)
        from autoslm.spec import JobSpec

        spec = JobSpec(run_id="dry", model="Qwen/Qwen3.5-4B", algorithm="grpo")
        status = runner.submit_job(spec, dry_run=True)
        assert status.state == "dry_run"
        assert runner.get_status("dry").spec["model"] == "Qwen/Qwen3.5-4B"


def test_mcp_handler_dry_run(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import autoslm.mcp.server as mcp
        import autoslm.runner as runner

        importlib.reload(runner)
        # fixed constant; redirect to tmp via monkeypatch so it's restored after the test.
        monkeypatch.setattr(runner, "RUNS_DIR", tmp)
        importlib.reload(mcp)
        result = mcp.handle(
            {
                "tool": "create_" + "train" + "ing_run",
                "args": {
                    "run_id": "mcp-dry",
                    "model": "Qwen/Qwen3.5-4B",
                    "algorithm": "grpo",
                    "environment": {"id": "primeintellect/gsm8k"},
                    "train": {"steps": 1, "seeds": [0], "hf_repo": "owner/runs"},
                    "gpu": {"type": "RTX 5090"},
                    "dry_run": True,
                },
            }
        )
        assert result["state"] == "dry_run"


def test_cli_train_dry_run():
    with tempfile.TemporaryDirectory() as tmp:
        config = os.path.join(tmp, "run.toml")
        runs = os.path.join(tmp, "runs")
        with open(config, "w") as f:
            f.write(
                'model = "Qwen/Qwen3.5-4B"\n'
                'algorithm = "grpo"\n'
                "[environment]\n"
                'id = "primeintellect/gsm8k"\n'
                "[train]\n"
                "steps = 1\n"
                "seeds = [0]\n"
                'hf_repo = "owner/runs"\n'
                "[gpu]\n"
                'type = "RTX 5090"\n'
            )
        env = os.environ.copy()
        env["AUTOSLM_RUNS_DIR"] = runs
        proc = subprocess.run(
            [sys.executable, "-m", "autoslm.cli.main", "train", config, "--dry-run"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        assert proc.returncode == 0, proc.stdout
        payload = json.loads(proc.stdout)
        assert payload["state"] == "dry_run"


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
            passed += 1
        except Exception:
            print("FAIL", fn.__name__)
            traceback.print_exc()
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
