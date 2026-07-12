"""CPU tests for the Flash MVP package."""

from __future__ import annotations

import importlib
import json
import os
import tempfile

import pytest


def test_catalog_validation():
    from flash.catalog import get_model, validate_model_for_algorithm

    info = get_model("Qwen/Qwen3.5-4B")
    assert "grpo" in info.algos
    # 9B is the bf16 GRPO tier (needs an 80 GB-class card; QLoRA was dropped).
    assert validate_model_for_algorithm("Qwen/Qwen3.5-9B", "grpo").id == "Qwen/Qwen3.5-9B"
    # An sft-only model still rejects grpo.
    from flash.catalog import MODELS, ModelInfo

    MODELS["test/sft-only"] = ModelInfo(
        id="test/sft-only",
        display_name="x",
        params="1B",
        params_b=1.0,
        algos=("sft",),
        min_vram_gb=12,
    )
    try:
        with pytest.raises(ValueError, match="not grpo"):
            validate_model_for_algorithm("test/sft-only", "grpo")
    finally:
        MODELS.pop("test/sft-only", None)


def test_config_to_job_spec():
    from flash.schema import spec_from_file

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "run.toml")
        with open(path, "w") as f:
            f.write(
                'model = "Qwen/Qwen3.5-4B"\n'
                'algorithm = "grpo"\n'
                "[environment]\n"
                'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
                "[train]\n"
                "epochs = 1\n"
                "max_examples = 10\n"
                'hf_repo = "owner/runs"\n'
                "[gpu]\n"
                'type = "RTX 5090"\n'
            )
        spec = spec_from_file(path, run_id="test-run")
        assert spec.run_id == "test-run"
        assert spec.phase == "rl"


def test_environment_registry():
    from flash.envs.registry import load_environment

    # Verifiers-only: there are no builtin envs and no default — an empty id is a hard
    # error (env loading itself is covered in test_envs_coverage).
    with pytest.raises(ValueError, match="no environment specified"):
        load_environment("")


def test_orchestrator_dry_run(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import flash.runner as runner

        importlib.reload(runner)
        # fixed constant; redirect to tmp via monkeypatch so it's restored after the test.
        monkeypatch.setattr(runner, "RUNS_DIR", tmp)
        from flash.spec import JobSpec

        spec = JobSpec(run_id="dry", model="Qwen/Qwen3.5-4B", algorithm="grpo")
        status = runner.submit_job(spec, dry_run=True)
        assert status.state == "dry_run"
        assert runner.get_status("dry").spec["model"] == "Qwen/Qwen3.5-4B"


def test_cli_train_dry_run(monkeypatch, capsys):
    # `flash train --dry-run` routes through the server (`create_run(dry_run=True)`) so the control
    # plane runs the real submit-time preflights without allocating a GPU; the CLI renders the
    # returned state=dry_run status. A fake client stands in for the server here.
    from flash.cli import main

    with tempfile.TemporaryDirectory() as tmp:
        config = os.path.join(tmp, "run.toml")
        with open(config, "w") as f:
            f.write(
                'model = "Qwen/Qwen3.5-4B"\n'
                'algorithm = "grpo"\n'
                "[environment]\n"
                'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
                # A user-set [train] hf_repo is silently ignored (platform-managed, assigned
                # per-run); the dry-run still validates and the resolved hf_repo comes back blank.
                "[train]\n"
                "epochs = 1\n"
                "max_examples = 1\n"
                'hf_repo = "owner/runs"\n'
                "[gpu]\n"
                'type = "RTX 5090"\n'
            )

        seen = {}

        class _FakeClient:
            def health(self):
                from flash import __version__
                from flash.schema import train_schema_metadata

                return {"version": __version__, "train_schema": train_schema_metadata()}

            def create_run(self, spec, runtime_secrets=None, dry_run=False):
                seen["dry_run"] = dry_run
                return {"run_id": "flash-dry-1", "state": "dry_run", "spec": spec}

        monkeypatch.setattr("flash.cli.commands.client_from_config", lambda: _FakeClient())

        rc = main(["train", config, "--dry-run"])
        assert rc == 0
        assert seen["dry_run"] is True
        payload = json.loads(capsys.readouterr().out)
        assert payload["state"] == "dry_run"
