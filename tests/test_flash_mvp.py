"""CPU tests for the Flash MVP package."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit


def test_catalog_validation():
    from flash.core.catalog import get_model, validate_model_for_algorithm

    info = get_model("Qwen/Qwen3.5-9B")
    assert "grpo" in info.algos
    # 9B is the bf16 GRPO tier (needs an 80 GB-class card; QLoRA was dropped).
    assert validate_model_for_algorithm("Qwen/Qwen3.5-9B", "grpo").id == "Qwen/Qwen3.5-9B"
    # An sft-only model still rejects grpo.
    from flash.core.catalog import MODELS, ModelInfo

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
    from flash.schema import spec_and_train_keys_from_file

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "run.toml")
        with open(path, "w") as f:
            f.write(
                'model = "Qwen/Qwen3.5-9B"\n'
                'algorithm = "grpo"\n'
                "[environment]\n"
                'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
                "[train]\n"
                "epochs = 1\n"
                "max_examples = 10\n"
                "[gpu]\n"
                ""
            )
        spec = spec_and_train_keys_from_file(path, run_id="test-run")[0]
        assert spec.run_id == "test-run"
        assert spec.phase == "rl"


def test_environment_registry():
    from flash.envs.loading.base import load_environment

    # Verifiers-only: there are no builtin envs and no default — an empty id is a hard
    # error (env loading itself is covered in test_envs_coverage).
    with pytest.raises(ValueError, match="no environment specified"):
        load_environment("")


def test_orchestrator_dry_run(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        # fixed constant; redirect to tmp via monkeypatch so it's restored after the test.
        monkeypatch.setattr(runner_state, "RUNS_DIR", tmp)
        from flash.core.spec import JobSpec, TrainSpec

        spec = JobSpec(
            run_id="dry",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(max_examples=8),
        )
        status = runner_submit.submit_job(spec, dry_run=True)
        assert status.state == "dry_run"
        assert runner_status.get_status("dry").spec["model"] == "Qwen/Qwen3.5-9B"


def test_cli_train_dry_run(monkeypatch, capsys):
    # `flash train --dry-run` routes through the server (`create_run(dry_run=True)`) so the control
    # plane runs the real submit-time preflights without allocating a GPU; the CLI renders the
    # returned state=dry_run status. A fake client stands in for the server here.
    from flash.cli.parsing.main import main

    with tempfile.TemporaryDirectory() as tmp:
        config = os.path.join(tmp, "run.toml")
        with open(config, "w") as f:
            f.write(
                'model = "Qwen/Qwen3.5-9B"\n'
                'project = "11111111-1111-4111-8111-111111111111"\n'
                'algorithm = "grpo"\n'
                "[environment]\n"
                'id = "github:freesolo-co/envs@main:gsm8k/environment.py"\n'
                "[train]\n"
                "epochs = 1\n"
                "max_examples = 1\n"
                "[gpu]\n"
                ""
            )

        seen = {}

        class _FakeClient:
            def create_run(
                self, spec, runtime_secrets=None, dry_run=False, client_train_schema=None
            ):
                seen["dry_run"] = dry_run
                seen["client_train_schema"] = client_train_schema
                return {"run_id": "flash-dry-1", "state": "dry_run", "spec": spec}

        monkeypatch.setattr(
            "flash.cli.commands.ops.train.client_from_config", lambda: _FakeClient()
        )

        rc = main(["train", config, "--dry-run"])
        assert rc == 0
        assert seen["dry_run"] is True
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert payload["state"] == "dry_run"
        assert "dry-run validated" not in captured.out
        assert "dry-run validated: config/schema" in captured.err
        assert "did NOT import or run your environment.py" in captured.err
