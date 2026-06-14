"""Spec plumbing not covered elsewhere: config validation error paths, JobSpec
serialization round-trips, worker env loading, run-id path containment, VRAM
estimates, and logging namespace helpers."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoslm.config_schema import ConfigError, spec_from_dict
from autoslm.worker_spec import JobSpec, load_job_spec_from_env

BASE_RAW = {
    "model": "Qwen/Qwen3-0.6B",
    "algorithm": "grpo",
    "environment": {"id": "gsm8k"},
    "train": {"steps": 10, "lora_rank": 8, "seeds": [0]},
    "gpu": {"type": "RTX 4090"},
}


def _raw(**overrides) -> dict:
    raw = json.loads(json.dumps(BASE_RAW))
    for key, value in overrides.items():
        section, _, leaf = key.partition(".")
        if leaf:
            raw.setdefault(section, {})[leaf] = value
        else:
            raw[section] = value
    return raw


# ---------------------------------------------------------------------------
# config_schema validation error paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"train.seeds": []}, "at least one seed"),
        ({"train.steps": 0}, "steps must be positive"),
        ({"train.lora_rank": 0}, "lora_rank must be positive"),
        ({"train.lora_alpha": 0}, "lora_alpha must be positive"),
        ({"train.lora_alpha": -8}, "lora_alpha must be positive"),
        ({"algorithm": "ppo"}, "unsupported algorithm"),
        ({"model_policy": "yolo"}, "model_policy"),
        ({"gpu.allow_unvalidated": "yes"}, "must be a boolean"),
    ],
)
def test_spec_validation_rejections(overrides, match) -> None:
    with pytest.raises(ConfigError, match=match):
        spec_from_dict(_raw(**overrides))


def test_sft_epochs_must_be_positive() -> None:
    raw = _raw(algorithm="sft")
    raw["train"] = {"epochs": 0, "lora_rank": 8, "seeds": [0]}
    with pytest.raises(ConfigError, match="epochs must be positive"):
        spec_from_dict(raw)


def test_missing_model_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must set `model`"):
        spec_from_dict({"algorithm": "sft"})


def test_relative_environment_path_is_anchored_to_config_dir(tmp_path) -> None:
    raw = _raw()
    raw["environment"] = {"id": "x", "path": "envs/custom"}
    spec = spec_from_dict(raw, base_dir=str(tmp_path))
    assert spec.environment.path == os.path.normpath(str(tmp_path / "envs" / "custom"))


# ---------------------------------------------------------------------------
# JobSpec serialization round-trips (what travels client -> server -> worker)
# ---------------------------------------------------------------------------


def test_job_spec_json_round_trip() -> None:
    spec = spec_from_dict(_raw(), run_id="rt-1")
    restored = JobSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.train.seeds == (0,)
    assert restored.phase == "rl"  # grpo's internal phase id


def test_load_job_spec_from_env_json_and_path(tmp_path, monkeypatch) -> None:
    spec = spec_from_dict(_raw(), run_id="env-1")

    monkeypatch.setenv("AUTOSLM_JOB_SPEC_JSON", spec.to_json())
    assert load_job_spec_from_env() == spec

    monkeypatch.delenv("AUTOSLM_JOB_SPEC_JSON")
    path = tmp_path / "spec.json"
    path.write_text(spec.to_json(), encoding="utf-8")
    monkeypatch.setenv("AUTOSLM_JOB_SPEC_PATH", str(path))
    assert load_job_spec_from_env() == spec

    monkeypatch.delenv("AUTOSLM_JOB_SPEC_PATH")
    assert load_job_spec_from_env() is None


# ---------------------------------------------------------------------------
# orchestrator: run-id containment + dry-run/list/cancel surface
# ---------------------------------------------------------------------------


def _fresh_orchestrator(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOSLM_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    import autoslm.orchestrator as orchestrator

    return importlib.reload(orchestrator)


def test_runs_file_path_rejects_traversal(tmp_path, monkeypatch) -> None:
    orch = _fresh_orchestrator(tmp_path, monkeypatch)
    for bad in ("../escape", "a/b", "", "x" * 200, ".hidden"):
        with pytest.raises(ValueError, match="invalid run_id"):
            orch.runs_file_path(bad, ".json")
    good = orch.runs_file_path("autoslm-123-abc", ".log")
    assert good.endswith("autoslm-123-abc.log")


def test_dry_run_submit_get_list_logs_cancel(tmp_path, monkeypatch) -> None:
    orch = _fresh_orchestrator(tmp_path, monkeypatch)
    spec = spec_from_dict(_raw())

    status = orch.submit_job(spec, dry_run=True)
    assert status.state == "dry_run"
    assert orch.get_status(status.run_id).state == "dry_run"
    assert status.run_id in [r.run_id for r in orch.list_runs()]
    assert orch.get_logs(status.run_id) == ""  # no log yet, no crash

    # terminal runs cancel as a no-op (state preserved)
    assert orch.cancel_run(status.run_id).state == "dry_run"

    with pytest.raises(FileNotFoundError, match="unknown run_id"):
        orch.get_status("autoslm-000-nope")


def test_artifacts_dir_and_adapter_prefix_helpers(tmp_path, monkeypatch) -> None:
    orch = _fresh_orchestrator(tmp_path, monkeypatch)
    spec = spec_from_dict(_raw(), run_id="autoslm-1-x")
    assert orch.artifacts_dir(spec).endswith(os.path.join("results", "runpod", "rl", "autoslm-1-x"))
    assert orch.adapter_prefix(spec) == "rl/autoslm-1-x/seed0"
    assert orch.adapter_prefix(spec, seed=3) == "rl/autoslm-1-x/seed3"


# ---------------------------------------------------------------------------
# engine.vram: fit estimates + offline param lookup
# ---------------------------------------------------------------------------


def test_vram_estimate_scales_with_params_and_algorithm() -> None:
    from autoslm.engine import vram

    sft_small = vram.estimate_vram_gb(0.6, "sft")
    sft_big = vram.estimate_vram_gb(8.0, "sft")
    grpo_big = vram.estimate_vram_gb(8.0, "grpo")
    assert sft_small < sft_big < grpo_big  # GRPO colocates vLLM on top of the trainer


def test_fetch_hf_params_is_offline_safe(monkeypatch) -> None:
    from autoslm.engine import vram

    monkeypatch.setenv("AUTOSLM_SKIP_NET", "1")
    assert vram.fetch_hf_params_b("any/model") is None


# ---------------------------------------------------------------------------
# _logging: namespace + level resolution
# ---------------------------------------------------------------------------


def test_get_logger_namespacing() -> None:
    from autoslm._logging import get_logger

    assert get_logger().name == "autoslm"
    assert get_logger("autoslm").name == "autoslm"
    assert get_logger("autoslm.flash").name == "autoslm.flash"
    assert get_logger("mymodule").name == "autoslm.mymodule"


def test_log_level_from_env(monkeypatch) -> None:
    from autoslm import _logging

    monkeypatch.setenv("AUTOSLM_LOG_LEVEL", "debug")
    assert _logging._level_from_env() == logging.DEBUG
    monkeypatch.setenv("AUTOSLM_LOG_LEVEL", "15")
    assert _logging._level_from_env() == 15
    monkeypatch.delenv("AUTOSLM_LOG_LEVEL")
    assert _logging._level_from_env(logging.WARNING) == logging.WARNING
