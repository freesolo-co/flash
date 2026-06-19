"""Spec plumbing not covered elsewhere: config validation error paths, JobSpec
serialization round-trips, worker env loading, run-id path containment, VRAM
estimates, and logging namespace helpers."""

from __future__ import annotations

import importlib
import json
import logging
import os

import pytest

from flash.schema import ConfigError, spec_from_dict
from flash.spec import JobSpec, load_job_spec_from_env

BASE_RAW = {
    "model": "Qwen/Qwen3.5-0.8B",
    "algorithm": "grpo",
    "environment": {"id": "primeintellect/gsm8k"},
    "train": {"steps": 10, "lora_rank": 8, "seeds": [0], "hf_repo": "owner/runs"},
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
# schema validation error paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"train.seeds": []}, "at least one seed"),
        ({"train.steps": 0}, "steps must be >= 1"),
        # lora_rank/alpha now parse via _train_int(minimum=1), so out-of-range values
        # are rejected at parse time with the shared ">= 1" message (a non-positive int
        # never reaches the later "must be positive" guard).
        ({"train.lora_rank": 0}, "lora_rank must be >= 1"),
        ({"train.lora_alpha": 0}, "lora_alpha must be >= 1"),
        ({"train.lora_alpha": -8}, "lora_alpha must be >= 1"),
        # bools must be rejected (bool is an int subclass: True would coerce to 1).
        ({"train.lora_rank": True}, "lora_rank must be an integer"),
        ({"train.lora_alpha": False}, "lora_alpha must be an integer"),
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
    with pytest.raises(ConfigError, match="epochs must be >= 1"):
        spec_from_dict(raw)


def test_missing_model_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must set `model`"):
        spec_from_dict({"algorithm": "sft"})


def test_missing_hf_repo_is_rejected() -> None:
    # [train] hf_repo is now REQUIRED (no operator HF_REPO default); a config without it fails.
    raw = _raw()
    raw["train"] = {"steps": 10, "lora_rank": 8, "seeds": [0]}
    with pytest.raises(ConfigError, match=r"train\.hf_repo is required"):
        spec_from_dict(raw)


def test_environment_path_is_rejected() -> None:
    # Local environment paths are gone; a `path` (alone or alongside `id`) must fail loudly.
    raw = _raw()
    raw["environment"] = {"path": "./environment.py"}
    with pytest.raises(ConfigError, match="local environment paths are no longer supported"):
        spec_from_dict(raw)
    raw["environment"] = {"id": "gsm8k", "path": "./environment.py"}
    with pytest.raises(ConfigError, match="local environment paths are no longer supported"):
        spec_from_dict(raw)


def test_bare_environment_id_is_rejected() -> None:
    # A bare id like "gsm8k" passes the presence check but the worker would run
    # `prime env install gsm8k` (invalid — Prime needs owner/name); reject it up front.
    for bad in ("gsm8k", "owner/", "/name", "a/b/c"):
        raw = _raw()
        raw["environment"] = {"id": bad}
        with pytest.raises(ConfigError, match=r"owner/name"):
            spec_from_dict(raw)


def test_bare_eval_env_id_is_rejected() -> None:
    # The eval env ([environment.params] eval_env_id) is also prime-installed on the worker, so a
    # bare eval id must be rejected up front (not fail after a GPU is provisioned).
    raw = _raw()
    raw["environment"] = {"id": "owner/train", "params": {"eval_env_id": "gsm8k"}}
    with pytest.raises(ConfigError, match=r"eval_env_id must be a published Prime Hub slug"):
        spec_from_dict(raw)
    # A full owner/name eval slug is accepted.
    raw = _raw()
    raw["environment"] = {"id": "owner/train", "params": {"eval_env_id": "owner/eval"}}
    spec_from_dict(raw)  # no raise


def test_environment_must_be_a_table() -> None:
    raw = _raw()
    raw["environment"] = "gsm8k"
    with pytest.raises(ConfigError, match=r"\[environment\] must be a table"):
        spec_from_dict(raw)


def test_jobspec_from_dict_rejects_path() -> None:
    # Defense-in-depth: a stale worker payload carrying a local path must be rejected.
    data = {
        "model": "Qwen/Qwen3-0.6B",
        "environment": {"id": "gsm8k", "path": "./environment.py"},
    }
    with pytest.raises(ValueError, match="local environment paths are no longer supported"):
        JobSpec.from_dict(data)


# ---------------------------------------------------------------------------
# JobSpec serialization round-trips (what travels client -> server -> worker)
# ---------------------------------------------------------------------------


def test_sft_caps_parse_from_toml() -> None:
    # [train].max_steps / max_examples are read by the worker; ensure spec_from_dict actually
    # parses them (they were defined on TrainSpec but silently dropped at parse time).
    spec = spec_from_dict(
        _raw(**{"train.max_steps": 50, "train.max_examples": 200}), run_id="caps-1"
    )
    assert spec.train.max_steps == 50
    assert spec.train.max_examples == 200
    # explicit 0 means "no cap" (not rejected); negatives are rejected.
    spec0 = spec_from_dict(_raw(**{"train.max_steps": 0}), run_id="caps-0")
    assert spec0.train.max_steps == 0
    with pytest.raises(ConfigError, match="max_examples must be >= 0"):
        spec_from_dict(_raw(**{"train.max_examples": -5}))


def test_job_spec_json_round_trip() -> None:
    spec = spec_from_dict(_raw(), run_id="rt-1")
    restored = JobSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.train.seeds == (0,)
    assert restored.phase == "rl"  # grpo's internal phase id


def test_load_job_spec_from_env_json_and_path(tmp_path, monkeypatch) -> None:
    spec = spec_from_dict(_raw(), run_id="env-1")

    monkeypatch.setenv("FLASH_JOB_SPEC_JSON", spec.to_json())
    assert load_job_spec_from_env() == spec

    monkeypatch.delenv("FLASH_JOB_SPEC_JSON")
    path = tmp_path / "spec.json"
    path.write_text(spec.to_json(), encoding="utf-8")
    monkeypatch.setenv("FLASH_JOB_SPEC_PATH", str(path))
    assert load_job_spec_from_env() == spec

    monkeypatch.delenv("FLASH_JOB_SPEC_PATH")
    assert load_job_spec_from_env() is None


# ---------------------------------------------------------------------------
# runner: run-id containment + dry-run/list/cancel surface
# ---------------------------------------------------------------------------


def _fresh_orchestrator(tmp_path, monkeypatch):
    import flash.runner as runner

    importlib.reload(runner)
    # Storage roots are fixed constants now; redirect to tmp for isolation.
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    return runner


def test_runs_file_path_rejects_traversal(tmp_path, monkeypatch) -> None:
    orch = _fresh_orchestrator(tmp_path, monkeypatch)
    for bad in ("../escape", "a/b", "", "x" * 200, ".hidden"):
        with pytest.raises(ValueError, match="invalid run_id"):
            orch.runs_file_path(bad, ".json")
    good = orch.runs_file_path("flash-123-abc", ".log")
    assert good.endswith("flash-123-abc.log")


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
        orch.get_status("flash-000-nope")


def test_artifacts_dir_and_adapter_prefix_helpers(tmp_path, monkeypatch) -> None:
    orch = _fresh_orchestrator(tmp_path, monkeypatch)
    spec = spec_from_dict(_raw(), run_id="flash-1-x")
    assert orch.artifacts_dir(spec).endswith(os.path.join("results", "runpod", "rl", "flash-1-x"))
    assert orch.adapter_prefix(spec) == "rl/flash-1-x/seed0"
    assert orch.adapter_prefix(spec, seed=3) == "rl/flash-1-x/seed3"


# ---------------------------------------------------------------------------
# engine.vram: fit estimates + offline param lookup
# ---------------------------------------------------------------------------


def test_vram_estimate_scales_with_params_and_algorithm() -> None:
    from flash.engine import vram

    sft_small = vram.estimate_vram_gb(0.6, "sft")
    sft_big = vram.estimate_vram_gb(8.0, "sft")
    grpo_big = vram.estimate_vram_gb(8.0, "grpo")
    assert sft_small < sft_big < grpo_big  # GRPO colocates vLLM on top of the trainer


def test_vram_sft_per_device_bs_is_managed_default(monkeypatch) -> None:
    # SFT micro-batch is a MANAGED default: build_worker_env no longer forwards SFT_PER_DEVICE_BS,
    # so the worker always runs the fixed default and the allocator must size against that SAME
    # fixed value. A control-plane process-env SFT_PER_DEVICE_BS must NOT move the estimate — sizing
    # a card for a micro-batch the worker never uses would under-route an SFT_PER_DEVICE_BS=1 env to
    # a too-small GPU that then OOMs at the default micro-batch 4.
    from flash.engine import vram

    monkeypatch.delenv("SFT_PER_DEVICE_BS", raising=False)
    base = vram.estimate_vram_gb(8.0, "sft", seq_len=4096, batch_size=32)
    monkeypatch.setenv("SFT_PER_DEVICE_BS", "8")
    assert vram.estimate_vram_gb(8.0, "sft", seq_len=4096, batch_size=32) == base  # env ignored
    monkeypatch.setenv("SFT_PER_DEVICE_BS", "1")
    assert vram.estimate_vram_gb(8.0, "sft", seq_len=4096, batch_size=32) == base  # env ignored
    monkeypatch.setenv("SFT_PER_DEVICE_BS", "not-an-int")
    assert vram.estimate_vram_gb(8.0, "sft", seq_len=4096, batch_size=32) == base  # env ignored


def test_fetch_hf_params_is_offline_safe(monkeypatch) -> None:
    from flash.engine import vram

    monkeypatch.setenv("FLASH_SKIP_NET", "1")
    assert vram.fetch_hf_params_b("any/model") is None


# ---------------------------------------------------------------------------
# _logging: namespace + level resolution
# ---------------------------------------------------------------------------


def test_get_logger_namespacing() -> None:
    from flash._logging import get_logger

    assert get_logger().name == "flash"
    assert get_logger("flash").name == "flash"
    assert get_logger("flash.providers").name == "flash.providers"
    assert get_logger("mymodule").name == "flash.mymodule"


def test_log_level_from_env(monkeypatch) -> None:
    from flash import _logging

    monkeypatch.setenv("FLASH_LOG_LEVEL", "debug")
    assert _logging._level_from_env() == logging.DEBUG
    monkeypatch.setenv("FLASH_LOG_LEVEL", "15")
    assert _logging._level_from_env() == 15
    monkeypatch.delenv("FLASH_LOG_LEVEL")
    assert _logging._level_from_env(logging.WARNING) == logging.WARNING
