"""Spec plumbing not covered elsewhere: config validation error paths, JobSpec
serialization round-trips, worker env loading, run-id path containment, VRAM
estimates, and logging namespace helpers."""

from __future__ import annotations

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
        # NOTE: model_policy is no longer a user knob (it's read from the FLASH_MODEL_POLICY env on
        # the control plane), so a bad user-supplied model_policy is ignored, not rejected here.
        # Unknown config sections/keys are rejected (not silently dropped → 16x-cost defaults).
        # The classic footgun: rollout knobs under a [grpo] table instead of [train].
        ({"grpo.group_size": 4}, "unknown config section"),
        ({"sft.epochs": 3}, r"under \[train\]"),
        ({"train.max_token": 256}, "unknown key"),  # typo of max_tokens
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


def test_hf_repo_is_managed_not_user_set() -> None:
    # [train] hf_repo is platform-managed (assigned server-side per run), so it is NEITHER
    # required NOR honored from a user config: a config without it parses fine, and a user-
    # supplied value is ignored (left blank for the control plane to assign at submit).
    raw = _raw()
    raw["train"] = {"steps": 10, "lora_rank": 8, "seeds": [0]}
    assert spec_from_dict(raw).train.hf_repo == ""
    raw["train"]["hf_repo"] = "someone-else/their-repo"
    assert spec_from_dict(raw).train.hf_repo == ""


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


def test_environment_must_be_a_table() -> None:
    raw = _raw()
    raw["environment"] = "gsm8k"
    with pytest.raises(ConfigError, match=r"\[environment\] must be a table"):
        spec_from_dict(raw)


@pytest.mark.parametrize("section", ["gpu", "environment", "train"])
def test_falsy_non_table_section_is_rejected_not_coerced(section: str) -> None:
    # A present-but-falsy non-dict (e.g. `gpu = false`) must hit the "must be a table" check,
    # not be silently coerced to {} by `or {}` (which would bypass validation). A MISSING
    # section still defaults to an empty table (covered by the happy-path tests).
    raw = _raw()
    raw[section] = False
    with pytest.raises(ConfigError, match=rf"\[{section}\] must be a table"):
        spec_from_dict(raw)


def test_environment_subfields_reject_wrong_types() -> None:
    # The [environment] sub-fields are consumed by EnvironmentSpec(...) via dict(... or {}) /
    # tuple(... or ()): a present-but-wrong-typed value would otherwise crash opaquely
    # (dict("x") / dict(1)) or silently misbehave (pip = "x" char-split into ('x',)). Each
    # must fail fast with a clear ConfigError instead.
    # A falsy non-table (params = false) is rejected too, mirroring the section-level rule that
    # `environment = false` must fail rather than silently coerce to {} and bypass intent.
    for bad in ("notatable", 123, False):
        raw = _raw()
        raw["environment"] = {"id": "primeintellect/gsm8k", "params": bad}
        with pytest.raises(ConfigError, match=r"\[environment\] params must be a table"):
            spec_from_dict(raw)
    for bad in ("notalist", 123, False):
        raw = _raw()
        raw["environment"] = {"id": "primeintellect/gsm8k", "pip": bad}
        with pytest.raises(ConfigError, match=r"\[environment\] pip must be a list of strings"):
            spec_from_dict(raw)
    raw = _raw()
    raw["environment"] = {"id": "primeintellect/gsm8k", "pip": ["ok", 123]}
    with pytest.raises(ConfigError, match=r"\[environment\] pip entries must be strings"):
        spec_from_dict(raw)


def test_environment_subfields_accept_valid_and_missing() -> None:
    # Missing sub-fields keep their defaults, and valid values pass through unchanged.
    raw = _raw()
    raw["environment"] = {"id": "primeintellect/gsm8k"}
    spec = spec_from_dict(raw)
    assert spec.environment.params == {}
    assert spec.environment.pip == ()
    raw = _raw()
    raw["environment"] = {
        "id": "primeintellect/gsm8k",
        "params": {"k": "v"},
        "pip": ["pkg==1.0"],
    }
    spec = spec_from_dict(raw)
    assert spec.environment.params == {"k": "v"}
    assert spec.environment.pip == ("pkg==1.0",)
    # An explicit None (e.g. JSON `null`) is treated as missing -> default, NOT rejected.
    raw = _raw()
    raw["environment"] = {"id": "primeintellect/gsm8k", "params": None, "pip": None}
    spec = spec_from_dict(raw)
    assert spec.environment.params == {}
    assert spec.environment.pip == ()


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


def test_gpu_provider_pin_parses_and_round_trips() -> None:
    # The opt-in [gpu] provider pin is a real user knob: it parses, normalizes (case/whitespace),
    # and round-trips through the worker JSON. Unset -> None (cross-provider cheapest-wins default).
    assert spec_from_dict(_raw()).gpu.provider is None
    for raw_val, want in (("vast", "vast"), ("runpod", "runpod"), (" VAST ", "vast")):
        spec = spec_from_dict(_raw(**{"gpu.provider": raw_val}), run_id="pin-1")
        assert spec.gpu.provider == want
        # survives the spec.to_json() -> worker re-parse round trip (asdict carries the field)
        assert JobSpec.from_json(spec.to_json()).gpu.provider == want


@pytest.mark.parametrize("bad", ["aws", "lambda", "", "RUNPOD;rm", 1])
def test_gpu_provider_pin_rejects_unknown_value(bad) -> None:
    # A typo / non-{runpod,vast} value must fail at parse time, not as an opaque submit-time error.
    # Empty string normalizes to None (unset), so it is NOT rejected — assert that separately.
    if bad == "":
        assert spec_from_dict(_raw(**{"gpu.provider": ""})).gpu.provider is None
        return
    match = "must be a string" if isinstance(bad, int) else "provider must be one of"
    with pytest.raises(ConfigError, match=match):
        spec_from_dict(_raw(**{"gpu.provider": bad}))


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
    from tests._helpers.runner import fresh_runner

    return fresh_runner(tmp_path, monkeypatch)


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
    # so the worker always runs the fixed default (4) and the allocator must size against that SAME
    # fixed value. A control-plane process-env SFT_PER_DEVICE_BS must NOT move the estimate — sizing
    # a card for a micro-batch the worker never uses would under-route an SFT_PER_DEVICE_BS=1 env to
    # a too-small GPU that then OOMs at the default micro-batch 4.
    from flash.engine import vram

    at_cap = vram.estimate_vram_gb(8.0, "sft", seq_len=4096, batch_size=4)
    above_cap = vram.estimate_vram_gb(8.0, "sft", seq_len=4096, batch_size=32)
    assert above_cap == at_cap  # batch_size above the per-device 4 is capped, not sized up
    below_cap = vram.estimate_vram_gb(8.0, "sft", seq_len=4096, batch_size=1)
    assert below_cap < at_cap  # micro-batch 1 reserves less activation VRAM
    # the removed env no longer changes the estimate (fully managed), whatever its value
    for val in ("8", "1", "not-an-int"):
        monkeypatch.setenv("SFT_PER_DEVICE_BS", val)
        assert vram.estimate_vram_gb(8.0, "sft", seq_len=4096, batch_size=32) == at_cap


def test_fetch_hf_params_is_offline_safe(monkeypatch) -> None:
    from flash.engine import vram

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


def test_configure_logging_verbosity() -> None:
    from flash import _logging

    _logging.configure_logging(verbosity=0)
    assert logging.getLogger("flash").level == logging.WARNING
    _logging.configure_logging(verbosity=1)
    assert logging.getLogger("flash").level == logging.INFO
    _logging.configure_logging(verbosity=2)
    assert logging.getLogger("flash").level == logging.DEBUG


# ---------------------------------------------------------------------------
# [worker_env] secret-key policy — [worker_env] is serialized into job_spec_json
# (persisted + logged), so secret-bearing keys must be rejected at parse time and set
# as real env vars instead. These cases pin the _is_secret_key heuristic so it doesn't
# drift into false positives (legit knobs) or false negatives (real secrets).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "HF_TOKEN",  # secret WORD: TOKEN
        "OPENAI_API_KEY",  # KEY qualified by API
        "AWS_SECRET_ACCESS_KEY",  # SECRET word + KEY qualified by SECRET/ACCESS
        "DB_PASSWORD",  # PASSWORD word
        "PRIME_API_KEY",
        "WANDB_API_KEY",
        "SOME_PRIVATE_KEY",  # KEY qualified by PRIVATE
        "MY_CREDENTIAL",
        "AUTH_KEY",  # KEY qualified by AUTH
        "SSH_KEY",  # KEY qualified by SSH
        "DEPLOY_KEY",  # KEY qualified by DEPLOY
        "GITHUB_PAT",  # PAT word (personal access token)
    ],
)
def test_worker_env_rejects_secret_keys(key: str) -> None:
    with pytest.raises(ConfigError, match="must not contain secret-bearing keys"):
        spec_from_dict(_raw(worker_env={key: "x"}))


@pytest.mark.parametrize(
    "key",
    [
        "RL_VLLM_GPU_UTIL",  # plain knob
        "SFT_PACKING",
        "RL_VLLM_MAX_BATCHED_TOKENS",  # word TOKENS, not the secret word TOKEN
        "SORT_KEY",  # bare KEY without a secret qualifier
        "WANDB_ENTITY",  # account routing, not a secret
        "FLASH_MLP_KERNEL",
        "VLLM_ATTENTION_BACKEND",
    ],
)
def test_worker_env_allows_non_secret_keys(key: str) -> None:
    spec = spec_from_dict(_raw(worker_env={key: "v"}))
    assert spec.worker_env[key] == "v"


@pytest.mark.parametrize("name", ["BAD=KEY", "", "BAD KEY", "X\tY"])
def test_worker_env_rejects_invalid_env_names(name: str) -> None:
    # Names subprocess.Popen(env=...) would reject on the worker (empty / '=' / whitespace) must
    # fail at parse time, not after a worker has been provisioned.
    with pytest.raises(ConfigError, match="invalid environment variable name"):
        spec_from_dict(_raw(worker_env={name: "v"}))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # Falsey string forms (the bug a plain bool() has: any non-empty string is True).
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("none", False),
        ("", False),
        ("  false  ", False),
        # Truthy string forms.
        ("true", True),
        ("1", True),
        ("yes", True),
        ("anything-else", True),
        # Already-typed values pass through bool().
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        (None, False),
    ],
)
def test_coerce_bool(value, expected) -> None:
    from flash.spec import coerce_bool

    assert coerce_bool(value) is expected
