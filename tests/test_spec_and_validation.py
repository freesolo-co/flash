"""Spec plumbing not covered elsewhere: config validation error paths, JobSpec
serialization round-trips, worker env loading, run-id path containment, VRAM
estimates, and logging namespace helpers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import fields

import pytest

from flash.schema import (
    TRAIN_KEY_MIN_VERSIONS,
    TRAIN_SCHEMA_KEYS,
    ConfigError,
    parse_adapter_revision,
    spec_from_dict,
    train_schema_metadata,
    validate_train_keys,
)
from flash.spec import GpuSpec, JobSpec, TrainSpec, load_job_spec_from_env

BASE_RAW = {
    "model": "Qwen/Qwen3.5-0.8B",
    "algorithm": "grpo",
    "environment": {"id": "freesolo/gsm8k"},
    "train": {"epochs": 1, "max_examples": 10, "lora_rank": 8, "hf_repo": "owner/runs"},
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


@pytest.mark.parametrize("step", ["00", "01"])
def test_parse_adapter_revision_rejects_zero_padded_steps(step):
    revision = f"run-a@step-{step}." + "a" * 40

    assert parse_adapter_revision(revision) is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        # `seeds` is no longer a valid [train] key (multi-seed removed); it's now rejected
        # as an unknown key rather than seed-validated.
        ({"train.seeds": [0]}, "unknown key"),
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
        # An unhashable model (TOML array / `[model]` table) used to TypeError on MODELS.get() -> 500;
        # it must be a clean ConfigError like every other scalar.
        ({"model": ["Qwen/Qwen3.5-4B"]}, "must be a model id string"),
        ({"model": {"id": "x"}}, "must be a model id string"),
        ({"model": "   "}, "must be a model id string"),
        # A truthy non-string algorithm used to AttributeError on .lower() (uncaught 500); it must be
        # a clean ConfigError (400) like seeds. Falsy values still default to "sft" (tested below).
        ({"algorithm": 5}, "algorithm must be a string"),
        ({"algorithm": ["grpo"]}, "algorithm must be a string"),
        ({"algorithm": True}, "algorithm must be a string"),
        # NOTE: model_policy is no longer a user knob (it's read from the FLASH_MODEL_POLICY env on
        # the control plane), so a bad user-supplied model_policy is ignored, not rejected here.
        # Unknown config sections/keys are rejected (not silently dropped → 16x-cost defaults).
        # The classic footgun: rollout knobs under a [grpo] table instead of [train].
        ({"grpo.group_size": 4}, "unknown config section"),
        ({"sft.epochs": 3}, r"under \[train\]"),
        ({"train.max_token": 256}, "unknown key"),  # typo of max_completion_tokens
        ({"train.max_length": 256}, "unknown key"),
        ({"train.max_tokens": 256}, "unknown key"),
        ({"train.rollout_request_timeout_seconds": 600}, "unknown key"),
        ({"train.rollout_request_max_attempts": 2}, "unknown key"),
        ({"train.rollout_stall_timeout_seconds": 60}, "unknown key"),
    ],
)
def test_spec_validation_rejections(overrides, match) -> None:
    with pytest.raises(ConfigError, match=match):
        spec_from_dict(_raw(**overrides))


def test_train_key_registry_is_derived_from_trainspec_metadata() -> None:
    train_fields = [item for item in fields(TrainSpec) if item.metadata.get("introduced_in")]

    assert "init_from_adapter_revision" not in TRAIN_SCHEMA_KEYS
    assert frozenset(item.name for item in train_fields) == TRAIN_SCHEMA_KEYS
    assert {
        item.name: item.metadata["introduced_in"] for item in train_fields
    } == TRAIN_KEY_MIN_VERSIONS
    assert train_schema_metadata() == {
        key: TRAIN_KEY_MIN_VERSIONS[key] for key in sorted(TRAIN_KEY_MIN_VERSIONS)
    }
    assert TRAIN_KEY_MIN_VERSIONS["hf_repo"] == "0.2.0"
    assert TRAIN_KEY_MIN_VERSIONS["max_context_tokens"] == "0.2.49"
    assert TRAIN_KEY_MIN_VERSIONS["max_completion_tokens"] == "0.2.49"
    assert TRAIN_KEY_MIN_VERSIONS["teacher_model"] == "0.2.56"
    assert TRAIN_KEY_MIN_VERSIONS["structured_outputs"] == "0.2.56"
    # opd has no auxiliary eos loss or user-facing eos-loss key.
    assert "opd_eos_loss_coef" not in TRAIN_KEY_MIN_VERSIONS
    assert {
        value
        for key, value in TRAIN_KEY_MIN_VERSIONS.items()
        if key
        not in {
            "max_context_tokens",
            "max_completion_tokens",
            "teacher_model",
            "structured_outputs",
        }
    } == {"0.2.0"}


def test_train_key_validator_rejects_unknown_names_only() -> None:
    validate_train_keys(TRAIN_SCHEMA_KEYS)
    with pytest.raises(ConfigError) as excinfo:
        validate_train_keys({"epochs", "removed_spelling"})

    message = str(excinfo.value)
    assert "unknown key(s): removed_spelling" in message
    assert "allowed:" in message


def test_historical_train_schema_shapes_are_immutable_source_snapshots() -> None:
    established = frozenset(
        {
            "advantage_clip",
            "batch_size",
            "epochs",
            "group_size",
            "hf_repo",
            "init_from_adapter",
            "kl_penalty_coef",
            "learning_rate",
            "lora_alpha",
            "lora_rank",
            "max_completion_tokens",
            "max_context_tokens",
            "max_examples",
            "max_steps",
            "opd_eos_loss_coef",
            "save_every",
            "stop_sequences",
            "temperature",
            "thinking_length_penalty_coef",
        }
    )
    historical_shapes = {
        "20c4452c": established,
        "699a8aab": established | {"structured_outputs"},
        "861571e7": established | {"structured_outputs", "teacher_model"},
    }
    baseline = {"epochs", "hf_repo", "max_examples"}

    # the historical snapshots are immutable and still carry opd_eos_loss_coef because those commits
    # did. opd has no auxiliary eos loss or user-facing eos-loss key, so current TRAIN_SCHEMA_KEYS
    # equals the latest historical shape minus that one key.
    assert historical_shapes["861571e7"] - {"opd_eos_loss_coef"} == TRAIN_SCHEMA_KEYS
    assert "opd_eos_loss_coef" not in TRAIN_SCHEMA_KEYS
    assert all(baseline <= shape for shape in historical_shapes.values())
    for key in ("structured_outputs", "teacher_model"):
        rejected_by = {commit for commit, shape in historical_shapes.items() if key not in shape}
        assert rejected_by == {
            "20c4452c",
            *({"699a8aab"} if key == "teacher_model" else set()),
        }


def test_sft_init_from_adapter_is_rejected_at_parse_time() -> None:
    with pytest.raises(ConfigError, match="SFT adapter continuation is not supported"):
        spec_from_dict(_raw(algorithm="sft", **{"train.init_from_adapter": "source-run"}))


@pytest.mark.parametrize(
    "lora_rank",
    [
        pytest.param(256, id="non-default"),
        pytest.param(32, id="default"),
        pytest.param(8, id="matching"),
        pytest.param(None, id="null"),
        pytest.param(0, id="invalid"),
    ],
)
def test_warmstart_rejects_explicit_child_rank(lora_rank) -> None:
    with pytest.raises(
        ConfigError,
        match=(
            r"train\.lora_rank cannot be set with train\.init_from_adapter because source adapter "
            r"rank metadata is authoritative"
        ),
    ):
        spec_from_dict(
            _raw(**{"train.init_from_adapter": "source-run", "train.lora_rank": lora_rank})
        )


def test_warmstart_accepts_omitted_child_rank_with_internal_placeholder() -> None:
    raw = _raw(**{"train.init_from_adapter": "source-run"})
    raw["train"].pop("lora_rank")

    spec = spec_from_dict(raw)

    assert spec.train.init_from_adapter == "source-run"
    assert spec.train.lora_rank == 32


def test_public_warmstart_serialization_omits_resolved_internal_fields() -> None:
    spec = JobSpec.from_dict(
        {
            **BASE_RAW,
            "train": {
                **BASE_RAW["train"],
                "init_from_adapter": "owner/runs:rl/source-run",
                "init_from_adapter_revision": "a" * 40,
                "lora_rank": 64,
                "lora_alpha": 128,
            },
        }
    )

    public = spec.to_dict()
    internal = spec.to_internal_dict()

    assert "init_from_adapter_revision" not in public["train"]
    assert "lora_rank" not in public["train"]
    assert internal["train"]["init_from_adapter_revision"] == "a" * 40
    assert internal["train"]["lora_rank"] == 64
    assert JobSpec.from_dict(internal) == spec


def test_public_warmstart_status_spec_round_trips_through_schema() -> None:
    raw = _raw(**{"train.init_from_adapter": "source-run"})
    raw["train"].pop("lora_rank")
    public = spec_from_dict(raw).to_dict()

    restored = spec_from_dict(public)

    assert restored.train.init_from_adapter == "source-run"
    assert restored.train.lora_rank == 32


@pytest.mark.parametrize("init_from_adapter", [None, "", "   "])
def test_blank_or_null_init_adapter_preserves_explicit_rank(init_from_adapter) -> None:
    spec = spec_from_dict(
        _raw(**{"train.init_from_adapter": init_from_adapter, "train.lora_rank": 8})
    )

    assert spec.train.init_from_adapter == ""
    assert spec.train.lora_rank == 8


def test_falsy_algorithm_defaults_to_sft() -> None:
    # The type guard must preserve the `(value or "sft")` semantics: falsy values default.
    # Supply SFT-valid [train] fields so validation reaches the algorithm assertion instead of
    # rejecting the now-default SFT run for a missing epochs/max_examples.
    for falsy in (None, "", 0):
        raw = _raw(algorithm=falsy, **{"train.epochs": 1, "train.max_examples": 8})
        assert spec_from_dict(raw).algorithm == "sft"


def test_sft_epochs_must_be_positive() -> None:
    raw = _raw(algorithm="sft")
    raw["train"] = {"epochs": 0, "max_examples": 8, "lora_rank": 8}
    with pytest.raises(ConfigError, match="epochs must be >= 1"):
        spec_from_dict(raw)


def test_sft_requires_positive_max_examples() -> None:
    raw = _raw(algorithm="sft")
    raw["train"] = {"epochs": 1, "lora_rank": 8}
    with pytest.raises(ConfigError, match=r"max_examples.*positive"):
        spec_from_dict(raw)
    raw["train"]["max_examples"] = 0
    with pytest.raises(ConfigError, match=r"max_examples.*positive"):
        spec_from_dict(raw)
    raw["train"]["max_examples"] = 8
    assert spec_from_dict(raw).train.max_examples == 8


def test_missing_model_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must set `model`"):
        spec_from_dict({"algorithm": "sft"})


def test_hf_repo_is_managed_not_user_set() -> None:
    # [train] hf_repo is platform-managed (assigned server-side per run), so it is NEITHER
    # required NOR honored from a user config: a config without it parses fine, and a user-
    # supplied value is ignored (left blank for the control plane to assign at submit).
    raw = _raw()
    raw["train"] = {"epochs": 1, "max_examples": 10, "lora_rank": 8}
    assert spec_from_dict(raw).train.hf_repo == ""
    raw["train"]["hf_repo"] = "someone-else/their-repo"
    assert spec_from_dict(raw).train.hf_repo == ""


def test_lora_rank_allows_rank128_for_small_serving_models() -> None:
    # The small dense tiers (default model Qwen3.5-0.8B) now serve rank-128 LoRA buffers.
    assert spec_from_dict(_raw(**{"train.lora_rank": 128})).train.lora_rank == 128


def test_lora_rank_must_fit_small_serving_cap() -> None:
    # Small-tier serving cap doubled 64 -> 128; rank 129 exceeds it.
    with pytest.raises(ConfigError, match="serving max_lora_rank=128"):
        spec_from_dict(_raw(**{"train.lora_rank": 129}))


def test_lora_rank_must_fit_large_serving_cap() -> None:
    # The 4B tier serving cap doubled 32 -> 64; rank 65 exceeds it.
    with pytest.raises(ConfigError, match="serving max_lora_rank=64"):
        spec_from_dict(_raw(model="Qwen/Qwen3.5-4B", **{"train.lora_rank": 65}))


def test_bare_environment_id_is_rejected() -> None:
    # A bare id like "gsm8k" passes the presence check but is not a Freesolo env slug;
    # reject it up front.
    for bad in (
        "gsm8k",
        "owner/",
        "/name",
        "a/b/c",
        "owner/..",
        "owner/.",
        "owner/na me",
        "owner/name:tag",
        "https://freesolo.co/owner/name",
        "github:owner/repo/extra@main:x/environment.py",
        "github:owner/repo@:x/environment.py",
        "github:owner/repo@main:../x.py",
        "github:owner/repo@main:/etc/passwd",
        "github:owner /repo@main:x/environment.py",
        "github:owner/repo@bad/ref:x/environment.py",
        "https://github.com/owner/repo/blob/main/../x.py",
        "https://github.com/owner/repo/blob/main:/etc/passwd",
        "https://github.com/owner/repo/blob/bad ref/x.py",
        "https://github.com/owner/repo/issues/1",
    ):
        raw = _raw()
        raw["environment"] = {"id": bad}
        with pytest.raises(ConfigError, match=r"Freesolo environment id"):
            spec_from_dict(raw)


def test_env_ref_validator_matches_adapter_acceptor() -> None:
    # The submit-time schema validator and the worker's environment acceptor parse ONE grammar;
    # they must agree exactly (accept <-> no raise, reject <-> raise) or a ref accepted at submit
    # could fail on the worker (or vice-versa). _require_environment_ref now delegates to the
    # adapter's is_freesolo_environment_id; this pins that alignment across the grammar's corners.
    from flash.envs.adapter import is_freesolo_environment_id
    from flash.schema.fields import _require_environment_ref

    corpus = [
        "ns/name",  # plain managed slug
        "github:owner/repo",
        "github:owner/repo@ref",
        "https://github.com/owner/repo/blob/dev/envs/e/environment.py",  # blob URL
        "  github:owner/repo@dev:envs/e/environment.py  ",  # whitespace-padded ref
        "  ns/name  ",  # whitespace-padded slug
        "https://github.com/owner/repo/blob/main/envs%2Fe/environment.py",  # %2F-encoded blob URL
        "github:owner/repo@main:../../etc/passwd",  # traversal attempt
        "https://github.com/owner/repo/blob/main/../x.py",  # traversal attempt
        "../../etc/passwd",
        "",
    ]

    def schema_accepts(value: str) -> bool:
        try:
            _require_environment_ref(value, "msg")
            return True
        except ConfigError:
            return False

    for value in corpus:
        assert schema_accepts(value) is is_freesolo_environment_id(value), value


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


def test_gpu_retry_and_wall_defaults_and_authored_values() -> None:
    defaults = GpuSpec()
    missing = _raw()
    missing["gpu"] = {}
    explicit_none = _raw(**{"gpu.max_retries": None, "gpu.max_wall_seconds": None})
    for raw in (missing, explicit_none):
        spec = spec_from_dict(raw)
        assert spec.gpu.max_retries == defaults.max_retries
        assert spec.gpu.max_wall_seconds == defaults.max_wall_seconds

    authored_raw = _raw(**{"gpu.max_retries": 7.0, "gpu.max_wall_seconds": 1234.0})
    authored_raw["gpu"].update(
        {"type": "not-a-real-gpu", "disk_gb": 999, "future_gpu_field": "ignored"}
    )
    authored = spec_from_dict(authored_raw)
    assert authored.gpu.max_retries == 7
    assert authored.gpu.max_wall_seconds == 1234
    assert authored.gpu.type == spec_from_dict(_raw()).gpu.type
    assert authored.gpu.disk_gb == defaults.disk_gb
    assert spec_from_dict(_raw(**{"gpu.max_retries": 0})).gpu.max_retries == 0


@pytest.mark.parametrize("key", ["max_retries", "max_wall_seconds"])
@pytest.mark.parametrize(
    ("value", "match"),
    [
        (True, "must be an integer"),
        ("5", "must be an integer"),
        (1.5, "must be a finite integer"),
        (float("inf"), "must be a finite integer"),
        (float("nan"), "must be a finite integer"),
    ],
)
def test_gpu_integer_fields_reject_invalid_values(key: str, value, match: str) -> None:
    with pytest.raises(ConfigError, match=rf"gpu\.{key} {match}"):
        spec_from_dict(_raw(**{f"gpu.{key}": value}))


def test_gpu_retry_and_wall_minimums() -> None:
    with pytest.raises(ConfigError, match=r"gpu\.max_retries must be >= 0"):
        spec_from_dict(_raw(**{"gpu.max_retries": -1}))
    for value in (59, 1, 0, -1, -3600):
        with pytest.raises(ConfigError, match=r"gpu\.max_wall_seconds must be >= 60"):
            spec_from_dict(_raw(**{"gpu.max_wall_seconds": value}))
    assert spec_from_dict(_raw(**{"gpu.max_wall_seconds": 60})).gpu.max_wall_seconds == 60


def test_environment_subfields_reject_wrong_types() -> None:
    # The [environment] sub-fields are consumed by EnvironmentSpec(...) via dict(... or {}) /
    # tuple(... or ()): a present-but-wrong-typed value would otherwise crash opaquely
    # (dict("x") / dict(1)) or silently misbehave (pip = "x" char-split into ('x',)). Each
    # must fail fast with a clear ConfigError instead.
    # A falsy non-table (params = false) is rejected too, mirroring the section-level rule that
    # `environment = false` must fail rather than silently coerce to {} and bypass intent.
    for bad in ("notatable", 123, False):
        raw = _raw()
        raw["environment"] = {
            "id": "github:freesolo-co/envs@main:gsm8k/environment.py",
            "params": bad,
        }
        with pytest.raises(ConfigError, match=r"\[environment\] params must be a table"):
            spec_from_dict(raw)
    for bad in ("notalist", 123, False):
        raw = _raw()
        raw["environment"] = {
            "id": "github:freesolo-co/envs@main:gsm8k/environment.py",
            "pip": bad,
        }
        with pytest.raises(ConfigError, match=r"\[environment\] pip must be a list of strings"):
            spec_from_dict(raw)
    raw = _raw()
    raw["environment"] = {
        "id": "github:freesolo-co/envs@main:gsm8k/environment.py",
        "pip": ["ok", 123],
    }
    with pytest.raises(ConfigError, match=r"\[environment\] pip entries must be strings"):
        spec_from_dict(raw)
    for bad in ("notalist", 123, False):
        raw = _raw()
        raw["environment"] = {
            "id": "github:freesolo-co/envs@main:gsm8k/environment.py",
            "secrets": bad,
        }
        with pytest.raises(ConfigError, match=r"\[environment\] secrets must be a list"):
            spec_from_dict(raw)
    raw = _raw()
    raw["environment"] = {
        "id": "github:freesolo-co/envs@main:gsm8k/environment.py",
        "secrets": ["OK_SECRET", 123],
    }
    with pytest.raises(ConfigError, match=r"\[environment\] secrets entries must be strings"):
        spec_from_dict(raw)
    raw = _raw()
    raw["environment"] = {
        "id": "github:freesolo-co/envs@main:gsm8k/environment.py",
        "secrets": ["BAD KEY"],
    }
    with pytest.raises(ConfigError, match=r"\[environment\] secrets has invalid"):
        spec_from_dict(raw)
    raw = _raw()
    raw["environment"] = {
        "id": "github:freesolo-co/envs@main:gsm8k/environment.py",
        "secrets": ["HF_TOKEN"],
    }
    with pytest.raises(ConfigError, match=r"platform-managed"):
        spec_from_dict(raw)


def test_grpo_environment_can_declare_fireworks_key() -> None:
    raw = _raw()
    raw["environment"] = {
        "id": "github:freesolo-co/envs@main:gsm8k/environment.py",
        "secrets": ["FIREWORKS_API_KEY"],
    }

    spec = spec_from_dict(raw)

    assert spec.algorithm == "grpo"
    assert spec.environment.secrets == ("FIREWORKS_API_KEY",)


def test_environment_subfields_accept_valid_and_missing() -> None:
    # Missing sub-fields keep their defaults, and valid values pass through unchanged.
    raw = _raw()
    raw["environment"] = {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"}
    spec = spec_from_dict(raw)
    assert spec.environment.params == {}
    assert spec.environment.pip == ()
    assert spec.environment.secrets == ()
    raw = _raw()
    raw["environment"] = {
        "id": "github:freesolo-co/envs@main:gsm8k/environment.py",
        "params": {"k": "v"},
        "pip": ["pkg==1.0"],
        "secrets": ["SERPAPI_API_KEY", "OPENAI_API_KEY", "SERPAPI_API_KEY"],
    }
    spec = spec_from_dict(raw)
    assert spec.environment.params == {"k": "v"}
    assert spec.environment.pip == ("pkg==1.0",)
    assert spec.environment.secrets == ("SERPAPI_API_KEY", "OPENAI_API_KEY")
    # An explicit None (e.g. JSON `null`) is treated as missing -> default, NOT rejected.
    raw = _raw()
    raw["environment"] = {
        "id": "github:freesolo-co/envs@main:gsm8k/environment.py",
        "params": None,
        "pip": None,
        "secrets": None,
    }
    spec = spec_from_dict(raw)
    assert spec.environment.params == {}
    assert spec.environment.pip == ()
    assert spec.environment.secrets == ()


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
    assert restored.phase == "rl"  # grpo's internal phase id


def test_gpu_public_fields_survive_payload_and_server_reparse() -> None:
    from flash.client.specs import spec_payload

    spec = spec_from_dict(
        _raw(**{"gpu.max_retries": 0, "gpu.max_wall_seconds": 60}), run_id="gpu-rt"
    )
    payload = spec_payload(spec)
    assert payload["gpu"]["max_retries"] == 0
    assert payload["gpu"]["max_wall_seconds"] == 60

    reparsed = spec_from_dict(payload, run_id="server-reparse")
    assert reparsed.gpu.max_retries == 0
    assert reparsed.gpu.max_wall_seconds == 60
    assert reparsed.gpu.type == spec.gpu.type


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


def test_programmatic_sft_submit_requires_max_examples(tmp_path, monkeypatch) -> None:
    from flash.spec import JobSpec

    orch = _fresh_orchestrator(tmp_path, monkeypatch)
    spec = JobSpec(run_id="sft-no-examples", model="Qwen/Qwen3.5-0.8B", algorithm="sft")
    with pytest.raises(ValueError, match=r"max_examples.*positive"):
        orch.submit_job(spec, dry_run=True)


def test_programmatic_sft_submit_rejects_adapter_continuation(tmp_path, monkeypatch) -> None:
    from flash.spec import JobSpec, TrainSpec

    orch = _fresh_orchestrator(tmp_path, monkeypatch)
    spec = JobSpec(
        run_id="sft-warmstart",
        model="Qwen/Qwen3.5-0.8B",
        algorithm="sft",
        train=TrainSpec(epochs=1, max_examples=8, init_from_adapter="source-run"),
    )
    with pytest.raises(ValueError, match="SFT adapter continuation is not supported"):
        orch.submit_job(spec, dry_run=True)


def test_artifacts_dir_and_adapter_prefix_helpers(tmp_path, monkeypatch) -> None:
    orch = _fresh_orchestrator(tmp_path, monkeypatch)
    spec = spec_from_dict(_raw(), run_id="flash-1-x")
    assert orch.artifacts_dir(spec).endswith(os.path.join("results", "runpod", "rl", "flash-1-x"))
    assert orch.adapter_prefix(spec) == "rl/flash-1-x"
    assert orch.adapter_ref(spec) is None

    d = spec.to_dict()
    d["train"] = {**d["train"], "hf_repo": "Freesolo-Co/flashrun-flash-1-x"}
    spec_with_repo = JobSpec.from_dict(d)
    assert orch.adapter_ref(spec_with_repo) == "Freesolo-Co/flashrun-flash-1-x:rl/flash-1-x"


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

    # Use a tiny vocab so this isolates the managed micro-batch cap rather than the dense-logits
    # per-device cap, which can floor both batch sizes to the same per-device value.
    at_cap = vram.estimate_vram_gb(8.0, "sft", seq_len=4096, batch_size=4, vocab=1)
    above_cap = vram.estimate_vram_gb(8.0, "sft", seq_len=4096, batch_size=32, vocab=1)
    assert above_cap == at_cap  # batch_size above the per-device 4 is capped, not sized up
    below_cap = vram.estimate_vram_gb(8.0, "sft", seq_len=4096, batch_size=1, vocab=1)
    assert below_cap < at_cap  # micro-batch 1 reserves less activation VRAM
    # the removed env no longer changes the estimate (fully managed), whatever its value
    for val in ("8", "1", "not-an-int"):
        monkeypatch.setenv("SFT_PER_DEVICE_BS", val)
        assert vram.estimate_vram_gb(8.0, "sft", seq_len=4096, batch_size=32, vocab=1) == at_cap


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
        "GITHUB_TOKEN",
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
