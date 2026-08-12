"""Spec plumbing not covered elsewhere: config validation error paths, JobSpec
serialization round-trips, worker env loading, run-id path containment, VRAM
estimates, and logging namespace helpers."""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager, nullcontext
from dataclasses import fields, replace

import pytest

from flash.core.spec import GpuSpec, JobSpec, TrainSpec, load_job_spec_from_env
from flash.schema import (
    TRAIN_KEY_MIN_VERSIONS,
    TRAIN_SCHEMA_KEYS,
    ConfigError,
    parse_adapter_revision,
    spec_and_train_keys_from_file,
    spec_from_dict,
    train_schema_metadata,
    validate_train_keys,
)

BASE_RAW = {
    "model": "Qwen/Qwen3.5-0.8B",
    "algorithm": "grpo",
    "project": "11111111-1111-4111-8111-111111111111",
    "environment": {"id": "freesolo/gsm8k"},
    "train": {"epochs": 1, "max_examples": 10, "lora_rank": 8},
    "gpu": {},
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


def _job_from_dict(data: dict) -> JobSpec:
    return JobSpec.from_dict({"project": "11111111-1111-4111-8111-111111111111", **data})


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
        # lora_rank and lora_alpha parse via _train_int(minimum=1), so out-of-range values are
        # rejected at parse time with the shared ">= 1" message (a non-positive int never reaches
        # the later "must be positive" guard).
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
        # gpu.count: cards per job, validated to 1..8; bools/non-ints/out-of-range are rejected.
        ({"gpu.count": 0}, "gpu.count must be between 1 and 8"),
        ({"gpu.count": 9}, "gpu.count must be between 1 and 8"),
        ({"gpu.count": True}, "gpu.count must be an integer"),
        ({"gpu.count": "two"}, "gpu.count must be an integer"),
        # `project` is the required canonical freesolo project id and must be a plain string.
        ({"project": ["p"]}, "project must be a string"),
        ({"project": 5}, "project must be a string"),
        ({"project": {"id": "p"}}, "project must be a string"),
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
    # hf_repo is platform-managed (no introduced_in), so it is absent from both the user-facing
    # train schema and the min-version registry.
    assert "hf_repo" not in TRAIN_SCHEMA_KEYS
    assert "hf_repo" not in TRAIN_KEY_MIN_VERSIONS
    assert TRAIN_KEY_MIN_VERSIONS["max_context_tokens"] == "0.2.49"
    assert TRAIN_KEY_MIN_VERSIONS["max_completion_tokens"] == "0.2.49"
    assert TRAIN_KEY_MIN_VERSIONS["teacher_model"] == "0.2.56"
    assert TRAIN_KEY_MIN_VERSIONS["structured_outputs"] == "0.2.56"
    assert TRAIN_KEY_MIN_VERSIONS["save_at_steps"] == "0.2.57"
    assert TRAIN_KEY_MIN_VERSIONS["credit_assignment"] == "1.0.2"
    assert TRAIN_KEY_MIN_VERSIONS["entropy_quantile"] == "1.0.15"
    # re-introduced as a user knob after being managed-and-derived; gated on the release that
    # restored it, not on lora_rank's original 0.2.0.
    assert TRAIN_KEY_MIN_VERSIONS["lora_alpha"] == "1.1.35"
    # the rl half of the optimizer-batch split. batch_size keeps its 0.2.0 gate because sft still
    # takes it; only the new rl name is gated on the release that introduced the split.
    assert TRAIN_KEY_MIN_VERSIONS["prompts_per_step"] == "1.1.43"
    assert TRAIN_KEY_MIN_VERSIONS["batch_size"] == "0.2.0"
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
            "save_at_steps",
            "credit_assignment",
            "entropy_quantile",
            "lora_alpha",
            "prompts_per_step",
        }
    } == {"0.2.0"}


def test_persisted_rollout_batch_survives_the_prompts_per_step_rename() -> None:
    """Regression: a run persisted before 1.1.43 carries the batch only as ``batch_size``.

    Production (main, 1.1.40) authors the rollout batch under ``batch_size`` and the workers read
    it there; dev reads ``prompts_per_step``. Every run in flight across that release therefore
    reparses through `from_dict` with the new key unset, and the worker falls back to the recipe
    default -- 64 for an authored 32 on grpo, which OOMs hardware rented for 32, and 8 on opd.
    `from_dict` is the recovery path (`platform/runtime.py` reattach/resubmit), NOT submission:
    the schema still rejects an authored `batch_size` on a live rollout spec.
    """
    from flash.engine.plan.recipe import RECIPE

    def _train(algorithm: str, train: dict) -> TrainSpec:
        return _job_from_dict(
            {
                "model": "Qwen/Qwen3.5-4B",
                "algorithm": algorithm,
                "environment": {"id": "github:owner/repo@main:env/environment.py"},
                "train": {"epochs": 1, "group_size": 4, **train},
            }
        ).train

    for algorithm, default in (
        ("grpo", RECIPE.rl.prompts_per_step),
        ("opd", RECIPE.opd.prompts_per_step),
    ):
        legacy = _train(algorithm, {"batch_size": 32})
        assert legacy.prompts_per_step == 32, algorithm
        # the value the broken read resumed on. per-algorithm, so the assertion cannot pass by
        # short-circuiting on the other one -- and so it fails loudly rather than matching by
        # coincidence if a recipe default ever becomes 32.
        assert default != 32, algorithm
        # the old key is MOVED, not copied: a spec carrying both names is rejected by the schema,
        # which would break the resubmit that recovery and `flash runs get` perform, and would let
        # `vram.py::_optimizer_batch_value` (which takes the larger of the two) size a card off the
        # stale value that ranking ignores.
        assert legacy.batch_size is None, algorithm

        # every persisted shape, asserted the same way: the migrated value, AND that the old key is
        # gone. checking only prompts_per_step would miss a retained batch_size, which is what
        # breaks the round trip below.
        for stored, expected in (
            # the pre-1.1.43 spelling, migrated.
            ({"batch_size": 32}, 32),
            # the current spelling, untouched.
            ({"prompts_per_step": 16}, 16),
            # a payload written mid-upgrade carrying BOTH: prompts_per_step wins, and the
            # superseded key goes with it rather than being left to re-emit.
            ({"batch_size": 32, "prompts_per_step": 16}, 16),
            ({"batch_size": 0, "prompts_per_step": 16}, 16),
            # a non-positive legacy value is discarded rather than migrated: `minimum=1` would have
            # rejected it at submission, so carrying it forward only fails later on a rented GPU.
            # discarded from BOTH names -- retaining it under the old one re-emits a rejected key.
            ({"batch_size": 0}, None),
            ({"batch_size": -5}, None),
        ):
            train = _train(algorithm, dict(stored))
            assert (train.prompts_per_step, train.batch_size) == (expected, None), (
                algorithm,
                stored,
            )
            # and the result must survive a full re-serialize -> reparse, which is what recovery
            # and `flash runs get` perform. a spec carrying both names raises ConfigError here.
            roundtrip = spec_from_dict(
                _job_from_dict(
                    {
                        "model": "Qwen/Qwen3.5-4B",
                        "algorithm": algorithm,
                        "environment": {"id": "github:owner/repo@main:env/environment.py"},
                        "train": {"epochs": 1, "group_size": 4, **stored},
                    }
                ).to_dict()
            )
            assert (roundtrip.train.prompts_per_step, roundtrip.train.batch_size) == (
                expected,
                None,
            ), (algorithm, stored)

    # sft is NOT migrated: there `batch_size` is a different quantity (examples per update, resolved
    # against a measured workload profile), so copying it into prompts_per_step would be wrong.
    sft = _train("sft", {"batch_size": 4})
    assert (sft.batch_size, sft.prompts_per_step) == (4, None)


def test_credit_assignment_defaults_accepts_and_roundtrips() -> None:
    default = spec_from_dict(_raw())
    assert default.train.credit_assignment == "per_episode"
    assert JobSpec.from_json(default.to_json()).train.credit_assignment == "per_episode"

    empty = spec_from_dict(_raw(**{"train.credit_assignment": "  "}))
    assert empty.train.credit_assignment == "per_episode"

    per_turn = spec_from_dict(_raw(**{"train.credit_assignment": " Per_Turn "}))
    assert per_turn.train.credit_assignment == "per_turn"
    assert per_turn.to_dict()["train"]["credit_assignment"] == "per_turn"
    assert JobSpec.from_json(per_turn.to_json()).train.credit_assignment == "per_turn"


def test_toml_config_requires_project(tmp_path) -> None:
    config = tmp_path / "train.toml"
    config.write_text(
        'model = "Qwen/Qwen3.5-0.8B"\n'
        'algorithm = "sft"\n'
        '[environment]\nid = "freesolo/gsm8k"\n'
        "[train]\nepochs = 1\nmax_examples = 1\n"
    )

    with pytest.raises(ConfigError, match="project is required and must be nonblank"):
        spec_and_train_keys_from_file(str(config), project_required=True)


def test_project_id_is_required_canonicalized_and_roundtrips() -> None:
    missing = _raw()
    missing.pop("project")
    with pytest.raises(ConfigError, match="project is required and must be nonblank"):
        spec_from_dict(missing, project_required=True)
    with pytest.raises(ConfigError, match="project is required and must be nonblank"):
        spec_from_dict(_raw(project="   "), project_required=True)
    with pytest.raises(ConfigError, match="project must be a valid UUID"):
        spec_from_dict(_raw(project="not-a-uuid"))
    assert JobSpec.from_dict({"model": "Qwen/Qwen3.5-0.8B"}).project == ""

    synthetic = JobSpec(model="Qwen/Qwen3.5-0.8B")
    assert synthetic.project == ""
    assert synthetic.to_dict()["project"] == ""
    from flash.client.specs import spec_payload

    with pytest.raises(ValueError, match="project is required and must be nonblank"):
        spec_payload(synthetic)

    project_id = "11111111-1111-4111-8111-111111111111"
    grouped = spec_from_dict(_raw(project=f"  {project_id.upper()}  "))
    assert grouped.project == project_id
    assert grouped.to_dict()["project"] == project_id
    assert _job_from_dict(grouped.to_dict()).project == project_id
    assert JobSpec.from_json(grouped.to_json()).project == project_id


@pytest.mark.parametrize("invalid", ["per_step", 1])
def test_credit_assignment_rejects_invalid_values(invalid: object) -> None:
    with pytest.raises(ConfigError) as excinfo:
        spec_from_dict(_raw(**{"train.credit_assignment": invalid}))
    assert str(excinfo.value) == (
        f'train.credit_assignment must be "per_episode" or "per_turn"; got {invalid!r}'
    )


def test_job_spec_from_dict_credit_assignment_validates_worker_boundary() -> None:
    # the worker-side deserialization boundary must round-trip valid modes, default a missing value,
    # and reject a malformed persisted/tampered value rather than silently downgrading to per-episode.
    payload = spec_from_dict(_raw(**{"train.credit_assignment": "per_turn"})).to_dict()
    assert _job_from_dict(payload).train.credit_assignment == "per_turn"

    payload["train"].pop("credit_assignment", None)
    assert _job_from_dict(payload).train.credit_assignment == "per_episode"

    payload["train"]["credit_assignment"] = "per_step"
    with pytest.raises(ValueError, match="credit_assignment must be one of"):
        _job_from_dict(payload)


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

    # historical snapshots remain exact to their commits, including fields now managed or removed.
    # current adds save_at_steps, credit_assignment, entropy_quantile, and prompts_per_step; opd eos
    # is removed. lora_alpha is in both: authorable then, managed in between, a user knob again now.
    # prompts_per_step is current-only: the rl half of the optimizer-batch split, which those
    # snapshots predate because back then batch_size still carried both meanings.
    assert historical_shapes["861571e7"] - {
        "opd_eos_loss_coef",
        "hf_repo",
        "advantage_clip",
    } == TRAIN_SCHEMA_KEYS - {
        "credit_assignment",
        "save_at_steps",
        "entropy_quantile",
        "prompts_per_step",
    }
    assert "opd_eos_loss_coef" not in TRAIN_SCHEMA_KEYS
    assert "advantage_clip" not in TRAIN_SCHEMA_KEYS
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


@pytest.mark.parametrize(
    "lora_alpha",
    [
        pytest.param(256, id="non-default"),
        pytest.param(16, id="matching-derived"),
        pytest.param(None, id="null"),
        pytest.param(0, id="invalid"),
    ],
)
def test_warmstart_rejects_explicit_child_alpha(lora_alpha) -> None:
    # the source adapter's alpha is authoritative, so authoring one is rejected rather than
    # silently overwritten by the inherited value.
    raw = _raw(**{"train.init_from_adapter": "source-run", "train.lora_alpha": lora_alpha})
    raw["train"].pop("lora_rank")
    with pytest.raises(
        ConfigError,
        match=(
            r"train\.lora_alpha cannot be set with train\.init_from_adapter because source adapter "
            r"alpha metadata is authoritative"
        ),
    ):
        spec_from_dict(raw)


@pytest.mark.parametrize(("lora_rank", "expected_alpha"), [(16, 32), (32, 64), (8, 16)])
def test_lora_alpha_defaults_to_twice_rank(lora_rank, expected_alpha) -> None:
    spec = spec_from_dict(_raw(**{"train.lora_rank": lora_rank}))
    assert spec.train.lora_alpha == expected_alpha


def test_default_lora_rank_defaults_alpha_to_64() -> None:
    raw = _raw()
    raw["train"].pop("lora_rank")
    spec = spec_from_dict(raw)
    assert spec.train.lora_rank == 32
    assert spec.train.lora_alpha == 64


def test_directly_constructed_trainspec_derives_alpha_from_rank() -> None:
    # A library caller building TrainSpec(...) directly must get the same 2 x rank default as the
    # parsed path. to_dict() no longer strips alpha, so a stale scalar default would be SUBMITTED
    # and trained with (rank 8 shipping alpha 64 instead of 16) rather than re-derived server-side.
    assert TrainSpec(lora_rank=8).lora_alpha == 16
    assert TrainSpec().lora_alpha == 64  # default rank 32
    assert TrainSpec(lora_rank=8, lora_alpha=48).lora_alpha == 48  # explicit still wins


def test_internal_from_dict_round_trips_stored_lora_alpha() -> None:
    # The internal carrier preserves a stored alpha so an authored value and a warm-start's
    # inherited parent alpha (which need not equal 2 x rank) survive control-plane -> worker
    # serialization; alpha falls back to 2 x rank only when the payload omits it.
    base = {
        "model": "Qwen/Qwen3.5-0.8B",
        "algorithm": "grpo",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "train": {"epochs": 1, "max_examples": 8, "lora_rank": 16, "lora_alpha": 48},
    }
    assert JobSpec.from_dict(base).train.lora_alpha == 48  # present -> round-trip
    absent = {**base, "train": {"epochs": 1, "max_examples": 8, "lora_rank": 16}}
    assert JobSpec.from_dict(absent).train.lora_alpha == 32  # absent -> derive 2 x rank


def test_internal_dict_emits_an_authored_lora_alpha_to_the_worker() -> None:
    # the EMISSION direction, not just from_dict: to_internal_dict() is what the worker rehydrates
    # from, so an authored alpha that never reached the internal carrier would silently train at
    # the derived 2 x rank scaling instead of the value the user wrote.
    spec = spec_from_dict(_raw(**{"train.lora_rank": 16, "train.lora_alpha": 48}))
    assert spec.to_internal_dict()["train"]["lora_alpha"] == 48
    derived = spec_from_dict(_raw(**{"train.lora_rank": 16}))
    assert derived.to_internal_dict()["train"]["lora_alpha"] == 32


def test_authored_lora_alpha_overrides_the_derived_default() -> None:
    # an authored alpha need not equal 2 x rank, and it survives the public round trip the client
    # submits and the server re-validates.
    spec = spec_from_dict(_raw(**{"train.lora_rank": 16, "train.lora_alpha": 48}))
    assert spec.train.lora_alpha == 48
    public = spec.to_dict()
    assert public["train"]["lora_alpha"] == 48
    assert spec_from_dict(public).train.lora_alpha == 48


def test_warmstart_accepts_omitted_child_rank_with_internal_placeholder() -> None:
    raw = _raw(**{"train.init_from_adapter": "source-run"})
    raw["train"].pop("lora_rank")

    spec = spec_from_dict(raw)

    assert spec.train.init_from_adapter == "source-run"
    assert spec.train.lora_rank == 32


def test_public_warmstart_serialization_omits_resolved_internal_fields() -> None:
    spec = _job_from_dict(
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
    assert _job_from_dict(internal) == spec


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


def test_sft_max_examples_is_an_optional_prefix_cap() -> None:
    # an sft quote is backed by a workload profile that materializes and tokenizes the real
    # dataset, so an omitted or zero cap means "every row" and is measured. requiring a row count
    # here would only be asking the user to supply the number the profile exists to measure.
    raw = _raw(algorithm="sft")
    raw["train"] = {"epochs": 1, "lora_rank": 8}
    assert spec_from_dict(raw).train.max_examples is None
    raw["train"]["max_examples"] = 0
    assert spec_from_dict(raw).train.max_examples == 0
    raw["train"]["max_examples"] = 8
    assert spec_from_dict(raw).train.max_examples == 8


def test_missing_model_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must set `model`"):
        spec_from_dict({"project": "11111111-1111-4111-8111-111111111111", "algorithm": "sft"})


def test_hf_repo_is_managed_not_user_set() -> None:
    # [train] hf_repo is platform-managed (assigned server-side per run), so it is NOT a user config
    # key: a config without it parses fine, and a user who sets it is rejected loudly rather than
    # having their value silently dropped.
    raw = _raw()
    raw["train"] = {"epochs": 1, "max_examples": 10, "lora_rank": 8}
    assert spec_from_dict(raw).train.hf_repo == ""
    raw["train"]["hf_repo"] = "someone-else/their-repo"
    with pytest.raises(ConfigError, match=r"\[train\] unknown key\(s\): hf_repo"):
        spec_from_dict(raw)


def test_lora_rank_allows_rank128_for_small_serving_models() -> None:
    # The small dense tiers (default model Qwen3.5-0.8B) now serve rank-128 LoRA buffers.
    assert spec_from_dict(_raw(**{"train.lora_rank": 128})).train.lora_rank == 128


def test_lora_rank_must_fit_small_serving_cap() -> None:
    # Small-tier serving cap doubled 64 -> 128; rank 129 exceeds it.
    with pytest.raises(ConfigError, match="serving max_lora_rank=128"):
        spec_from_dict(_raw(**{"train.lora_rank": 129}))


def test_lora_rank_must_fit_large_serving_cap() -> None:
    from flash.core.catalog import serving_lora_rank_cap

    # the 27B is the rank-64 tier. this case used the 4B, which was rank-64 when it was written and
    # is now rank-128 -- leaving it there would have made this a duplicate of the small-cap test
    # above rather than coverage of the lower cap. derived from the catalog so it tracks the tier.
    cap = serving_lora_rank_cap("Qwen/Qwen3.6-27B")
    assert cap is not None
    with pytest.raises(ConfigError, match=f"serving max_lora_rank={cap}"):
        spec_from_dict(_raw(model="Qwen/Qwen3.6-27B", **{"train.lora_rank": cap + 1}))


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


# the flat [train] table is shared by all three algorithms, so a knob a run's algorithm cannot
# consume used to parse clean and do nothing. (algorithm, knob, value) pairs that must be rejected.
_INAPPLICABLE_CASES = [
    ("sft", "group_size", 8),
    ("sft", "temperature", 0.7),
    ("sft", "max_completion_tokens", 512),
    ("sft", "kl_penalty_coef", 0.5),
    ("sft", "entropy_quantile", 0.5),
    ("sft", "thinking_length_penalty_coef", 0.1),
    ("sft", "teacher_model", "glm-5.2"),
    ("sft", "credit_assignment", "per_turn"),
    ("sft", "stop_sequences", ["END"]),
    ("sft", "structured_outputs", '{"type": "object"}'),
    ("opd", "entropy_quantile", 0.5),
    ("opd", "thinking_length_penalty_coef", 0.1),
    ("opd", "credit_assignment", "per_turn"),
    ("grpo", "teacher_model", "glm-5.2"),
]

# the same knobs on an algorithm whose worker DOES read them. without this direction a validator
# that rejected everything everywhere would pass the rejection tests above.
_APPLICABLE_CASES = [
    ("grpo", "group_size", 8),
    ("opd", "group_size", 2),
    ("grpo", "temperature", 0.7),
    ("opd", "temperature", 0.7),
    ("grpo", "max_completion_tokens", 512),
    ("opd", "max_completion_tokens", 512),
    ("grpo", "kl_penalty_coef", 0.5),
    ("opd", "kl_penalty_coef", 0.5),
    ("grpo", "stop_sequences", ["END"]),
    ("opd", "stop_sequences", ["END"]),
    ("grpo", "structured_outputs", '{"type": "object"}'),
    ("opd", "structured_outputs", '{"type": "object"}'),
    ("grpo", "entropy_quantile", 0.5),
    ("grpo", "thinking_length_penalty_coef", 0.1),
    ("grpo", "credit_assignment", "per_turn"),
    ("opd", "teacher_model", "glm-5.2"),
]


@pytest.mark.parametrize(("algorithm", "knob", "value"), _INAPPLICABLE_CASES)
def test_train_knob_rejected_by_algorithms_that_cannot_consume_it(algorithm, knob, value) -> None:
    raw = _raw(algorithm=algorithm)
    raw["train"][knob] = value
    with pytest.raises(ConfigError, match=rf"train\.{knob}"):
        spec_from_dict(raw)


@pytest.mark.parametrize(("algorithm", "knob", "value"), _APPLICABLE_CASES)
def test_train_knob_accepted_by_algorithms_that_consume_it(algorithm, knob, value) -> None:
    raw = _raw(algorithm=algorithm)
    raw["train"][knob] = value
    assert spec_from_dict(raw).algorithm == algorithm


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_full_public_dict_round_trips_despite_knob_scoping(algorithm) -> None:
    """Round-trip every serialized TrainSpec field through algorithm scoping.

    Scope on meaningful values, not key presence, because ``to_dict()`` includes unauthored fields.
    """
    spec = spec_from_dict(_raw(algorithm=algorithm))

    restored = spec_from_dict(spec.to_dict())

    assert restored.algorithm == algorithm
    assert restored.train.epochs == spec.train.epochs


def test_advantage_clip_is_no_longer_a_config_key() -> None:
    # parsed, range-validated, and shipped to the worker, which then explicitly did not apply it.
    raw = _raw(algorithm="grpo")
    raw["train"]["advantage_clip"] = 1.5
    with pytest.raises(ConfigError, match=r"unknown key\(s\): advantage_clip"):
        spec_from_dict(raw)


def test_jobspec_drops_advantage_clip_from_persisted_records() -> None:
    """A run provisioned before #968 still parses; recovery would otherwise fail it and kill its worker."""
    original = _job_from_dict({})
    persisted = original.to_internal_dict()
    persisted["train"]["advantage_clip"] = 1.5

    restored = JobSpec.from_dict(persisted)

    # dropped, not resurrected: the key is gone from the schema and never reaches a worker.
    assert restored.train == original.train
    assert not hasattr(restored.train, "advantage_clip")


def test_jobspec_still_rejects_train_keys_that_were_never_tolerated() -> None:
    """The drop set is one key, not an amnesty on every removed field.

    `seeds` has its own rejection contract (test_from_dict_rejects_removed_legacy_train_seeds,
    #536), so widening the drop set speculatively would silently overturn it.
    """
    with pytest.raises(ValueError, match=r"train has unknown key\(s\): seeds"):
        JobSpec.from_dict({"train": {"seeds": [0, 1]}})


def test_environment_pip_is_authorable() -> None:
    """A scorer's third-party imports have no other declaration path onto the worker."""
    raw = _raw()
    raw["environment"]["pip"] = ["pymongo>=4.6", "rapidfuzz"]
    assert spec_from_dict(raw).environment.pip == ("pymongo>=4.6", "rapidfuzz")


def test_environment_pip_rejects_malformed_entries() -> None:
    """Malformed requirements must fail at parse, not mid-install with the GPU already billing."""
    raw = _raw()
    raw["environment"]["pip"] = "pymongo"
    with pytest.raises(ConfigError, match=r"not a string: use \[\"pymongo\"\]"):
        spec_from_dict(raw)

    raw = _raw()
    raw["environment"]["pip"] = ["pymongo", 7]
    with pytest.raises(ConfigError, match="non-empty requirement strings"):
        spec_from_dict(raw)

    raw = _raw()
    raw["environment"]["pip"] = ["pymongo", "   "]
    with pytest.raises(ConfigError, match="non-empty requirement strings"):
        spec_from_dict(raw)


def test_environment_pip_rejects_pip_options() -> None:
    """Entries are spliced into `python -m pip install`, so an option flag is not a requirement.

    `--no-deps` would suppress the dependencies of the mandatory freesolo worker requirement, and
    `--target` would redirect where it lands -- both reachable from a field that only names packages.
    """
    for option in ("--no-deps", "--target=/tmp/deps", "-e ."):
        raw = _raw()
        raw["environment"]["pip"] = ["pymongo>=4.6", option]
        with pytest.raises(ConfigError, match="must be requirements, not pip options"):
            spec_from_dict(raw)


def test_environment_pip_accepts_spaced_requirements() -> None:
    """Whitespace inside an entry is not a defect: both install paths pass one entry as one argv.

    ``subprocess.run([sys.executable, "-m", "pip", "install", *extra_pip])`` never goes through a
    shell, so a spaced PEP 508 requirement arrives at pip as a single operand and installs. Version
    specs, parenthesized clauses, spaced extras, markers and direct references all rely on this.
    """
    for entry in (
        "pymongo >= 4.6",
        "pkg (>=1.0)",
        "pkg [extra1, extra2] >=1.0",
        'pkg; python_version < "3.12"',
        "pkg @ https://host/a-1.0.whl",
    ):
        raw = _raw()
        raw["environment"]["pip"] = [entry]
        assert spec_from_dict(raw).environment.pip == (entry,)


def test_environment_pip_rejection_messages_never_echo_a_url() -> None:
    """Every rejection path must redact a URL, including the ones the credential guard never sees.

    These messages are printed by the CLI and returned verbatim as the server's HTTP error detail
    (``flash/server/platform/deps.py`` raises ``HTTPException(400, detail=str(exc))``), so quoting a
    value back copies it into terminals, CI output and API logs. The scalar, non-string and
    pip-option branches all raise BEFORE the URL credential guard is reached, so each has to redact
    on its own -- an option is credential-bearing in its own right via
    ``--extra-index-url=https://user:token@host``.
    """
    secret = "ghp_SECRETTOKEN"
    for value in (
        f"git+https://{secret}@github.com/org/repo.git",  # scalar: rejected as "not a list"
        [f"--extra-index-url=https://user:{secret}@host/simple"],  # pip option
        [f"git+https://{secret}@h/r.git".encode()],  # non-string entry
        [{"url": f"https://{secret}@h"}],  # non-string entry, nested
        [f"git+https://{secret}@github.com/o/r.git"],  # reaches the credential guard
        [f"https://h/p-1.0.whl?private_token={secret}"],  # query-string credential
    ):
        raw = _raw()
        raw["environment"]["pip"] = value
        with pytest.raises(ConfigError) as caught:
            spec_from_dict(raw)
        assert secret not in str(caught.value)

    # redaction is scoped to URL-shaped input: an ordinary typo still names itself, or the message
    # would stop being actionable for the mistakes that are not credentials.
    raw = _raw()
    raw["environment"]["pip"] = ["--no-deps"]
    with pytest.raises(ConfigError, match="--no-deps"):
        spec_from_dict(raw)


def test_environment_pip_rejects_url_credentials() -> None:
    """A spec is not a secret store: pip entries are persisted and uploaded verbatim.

    ``RunStatus.spec`` keeps the authored value and the worker's ``metrics.json`` carries it inside
    ``notes.job_spec``, so a token in a direct or VCS URL would land on disk and in the run log.
    """
    for url in (
        "git+https://user:ghp_SECRETTOKEN@github.com/org/repo.git#egg=pkg",
        "https://tok:s3cret@example.com/pkg-1.0.tar.gz",
        "pkg @ https://x:y@host/a.whl",
        "git+ssh://git:deploykey@host/repo.git",
        # ANY nonempty userinfo, not just `user:password`. a github token is conventionally passed
        # username-only, so requiring a literal colon would miss the most likely leak outright...
        "git+https://ghp_SECRETTOKEN@github.com/org/repo.git",
        # ...and the separator can arrive percent-encoded, which is the same credential.
        "git+https://user%3As3cret@github.com/org/repo.git",
        # a query string carries credentials just as well as userinfo does, and naming a package
        # never needs one: a private-index token and a presigned object-store signature.
        "pkg @ https://host/pkg-1.0.whl?private_token=s3cret",
        "https://host/pkg-1.0.whl?X-Amz-Signature=deploykey",
    ):
        raw = _raw()
        raw["environment"]["pip"] = [url]
        with pytest.raises(ConfigError, match="must not embed credentials") as caught:
            spec_from_dict(raw)
        # the message must not quote the requirement back -- that would copy the credential into
        # the very logs this rejection exists to keep it out of.
        for secret in ("ghp_SECRETTOKEN", "s3cret", "deploykey"):
            assert secret not in str(caught.value)

    # unauthenticated direct and VCS URLs stay usable; only inline userinfo is refused.
    for url in (
        "pkg @ https://host/a-1.0.whl",
        "git+https://github.com/org/repo.git#egg=pkg",
        "pymongo>=4.6",
        # a VCS ref pin puts `@` AFTER the authority, so it must not read as userinfo.
        "git+https://github.com/org/repo.git@v1.2.3",
    ):
        raw = _raw()
        raw["environment"]["pip"] = [url]
        assert spec_from_dict(raw).environment.pip == (url,)


def test_submit_payload_carries_the_pip_key() -> None:
    """spec_payload is what the CLI actually sends, and the server re-parses it with this parser."""
    from flash.client.specs import spec_payload

    raw = _raw()
    raw["environment"]["pip"] = ["pymongo>=4.6"]
    spec = spec_from_dict(raw)
    payload = spec_payload(spec, authored_train_keys=frozenset({"epochs"}))

    # dropping it here would strand the requirement on the client: the worker installs from payload.
    # a tuple in memory, a JSON array on the wire, exactly as the sibling `secrets` field travels.
    assert tuple(payload["environment"]["pip"]) == ("pymongo>=4.6",)
    assert json.loads(json.dumps(payload))["environment"]["pip"] == ["pymongo>=4.6"]
    assert spec_from_dict(payload).environment.pip == ("pymongo>=4.6",)


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


def test_gpu_retry_and_wall_are_managed_defaults_not_user_authored() -> None:
    # max_retries / max_wall_seconds are platform-managed lifecycle policy: a user config never sets
    # them (the GpuSpec default applies), and authoring either - with any value - is rejected loudly
    # as an unknown key rather than silently honored.
    defaults = GpuSpec()
    spec = spec_from_dict(_raw())
    assert spec.gpu.max_retries == defaults.max_retries
    assert spec.gpu.max_wall_seconds == defaults.max_wall_seconds

    for managed in ("max_retries", "max_wall_seconds"):
        raw = _raw()
        raw["gpu"][managed] = 7
        with pytest.raises(ConfigError, match=rf"\[gpu\] unknown key\(s\): {managed}"):
            spec_from_dict(raw)

    unknown = _raw()
    unknown["gpu"]["future_gpu_field"] = "rejected"
    with pytest.raises(ConfigError, match=r"\[gpu\] unknown key\(s\): future_gpu_field"):
        spec_from_dict(unknown)


def test_environment_subfields_reject_wrong_types() -> None:
    # The [environment] sub-fields are consumed by EnvironmentSpec(...) via dict(... or {}):
    # a present-but-wrong-typed value would otherwise crash opaquely (dict("x") / dict(1)).
    # Each must fail fast with a clear ConfigError instead.
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


@pytest.mark.parametrize(
    "key",
    [
        "PARASAIL_API_KEY",
        "FLASH_PUBLIC_URL",
        "FLASH_TEACHER_CAPABILITY",
    ],
)
def test_environment_cannot_declare_managed_teacher_transport_or_credentials(key) -> None:
    raw = _raw()
    raw["environment"] = {
        "id": "github:freesolo-co/envs@main:gsm8k/environment.py",
        "secrets": [key],
    }

    with pytest.raises(ConfigError, match="platform-managed"):
        spec_from_dict(raw)


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
        "secrets": ["SERPAPI_API_KEY", "OPENAI_API_KEY", "SERPAPI_API_KEY"],
    }
    spec = spec_from_dict(raw)
    assert spec.environment.params == {"k": "v"}
    # pip is authorable but omitted here, so it stays at its default.
    assert spec.environment.pip == ()
    assert spec.environment.secrets == ("SERPAPI_API_KEY", "OPENAI_API_KEY")
    # An explicit None (e.g. JSON `null`) is treated as missing -> default, NOT rejected.
    raw = _raw()
    raw["environment"] = {
        "id": "github:freesolo-co/envs@main:gsm8k/environment.py",
        "params": None,
        "secrets": None,
    }
    spec = spec_from_dict(raw)
    assert spec.environment.params == {}
    assert spec.environment.pip == ()
    assert spec.environment.secrets == ()


def test_jobspec_from_dict_rejects_path() -> None:
    # Defense-in-depth: a stale worker payload carrying a local path must be rejected.
    data = {
        "project": "11111111-1111-4111-8111-111111111111",
        "model": "Qwen/Qwen3-0.6B",
        "environment": {"id": "gsm8k", "path": "./environment.py"},
    }
    with pytest.raises(ValueError, match="local environment paths are no longer supported"):
        _job_from_dict(data)


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
    # explicit non-positive max_steps (0 or negative) is not rejected: it canonicalizes to none so a
    # single sentinel means "use the derived horizon" (max_examples still rejects negatives below).
    for bad in (0, -7):
        spec_np = spec_from_dict(_raw(**{"train.max_steps": bad}), run_id="caps-np")
        assert spec_np.train.max_steps is None
    with pytest.raises(ConfigError, match="max_examples must be >= 0"):
        spec_from_dict(_raw(**{"train.max_examples": -5}))


def test_job_spec_json_round_trip() -> None:
    spec = spec_from_dict(_raw(), run_id="rt-1")
    restored = JobSpec.from_json(spec.to_json())
    assert restored == spec
    assert restored.phase == "rl"  # grpo's internal phase id


def test_gpu_public_fields_survive_payload_and_server_reparse() -> None:
    from flash.client.specs import spec_payload

    defaults = GpuSpec()
    spec = spec_from_dict(
        _raw(**{"gpu.provider": " RunPod ", "gpu.type": "rtx-4090"}),
        run_id="gpu-rt",
    )
    payload = spec_payload(spec)
    # the public payload carries the user-authorable gpu knobs and omits managed lifecycle policy.
    assert payload["gpu"]["provider"] == "runpod"
    assert payload["gpu"]["type"] == "RTX 4090"
    assert "max_retries" not in payload["gpu"]
    assert "max_wall_seconds" not in payload["gpu"]

    reparsed = spec_from_dict(payload, run_id="server-reparse")
    assert reparsed.gpu.provider == "runpod"
    assert reparsed.gpu.type == "RTX 4090"
    assert reparsed.gpu.type == spec.gpu.type
    # managed lifecycle fields reconstitute to their defaults on the server reparse.
    assert reparsed.gpu.max_retries == defaults.max_retries
    assert reparsed.gpu.max_wall_seconds == defaults.max_wall_seconds


def test_gpu_constraints_reject_unknown_unsupported_or_undersized_values() -> None:
    with pytest.raises(ConfigError, match=r"gpu\.provider"):
        spec_from_dict(_raw(**{"gpu.provider": "aws"}))
    with pytest.raises(ConfigError, match=r"gpu\.type"):
        spec_from_dict(_raw(**{"gpu.type": "Tesla T4"}))
    with pytest.raises(ConfigError, match=r"unsupported gpu 'RTX A6000'"):
        spec_from_dict(_raw(**{"gpu.type": "RTX A6000"}))
    with pytest.raises(ConfigError, match="requires at least"):
        spec_from_dict(
            _raw(
                model="Qwen/Qwen3.5-9B",
                **{"gpu.type": "RTX 4090"},
            )
        )
    with pytest.raises(ConfigError, match="cannot provision"):
        spec_from_dict(
            _raw(
                **{
                    "gpu.provider": "lambda",
                    "gpu.type": "RTX 4090",
                }
            )
        )


def test_gpu_type_pins_and_unset_stays_auto(capsys) -> None:
    automatic_raw = _raw()
    automatic_raw["gpu"].pop("type", None)

    automatic = spec_from_dict(automatic_raw)
    pinned = spec_from_dict(_raw(**{"gpu.type": "B200"}))

    assert automatic.gpu.type == ""
    assert pinned.gpu.type == "B200"
    assert capsys.readouterr().err == ""


def test_removed_gpu_pin_key_is_rejected_as_unknown() -> None:
    removed_key = "exact" + "_type"
    raw = _raw()
    raw["gpu"][removed_key] = "H100"

    with pytest.raises(ConfigError, match=r"\[gpu\] unknown key"):
        spec_from_dict(raw)
    with pytest.raises(ValueError, match=r"gpu has unknown key"):
        _job_from_dict({"gpu": {removed_key: "H100"}})


def test_persisted_gpu_type_is_canonicalized_and_validated() -> None:
    assert _job_from_dict({"gpu": {"type": " h100 "}}).gpu.type == "H100"
    with pytest.raises(TypeError, match=r"gpu\.type must be a string"):
        _job_from_dict({"gpu": {"type": 1}})
    with pytest.raises(ValueError, match=r"gpu\.type: unsupported gpu 'H10O'"):
        _job_from_dict({"gpu": {"type": "H10O"}})
    with pytest.raises(ValueError, match=r"unsupported gpu 'RTX A6000'"):
        _job_from_dict({"gpu": {"type": "RTX A6000"}})


def test_removed_model_revision_config_key_has_targeted_remedy() -> None:
    message = (
        "config key `model_revision` was removed because Flash-managed serving loads a "
        "pre-quantized FP8 checkpoint resolved per base model, so it cannot honor an arbitrary "
        "upstream commit and an authored pin made the run undeployable. If you need a fixed "
        "upstream base, use `flash models export`; the exported adapter records Flash's resolved "
        "base revision for Hugging Face loading."
    )

    for value in ("main", "", None, 123, False, ["main"], {"revision": "main"}):
        with pytest.raises(ConfigError) as exc_info:
            spec_from_dict(_raw(model_revision=value))
        assert str(exc_info.value) == message


def test_model_revision_stays_available_to_internal_round_trips() -> None:
    persisted = {**JobSpec().to_internal_dict(), "model_revision": "  refs/pr/123  "}

    spec = JobSpec.from_dict(persisted)

    assert spec.model_revision == "refs/pr/123"
    assert JobSpec.from_json(spec.to_json()).model_revision == "refs/pr/123"
    assert spec.to_internal_dict()["model_revision"] == "refs/pr/123"
    assert "model_revision" not in spec.to_dict()

    for value in (None, 123, False, ["main"], {"revision": "main"}):
        with pytest.raises(TypeError, match="model_revision must be a string"):
            _job_from_dict({"model_revision": value})


def test_model_revision_auto_is_platform_managed_and_stripped_from_the_public_spec() -> None:
    """The marker records WHO pinned the base model, and only the runner may set it."""
    auto = replace(
        spec_from_dict(_raw()),
        model_revision="c" * 40,
        model_revision_auto=True,
    )
    public = auto.to_dict()
    assert "model_revision" not in public
    assert "model_revision_auto" not in public
    assert spec_from_dict(public).model_revision == ""
    assert auto.to_internal_dict()["model_revision_auto"] is True
    assert JobSpec.from_dict(auto.to_internal_dict()).model_revision_auto is True
    assert JobSpec.from_json(auto.to_json()).model_revision_auto is True

    # a user cannot forge it: it is absent from _TOP_LEVEL_KEYS, so the public parser refuses it
    with pytest.raises(ConfigError, match=r"unknown config key\(s\): model_revision_auto"):
        spec_from_dict(_raw(model_revision_auto=True))

    # the marker qualifies a pin and cannot outlive one
    assert replace(auto, model_revision="").model_revision_auto is False


def test_model_revision_auto_does_not_change_pre_existing_preparation_digests() -> None:
    """A snapshot prepared before this field existed must still rehash to its stored digest.

    `_preparation_digest` has to reproduce the bytes that were hashed, not today's serialization,
    or a still-valid warm-start or workload-profile run fails integrity validation on recovery.
    """
    from flash.runner.preparation import _preparation_digest

    unmarked = JobSpec(model="Qwen/Qwen3.5-9B", algorithm="sft", model_revision="c" * 40)

    # rebuild the pre-upgrade bytes: the same payload this build produces, minus the key that did
    # not exist then. mirrors _preparation_digest's own omission list rather than re-deriving it,
    # so the control cannot drift from the code under test.
    worker_payload = unmarked.to_internal_dict()
    for key in (
        "workload_profile_kind",
        "workload_profile_input_digest",
        "workload_profile_producer_version",
        "workload_profile",
    ):
        if not worker_payload.get(key):
            worker_payload.pop(key, None)
    worker_payload.pop("model_revision_auto", None)
    worker_payload.pop("gpu_count_auto", None)
    public_payload = unmarked.to_dict()  # to_dict() already strips the markers
    # the old plane popped `[environment] pip` from every public payload, so its bytes carried no
    # such key. mirrors _preparation_digest's drop-when-empty for the same reason as the list above.
    if not public_payload["environment"].get("pip"):
        public_payload["environment"].pop("pip", None)

    payload = json.dumps(
        {
            "version": 1,
            "public_spec": public_payload,
            "worker_spec": worker_payload,
            "adapter_identity": None,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    import hashlib

    legacy = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    assert _preparation_digest(unmarked, unmarked, None) == legacy

    # a marked run still binds the marker into its digest, so tampering remains detectable
    marked = replace(unmarked, model_revision_auto=True)
    assert _preparation_digest(marked, marked, None) != legacy


def test_resubmitting_a_public_spec_gets_a_fresh_runner_pin(monkeypatch) -> None:
    """Round-tripping an auto-pinned run's public spec must keep the pin platform-managed."""
    from flash.runner.preparation import _resolve_model_revision

    resolved_sha = "d" * 40

    class _Api:
        def __init__(self, *a, **k) -> None: ...

        def model_info(self, model, revision=None):
            _Api.asked_for = revision
            return type("_Info", (), {"sha": revision or resolved_sha})

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)

    auto = _resolve_model_revision(
        spec_from_dict(_raw(model="Qwen/Qwen3.5-9B", algorithm="sft")), required=True
    )
    assert auto.model_revision == resolved_sha
    assert auto.model_revision_auto is True

    rerun = _resolve_model_revision(spec_from_dict(auto.to_dict()), required=True)
    assert _Api.asked_for is None
    assert rerun.model_revision == resolved_sha
    assert rerun.model_revision_auto is True


def test_removing_model_revision_from_public_specs_keeps_new_digests_stable() -> None:
    from flash.runner.preparation import _preparation_digest

    public = spec_from_dict(_raw(model="Qwen/Qwen3.5-9B", algorithm="sft"))
    worker = replace(public, model_revision="a" * 40, model_revision_auto=True)
    stored = public.to_dict()

    assert "model_revision" not in stored
    assert _preparation_digest(public, worker, None) == _preparation_digest(
        JobSpec.from_dict(stored), worker, None
    )


@pytest.mark.parametrize("revision", ["", "a" * 40])
def test_pre_removal_public_model_revision_keeps_its_preparation_digest(revision) -> None:
    """A stored public revision key must rehash under the exact pre-removal payload."""
    import flash.runner as runner
    stored_public_spec = {
        **spec_from_dict(_raw(model="Qwen/Qwen3.5-9B", algorithm="sft")).to_dict(),
        "model_revision": revision,
    }
    stored_worker_spec = {
        **JobSpec.from_dict(stored_public_spec).to_internal_dict(),
        "run_id": "legacy-revision",
    }
    public = JobSpec.from_dict(stored_public_spec)
    worker = JobSpec.from_dict(stored_worker_spec)
    legacy_digest = runner._preparation_digest(
        public,
        worker,
        None,
        legacy_public_keys={"model_revision": revision},
    )
    status = runner.RunStatus(
        state="running",
        run_id="legacy-revision",
        spec=stored_public_spec,
        effective_preparation={
            "worker_spec": stored_worker_spec,
            "preparation_digest": legacy_digest,
            "workload_profile": {"legacy": True},
        },
    )
    status.effective_preparation["worker_spec"]["workload_profile_kind"] = "sft"
    status.effective_preparation["worker_spec"]["workload_profile_input_digest"] = "b" * 64
    status.effective_preparation["worker_spec"]["workload_profile_producer_version"] = "1.1.55"
    status.effective_preparation["worker_spec"]["workload_profile"] = {"legacy": True}
    worker = JobSpec.from_dict(status.effective_preparation["worker_spec"])
    status.effective_preparation["preparation_digest"] = runner._preparation_digest(
        public,
        worker,
        None,
        legacy_public_keys={"model_revision": revision},
    )

    recovered = runner.effective_spec_from_status(status)

    assert recovered.model_revision == revision
    assert recovered.model_revision_auto is False


@contextmanager
def _serializing_without_prompts_per_step():
    """Serialize `JobSpec` the way 1.1.40 did: with no ``prompts_per_step`` key at all.

    That build predates the field, so its payload carried no such key -- which hashes differently
    from the explicit null today's dataclass always emits. A digest meant to stand in for a real
    persisted snapshot has to be taken over those historical bytes, or the test asserts against a
    shape production never wrote.
    """
    original_internal, original_public = JobSpec.to_internal_dict, JobSpec.to_dict

    def _drop(emit):
        return lambda self: {
            **emit(self),
            "train": {k: v for k, v in emit(self)["train"].items() if k != "prompts_per_step"},
        }

    JobSpec.to_internal_dict, JobSpec.to_dict = _drop(original_internal), _drop(original_public)
    try:
        yield
    finally:
        JobSpec.to_internal_dict, JobSpec.to_dict = original_internal, original_public


def test_migrating_a_legacy_rollout_batch_keeps_its_preparation_digest_valid() -> None:
    """Recovering a persisted rollout snapshot must not fail integrity validation.

    The digest hashes the bytes that were STORED, and `from_dict` now MOVES the rollout batch from
    ``batch_size`` onto ``prompts_per_step``. Rehashing a snapshot with today's parse can therefore
    yield different bytes than were hashed at persist time -- rejecting exactly the warm-start
    grpo/opd runs this migration exists to rescue.

    Every stored shape is covered, because three of them differ only in ways an assertion about
    "the legacy value" cannot see: 1.1.40 predates the field and stored NO key, 1.1.43+ stored an
    explicit null, a mid-upgrade payload stored BOTH names, and a snapshot that authored no batch
    at all has nothing to migrate yet still changes shape. The last two were regressions found in
    review after an earlier fix keyed only on "is there a legacy value to move".

    Tamper and modern controls are included, since an assertion that only checked recovery would
    also pass if the digest stopped binding the batch entirely.
    """
    import flash.runner as runner

    def _spec(batch_size, prompts_per_step, *, algorithm="grpo"):
        spec = replace(
            JobSpec.from_dict(
                {
                    "model": "Qwen/Qwen3.5-4B",
                    "algorithm": algorithm,
                    "run_id": "legacy",
                    "environment": {"id": "github:owner/repo@main:env/environment.py"},
                    "gpu": {"type": "H100", "count": 1},
                    "train": {"epochs": 1, "group_size": 4},
                }
            ),
            model_revision="a" * 40,
            model_revision_auto=True,  # one of the two triggers for the digest check
        )
        # force the shape the OLD build parsed, bypassing today's migration.
        return replace(
            spec,
            train=replace(spec.train, batch_size=batch_size, prompts_per_step=prompts_per_step),
        )

    def _stored(batch_size, prompts_per_step, *, key_absent):
        """A snapshot as the pre-fix build persisted it, digested over exactly ITS bytes.

        `key_absent` reproduces 1.1.40, which predates ``prompts_per_step`` entirely -- so its
        payload carried no such key, which hashes differently from an explicit null. Serializing
        through today's dataclass always emits the key, so the digest has to be taken over the
        historical bytes or the test would assert against a shape production never wrote.
        """
        spec = _spec(batch_size, prompts_per_step)
        worker, public = spec.to_internal_dict(), spec.to_dict()
        if key_absent:
            worker["train"].pop("prompts_per_step", None)
            public["train"].pop("prompts_per_step", None)
        with _serializing_without_prompts_per_step() if key_absent else nullcontext():
            digest = runner._preparation_digest(spec, spec, None)
        return worker, public, digest

    def _recover(worker, public, digest):
        return runner.effective_spec_from_status(
            runner.RunStatus(
                state="running",
                run_id="legacy",
                spec=public,
                effective_preparation={"worker_spec": worker, "preparation_digest": digest},
            )
        )

    # (label, stored batch_size, stored prompts_per_step, key absent, migrated prompts_per_step)
    for label, batch_size, prompts_per_step, key_absent, expected in (
        ("1.1.40 authored 32", 32, None, True, 32),
        # nothing to migrate, but the reparse still emits a key the stored bytes lacked.
        ("1.1.40 authored nothing", None, None, True, None),
        ("1.1.43+ authored 32", 32, None, False, 32),
        ("1.1.43+ authored nothing", None, None, False, None),
        # mid-upgrade: both names stored. the migration drops the old one, so it still has to be
        # rehashed under the stored spelling.
        ("mixed both names", 32, 16, False, 16),
        ("modern new name only", None, 16, False, 16),
        # a non-positive legacy value is discarded by the parser, changing the shape again.
        ("legacy non-positive", 0, None, False, None),
    ):
        worker, public, digest = _stored(batch_size, prompts_per_step, key_absent=key_absent)
        recovered = _recover(worker, public, digest)
        assert (recovered.train.prompts_per_step, recovered.train.batch_size) == (
            expected,
            None,
        ), label

        # control: the digest still binds both names on BOTH halves. without this, "recovery works"
        # would also be satisfied by a restore that let any value through.
        #
        # each half is tampered ALONE. Changing both together hides a hole in either one, which is
        # how a public-side gap got through review: the parse DROPS a superseded `batch_size`, so
        # `_validate_effective_spec` never compares it, and a restore that fed the worker's reading
        # to the public payload overwrote the tampered value before hashing. Only the public spec
        # is user-visible, so that half is the one an attacker can reach.
        for key in ("batch_size", "prompts_per_step"):
            for half in ("worker", "public"):
                source = worker if half == "worker" else public
                if key not in source["train"]:
                    continue
                tampered = {**source, "train": {**source["train"], key: 999}}
                # either rejection is correct: a value the parse KEEPS is caught structurally by
                # `_validate_effective_spec` (public/worker mismatch) before the digest is even
                # reached, and one it DROPS reaches the digest. What must never happen is the
                # tamper being accepted, so both messages are allowed and neither is required.
                with pytest.raises(
                    ValueError,
                    match=r"failed integrity validation|does not match the public run",
                ):
                    _recover(
                        tampered if half == "worker" else worker,
                        tampered if half == "public" else public,
                        digest,
                    )

    # sft authors `batch_size` under its CURRENT name, so from_dict leaves it alone and the digest
    # must not replay anything for it.
    sft = JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "run_id": "sft-digest",
            "environment": {"id": "github:owner/repo@main:env/environment.py"},
            "gpu": {"type": "H100", "count": 1},
            "train": {"epochs": 1, "batch_size": 4},
        }
    )
    assert runner._stored_rollout_batch_spelling(sft.to_internal_dict()) is None


def test_re_persisting_a_legacy_rollout_run_keeps_it_recoverable(tmp_path, monkeypatch) -> None:
    """A quote refresh or realloc must not write a digest the next read cannot reproduce.

    `status.spec` is never rewritten, so a legacy rollout run keeps the old batch spelling for life
    and every READ replays it. The re-persist path rewrites the worker half and rehashes -- so if it
    hashes without that replay, the run recovers until its first quote refresh and fails afterwards,
    which is worse than failing outright because it passes every check at submission time.

    Repeated because re-persist has to be idempotent: the second one hashes bytes the first wrote.
    """
    import importlib

    import flash.runner as runner

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    from flash.runner.submit import _persist_effective_worker_spec

    for index, (batch_size, prompts_per_step, key_absent) in enumerate(
        (
            (32, None, True),  # 1.1.40: old name only, new key absent
            (None, None, True),  # 1.1.40: nothing authored
            (32, None, False),  # 1.1.43+: old name, new key null
            (32, 16, False),  # mid-upgrade: both names stored
            (None, 16, False),  # modern
        )
    ):
        run_id = f"repersist-{index}"
        spec = replace(
            JobSpec.from_dict(
                {
                    "model": "Qwen/Qwen3.5-4B",
                    "algorithm": "grpo",
                    "run_id": run_id,
                    "project": "11111111-1111-4111-8111-111111111111",
                    "environment": {"id": "github:owner/repo@main:env/environment.py"},
                    "gpu": {"type": "H100", "count": 1},
                    "train": {"epochs": 1, "group_size": 4, "lora_rank": 16},
                }
            ),
            model_revision="a" * 40,
            model_revision_auto=True,  # trips the digest gate
        )
        spec = replace(
            spec,
            train=replace(spec.train, batch_size=batch_size, prompts_per_step=prompts_per_step),
        )
        worker, public = spec.to_internal_dict(), spec.to_dict()
        if key_absent:
            worker["train"].pop("prompts_per_step", None)
            public["train"].pop("prompts_per_step", None)
        with _serializing_without_prompts_per_step() if key_absent else nullcontext():
            digest = runner._preparation_digest(spec, spec, None)
        runner._save_status(
            runner.RunStatus(
                state="queued",
                run_id=run_id,
                spec=public,
                effective_preparation={"worker_spec": worker, "preparation_digest": digest},
            )
        )

        before = runner.effective_spec_from_status(runner.get_status(run_id))
        for _ in range(2):
            stored = runner.get_status(run_id).effective_preparation["worker_spec"]
            _persist_effective_worker_spec(JobSpec.from_dict(stored))
            after = runner.effective_spec_from_status(runner.get_status(run_id))
            # recoverable AND unchanged: a digest that merely validates is not enough if the batch
            # it certifies drifted.
            assert (after.train.prompts_per_step, after.train.batch_size) == (
                before.train.prompts_per_step,
                None,
            ), (run_id, batch_size, prompts_per_step)


def test_effective_spec_validation_accepts_the_asymmetric_auto_pin_shape() -> None:
    """The public/worker structural compare must tolerate the marker living on one half only.

    Submit persists `spec=public_spec.to_dict()`, and to_dict() strips the marker, so a real
    auto-pinned run is asymmetric by construction: the worker half carries True, and the public
    half rebuilt from the stored dict reads the False default. `_validate_effective_spec` compares
    the two structurally, so without an exclusion it raises for every auto-pinned run -- turning
    the 400 this PR removes into a 409 and leaving those runs exactly as undeployable.

    Built by round-tripping through to_dict() rather than by hand, so the test cannot assert a
    shape that submit does not actually produce.
    """
    from flash.runner.preparation import _validate_effective_spec

    public = spec_from_dict(_raw(model="Qwen/Qwen3.5-9B", algorithm="sft"))
    worker = replace(public, model_revision="a" * 40, model_revision_auto=True)
    reparsed_public = JobSpec.from_dict(worker.to_dict())
    assert reparsed_public.model_revision_auto is False
    assert reparsed_public.model_revision == ""
    assert worker.model_revision == "a" * 40

    _validate_effective_spec(reparsed_public, worker)  # raises if the exclusion is missing

    # historical authored pins remain compared on stored public payloads, so changing the worker's
    # revision still fails closed for already-persisted runs.
    authored_public = JobSpec.from_dict({**worker.to_dict(), "model_revision": "a" * 40})
    authored_worker = replace(worker, model_revision_auto=False)
    for tampered in (
        replace(authored_worker, model_revision="b" * 40),
        replace(authored_worker, model_revision="b" * 40, model_revision_auto=True),
    ):
        with pytest.raises(ValueError, match="does not match the public run"):
            _validate_effective_spec(authored_public, tampered)


def test_unknown_top_level_scalar_and_jobspec_gpu_shapes_fail_closed() -> None:
    with pytest.raises(ConfigError, match=r"unknown config key\(s\): model_revison"):
        spec_from_dict(_raw(model_revison="main"))

    for gpu in (False, "H100", ["H100"]):
        with pytest.raises(TypeError, match="gpu must be an object"):
            _job_from_dict({"gpu": gpu})
    with pytest.raises(ValueError, match=r"gpu has unknown key\(s\): exact_typ"):
        _job_from_dict({"gpu": {"exact_typ": "H100"}})
    with pytest.raises(TypeError, match=r"gpu\.provider must be a string"):
        _job_from_dict({"gpu": {"provider": 1}})
    with pytest.raises(TypeError, match=r"gpu\.type must be a string"):
        _job_from_dict({"gpu": {"type": 1}})
    with pytest.raises(ValueError, match="cannot provision"):
        _job_from_dict({"gpu": {"provider": "lambda", "type": "RTX 4090"}})

    restored = _job_from_dict({"gpu": {"provider": " LAMBDA ", "type": "h100"}})
    assert restored.gpu.provider == "lambda"
    assert restored.gpu.type == "H100"
    assert _job_from_dict({}).gpu.provider == ""
    assert _job_from_dict({}).gpu.type == ""


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


def test_programmatic_sft_submit_fails_closed_without_a_profilable_environment(
    tmp_path, monkeypatch
) -> None:
    # sft is quoted from a workload profile that tokenizes the real dataset, so a spec with no
    # environment to profile has no measurable workload. it must fail closed rather than fall back
    # to an assumed row count -- including on the dry-run preview, which previews a real submit.
    from flash.core.spec import JobSpec

    orch = _fresh_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(
        orch, "_resolve_model_revision", lambda s, **_kw: replace(s, model_revision="a" * 40, model_revision_auto=True)
    )
    spec = JobSpec(
        run_id="sft-no-environment",
        model="Qwen/Qwen3.5-0.8B",
        algorithm="sft",
        project="11111111-1111-4111-8111-111111111111",
    )
    with pytest.raises(orch.WorkloadProfileUnavailable, match="requires an environment id"):
        orch.submit_job(spec, dry_run=True)
    with pytest.raises(FileNotFoundError):
        orch.get_status(spec.run_id)


def test_programmatic_sft_submit_rejects_adapter_continuation(tmp_path, monkeypatch) -> None:
    from flash.core.spec import JobSpec, TrainSpec

    orch = _fresh_orchestrator(tmp_path, monkeypatch)
    # offline: sft requires a resolved model_revision, which calls HfApi().model_info(). Stub it
    # the same way the sibling test above does, so the adapter-continuation rejection this test is
    # about is what fails -- not an unrelated network lookup on a disconnected runner.
    monkeypatch.setattr(
        orch, "_resolve_model_revision", lambda s, **_kw: replace(s, model_revision="a" * 40, model_revision_auto=True)
    )
    spec = JobSpec(
        run_id="sft-warmstart",
        model="Qwen/Qwen3.5-0.8B",
        algorithm="sft",
        project="11111111-1111-4111-8111-111111111111",
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

    # hf_repo and run_id are platform-managed: they survive the INTERNAL round trip
    # (to_internal_dict -> from_dict), which is what the worker/control plane use, not to_dict().
    d = spec.to_internal_dict()
    d["train"] = {**d["train"], "hf_repo": "Freesolo-Co/flashrun-flash-1-x"}
    spec_with_repo = _job_from_dict(d)
    assert orch.adapter_ref(spec_with_repo) == "Freesolo-Co/flashrun-flash-1-x:rl/flash-1-x"


# ---------------------------------------------------------------------------
# engine.vram: fit estimates + offline param lookup
# ---------------------------------------------------------------------------


def test_vram_estimate_scales_with_params_and_algorithm() -> None:
    from flash.engine.plan import vram

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
    from flash.engine.plan import vram

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


# ---------------------------------------------------------------------------
# _logging: namespace + level resolution
# ---------------------------------------------------------------------------


def test_get_logger_namespacing() -> None:
    from flash._internal.logging import get_logger

    assert get_logger().name == "flash"
    assert get_logger("flash").name == "flash"
    assert get_logger("flash.providers").name == "flash.providers"
    assert get_logger("mymodule").name == "flash.mymodule"


def test_configure_logging_verbosity() -> None:
    from flash._internal import logging as _logging

    _logging.configure_logging(verbosity=0)
    assert logging.getLogger("flash").level == logging.WARNING
    _logging.configure_logging(verbosity=1)
    assert logging.getLogger("flash").level == logging.INFO
    _logging.configure_logging(verbosity=2)
    assert logging.getLogger("flash").level == logging.DEBUG


def test_removed_worker_environment_table_is_rejected(tmp_path) -> None:
    # the deleted per-run env override table is now an unknown section, not a silently ignored one:
    # a config that still carries it must fail loudly rather than train with the overrides dropped.
    path = tmp_path / "removed-worker-environment.toml"
    path.write_text(
        'model = "Qwen/Qwen3.5-0.8B"\nalgorithm = "grpo"\n[worker_env]\nCUSTOM_FLAG = "value"\n'
    )

    with pytest.raises(ConfigError) as exc_info:
        spec_and_train_keys_from_file(str(path))

    message = str(exc_info.value)
    assert "unknown config section(s): worker_env" in message
    assert "(allowed tables: environment, train, gpu, wandb)" in message


@pytest.mark.parametrize("stored", [{}, {"CUSTOM_FLAG": "value"}])
def test_a_run_persisted_before_the_worker_env_removal_still_reloads(stored) -> None:
    """A record written by the OLD plane must survive the upgrade that drops the field.

    Specs were persisted with asdict, so EVERY record the pre-upgrade plane wrote names worker_env,
    including the defaulted empty one. Stored records are never rewritten and from_dict is strict, so
    without the dropped-key tolerance the first reload after deploy raises and a still-running job
    loses its recovery, deploy, and serving paths.
    """
    persisted = {**JobSpec().to_internal_dict(), "worker_env": stored}

    spec = JobSpec.from_dict(persisted)

    # tolerated on read, but genuinely gone: the value must not come back as an attribute.
    assert not hasattr(spec, "worker_env")


def test_the_dropped_worker_env_key_is_tolerated_on_read_only_never_authored() -> None:
    """Tolerance must not quietly re-open the table as an authorable one.

    from_dict ignores it so old RECORDS load; the schema layer still rejects it so a CONFIG naming it
    fails loudly rather than training with its overrides silently discarded.
    """
    with pytest.raises(ConfigError, match="unknown config section"):
        spec_from_dict(_raw(worker_env={"CUSTOM_FLAG": "value"}))


def test_an_unknown_top_level_key_is_still_rejected_on_read() -> None:
    # the tolerance is scoped to keys the spec itself dropped, so it cannot become a general
    # accept-anything hole in the persisted-spec reader.
    with pytest.raises(ValueError, match="unknown key"):
        JobSpec.from_dict({**JobSpec().to_internal_dict(), "not_a_real_key": 1})


def test_a_pre_upgrade_run_that_authored_overrides_is_told_they_stopped_applying(caplog) -> None:
    """Tolerating the key must not make the behavior change silent.

    A run submitted with overrides keeps them in its record forever, but nothing forwards them now,
    so it trains on managed defaults instead of what was authored. Reloading without a word would
    make that indistinguishable from a run that never set them -- the operator reading logs after an
    unexpected result would have nothing pointing at the cause.
    """
    persisted = {
        **JobSpec().to_internal_dict(),
        "run_id": "run-legacy-1",
        "worker_env": {"FLASH_VERL_PYTHON": "/custom/verl/bin/python"},
    }

    with caplog.at_level(logging.WARNING, logger="flash.spec"):
        JobSpec.from_dict(persisted)

    assert "FLASH_VERL_PYTHON" in caplog.text
    assert "run-legacy-1" in caplog.text
    assert "NOT" in caplog.text


def test_the_warning_names_the_run_or_is_not_emitted_at_all(caplog) -> None:
    """An unidentified warning is worse than none: it cannot be acted on, and it duplicates.

    The same stored run is read through both shapes -- the internal worker spec (asdict, keeps
    run_id) and the public spec (to_dict, pops it). Warning on the public read would emit a second
    line naming no run, which an operator cannot map back to anything, alongside the identified line
    the worker-spec read already produced.
    """
    spec = JobSpec.from_dict(
        {**JobSpec().to_internal_dict(), "run_id": "run-legacy-1"},
    )
    dropped = {"FLASH_VERL_PYTHON": "/custom/verl/bin/python"}

    with caplog.at_level(logging.WARNING, logger="flash.spec"):
        JobSpec.from_dict({**spec.to_internal_dict(), "worker_env": dropped})
    assert "run-legacy-1" in caplog.text

    caplog.clear()
    # the public shape pops run_id, so this read stays quiet rather than saying "run <unknown>".
    public = spec.to_dict()
    assert "run_id" not in public
    with caplog.at_level(logging.WARNING, logger="flash.spec"):
        JobSpec.from_dict({**public, "worker_env": dropped})
    assert caplog.text == ""


def test_a_record_without_the_dropped_key_says_nothing(caplog) -> None:
    # every record the pre-upgrade plane wrote names the key, defaulted-empty included. warning on
    # the empty ones would fire on effectively every reload and train operators to ignore it.
    persisted = {**JobSpec().to_internal_dict(), "worker_env": {}}

    with caplog.at_level(logging.WARNING, logger="flash.spec"):
        JobSpec.from_dict(persisted)

    assert caplog.text == ""


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
    from flash.core.spec import coerce_bool

    assert coerce_bool(value) is expected


# ---------------------------------------------------------------------------
# gpu.count (multi-gpu job spec)
# ---------------------------------------------------------------------------


def test_unset_gpu_count_keeps_the_digest_stable_integer_placeholder() -> None:
    parsed = spec_from_dict(_raw())
    assert parsed.gpu.count == 1
    assert parsed.gpu_count_auto is True
    assert GpuSpec().count == 1

    # public serialization is part of the preparation digest. keep the historical integer key shape;
    # only the internal marker may distinguish this placeholder from an authored count=1.
    assert parsed.to_dict()["gpu"]["count"] == 1
    assert isinstance(parsed.to_dict()["gpu"]["count"], int)
    assert "gpu_count_auto" not in parsed.to_dict()

    internal = _job_from_dict(parsed.to_internal_dict())
    assert internal.gpu.count == 1
    assert internal.gpu_count_auto is True
    assert JobSpec.from_json(parsed.to_json()).gpu_count_auto is True

    from flash.runner.supervise.lifecycle import _spec_with_gpu

    # the marker is PROVENANCE ("the author omitted gpu.count"), so it survives the resolved shape.
    # it is the only surviving record of that fact -- the public halves of an auto-sized and an
    # authored single-card run are byte-identical -- so clearing it here made a recovered
    # auto-sized run re-allocate hard-pinned to one card.
    resolved = _spec_with_gpu(internal, "H200", 2)
    assert resolved.gpu.count == 2
    assert resolved.gpu_count_auto is True
    with pytest.raises(ConfigError, match=r"unknown config key\(s\): gpu_count_auto"):
        spec_from_dict(_raw(gpu_count_auto=True))


def test_authored_gpu_count_parses_and_roundtrips() -> None:
    parsed = spec_from_dict(_raw(**{"gpu.count": 4}))
    assert parsed.gpu.count == 4
    assert parsed.gpu_count_auto is False
    # count survives both serialization hops (asdict-based to_dict / to_json).
    assert _job_from_dict(parsed.to_dict()).gpu.count == 4
    assert JobSpec.from_json(parsed.to_json()).gpu.count == 4


def test_explicit_gpu_count_one_is_not_auto() -> None:
    parsed = spec_from_dict(_raw(**{"gpu.count": 1}))
    assert parsed.gpu.count == 1
    assert parsed.gpu_count_auto is False


def test_gpu_type_without_count_keeps_the_single_card_pin() -> None:
    parsed = spec_from_dict(_raw(**{"gpu.type": "B200"}))
    assert parsed.gpu.count == 1
    assert parsed.gpu_count_auto is False


@pytest.mark.parametrize("good", [1, 8])
def test_gpu_spec_direct_construction_accepts_valid_count(good: int) -> None:
    assert GpuSpec(count=good).count == good


@pytest.mark.parametrize(
    ("bad", "exc"),
    [
        (0, ValueError),
        (-1, ValueError),
        (9, ValueError),
        (True, TypeError),  # bool is an int subclass; must be rejected
        ("two", TypeError),
    ],
)
def test_gpu_spec_direct_construction_rejects_bad_count(bad: object, exc: type) -> None:
    with pytest.raises(exc):
        GpuSpec(count=bad)


def test_gpu_count_of_reads_spec_and_defaults() -> None:
    from flash.core.spec import gpu_count_of

    assert gpu_count_of(None) == 1  # no spec -> single gpu
    assert (
        gpu_count_of(JobSpec(project="11111111-1111-4111-8111-111111111111")) == 1
    )  # default gpu count
    assert (
        gpu_count_of(JobSpec(project="11111111-1111-4111-8111-111111111111", gpu=GpuSpec(count=3)))
        == 3
    )


# ---------------------------------------------------------------------------
# the optimizer batch is a different quantity per algorithm, so it has a different name
# ---------------------------------------------------------------------------


def test_sft_batch_size_is_rejected_on_rl_and_names_the_right_key() -> None:
    """The trap this split exists to close.

    `batch_size = 1` is the standard sft out-of-memory workaround. Copied into a grpo/opd config it
    used to parse and silently mean one prompt per optimizer update -- the run trained, logged and
    billed, and nothing errored. It must now fail at parse and say which key to use instead.
    """
    for algorithm in ("grpo", "opd"):
        with pytest.raises(ConfigError) as excinfo:
            spec_from_dict(_raw(algorithm=algorithm, **{"train.batch_size": 1}))

        message = str(excinfo.value)
        assert "batch_size does not apply to" in message
        assert "prompts_per_step" in message  # the remedy names the key that works


def test_rl_prompts_per_step_is_rejected_on_sft_and_names_the_right_key() -> None:
    with pytest.raises(ConfigError) as excinfo:
        spec_from_dict(_raw(algorithm="sft", **{"train.prompts_per_step": 8}))

    message = str(excinfo.value)
    assert "prompts_per_step does not apply to sft" in message
    assert "batch_size" in message


def test_each_algorithm_still_accepts_its_own_optimizer_batch_key() -> None:
    """The rejection is per-algorithm, not a blanket ban on either name."""
    assert spec_from_dict(_raw(algorithm="sft", **{"train.batch_size": 4})).train.batch_size == 4
    for algorithm in ("grpo", "opd"):
        spec = spec_from_dict(_raw(algorithm=algorithm, **{"train.prompts_per_step": 16}))
        assert spec.train.prompts_per_step == 16
        assert spec.train.batch_size is None


def test_neither_optimizer_batch_key_is_required() -> None:
    """Both are optional; an unset key leaves the worker's tuned recipe default in place."""
    for algorithm in ("sft", "grpo", "opd"):
        spec = spec_from_dict(_raw(algorithm=algorithm))
        assert spec.train.batch_size is None
        assert spec.train.prompts_per_step is None
