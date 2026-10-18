"""Spec plumbing not covered elsewhere: config validation error paths, JobSpec
serialization round-trips, worker env loading, run-id path containment, VRAM
estimates, and logging namespace helpers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import fields, replace

import pytest

import flash.runner.lifecycle.preparation as runner_preparation
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit
import flash.runner.supervise.deploy as runner_deploy
from flash.core.spec import (
    GpuSpec,
    JobSpec,
    TrainSpec,
    attributed_gpu_type,
    load_job_spec_from_env,
    persisted_gpu_head,
    persisted_gpu_types,
)
from flash.schema import (
    TRAIN_KEY_MIN_VERSIONS,
    TRAIN_SCHEMA_KEYS,
    ConfigError,
    format_checkpoint_ref,
    parse_checkpoint_ref,
    spec_and_train_keys_from_file,
    spec_from_dict,
    train_schema_metadata,
    validate_train_keys,
)

BASE_RAW = {
    "model": "Qwen/Qwen3.5-9B",
    "algorithm": "grpo",
    "project": "11111111-1111-4111-8111-111111111111",
    "environment": {"id": "freesolo/math-agent/gsm8k"},
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
    payload = {"project": "11111111-1111-4111-8111-111111111111", **data}
    if "train" not in payload:
        payload["train"] = {"credit_assignment": "per_episode"}
    elif isinstance(payload["train"], dict) and "credit_assignment" not in payload["train"]:
        payload["train"] = {**payload["train"], "credit_assignment": "per_episode"}
    return JobSpec.from_dict(payload)


# ---------------------------------------------------------------------------
# schema validation error paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "checkpoint_id",
    [
        "run-a",
        "run-a/step-00",
        "run-a/step-01",
        "run-a/current",
        "run-a/final/extra",
        " run-a/final",
        "run-a/final ",
        "run-a@final." + "a" * 40,
    ],
)
def test_parse_checkpoint_ref_rejects_noncanonical_values(checkpoint_id):
    assert parse_checkpoint_ref(checkpoint_id) is None


def test_checkpoint_ref_round_trip_requires_explicit_checkpoint():
    assert parse_checkpoint_ref("run-a/final") == ("run-a", None)
    assert parse_checkpoint_ref("run-a/step-0") == ("run-a", 0)
    assert parse_checkpoint_ref("run-a/step-42") == ("run-a", 42)
    assert format_checkpoint_ref("run-a") == "run-a/final"
    assert format_checkpoint_ref("run-a", 42) == "run-a/step-42"


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
        ({"model": ["Qwen/Qwen3.5-9B"]}, "must be a model id string"),
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
        'model = "Qwen/Qwen3.5-9B"\n'
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
    persisted = JobSpec(model="Qwen/Qwen3.5-9B").to_internal_dict()
    assert JobSpec.from_dict(persisted).project == ""

    synthetic = JobSpec(model="Qwen/Qwen3.5-9B")
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
    payload = spec_from_dict(_raw(**{"train.credit_assignment": "per_turn"})).to_dict()
    assert _job_from_dict(payload).train.credit_assignment == "per_turn"

    payload["train"].pop("credit_assignment", None)
    assert JobSpec.from_dict(payload).train.credit_assignment == "per_episode"

    for invalid in (None, "", "  ", "per_step"):
        payload["train"]["credit_assignment"] = invalid
        with pytest.raises(ValueError, match="credit_assignment must be one of"):
            JobSpec.from_dict(payload)


def test_job_spec_from_dict_rejects_nonpositive_save_every() -> None:
    payload = spec_from_dict(_raw(**{"train.save_every": 20})).to_dict()
    assert _job_from_dict(payload).train.save_every == 20

    for invalid in (0, -1, -20):
        payload["train"]["save_every"] = invalid
        with pytest.raises(ValueError, match=r"train\.save_every must be positive"):
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


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_init_from_adapter_parses_for_every_target_algorithm(algorithm) -> None:
    """Warm start is target-algorithm agnostic: every algorithm may continue an adapter.

    SFT used to be rejected here. The restriction described the retired trl SFT backend; the verl
    backend loads a warm-start adapter exactly like GRPO and OPD do, so all nine source/target
    combinations parse.
    """
    raw = _raw(algorithm=algorithm, **{"train.init_from_adapter": "source-run/final"})
    # the source adapter's rank/alpha metadata is authoritative for every warm start, so authoring
    # either alongside init_from_adapter stays rejected; BASE_RAW presets a rank.
    raw["train"].pop("lora_rank")

    spec = spec_from_dict(raw)

    assert spec.algorithm == algorithm
    assert spec.train.init_from_adapter == "source-run/final"


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
            _raw(**{"train.init_from_adapter": "source-run/final", "train.lora_rank": lora_rank})
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
    raw = _raw(**{"train.init_from_adapter": "source-run/final", "train.lora_alpha": lora_alpha})
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
        "model": "Qwen/Qwen3.5-9B",
        "algorithm": "grpo",
        "environment": {"id": "github:owner/repo@main:env/environment.py"},
        "train": {
            "epochs": 1,
            "max_examples": 8,
            "lora_rank": 16,
            "lora_alpha": 48,
            "credit_assignment": "per_episode",
        },
    }
    assert JobSpec.from_dict(base).train.lora_alpha == 48  # present -> round-trip
    absent = {
        **base,
        "train": {
            "epochs": 1,
            "max_examples": 8,
            "lora_rank": 16,
            "credit_assignment": "per_episode",
        },
    }
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
    raw = _raw(**{"train.init_from_adapter": "source-run/final"})
    raw["train"].pop("lora_rank")

    spec = spec_from_dict(raw)

    assert spec.train.init_from_adapter == "source-run/final"
    assert spec.train.lora_rank == 32


def test_gpu_sizing_consumes_the_canonical_warmstart_reference() -> None:
    """An invalid reference must not be approximated as an unresolved warm start.

    This intentionally pins the only error-order change from sharing the canonical parse: the bad
    reference is reported before rank-dependent gpu sizing can reinterpret its non-empty text.
    """
    raw = _raw(
        model="Qwen/Qwen3.6-35B-A3B",
        **{"gpu.type": "A100 PCIe", "train.init_from_adapter": "not/a/checkpoint/ref"},
    )
    raw["train"].pop("lora_rank")

    with pytest.raises(ConfigError, match=r"train\.init_from_adapter must be `<run_id>/final`"):
        spec_from_dict(raw)


def test_warmstart_placeholder_rank_does_not_reject_a_source_rank_that_fits_b200() -> None:
    from flash.providers.core.allocator import required_vram_gb

    train = {
        "epochs": 1,
        "max_examples": 10,
        "prompts_per_step": 8,
        "group_size": 4,
        "max_context_tokens": 1536,
        "max_completion_tokens": 512,
    }
    base = _raw(
        model="Qwen/Qwen3.6-35B-A3B",
        algorithm="grpo",
        **{
            "gpu.type": "B200",
            **{f"train.{key}": value for key, value in train.items()},
        },
    )
    base["train"].pop("lora_rank")

    warm = spec_from_dict(
        {
            **base,
            "train": {**base["train"], "init_from_adapter": "source-rank-4/final"},
        }
    )
    cold_rank4 = spec_from_dict({**base, "train": {**base["train"], "lora_rank": 4}})

    assert warm.train.lora_rank == 32
    assert cold_rank4.train.lora_rank == 4
    assert (
        required_vram_gb(warm.model, warm.algorithm, train={**base["train"], "lora_rank": 4}) == 180
    )
    assert (
        required_vram_gb(warm.model, warm.algorithm, train={**base["train"], "lora_rank": 32})
        == 199
    )
    with pytest.raises(ConfigError, match=r"requires at least 199 GB"):
        spec_from_dict({**base, "train": {**base["train"], "lora_rank": 32}})


def test_parse_time_vram_rejection_names_the_wider_shape_that_fits() -> None:
    """A single-card pin a second card would satisfy is a one-flag fix, not a dead end.

    The allocator ends every fit rejection with ``wider_shape_remedy``; this parse-time check
    did not, so an authored ``count = 1`` on the largest card Flash manages was rejected with no
    remedy at all. A user reading it concludes there is nothing left to rent, when ``--gpus 2``
    admits the same run.
    """
    train = {
        "epochs": 1,
        "max_examples": 10,
        "prompts_per_step": 8,
        "group_size": 4,
        "max_context_tokens": 1536,
        "max_completion_tokens": 512,
        "lora_rank": 32,
    }
    raw = _raw(
        model="Qwen/Qwen3.6-35B-A3B",
        algorithm="grpo",
        **{
            "gpu.type": "B200",
            "gpu.count": 1,
            **{f"train.{key}": value for key, value in train.items()},
        },
    )
    with pytest.raises(ConfigError) as excinfo:
        spec_from_dict(raw)
    message = str(excinfo.value)
    assert "requires at least 199 GB" in message
    assert "--gpus 2" in message, f"rejection gave no remedy: {message}"
    # and the suggestion has to be true: the same config at that count must parse.
    widened = spec_from_dict({**raw, "gpu": {**raw["gpu"], "count": 2}})
    assert widened.gpu.count == 2

    # a HARD provider pin narrows the pool, so a class whose wider shape that provider does not
    # freely rent must not be advertised. lambda names its card count in the instance type.
    with pytest.raises(ConfigError) as pinned:
        spec_from_dict({**raw, "gpu": {**raw["gpu"], "provider": "lambda"}})
    pinned_message = str(pinned.value)
    assert "requires at least 199 GB" in pinned_message
    assert "--gpus" not in pinned_message, (
        f"a lambda-pinned run was sent to a wider shape its pin may not carry: {pinned_message}"
    )


def test_warmstart_still_rejects_a_shape_impossible_at_the_minimum_rank() -> None:
    raw = _raw(
        model="Qwen/Qwen3.6-35B-A3B",
        **{
            "gpu.type": "B200",
            "train.init_from_adapter": "source-run/final",
            "train.max_context_tokens": 32768,
            "train.max_completion_tokens": 512,
            "train.prompts_per_step": 8,
            "train.group_size": 4,
        },
    )
    raw["train"].pop("lora_rank")

    # the pinned class is named, because the exact-card gate reaches this shape before the pool-wide
    # search does. the requirement is the same 216 GB either way.
    with pytest.raises(ConfigError, match=r"gpu\.type 'B200' .* at least 216 GB"):
        spec_from_dict(raw)


def test_warmstart_still_validates_the_authored_card_at_the_minimum_rank() -> None:
    """Relaxing the rank must not switch the exact-card gate off.

    That gate is the only parse-time check of an authored `gpu.type`; skipping it let a pin that
    cannot hold the run at ANY rank parse, because the fallback search ranks the whole pool rather
    than the pinned class. rank 1 is a true vram lower bound, so a card rejected against it cannot fit
    whatever rank the source turns out to have.
    """
    raw = _raw(
        model="Qwen/Qwen3.6-35B-A3B",
        **{"gpu.type": "A100 PCIe", "train.init_from_adapter": "source-run/final"},
    )
    raw["train"].pop("lora_rank")

    with pytest.raises(ConfigError, match=r"gpu\.type 'A100 PCIe' has 80 GB VRAM"):
        spec_from_dict(raw)


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
    raw = _raw(**{"train.init_from_adapter": "source-run/final"})
    raw["train"].pop("lora_rank")
    public = spec_from_dict(raw).to_dict()

    restored = spec_from_dict(public)

    assert restored.train.init_from_adapter == "source-run/final"
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


def test_lora_rank_allows_rank128_for_default_serving_model() -> None:
    # the default 9b serving profile supports rank-128 lora buffers.
    assert spec_from_dict(_raw(**{"train.lora_rank": 128})).train.lora_rank == 128


def test_lora_rank_must_fit_default_serving_cap() -> None:
    # rank 129 exceeds the default 9b serving profile's rank-128 cap.
    with pytest.raises(ConfigError, match="serving max_lora_rank=128"):
        spec_from_dict(_raw(**{"train.lora_rank": 129}))


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_qwen38_training_rank_must_fit_the_approved_serving_cap(algorithm: str) -> None:
    from flash.core.catalog import serving_lora_rank_cap

    assert serving_lora_rank_cap("Qwen/Qwen3.8-27B") == 64
    assert (
        spec_from_dict(
            _raw(
                model="Qwen/Qwen3.8-27B",
                algorithm=algorithm,
                **{"train.lora_rank": 64},
            )
        ).train.lora_rank
        == 64
    )
    with pytest.raises(ConfigError, match="serving max_lora_rank=64"):
        spec_from_dict(
            _raw(
                model="Qwen/Qwen3.8-27B",
                algorithm=algorithm,
                **{"train.lora_rank": 65},
            )
        )


def test_bare_environment_id_is_rejected() -> None:
    # A bare id like "gsm8k", or a two-segment one that predates per-project names, passes the
    # presence check but is not a Freesolo env slug; reject it up front.
    for bad in (
        "gsm8k",
        "owner/name",
        "owner/project/",
        "/project/name",
        "a/b/c/d",
        "owner/project/..",
        "owner/project/.",
        "owner/project/na me",
        "owner/project/name:tag",
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
    from flash.envs.loading.adapter import is_freesolo_environment_id
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


@pytest.mark.parametrize("unknown_key", ["path", "pth", "totally_made_up"])
def test_jobspec_from_dict_rejects_unknown_environment_keys(unknown_key) -> None:
    data = {
        "project": "11111111-1111-4111-8111-111111111111",
        "model": "Qwen/Qwen3-0.6B",
        "environment": {"id": "gsm8k", unknown_key: "./environment.py"},
    }
    with pytest.raises(ValueError, match=rf"environment has unknown key\(s\): {unknown_key}"):
        _job_from_dict(data)


def test_jobspec_from_dict_keeps_valid_environment_package() -> None:
    data = {
        "project": "11111111-1111-4111-8111-111111111111",
        "model": "Qwen/Qwen3-0.6B",
        "environment": {
            "id": "owner/project/gsm8k",
            "resolved_sha": "d" * 40,
            "package": {
                "artifact_revision": "a" * 40,
                "archive_sha256": "b" * 64,
                "manifest_sha256": "c" * 64,
            },
        },
    }

    spec = _job_from_dict(data)

    assert spec.environment.package is not None
    assert spec.environment.package.artifact_revision == "a" * 40


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
        _raw(**{"gpu.provider": " RunPod ", "gpu.type": "A100 PCIe"}),
        run_id="gpu-rt",
    )
    payload = spec_payload(spec)
    # the public payload carries the user-authorable gpu knobs and omits managed lifecycle policy.
    assert payload["gpu"]["provider"] == "runpod"
    assert payload["gpu"]["type"] == "A100 PCIe"
    assert "max_retries" not in payload["gpu"]
    assert "max_wall_seconds" not in payload["gpu"]

    reparsed = spec_from_dict(payload, run_id="server-reparse")
    assert reparsed.gpu.provider == "runpod"
    assert reparsed.gpu.type == "A100 PCIe"
    assert reparsed.gpu.type == spec.gpu.type
    # managed lifecycle fields reconstitute to their defaults on the server reparse.
    assert reparsed.gpu.max_retries == defaults.max_retries
    assert reparsed.gpu.max_wall_seconds == defaults.max_wall_seconds


def test_gpu_provider_preferences_validate_dedupe_and_round_trip_in_order() -> None:
    spec = spec_from_dict(
        _raw(**{"gpu.providers": [" Vast ", "runpod", "vast", "lambda", "runpod"]}),
        run_id="gpu-provider-preferences",
    )

    assert spec.gpu.providers == ("vast", "runpod", "lambda")
    public = spec.to_dict()
    assert public["gpu"]["providers"] == ("vast", "runpod", "lambda")
    assert JobSpec.from_dict(public).gpu.providers == ("vast", "runpod", "lambda")
    assert JobSpec.from_json(spec.to_json()).gpu.providers == ("vast", "runpod", "lambda")


def test_gpu_provider_preferences_reject_invalid_authoring() -> None:
    with pytest.raises(ValueError, match="must name at least one provider"):
        GpuSpec(providers=[])
    with pytest.raises(ConfigError, match="must name at least one provider"):
        spec_from_dict(_raw(**{"gpu.providers": []}))
    with pytest.raises(ConfigError, match=r"one of runpod, lambda, vast"):
        spec_from_dict(_raw(**{"gpu.providers": ["aws"]}))
    with pytest.raises(ConfigError, match=r"gpu\.provider and gpu\.providers cannot both be set"):
        spec_from_dict(_raw(**{"gpu.provider": "runpod", "gpu.providers": ["vast"]}))


def test_cost_quote_preserves_soft_provider_preference(monkeypatch) -> None:
    """The quote follows the authored preference, on a plane configured to serve it.

    The harness deletes the lambda/vast keys and injects only a RunPod pool, so the preference has
    to be made reachable explicitly -- quoting lambda on the bare fixture plane would assert the
    very defect `test_cost_quote_skips_a_preference_this_plane_cannot_provision` guards against.
    """
    from flash.cost.analytical import estimate_cost
    from flash.cost.spec import runconfig_from_spec
    from flash.providers.core import registry as providers_registry

    spec = spec_from_dict(_raw(**{"gpu.providers": ["lambda", "vast"]}))
    config = runconfig_from_spec(spec)

    assert config.provider == "auto"
    assert config.providers == ("lambda", "vast")
    monkeypatch.setattr(
        providers_registry, "available_providers", lambda: ("runpod", "lambda", "vast")
    )
    assert estimate_cost(config).provider == "lambda"


def test_cost_quote_skips_a_preference_this_plane_cannot_provision(monkeypatch) -> None:
    """The quote must price what allocation would really rent, not an unreachable preference.

    ``allocate()`` searches the CONFIGURED providers, so a preference naming one this plane has no
    credentials for is silently ignored there. Quoting it anyway prices a shape the run can never
    get, and the server's affordability check runs on that estimate -- so a balance that covers the
    real allocation can be refused with a 402.
    """
    from flash.cost.analytical import estimate_cost
    from flash.cost.spec import runconfig_from_spec
    from flash.providers.core import registry as providers_registry

    spec = spec_from_dict(_raw(**{"gpu.providers": ["lambda"]}))
    config = runconfig_from_spec(spec)

    monkeypatch.setattr(providers_registry, "available_providers", lambda: ("runpod",))
    unreachable = estimate_cost(config)
    assert unreachable.provider == "runpod"

    # the same preference is still honored on a plane that can actually provision it.
    monkeypatch.setattr(providers_registry, "available_providers", lambda: ("runpod", "lambda"))
    assert estimate_cost(config).provider == "lambda"


def test_cost_quote_refuses_a_shape_no_configured_provider_can_rent(monkeypatch) -> None:
    """Exhausting the eligible set must raise, not fall through to an unrestricted `auto` quote.

    A vast-only plane cannot rent a B200. Ranking the registered RunPod pool anyway returns a quote
    for hardware this plane has no credentials for, which then passes the affordability check and
    only fails once live allocation runs -- after the run is recorded.
    """
    from flash.cost.analytical import estimate_cost
    from flash.cost.spec import runconfig_from_spec
    from flash.providers.core import registry as providers_registry

    spec = spec_from_dict(_raw(**{"gpu.providers": ["vast"], "gpu.type": "B200"}))
    config = runconfig_from_spec(spec)

    monkeypatch.setattr(providers_registry, "available_providers", lambda: ("vast",))
    with pytest.raises(ValueError, match="vast"):
        estimate_cost(config)

    # a plane that can actually rent the class still quotes it.
    monkeypatch.setattr(providers_registry, "available_providers", lambda: ("runpod", "vast"))
    assert estimate_cost(config).provider == "runpod"


def test_cost_quote_prices_the_cheapest_acceptable_gpu_class() -> None:
    """An ordered `[gpu] type` list must be quoted over the whole acceptable set.

    ``allocate()`` cost-ranks every class ``acceptable_types`` returns, so pricing only the head
    quotes a shape the run may never be given: a ["B200", "H100"] run is quoted ~3x the H100 the
    allocator would really rent, and the submit-time affordability precheck can refuse a run the
    organization can afford.
    """
    from flash.cost.analytical import estimate_cost
    from flash.cost.spec import runconfig_from_spec

    listed = spec_from_dict(_raw(**{"gpu.type": ["B200", "H100"]}))
    assert listed.gpu.acceptable_types == ("B200", "H100")

    config = runconfig_from_spec(listed)
    assert config.gpu_type == "B200"
    assert config.gpu_type_fallbacks == ("H100",)

    head_only = estimate_cost(runconfig_from_spec(spec_from_dict(_raw(**{"gpu.type": "B200"}))))
    fallback_only = estimate_cost(runconfig_from_spec(spec_from_dict(_raw(**{"gpu.type": "H100"}))))
    quoted = estimate_cost(config)

    # the acceptable fallback is genuinely cheaper here, so the head-only quote is an overquote.
    assert fallback_only.total_usd < head_only.total_usd
    assert quoted.gpu == "H100"
    assert quoted.total_usd == pytest.approx(fallback_only.total_usd)

    # authoring order does not change the answer: allocation ranks on cost, not position.
    reversed_order = estimate_cost(
        runconfig_from_spec(spec_from_dict(_raw(**{"gpu.type": ["H100", "B200"]})))
    )
    assert reversed_order.gpu == "H100"

    # a bare pin stays a hard pin -- carrying fallbacks must not widen a single authored class.
    assert (
        estimate_cost(runconfig_from_spec(spec_from_dict(_raw(**{"gpu.type": "B200"})))).gpu
        == "B200"
    )


def test_cost_quote_no_fit_error_names_every_acceptable_class() -> None:
    """The rejection must name the whole declared set, not just the head that happened to be first.

    Built as a RunConfig rather than a spec: the parse gate rejects an unfittable pin before the
    quote runs, so the spec door cannot reach this branch.
    """
    from flash.cost.analytical import estimate_cost
    from flash.cost.types import RunConfig

    config = RunConfig(
        model_id="Qwen/Qwen3.6-35B-A3B",
        method="grpo",
        steps=10,
        batch_size=8,
        group_size=4,
        gpu_count=1,
        gpu_type="RTX 4090",
        gpu_type_fallbacks=("RTX 5090",),
    )
    with pytest.raises(ValueError, match="cannot fit this run") as exc:
        estimate_cost(config)
    assert "RTX 4090" in str(exc.value)
    assert "RTX 5090" in str(exc.value)


def test_soft_provider_preference_does_not_reject_an_ineligible_gpu_type_pair() -> None:
    spec = spec_from_dict(_raw(**{"gpu.providers": ["lambda"], "gpu.type": "A100 PCIe"}))

    assert spec.gpu.providers == ("lambda",)
    assert spec.gpu.type == "A100 PCIe"


def test_persisted_gpu_provider_preferences_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="must name at least one provider"):
        _job_from_dict({"gpu": {"providers": []}})
    with pytest.raises(ValueError, match=r"one of runpod, lambda, vast"):
        _job_from_dict({"gpu": {"providers": ["aws"]}})
    with pytest.raises(ValueError, match=r"gpu\.provider and gpu\.providers cannot both be set"):
        _job_from_dict({"gpu": {"provider": "runpod", "providers": ["vast"]}})


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


def test_gpu_type_accepts_an_ordered_list_of_acceptable_classes() -> None:
    """An ordered list preserves every acceptable class behind one concrete head."""
    spec = spec_from_dict(_raw(**{"gpu.type": ["A100 PCIe", "A100 SXM"]}))

    assert spec.gpu.type == "A100 PCIe"
    assert spec.gpu.type_fallbacks == ("A100 SXM",)
    assert spec.gpu.acceptable_types == ("A100 PCIe", "A100 SXM")

    # aliases canonicalize per entry, exactly as the scalar form does.
    assert spec_from_dict(_raw(**{"gpu.type": [" a100 pcie ", "h100"]})).gpu.acceptable_types == (
        "A100 PCIe",
        "H100",
    )
    # a one-element list is exactly a scalar pin, fallbacks included (there are none).
    lone = spec_from_dict(_raw(**{"gpu.type": ["B200"]}))
    assert (lone.gpu.type, lone.gpu.type_fallbacks) == ("B200", ())
    assert (
        lone.gpu.acceptable_types
        == spec_from_dict(_raw(**{"gpu.type": "B200"})).gpu.acceptable_types
    )
    # a repeat asks for nothing the first mention did not, and the first position is preference.
    assert spec_from_dict(
        _raw(**{"gpu.type": ["H100", "A100 PCIe", "H100"]})
    ).gpu.acceptable_types == ("H100", "A100 PCIe")


def test_gpu_type_list_rejects_empty_and_unusable_entries() -> None:
    """Every named class must be valid, reachable, and large enough."""
    with pytest.raises(ConfigError, match=r"must not be an empty list"):
        spec_from_dict(_raw(**{"gpu.type": []}))
    with pytest.raises(ConfigError, match=r"unsupported gpu"):
        spec_from_dict(_raw(**{"gpu.type": ["A100 PCIe", "Tesla T4"]}))
    with pytest.raises(ConfigError, match=r"gpu\.type entries must be strings"):
        spec_from_dict(_raw(**{"gpu.type": ["A100 PCIe", 1]}))
    with pytest.raises(ConfigError, match=r"gpu\.type entries must not be empty"):
        spec_from_dict(_raw(**{"gpu.type": ["A100 PCIe", "  "]}))
    # the vram floor applies to the whole list, so a fallback too small to hold the run is refused.
    with pytest.raises(ConfigError, match="requires at least"):
        spec_from_dict(_raw(model="Qwen/Qwen3.5-9B", **{"gpu.type": ["A100 PCIe", "RTX 4090"]}))
    # and so does provider compatibility.
    with pytest.raises(ConfigError, match="cannot provision"):
        spec_from_dict(_raw(**{"gpu.provider": "lambda", "gpu.type": ["H100", "RTX 4090"]}))


def test_gpu_type_list_round_trips_through_the_public_payload() -> None:
    """Public serialization restores the authored list spelling."""
    from flash.client.specs import spec_payload

    spec = spec_from_dict(_raw(**{"gpu.type": ["A100 PCIe", "A100 SXM"]}), run_id="gpu-list")
    payload = spec_payload(spec)

    assert payload["gpu"]["type"] == ["A100 PCIe", "A100 SXM"]
    assert "type_fallbacks" not in payload["gpu"]

    reparsed = spec_from_dict(payload, run_id="server-reparse")
    assert reparsed.gpu.acceptable_types == spec.gpu.acceptable_types


def test_authoring_gpu_type_fallbacks_directly_is_rejected() -> None:
    """The internal split form is not public configuration."""
    raw = _raw()
    raw["gpu"]["type_fallbacks"] = ["A100 SXM"]

    with pytest.raises(ConfigError, match=r"\[gpu\] unknown key"):
        spec_from_dict(raw)


def test_persisted_gpu_type_fallbacks_are_canonicalized_and_require_a_head() -> None:
    """Persisted fallback classes receive the same validation as the head."""
    assert _job_from_dict(
        {"gpu": {"type": "h100", "type_fallbacks": [" a100 pcie "]}}
    ).gpu.acceptable_types == ("H100", "A100 PCIe")
    with pytest.raises(ValueError, match=r"unsupported gpu"):
        _job_from_dict({"gpu": {"type": "H100", "type_fallbacks": ["H10O"]}})
    with pytest.raises(TypeError, match=r"must be a list of strings"):
        _job_from_dict({"gpu": {"type": "H100", "type_fallbacks": "A100 PCIe"}})
    with pytest.raises(ValueError, match=r"requires gpu\.type"):
        _job_from_dict({"gpu": {"type_fallbacks": ["H100"]}})


def test_gpu_spec_guards_direct_ordered_pin_construction() -> None:
    spec = GpuSpec(type=" h100 ", type_fallbacks=[" a100 pcie ", "H100"])
    assert spec.acceptable_types == ("H100", "A100 PCIe")

    with pytest.raises(TypeError, match=r"must be a list of strings"):
        GpuSpec(type="H100", type_fallbacks="A100 PCIe")
    with pytest.raises(TypeError, match=r"entry must be a string"):
        GpuSpec(type="H100", type_fallbacks=(123,))
    with pytest.raises(ValueError, match=r"unsupported gpu"):
        GpuSpec(type="H100", type_fallbacks=("H10O",))


def test_persisted_ordered_gpu_pin_survives_a_status_reload() -> None:
    """The public list form remains stable across repeated status reloads."""
    spec = _job_from_dict({"gpu": {"type": "A100 PCIe", "type_fallbacks": ["A100 SXM"]}})

    persisted = spec.to_dict()
    assert persisted["gpu"]["type"] == ["A100 PCIe", "A100 SXM"]

    reloaded = JobSpec.from_dict(persisted)
    assert reloaded.gpu.acceptable_types == spec.gpu.acceptable_types
    # and the reload is stable, because a record is re-read on every hop, not just the first.
    assert JobSpec.from_dict(reloaded.to_dict()).gpu.acceptable_types == spec.gpu.acceptable_types

    # the two spellings are alternatives, never combined: one of them has to be the authored order.
    with pytest.raises(ValueError, match=r"cannot both be set"):
        _job_from_dict({"gpu": {"type": ["H100", "B200"], "type_fallbacks": ["A100 SXM"]}})
    with pytest.raises(ValueError, match=r"at least one gpu"):
        _job_from_dict({"gpu": {"type": []}})


def test_persisted_gpu_head_reads_an_unparseable_spec_without_raising() -> None:
    """The raw reader remains total when a persisted spec cannot be parsed."""
    assert persisted_gpu_head({"gpu": {"type": ["A100 PCIe", "A100 SXM"]}}) == "A100 PCIe"
    assert persisted_gpu_head({"gpu": {"type": "H200"}}) == "H200"
    # malformed or absent -> "" (falsey), which every call site already guards on. no raise.
    for malformed in (
        None,
        {},
        {"gpu": {}},
        {"gpu": {"type": ""}},
        {"gpu": {"type": []}},
        {"gpu": {"type": None}},
        {"gpu": "not-a-dict"},
        {"gpu": {"type": 7}},
        "truthy-string",
        ["truthy-list"],
    ):
        assert persisted_gpu_head(malformed) == ""


def test_persisted_gpu_types_names_every_acceptable_class() -> None:
    """The raw reader returns every usable class without raising."""
    assert persisted_gpu_types({"gpu": {"type": ["A100 PCIe", "A100 SXM"]}}) == (
        "A100 PCIe",
        "A100 SXM",
    )
    # a scalar pin is the one-entry case, so callers need no separate spelling for it.
    assert persisted_gpu_types({"gpu": {"type": "H200"}}) == ("H200",)
    # authored order is preserved and duplicates collapse, so the index cannot double-count.
    assert persisted_gpu_types({"gpu": {"type": ["H200", "A100 PCIe", "H200"]}}) == (
        "H200",
        "A100 PCIe",
    )
    # junk entries are dropped rather than raising, and the good ones still survive.
    assert persisted_gpu_types({"gpu": {"type": ["H200", 7, "", None]}}) == ("H200",)
    for malformed in (
        None,
        {},
        {"gpu": {}},
        {"gpu": {"type": ""}},
        {"gpu": {"type": []}},
        {"gpu": {"type": None}},
        {"gpu": "not-a-dict"},
        {"gpu": {"type": 7}},
        "truthy-string",
        ["truthy-list"],
    ):
        assert persisted_gpu_types(malformed) == ()


def test_attributed_gpu_type_prefers_effective_worker_spec_and_is_total() -> None:
    status = {
        "remote": None,
        "effective_preparation": {"worker_spec": {"gpu": {"type": "A100 SXM"}}},
        "spec": {"gpu": {"type": ["RTX 5090", "A100 SXM"]}},
    }
    assert attributed_gpu_type(status) == "A100 SXM"
    status["remote"] = {"allocated_gpu": "H200"}
    assert attributed_gpu_type(status) == "H200"
    status["remote"] = None
    status["realized_cost_remote"] = {"allocated_gpu": "A100 PCIe"}
    assert attributed_gpu_type(status) == "A100 PCIe"
    for malformed in ("status", ["status"], {"remote": []}, {"effective_preparation": "bad"}):
        assert attributed_gpu_type(malformed) == ""


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


def test_runner_model_revision_stays_available_to_internal_round_trips() -> None:
    revision = "a" * 40
    persisted = {
        **JobSpec().to_internal_dict(),
        "model_revision": revision,
        "model_revision_auto": True,
    }

    spec = JobSpec.from_dict(persisted)

    assert spec.model_revision == revision
    assert JobSpec.from_json(spec.to_json()).model_revision == revision
    assert spec.to_internal_dict()["model_revision"] == revision
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


def test_model_revision_force_pin_is_internal_only_and_round_trips() -> None:
    revision = "d" * 40
    forced = replace(
        spec_from_dict(_raw()),
        model_revision=revision,
        model_revision_auto=True,
        model_revision_force_pin=True,
    )

    public = forced.to_dict()
    internal = forced.to_internal_dict()

    assert "model_revision_force_pin" not in public
    assert internal["model_revision_force_pin"] is True
    assert JobSpec.from_dict(internal) == forced
    assert JobSpec.from_json(forced.to_json()) == forced
    with pytest.raises(ConfigError, match=r"unknown config key\(s\): model_revision_force_pin"):
        spec_from_dict(_raw(model_revision_force_pin=True))


@pytest.mark.parametrize("field_name", ["model_revision_auto", "model_revision_force_pin"])
@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, 1.0, (), []])
def test_model_revision_markers_require_bool_for_direct_construction_and_replace(
    field_name, value
) -> None:
    revision = "a" * 40
    constructor_kwargs = {"model_revision": revision}
    if field_name == "model_revision_force_pin":
        constructor_kwargs["model_revision_auto"] = True
    constructor_kwargs[field_name] = value

    with pytest.raises(TypeError, match=rf"{field_name} must be a boolean"):
        JobSpec(**constructor_kwargs)

    valid = JobSpec(model_revision=revision, model_revision_auto=True)
    with pytest.raises(TypeError, match=rf"{field_name} must be a boolean"):
        replace(valid, **{field_name: value})


@pytest.mark.parametrize(
    ("model_revision_auto", "model_revision_force_pin"),
    [(True, False), (True, True)],
)
def test_model_revision_markers_accept_valid_bool_states(
    model_revision_auto, model_revision_force_pin
) -> None:
    spec = JobSpec(
        model_revision="a" * 40,
        model_revision_auto=model_revision_auto,
        model_revision_force_pin=model_revision_force_pin,
    )

    assert spec.model_revision_auto is model_revision_auto
    assert spec.model_revision_force_pin is model_revision_force_pin
    assert replace(spec) == spec


@pytest.mark.parametrize(
    ("auto_raw", "force_raw", "expected"),
    [
        ("true", "false", (True, False)),
        (1, 0, (True, False)),
        ("true", "true", (True, True)),
    ],
)
def test_model_revision_marker_from_dict_coercion_and_roundtrip_are_unchanged(
    auto_raw, force_raw, expected
) -> None:
    payload = JobSpec(model_revision="a" * 40).to_internal_dict()
    payload.update(model_revision_auto=auto_raw, model_revision_force_pin=force_raw)

    restored = JobSpec.from_dict(payload)

    assert (restored.model_revision_auto, restored.model_revision_force_pin) == expected
    assert JobSpec.from_dict(restored.to_internal_dict()) == restored


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_revision_force_pin": True},
        {
            "model_revision": "main",
            "model_revision_auto": True,
            "model_revision_force_pin": True,
        },
        {
            "model_revision": "A" * 40,
            "model_revision_auto": True,
            "model_revision_force_pin": True,
        },
    ],
)
def test_model_revision_force_pin_rejects_invalid_internal_states(overrides) -> None:
    with pytest.raises(ValueError, match="model_revision_force_pin requires"):
        _job_from_dict(overrides)


def test_ordered_gpu_pin_changes_the_preparation_digest() -> None:
    from flash.runner.lifecycle.preparation import _preparation_digest

    scalar = JobSpec(
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        model_revision="c" * 40,
        gpu=GpuSpec(type="A100 PCIe", count=1, provider="runpod"),
    )
    ordered = replace(
        scalar,
        gpu=GpuSpec(type="A100 PCIe", type_fallbacks=("A100 SXM",), count=1, provider="runpod"),
    )
    other = replace(
        scalar, gpu=GpuSpec(type="A100 PCIe", type_fallbacks=("H100",), count=1, provider="runpod")
    )

    assert _preparation_digest(ordered, ordered, None) != _preparation_digest(scalar, scalar, None)
    assert _preparation_digest(ordered, ordered, None) != _preparation_digest(other, other, None)


def test_resubmitting_a_public_spec_gets_a_fresh_runner_pin(monkeypatch) -> None:
    """Round-tripping an auto-pinned run's public spec must keep the pin platform-managed."""
    from flash.runner.lifecycle.preparation import _resolve_model_revision

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


def test_forced_sft_model_revision_verifies_and_retains_the_exact_pin(monkeypatch) -> None:
    from flash.runner.lifecycle.preparation import _resolve_model_revision

    exact = "a" * 40
    moving_head = "b" * 40
    asked_for = []

    class _Api:
        def __init__(self, *a, **k) -> None: ...

        def model_info(self, model, revision=None):
            asked_for.append(revision)
            return type("_Info", (), {"sha": moving_head if revision is None else revision})

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    forced = _job_from_dict(
        {
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "sft",
            "model_revision": exact,
            "model_revision_auto": True,
            "model_revision_force_pin": True,
        }
    )

    resolved = _resolve_model_revision(forced, required=True)

    assert asked_for == [exact]
    assert resolved.model_revision == exact
    assert resolved.model_revision_auto is True
    assert resolved.model_revision_force_pin is False
    assert resolved.to_internal_dict()["model_revision_force_pin"] is False


@pytest.mark.parametrize("reported", ["b" * 40, "A" * 40])
def test_forced_model_revision_rejects_a_mismatched_hub_resolution(monkeypatch, reported) -> None:
    from flash.runner.lifecycle.preparation import _resolve_model_revision

    class _Api:
        def __init__(self, *a, **k) -> None: ...

        def model_info(self, model, revision=None):
            return type("_Info", (), {"sha": reported})

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    forced = _job_from_dict(
        {
            "model_revision": "a" * 40,
            "model_revision_auto": True,
            "model_revision_force_pin": True,
        }
    )

    with pytest.raises(ValueError, match="could not resolve model_revision"):
        _resolve_model_revision(forced, required=True)


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_forced_model_revision_verifies_for_rollout_algorithms(monkeypatch, algorithm) -> None:
    from flash.runner.lifecycle.preparation import _resolve_model_revision

    exact = "c" * 40
    asked_for = []

    class _Api:
        def __init__(self, *a, **k) -> None: ...

        def model_info(self, model, revision=None):
            asked_for.append(revision)
            return type("_Info", (), {"sha": exact})

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    forced = _job_from_dict(
        {
            "algorithm": algorithm,
            "model_revision": exact,
            "model_revision_auto": True,
            "model_revision_force_pin": True,
        }
    )

    resolved = _resolve_model_revision(forced, required=False)

    assert asked_for == [exact]
    assert resolved.model_revision == exact
    assert resolved.model_revision_auto is True
    assert resolved.model_revision_force_pin is False


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_qwen38_catalog_revision_is_forced_and_verified(monkeypatch, algorithm) -> None:
    from flash.runner.lifecycle.preparation import _resolve_model_revision

    exact = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    asked_for = []

    class _Api:
        def __init__(self, *a, **k) -> None: ...

        def model_info(self, model, revision=None):
            asked_for.append((model, revision))
            return type("_Info", (), {"sha": exact})

    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    spec = _job_from_dict({"model": "Qwen/Qwen3.8-27B", "algorithm": algorithm})

    resolved = _resolve_model_revision(spec, required=algorithm == "sft")

    assert asked_for == [("Qwen/Qwen3.8-27B", exact)]
    assert resolved.model_revision == exact
    assert resolved.model_revision_auto is True
    assert resolved.model_revision_force_pin is False


def test_qwen38_catalog_revision_rejects_inherited_qwen36_pin() -> None:
    from flash.runner.lifecycle.preparation import _resolve_model_revision

    inherited = _job_from_dict(
        {
            "model": "Qwen/Qwen3.8-27B",
            "algorithm": "grpo",
            "model_revision": "a" * 40,
            "model_revision_auto": True,
        }
    )
    with pytest.raises(ValueError, match="requires immutable revision"):
        _resolve_model_revision(inherited)


def test_unmanaged_model_revision_is_rejected_and_runner_pin_is_unchanged() -> None:
    from flash.runner.lifecycle.preparation import _resolve_model_revision

    with pytest.raises(ValueError, match="model_revision requires model_revision_auto=True"):
        _job_from_dict({"model_revision": "release-tag"})
    runner_pin = _job_from_dict({"model_revision": "d" * 40, "model_revision_auto": True})

    assert _resolve_model_revision(runner_pin, required=False) == runner_pin
    assert _resolve_model_revision(runner_pin, required=True) == runner_pin


def test_removing_model_revision_from_public_specs_keeps_new_digests_stable() -> None:
    from flash.runner.lifecycle.preparation import _preparation_digest

    public = spec_from_dict(_raw(model="Qwen/Qwen3.5-9B", algorithm="sft"))
    worker = replace(public, model_revision="a" * 40, model_revision_auto=True)
    stored = public.to_dict()

    assert "model_revision" not in stored
    assert _preparation_digest(public, worker, None) == _preparation_digest(
        JobSpec.from_dict(stored), worker, None
    )


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
    from flash.runner.lifecycle.preparation import _validate_effective_spec

    public = spec_from_dict(_raw(model="Qwen/Qwen3.5-9B", algorithm="sft"))
    worker = replace(public, model_revision="a" * 40, model_revision_auto=True)
    reparsed_public = JobSpec.from_dict(worker.to_dict())
    assert reparsed_public.model_revision_auto is False
    assert reparsed_public.model_revision == ""
    assert worker.model_revision == "a" * 40

    _validate_effective_spec(reparsed_public, worker)  # raises if the exclusion is missing


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


def _fresh_orchestrator(tmp_path, monkeypatch) -> None:
    from tests._helpers.runner import fresh_runner

    fresh_runner(tmp_path, monkeypatch)


def test_runs_file_path_rejects_traversal(tmp_path, monkeypatch) -> None:
    _fresh_orchestrator(tmp_path, monkeypatch)
    for bad in ("../escape", "a/b", "", "x" * 200, ".hidden"):
        with pytest.raises(ValueError, match="invalid run_id"):
            runner_state.runs_file_path(bad, ".json")
    good = runner_state.runs_file_path("flash-123-abc", ".log")
    assert good.endswith("flash-123-abc.log")


def test_dry_run_submit_get_list_logs_cancel(tmp_path, monkeypatch) -> None:
    _fresh_orchestrator(tmp_path, monkeypatch)
    spec = spec_from_dict(_raw())

    status = runner_submit.submit_job(spec, dry_run=True)
    assert status.state == "dry_run"
    assert runner_status.get_status(status.run_id).state == "dry_run"
    assert status.run_id in [r.run_id for r in runner_status.list_runs()]
    assert runner_status.get_logs(status.run_id) == ""  # no log yet, no crash

    # terminal runs cancel as a no-op (state preserved)
    assert runner_deploy.cancel_run(status.run_id).state == "dry_run"

    with pytest.raises(FileNotFoundError, match="unknown run_id"):
        runner_status.get_status("flash-000-nope")


def test_programmatic_sft_submit_fails_closed_without_a_profilable_environment(
    tmp_path, monkeypatch
) -> None:
    # sft is quoted from a workload profile that tokenizes the real dataset, so a spec with no
    # environment to profile has no measurable workload. it must fail closed rather than fall back
    # to an assumed row count -- including on the dry-run preview, which previews a real submit.
    from flash.core.spec import JobSpec

    _fresh_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner_preparation,
        "_resolve_model_revision",
        lambda s, **_kw: replace(s, model_revision="a" * 40, model_revision_auto=True),
    )
    spec = JobSpec(
        run_id="sft-no-environment",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        project="11111111-1111-4111-8111-111111111111",
    )
    with pytest.raises(
        runner_preparation.WorkloadProfileUnavailable, match="requires an environment id"
    ):
        runner_submit.submit_job(spec, dry_run=True)
    with pytest.raises(FileNotFoundError):
        runner_status.get_status(spec.run_id)


@pytest.mark.parametrize("algorithm", ["sft", "grpo", "opd"])
def test_adapter_continuation_preparation_is_target_algorithm_agnostic(
    tmp_path, monkeypatch, algorithm
) -> None:
    """Preparation diagnoses the SOURCE run, never the target algorithm.

    SFT used to be refused outright here. Now all three targets reach the same source resolution,
    so an unknown source produces one identical message rather than SFT failing earlier and for a
    different reason.
    """
    from flash.core.spec import JobSpec, TrainSpec

    _fresh_orchestrator(tmp_path, monkeypatch)
    spec = JobSpec(
        run_id=f"{algorithm}-warmstart",
        model="Qwen/Qwen3.5-9B",
        algorithm=algorithm,
        project="11111111-1111-4111-8111-111111111111",
        train=TrainSpec(epochs=1, max_examples=8, init_from_adapter="source-run/final"),
    )

    with pytest.raises(ValueError, match="references unknown run 'source-run'"):
        runner_preparation._prepare_init_from_adapter(spec)


def test_artifacts_dir_and_adapter_prefix_helpers(tmp_path, monkeypatch) -> None:
    _fresh_orchestrator(tmp_path, monkeypatch)
    spec = spec_from_dict(_raw(), run_id="flash-1-x")
    assert runner_state.artifacts_dir(spec).endswith(
        os.path.join("results", "runpod", "rl", "flash-1-x")
    )
    assert runner_state.adapter_prefix(spec) == "rl/flash-1-x"
    assert runner_state.adapter_ref(spec) is None

    # hf_repo and run_id are platform-managed: they survive the INTERNAL round trip
    # (to_internal_dict -> from_dict), which is what the worker/control plane use, not to_dict().
    d = spec.to_internal_dict()
    d["train"] = {**d["train"], "hf_repo": "Freesolo-Co/flashrun-flash-1-x"}
    spec_with_repo = _job_from_dict(d)
    assert runner_state.adapter_ref(spec_with_repo) == "Freesolo-Co/flashrun-flash-1-x:rl/flash-1-x"


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
        'model = "Qwen/Qwen3.5-9B"\nalgorithm = "grpo"\n[worker_env]\nCUSTOM_FLAG = "value"\n'
    )

    with pytest.raises(ConfigError) as exc_info:
        spec_and_train_keys_from_file(str(path))

    message = str(exc_info.value)
    assert "unknown config section(s): worker_env" in message
    assert "(allowed tables: environment, train, gpu, wandb)" in message


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


@pytest.mark.parametrize("algorithm", ["grpo", "opd"])
def test_persisted_rl_batch_size_is_rejected_and_names_the_right_key(algorithm) -> None:
    record = JobSpec(algorithm=algorithm).to_internal_dict()
    record["train"]["batch_size"] = 16
    record["train"].pop("prompts_per_step", None)

    with pytest.raises(ValueError, match=r"batch_size.*prompts_per_step") as excinfo:
        JobSpec.from_dict(record)

    message = str(excinfo.value)
    assert f"batch_size does not apply to {algorithm}" in message
    assert "prompts_per_step" in message


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
