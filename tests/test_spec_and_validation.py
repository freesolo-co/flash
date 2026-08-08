"""Spec plumbing not covered elsewhere: config validation error paths, JobSpec
serialization round-trips, worker env loading, run-id path containment, VRAM
estimates, and logging namespace helpers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import fields, replace

import pytest

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
from flash.spec import GpuSpec, JobSpec, TrainSpec, load_job_spec_from_env

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
    assert TRAIN_KEY_MIN_VERSIONS["lora_alpha"] == "1.1.31"
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
    # current adds save_at_steps, credit_assignment, and entropy_quantile; opd eos is removed.
    # lora_alpha is in both: authorable then, managed in between, and a user knob again now.
    assert historical_shapes["861571e7"] - {
        "opd_eos_loss_coef",
        "hf_repo",
        "advantage_clip",
    } == TRAIN_SCHEMA_KEYS - {
        "credit_assignment",
        "save_at_steps",
        "entropy_quantile",
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
    from flash.catalog import serving_lora_rank_cap

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


def test_environment_pip_is_platform_managed() -> None:
    raw = _raw()
    raw["environment"]["pip"] = ["freesolo==1.2.3"]
    with pytest.raises(ConfigError, match=r"\[environment\] unknown key\(s\): pip"):
        spec_from_dict(raw)


def test_submit_payload_round_trips_without_a_pip_key() -> None:
    """spec_payload is what the CLI actually sends, and the server re-parses it with this parser."""
    from flash.client.specs import spec_payload

    spec = spec_from_dict(_raw())
    payload = spec_payload(spec, authored_train_keys=frozenset({"epochs"}))

    assert "pip" not in payload["environment"]
    assert spec_from_dict(payload).environment.id == spec.environment.id


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
        "FLASH_CONTROL_PANEL_URL",
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
    # pip is platform-managed: never authored, so it stays at its default here.
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


def test_model_revision_strips_round_trips_and_rejects_non_strings() -> None:
    from flash.client.specs import spec_payload

    spec = spec_from_dict(_raw(model_revision="  refs/pr/123  "), run_id="revision-rt")
    assert spec.model_revision == "refs/pr/123"
    assert spec_payload(spec)["model_revision"] == "refs/pr/123"
    assert JobSpec.from_json(spec.to_json()).model_revision == "refs/pr/123"
    assert spec_from_dict(_raw(model_revision="   ")).model_revision == ""

    for value in (None, 123, False, ["main"], {"revision": "main"}):
        with pytest.raises(ConfigError, match="model_revision must be a string"):
            spec_from_dict(_raw(model_revision=value))
        with pytest.raises(TypeError, match="model_revision must be a string"):
            _job_from_dict({"model_revision": value})


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
    from flash.spec import JobSpec

    orch = _fresh_orchestrator(tmp_path, monkeypatch)
    monkeypatch.setattr(
        orch, "_resolve_model_revision", lambda s, **_kw: replace(s, model_revision="a" * 40)
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
    from flash.spec import JobSpec, TrainSpec

    orch = _fresh_orchestrator(tmp_path, monkeypatch)
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
    from flash.spec import coerce_bool

    assert coerce_bool(value) is expected


# ---------------------------------------------------------------------------
# gpu.count (multi-gpu job spec)
# ---------------------------------------------------------------------------


def test_gpu_count_defaults_to_one() -> None:
    assert spec_from_dict(_raw()).gpu.count == 1
    assert GpuSpec().count == 1


def test_gpu_count_parses_and_roundtrips() -> None:
    parsed = spec_from_dict(_raw(**{"gpu.count": 4}))
    assert parsed.gpu.count == 4
    # count survives both serialization hops (asdict-based to_dict / to_json).
    assert _job_from_dict(parsed.to_dict()).gpu.count == 4
    assert JobSpec.from_json(parsed.to_json()).gpu.count == 4


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
    from flash.spec import gpu_count_of

    assert gpu_count_of(None) == 1  # no spec -> single gpu
    assert (
        gpu_count_of(JobSpec(project="11111111-1111-4111-8111-111111111111")) == 1
    )  # default gpu count
    assert (
        gpu_count_of(JobSpec(project="11111111-1111-4111-8111-111111111111", gpu=GpuSpec(count=3)))
        == 3
    )
