from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from flash.adapters.lora_rank import (
    adapter_artifact_identity,
    alpha_from_adapter_config,
    inspect_adapter_config,
    preflight_init_adapter_lora_rank,
    rank_from_adapter_config,
    resolve_adapter_ref,
)
from flash.schema import spec_from_dict

_ADAPTER_REF = "owner/runs:sft/sft-run"


def _spec(*, rank: int = 16, model: str = "Qwen/Qwen3.5-9B"):
    # a warm-start child cannot author rank or alpha, so the child spec only carries the rank
    # default; the adapter-side alpha the preflight inspects comes from the loaded
    # adapter_config.json, independent of the child spec.
    spec = spec_from_dict(
        {
            "model": model,
            "algorithm": "grpo",
            "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
            "train": {
                "epochs": 1,
                "max_examples": 8,
                "lora_rank": rank,
            },
        }
    )
    return replace(spec, train=replace(spec.train, init_from_adapter=_ADAPTER_REF))


def _config(**overrides):
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "Qwen/Qwen3.5-9B",
        "r": 16,
        "lora_alpha": 32,
        # every peft>=0.19 save carries this key, null for a multimodal run, so a config without it
        # is not a shape any supported writer produces. present here because submit rejects an
        # unmarked adapter outright: omitting it would make these rank/alpha cases exercise that
        # rejection instead of what they are about.
        "exclude_modules": None,
    }
    config.update(overrides)
    return config


def test_rank_and_alpha_use_maximum_pattern_values():
    config = _config(
        rank_pattern={"a": 64, "b": 8},
        alpha_pattern={"a": 128, "b": 16},
    )
    assert rank_from_adapter_config(config, source="adapter") == 64
    assert alpha_from_adapter_config(config, source="adapter") == 128


def test_preflight_accepts_child_rank_and_alpha_mismatches():
    metadata = preflight_init_adapter_lora_rank(
        _spec(rank=8),
        config_loader=lambda _ref, _token, _revision: _config(r=64, lora_alpha=128),
    )
    assert metadata is not None
    assert metadata.rank == 64
    assert metadata.alpha == 128


def test_preflight_rejects_adapter_rank_above_serving_cap():
    from flash.core.catalog import serving_lora_rank_cap

    model = "Qwen/Qwen3.5-9B"
    # read the active hosted model's cap so catalog changes cannot make the rejection case fit.
    cap = serving_lora_rank_cap(model)
    assert cap is not None
    adapter_rank = cap + 1
    with pytest.raises(ValueError, match=rf"rank {adapter_rank}.*serving max_lora_rank={cap}"):
        preflight_init_adapter_lora_rank(
            _spec(model=model),
            config_loader=lambda _ref, _token, _revision: _config(
                r=adapter_rank, base_model_name_or_path=model
            ),
        )


@pytest.mark.parametrize("adapter_rank", [64, 65])
def test_qwen38_customer_serving_rank_envelope(adapter_rank: int):
    from flash.core.catalog import serving_lora_rank_cap

    model = "Qwen/Qwen3.8-27B"
    assert serving_lora_rank_cap(model) == 64
    if adapter_rank == 64:
        metadata = preflight_init_adapter_lora_rank(
            _spec(model=model),
            config_loader=lambda _ref, _token, _revision: _config(
                r=adapter_rank,
                lora_alpha=128,
                base_model_name_or_path=model,
            ),
        )
        assert metadata is not None
        assert metadata.rank == adapter_rank
    else:
        with pytest.raises(ValueError, match=r"rank 65.*serving max_lora_rank=64"):
            preflight_init_adapter_lora_rank(
                _spec(model=model),
                config_loader=lambda _ref, _token, _revision: _config(
                    r=adapter_rank,
                    lora_alpha=130,
                    base_model_name_or_path=model,
                ),
            )


@pytest.mark.parametrize("field", ["rank_pattern", "alpha_pattern"])
@pytest.mark.parametrize("value", [0, "", [], False])
def test_preflight_rejects_invalid_patterns(field, value):
    with pytest.raises(ValueError, match=field):
        preflight_init_adapter_lora_rank(
            _spec(), config_loader=lambda _ref, _token, _revision: _config(**{field: value})
        )


@pytest.mark.parametrize("value", [None, "", "IA3"])
def test_inspection_requires_lora_peft_type(value):
    with pytest.raises(ValueError, match="peft_type must be LORA"):
        inspect_adapter_config(
            _config(peft_type=value), source="adapter", target_model="Qwen/Qwen3.5-9B"
        )


def test_inspection_rejects_incompatible_task_type():
    with pytest.raises(ValueError, match="task_type must be CAUSAL_LM"):
        inspect_adapter_config(
            _config(task_type="SEQ_CLS"), source="adapter", target_model="Qwen/Qwen3.5-9B"
        )


def test_inspection_rejects_incompatible_base_model():
    with pytest.raises(ValueError, match=r"base model.*does not match target model"):
        inspect_adapter_config(
            _config(base_model_name_or_path="Qwen/Qwen3.8-27B"),
            source="adapter",
            target_model="Qwen/Qwen3.5-9B",
        )


def test_inspection_requires_the_adapter_to_name_its_base_model():
    # a blank base model must not read as "no opinion": that made the base-model comparison below
    # skip itself, so an adapter trained on a DIFFERENT base passed preflight and was inherited
    # into the run. every flash-published adapter is stamped by the exporter, so a blank value is
    # an artifact that predates it and must fail loudly rather than silently disable the check.
    with pytest.raises(ValueError, match="does not name its base model"):
        inspect_adapter_config(
            _config(base_model_name_or_path=""),
            source="adapter",
            target_model="Qwen/Qwen3.5-9B",
        )


def test_inspection_requires_a_causal_lm_task_type():
    # same failure direction: a blank task_type skipped the check instead of failing it.
    with pytest.raises(ValueError, match="task_type must be CAUSAL_LM"):
        inspect_adapter_config(
            _config(task_type=""),
            source="adapter",
            target_model="Qwen/Qwen3.5-9B",
        )


def test_inspection_requires_alpha_metadata():
    config = _config()
    config.pop("lora_alpha")
    with pytest.raises(ValueError, match="no LoRA alpha metadata"):
        inspect_adapter_config(config, source="adapter", target_model="Qwen/Qwen3.5-9B")


@pytest.mark.parametrize(
    "value",
    [True, False, 1.5, float("nan"), float("inf"), [], {}, "1.5", "nan", "inf", "x"],
)
def test_rank_metadata_rejects_non_integral_or_malformed_values(value):
    with pytest.raises(ValueError, match="invalid r"):
        rank_from_adapter_config(_config(r=value), source="adapter")


@pytest.mark.parametrize("value", [1, 1.0, "1", "+1", "1.0", "1e0"])
def test_rank_metadata_accepts_deliberate_integral_representations(value):
    assert rank_from_adapter_config(_config(r=value), source="adapter") == 1


def test_rank_metadata_parses_large_integral_decimal_string_exactly():
    assert (
        rank_from_adapter_config(_config(r="9007199254740993.0"), source="adapter")
        == 9007199254740993
    )


def test_rank_metadata_rejects_inexact_decimal_string():
    with pytest.raises(ValueError, match="invalid r"):
        rank_from_adapter_config(_config(r="1.0000000000000001"), source="adapter")


@pytest.mark.parametrize("value", [Decimal("1"), Decimal("1.0"), Decimal("1E0")])
def test_rank_metadata_accepts_integral_decimal_values(value):
    # load_hf_adapter_config parses json floats as decimal, so decimal instances reach the parser.
    assert rank_from_adapter_config(_config(r=value), source="adapter") == 1


@pytest.mark.parametrize(
    "value",
    [Decimal("1.5"), Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity"), Decimal("-Infinity")],
)
def test_rank_metadata_rejects_non_integral_decimal_values(value):
    with pytest.raises(ValueError, match="invalid r"):
        rank_from_adapter_config(_config(r=value), source="adapter")


@pytest.mark.parametrize("value", [Decimal("0"), Decimal("-1")])
def test_rank_metadata_rejects_non_positive_decimal_values(value):
    with pytest.raises(ValueError, match="non-positive r"):
        rank_from_adapter_config(_config(r=value), source="adapter")


_FLOAT_CONFIG_JSON = (
    '{"peft_type": "LORA", "task_type": "CAUSAL_LM",'
    ' "base_model_name_or_path": "Qwen/Qwen3.5-9B",'
    ' "r": 16.0, "lora_alpha": 64.0, "lora_dropout": 0.05}'
)


def _parsed_config(text: str = _FLOAT_CONFIG_JSON):
    # mirror load_hf_adapter_config: floats arrive as decimal, not float.
    return json.loads(text, parse_float=Decimal)


def test_preflight_parses_decimal_typed_topology_from_real_json():
    config = _parsed_config()
    assert isinstance(config["r"], Decimal)
    assert isinstance(config["lora_alpha"], Decimal)
    metadata = preflight_init_adapter_lora_rank(
        _spec(), config_loader=lambda _ref, _token, _revision: config
    )
    assert metadata is not None
    assert metadata.rank == 16
    assert metadata.alpha == 64


def test_adapter_identity_binds_config_and_weight_metadata(monkeypatch):
    import huggingface_hub

    state = {"oid": "sha256:weights-v1"}

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_tree(self, repo_id, path_in_repo, repo_type, recursive=False, revision=None):
            assert repo_id == "owner/runs"
            assert repo_type == "dataset"
            assert revision is None
            return [
                SimpleNamespace(
                    path="sft/sft-run/adapter/adapter_model.safetensors",
                    blob_id=None,
                    size=123,
                    # attribute-style, as `list_repo_tree` really returns it: `RepoFile.__init__`
                    # builds a `BlobLfsInfo` dataclass, never a mapping.
                    lfs=SimpleNamespace(sha256=state["oid"], size=123),
                )
            ]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    first = adapter_artifact_identity(_ADAPTER_REF, _config(), token="token")
    second = adapter_artifact_identity(_ADAPTER_REF, _config(), token="token")
    assert first == second
    assert first.weight_filename == "adapter_model.safetensors"
    changed_config = adapter_artifact_identity(_ADAPTER_REF, _config(lora_alpha=64), token="token")
    assert changed_config.digest != first.digest
    state["oid"] = "sha256:weights-v2"
    changed_weight = adapter_artifact_identity(_ADAPTER_REF, _config(), token="token")
    assert changed_weight.digest != first.digest


def test_adapter_identity_digests_decimal_config_values_exactly(monkeypatch):
    import huggingface_hub

    class FakeApi:
        def __init__(self, token=None):
            self.token = token

        def list_repo_tree(self, repo_id, path_in_repo, repo_type, recursive=False, revision=None):
            return [
                SimpleNamespace(
                    path="sft/sft-run/adapter/adapter_model.safetensors",
                    blob_id=None,
                    size=123,
                    lfs=SimpleNamespace(sha256="sha256:weights-v1", size=123),
                )
            ]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    first = adapter_artifact_identity(_ADAPTER_REF, _parsed_config(), token="token")
    second = adapter_artifact_identity(_ADAPTER_REF, _parsed_config(), token="token")
    assert first == second
    changed_float = adapter_artifact_identity(
        _ADAPTER_REF,
        _parsed_config(_FLOAT_CONFIG_JSON.replace("0.05", "0.10")),
        token="token",
    )
    assert changed_float.digest != first.digest
    # a json string spelling the same numeral must not collide with the decimal value.
    string_typed = adapter_artifact_identity(
        _ADAPTER_REF,
        _parsed_config(
            _FLOAT_CONFIG_JSON.replace('"lora_dropout": 0.05', '"lora_dropout": "0.05"')
        ),
        token="token",
    )
    assert string_typed.digest != first.digest
    magic_object = adapter_artifact_identity(
        _ADAPTER_REF,
        _parsed_config(
            _FLOAT_CONFIG_JSON.replace(
                '"lora_dropout": 0.05',
                '"lora_dropout": {"__decimal__": "0.05"}',
            )
        ),
        token="token",
    )
    assert magic_object.digest != first.digest
    assert magic_object.config_sha256 != first.config_sha256

    integer_config = _config()
    integer_identity = adapter_artifact_identity(_ADAPTER_REF, integer_config, token="token")
    legacy_bytes = json.dumps(
        integer_config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert integer_identity.config_sha256 == hashlib.sha256(legacy_bytes).hexdigest()


def test_lora_rank_uses_schema_adapter_storage_ref_parser():
    assert resolve_adapter_ref("owner/runs:sft/source-run/checkpoints/step-40") == (
        "owner/runs",
        "sft/source-run/checkpoints/step-40",
    )


_FUSED_MODEL = "Qwen/Qwen3.6-35B-A3B"
_FUSED_TARGETS = [
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
]
_MISSING = object()


@pytest.mark.parametrize("targets", [_FUSED_TARGETS, list(reversed(_FUSED_TARGETS))])
def test_fused_expert_config_accepts_exact_target_order_variants(targets):
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config

    validate_fused_expert_adapter_config(
        {"r": 16, "target_parameters": targets, "target_modules": ["q_proj"]},
        _FUSED_MODEL,
    )


@pytest.mark.parametrize(
    "targets",
    [
        pytest.param(_MISSING, id="missing"),
        pytest.param(None, id="null"),
        pytest.param([], id="empty"),
        pytest.param("mlp.experts.gate_up_proj", id="string"),
        pytest.param(["mlp.experts.gate_up_proj"], id="partial"),
        pytest.param([*_FUSED_TARGETS, "mlp.router"], id="extra"),
        pytest.param([_FUSED_TARGETS[0], _FUSED_TARGETS[0]], id="duplicate"),
        pytest.param([_FUSED_TARGETS[0], 7], id="non-string"),
    ],
)
def test_fused_expert_config_rejects_noncanonical_target_parameters(targets):
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config

    config = {"r": 16, "target_modules": ["q_proj"]}
    if targets is not _MISSING:
        config["target_parameters"] = targets
    with pytest.raises(ValueError, match=r"target_parameters|fused expert targets"):
        validate_fused_expert_adapter_config(config, _FUSED_MODEL)


@pytest.mark.parametrize(
    "modules",
    [
        pytest.param("all-linear", id="string"),
        pytest.param(["q_proj", "v_proj"], id="list"),
    ],
)
def test_fused_expert_config_accepts_supported_target_module_shapes(modules):
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config

    validate_fused_expert_adapter_config(
        {
            "r": 16,
            "target_parameters": list(_FUSED_TARGETS),
            "target_modules": modules,
        },
        _FUSED_MODEL,
    )


@pytest.mark.parametrize(
    "modules",
    [
        pytest.param(_MISSING, id="missing"),
        pytest.param(None, id="null"),
    ],
)
def test_fused_expert_config_rejects_missing_ordinary_targets(modules):
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config

    config = {"r": 16, "target_parameters": list(_FUSED_TARGETS)}
    if modules is not _MISSING:
        config["target_modules"] = modules
    with pytest.raises(ValueError, match="target_modules"):
        validate_fused_expert_adapter_config(config, _FUSED_MODEL)


@pytest.mark.parametrize(
    "modules",
    [
        pytest.param("experts", id="synthetic-string-experts"),
        pytest.param("base_layer", id="synthetic-string-base-layer"),
        pytest.param(r".*\.mlp\.experts", id="synthetic-regex"),
        pytest.param("[", id="invalid-regex"),
        pytest.param(["q_proj", "experts"], id="synthetic-list-experts"),
        pytest.param(["base_layer", "q_proj"], id="synthetic-list-base-layer"),
        pytest.param(["mlp.experts"], id="synthetic-list-owner"),
        pytest.param(["experts.base_layer"], id="synthetic-list-nested"),
        pytest.param(["model.layers.0.mlp.experts"], id="synthetic-list-qualified"),
        pytest.param([], id="empty-list"),
        pytest.param(["q_proj", ""], id="empty-list-entry"),
        pytest.param(7, id="integer"),
        pytest.param({"q_proj"}, id="set"),
        pytest.param(("q_proj",), id="tuple"),
        pytest.param(["q_proj", 7], id="non-string-list-entry"),
    ],
)
def test_fused_expert_config_rejects_synthetic_or_malformed_target_modules(modules):
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config

    with pytest.raises(ValueError, match="target_modules"):
        validate_fused_expert_adapter_config(
            {
                "r": 16,
                "target_parameters": list(_FUSED_TARGETS),
                "target_modules": modules,
            },
            _FUSED_MODEL,
        )


def test_fused_expert_config_accepts_per_target_rank_patterns_and_scalar_fallback():
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config

    validate_fused_expert_adapter_config(
        {
            "r": 16,
            "target_parameters": list(_FUSED_TARGETS),
            "target_modules": ["q_proj"],
            "rank_pattern": {"mlp.experts.gate_up_proj": 8},
        },
        _FUSED_MODEL,
    )


def test_fused_expert_config_accepts_overrides_for_every_declared_target():
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config

    validate_fused_expert_adapter_config(
        {
            "target_parameters": list(_FUSED_TARGETS),
            "target_modules": ["q_proj", "v_proj"],
            "rank_pattern": {
                "mlp.experts.gate_up_proj": 8,
                "mlp.experts.down_proj": 4,
                "q_proj": 16,
                "v_proj": 8,
            },
        },
        _FUSED_MODEL,
    )


@pytest.mark.parametrize(
    "rank_pattern",
    [
        pytest.param([], id="list"),
        pytest.param("mlp.experts", id="string"),
        pytest.param({"": 16}, id="empty-pattern"),
        pytest.param({"mlp.experts": 0}, id="zero-rank"),
        pytest.param({"mlp.experts": -1}, id="negative-rank"),
        pytest.param({"mlp.experts": True}, id="bool-rank"),
        pytest.param({"mlp.experts": 1.5}, id="float-rank"),
    ],
)
def test_fused_expert_config_rejects_malformed_rank_patterns(rank_pattern):
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config

    with pytest.raises(ValueError, match="rank_pattern"):
        validate_fused_expert_adapter_config(
            {
                "r": 16,
                "target_parameters": list(_FUSED_TARGETS),
                "target_modules": ["q_proj"],
                "rank_pattern": rank_pattern,
            },
            _FUSED_MODEL,
        )


def test_fused_expert_config_rejects_malformed_rank_pattern_regex():
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config

    with pytest.raises(ValueError, match="rank_pattern"):
        validate_fused_expert_adapter_config(
            {
                "r": 16,
                "target_parameters": list(_FUSED_TARGETS),
                "target_modules": ["q_proj"],
                "rank_pattern": {"[": 8},
            },
            _FUSED_MODEL,
        )


def test_fused_expert_config_rejects_unresolved_fused_target_rank():
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config

    with pytest.raises(ValueError, match="no resolved LoRA rank"):
        validate_fused_expert_adapter_config(
            {
                "target_parameters": list(_FUSED_TARGETS),
                "target_modules": ["q_proj"],
                "rank_pattern": {"mlp.experts.gate_up_proj": 8},
            },
            _FUSED_MODEL,
        )


def test_fused_expert_config_rejects_unresolved_ordinary_target_rank():
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config

    with pytest.raises(ValueError, match="ordinary target_modules"):
        validate_fused_expert_adapter_config(
            {
                "target_parameters": list(_FUSED_TARGETS),
                "target_modules": ["q_proj"],
                "rank_pattern": {
                    "mlp.experts.gate_up_proj": 8,
                    "mlp.experts.down_proj": 4,
                },
            },
            _FUSED_MODEL,
        )


def test_fused_expert_config_is_a_noop_for_non_fused_models():
    from flash.adapters.fused_experts import validate_fused_expert_adapter_config

    malformed = {"target_parameters": None, "target_modules": {"experts"}}
    validate_fused_expert_adapter_config(malformed, "Qwen/Qwen3.5-9B")


def _patch_fused_submit_preflight(
    monkeypatch, config, *, reject_config, child_rank=16, resolved_rank=16
):
    import flash.adapters.fused_experts as fused_experts
    import flash.adapters.lora_rank as lora_rank
    import flash.runner.lifecycle.preparation as preparation
    import flash.runner.results.checkpoints as checkpoints

    target_spec = _spec(rank=child_rank, model=_FUSED_MODEL)
    target_spec = replace(
        target_spec,
        train=replace(target_spec.train, init_from_adapter="source-run/final"),
    )
    source_spec = replace(
        target_spec,
        train=replace(
            target_spec.train,
            init_from_adapter="",
            hf_repo="owner/runs",
        ),
    )
    status = SimpleNamespace(state="done")
    events = []

    def load_config(_ref, _token, _revision):
        events.append(("load", config))
        return config

    def validate_config(seen, model_id):
        assert seen is config
        assert model_id == _FUSED_MODEL
        events.append(("validate", seen))
        if reject_config:
            raise ValueError("invalid fused config")

    def preflight(spec, *, token, config_loader):
        assert spec.model == _FUSED_MODEL
        seen = config_loader("unused", token, "unused")
        assert seen is config
        events.append(("preflight", seen))
        return SimpleNamespace(rank=resolved_rank, alpha=2 * resolved_rank)

    monkeypatch.setattr(preparation.status_ops, "get_status", lambda _run_id: status)
    monkeypatch.setattr(
        preparation,
        "_warmstart_source_is_authorized",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        preparation.status_ops,
        "effective_spec_from_status",
        lambda _status: source_spec,
    )
    monkeypatch.setattr(preparation, "_adopted_warmstart_revision", lambda spec, _source: spec)
    monkeypatch.setattr(checkpoints, "adapter_artifact_exists", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(lora_rank, "resolve_hf_dataset_revision", lambda *_args: "revision")
    monkeypatch.setattr(lora_rank, "load_hf_adapter_config", load_config)
    monkeypatch.setattr(lora_rank, "preflight_init_adapter_lora_rank", preflight)
    monkeypatch.setattr(
        lora_rank,
        "adapter_artifact_identity",
        lambda *_args, **_kwargs: SimpleNamespace(to_dict=dict),
    )
    monkeypatch.setattr(fused_experts, "validate_fused_expert_adapter_config", validate_config)
    return preparation, target_spec, events


def test_submit_rejects_fused_config_before_rank_preflight(monkeypatch):
    config = _config(target_parameters=None)
    preparation, target_spec, events = _patch_fused_submit_preflight(
        monkeypatch, config, reject_config=True
    )

    with pytest.raises(ValueError, match="invalid fused config"):
        preparation._prepare_init_from_adapter_inner(target_spec)

    assert events == [("load", config), ("validate", config)]


def test_submit_rejects_an_unmarked_adapter_before_any_gpu_is_allocated(monkeypatch):
    """An adapter with no modality marker must fail at submit, not on the rented GPU.

    The marker decides which module surface the run trains, so an unmarked source is unusable by
    every algorithm -- the answer never depends on anything only the worker knows. The worker still
    re-checks the bytes it downloads; what this pins is that the control plane, which already holds
    this exact config, does not defer a decision it can make for free into a paid allocation.
    """
    config = _config()
    del config["exclude_modules"]
    preparation, target_spec, events = _patch_fused_submit_preflight(
        monkeypatch, config, reject_config=False
    )

    with pytest.raises(ValueError, match="required exclude_modules modality marker"):
        preparation._prepare_init_from_adapter_inner(target_spec)

    # rejected on the loaded config alone: nothing downstream ran, so no later stage can be what
    # caught it and no allocation could have happened first.
    assert events == [("load", config)]


def test_submit_passes_the_loaded_config_to_validation_then_rank_preflight(monkeypatch):
    config = _config(target_parameters=list(_FUSED_TARGETS))
    preparation, target_spec, events = _patch_fused_submit_preflight(
        monkeypatch, config, reject_config=False
    )

    preparation._prepare_init_from_adapter_inner(target_spec)

    assert events == [
        ("load", config),
        ("validate", config),
        ("preflight", config),
    ]


def test_preparation_returns_resolved_source_rank_on_worker_spec(monkeypatch):
    from flash.providers.core.allocator import required_vram_gb

    config = _config(target_parameters=list(_FUSED_TARGETS))
    preparation, target_spec, _events = _patch_fused_submit_preflight(
        monkeypatch,
        config,
        reject_config=False,
        child_rank=32,
        resolved_rank=4,
    )

    _public_spec, worker_spec, _identity, _source_context = (
        preparation._prepare_init_from_adapter_inner(target_spec)
    )

    assert target_spec.train.lora_rank == 32
    assert worker_spec.train.lora_rank == 4
    assert required_vram_gb(
        worker_spec.model, worker_spec.algorithm, train=worker_spec.train
    ) < required_vram_gb(target_spec.model, target_spec.algorithm, train=target_spec.train)
