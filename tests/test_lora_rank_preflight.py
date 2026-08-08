from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from flash.lora_rank import (
    adapter_artifact_identity,
    alpha_from_adapter_config,
    inspect_adapter_config,
    preflight_init_adapter_lora_rank,
    rank_from_adapter_config,
    resolve_adapter_ref,
)
from flash.schema import spec_from_dict

_ADAPTER_REF = "owner/runs:sft/sft-run"


def _spec(*, rank: int = 16, model: str = "Qwen/Qwen3.5-4B"):
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
        "base_model_name_or_path": "Qwen/Qwen3.5-4B",
        "r": 16,
        "lora_alpha": 32,
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


@pytest.mark.parametrize("model", ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-27B"])
def test_preflight_rejects_adapter_rank_above_serving_cap(model):
    from flash.catalog import serving_lora_rank_cap

    # each tier's own cap, read from the catalog. these were pinned to a shared literal 64, which
    # silently stopped testing the 4B once its cap rose to 128 -- rank 96 then FITS.
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
            _config(peft_type=value), source="adapter", target_model="Qwen/Qwen3.5-4B"
        )


def test_inspection_rejects_incompatible_task_type():
    with pytest.raises(ValueError, match="task_type must be CAUSAL_LM"):
        inspect_adapter_config(
            _config(task_type="SEQ_CLS"), source="adapter", target_model="Qwen/Qwen3.5-4B"
        )


def test_inspection_rejects_incompatible_base_model():
    with pytest.raises(ValueError, match=r"base model.*does not match target model"):
        inspect_adapter_config(
            _config(base_model_name_or_path="Qwen/Qwen3.5-0.8B"),
            source="adapter",
            target_model="Qwen/Qwen3.5-4B",
        )


def test_inspection_allows_empty_optional_compatibility_fields():
    metadata = inspect_adapter_config(
        _config(base_model_name_or_path="", task_type=""),
        source="adapter",
        target_model="Qwen/Qwen3.5-4B",
    )
    assert metadata.rank == 16
    assert metadata.alpha == 32


def test_inspection_requires_alpha_metadata():
    config = _config()
    config.pop("lora_alpha")
    with pytest.raises(ValueError, match="no LoRA alpha metadata"):
        inspect_adapter_config(config, source="adapter", target_model="Qwen/Qwen3.5-4B")


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
    ' "base_model_name_or_path": "Qwen/Qwen3.5-4B",'
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

        def get_paths_info(self, repo_id, paths, repo_type, revision=None):
            assert repo_id == "owner/runs"
            assert repo_type == "dataset"
            assert revision is None
            return [
                SimpleNamespace(
                    path="sft/sft-run/adapter/adapter_model.safetensors",
                    blob_id=None,
                    size=123,
                    lfs={"sha256": state["oid"], "size": 123},
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

        def get_paths_info(self, repo_id, paths, repo_type, revision=None):
            return [
                SimpleNamespace(
                    path="sft/sft-run/adapter/adapter_model.safetensors",
                    blob_id=None,
                    size=123,
                    lfs={"sha256": "sha256:weights-v1", "size": 123},
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
