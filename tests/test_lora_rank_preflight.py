from __future__ import annotations

from dataclasses import replace

import pytest

from flash.lora_rank import preflight_init_adapter_lora_rank, resolve_adapter_ref
from flash.schema import spec_from_dict

_ADAPTER_REF = "owner/runs:sft/sft-run"


def _spec(*, model: str = "Qwen/Qwen3.5-4B", rank: int = 16, algorithm: str = "grpo"):
    train = {
        "epochs": 1,
        "max_examples": 8,
        "lora_rank": rank,
    }
    spec = spec_from_dict(
        {
            "model": model,
            "algorithm": algorithm,
            "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
            "train": train,
        }
    )
    return replace(spec, train=replace(spec.train, init_from_adapter=_ADAPTER_REF))


def _loader(config):
    return lambda adapter_ref, token: config


def test_init_adapter_preflight_rejects_adapter_rank_above_serving_cap():
    # Qwen3.5-4B serving cap is now max_lora_rank=64 (doubled from 32); a rank-96 adapter still exceeds it.
    spec = _spec(rank=16)

    with pytest.raises(ValueError, match=r"has rank 96.*serving max_lora_rank=64"):
        preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 96}))


def test_init_adapter_preflight_allows_empty_vl_patterns():
    spec = _spec(rank=16)

    preflight_init_adapter_lora_rank(
        spec,
        config_loader=_loader({"r": 16, "rank_pattern": {}, "alpha_pattern": {}}),
    )


@pytest.mark.parametrize("value", [0, "", [], False])
def test_init_adapter_preflight_rejects_falsey_invalid_rank_pattern(value):
    # rank_from_adapter_config rejects a non-Mapping rank_pattern (falsey-but-not-None values are the
    # tricky cases); alpha_pattern is no longer inspected by the single-adapter rank preflight.
    spec = _spec(rank=16)

    with pytest.raises(ValueError, match="rank_pattern"):
        preflight_init_adapter_lora_rank(
            spec,
            config_loader=_loader({"r": 16, "rank_pattern": value}),
        )


def test_init_adapter_preflight_checks_adapter_rank_for_sft_warm_start():
    # Qwen3.5-0.8B serving cap is now max_lora_rank=128 (doubled from 64); a rank-160 adapter exceeds it.
    spec = _spec(model="Qwen/Qwen3.5-0.8B", rank=32, algorithm="sft")

    with pytest.raises(ValueError, match=r"has rank 160.*serving max_lora_rank=128"):
        preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 160}))


def test_lora_rank_uses_schema_adapter_storage_ref_parser():
    assert resolve_adapter_ref("owner/runs:sft/source-run/checkpoints/step-40") == (
        "owner/runs",
        "sft/source-run/checkpoints/step-40",
    )
