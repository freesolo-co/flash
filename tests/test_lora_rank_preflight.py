from __future__ import annotations

from dataclasses import replace

import pytest

from flash.lora_rank import preflight_init_adapter_lora_rank
from flash.schema import spec_from_dict

_ADAPTER_REF = "owner/runs:sft/sft-run"


def _spec(*, model: str = "Qwen/Qwen3.5-4B", rank: int = 16, algorithm: str = "grpo"):
    spec = spec_from_dict(
        {
            "model": model,
            "algorithm": algorithm,
            "environment": {"id": "github:freesolo-co/envs@main:gsm8k/environment.py"},
            "train": {
                "steps": 1,
                "lora_rank": rank,
            },
        }
    )
    return replace(spec, train=replace(spec.train, init_from_adapter=_ADAPTER_REF))


def _loader(config):
    return lambda adapter_ref, token: config


def test_init_adapter_preflight_rejects_adapter_rank_above_serving_cap():
    spec = _spec(rank=16)

    with pytest.raises(ValueError, match=r"has rank 96.*serving max_lora_rank=64"):
        preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 96}))


def test_init_adapter_preflight_rejects_vl_recombined_rank_before_training():
    spec = _spec(rank=16)

    with pytest.raises(
        ValueError,
        match=r"rank 72 \(SFT rank 56 \+ GRPO rank 16\).*set GRPO train\.lora_rank <= 8",
    ):
        preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 56}))


def test_init_adapter_preflight_allows_vl_recombined_rank_at_serving_cap():
    spec = _spec(rank=16)

    preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 48}))


def test_init_adapter_preflight_rejects_non_uniform_vl_rank_pattern():
    spec = _spec(rank=16)

    with pytest.raises(ValueError, match="rank_pattern"):
        preflight_init_adapter_lora_rank(
            spec,
            config_loader=_loader({"r": 16, "rank_pattern": {"q_proj": 8}}),
        )


def test_init_adapter_preflight_checks_adapter_rank_for_sft_warm_start():
    spec = _spec(model="Qwen/Qwen3.5-0.8B", rank=32, algorithm="sft")

    with pytest.raises(ValueError, match=r"has rank 160.*serving max_lora_rank=128"):
        preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 160}))
