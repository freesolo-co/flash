from __future__ import annotations

from dataclasses import replace

import pytest

from flash.lora_rank import preflight_init_adapter_lora_rank, resolve_adapter_ref
from flash.schema import spec_from_dict

_ADAPTER_REF = "owner/runs:sft/sft-run"


def _spec(*, model: str = "Qwen/Qwen3.5-4B", rank: int = 16, algorithm: str = "grpo"):
    train = {
        "steps": 1,
        "lora_rank": rank,
    }
    if algorithm == "sft":
        train["max_examples"] = 8
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
    spec = _spec(rank=16)

    with pytest.raises(ValueError, match=r"has rank 96.*serving max_lora_rank=32"):
        preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 96}))


def test_init_adapter_preflight_rejects_vl_recombined_rank_before_training():
    spec = _spec(rank=16)

    with pytest.raises(
        ValueError,
        match=r"rank 40 \(SFT rank 24 \+ GRPO rank 16\).*set GRPO train\.lora_rank <= 8",
    ):
        preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 24}))


def test_init_adapter_preflight_allows_vl_recombined_rank_at_serving_cap():
    spec = _spec(rank=16)

    preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 16}))


def test_init_adapter_preflight_rejects_opd_vl_recombined_rank_before_training():
    # opd VL warm-start ALSO stacks SFT⊕opd at deploy (recombined_warmstart_adapter_dir), so the
    # recombined rank must fit the serving cap — same preflight as GRPO, worded for opd.
    spec = _spec(rank=16, algorithm="opd")

    with pytest.raises(
        ValueError,
        match=r"SFT\+OPD adapter would be rank 40 \(SFT rank 24 \+ OPD rank 16\).*"
        r"set OPD train\.lora_rank <= 8",
    ):
        preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 24}))


def test_init_adapter_preflight_allows_opd_vl_recombined_rank_at_serving_cap():
    spec = _spec(rank=16, algorithm="opd")

    preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 16}))  # 16+16=32=cap


def test_init_adapter_preflight_allows_empty_vl_patterns():
    spec = _spec(rank=16)

    preflight_init_adapter_lora_rank(
        spec,
        config_loader=_loader({"r": 16, "rank_pattern": {}, "alpha_pattern": {}}),
    )


def test_init_adapter_preflight_rejects_non_uniform_vl_rank_pattern():
    spec = _spec(rank=16)

    with pytest.raises(ValueError, match="rank_pattern"):
        preflight_init_adapter_lora_rank(
            spec,
            config_loader=_loader({"r": 16, "rank_pattern": {"q_proj": 8}}),
        )


@pytest.mark.parametrize("key", ["rank_pattern", "alpha_pattern"])
@pytest.mark.parametrize("value", [0, "", [], False])
def test_init_adapter_preflight_rejects_falsey_invalid_vl_patterns(key, value):
    spec = _spec(rank=16)

    with pytest.raises(ValueError, match=key):
        preflight_init_adapter_lora_rank(
            spec,
            config_loader=_loader({"r": 16, key: value}),
        )


def test_init_adapter_preflight_checks_adapter_rank_for_sft_warm_start():
    spec = _spec(model="Qwen/Qwen3.5-0.8B", rank=32, algorithm="sft")

    with pytest.raises(ValueError, match=r"has rank 96.*serving max_lora_rank=64"):
        preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 96}))


def test_lora_rank_uses_schema_adapter_storage_ref_parser():
    assert resolve_adapter_ref("owner/runs:sft/source-run/checkpoints/step-40") == (
        "owner/runs",
        "sft/source-run/checkpoints/step-40",
    )
