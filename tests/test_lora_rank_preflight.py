from __future__ import annotations

from dataclasses import replace

import pytest

from flash.lora_rank import (
    preflight_init_adapter_lora_rank,
    preflight_opd_reference_lora_rank,
    resolve_adapter_ref,
)
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


def test_init_adapter_preflight_rejects_lora_rank_below_source_rank():
    # The reported bug: a rank-64 source with the default lora_rank=32 fits the serving cap (64) but
    # would be cost/allocator/GRPO-sleep sized as rank 32 and then run/serve at the source's rank 64.
    spec = _spec(model="Qwen/Qwen3.5-4B", rank=32)

    with pytest.raises(
        ValueError, match=r"train\.lora_rank=32 does not match.*rank 64.*Set train\.lora_rank=64"
    ):
        preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 64}))


def test_init_adapter_preflight_rejects_lora_rank_above_source_rank():
    # A too-high lora_rank is also a mismatch: the continued adapter is rank 16, so sizing at 32 is
    # wrong (and misleading) even though it merely over-provisions rather than OOMs.
    spec = _spec(model="Qwen/Qwen3.5-4B", rank=32)

    with pytest.raises(ValueError, match=r"train\.lora_rank=32 does not match.*rank 16"):
        preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 16}))


def test_init_adapter_preflight_allows_lora_rank_matching_source_rank():
    # The common warm-start: lora_rank equals the continued source adapter's rank -> passes.
    spec = _spec(model="Qwen/Qwen3.5-4B", rank=64)

    preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 64}))


def test_init_adapter_preflight_serving_cap_precedes_rank_mismatch():
    # When the source both exceeds the cap AND mismatches lora_rank, the cap violation (the harder
    # blocker) is reported first so the user fixes the undeployable rank before the sizing mismatch.
    spec = _spec(model="Qwen/Qwen3.5-4B", rank=32)

    with pytest.raises(ValueError, match=r"has rank 128.*serving max_lora_rank=64"):
        preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 128}))


def _open_policy_spec(*, rank: int):
    # Open-policy / uncataloged model -> serving_lora_rank_cap is None (no serving entry).
    from flash.spec import JobSpec, TrainSpec

    return JobSpec(
        model="mistralai/Mistral-7B-v0.1",
        algorithm="grpo",
        model_policy="allow",
        train=TrainSpec(
            epochs=1, max_examples=8, lora_rank=rank, init_from_adapter=_ADAPTER_REF
        ),
    )


def test_init_adapter_preflight_rejects_rank_mismatch_for_uncapped_model():
    # No serving cap, but the mismatch is cap-INDEPENDENT: cost/allocator/GRPO-sleep still read
    # train.lora_rank, so a rank-64 source with lora_rank=32 is billed and placed at 32 then OOMs at 64.
    spec = _open_policy_spec(rank=32)

    with pytest.raises(ValueError, match=r"train\.lora_rank=32 does not match.*rank 64"):
        preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 64}))


def test_init_adapter_preflight_allows_matching_rank_for_uncapped_model():
    # A capless model with lora_rank equal to the continued adapter's rank passes.
    spec = _open_policy_spec(rank=64)

    preflight_init_adapter_lora_rank(spec, config_loader=_loader({"r": 64}))


def test_opd_reference_rank_preflight_is_independent_from_policy_rank():
    from flash.spec import JobSpec, TrainSpec

    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(
            lora_rank=8,
            opd_reference_adapter=_ADAPTER_REF,
            opd_reference_lora_rank=128,
            opd_objective_ids=("c06",),
        ),
    )

    verified = preflight_opd_reference_lora_rank(
        spec, config_loader=_loader({"r": 64, "rank_pattern": {"layer": 128}})
    )
    assert verified.train.lora_rank == 8
    assert verified.train.opd_reference_lora_rank == 128


def test_opd_reference_rank_preflight_rejects_source_metadata_mismatch():
    from flash.spec import JobSpec, TrainSpec

    spec = JobSpec(
        model="Qwen/Qwen3.5-4B",
        algorithm="opd",
        train=TrainSpec(
            lora_rank=8,
            opd_reference_adapter=_ADAPTER_REF,
            opd_reference_lora_rank=64,
            opd_objective_ids=("c11",),
        ),
    )

    with pytest.raises(ValueError, match=r"has rank 128.*source SFT run records.*64"):
        preflight_opd_reference_lora_rank(spec, config_loader=_loader({"r": 128}))


def test_lora_rank_uses_schema_adapter_storage_ref_parser():
    assert resolve_adapter_ref("owner/runs:sft/source-run/checkpoints/step-40") == (
        "owner/runs",
        "sft/source-run/checkpoints/step-40",
    )
