"""Trainer:inference split selection + the reasonable ratio grid for disaggregated GRPO."""

from __future__ import annotations

import pytest

from flash.engine.rollout_bench import (
    ratio_grid,
    select_rollout_split,
    validate_disaggregated_requirement,
)


def test_colocate_is_inference_gpus_zero():
    s = select_rollout_split(1, 0)
    assert s.mode == "colocate"
    assert s.label == "colocate"
    assert s.train_gpus == 1
    assert s.infer_gpus == 0
    assert s.train_devices == (0,)
    assert s.infer_devices == ()


def test_disaggregated_pins_inference_to_first_devices():
    # The vLLM server takes the FIRST devices (device 0 → valid NVML index); the trainer the rest.
    s = select_rollout_split(2, 1)  # 1:1
    assert s.mode == "disaggregated"
    assert s.label == "1:1"
    assert s.infer_devices == (0,)
    assert s.train_devices == (1,)

    s = select_rollout_split(4, 1)  # 3:1
    assert s.label == "3:1"
    assert s.infer_devices == (0,)
    assert s.train_devices == (1, 2, 3)

    s = select_rollout_split(3, 2)  # 1:2
    assert s.label == "1:2"
    assert s.infer_devices == (0, 1)
    assert s.train_devices == (2,)


def test_invalid_splits_raise():
    with pytest.raises(ValueError, match="total_gpus must be >= 1"):
        select_rollout_split(0, 0)
    with pytest.raises(ValueError, match="must be < total_gpus"):
        select_rollout_split(2, 2)  # no GPU left to train
    with pytest.raises(ValueError, match="must be < total_gpus"):
        select_rollout_split(2, 3)
    with pytest.raises(ValueError, match="inference_gpus must be >= 0"):
        select_rollout_split(2, -1)


def test_ratio_grid_starts_colocate_and_excludes_absurd_splits():
    grid = ratio_grid(max_gpus=4)
    labels = [s.label for s in grid]
    # colocate is always first (the baseline row of the benchmark table)
    assert labels[0] == "colocate"
    # the reasonable splits within a 4-GPU node, ordered by (total gpus, infer gpus)
    assert labels == ["colocate", "1:1", "2:1", "1:2", "3:1", "2:2", "1:3"]
    # no absurd imbalance (e.g. no 1:3 / 3:1 beyond the cap is fine; nothing past _MAX_RATIO=3)
    for s in grid[1:]:
        assert max(s.train_gpus, s.infer_gpus) / min(s.train_gpus, s.infer_gpus) <= 3


def test_ratio_grid_caps_at_max_gpus():
    grid = ratio_grid(max_gpus=2)
    assert [s.label for s in grid] == ["colocate", "1:1"]
    # every config is a valid, self-consistent split
    for s in grid:
        assert s.train_gpus + s.infer_gpus == s.total_gpus
        assert len(s.train_devices) == s.train_gpus
        assert len(s.infer_devices) == s.infer_gpus


def test_disaggregated_requirement_rejects_colocate_grpo():
    # requires_disaggregated + GRPO + colocate (inference_gpus=0) -> rejected
    with pytest.raises(ValueError, match="disaggregated GRPO path"):
        validate_disaggregated_requirement(
            requires_disaggregated=True, algorithm="grpo", inference_gpus=0
        )
    # GRPO with dedicated inference GPUs is allowed
    validate_disaggregated_requirement(
        requires_disaggregated=True, algorithm="grpo", inference_gpus=1
    )
    # SFT (no rollout engine) is always allowed
    validate_disaggregated_requirement(
        requires_disaggregated=True, algorithm="sft", inference_gpus=0
    )
    # a normal model is unaffected
    validate_disaggregated_requirement(
        requires_disaggregated=False, algorithm="grpo", inference_gpus=0
    )


def test_single_trainer_only_rejects_multi_trainer_split():
    # single_trainer_only (the 35B) cannot replicate the policy across >1 trainer card (plain DDP).
    # A 2:2 split (count=4, inference_gpus=2 -> 2 trainer GPUs) must be rejected at submit.
    with pytest.raises(ValueError, match="SINGLE-trainer"):
        validate_disaggregated_requirement(
            requires_disaggregated=True,
            algorithm="grpo",
            inference_gpus=2,
            single_trainer_only=True,
            gpu_count=4,
        )
    # 1:N (single trainer) is allowed: count=3, inference_gpus=2 -> 1 trainer GPU.
    validate_disaggregated_requirement(
        requires_disaggregated=True,
        algorithm="grpo",
        inference_gpus=2,
        single_trainer_only=True,
        gpu_count=3,
    )
    # gpu_count omitted (old callers) -> the multi-trainer check is skipped, no crash.
    validate_disaggregated_requirement(
        requires_disaggregated=True, algorithm="grpo", inference_gpus=1, single_trainer_only=True
    )
    # a non-single-trainer model can take a multi-trainer split.
    validate_disaggregated_requirement(
        requires_disaggregated=False,
        algorithm="grpo",
        inference_gpus=2,
        single_trainer_only=False,
        gpu_count=4,
    )


def test_catalog_35b_is_disaggregated_only():
    from flash.catalog import get_model

    m = get_model("Qwen/Qwen3.6-35B-A3B")
    assert m.requires_disaggregated is True
    assert m.single_trainer_only is True
    assert "grpo" in m.algos


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
