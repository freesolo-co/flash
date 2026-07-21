from __future__ import annotations

import pytest

from flash.lora_rank import preflight_train_context_within_serving
from flash.spec import JobSpec, TrainSpec


def _spec(
    *,
    model: str,
    algorithm: str,
    max_context_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    thinking: bool = False,
    model_policy: str = "catalog",
) -> JobSpec:
    # Built directly (not via spec_from_dict) so the unit under test is the context preflight alone,
    # not the allocator VRAM sizing spec_from_dict also runs.
    return JobSpec(
        model=model,
        algorithm=algorithm,
        thinking=thinking,
        model_policy=model_policy,
        train=TrainSpec(
            max_context_tokens=max_context_tokens,
            max_completion_tokens=max_completion_tokens,
        ),
    )


def test_sft_max_context_tokens_above_serving_cap_rejected():
    # 4b serves at max_model_len=32768; a 40000-token sft context exceeds that boundary.
    spec = _spec(model="Qwen/Qwen3.5-4B", algorithm="sft", max_context_tokens=40000)
    with pytest.raises(
        ValueError, match=r"train\.max_context_tokens=40000 exceeds .*max_model_len=32768"
    ):
        preflight_train_context_within_serving(spec)


def test_sft_max_context_tokens_at_cap_allowed():
    preflight_train_context_within_serving(
        _spec(model="Qwen/Qwen3.5-4B", algorithm="sft", max_context_tokens=32768)
    )


def test_27b_sft_context_at_serving_cap_allowed():
    preflight_train_context_within_serving(
        _spec(model="Qwen/Qwen3.6-27B", algorithm="sft", max_context_tokens=32768)
    )


def test_27b_sft_context_above_serving_cap_rejected():
    spec = _spec(model="Qwen/Qwen3.6-27B", algorithm="sft", max_context_tokens=32769)
    with pytest.raises(ValueError, match=r"exceeds .*serving max_model_len=32768"):
        preflight_train_context_within_serving(spec)


def test_27b_grpo_effective_rollout_above_serving_cap_rejected():
    spec = _spec(
        model="Qwen/Qwen3.6-27B",
        algorithm="grpo",
        max_completion_tokens=40000,
    )
    with pytest.raises(ValueError, match=r"exceeds .*serving max_model_len=32768"):
        preflight_train_context_within_serving(spec)


def test_sft_unset_max_context_tokens_allowed():
    # Unset -> the worker's small recipe default, always within the cap.
    preflight_train_context_within_serving(_spec(model="Qwen/Qwen3.5-4B", algorithm="sft"))


def test_35b_serves_4096_so_8192_sft_context_rejected():
    # The 35B now serves at 4096 (weight-bound 6x64 ceiling), so an 8192 SFT context is rejected.
    spec = _spec(model="Qwen/Qwen3.6-35B-A3B", algorithm="sft", max_context_tokens=8192)
    with pytest.raises(ValueError, match=r"exceeds .*serving max_model_len=4096"):
        preflight_train_context_within_serving(spec)


def test_35b_4096_context_allowed():
    preflight_train_context_within_serving(
        _spec(model="Qwen/Qwen3.6-35B-A3B", algorithm="sft", max_context_tokens=4096)
    )


def test_grpo_unset_rollout_within_35b_cap_allowed():
    # Unset GRPO rollout defaults (max_prompt 2048 + completion) stay under 4096, even for thinking.
    preflight_train_context_within_serving(
        _spec(model="Qwen/Qwen3.6-35B-A3B", algorithm="grpo", thinking=True)
    )


def test_grpo_big_max_completion_tokens_pushes_rollout_over_cap_rejected():
    # a large completion budget makes prompt+completion exceed the served context.
    spec = _spec(model="Qwen/Qwen3.5-4B", algorithm="grpo", max_completion_tokens=40000)
    with pytest.raises(ValueError, match=r"exceeds .*serving max_model_len=32768"):
        preflight_train_context_within_serving(spec)


def test_opd_rollout_context_above_serving_cap_rejected():
    spec = _spec(
        model="Qwen/Qwen3.6-35B-A3B",
        algorithm="opd",
        max_completion_tokens=3500,
    )
    with pytest.raises(
        ValueError,
        match=r"OPD rollout prompt\+completion\)=4524 exceeds .*max_model_len=4096",
    ):
        preflight_train_context_within_serving(spec)


def test_opd_rollout_context_within_serving_cap_allowed():
    preflight_train_context_within_serving(
        _spec(
            model="Qwen/Qwen3.6-35B-A3B",
            algorithm="opd",
            max_completion_tokens=3000,
        )
    )


def test_open_policy_uncataloged_model_skipped():
    # No serving entry -> serving_context_cap is None -> preflight is a no-op even for a huge context.
    spec = _spec(
        model="mistralai/Mistral-7B-v0.1",
        algorithm="sft",
        max_context_tokens=32768,
        model_policy="allow",
    )
    preflight_train_context_within_serving(spec)
