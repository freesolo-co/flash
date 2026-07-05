from __future__ import annotations

import pytest

from flash.lora_rank import preflight_train_context_within_serving
from flash.spec import JobSpec, TrainSpec


def _spec(
    *,
    model: str,
    algorithm: str,
    max_length: int | None = None,
    max_tokens: int | None = None,
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
        train=TrainSpec(max_length=max_length, max_tokens=max_tokens),
    )


def test_sft_max_length_above_serving_cap_rejected():
    # 4B serves at max_model_len=8192; a 16384 SFT context is longer than it is ever served.
    spec = _spec(model="Qwen/Qwen3.5-4B", algorithm="sft", max_length=16384)
    with pytest.raises(ValueError, match=r"train\.max_length=16384 exceeds .*max_model_len=8192"):
        preflight_train_context_within_serving(spec)


def test_sft_max_length_at_cap_allowed():
    preflight_train_context_within_serving(
        _spec(model="Qwen/Qwen3.5-4B", algorithm="sft", max_length=8192)
    )


def test_sft_unset_max_length_allowed():
    # Unset -> the worker's small recipe default, always within the cap.
    preflight_train_context_within_serving(_spec(model="Qwen/Qwen3.5-4B", algorithm="sft"))


def test_35b_serves_4096_so_8192_sft_context_rejected():
    # The 35B now serves at 4096 (weight-bound 6x64 ceiling), so an 8192 SFT context is rejected.
    spec = _spec(model="Qwen/Qwen3.6-35B-A3B", algorithm="sft", max_length=8192)
    with pytest.raises(ValueError, match=r"exceeds .*serving max_model_len=4096"):
        preflight_train_context_within_serving(spec)


def test_35b_4096_context_allowed():
    preflight_train_context_within_serving(
        _spec(model="Qwen/Qwen3.6-35B-A3B", algorithm="sft", max_length=4096)
    )


def test_grpo_unset_rollout_within_35b_cap_allowed():
    # Unset GRPO rollout defaults (max_prompt 2048 + completion) stay under 4096, even for thinking.
    preflight_train_context_within_serving(
        _spec(model="Qwen/Qwen3.6-35B-A3B", algorithm="grpo", thinking=True)
    )


def test_grpo_big_max_tokens_pushes_rollout_over_cap_rejected():
    # A large completion budget makes prompt+completion (grpo_rollout_seq_len) exceed the served ctx.
    spec = _spec(model="Qwen/Qwen3.5-4B", algorithm="grpo", max_tokens=8192)
    with pytest.raises(ValueError, match=r"exceeds .*serving max_model_len=8192"):
        preflight_train_context_within_serving(spec)


def test_open_policy_uncataloged_model_skipped():
    # No serving entry -> serving_context_cap is None -> preflight is a no-op even for a huge context.
    spec = _spec(
        model="mistralai/Mistral-7B-v0.1",
        algorithm="sft",
        max_length=32768,
        model_policy="allow",
    )
    preflight_train_context_within_serving(spec)
