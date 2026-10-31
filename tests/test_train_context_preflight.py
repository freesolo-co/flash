from __future__ import annotations

import pytest

from flash.adapters.lora_rank import preflight_train_context_within_serving
from flash.core.spec import JobSpec, TrainSpec


def _spec(
    *,
    model: str,
    algorithm: str,
    max_context_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    thinking: bool = False,
) -> JobSpec:
    # Built directly (not via spec_from_dict) so the unit under test is the context preflight alone,
    # not the allocator VRAM sizing spec_from_dict also runs.
    return JobSpec(
        model=model,
        algorithm=algorithm,
        thinking=thinking,
        train=TrainSpec(
            max_context_tokens=max_context_tokens,
            max_completion_tokens=max_completion_tokens,
        ),
    )


def test_sft_max_context_tokens_above_serving_cap_rejected():
    # The 9B serves at native 256k; reject contexts above the advertised window.
    spec = _spec(model="Qwen/Qwen3.5-9B", algorithm="sft", max_context_tokens=262145)
    with pytest.raises(
        ValueError, match=r"train\.max_context_tokens=262145 exceeds .*max_model_len=262144"
    ):
        preflight_train_context_within_serving(spec)


def test_sft_max_context_tokens_at_cap_allowed():
    preflight_train_context_within_serving(
        _spec(model="Qwen/Qwen3.5-9B", algorithm="sft", max_context_tokens=262144)
    )


@pytest.mark.parametrize(
    ("algorithm", "kwargs"),
    [("sft", {"max_context_tokens": 262144}), ("grpo", {"max_completion_tokens": 260096})],
)
def test_qwen38_context_at_approved_serving_cap_allowed(algorithm: str, kwargs: dict):
    preflight_train_context_within_serving(
        _spec(model="Qwen/Qwen3.8-27B", algorithm=algorithm, **kwargs)
    )


@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
def test_qwen38_context_above_approved_serving_cap_rejected(algorithm: str):
    kwargs = (
        {"max_context_tokens": 262145} if algorithm == "sft" else {"max_completion_tokens": 261121}
    )
    with pytest.raises(ValueError, match=r"exceeds .*serving max_model_len=262144"):
        preflight_train_context_within_serving(
            _spec(model="Qwen/Qwen3.8-27B", algorithm=algorithm, **kwargs)
        )


def test_sft_unset_max_context_tokens_allowed():
    # Unset -> the worker's small recipe default, always within the cap.
    preflight_train_context_within_serving(_spec(model="Qwen/Qwen3.5-9B", algorithm="sft"))


def test_35b_262144_context_allowed():
    preflight_train_context_within_serving(
        _spec(model="Qwen/Qwen3.6-35B-A3B", algorithm="sft", max_context_tokens=262144)
    )


def test_35b_context_above_serving_cap_rejected():
    spec = _spec(model="Qwen/Qwen3.6-35B-A3B", algorithm="sft", max_context_tokens=262145)
    with pytest.raises(ValueError, match=r"exceeds .*serving max_model_len=262144"):
        preflight_train_context_within_serving(spec)


def test_grpo_unset_rollout_within_35b_cap_allowed():
    # unset grpo rollout defaults (max_prompt 2048 + completion) stay under 262144, even for thinking.
    preflight_train_context_within_serving(
        _spec(model="Qwen/Qwen3.6-35B-A3B", algorithm="grpo", thinking=True)
    )


def test_grpo_big_max_completion_tokens_pushes_rollout_over_cap_rejected():
    # a large completion budget makes prompt+completion exceed the served context.
    spec = _spec(model="Qwen/Qwen3.5-9B", algorithm="grpo", max_completion_tokens=262144)
    with pytest.raises(ValueError, match=r"exceeds .*serving max_model_len=262144"):
        preflight_train_context_within_serving(spec)


def test_a_model_without_a_serving_entry_is_skipped(monkeypatch):
    # clear an active model's serving entry to prove this branch follows catalog state rather than
    # relying only on the naturally inactive hosted candidate.
    from dataclasses import replace

    import flash.core.catalog as catalog

    model = "Qwen/Qwen3.5-9B"
    monkeypatch.setitem(catalog.MODELS, model, replace(catalog.MODELS[model], serving=None))
    preflight_train_context_within_serving(
        _spec(model=model, algorithm="sft", max_context_tokens=262144)
    )
