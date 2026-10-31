import pytest

from flash.adapters.lora_rank import preflight_train_context_within_serving
from flash.core.catalog import serving_context_cap
from flash.core.spec import JobSpec


@pytest.mark.parametrize(
    ("model", "expected_cap"),
    [
        ("Qwen/Qwen3.5-0.8B", 262144),
        ("Qwen/Qwen3.5-4B", 262144),
        ("Qwen/Qwen3.5-9B", 262144),
        ("Qwen/Qwen3.8-27B", 262144),
        ("Qwen/Qwen3.6-35B-A3B", 262144),
    ],
)
def test_serving_context_cap(model: str, expected_cap: int | None) -> None:
    assert serving_context_cap(model) == expected_cap


def _sft_spec(max_context_tokens: int, model: str = "Qwen/Qwen3.5-9B") -> JobSpec:
    return JobSpec.from_dict(
        {
            "model": model,
            "algorithm": "sft",
            "train": {"max_context_tokens": max_context_tokens},
        }
    )


@pytest.mark.parametrize("model", ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"])
def test_training_context_preflight_allows_serving_cap(model: str) -> None:
    preflight_train_context_within_serving(_sft_spec(262144, model))


@pytest.mark.parametrize("model", ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"])
def test_training_context_preflight_rejects_above_serving_cap(model: str) -> None:
    with pytest.raises(ValueError, match=r"max_context_tokens=262145.*max_model_len=262144"):
        preflight_train_context_within_serving(_sft_spec(262145, model))


@pytest.mark.parametrize("max_context_tokens", [262144, 262145])
def test_qwen38_context_envelope(max_context_tokens: int) -> None:
    spec = _sft_spec(max_context_tokens, model="Qwen/Qwen3.8-27B")
    if max_context_tokens == 262144:
        preflight_train_context_within_serving(spec)
    else:
        with pytest.raises(ValueError, match=r"max_context_tokens=262145.*max_model_len=262144"):
            preflight_train_context_within_serving(spec)
