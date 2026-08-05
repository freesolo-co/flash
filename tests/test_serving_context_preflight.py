import pytest

from flash.catalog import serving_context_cap
from flash.lora_rank import preflight_train_context_within_serving
from flash.spec import JobSpec


@pytest.mark.parametrize(
    ("model", "expected_cap"),
    [
        ("Qwen/Qwen3.5-0.8B", 32768),
        ("Qwen/Qwen3.5-2B", 32768),
        ("Qwen/Qwen3.5-4B", 32768),
        ("Qwen/Qwen3.5-9B", 32768),
        ("Qwen/Qwen3.6-27B", 32768),
        ("Qwen/Qwen3.6-35B-A3B", 32768),
    ],
)
def test_serving_context_cap(model: str, expected_cap: int) -> None:
    assert serving_context_cap(model) == expected_cap


def _sft_spec(max_context_tokens: int) -> JobSpec:
    return JobSpec.from_dict(
        {
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "sft",
            "train": {"max_context_tokens": max_context_tokens},
        }
    )


def test_training_context_preflight_allows_serving_cap() -> None:
    preflight_train_context_within_serving(_sft_spec(32768))


def test_training_context_preflight_rejects_above_serving_cap() -> None:
    with pytest.raises(ValueError, match=r"max_context_tokens=40000.*max_model_len=32768"):
        preflight_train_context_within_serving(_sft_spec(40000))
