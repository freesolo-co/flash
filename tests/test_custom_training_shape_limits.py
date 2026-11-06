"""Custom shapes use common model limits and retain single-card admission checks."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flash.adapters.lora_rank import preflight_train_context_within_serving
from flash.core.catalog import MODELS, serving_context_cap, serving_lora_rank_cap
from flash.custom.validation import validate_operation
from flash.providers.core.base import provisional_gpu
from flash.schema import ConfigError, spec_from_dict


def custom_spec(model="Qwen/Qwen3.5-9B", **train):
    return spec_from_dict(
        {
            "model": model,
            "algorithm": "custom",
            "custom": {"file": "algorithm.py"},
            "train": train,
        }
    )


@pytest.mark.parametrize(
    ("model", "rank", "context"),
    [
        ("Qwen/Qwen3.5-9B", 128, 8192),
        ("Qwen/Qwen3.5-0.8B", 16, 16384),
        ("Qwen/Qwen3.5-4B", 64, 8192),
    ],
)
def test_extended_shapes_survive_roundtrip_and_fit_a_single_gpu(model, rank, context):
    spec = custom_spec(model, lora_rank=rank, lora_alpha=512, max_context_tokens=context)
    assert spec.train.lora_rank == rank
    assert spec.train.lora_alpha == 512
    assert spec.train.max_context_tokens == context
    assert spec_from_dict(spec.to_dict()).train == spec.train
    preflight_train_context_within_serving(spec)
    assert provisional_gpu(model, "custom", train=spec.train)


@pytest.mark.parametrize("model", MODELS)
def test_custom_retains_model_specific_rank_cap(model):
    with pytest.raises(ConfigError, match="serving max_lora_rank"):
        custom_spec(model, lora_rank=serving_lora_rank_cap(model) + 1)


def test_custom_context_uses_shared_model_cap():
    spec = custom_spec()
    cap = serving_context_cap(spec.model)
    # Isolate model context validation from the independent single-GPU memory ceiling.
    preflight_train_context_within_serving(
        replace(spec, train=replace(spec.train, max_context_tokens=cap))
    )
    with pytest.raises(ValueError, match="serving max_model_len"):
        preflight_train_context_within_serving(
            replace(spec, train=replace(spec.train, max_context_tokens=cap + 1))
        )


@pytest.mark.parametrize(
    "train",
    [
        {"max_context_tokens": 1},
        {"max_context_tokens": True},
        {"lora_rank": 0},
        {"lora_alpha": 0},
    ],
)
def test_invalid_shapes_still_rejected(train):
    with pytest.raises(ConfigError):
        custom_spec(**train)


def test_context_that_cannot_fit_one_gpu_is_rejected():
    with pytest.raises(ConfigError):
        custom_spec(max_context_tokens=262144)


def test_operations_obey_configured_context_above_old_ceiling():
    backend = SimpleNamespace(max_tokens=8192, max_batch=1, vocab_size=100)
    row = {"tokens": [1] * 8192, "weights": [1.0] * 8191}
    parsed = validate_operation(backend, "forward_backward", {"data": [row]})
    assert len(parsed["data"][0].tokens) == 8192
    validate_operation(backend, "sample", {"prompt": [1] * 8191, "max_tokens": 1})
    with pytest.raises(ValueError, match="2-8192"):
        validate_operation(backend, "forward", {"data": [{"tokens": [1] * 8193}]})
    with pytest.raises(ValueError, match="max_tokens"):
        validate_operation(backend, "sample", {"prompt": [1] * 8191, "max_tokens": 2})


def test_worker_passes_extended_shape_to_backend(monkeypatch):
    from flash.custom import worker

    spec = custom_spec(lora_rank=128, lora_alpha=512, max_context_tokens=8192)
    context = SimpleNamespace(config=SimpleNamespace(spec=spec, attempt=0))
    load = Mock(side_effect=RuntimeError("stop before loading GPU weights"))
    monkeypatch.setattr(worker, "RunCheckpoints", Mock())
    monkeypatch.setattr(worker, "heartbeat", Mock())
    monkeypatch.setattr(worker, "liveness_heartbeat", lambda *a, **k: nullcontext())
    monkeypatch.setattr(worker, "load_backend", load)
    with pytest.raises(RuntimeError, match="stop before loading GPU weights"):
        worker.run(context=context)
    assert load.call_args.kwargs["lora_rank"] == 128
    assert load.call_args.kwargs["lora_alpha"] == 512
    assert load.call_args.kwargs["max_tokens"] == 8192
