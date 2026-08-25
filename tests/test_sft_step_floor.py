"""The sft estimate excludes one-time framework setup and retains recurring step work."""

from __future__ import annotations

import pytest

from flash.cost.analytical import (
    SFT_STEP_FLOOR_SECONDS,
    compile_seconds,
    estimate_cost,
    required_save_overhead_seconds,
    sft_step_floor_seconds,
    sharded_step_seconds,
)
from flash.cost.types import RunConfig


def _sft(**overrides):
    base = {
        "model_id": "Qwen/Qwen3.5-9B",
        "method": "sft",
        "steps": 32,
        "seq_len": 1024,
        "batch_size": 8,
        "gpu_type": "H100",
        "provider": "runpod",
    }
    base.update(overrides)
    return RunConfig(**base)


def test_sft_step_floor_has_no_one_time_startup_block():
    assert sft_step_floor_seconds(0) == 0.0
    assert sft_step_floor_seconds(-5) == 0.0
    assert sft_step_floor_seconds(32) == pytest.approx(SFT_STEP_FLOOR_SECONDS * 32)


def test_sft_step_floor_is_independent_of_model_size():
    assert sft_step_floor_seconds(_sft(model_id="Qwen/Qwen3.5-9B").steps) == (
        sft_step_floor_seconds(_sft(model_id="Qwen/Qwen3.6-35B-A3B").steps)
    )


def test_sft_quote_is_exactly_training_work_without_framework_setup():
    config = _sft(model_id="Qwen/Qwen3.5-9B", steps=2)
    quote = estimate_cost(config)
    expected = (
        compile_seconds(config, quote.gpu)
        + config.steps * sharded_step_seconds(config, quote.gpu, quote.gpu_count, quote.provider)
        + required_save_overhead_seconds(config)
        + sft_step_floor_seconds(config.steps)
    )
    assert quote.train_seconds == pytest.approx(expected)


def test_sft_step_floor_reaches_both_step_and_token_quote_paths(monkeypatch):
    import flash.cost.analytical as analytical

    step_config = _sft()
    token_config = _sft(train_tokens=32197.0)
    step_with = estimate_cost(step_config).train_seconds
    token_with = estimate_cost(token_config).train_seconds

    monkeypatch.setattr(analytical, "SFT_STEP_FLOOR_SECONDS", 0.0)
    step_without = estimate_cost(step_config).train_seconds
    token_without = estimate_cost(token_config).train_seconds

    floor = SFT_STEP_FLOOR_SECONDS * step_config.steps
    assert step_with - step_without == pytest.approx(floor)
    assert token_with - token_without == pytest.approx(floor)


@pytest.mark.parametrize("method", ["grpo", "opd"])
def test_rollout_quotes_do_not_pay_the_sft_step_floor(method, monkeypatch):
    import flash.cost.analytical as analytical

    config = RunConfig(
        model_id="Qwen/Qwen3.5-9B",
        method=method,
        steps=8,
        seq_len=1024,
        completion_len=512,
        batch_size=8,
        group_size=4,
        gpu_type="B200",
        provider="runpod",
    )
    before = estimate_cost(config).train_seconds
    monkeypatch.setattr(analytical, "SFT_STEP_FLOOR_SECONDS", 9_999.0)
    assert estimate_cost(config).train_seconds == pytest.approx(before)
