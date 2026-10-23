"""Checkpointing follows workload memory, independent of model names and GPU families."""

from dataclasses import replace

import pytest

from flash.core.catalog import MODELS
from flash.engine.plan.vram import sft_gc_off_peak_gb, sft_grad_checkpoint_can_disable
from flash.engine.worker.perf.memory import grad_checkpointing_on


def dense_gate(**overrides):
    options = {
        "model_id": "Qwen/Qwen3.5-9B",
        "max_length": 496,
        "allow_disable": True,
        "card_vram_gb": 47.695921152,
        "hidden": 4096,
        "num_layers": 32,
        "fused_ce": True,
        "per_device_bs": 1,
        "lora_rank": 8,
    }
    return grad_checkpointing_on(**(options | overrides))


@pytest.mark.parametrize(
    "override",
    [
        {},  # The original paid benchmark still disables checkpointing.
        {"revision": "f" * 40},
        {"max_length": 513},
        {"max_length": 2048, "card_vram_gb": 80},
        {"lora_rank": 64},
        {"per_device_bs": 2, "card_vram_gb": 80},
        {"card_vram_gb": 45},
        {"card_vram_gb": 96},
        {"hidden": 5120, "num_layers": 40, "card_vram_gb": 80},
    ],
)
def test_any_workload_that_fits_can_skip_recompute(override):
    assert dense_gate(**override) is False


def test_different_dense_model_uses_its_own_geometry():
    model = MODELS["Qwen/Qwen3.8-27B"]
    options = {"model_id": model.id, "hidden": model.hidden_size, "num_layers": model.num_layers}
    assert dense_gate(**options, card_vram_gb=96) is False
    assert dense_gate(**options, card_vram_gb=48) is True


def test_moe_has_no_fixed_minimum_card_size():
    assert (
        grad_checkpointing_on(
            "Qwen/Qwen3.6-35B-A3B",
            128,
            allow_disable=True,
            card_vram_gb=115,
            active_params_b=3.0,
            hidden=2048,
            num_layers=40,
            fused_ce=True,
            per_device_bs=1,
            lora_rank=8,
        )
        is False
    )


def test_model_name_does_not_control_memory_fit(monkeypatch):
    renamed = replace(MODELS["Qwen/Qwen3.5-9B"], id="test/renamed-dense")
    monkeypatch.setitem(MODELS, renamed.id, renamed)
    assert dense_gate(model_id=renamed.id) is False
    assert dense_gate(model_id=renamed.id, card_vram_gb=24) is True


@pytest.mark.parametrize(
    "override",
    [
        {"max_length": 4096},
        {"per_device_bs": 4},
        {"card_vram_gb": 24},
        {"max_length": 0},
        {"max_length": -1},
        {"hidden": 0},
        {"num_layers": -1},
        {"per_device_bs": 0},
        {"per_device_bs": -1},
        {"lora_rank": 0},
        {"card_vram_gb": float("inf")},
        {"card_vram_gb": float("nan")},
        {"fused_ce": False},
        {"allow_disable": False},
        {"model_id": "test/unknown-model"},
    ],
)
def test_insufficient_or_unknown_memory_keeps_checkpointing(override):
    assert dense_gate(**override) is True


def test_small_model_still_checks_memory_in_sft(monkeypatch):
    small = replace(MODELS["Qwen/Qwen3.5-9B"], id="test/small-model", params_b=1.0)
    monkeypatch.setitem(MODELS, small.id, small)
    # The legacy default remains available outside SFT's memory-aware decision.
    assert grad_checkpointing_on(small.id, 496) is False
    assert dense_gate(model_id=small.id, card_vram_gb=48) is False
    assert dense_gate(model_id=small.id, card_vram_gb=8) is True
    assert dense_gate(model_id=small.id, hidden=0) is True


@pytest.mark.parametrize(("active_params_b", "margin"), [(None, 8.0), (3.0, 18.0)])
def test_fit_boundary_includes_architecture_safety_margin(active_params_b, margin):
    model = MODELS["Qwen/Qwen3.6-35B-A3B" if active_params_b else "Qwen/Qwen3.5-9B"]
    options = {
        "active_params_b": active_params_b,
        "seq_len": 496,
        "hidden": model.hidden_size,
        "num_layers": model.num_layers,
        "batch": 1,
        "lora_rank": 8,
        "model_info": model,
    }
    peak = sft_gc_off_peak_gb(model.params_b, **options)
    assert not sft_grad_checkpoint_can_disable(
        model.params_b, card_vram_gb=peak + margin - 0.01, **options
    )
    assert sft_grad_checkpoint_can_disable(model.params_b, card_vram_gb=peak + margin, **options)


@pytest.mark.parametrize(
    "override",
    [
        {"params_b": 0},
        {"params_b": float("nan")},
        {"batch": 0},
        {"hidden": -1},
        {"seq_len": -1},
        {"num_layers": 0},
        {"lora_rank": -1},
        {"active_params_b": -1},
        {"active_params_b": float("nan")},
        {"active_params_b": 100},
        {"quant": "unknown"},
        {"card_vram_gb": float("inf")},
        {"margin_gb": -1},
        {"margin_gb": float("nan")},
    ],
)
def test_estimator_rejects_invalid_inputs(override):
    options = {
        "params_b": 9.7,
        "active_params_b": None,
        "seq_len": 496,
        "hidden": 4096,
        "num_layers": 32,
        "batch": 1,
        "lora_rank": 8,
        "card_vram_gb": 48,
    }
    assert not sft_grad_checkpoint_can_disable(**(options | override))
