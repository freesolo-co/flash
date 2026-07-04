from __future__ import annotations

import types

import pytest

from flash.engine.worker import lora as W

_BASELINE_QWEN3_6 = r"(^|.*\.)(visual|vision_tower|multi_modal_projector|mtp)(\..*|$)"


@pytest.fixture
def fake_config(monkeypatch):
    """Stub transformers.AutoConfig.from_pretrained to a chosen model_type (no network)."""

    def _set(model_type: str):
        class _AutoConfig:
            @staticmethod
            def from_pretrained(*_a, **_k):
                return types.SimpleNamespace(model_type=model_type)

        import transformers

        monkeypatch.setattr(transformers, "AutoConfig", _AutoConfig)

    return _set


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    monkeypatch.delenv("FLASH_LORA_ATTN_ROUTER_ONLY", raising=False)


def test_baseline_unset_is_byte_identical(fake_config):
    # Flag unset -> qwen3_6 returns exactly the vision-tower exclude regex (byte-identical to main).
    fake_config("qwen3_6")
    assert W.lora_exclude_modules("x") == _BASELINE_QWEN3_6


def test_attn_router_flag_adds_moe_segments(fake_config, monkeypatch):
    fake_config("qwen3_6")
    monkeypatch.setenv("FLASH_LORA_ATTN_ROUTER_ONLY", "1")
    rgx = W.lora_exclude_modules("x")
    for seg in ("experts", "shared_expert", "shared_expert_gate", "linear_attn"):
        assert seg in rgx, seg
    # Baseline VL segments are still present (union, not replacement).
    for seg in ("visual", "vision_tower", "multi_modal_projector", "mtp"):
        assert seg in rgx, seg


@pytest.mark.parametrize("flag", ["0", "false", "off", "", "no", "none"])
def test_falsey_flag_is_baseline(fake_config, monkeypatch, flag):
    fake_config("qwen3_6")
    monkeypatch.setenv("FLASH_LORA_ATTN_ROUTER_ONLY", flag)
    assert W.lora_exclude_modules("x") == _BASELINE_QWEN3_6


def test_non_vl_model_returns_none_in_both_states(fake_config, monkeypatch):
    fake_config("llama")
    assert W.lora_exclude_modules("x") is None
    monkeypatch.setenv("FLASH_LORA_ATTN_ROUTER_ONLY", "1")
    # A non-MoE/non-VL model has no VL segments; the flag alone must NOT start excluding on it
    # (the exclude only augments an existing VL segment set for the routed-MoE checkpoints).
    assert W.lora_exclude_modules("x") is None


def test_attn_router_regex_matches_expert_and_shared_expert_paths(fake_config, monkeypatch):
    fake_config("qwen3_6")
    monkeypatch.setenv("FLASH_LORA_ATTN_ROUTER_ONLY", "1")
    import re

    pat = re.compile(W.lora_exclude_modules("x"))
    # Should EXCLUDE experts / shared_expert / linear_attn ...
    assert pat.fullmatch("model.language_model.layers.5.mlp.experts.gate_up_proj")
    assert pat.fullmatch("model.language_model.layers.5.mlp.shared_expert.down_proj")
    assert pat.fullmatch("model.language_model.layers.5.mlp.shared_expert_gate")
    assert pat.fullmatch("model.language_model.layers.2.linear_attn.in_proj_qkv")
    # ... but KEEP attention + router (must NOT match).
    assert not pat.fullmatch("model.language_model.layers.3.self_attn.q_proj")
    assert not pat.fullmatch("model.language_model.layers.3.self_attn.o_proj")
    assert not pat.fullmatch("model.language_model.layers.5.mlp.gate")
