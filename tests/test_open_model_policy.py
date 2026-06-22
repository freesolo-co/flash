"""Open-model policy + VRAM estimator unit tests (CPU-only, no network)."""

from __future__ import annotations

import pytest

from flash.catalog import resolve_model
from flash.engine.vram import VramEstimate, check_fit, estimate_vram_gb
from flash.schema import ConfigError, spec_from_dict
from tests._helpers.specs import raw_spec as _raw


def test_catalog_policy_rejects_unlisted_with_hint():
    with pytest.raises(ValueError, match="model_policy"):
        resolve_model("some-org/some-model", "sft", policy="catalog")


def test_catalog_model_resolves_normally():
    info = resolve_model("Qwen/Qwen3.5-4B", "grpo", policy="catalog")
    assert info.id == "Qwen/Qwen3.5-4B"


def test_minicpm5_in_catalog():
    info = resolve_model("openbmb/MiniCPM5-1B", "grpo", policy="catalog")
    assert info.recommended_gpu == "RTX 4090"
    assert "grpo" in info.algos
    assert "sft" in info.algos


def test_allow_policy_accepts_fitting_model(monkeypatch):
    monkeypatch.setattr("flash.engine.vram.fetch_hf_params_b", lambda model_id, **k: 1.2, raising=True)
    info = resolve_model("acme/tiny-1b", "grpo", policy="allow", gpu="RTX 4090")
    assert info.thinking == "unknown"
    assert "open-model policy" in info.notes


def test_allow_policy_blocks_provably_too_big(monkeypatch):
    monkeypatch.setattr(
        "flash.engine.vram.fetch_hf_params_b", lambda model_id, **k: 70.0, raising=True
    )
    with pytest.raises(ValueError, match="does not fit"):
        resolve_model("acme/huge-70b", "sft", policy="allow", gpu="RTX 4090")


def test_allow_policy_unknown_size_warns_but_allows(monkeypatch, capsys):
    monkeypatch.setattr(
        "flash.engine.vram.fetch_hf_params_b", lambda model_id, **k: None, raising=True
    )
    info = resolve_model("acme/mystery", "sft", policy="allow", gpu="RTX 5090")
    assert info.id == "acme/mystery"
    assert "warning" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Estimator sanity: calibrated against catalog anchors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("params_b", "algo", "quant", "gpu", "expected"),
    [
        (4.0, "grpo", "bf16", "RTX 5090", "fits"),  # Qwen3-4B colocate on 32 GB (measured)
        (4.0, "sft", "bf16", "RTX 4090", "fits"),
        (9.65, "sft", "bf16", "RTX 5090", "fits"),  # Qwen3.5-9B SFT
        (36.0, "sft", "4bit-qlora", "RTX 5090", "tight"),  # Qwen3.6-35B-A3B QLoRA
        (36.0, "sft", "bf16", "RTX 5090", "too_big"),  # 72 GB of weights
        (36.0, "grpo", "4bit-qlora", "RTX 5090", "too_big"),  # 2 copies + KV ~55 GB >> 32 GB
    ],
)
def test_estimator_anchors(monkeypatch, params_b, algo, quant, gpu, expected):
    est = check_fit("anchor/model", algo, gpu, quant=quant, params_b=params_b)
    assert isinstance(est, VramEstimate)
    assert est.verdict == expected, est.describe()


def test_grpo_needs_more_than_sft():
    assert estimate_vram_gb(4.0, "grpo") > estimate_vram_gb(4.0, "sft")


# ---------------------------------------------------------------------------
# Config schema plumbing
# ---------------------------------------------------------------------------
def test_spec_model_policy_default_catalog():
    spec = spec_from_dict(_raw())
    assert spec.model_policy == "catalog"


def test_spec_model_policy_allow_via_operator_env(monkeypatch):
    # model_policy is operator-controlled via the FLASH_MODEL_POLICY control-plane env, NOT a user
    # knob: setting the env to "allow" lets a fitting non-catalog model through.
    monkeypatch.setenv("FLASH_MODEL_POLICY", "allow")
    monkeypatch.setattr("flash.engine.vram.fetch_hf_params_b", lambda model_id, **k: 1.0, raising=True)
    spec = spec_from_dict(_raw(model="acme/tiny-1b"))
    assert spec.model_policy == "allow"
    spec2 = spec.from_dict(spec.to_dict())
    assert spec2.model_policy == "allow"


def test_spec_user_model_policy_is_ignored(monkeypatch):
    # A user-supplied model_policy in the config is ignored (the operator env governs); the
    # default policy stays "catalog" even when the user asks for "allow".
    monkeypatch.delenv("FLASH_MODEL_POLICY", raising=False)
    spec = spec_from_dict(_raw(model="Qwen/Qwen3.5-4B", model_policy="allow"))
    assert spec.model_policy == "catalog"


def test_operator_model_policy_invalid(monkeypatch):
    monkeypatch.setenv("FLASH_MODEL_POLICY", "yolo")
    with pytest.raises(ConfigError, match="FLASH_MODEL_POLICY"):
        spec_from_dict(_raw())


def test_spec_unlisted_model_under_catalog_policy_fails():
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(_raw(model="acme/unlisted"))
    assert "model_policy" in str(ei.value)
