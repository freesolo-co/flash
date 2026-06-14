"""Open-model policy + VRAM estimator unit tests (CPU-only, no network)."""

from __future__ import annotations

import pytest

from autoslm.catalog import resolve_model
from autoslm.config_schema import ConfigError, spec_from_dict
from autoslm.engine.vram import VramEstimate, check_fit, estimate_vram_gb


def test_catalog_policy_rejects_unlisted_with_hint():
    with pytest.raises(ValueError, match="model_policy"):
        resolve_model("some-org/some-model", "sft", policy="catalog")


def test_catalog_model_resolves_normally():
    info = resolve_model("Qwen/Qwen3-4B-Instruct-2507", "grpo", policy="catalog")
    assert info.id == "Qwen/Qwen3-4B-Instruct-2507"


def test_minicpm5_in_catalog():
    info = resolve_model("openbmb/MiniCPM5-1B", "grpo", policy="catalog")
    assert info.recommended_gpu == "RTX 4090"
    assert "grpo" in info.algos
    assert "sft" in info.algos


def test_allow_policy_accepts_fitting_model(monkeypatch):
    monkeypatch.setattr("autoslm.engine.vram.fetch_hf_params_b", lambda model_id: 1.2, raising=True)
    info = resolve_model("acme/tiny-1b", "grpo", policy="allow", gpu="RTX 4090")
    assert info.experimental
    assert "open-model policy" in info.notes


def test_allow_policy_blocks_provably_too_big(monkeypatch):
    monkeypatch.setattr(
        "autoslm.engine.vram.fetch_hf_params_b", lambda model_id: 70.0, raising=True
    )
    with pytest.raises(ValueError, match="does not fit"):
        resolve_model("acme/huge-70b", "sft", policy="allow", gpu="RTX 4090")


def test_allow_policy_unknown_size_warns_but_allows(monkeypatch, capsys):
    monkeypatch.setattr(
        "autoslm.engine.vram.fetch_hf_params_b", lambda model_id: None, raising=True
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
        (36.0, "sft", "4bit-qlora", "RTX 5090", "fits"),  # Qwen3.6-35B-A3B QLoRA
        (36.0, "sft", "bf16", "RTX 5090", "too_big"),  # 72 GB of weights
        (36.0, "grpo", "4bit-qlora", "RTX 5090", "tight"),  # excluded tier, est ~33 GB
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
def _raw(model="Qwen/Qwen3-0.6B", **kw):
    d = {
        "model": model,
        "algorithm": "sft",
        "train": {"epochs": 1},
        "environment": {"id": "owner/env"},  # any verifiers/Hub slug (not loaded here)
    }
    d.update(kw)
    return d


def test_spec_model_policy_default_catalog():
    spec = spec_from_dict(_raw())
    assert spec.model_policy == "catalog"


def test_spec_model_policy_allow_roundtrip(monkeypatch):
    monkeypatch.setattr("autoslm.engine.vram.fetch_hf_params_b", lambda model_id: 1.0, raising=True)
    spec = spec_from_dict(_raw(model="acme/tiny-1b", model_policy="allow"))
    assert spec.model_policy == "allow"
    spec2 = spec.from_dict(spec.to_dict())
    assert spec2.model_policy == "allow"


def test_spec_model_policy_invalid():
    with pytest.raises(ConfigError):
        spec_from_dict(_raw(model_policy="yolo"))


def test_spec_unlisted_model_under_catalog_policy_fails():
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(_raw(model="acme/unlisted"))
    assert "model_policy" in str(ei.value)
