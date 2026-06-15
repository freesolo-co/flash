"""Thinking-mode config plumbing tests (CPU-only, no network).

The `thinking` flag is a single per-run boolean: parsed/validated in spec_from_dict
against the catalog's per-model capability, carried by JobSpec end to end.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autoslm.catalog import ModelInfo
from autoslm.schema import ConfigError, spec_from_dict, spec_from_file
from autoslm.spec import JobSpec


def _raw(model="Qwen/Qwen3.5-0.8B", **kw):
    d = {
        "model": model,
        "algorithm": "sft",
        "train": {"epochs": 1, "hf_repo": "owner/runs"},
        "environment": {"id": "owner/env"},  # any verifiers/Hub slug (not loaded here)
    }
    d.update(kw)
    return d


def test_thinking_defaults_true():
    spec = spec_from_dict(_raw())  # reasoning mode is ON by default
    assert spec.thinking is True


def test_thinking_can_be_disabled():
    spec = spec_from_dict(_raw(thinking=False))
    assert spec.thinking is False


def test_thinking_true_on_hybrid_model():
    spec = spec_from_dict(_raw(thinking=True))  # Qwen3.5-0.8B is hybrid
    assert spec.thinking is True


def test_thinking_rejected_for_non_thinking_model():
    # No catalog entry is thinking="none" anymore, so inject a temporary one to exercise
    # the "this model can't think" rejection path.
    from autoslm import catalog
    from autoslm.catalog import ModelInfo

    catalog.MODELS["test/none-think"] = ModelInfo(
        id="test/none-think",
        display_name="none",
        params="1B",
        algos=("sft", "grpo"),
        min_vram_gb=12,
        thinking="none",
    )
    try:
        with pytest.raises(ConfigError) as ei:
            spec_from_dict(_raw(model="test/none-think", thinking=True))
        assert "thinking" in str(ei.value)
    finally:
        catalog.MODELS.pop("test/none-think", None)


def test_always_thinking_model_requires_flag(monkeypatch):
    # No curated always-thinker yet; simulate one via the resolver. Stub the GPU policy
    # too so the unlisted id never triggers the network-backed open-model sizing path.
    info = ModelInfo(
        id="acme/r1-distill",
        display_name="acme r1",
        params="1.5B",
        algos=("sft", "grpo"),
        min_vram_gb=12,
        thinking="always",
    )
    monkeypatch.setattr("autoslm.schema.resolve_model", lambda *a, **k: info)
    monkeypatch.setattr("autoslm.schema.resolve_gpu_policy", lambda *a, **k: "RTX 5090")
    # An always-thinker can't run with thinking explicitly OFF (thinking now defaults ON, so
    # the rejection only triggers on an explicit thinking=false).
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(_raw(model="acme/r1-distill", model_policy="allow", thinking=False))
    assert "thinking = true" in str(ei.value)
    spec = spec_from_dict(_raw(model="acme/r1-distill", model_policy="allow", thinking=True))
    assert spec.thinking is True


def test_thinking_unknown_capability_warns_but_allows(monkeypatch, capsys):
    # Open-model-policy entries resolve to thinking="unknown": the run proceeds with a
    # warning rather than a hard error. Stub the resolver + GPU policy (no network).
    info = ModelInfo(
        id="acme/tiny-1b",
        display_name="acme tiny",
        params="1B",
        algos=("sft", "grpo"),
        min_vram_gb=12,
        thinking="unknown",
    )
    monkeypatch.setattr("autoslm.schema.resolve_model", lambda *a, **k: info)
    monkeypatch.setattr("autoslm.schema.resolve_gpu_policy", lambda *a, **k: "RTX 5090")
    spec = spec_from_dict(_raw(model="acme/tiny-1b", model_policy="allow", thinking=True))
    assert spec.thinking is True
    assert "warning" in capsys.readouterr().out


def test_thinking_must_be_boolean():
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(_raw(thinking="yes"))
    assert "boolean" in str(ei.value)


def test_thinking_roundtrips_through_dict():
    spec = spec_from_dict(_raw(thinking=True))
    spec2 = JobSpec.from_dict(spec.to_dict())
    assert spec2.thinking is True
    # a dict without the field gets the default (now True)
    assert JobSpec.from_dict({"model": "Qwen/Qwen3.5-0.8B"}).thinking is True


def test_thinking_set_override(tmp_path):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        'model = "Qwen/Qwen3.5-0.8B"\nalgorithm = "sft"\n\n'
        '[environment]\nid = "owner/env"\n\n[train]\nepochs = 1\nhf_repo = "owner/runs"\n'
    )
    spec = spec_from_file(str(cfg), overrides=["thinking=true"])
    assert spec.thinking is True
