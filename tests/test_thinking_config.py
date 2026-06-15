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


def _raw(model="Qwen/Qwen3-0.6B", **kw):
    d = {
        "model": model,
        "algorithm": "sft",
        "train": {"epochs": 1, "hf_repo": "owner/runs"},
        "environment": {"id": "owner/env"},  # any verifiers/Hub slug (not loaded here)
    }
    d.update(kw)
    return d


def test_thinking_defaults_false():
    spec = spec_from_dict(_raw())
    assert spec.thinking is False


def test_thinking_true_on_hybrid_model():
    spec = spec_from_dict(_raw(thinking=True))  # Qwen3-0.6B is hybrid
    assert spec.thinking is True


def test_thinking_rejected_for_non_thinking_model():
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(_raw(model="Qwen/Qwen3-4B-Instruct-2507", thinking=True))
    assert "thinking" in str(ei.value)


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
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(_raw(model="acme/r1-distill", model_policy="allow"))
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
        experimental=True,
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
    # old persisted statuses without the field default to False
    assert JobSpec.from_dict({"model": "Qwen/Qwen3-0.6B"}).thinking is False


def test_thinking_set_override(tmp_path):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        'model = "Qwen/Qwen3-0.6B"\nalgorithm = "sft"\n\n'
        '[environment]\nid = "owner/env"\n\n[train]\nepochs = 1\nhf_repo = "owner/runs"\n'
    )
    spec = spec_from_file(str(cfg), overrides=["thinking=true"])
    assert spec.thinking is True
