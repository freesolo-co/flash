"""Thinking-mode config plumbing tests (CPU-only, no network).

The `thinking` flag is a single per-run boolean: parsed/validated in spec_from_dict
against the catalog's per-model capability, carried by JobSpec end to end.
"""

from __future__ import annotations

import pytest

from flash.core.catalog import ModelInfo
from flash.core.spec import JobSpec
from flash.schema import ConfigError, spec_and_train_keys_from_file, spec_from_dict
from tests._helpers.specs import raw_spec as _raw


def test_thinking_defaults_false():
    spec = spec_from_dict(_raw())  # reasoning mode is OFF by default (operator preference)
    assert spec.thinking is False


def test_thinking_can_be_disabled():
    spec = spec_from_dict(_raw(thinking=False))
    assert spec.thinking is False


def test_thinking_true_on_hybrid_model():
    spec = spec_from_dict(_raw(thinking=True))  # Qwen3.5-0.8B is hybrid
    assert spec.thinking is True


def test_thinking_rejected_for_non_thinking_model():
    # No catalog entry is thinking="none" anymore, so inject a temporary one to exercise
    # the "this model can't think" rejection path.
    from flash.core import catalog
    from flash.core.catalog import ModelInfo

    catalog.MODELS["test/none-think"] = ModelInfo(
        id="test/none-think",
        display_name="none",
        params="1B",
        params_b=1.0,
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
    # No curated always-thinker yet; simulate one via the resolver. Stub the provisional GPU
    # sizing too, so the unlisted id is never resolved through the real catalog path.
    info = ModelInfo(
        id="acme/r1-distill",
        display_name="acme r1",
        params="1.5B",
        params_b=1.5,
        algos=("sft", "grpo"),
        min_vram_gb=12,
        thinking="always",
    )
    monkeypatch.setattr("flash.schema.resolve_model", lambda *a, **k: info)
    monkeypatch.setattr("flash.schema.provisional_gpu", lambda *a, **k: "RTX 5090")
    # An always-thinker can't run with thinking OFF (the default); the rejection triggers on the
    # default-off as well as an explicit thinking=false.
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(_raw(model="acme/r1-distill", thinking=False))
    assert "thinking = true" in str(ei.value)
    spec = spec_from_dict(_raw(model="acme/r1-distill", thinking=True))
    assert spec.thinking is True


def test_every_catalog_entry_states_a_thinking_capability():
    """No trainable model has an UNKNOWN thinking capability, so nothing is admitted on a warning.

    `thinking="unknown"` was produced by exactly one thing: the open-model path synthesizing a
    ModelInfo for an id nobody had curated. Its chat template was never inspected, so the parser
    admitted the run with a warning instead of a verdict. Every trainable model is now curated and
    states its capability, which is what lets the two hard rejections above BE hard.
    """
    from flash.core.catalog import MODELS

    for model_id, info in MODELS.items():
        assert info.thinking in ("none", "always", "hybrid"), (
            f"{model_id} states thinking={info.thinking!r}; a curated entry must state a real "
            f"capability so the parser can accept or reject rather than warn"
        )


def test_thinking_must_be_boolean():
    with pytest.raises(ConfigError) as ei:
        spec_from_dict(_raw(thinking="yes"))
    assert "boolean" in str(ei.value)


def test_thinking_roundtrips_through_dict():
    spec = spec_from_dict(_raw(thinking=True))
    spec2 = JobSpec.from_dict(spec.to_dict())
    assert spec2.thinking is True
    # a dict without the field gets the default (OFF)
    assert JobSpec.from_dict({"model": "Qwen/Qwen3.5-9B"}).thinking is False


def test_thinking_set_override(tmp_path):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        'model = "Qwen/Qwen3.5-9B"\nalgorithm = "sft"\n\n'
        '[environment]\nid = "github:owner/repo@main:env/environment.py"\n\n'
        "[train]\nepochs = 1\nmax_examples = 8\n"
    )
    spec = spec_and_train_keys_from_file(str(cfg), overrides=["thinking=true"])[0]
    assert spec.thinking is True
