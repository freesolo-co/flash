"""Tests for flash.engine.vram.resolve_params_b — the single model-size resolver shared by the
worker (run_sft / run_rl) and the cost estimator (cost.spec), so they can never drift on how big a
model is (the >=3B fused-CE gate and the colocate per-device cap both hinge on it)."""

from __future__ import annotations


def test_resolve_params_b_uses_catalog_params_b_float():
    # Curated catalog models return their numeric params_b stat directly (no HF fetch).
    from flash.engine.vram import resolve_params_b

    assert resolve_params_b("Qwen/Qwen3.5-0.8B") == 0.9
    assert resolve_params_b("Qwen/Qwen3.5-4B") == 4.7
    assert resolve_params_b("Qwen/Qwen3.5-9B") == 9.7


def test_resolve_params_b_falls_back_to_params_string(monkeypatch):
    """A catalog entry lacking the numeric params_b but carrying a ``params`` display string is
    parsed (run_sft's fallback) — and the HF fetch is NOT consulted."""
    import types

    import flash.catalog as catalog
    from flash.engine import vram

    fake = types.SimpleNamespace(params_b=0.0, params="6.5B (text-only fine-tune)")
    monkeypatch.setitem(catalog.MODELS, "acme/with-string", fake)
    calls = []
    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda mid: calls.append(mid))
    assert vram.resolve_params_b("acme/with-string") == 6.5
    assert calls == []  # the string parse short-circuits the HF fetch


def test_resolve_params_b_open_model_uses_hf_metadata(monkeypatch):
    """An uncataloged (open-policy) model has no catalog entry -> fetch the real HF safetensors
    param count."""
    from flash.engine import vram

    monkeypatch.setattr(
        vram, "fetch_hf_params_b", lambda mid: 7.2 if mid == "acme/open-7b" else None
    )
    assert vram.resolve_params_b("acme/open-7b") == 7.2


def test_resolve_params_b_none_when_uncataloged_and_no_network(monkeypatch):
    """Best-effort: an uncataloged model with no HF metadata returns None (callers degrade to the
    size-unknown path — memory-safe fused-CE gate, loose colocate cap), never an error."""
    from flash.engine import vram

    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda mid: None)
    assert vram.resolve_params_b("acme/totally-unknown") is None
