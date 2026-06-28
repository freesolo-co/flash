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


def test_resolve_params_b_zero_falls_through_to_hf_not_string(monkeypatch):
    """An entry with params_b=0.0 (no curated size) falls through to the HF fetch — the ``params``
    display string is NOT parsed. params_b_from_str was removed; the string is display-only now, so a
    "6.5B" display must NOT be mistaken for the size (the HF metadata is the only fallback)."""
    import types

    import flash.catalog as catalog
    from flash.engine import vram

    fake = types.SimpleNamespace(params_b=0.0, params="6.5B (text-only fine-tune)")
    monkeypatch.setitem(catalog.MODELS, "acme/with-string", fake)
    calls = []
    monkeypatch.setattr(vram, "fetch_hf_params_b", lambda mid: calls.append(mid) or 7.0)
    # 7.0 from the HF fetch, NOT 6.5 from the display string (which is now ignored).
    assert vram.resolve_params_b("acme/with-string") == 7.0
    assert calls == ["acme/with-string"]  # the HF fetch IS consulted (string no longer short-circuits)


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
