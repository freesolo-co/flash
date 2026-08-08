"""Tests for flash.engine.plan.vram.resolve_params_b — the single model-size resolver shared by the
worker (run_sft / run_rl) and the cost estimator (cost.spec), so they can never drift on how big a
model is (the >=3B fused-CE gate and the colocate per-device cap both hinge on it)."""

from __future__ import annotations


def test_resolve_params_b_uses_catalog_params_b_float():
    # Curated catalog models return their numeric params_b stat directly (no HF fetch).
    from flash.engine.plan.vram import resolve_params_b

    assert resolve_params_b("Qwen/Qwen3.5-0.8B") == 0.9
    assert resolve_params_b("Qwen/Qwen3.5-4B") == 4.7
    assert resolve_params_b("Qwen/Qwen3.5-9B") == 9.7
    assert resolve_params_b("Qwen/Qwen3.6-27B") == 27.0


def test_resolve_params_b_pinned_revision_reads_the_pinned_size(monkeypatch):
    """A PIN resolves to the pinned commit's real size, not the catalog's default-revision stat.

    The catalog states a size for its default revision only. A run pinned to an older or variant
    commit can genuinely differ, and sizing it from the catalog would quote the wrong model -- so the
    pinned path fetches real geometry and the catalog number must NOT win.
    """
    from flash.engine.plan import vram

    monkeypatch.setattr(
        vram, "_validated_revision_geometry", lambda _mid, _rev, _info: (9.1, 151936)
    )
    assert vram.resolve_params_b("Qwen/Qwen3.5-9B", "a" * 40) == 9.1
    # ...and the unpinned call still reads the catalog, so the assert above is a real difference.
    assert vram.resolve_params_b("Qwen/Qwen3.5-9B") == 9.7


def test_resolve_params_b_none_when_uncataloged(monkeypatch):
    """Best-effort: an uncataloged id returns None, and never reaches the network to guess.

    Submit rejects uncataloged models, so only a stale caller can produce one here. Callers degrade
    to the size-unknown path (memory-safe fused-CE gate, loose colocate cap) rather than raising on
    the allocation path -- but a size must never be invented for an id the catalog does not state.
    """
    from flash.engine.plan import vram

    def _boom(*_args, **_kwargs):
        raise AssertionError("an uncataloged id must not be sized over the network")

    monkeypatch.setattr(vram, "fetch_hf_params_b", _boom, raising=False)
    assert vram.resolve_params_b("acme/totally-unknown") is None
    assert vram.resolve_params_b("acme/totally-unknown", "a" * 40) is None
