"""Tests for flash.engine.plan.vram.resolve_params_b — the single model-size resolver shared by the
worker (run_sft / run_rl) and the cost estimator (cost.spec), so they can never drift on how big a
model is (the >=3B fused-CE gate and the colocate per-device cap both hinge on it)."""

from __future__ import annotations


def test_resolve_params_b_uses_catalog_params_b_float():
    # Curated catalog models return their numeric params_b stat directly (no HF fetch).
    from flash.engine.plan.vram import resolve_params_b

    assert resolve_params_b("Qwen/Qwen3.5-9B") == 9.7
    assert resolve_params_b("Qwen/Qwen3.8-27B") == 27.781427952
    assert resolve_params_b("Qwen/Qwen3.6-35B-A3B") == 35.0


def test_qwen38_exact_parameter_count_reaches_vram_disk_and_cost() -> None:
    from flash.core.catalog import resolve_model
    from flash.cost import RunConfig, estimate_cost
    from flash.cost.analytical import setup_seconds
    from flash.providers.core.allocator import required_vram_gb

    model = "Qwen/Qwen3.8-27B"
    config = RunConfig(
        model,
        "sft",
        10,
        batch_size=4,
        seq_len=1024,
        sft_retained_examples=40,
    )

    assert (
        required_vram_gb(
            model,
            "sft",
            train={"batch_size": 4, "max_context_tokens": 1024, "lora_rank": 32},
        )
        == 80
    )
    assert resolve_model(model, "sft").min_disk_gb == 232
    assert setup_seconds(config) == 583.9071397600001
    quote = estimate_cost(config)
    assert quote.total_usd == 0.045268060397964854
    assert quote.required_vram_gb == 80


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


def test_resolve_params_b_none_when_uncataloged():
    """Best-effort: an uncataloged id returns None rather than inventing a size.

    Submit rejects uncataloged models, so only a stale caller can produce one here. Callers degrade
    to the size-unknown path (memory-safe fused-CE gate, loose colocate cap) rather than raising on
    the allocation path.
    """
    from flash.engine.plan import vram

    assert vram.resolve_params_b("acme/totally-unknown") is None
    assert vram.resolve_params_b("acme/totally-unknown", "a" * 40) is None
