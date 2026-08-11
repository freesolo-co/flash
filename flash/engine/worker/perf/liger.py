"""Liger fused-kernel gating for the fine-tuning worker."""

from __future__ import annotations

from flash.engine.plan.vram import _LIGER_MIN_PARAMS_B

# 1B-class models measured net-negative on every arch; only pay off on large models.
_LIGER_MIN_PARAMS = _LIGER_MIN_PARAMS_B * 1e9


def _estimate_params(cfg) -> float:
    """Rough param count from a HF config; reads text_config for multimodal models (e.g. Qwen3.5-VL) to avoid underestimating."""
    tc = getattr(cfg, "text_config", None)
    src = cfg if getattr(cfg, "hidden_size", 0) else (tc or cfg)
    h = getattr(src, "hidden_size", 0) or 0
    v = getattr(src, "vocab_size", 0) or getattr(cfg, "vocab_size", 0) or 0
    n = getattr(src, "num_hidden_layers", 0) or 0
    tied = getattr(src, "tie_word_embeddings", getattr(cfg, "tie_word_embeddings", False))
    emb = v * h * (1 if tied else 2)
    blocks = n * 12 * h * h  # ~12 h^2 per transformer block (attn + MLP)
    return float(emb + blocks)


def _liger_default_for_model(model_id: str, revision: str = "") -> bool:
    """Return True if the model is large enough for Liger's fused-CE to be a net win.

    A cataloged model answers from its own ``params_b`` rather than the config probe below. The
    probe reconstructs a parameter count as ``embeddings + 12 h^2 per layer``, which describes a
    DENSE transformer: it cannot see an expert stack, so it reads Qwen3.6-35B-A3B (35B total) as
    3.03B -- 1% over this threshold, on a model that clears it 11x over. Nothing about that margin
    is load-bearing, and it decides gradient checkpointing (``_memory_mode``), so a catalog whose
    vocab or depth shifted slightly would silently turn GC off on a 35B model.

    The probe is also fail-open for this caller: it returns False on ANY exception, and False here
    means "small model, no checkpointing needed". A network blip or a rate-limited HF read would
    therefore disable gradient checkpointing on a 35B model and OOM the run. The catalog is local,
    exact, and always available, so it answers first; the probe remains for uncataloged models,
    which is the supported way to fork the catalog and add one.

    Resolved through ``resolve_params_b``, the shared worker/cost accessor, so this gate can never
    disagree with the size the allocator and the VRAM equations use. Called without a revision on
    purpose: a pinned commit resolves size by fetching the HF safetensors index, which is the
    network read this short-circuit exists to avoid, and no cataloged model sits near enough to the
    3B threshold for the +/-5% revision tolerance to change this answer.
    """
    from flash.engine.plan.vram import resolve_params_b

    catalog_params_b = resolve_params_b(model_id)
    if catalog_params_b is not None:
        return catalog_params_b >= _LIGER_MIN_PARAMS_B
    try:
        from transformers import AutoConfig

        from flash.engine.worker.io.hf import model_revision_kwargs

        cfg = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            **model_revision_kwargs(revision),
        )
        return _estimate_params(cfg) >= _LIGER_MIN_PARAMS
    except Exception as e:
        print("liger model-size probe failed (default off):", e)
        return False
