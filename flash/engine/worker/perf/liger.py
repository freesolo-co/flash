"""Liger fused-kernel gating for the fine-tuning worker."""

from __future__ import annotations

from flash.engine.vram import _LIGER_MIN_PARAMS_B

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


def _liger_default_for_model(model_id: str) -> bool:
    """Return True if the model is large enough for Liger's fused-CE to be a net win."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        return _estimate_params(cfg) >= _LIGER_MIN_PARAMS
    except Exception as e:
        print("liger model-size probe failed (default off):", e)
        return False


def liger_on(default_on: bool) -> bool:
    """Whether to enable Liger. Requires CUDA and liger_kernel importable — flash[gpu] doesn't ship it."""
    if not default_on:
        return False
    try:
        import importlib.util

        import torch

        return bool(
            torch.cuda.is_available() and importlib.util.find_spec("liger_kernel") is not None
        )
    except Exception:
        return False
