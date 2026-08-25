"""Memory-mode / grad-checkpointing / optimizer defaults for the fine-tuning worker."""

from __future__ import annotations

from flash.engine.plan.vram import _LIGER_LONG_CTX_TOKENS
from flash.engine.worker.perf.liger import _liger_default_for_model


def _memory_mode(model_id: str, max_length: int = 0, *, revision: str = "") -> bool:
    """True for large models or long contexts; False for small+short (optimize for speed)."""
    if max_length and max_length >= _LIGER_LONG_CTX_TOKENS:
        return True
    return _liger_default_for_model(model_id, revision=revision)


def grad_checkpointing_on(
    model_id: str,
    max_length: int = 0,
    *,
    allow_disable: bool = False,
    card_vram_gb: float = 0.0,
    capability: tuple[int, int] | None = None,
    active_params_b: float | None = None,
    hidden: int = 0,
    num_layers: int = 0,
    fused_ce: bool = False,
    per_device_bs: int = 0,
    lora_rank: int = 32,
    revision: str = "",
) -> bool:
    """Gradient-checkpointing default, with an SFT-only opt-out when GC-off fits."""
    base = _memory_mode(model_id, max_length, revision=revision)
    if not allow_disable:
        return base
    if not base:
        return False
    can_consider_gc_off = (
        fused_ce
        and active_params_b
        and card_vram_gb
        and card_vram_gb >= 120.0
        and (capability is None or capability >= (9, 0))
    )
    if not can_consider_gc_off:
        return True
    from flash.core.catalog import MODELS

    info = MODELS.get(model_id)
    params_b = float(getattr(info, "params_b", 0.0) or 0.0) if info else 0.0
    if params_b <= 0:
        return True
    from flash.engine.plan.vram import sft_grad_checkpoint_can_disable

    if sft_grad_checkpoint_can_disable(
        params_b,
        active_params_b=active_params_b,
        seq_len=max_length,
        hidden=hidden,
        num_layers=num_layers,
        card_vram_gb=card_vram_gb,
        batch=per_device_bs or 4,
        lora_rank=lora_rank,
        quant=(getattr(info, "quant", "bf16") or "bf16") if info else "bf16",
        model_info=info,
    ):
        print(
            f"[sft] gradient checkpointing OFF: GC-off peak fits {card_vram_gb:.0f} GB "
            f"(active={active_params_b}B, seq={max_length}, {num_layers}L x {hidden}h, fused CE)"
        )
        return False
    return True


def _is_gdn_hybrid_family(model_id: str) -> bool:
    """Minimal offline fallback for an uncataloged Qwen GatedDeltaNet fork."""
    mid = (model_id or "").lower()
    return any(
        token in mid for token in ("qwen3.5", "qwen3_5", "qwen3.6", "qwen3_6", "qwen3.8", "qwen3_8")
    )


def grpo_use_reentrant(model_id: str) -> bool:
    """Whether GRPO gradient checkpointing must use REENTRANT recompute for this model.

    MoE models AND GatedDeltaNet (GDN) hybrids need it. Non-reentrant checkpointing
    (``use_reentrant=False``) asserts that every recomputed activation's metadata matches the
    forward pass and dies on the FIRST backward — before a single optimizer step — with
    ``torch.utils.checkpoint: Recomputed values ... different metadata`` whenever a decoder layer
    contains a custom, data-dependent kernel whose saved-for-backward tensors the recompute lays out
    differently:

    - MoE (Qwen3.6-35B-A3B): the router re-dispatches tokens on recompute, so the grouped
      expert-buffer shapes differ (forward expert-dispatch tokens 28192 vs recompute 3524 ==
      group_size x). This is what #429 fixed.
    - GDN hybrids (Qwen3.5/3.6/3.8): FlashAttention-2 varlen-unpad on the full-attention layers,
      the fused GatedDeltaNet chunk-scan on the linear-attention layers, and the fused Triton
      kernels each save shape-/data-dependent tensors that the non-reentrant metadata-equality check
      can't positionally reconcile (live-confirmed on Qwen3.5-0.8B GRPO / RTX 4090: forward packed
      varlen ``[1636, ...]`` vs recompute padded ``[1024, ...]``). Same failure mode as MoE.

    Reentrant checkpointing re-runs the forward inside the same autograd context over the same
    closed-over inputs (mask/position_ids threaded via the ``partial``; ``use_cache=False``) and does
    NOT assert metadata equality, so it tolerates these recomputes and produces correct gradients.
    A non-MoE, non-GDN dense model keeps the faster, lower-overhead non-reentrant path (no catalog
    entry is one today; see ``_is_gdn_hybrid_family``).
    """
    from flash.core.catalog import MODELS

    info = MODELS.get(model_id)
    if info is not None:
        return info.is_moe or info.num_linear_attention_layers > 0
    return _is_gdn_hybrid_family(model_id)
