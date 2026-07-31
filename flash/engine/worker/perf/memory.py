"""Memory-mode / grad-checkpointing / optimizer defaults for the fine-tuning worker."""

from __future__ import annotations

from flash.engine.vram import _LIGER_LONG_CTX_TOKENS
from flash.engine.worker.perf.liger import _liger_default_for_model

_LONG_CONTEXT_TOKENS = _LIGER_LONG_CTX_TOKENS


def _memory_mode(model_id: str, max_length: int = 0, *, revision: str = "") -> bool:
    """True for large models or long contexts; False for small+short (optimize for speed)."""
    if max_length and max_length >= _LONG_CONTEXT_TOKENS:
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
    from flash.catalog import MODELS

    info = MODELS.get(model_id)
    params_b = float(getattr(info, "params_b", 0.0) or 0.0) if info else 0.0
    if params_b <= 0:
        return True
    from flash.engine.vram import sft_grad_checkpoint_can_disable

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
    """Offline family check for a Qwen3.5/3.6 GatedDeltaNet hybrid (no network/config probe).

    Every curated model is a Qwen3.5/3.6 GDN hybrid; uncataloged non-Qwen open models are not. Kept
    as a string check so ``grpo_use_reentrant``
    stays pure and hermetic (callable at config-build time, unit-testable without HF access).
    """
    mid = (model_id or "").lower()
    return any(token in mid for token in ("qwen3.5", "qwen3_5", "qwen3.6", "qwen3_6"))


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
    - GDN hybrids (Qwen3.5/3.6 dense): FlashAttention-2 varlen-unpad on the full-attention layers,
      the fused GatedDeltaNet chunk-scan on the linear-attention layers, and chalk's fused Triton
      kernels each save shape-/data-dependent tensors that the non-reentrant metadata-equality check
      can't positionally reconcile (live-confirmed on Qwen3.5-0.8B GRPO / RTX 4090: forward packed
      varlen ``[1636, ...]`` vs recompute padded ``[1024, ...]``). Same failure mode as MoE.

    Reentrant checkpointing re-runs the forward inside the same autograd context over the same
    closed-over inputs (mask/position_ids threaded via the ``partial``; ``use_cache=False``) and does
    NOT assert metadata equality, so it tolerates these recomputes and produces correct gradients.
    Uncataloged non-GDN dense open models keep the faster, lower-overhead non-reentrant path.
    """
    from flash.catalog import MODELS

    info = MODELS.get(model_id)
    if info is not None and info.is_moe:
        return True
    return _is_gdn_hybrid_family(model_id)


def enable_multimodal_input_require_grads(model) -> object | None:
    """Make the vision patch embeddings differentiable for reentrant checkpointing."""
    if model is None:
        return None

    existing_handle = getattr(model, "_mm_vision_require_grads_hook", None)
    if existing_handle is not None:
        existing_handle.remove()
        model._mm_vision_require_grads_hook = None

    get_base_model = getattr(model, "get_base_model", None)
    if callable(get_base_model):
        base_model = get_base_model()
    else:
        peft_base_model = getattr(model, "base_model", None)
        base_model = getattr(peft_base_model, "model", model)

    for module_path, module in base_model.named_modules():
        if not module_path.endswith("visual.patch_embed"):
            continue

        def _require_output_grad(_module, _inputs, output):
            import torch

            tensor = output[0] if isinstance(output, tuple) and output else output
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                tensor.requires_grad_(True)
            return

        handle = module.register_forward_hook(_require_output_grad)
        model._mm_vision_require_grads_hook = handle
        print(f"[multimodal] vision input gradients enabled at {module_path}")
        return handle

    return None


def make_multimodal_input_require_grads_callback():
    """Return a trainer callback that installs the vision input-gradient hook at train start."""
    from transformers import TrainerCallback

    class _MultimodalInputRequireGrads(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            enable_multimodal_input_require_grads(kwargs.get("model"))
            return control

    return _MultimodalInputRequireGrads()


def fused_optim_name() -> str:
    """8-bit paged AdamW: optimizer state paged to host RAM, fits smaller GPUs."""
    return "paged_adamw_8bit"
