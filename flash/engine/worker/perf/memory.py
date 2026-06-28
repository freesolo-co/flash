"""Memory-mode / grad-checkpointing / optimizer defaults for the fine-tuning worker."""

from __future__ import annotations

from flash.engine.vram import _LIGER_LONG_CTX_TOKENS
from flash.engine.worker.perf.liger import _liger_default_for_model

_LONG_CONTEXT_TOKENS = _LIGER_LONG_CTX_TOKENS


def _memory_mode(model_id: str, max_length: int = 0) -> bool:
    """True for large models or long contexts; False for small+short (optimize for speed)."""
    if max_length and max_length >= _LONG_CONTEXT_TOKENS:
        return True
    return _liger_default_for_model(model_id)


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
) -> bool:
    """Gradient-checkpointing default, with an SFT-only opt-out when GC-off fits."""
    base = _memory_mode(model_id, max_length)
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
    ):
        print(
            f"[sft] gradient checkpointing OFF: GC-off peak fits {card_vram_gb:.0f} GB "
            f"(active={active_params_b}B, seq={max_length}, {num_layers}L x {hidden}h, fused CE)"
        )
        return False
    return True


def grpo_sleep_mode(
    model_id: str,
    *,
    max_length: int = 0,
    group_size: int = 8,
    max_tokens: int | None = None,
    lora_rank: int = 32,
    thinking: bool = False,
    card_vram_gb: float = 0.0,
    fp8_kv: bool = False,
) -> bool:
    """Whether colocated-vLLM GRPO should offload the rollout engine between steps."""
    from flash.catalog import MODELS
    from flash.engine.vram import grpo_fits_resident, grpo_rollout_seq_len

    _info = MODELS.get(model_id)
    _sleep_broken = bool(_info is not None and getattr(_info, "sleep_unsupported", False))
    seq_len = grpo_rollout_seq_len(max_length, max_tokens, thinking)
    if not _memory_mode(model_id, seq_len) and not _sleep_broken:
        return False
    _fits = None
    if card_vram_gb and card_vram_gb > 0:
        try:
            _fits = grpo_fits_resident(
                model_id,
                seq_len=seq_len,
                max_tokens=max_tokens,
                lora_rank=lora_rank,
                group_size=group_size,
                thinking=thinking,
                card_vram_gb=card_vram_gb,
                fp8_kv=fp8_kv,
            )
        except Exception as e:
            print("[rl] grpo sleep-mode resident check skipped:", e)
    if _fits:
        return False  # fits resident -> skip the (buggy, slow) sleep/wake cycle
    if _sleep_broken:
        # vLLM sleep is NON-FUNCTIONAL for this model (the wake/reload HANGS -- see ModelInfo
        # .sleep_unsupported). NEVER return True (that would route to the hang). If we positively know
        # it doesn't fit resident, REJECT with a clear error; if we couldn't check (no card info),
        # attempt RESIDENT anyway -- a possible OOM beats a guaranteed wake-hang.
        if _fits is False:
            raise ValueError(
                f"{model_id}: GRPO config (engine context ~{seq_len} tok, group={group_size}) does NOT "
                f"fit RESIDENT on {card_vram_gb:.0f} GB and vLLM sleep mode HANGS this model "
                f"(resident-only). Reduce [train].max_length and/or group_size to fit resident."
            )
        return False
    return True


def fused_optim_name() -> str:
    """8-bit paged AdamW: optimizer state paged to host RAM, fits smaller GPUs."""
    return "paged_adamw_8bit"
