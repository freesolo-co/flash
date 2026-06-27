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


def grad_checkpointing_on(model_id: str, max_length: int = 0) -> bool:
    """ON for large models / long context; OFF for small+short (trades ~25% speed for activation memory)."""
    return _memory_mode(model_id, max_length)


def grpo_sleep_mode(
    model_id: str,
    *,
    max_length: int = 0,
    group_size: int = 8,
    max_tokens: int | None = None,
    lora_rank: int = 32,
    thinking: bool = False,
    card_vram_gb: float = 0.0,
) -> bool:
    """Whether colocated-vLLM GRPO should enable vLLM sleep mode (offload rollout engine between steps).

    Sleep/wake stalls the rollout on large models — only enable when the run genuinely can't fit resident."""
    from flash.engine.vram import grpo_fits_resident, grpo_rollout_seq_len

    # Use the actual rollout seq_len (not raw max_length): a 0 max_length would skip the resident-fit check for long max_tokens runs.
    seq_len = grpo_rollout_seq_len(max_length, max_tokens, thinking)
    if not _memory_mode(model_id, seq_len):
        return False
    if card_vram_gb and card_vram_gb > 0:
        try:
            if grpo_fits_resident(
                model_id,
                seq_len=seq_len,
                max_tokens=max_tokens,
                lora_rank=lora_rank,
                group_size=group_size,
                thinking=thinking,
                card_vram_gb=card_vram_gb,
            ):
                return False  # fits resident -> skip sleep/wake (buggy on large-model GRPO path)
        except Exception as e:
            print("[rl] grpo sleep-mode resident check skipped:", e)
    return True


def fused_optim_name() -> str:
    """8-bit paged AdamW: optimizer state paged to host RAM, fits smaller GPUs."""
    return "paged_adamw_8bit"
