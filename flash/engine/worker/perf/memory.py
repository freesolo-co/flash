"""Memory-mode / grad-checkpointing / optimizer defaults for the fine-tuning worker.

"Memory mode" = a large model (fused-CE memory win) OR a long context (activations/KV dominate).
The long-context threshold is the canonical value in ``flash.engine.vram``; the model-size half
reuses the Liger size gate. Heavy deps stay lazy so this is CPU-importable.
"""

from __future__ import annotations

# canonical value in flash.engine.vram (single source of truth, shared with the cost estimator).
from flash.engine.vram import _LIGER_LONG_CTX_TOKENS
from flash.engine.worker.perf.liger import _liger_default_for_model

# Long-context runs are memory-bound (activations + vLLM KV cache scale with sequence length), so
# they need the memory features even on a SMALL model — PR #174 measured a 1B model OOM on GRPO at
# 4096 ctx in speed mode, but it fits in memory mode. So "memory mode" = large model OR long ctx.
_LONG_CONTEXT_TOKENS = _LIGER_LONG_CTX_TOKENS  # canonical value in flash.engine.vram


def _memory_mode(model_id: str, max_length: int = 0) -> bool:
    """Whether to default the memory-saving features (Liger, grad-checkpointing, vLLM sleep) ON:
    a large model (fused-CE memory win) OR a long context (activations/KV dominate). Small model +
    short context -> off (optimize for speed)."""
    if max_length and max_length >= _LONG_CONTEXT_TOKENS:
        return True
    return _liger_default_for_model(model_id)


def grad_checkpointing_on(model_id: str, max_length: int = 0) -> bool:
    """Gradient checkpointing recomputes the forward in backward (~25% slower) to save activation
    memory — a MEMORY feature, not speed. ON for large models / long context that need the
    headroom; OFF for small+short runs that fit without it (the speed win)."""
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
    """Whether colocated-vLLM GRPO should enable vLLM sleep mode (offload the rollout engine
    between steps).

    Sleep mode trades a large per-step cost for memory, and on the large-model GRPO path the
    sleep/wake cycle STALLS the colocated rollout (the rollout produces unparseable completions and
    then the worker hangs). So enable it ONLY when the run genuinely can't fit RESIDENT on the card:
    when the policy + colocated rollout engine + training peak all fit on ``card_vram_gb`` (the
    common case on an allocator-sized card), skip sleep mode entirely. Falls back to the
    size/context gate (``_memory_mode``) when the card VRAM is unknown."""
    if not _memory_mode(model_id, max_length):
        return False  # small / short-context -> never needed
    if card_vram_gb and card_vram_gb > 0:
        try:
            from flash.engine.vram import grpo_fits_resident, grpo_rollout_seq_len

            if grpo_fits_resident(
                model_id,
                # Size the resident-fit check to the engine context run_rl() actually launches
                # (max(1024, prompt+completion) when [train].max_length is unset, 2368 default / 3584
                # thinking), NOT a flat 1024 -- otherwise a marginal card is wrongly told the run fits
                # resident and sleep is disabled, risking an OOM on the real longer rollout.
                seq_len=grpo_rollout_seq_len(max_length, max_tokens, thinking),
                max_tokens=max_tokens,
                lora_rank=lora_rank,
                group_size=group_size,
                thinking=thinking,
                card_vram_gb=card_vram_gb,
            ):
                return False  # fits resident -> skip the (buggy, slow) sleep/wake cycle
        except Exception as e:
            print("[rl] grpo sleep-mode resident check skipped:", e)
    return True


def fused_optim_name() -> str:
    """TRL/HF ``optim`` value: 8-bit paged AdamW (bitsandbytes int8 optimizer state paged to host
    RAM). It fits a smaller/cheaper GPU and is the better default across the catalog."""
    return "paged_adamw_8bit"
