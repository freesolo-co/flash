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
    """Gradient checkpointing recomputes the forward in backward (one extra forward ~= +33% compute)
    to save activation memory — a MEMORY feature, not speed. ON for large models / long context that
    need the headroom; OFF for small+short runs that fit without it (the speed win).

    SFT GC-OFF opportunity (``allow_disable=True``, the SFT path only — NOT the colocated GRPO
    trainer at rl.py, where two ~70 GB weight copies + KV leave no room to drop GC): for a big MoE
    whose ACTIVE backbone is cheap (35B-A3B) on a BIG datacenter card (>=120 GB H200/B200, sm90+),
    with fused CE removing the [B,T,vocab] logit spike, the no-recompute activations FIT with margin —
    so GC is pure tax. Turn it OFF when ``sft_grad_checkpoint_can_disable`` confirms the GC-off peak
    fits the REAL card (the allocator sized for the GC-*on* peak, so this is the independent guard).
    Legacy / GRPO callers (``allow_disable=False``) keep the exact old behavior."""
    base = _memory_mode(model_id, max_length)
    if not allow_disable:
        return base  # GRPO trainer + legacy callers: unchanged (memory-mode default)
    if not base:
        return False  # small + short -> already off (the speed default)
    # Auto GC-off: big MoE (cheap active backbone) + fused CE + big datacenter card + the GC-off peak
    # fits with margin. Every clause is conservative; any unknown keeps GC ON.
    if (
        fused_ce
        and active_params_b
        and card_vram_gb
        and card_vram_gb >= 120.0
        and (capability is None or capability >= (9, 0))
    ):
        from flash.catalog import MODELS

        info = MODELS.get(model_id)
        params_b = float(getattr(info, "params_b", 0.0) or 0.0) if info else 0.0
        quant = (getattr(info, "quant", "bf16") or "bf16") if info else "bf16"
        if params_b > 0:
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
                quant=quant,
            ):
                print(
                    f"[sft] gradient checkpointing OFF: GC-off peak fits {card_vram_gb:.0f} GB "
                    f"(active={active_params_b}B, seq={max_length}, {num_layers}L x {hidden}h, "
                    "fused CE) -> drop the ~+33% recompute tax"
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
    """Whether colocated-vLLM GRPO should enable vLLM sleep mode (offload the rollout engine
    between steps).

    Sleep mode trades a large per-step cost for memory, and on the large-model GRPO path the
    sleep/wake cycle STALLS the colocated rollout (the rollout produces unparseable completions and
    then the worker hangs). So enable it ONLY when the run genuinely can't fit RESIDENT on the card:
    when the policy + colocated rollout engine + training peak all fit on ``card_vram_gb`` (the
    common case on an allocator-sized card), skip sleep mode entirely. Falls back to the
    size/context gate (``_memory_mode``) when the card VRAM is unknown."""
    # Gate on the rollout length run_rl() ACTUALLY launches (max(1024, prompt+completion) when
    # [train].max_length is unset -- 2368 default / 3584 thinking), NOT the raw max_length. With
    # max_length unset (0) the size/context pre-filter would see a 0-length "short" run and early-
    # exit for a sub-3B model, skipping the resident-fit check that a long max_tokens rollout needs.
    from flash.catalog import MODELS
    from flash.engine.vram import grpo_fits_resident, grpo_rollout_seq_len

    _info = MODELS.get(model_id)
    _sleep_broken = bool(_info is not None and getattr(_info, "sleep_unsupported", False))
    seq_len = grpo_rollout_seq_len(max_length, max_tokens, thinking)
    if not _memory_mode(model_id, seq_len) and not _sleep_broken:
        return False  # small model AND genuinely short rollout -> never needed
    _fits = None  # None = couldn't check (no card info)
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
    """TRL/HF ``optim`` value: 8-bit paged AdamW (bitsandbytes int8 optimizer state paged to host
    RAM). It fits a smaller/cheaper GPU and is the better default across the catalog."""
    return "paged_adamw_8bit"
