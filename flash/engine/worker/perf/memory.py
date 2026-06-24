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


def fused_optim_name() -> str:
    """TRL/HF ``optim`` value: 8-bit paged AdamW (bitsandbytes int8 optimizer state paged to host
    RAM). It fits a smaller/cheaper GPU and is the better default across the catalog."""
    return "paged_adamw_8bit"
