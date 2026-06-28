"""LoRA+ optimizer-class selection — mirrors catalog ``optim`` instead of silently forcing fp32 AdamW."""

from __future__ import annotations


def loraplus_optimizer_cls(optim_name: str):
    """Return ``(optimizer_cls, extra_kwargs)`` for PEFT's LoRA+ optimizer builder."""
    import torch as _torch

    # str() + lower(): TRL normalizes optim to an OptimizerNames enum, str() gives "OptimizerNames.PAGED_ADAMW_8BIT"
    if "8bit" in str(optim_name or "").lower():
        try:
            import bitsandbytes as bnb

            return bnb.optim.PagedAdamW8bit, {}
        except Exception as e:  # bnb missing / no CUDA build -> safe fp32 fallback
            print(f"[lora+] bitsandbytes 8-bit optimizer unavailable ({e}); using fp32 AdamW")
    return _torch.optim.AdamW, {}
