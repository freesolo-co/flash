"""Attention-implementation selection: maps CUDA compute capability to best attn_implementation."""

from __future__ import annotations

import contextlib


def _attn_impl_for_capability(
    major: int, minor: int = 0, *, fa3_available: bool = False, fa2_available: bool = False
) -> str | None:
    """Map CUDA compute capability to a Transformers attn_implementation override."""
    if major == 9 and fa3_available:
        return "flash_attention_3"
    if major == 8 and minor in (0, 6, 9) and fa2_available:  # gate minor: exclude sm87 Jetson Orin
        return "flash_attention_2"
    if major in (10, 12):
        # Blackwell: FA3/FA4 need TMEM sm120 lacks; FA2 sm100 SASS coverage unverified -> cuDNN SDPA.
        return "sdpa"
    return None


def _flash_attn_3_available() -> bool:
    """True when flash_attn_interface is importable (FA3 usable by transformers).

    Uses a guarded import, not find_spec — a broken install (ABI mismatch) reads as unavailable."""
    try:
        from transformers.utils import is_flash_attn_3_available

        return bool(is_flash_attn_3_available())
    except Exception:
        try:
            import flash_attn_interface  # noqa: F401  (guarded: verifies real importability)

            return True
        except Exception:
            return False


def _flash_attn_available() -> bool:
    """True when the flash_attn (FA2) wheel is importable.

    Uses a guarded import, not find_spec — a broken install (ABI mismatch) reads as unavailable."""
    try:
        from transformers.utils import is_flash_attn_2_available

        return bool(is_flash_attn_2_available())
    except Exception:
        try:
            import flash_attn  # noqa: F401  (guarded: verifies real importability)

            return True
        except Exception:
            return False


def optimal_attn_impl() -> str | None:
    """Best ``attn_implementation`` for the live GPU (None = leave transformers' default)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
    except Exception as e:
        print("optimal_attn_impl probe failed:", e)
        return None
    fa2 = _flash_attn_available()
    fa3 = _flash_attn_3_available() if major == 9 else False
    impl = _attn_impl_for_capability(major, minor, fa3_available=fa3, fa2_available=fa2)
    if impl in ("flash_attention_2", "flash_attention_3"):
        ver = "FlashAttention-3" if impl == "flash_attention_3" else "FlashAttention-2"
        print(
            f"[attn] sm{major}{minor} -> attn_implementation={impl} ({ver}, full-attention layers)"
        )
    elif major == 9 and not fa3:
        print(f"[attn] sm{major}{minor}: FA3 unavailable (flash_attn_interface absent) -> SDPA")
    elif major in (10, 12):
        _tier = "datacenter" if major == 10 else "consumer"
        print(
            f"[attn] sm{major}{minor} ({_tier} Blackwell) -> SDPA/cuDNN "
            "(FA3/FA4 need TMEM; FA2 Blackwell-SASS coverage unverified)"
        )
    elif not fa2:
        print(f"[attn] sm{major}{minor}: flash_attn wheel absent -> SDPA")
    return impl


def _sdpa_cudnn_ctx(attn_impl: str | None):
    """Force cuDNN SDPA backend on Blackwell (attn_impl=="sdpa"); no-op otherwise."""
    if attn_impl != "sdpa":
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # MATH must be last-resort: omitting it crashes sm120 GRPO ("No available kernel", measured).
        return sdpa_kernel(
            [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
            set_priority=True,
        )
    except Exception as e:
        print("[attn] cuDNN SDPA backend unavailable, using default SDPA:", e)
        return contextlib.nullcontext()
