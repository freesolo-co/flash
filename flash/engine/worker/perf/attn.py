"""Attention-implementation selection for the fine-tuning worker.

Pure / CPU-importable probes that map a CUDA compute capability to the best per-arch
``attn_implementation`` and force the cuDNN SDPA backend on consumer Blackwell. Torch and
transformers are imported lazily inside the functions so this stays a leaf module.
"""

from __future__ import annotations

import contextlib


def _attn_impl_for_capability(
    major: int, minor: int = 0, *, fa3_available: bool = False, fa2_available: bool = False
) -> str | None:
    """Map a CUDA compute capability to the trainer ``attn_implementation`` — the best per-arch
    FlashAttention kernel for the model's FULL-attention (softmax) layers, so SFT *and* GRPO use a
    real flash kernel on every arch where one exists (not plain SDPA). The Qwen3.5/3.6
    Gated-DeltaNet *linear*-attention layers always keep their own path (fla, or the native
    pure-PyTorch delta rule once fla is dropped on Hopper) — FlashAttention does not apply to linear
    attention.

    Each arch maps to its ONE best flash kernel; the fallback is UNIFORM — plain SDPA on every arch
    when that kernel's package is absent (no special FA3->FA2 chain on Hopper):
      * Hopper (sm90, H100/H200): "flash_attention_3" — FA3's warp-specialized async kernels are the
        fastest exact attention on Hopper; transformers routes it to the LOCAL ``flash_attn_interface``
        (no HF Kernels-Hub, whose torch2.10 versions break ``import transformers``). FA3 is baked into
        the worker image by default (Dockerfile FLASH_ATTN_3_SPEC), so ``fa3_available`` is normally
        True; absent -> plain SDPA, same as every other arch.
      * Ampere (sm80 A100 / sm86 3090·A6000) + Ada (sm89 4090·L40S): "flash_attention_2" when the
        ``flash_attn`` wheel is importable (``fa2_available``) — FA3 does NOT support these archs.
      * consumer Blackwell (sm120 5090 / RTX Pro): "sdpa" forced to the cuDNN backend. THE ONE arch
        with no flash: FA3/FA4 need TMEM/tcgen05 that sm120 lacks, and the prebuilt FA2 CUDA wheel's
        sm120 coverage is unverified, so cuDNN SDPA is the validated best here.
      * anything else / flash unavailable -> None: transformers picks SDPA (already flash-backed on
        Ampere/Ada/Hopper).
    Pure function (no torch / no imports) so it's unit-testable on CPU; ``fa2_available`` /
    ``fa3_available`` are the caller's probes (``optimal_attn_impl``). The big LoRA win is still the
    Liger/chalk fused kernels; flash helps only the ~25% full-attention layers of the hybrid arch."""
    if major == 9 and fa3_available:  # Hopper: FA3 is the arch's best flash kernel
        return "flash_attention_3"
    if major == 8 and minor in (0, 6, 9) and fa2_available:  # Ampere 8.0/8.6 + Ada 8.9 ONLY: FA2
        # (gate the minor so an unsupported sm8x like sm87 Jetson Orin doesn't get FA2 forced on it)
        return "flash_attention_2"
    if (
        major == 12
    ):  # consumer Blackwell: cuDNN SDPA (the one exception — FA3/FA4 need TMEM/tcgen05)
        return "sdpa"
    return None  # the arch's flash kernel is absent -> plain SDPA (the SAME fallback on every arch)


def _flash_attn_3_available() -> bool:
    """True when FlashAttention-3 is usable by transformers on this worker — i.e. the
    ``flash_attn_interface`` module (the ``flash-attn-3`` Hopper build) is importable.

    transformers' ``flash_attention_3`` path does ``from flash_attn_interface import
    flash_attn_func, ...`` (modeling_flash_attention_utils), so a present module is exactly what
    makes ``attn_implementation="flash_attention_3"`` resolve WITHOUT the HF Kernels-Hub. Prefer
    transformers' own ``is_flash_attn_3_available`` probe (it verifies real importability). Only if
    that probe is itself unavailable (transformers not importable here) fall back to a GUARDED import
    of ``flash_attn_interface`` — NOT a bare ``find_spec``, so an on-disk-but-broken install (ABI
    mismatch / missing .so) reads as unavailable instead of a false positive that would later crash
    transformers at model load. FA3 is used whenever it's importable — fixed, no disable knob."""
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
    """True when the ``flash_attn`` (FA2) wheel is importable (baked into the worker image).

    Drives the FA2 ``attn_implementation`` selection on Ampere/Ada (via ``_attn_impl_for_capability``)
    AND the SFT packing default on every arch. ``_attn_impl_for_capability`` itself never picks FA2 on
    Hopper (FA3, else uniform SDPA); FA2 re-enters there ONLY through the SFT packing path, which
    forces FA2 varlen when ``optimal_attn_impl`` returned None (Hopper without FA3). On sm120 the
    selector returns ``"sdpa"`` and run_sft DISABLES packing instead (consumer Blackwell stays plain
    SDPA — no flash), so sm120 never forces FA2. Packing rationale: TRL's ``packing_strategy='bfd'``
    produces flattened/padding-free
    batches whose example boundaries are carried by ``position_ids`` and enforced ONLY by a
    varlen-capable attention impl (FA2/FA3/flex). Under plain SDPA, packed examples attend ACROSS
    boundaries (silent quality loss). find_spec only — no import side effects (no CUDA init). FA2 is
    used whenever the wheel is importable — fixed, no disable knob."""
    try:
        import importlib.util

        return importlib.util.find_spec("flash_attn") is not None
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
    fa2 = _flash_attn_available()  # FA2 wheel importable (Ampere/Ada/Hopper)
    # Probe FA3 only on Hopper (the only arch it selects it for) so a non-Hopper run never imports
    # the transformers FA3 helpers needlessly.
    fa3 = _flash_attn_3_available() if major == 9 else False
    impl = _attn_impl_for_capability(major, minor, fa3_available=fa3, fa2_available=fa2)
    if impl in ("flash_attention_2", "flash_attention_3"):
        ver = "FlashAttention-3" if impl == "flash_attention_3" else "FlashAttention-2"
        print(
            f"[attn] sm{major}{minor} -> attn_implementation={impl} ({ver}, full-attention layers)"
        )
    elif major == 9 and not fa3:
        # Hopper but FA3 not selected -> plain SDPA (uniform fallback). FA3 is baked into the worker
        # image by default, so this means flash_attn_interface is absent/broken — check the install.
        print(f"[attn] sm{major}{minor}: FA3 unavailable (flash_attn_interface absent) -> SDPA")
    elif major == 12:  # the only arch that returns impl=="sdpa" -> this branch covers all of it
        print(
            f"[attn] sm{major}{minor} (consumer Blackwell) -> SDPA/cuDNN (FA3/FA4 need TMEM; n/a on sm120)"
        )
    elif not fa2:
        print(f"[attn] sm{major}{minor}: flash_attn wheel absent -> SDPA")
    return impl


def _sdpa_cudnn_ctx(attn_impl: str | None):
    """Context forcing the cuDNN SDPA backend (real Blackwell-consumer kernels) when we fell
    back to plain SDPA on sm120; a no-op context otherwise. Best-effort."""
    if attn_impl != "sdpa":
        return contextlib.nullcontext()
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # Priority-ordered: prefer the fast cuDNN/flash/efficient kernels, but ALWAYS include MATH
        # as the final fallback. Restricting to only [CUDNN, EFFICIENT] makes sm120 GRPO crash with
        # "RuntimeError: No available kernel" when neither has a kernel for the completion-batch
        # attention shape (MEASURED: Qwen3.5 GRPO on RTX 5090). MATH is universal, so the candidate
        # set is never empty; set_priority keeps cuDNN first whenever it CAN serve the shape (SFT
        # fast path unchanged), only falling through for the shapes cuDNN/efficient reject.
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
