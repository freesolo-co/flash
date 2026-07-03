"""Optional chalk GPU kernels (the ``freesolo-chalk`` package).

Standalone Triton kernels for qwen3.5/3.6. Flash does not ask TRL to apply Liger; chalk owns the
RMSNorm, SwiGLU, fused-linear-CE, RoPE, LoRA-delta, trainable attention epilogue, embedding, and GDN
patches. Degrades to a no-op if freesolo-chalk is not installed.
"""

from __future__ import annotations

from collections.abc import Mapping

from flash._logging import get_logger

log = get_logger(__name__)

_KERNELS: list[tuple[str, bool]] = [
    ("rope", True),
    ("rmsnorm", True),
    ("swiglu", True),
    ("fused_linear_cross_entropy", True),
    ("fused_lora_delta", True),
    ("trainable_attn_epilogue", True),
    ("fused_embedding", True),
    ("gdn", True),
    ("attn_epilogue", False),  # off (eval-only; needs q/k/v out of LoRA)
    ("fp8_frozen_base", False),  # off by default: speed/memory tradeoff, enable only after run A/B
]


def active_kernels(report: Mapping[str, object] | None) -> list[str]:
    """Return kernels that engaged (truthy, non-error) in an apply report, excluding ``liger``."""
    return sorted(
        k
        for k, v in (report or {}).items()
        if k != "liger" and v not in (False, None) and not (isinstance(v, dict) and "error" in v)
    )


def chalk_supports_fused_ce_model(model_id: str | None) -> bool:
    """True when chalk's standalone fused-CE patch supports this model family."""
    mid = (model_id or "").lower()
    return any(token in mid for token in ("qwen3.5", "qwen3_5", "qwen3.6", "qwen3_6"))


def chalk_fused_ce_available(model_id: str | None = None) -> bool:
    """Best-effort preflight for SFT fused-CE sizing before the trainer exists."""
    if model_id is not None and not chalk_supports_fused_ce_model(model_id):
        return False
    try:
        from chalk.transformers import apply_chalk_kernel_to_qwen35 as _apply
    except Exception:
        return False
    return callable(_apply)


def install_chalk_kernels(model=None) -> dict:
    """Apply chalk standalone kernels to ``model``; call AFTER TRL builds the trainer.

    Returns chalk's per-kernel report, or ``{}`` when freesolo-chalk isn't installed.
    """
    if model is None:
        return {}

    kwargs = dict(_KERNELS)
    try:
        from chalk.transformers import apply_chalk_kernel_to_qwen35
    except ImportError:
        log.info(
            "freesolo-chalk is not installed on this worker (set FLASH_CHALK_SPEC to an installable "
            "spec, or check the default install); chalk kernels off, using eager kernels."
        )
        return {}
    except Exception as e:
        log.warning("chalk import failed (ignored, kernels disabled): %s", e)
        return {}

    try:
        # liger=False: flash uses chalk standalone. TRL must not have applied Liger first.
        report = apply_chalk_kernel_to_qwen35(model, liger=False, **kwargs)
    except Exception as e:  # never block training on the optional kernel stack
        log.warning("chalk apply failed (ignored, kernels disabled): %s", e)
        return {}

    active = active_kernels(report)
    if active:
        log.info("chalk kernels active: %s", ", ".join(active))
    return report or {}
