"""Optional chalk GPU kernels (the ``freesolo-chalk`` package).

Gap-filling Triton kernels for qwen3.5 applied on top of Liger (liger=False; TRL owns Liger).
Degrades to a no-op if freesolo-chalk is not installed.
"""

from __future__ import annotations

from collections.abc import Mapping

from flash._logging import get_logger

log = get_logger(__name__)

_KERNELS: list[tuple[str, bool]] = [
    ("rope", True),
    ("fused_lora_delta", True),
    ("fused_embedding", True),
    ("fused_mlp", False),  # off (Liger owns MLP/SwiGLU)
    ("attn_epilogue", False),  # off (eval-only; needs q/k/v out of LoRA)
    ("fp8_frozen_base", False),  # off (Hopper sm_90+ only)
]


def active_kernels(report: Mapping[str, object] | None) -> list[str]:
    """Return kernels that engaged (truthy, non-error) in an apply report, excluding ``liger``."""
    return sorted(
        k
        for k, v in (report or {}).items()
        if k != "liger" and v not in (False, None) and not (isinstance(v, dict) and "error" in v)
    )


def install_chalk_kernels(model=None) -> dict:
    """Apply chalk's gap-filling kernels to ``model``; call AFTER TRL builds the trainer.

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
            "spec, or check the default PyPI install); chalk kernels off, using eager/Liger."
        )
        return {}
    except Exception as e:
        log.warning("chalk import failed (ignored, kernels disabled): %s", e)
        return {}

    try:
        # liger=False: TRL already applied Liger; chalk composes on top of the live Liger modules.
        report = apply_chalk_kernel_to_qwen35(model, liger=False, **kwargs)
    except Exception as e:  # never block training on the optional kernel stack
        log.warning("chalk apply failed (ignored, kernels disabled): %s", e)
        return {}

    active = active_kernels(report)
    if active:
        log.info("chalk kernels active: %s", ", ".join(active))
    return report or {}
