"""Optional chalk GPU kernels (the ``freesolo-chalk`` package).

Chalk holds Freesolo's hand-written Triton/CUDA kernels that complement Liger: the RoPE the
qwen3.5 hybrid arch needs (Liger refuses it), the LoRA-delta matmul, fused MLP, the QKV
norm+RoPE attention epilogue, embedding gather, and FP8 frozen-base GEMMs.

Chalk ships a Liger-style one-call entry point, ``apply_chalk_kernel_to_qwen35(model, ...)``,
mirroring ``apply_liger_kernel_to_qwen3``: enablement is the call itself (no env flag), each kernel
is a boolean keyword, and it NEVER raises on a kernel failure (every installer self-tests +
arch-gates and falls back to the eager/Liger path; a no-op off-GPU). flash applies it
AUTOMATICALLY — like Liger — after the trainer builds the model, with the **gap-filling** kernels
Liger leaves on the eager path ON BY DEFAULT: RoPE, the LoRA-delta matmul, and embedding gather.
The kernels that OVERLAP Liger (fused MLP / SwiGLU — Liger owns MLP) or are situational (the
eval-only QKV epilogue, the Hopper-only FP8 frozen base) stay OPT-IN.

Liger is applied by TRL (``use_liger_kernel``); chalk composes ON TOP of the live Liger modules,
so flash calls chalk with ``liger=False``. Kernel selection is FIXED (deterministic): the
gap-fillers run and the overlapping/situational kernels stay off — there is no env override. If
``freesolo-chalk`` isn't installed (no ``FLASH_CHALK_SPEC``, or on the control plane) the whole
module degrades to a no-op.
"""

from __future__ import annotations

from collections.abc import Mapping

from flash._logging import get_logger

log = get_logger(__name__)

# Chalk kernel table: (apply_chalk_kernel_to_qwen35 keyword, enabled). Selection is FIXED — there
# is no env override; the values here are exactly what runs on every supported run.
# The GAP-FILLERS that complement Liger are ON — applied automatically like apply_liger_kernel —
# because each chalk installer self-tests on install and falls back to the eager/Liger path on any
# failure, so always-applying them is safe:
#   * rope             — the RoPE Liger REFUSES on the qwen3.5 hybrid arch (its only real gap)
#   * fused_lora_delta — the LoRA-delta matmul on the trainable path (Liger doesn't touch adapters)
#   * fused_embedding  — the embedding gather (Liger doesn't touch it)
# The OVERLAPPING / situational kernels stay OFF: the fused MLP overlaps Liger's SwiGLU (Liger owns
# MLP), the attn epilogue is eval-only (needs q/k/v out of LORA_TARGETS), and the FP8 frozen base is
# Hopper sm_90+ only. The keyword is exactly chalk's apply_chalk_kernel_to_qwen35 kwarg.
_KERNELS: list[tuple[str, bool]] = [
    ("rope", True),
    ("fused_lora_delta", True),
    ("fused_embedding", True),
    ("fused_mlp", False),  # off (Liger owns MLP/SwiGLU)
    ("attn_epilogue", False),  # off (eval-only; needs q/k/v out of LoRA)
    ("fp8_frozen_base", False),  # off (Hopper sm_90+ only)
]


def _enabled_kwargs() -> dict[str, bool]:
    """The fixed ``apply_chalk_kernel_to_qwen35`` boolean kwargs (gap-fillers on, the rest off)."""
    return dict(_KERNELS)


def active_kernels(report: Mapping[str, object] | None) -> list[str]:
    """The chalk kernels that actually ENGAGED (truthy, non-error result) in an apply report.

    For a metrics note recording which kernels ran (so chalk engagement is verifiable without the
    console). Excludes ``liger`` (TRL applies Liger; chalk's report carries it as False here).
    """
    return sorted(
        k
        for k, v in (report or {}).items()
        if k != "liger" and v not in (False, None) and not (isinstance(v, dict) and "error" in v)
    )


def install_chalk_kernels(model=None) -> dict:
    """Apply chalk's gap-filling kernels to ``model`` — ON by default (like Liger), flags override.

    Uses chalk's Liger-style entry point ``apply_chalk_kernel_to_qwen35(model, liger=False, ...)``:
    Liger is already applied by TRL (``use_liger_kernel``), so chalk composes on top of the live
    Liger modules. Each kernel is a fixed boolean (gap-fillers on, the rest off). Returns chalk's
    per-kernel report, or ``{}`` when there is no model yet or freesolo-chalk isn't installed.

    chalk's apply patches the LIVE module, so the worker calls this AFTER TRL builds the trainer
    (``model=trainer.model``); ``model is None`` is a safe no-op kept for defensive callers.
    """
    if model is None:
        # chalk's apply patches the materialized module -> nothing to do before the model is built.
        return {}

    kwargs = _enabled_kwargs()
    try:
        from chalk.transformers import apply_chalk_kernel_to_qwen35
    except ImportError:
        # chalk is installed by default (PyPI; chalk_extra_pip), so this only fires if an install
        # was disabled/failed. Always safe: the kernels degrade to the eager/Liger path. Only the
        # post-build call reaches this import (the pre-build pass returns early), so it logs at most
        # once per run — no per-process dedup needed.
        log.info(
            "freesolo-chalk is not installed on this worker (set FLASH_CHALK_SPEC to an installable "
            "spec, or check the default PyPI install); chalk kernels off, using eager/Liger."
        )
        return {}
    except Exception as e:
        # A partially-installed / version-incompatible chalk can raise non-ImportError errors at
        # import time (e.g. a Triton/torch mismatch). This hook must never abort training.
        log.warning("chalk import failed (ignored, kernels disabled): %s", e)
        return {}

    try:
        # liger=False: TRL already applied Liger (use_liger_kernel); chalk composes on the live
        # Liger modules. apply_chalk_kernel_to_qwen35 never raises on a per-kernel failure, but
        # guard the call itself so a chalk API/version skew can never abort training.
        report = apply_chalk_kernel_to_qwen35(model, liger=False, **kwargs)
    except Exception as e:  # never block training on the optional kernel stack
        log.warning("chalk apply failed (ignored, kernels disabled): %s", e)
        return {}

    active = active_kernels(report)
    if active:
        log.info("chalk kernels active: %s", ", ".join(active))
    return report or {}
