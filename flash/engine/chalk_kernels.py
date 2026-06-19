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
so flash calls chalk with ``liger=False``. Per-kernel ``FLASH_*`` flags are OVERRIDES:
``FLASH_<K>=0`` disables a default-on kernel, ``FLASH_<K>=1`` enables an opt-in one. If
``freesolo-chalk`` isn't installed (no ``FLASH_CHALK_SPEC``, or on the control plane) the whole
module degrades to a no-op.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from flash._logging import get_logger

log = get_logger(__name__)

# Chalk kernel table: (FLASH_* flag, apply_chalk_kernel_to_qwen35 keyword, default_on).
# The GAP-FILLERS that complement Liger are ON BY DEFAULT — applied automatically like
# apply_liger_kernel — because each chalk installer self-tests on install and falls back to the
# eager/Liger path on any failure, so always-applying them is safe:
#   * rope             — the RoPE Liger REFUSES on the qwen3.5 hybrid arch (its only real gap)
#   * fused_lora_delta — the LoRA-delta matmul on the trainable path (Liger doesn't touch adapters)
#   * fused_embedding  — the embedding gather (Liger doesn't touch it)
# The OVERLAPPING / situational kernels stay OPT-IN (default off): the fused MLP overlaps Liger's
# SwiGLU (Liger owns MLP), the attn epilogue is eval-only (needs q/k/v out of LORA_TARGETS), and the
# FP8 frozen base is Hopper sm_90+ only. FLASH_<K>=0 turns a default-on kernel OFF; FLASH_<K>=1
# turns a default-off one ON. The keyword is exactly chalk's apply_chalk_kernel_to_qwen35 kwarg.
_KERNELS: list[tuple[str, str, bool]] = [
    ("FLASH_ROPE_KERNEL", "rope", True),
    ("FLASH_TRITON_LORA", "fused_lora_delta", True),
    ("FLASH_EMBED_KERNEL", "fused_embedding", True),
    ("FLASH_MLP_KERNEL", "fused_mlp", False),  # opt-in (Liger owns MLP/SwiGLU)
    ("FLASH_QKV_KERNEL", "attn_epilogue", False),  # opt-in (eval-only; needs q/k/v out of LoRA)
    ("FLASH_FP8_BASE", "fp8_frozen_base", False),  # opt-in (Hopper sm_90+ only)
]


def _flag_on(value: str | None, default_on: bool) -> bool:
    """Resolve one FLASH_* flag value: ON when set truthy, OFF when set falsey, and the DEFAULT
    when the flag is UNSET *or empty/whitespace* (an empty value reads as unset, not OFF) — so the
    gap-fillers run by default and FLASH_<K>=0 disables them."""
    if value is None or not value.strip():
        return default_on
    return value.strip().lower() not in ("0", "false", "no", "off")


def _kernel_on(flag: str, default_on: bool) -> bool:
    """As :func:`_flag_on`, reading ``flag`` from this process's ``os.environ``."""
    return _flag_on(os.environ.get(flag), default_on)


def is_chalk_enabled(env: Mapping[str, str]) -> bool:
    """True if ANY chalk kernel is enabled in ``env`` (so chalk must be installed).

    Resolves every kernel's FLASH_* flag in ``env`` against its default (gap-fillers default-on),
    so it is True for a normal run and False only when every otherwise-enabled kernel is explicitly
    set to 0. The single source of truth for "is chalk selected"; ``providers.runpod.train`` uses
    this instead of re-implementing the flag parsing against the ``_KERNELS`` table.
    """
    return any(_flag_on(env.get(flag), default_on) for flag, _kw, default_on in _KERNELS)


def _enabled_kwargs() -> dict[str, bool]:
    """The ``apply_chalk_kernel_to_qwen35`` boolean kwargs resolved from the FLASH_* flags.

    Gap-fillers default-on, overlapping/situational kernels default-off; FLASH_<K> overrides.
    """
    return {kw: _kernel_on(flag, default_on) for flag, kw, default_on in _KERNELS}


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
    Liger modules. Each kernel is a boolean resolved from its ``FLASH_*`` flag (gap-fillers
    default-on). Returns chalk's per-kernel report, or ``{}`` when every kernel is disabled, there is
    no model yet, or freesolo-chalk isn't installed.

    chalk's apply patches the LIVE module, so the worker calls this AFTER TRL builds the trainer
    (``model=trainer.model``); ``model is None`` is a safe no-op kept for defensive callers.
    """
    if model is None:
        # chalk's apply patches the materialized module -> nothing to do before the model is built.
        return {}

    kwargs = _enabled_kwargs()
    if not any(kwargs.values()):
        # Every kernel explicitly disabled (FLASH_<K>=0 across the gap-fillers) -> nothing to do.
        return {}

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
