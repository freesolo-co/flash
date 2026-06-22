"""chalk GPU kernels (the ``freesolo-chalk`` package) — flash's STANDALONE kernel stack.

chalk holds Freesolo's hand-written Triton/CUDA kernels for the Qwen3.5/3.6 layers. flash no
longer uses Liger at all: chalk runs standalone and supplies EVERY fused kernel — its own
RMSNorm, SwiGLU, and fused-linear-cross-entropy (the layers Liger used to own, which chalk now
beats or ties on every GPU arch), PLUS the kernels Liger never had: the RoPE the qwen3.5 hybrid
arch needs (Liger refused it), the LoRA-delta matmul, embedding gather, fused MLP, the QKV
norm+RoPE attention epilogue, and FP8 frozen-base GEMMs.

chalk ships a Liger-style one-call entry point, ``apply_chalk_kernel_to_qwen35(model, ...)``:
enablement is the call itself (no env flag), each kernel is a boolean keyword, and it NEVER
raises on a kernel failure (every installer self-tests + arch-gates and falls back to the eager
path; a no-op off-GPU). flash applies it AUTOMATICALLY after the trainer builds the model, with
``liger=False`` (CHALK STANDALONE — chalk installs its OWN rms_norm/swiglu/FLCE, zero Liger).

The PRODUCTION kernels are ON BY DEFAULT — chalk replaces Liger for everything: rms_norm, swiglu,
fused-linear-CE (the FLCE that keeps the ~248k-vocab logits from materializing — flash's
large-vocab OOM protection now comes from chalk, not Liger), RoPE, the LoRA-delta matmul, and
embedding gather. Situational kernels stay OPT-IN: the fused MLP (overlaps swiglu, measured
net-negative on H100), the eval-only QKV epilogue (needs q/k/v out of LoRA targets), and the
Hopper-only FP8 frozen base. Per-kernel ``FLASH_*`` flags are OVERRIDES: ``FLASH_<K>=0`` disables
a default-on kernel, ``FLASH_<K>=1`` enables an opt-in one. If ``freesolo-chalk`` isn't installed
(e.g. on the control plane) the whole module degrades to a no-op.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from flash._logging import get_logger

log = get_logger(__name__)

# Chalk kernel table: (FLASH_* flag, apply_chalk_kernel_to_qwen35 keyword, default_on).
# chalk is the STANDALONE stack (no Liger) — every PRODUCTION fused kernel is ON BY DEFAULT,
# applied automatically after the trainer builds the model. Each chalk installer self-tests on
# install and falls back to the eager path on any failure, so always-applying them is safe:
#   * rmsnorm                     — chalk's fused RMSNorm (replaces Liger's)
#   * swiglu                      — chalk's fused SwiGLU activation (replaces Liger's)
#   * fused_linear_cross_entropy  — chalk's chunked LM-head+CE; keeps the ~248k-vocab logits from
#                                   materializing (the large-vocab OOM protection — was Liger's FLCE)
#   * rope                        — the RoPE Liger REFUSED on the qwen3.5 hybrid arch
#   * fused_lora_delta            — the LoRA-delta matmul on the trainable path (adapters)
#   * fused_embedding             — the embedding gather
# Situational kernels stay OPT-IN (default off): the fused MLP overlaps swiglu (measured
# net-negative on H100), the attn epilogue is eval-only (needs q/k/v out of LORA_TARGETS), and the
# FP8 frozen base is Hopper sm_90+ only. FLASH_<K>=0 turns a default-on kernel OFF; FLASH_<K>=1
# turns a default-off one ON. The keyword is exactly chalk's apply_chalk_kernel_to_qwen35 kwarg.
_KERNELS: list[tuple[str, str, bool]] = [
    ("FLASH_RMSNORM_KERNEL", "rmsnorm", True),
    ("FLASH_SWIGLU_KERNEL", "swiglu", True),
    ("FLASH_FLCE_KERNEL", "fused_linear_cross_entropy", True),
    ("FLASH_ROPE_KERNEL", "rope", True),
    ("FLASH_TRITON_LORA", "fused_lora_delta", True),
    ("FLASH_EMBED_KERNEL", "fused_embedding", True),
    ("FLASH_MLP_KERNEL", "fused_mlp", False),  # opt-in (overlaps swiglu; net-negative on H100)
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


def kernel_on(flag: str, default_on: bool) -> bool:
    """Is a single chalk kernel ``flag`` enabled in this process's ``os.environ``?

    Resolves one FLASH_* flag against its default (see :func:`_flag_on`).
    """
    return _flag_on(os.environ.get(flag), default_on)


def is_chalk_enabled(env: Mapping[str, str]) -> bool:
    """True if ANY chalk kernel is enabled in ``env`` (so chalk must be installed).

    Resolves every kernel's FLASH_* flag in ``env`` against its default (gap-fillers default-on),
    so it is True for a normal run and False only when every otherwise-enabled kernel is explicitly
    set to 0. The single source of truth for "is chalk selected"; ``providers.runpod.train`` uses
    this instead of re-implementing the flag parsing against the ``_KERNELS`` table.
    """
    return any(_flag_on(env.get(flag), default_on) for flag, _kw, default_on in _KERNELS)


def flce_flag_on() -> bool:
    """Is chalk's fused-linear cross-entropy (FLCE) ENABLED by flag (``FLASH_FLCE_KERNEL``)?

    The single source of truth for the operator's FLCE setting (default-on; an operator disables it
    with ``FLASH_FLCE_KERNEL=0``). The VRAM/cost estimator gates on THIS — it runs on the control
    plane (where freesolo-chalk is intentionally absent) but provisions/prices for a worker that DOES
    have chalk, so it must bank the fused-CE saving whenever the operator left FLCE on, while still
    dropping it (cap binds) when the operator turned FLCE off. ``run_sft`` gates on the stricter
    :func:`fused_ce_available` (this AND chalk actually importable on the worker).
    """
    return kernel_on("FLASH_FLCE_KERNEL", True)


def _chalk_importable() -> bool:
    """True when the ``freesolo-chalk`` package (import name ``chalk``) can be imported here.

    ``find_spec`` only — no import side effects (and no CUDA init) — and it probes the SAME thing
    ``install_chalk_kernels`` does (``from chalk.transformers import ...``), so "chalk is available"
    is decided one way. chalk is optional: it is absent on the control plane and may be missing /
    failed-to-install on a worker, in which case its FLCE (the large-vocab logits fuser) never runs.
    """
    try:
        import importlib.util

        return importlib.util.find_spec("chalk") is not None
    except Exception:
        # A broken/partial chalk install can make find_spec itself raise -> treat as unavailable.
        return False


def fused_ce_available() -> bool:
    """Can chalk's fused-linear cross-entropy (FLCE) ACTUALLY run in THIS process?

    For the WORKER (``run_sft``), which is the process that actually runs the kernel and so must
    protect against a missing install. True ONLY when BOTH hold:
      * the ``FLASH_FLCE_KERNEL`` flag is on (:func:`flce_flag_on`), AND
      * ``freesolo-chalk`` is importable (it supplies FLCE; a no-op when absent).

    The fused-CE memory saving (the [per_device, seq, vocab] fp32 logits never materialize) only
    happens when both are true. Gating on the flag ALONE wrongly assumes the saving whenever chalk is
    missing/failed-to-install, so the worker would skip the large-vocab logits cap and OOM on
    ~248k-vocab runs. CONSERVATIVE by construction: any uncertainty -> False -> the cap binds (no
    OOM). (The offline estimator instead uses :func:`flce_flag_on`: it can't import chalk on the
    control plane but provisions for a worker that has it, so it banks the saving on the flag alone —
    using this stricter check there would under-bank and over-provision EVERY ≥3B / long-ctx run.)
    """
    return flce_flag_on() and _chalk_importable()


def _enabled_kwargs() -> dict[str, bool]:
    """The ``apply_chalk_kernel_to_qwen35`` boolean kwargs resolved from the FLASH_* flags.

    Gap-fillers default-on, overlapping/situational kernels default-off; FLASH_<K> overrides.
    """
    return {kw: kernel_on(flag, default_on) for flag, kw, default_on in _KERNELS}


def active_kernels(report: Mapping[str, object] | None) -> list[str]:
    """The chalk kernels that actually ENGAGED (truthy, non-error result) in an apply report.

    For a metrics note recording which kernels ran (so chalk engagement is verifiable without the
    console). Excludes ``liger`` (chalk runs standalone; its report carries ``liger`` as False).
    """
    return sorted(
        k
        for k, v in (report or {}).items()
        if k != "liger" and v not in (False, None) and not (isinstance(v, dict) and "error" in v)
    )


def install_chalk_kernels(model=None) -> dict:
    """Apply chalk's STANDALONE kernel stack to ``model`` — production kernels ON by default.

    Uses chalk's one-call entry point ``apply_chalk_kernel_to_qwen35(model, liger=False, ...)``:
    ``liger=False`` means chalk installs its OWN rms_norm/swiglu/FLCE (plus rope/lora-delta/
    embedding) — flash does NOT use Liger. Each kernel is a boolean resolved from its ``FLASH_*``
    flag (production kernels default-on). Returns chalk's per-kernel report, or ``{}`` when every
    kernel is disabled, there is no model yet, or freesolo-chalk isn't installed.

    chalk's apply patches the LIVE module, so the worker calls this AFTER the trainer builds the
    model (``model=trainer.model``); ``model is None`` is a safe no-op kept for defensive callers.
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
        # chalk is installed by default (baked into the worker image + chalk_extra_pip), so this only
        # fires if an install was disabled/failed. Always safe: the kernels degrade to eager. Only the
        # post-build call reaches this import (the pre-build pass returns early), so it logs at most
        # once per run — no per-process dedup needed.
        log.info(
            "freesolo-chalk is not installed on this worker (set FLASH_CHALK_SPEC to an installable "
            "spec, or check the default PyPI install); chalk kernels off, using the eager path."
        )
        return {}
    except Exception as e:
        # A partially-installed / version-incompatible chalk can raise non-ImportError errors at
        # import time (e.g. a Triton/torch mismatch). This hook must never abort training.
        log.warning("chalk import failed (ignored, kernels disabled): %s", e)
        return {}

    try:
        # liger=False: CHALK STANDALONE — chalk installs its OWN rms_norm/swiglu/FLCE (+ rope/
        # lora-delta/embedding); flash does NOT enable TRL's Liger. apply_chalk_kernel_to_qwen35
        # never raises on a per-kernel failure, but guard the call itself so a chalk API/version
        # skew can never abort training.
        report = apply_chalk_kernel_to_qwen35(model, liger=False, **kwargs)
    except Exception as e:  # never block training on the optional kernel stack
        log.warning("chalk apply failed (ignored, kernels disabled): %s", e)
        return {}

    active = active_kernels(report)
    if active:
        log.info("chalk kernels active: %s", ", ".join(active))
    return report or {}
