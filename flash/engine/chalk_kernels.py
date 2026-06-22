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
# Situational kernels stay OPT-IN (default off): the attn epilogue is eval-only (needs q/k/v out of
# LORA_TARGETS), and the FP8 frozen base is Hopper sm_90+ only. FLASH_<K>=0 turns a default-on kernel
# OFF; FLASH_<K>=1 turns a default-off one ON. The keyword is exactly chalk's
# apply_chalk_kernel_to_qwen35 kwarg. (The bf16 fused-MLP kernel was REMOVED in freesolo-chalk 0.3.2
# — verified net-negative everywhere and eval-only; its activation fusion is the one swiglu already
# does. The Hopper/Blackwell base-GEMM lever is the FP8 frozen base below.)
# LARGER-DIM VERIFIED (freesolo-chalk 0.4.7 + 0.4.8): the FULL kernel set (rmsnorm/swiglu/FLCE/
# trainable QK-norm+RoPE/output-gate/GDN conv+SiLU/GDN native-GVA/lora-delta) is correct fwd+bwd (vs
# fp32 oracle, grad rel-err ~1e-3) AND wins speed in the production GRPO regime at the REAL 9B (hidden
# 4096, MLP inter 12288), 35B-A3B (MoE inter 512) and Qwen3.6 dims + head counts — not just the
# 0.8B/2B/4B dims everything was benched at. 0.4.7 verified A40(sm86)+A100(sm80); 0.4.8 completes the
# matrix on H100(sm90)+5090(sm120) for the newest kernels (out_gate/GVA/trainable-QK at 9B/35B dims).
# ANY_BAD_CORRECTNESS=False on ALL FOUR arches (sm80/86/90/120); worst rel-err ~4.2e-3 (GVA dk grad,
# << 2e-2 tol). Rules out a larger-dim chalk kernel bug as the Qwen3.5-flat root cause on every arch.
# No source/flag change (verification releases).
_KERNELS: list[tuple[str, str, bool]] = [
    ("FLASH_RMSNORM_KERNEL", "rmsnorm", True),
    ("FLASH_SWIGLU_KERNEL", "swiglu", True),
    ("FLASH_FLCE_KERNEL", "fused_linear_cross_entropy", True),
    ("FLASH_ROPE_KERNEL", "rope", True),
    ("FLASH_TRITON_LORA", "fused_lora_delta", True),
    ("FLASH_EMBED_KERNEL", "fused_embedding", True),
    ("FLASH_GDN_KERNEL", "gdn", True),  # chalk-native GDN conv+SiLU (freesolo-chalk 0.4.2) + native-GVA scan pre-op (0.4.5); fla still owns the scan. Full GDN block +13-15% E2E at the production batch/seq regime, VERIFIED H100/A100/A40 + 5090 (0.4.4 arch matrix). Replaces the eager F.silu(F.conv1d) flash runs (causal_conv1d omitted: sdist build fails). 0.4.5: on non-Hopper (sm80/86/120) SKIPS the eager q/k repeat_interleave (lets fla broadcast GVA natively) -> +1.06-1.14x fwd+bwd + ~1.04x less peak mem, VERIFIED A100/A40/5090; gated OFF on Hopper sm90 where fla's tilelang chunk_bwd refuses GVA. No new flag (auto-gated by device cap inside install).
    ("FLASH_TRAINABLE_QKV", "trainable_attn_epilogue", True),  # chalk-native TRAINABLE fused QK-norm+partial-RoPE (freesolo-chalk 0.4.3, fwd+bwd). ENGAGES in training with q/k/v in LoRA (the production all-linear case the eval-only attn_epilogue could never hit). Replaces the eager q_norm/k_norm + apply_rotary_pos_emb stack; geomean fwd+bwd H100 2.85x / A100 3.42x / A40 6.32x / 5090 3.18x (0.4.4 arch matrix). 0.4.6: ALSO fuses the post-attention OUTPUT gate (attn_out*sigmoid(gate) before o_proj) into a chalk-native fwd+bwd kernel (independently self-test gated; if only that declines, QK-norm still fuses + gate stays eager) -> +1.47-1.50x fwd+bwd at the production GRPO regime (t=B*L>=4096) + peak-mem win every shape, VERIFIED A40/A100/H100; E2E real-attention grads 6.2e-3. Self-test gated -> eager on any failure. No new flag (auto-gated inside install).
    ("FLASH_QKV_KERNEL", "attn_epilogue", False),  # opt-in (eval-only; needs q/k/v out of LoRA). chalk skips the trainable one when this is on (they patch the same forward).
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
        # Version-skew safety: drop any kwarg the INSTALLED chalk's apply doesn't accept (e.g. a newer
        # flag like `gdn` against an older baked chalk wheel) so unknown kwargs degrade that one kernel
        # to eager instead of TypeError-ing the whole stack. (No-op if apply takes **kwargs.)
        try:
            import inspect

            _params = inspect.signature(apply_chalk_kernel_to_qwen35).parameters
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in _params.values()):
                _dropped = [k for k in kwargs if k not in _params]
                if _dropped:
                    log.info("chalk apply: dropping kwargs unsupported by installed chalk: %s", _dropped)
                kwargs = {k: v for k, v in kwargs.items() if k in _params}
        except (ValueError, TypeError):
            pass  # builtins/odd callables expose no signature — pass kwargs as-is
        report = apply_chalk_kernel_to_qwen35(model, liger=False, **kwargs)
    except Exception as e:  # never block training on the optional kernel stack
        log.warning("chalk apply failed (ignored, kernels disabled): %s", e)
        return {}

    active = active_kernels(report)
    if active:
        log.info("chalk kernels active: %s", ", ".join(active))
    return report or {}
