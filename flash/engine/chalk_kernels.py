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


def fp8_base_engaged(report: Mapping[str, object] | None) -> bool:
    """True iff chalk's ``fp8_frozen_base`` actually wrapped at least one Linear.

    Its report is a dict ``{installed, attn, mlp, ...}`` even on a no-op (``installed: 0`` on a
    non-FP8 GPU / unsupported model), so — unlike ``active_kernels`` — engagement must be read off
    the ``installed`` count, not mere dict-presence."""
    r = (report or {}).get("fp8_frozen_base")
    if isinstance(r, dict):
        if "error" in r:
            return False
        try:
            return int(r.get("installed", 0) or 0) > 0
        except (TypeError, ValueError):
            return False
    return bool(r)


def _fp8_kwargs(apply_fn) -> dict:
    """FP8 frozen-base kwargs for ``apply_chalk_kernel_to_qwen35``, feature-detected against the
    installed chalk so an older build (missing the memory-mode knobs) can't ``TypeError`` and
    silently disable the WHOLE kernel stack.

    ``fp8_frozen_base`` turns the frozen-base forward GEMM into FP8 e4m3 (chalk applies it to the
    PEFT LoRA base too, so it covers flash's ``all-linear`` LoRA). ``fp8_no_wcache`` re-quantizes
    the fp8 weight each forward instead of caching a persistent fp8 copy: peak memory stays ~bf16
    baseline (the cached default adds ~+50% of the frozen-weight memory) and — unlike ``free_base``
    — it keeps the bf16 base param intact, so GRPO's colocated-vLLM weight sync still reads real
    bf16 weights. Both are speed<->memory knobs, never trained-weight changes."""
    try:
        import inspect

        params = set(inspect.signature(apply_fn).parameters)
    except (ValueError, TypeError):
        params = set()
    kw = {"fp8_frozen_base": True}
    if "fp8_no_wcache" in params:
        kw["fp8_no_wcache"] = True
    return kw


def install_chalk_kernels(
    model=None, *, fp8: bool = False, fused_ce: bool = True, grad_checkpointing: bool = False
) -> dict:
    """Apply chalk standalone kernels to ``model``; call AFTER TRL builds the trainer.

    ``fp8=True`` additionally enables the FP8 frozen-base GEMM (chalk ``fp8_frozen_base``): a
    QLoRA-style forward-compute change on the FROZEN base only (trainable LoRA adapters + optimizer
    stay bf16). It self-gates inside chalk to FP8 hardware (Ada/Hopper/Blackwell, sm_89+) and to
    Qwen3.5/3.6, so on an A100/older worker or an unsupported model it no-ops and training stays
    bf16 — logged below so the A/B stays honest.

    ``fused_ce=False`` turns OFF chalk's fused linear-cross-entropy. chalk's fused-CE binds the
    causal-LM forward to return the fused loss with ``logits=None``; that is incompatible with TRL's
    ``SFTTrainer.compute_loss`` (it reads ``outputs.logits`` for a token-accuracy metric and only
    skips it when ``use_liger_kernel=True``, which TRL rejects unless liger-kernel is installed —
    it is not on this worker). The SFT path passes ``fused_ce=False`` so the model returns real
    logits; GRPO keeps the default. ``fp8_frozen_base`` is independent, so FP8 still engages either
    way.

    ``grad_checkpointing=True`` turns OFF chalk's ``fused_embedding``. That kernel stashes the step's
    ``input_ids`` on ``embed_tokens`` and consumes them in layer-0's ``input_layernorm`` (taking the
    fused path because the base embed/norm are frozen). Under non-reentrant gradient checkpointing,
    ``embed_tokens`` runs only on the RECORD pass (it's outside the checkpointed decoder layers); the
    backward RECOMPUTE finds the stash cleared, so layer-0 norm flips to eager RMSNorm — the two
    passes enroll a different saved-tensor sequence and ``torch.utils.checkpoint`` raises
    ``CheckpointError`` at the first backward (crashes bf16 and fp8 alike, any arch). Disabling it
    makes both passes take the identical eager path. It's a single-norm micro-opt, so the only cost
    is losing that fusion whenever GC is on.

    Returns chalk's per-kernel report, or ``{}`` when freesolo-chalk isn't installed.
    """
    if model is None:
        return {}

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

    kwargs = dict(_KERNELS)
    if not fused_ce:
        kwargs["fused_linear_cross_entropy"] = False
    if grad_checkpointing:
        kwargs["fused_embedding"] = False
        log.info(
            "chalk fused_embedding disabled (gradient checkpointing on): its layer-0 input_ids stash "
            "isn't replayed on the backward recompute -> torch.utils.checkpoint CheckpointError"
        )
    if fp8:
        kwargs.update(_fp8_kwargs(apply_chalk_kernel_to_qwen35))
        log.info("chalk fp8 requested: enabling fp8_frozen_base (QLoRA-style FP8 frozen-base GEMM)")

    try:
        # liger=False: flash uses chalk standalone. TRL must not have applied Liger first.
        report = apply_chalk_kernel_to_qwen35(model, liger=False, **kwargs)
    except Exception as e:  # never block training on the optional kernel stack
        log.warning("chalk apply failed (ignored, kernels disabled): %s", e)
        return {}

    active = active_kernels(report)
    if active:
        log.info("chalk kernels active: %s", ", ".join(active))
    if fp8:
        # Honest A/B: distinguish "fp8 engaged" from "fp8 requested but the worker/model couldn't
        # run it" (chalk returns a falsy/`installed: 0` report for fp8_frozen_base then).
        if fp8_base_engaged(report):
            n = report.get("fp8_frozen_base", {})
            n = n.get("installed") if isinstance(n, dict) else n
            log.info("chalk fp8 ENGAGED: %s frozen-base Linear(s) running FP8 e4m3 GEMM", n)
        else:
            log.warning(
                "chalk fp8 requested but did NOT engage (needs sm_89+ FP8 hardware "
                "[Ada/Hopper/Blackwell] and a Qwen3.5/3.6 model); training runs in bf16"
            )
    return report or {}
