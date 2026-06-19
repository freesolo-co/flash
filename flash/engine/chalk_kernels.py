"""Optional chalk GPU kernels (the ``freesolo-chalk`` package).

Chalk holds Freesolo's hand-written Triton/CUDA kernels that complement Liger
(fused GEMMs, the LoRA-delta matmuls, the QKV norm+RoPE epilogue, embedding gather,
FP8 frozen-base GEMMs). Every chalk installer is **opt-in** (off unless its ``CHALK_*``
env flag is set), arch-gated, runs a numeric self-test on install, patches only frozen
``nn.Linear`` layers, and silently falls back to the eager / Liger path on any failure.

So :func:`install_chalk_kernels` is always safe to call: with the flags unset (the
default) every installer no-ops, and if ``freesolo-chalk`` isn't installed at all (e.g.
on the control plane) this whole module degrades to a no-op.
"""

from __future__ import annotations

from flash._logging import get_logger

log = get_logger(__name__)


def install_chalk_kernels(model=None) -> dict:
    """Install whichever chalk kernels are enabled via ``CHALK_*`` env flags.

    Call once per training run. Pass ``model`` (the built ``nn.Module``) to enable the
    instance-level kernels (FP8 base, embedding, FP8 MLP); the class/function-level
    kernels (LoRA delta, fused MLP, QKV epilogue, RoPE) install with ``model=None`` too,
    so this can also be called *before* the trainer builds the model.

    Returns a ``{installer_name: result}`` map (``{}`` when chalk isn't installed).
    """
    try:
        import chalk.transformers as ck
    except ImportError:
        # chalk not installed (control plane, or a worker without the gpu extra) — nothing to do.
        return {}

    results: dict[str, object] = {}

    # (name, needs_model) — model=None installers patch the model class / global fn.
    installers = [
        ("install_lora", False),
        ("install_qwen35_mlp", False),
        ("install_qwen35_qkv", False),
        ("install_qwen35_rope", False),
        ("install_qwen35_mlp_fp8", True),
        ("install_fp8_base", True),
        ("install_qwen35_embedding", True),
    ]
    for name, needs_model in installers:
        fn = getattr(ck, name, None)
        if fn is None:
            continue
        if needs_model and model is None:
            continue  # instance-level kernel; needs the built model
        try:
            results[name] = fn(model) if needs_model else fn()
        except Exception as e:  # never block training on an optional kernel
            log.warning("chalk %s failed (ignored): %s", name, e)
            results[name] = f"error: {e}"

    enabled = {k: v for k, v in results.items() if v not in (False, None) and not str(v).startswith("error")}
    if enabled:
        log.info("chalk kernels active: %s", ", ".join(enabled))
    return results
