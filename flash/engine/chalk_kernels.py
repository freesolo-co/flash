"""Optional chalk GPU kernels (the ``freesolo-chalk`` package).

Chalk holds Freesolo's hand-written Triton/CUDA kernels that complement Liger
(fused GEMMs, the LoRA-delta matmuls, the QKV norm+RoPE epilogue, embedding gather,
FP8 frozen-base GEMMs).

Chalk follows the **install-on-call** model (mirroring Liger's
``apply_liger_kernel_to_qwen3``): chalk reads NO env vars — *calling* an installer IS
the opt-in. Each installer then arch-gates itself, runs a numeric self-test on install,
patches only frozen ``nn.Linear`` layers, and silently falls back to the eager / Liger
path (a no-op on CPU / the control plane) on any failure.

Chalk is applied AUTOMATICALLY, just like Liger: :func:`install_chalk_kernels` turns the
**gap-filling** kernels ON BY DEFAULT (RoPE, the QKV norm+RoPE epilogue, the LoRA-delta
matmul, embedding gather — exactly the ops Liger leaves on the eager path) and only needs
``freesolo-chalk`` installed on the worker (via ``FLASH_CHALK_SPEC``). The self-test +
eager fallback in every installer make default-on safe. Per-kernel ``FLASH_*`` flags are
OVERRIDES: ``FLASH_<K>=0`` disables a default-on kernel, ``FLASH_<K>=1`` enables an opt-in
one (the fused MLP that overlaps Liger's SwiGLU, and the FP8 frozen-base GEMMs). If
``freesolo-chalk`` isn't installed at all (e.g. on the control plane) the whole module
degrades to a no-op.
"""

from __future__ import annotations

import os

from flash._logging import get_logger

log = get_logger(__name__)


def _truthy(name: str) -> bool:
    """A FLASH_* flag counts as ON when set to a non-empty, non-false value."""
    v = os.environ.get(name)
    return v is not None and v.strip().lower() not in ("", "0", "false", "no", "off")


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        log.warning("ignoring non-integer %s=%r (using %d)", name, raw, default)
        return default


# Chalk kernel table: (FLASH_* flag, chalk installer, needs_model, default_on).
# The GAP-FILLERS that complement Liger are ON BY DEFAULT — applied automatically like
# apply_liger_kernel — because each chalk installer self-tests on install and falls back to the
# eager/Liger path on any failure, so always-applying them is safe:
#   * RoPE  — the op Liger REFUSES on the qwen3.5 hybrid arch (its only real gap)
#   * QKV   — the q/k-norm + RoPE attention epilogue (Liger doesn't fuse it)
#   * LoRA  — the LoRA-delta matmul on the trainable path (Liger doesn't touch adapters)
#   * embed — the embedding gather (Liger doesn't touch it)
# The AGGRESSIVE kernels stay OPT-IN (default off): the fused MLP OVERLAPS Liger's SwiGLU (would
# double-patch), and the FP8 frozen-base GEMMs are a precision trade-off. FLASH_<KERNEL>=0 turns a
# default-on kernel OFF; FLASH_<KERNEL>=1 turns a default-off one ON.
_KERNELS: list[tuple[str, str, bool, bool]] = [
    ("FLASH_ROPE_KERNEL", "install_qwen35_rope", False, True),
    ("FLASH_QKV_KERNEL", "install_qwen35_qkv", False, True),
    ("FLASH_TRITON_LORA", "install_lora", False, True),
    ("FLASH_EMBED_KERNEL", "install_qwen35_embedding", True, True),
    ("FLASH_MLP_KERNEL", "install_qwen35_mlp", False, False),  # opt-in (overlaps Liger SwiGLU)
    ("FLASH_MLP_FP8", "install_qwen35_mlp_fp8", True, False),  # opt-in (FP8 precision trade-off)
    ("FLASH_FP8_BASE", "install_fp8_base", True, False),  # opt-in (FP8 precision trade-off)
]


def _kernel_on(flag: str, default_on: bool) -> bool:
    """A kernel is ON when its FLASH_* flag is set truthy, OFF when set falsey, and the DEFAULT
    when the flag is unset — so the gap-fillers run by default and FLASH_<K>=0 disables them."""
    v = os.environ.get(flag)
    if v is None or not v.strip():
        return default_on
    return v.strip().lower() not in ("0", "false", "no", "off")


def _selected_installers() -> list[tuple[str, bool, dict]]:
    """The chalk installers to run this pass — gap-fillers ON by default, FLASH_* flags override.

    Returns ``(installer_name, needs_model, kwargs)`` for each ENABLED kernel. ``needs_model`` marks
    the instance-level installers (patch the materialized ``nn.Module``, skipped pre-build); the
    class/function-level ones install with no model.
    """
    selected: list[tuple[str, bool, dict]] = []
    for flag, installer, needs_model, default_on in _KERNELS:
        if not _kernel_on(flag, default_on):
            continue
        kwargs: dict = {}
        if installer == "install_qwen35_mlp_fp8":
            # FLASH_MLP_FP8_DOWN selects whether the down-projection is fused too (chalk default).
            down = os.environ.get("FLASH_MLP_FP8_DOWN")
            if down is not None:
                kwargs = {"down": _truthy("FLASH_MLP_FP8_DOWN")}
        elif installer == "install_fp8_base":
            # Only forward the scope knobs the operator set so chalk's own defaults apply otherwise.
            if os.environ.get("FLASH_FP8_BASE_ATTN") is not None:
                kwargs["attn"] = _truthy("FLASH_FP8_BASE_ATTN")
            if os.environ.get("FLASH_FP8_BASE_MLP") is not None:
                kwargs["mlp"] = _truthy("FLASH_FP8_BASE_MLP")
            if os.environ.get("FLASH_FP8_BASE_MIN_K") is not None:
                kwargs["min_k"] = _int_env("FLASH_FP8_BASE_MIN_K", 256)
        selected.append((installer, needs_model, kwargs))
    return selected


def install_chalk_kernels(model=None) -> dict:
    """Install chalk's gap-filling kernels — ON by default (like Liger), FLASH_* flags override.

    The default-on gap-fillers (RoPE, QKV, LoRA-delta, embedding) plus any opt-in kernel enabled
    via ``FLASH_<K>=1`` are applied (see :func:`_selected_installers`). Returns ``{}`` when every
    kernel is disabled, ``freesolo-chalk`` isn't installed, or no selected kernel belongs to this
    pass.

    Call this TWICE per training run — exactly the two passes the worker makes — and each kernel
    installs on the ONE pass it belongs to (never twice):

    * ``install_chalk_kernels()`` (``model is None``, the PRE-build pass) installs only the
      class/function-level kernels (LoRA delta, fused MLP/QKV, RoPE) — global monkeypatches that
      must be in place before the trainer builds the model. Instance-level kernels are skipped (no
      module yet).
    * ``install_chalk_kernels(trainer.model)`` (``model is not None``, the POST-build pass) installs
      only the instance-level kernels (FP8 base, embedding, FP8 MLP) against the materialized
      module. The class/function-level monkeypatches were already applied on the pre-build pass, so
      re-running them here would double-patch — they are deliberately NOT re-run.
    """
    selected = _selected_installers()
    if not selected:
        # Every kernel explicitly disabled (FLASH_<K>=0 across the gap-fillers) -> nothing to do.
        return {}

    try:
        import chalk.transformers as ck
    except ImportError:
        # chalk's gap-fillers are default-on, but freesolo-chalk isn't installed on this worker
        # (no FLASH_CHALK_SPEC, or running on the control plane). Documented as always safe: the
        # kernels degrade to the eager/Liger path. Log once at info so it's visible but not noisy.
        log.info(
            "chalk gap-filling kernels are default-on but freesolo-chalk is not installed "
            "(set FLASH_CHALK_SPEC to an installable spec on the worker); using eager/Liger."
        )
        return {}
    except Exception as e:
        # A partially-installed / version-incompatible chalk can raise non-ImportError errors at
        # import time (e.g. a Triton/torch mismatch surfaced on import of a kernel module). This
        # hook is documented as always safe and must never abort training, so degrade to a no-op.
        log.warning("chalk import failed (ignored, kernels disabled): %s", e)
        return {}

    results: dict[str, object] = {}
    for name, needs_model, kwargs in selected:
        # Each kernel installs on exactly ONE of the two passes, so the second call never
        # re-applies what the first already installed:
        #   * class/function-level (needs_model=False): global monkeypatches -> PRE-build pass only
        #     (model is None). Skipping them when a model is present avoids re-installing the global
        #     patch the pre-build call already applied (double-patch).
        #   * instance-level (needs_model=True): patch the built nn.Module -> POST-build pass only
        #     (model is not None); there is no module to patch on the pre-build pass.
        if needs_model != (model is not None):
            continue
        fn = getattr(ck, name, None)
        if fn is None:
            log.warning("chalk has no installer %s (skipping)", name)
            continue
        try:
            results[name] = fn(model, **kwargs) if needs_model else fn(**kwargs)
        except Exception as e:  # never block training on an optional kernel
            log.warning("chalk %s failed (ignored): %s", name, e)
            results[name] = f"error: {e}"

    enabled = {k: v for k, v in results.items() if v not in (False, None) and not str(v).startswith("error")}
    if enabled:
        log.info("chalk kernels active: %s", ", ".join(enabled))
    return results
