"""Optional chalk GPU kernels (the ``freesolo-chalk`` package).

Chalk holds Freesolo's hand-written Triton/CUDA kernels that complement Liger
(fused GEMMs, the LoRA-delta matmuls, the QKV norm+RoPE epilogue, embedding gather,
FP8 frozen-base GEMMs).

Chalk follows the **install-on-call** model (mirroring Liger's
``apply_liger_kernel_to_qwen3``): chalk reads NO env vars — *calling* an installer IS
the opt-in. Each installer then arch-gates itself, runs a numeric self-test on install,
patches only frozen ``nn.Linear`` layers, and silently falls back to the eager / Liger
path (a no-op on CPU / the control plane) on any failure.

Because chalk no longer gates itself, the FLASH WORKER must decide *which* installers to
call. :func:`install_chalk_kernels` reads per-kernel ``FLASH_*`` selection flags (set by
the operator, forwarded to the worker by ``providers.runpod.train.build_worker_env``) and
calls ONLY the selected chalk installers. With NO flags set (the default) it calls
nothing, so it is always safe to invoke; and if ``freesolo-chalk`` isn't installed at all
(e.g. on the control plane, or a worker without a chalk spec) the whole module degrades to
a no-op.
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


def _selected_installers() -> list[tuple[str, bool, dict]]:
    """The chalk installers the operator selected via ``FLASH_*`` flags.

    Returns ``(installer_name, needs_model, kwargs)`` for each ENABLED kernel. ``needs_model``
    marks the instance-level installers (they patch the materialized ``nn.Module`` and must be
    skipped when no model is available yet); the class/function-level ones install with no model.

    Default (no flags set) is an empty list -> :func:`install_chalk_kernels` calls nothing.
    """
    selected: list[tuple[str, bool, dict]] = []
    # Class/function-level kernels (patch the model class / a global fn; no model arg needed).
    # chalk signatures: install_qwen35_mlp(model=None), install_qwen35_qkv(model=None),
    # install_lora() [no model param], install_qwen35_rope() [no model param].
    if _truthy("FLASH_MLP_KERNEL"):
        selected.append(("install_qwen35_mlp", False, {}))
    if _truthy("FLASH_QKV_KERNEL"):
        selected.append(("install_qwen35_qkv", False, {}))
    if _truthy("FLASH_TRITON_LORA"):
        selected.append(("install_lora", False, {}))
    if _truthy("FLASH_ROPE_KERNEL"):
        selected.append(("install_qwen35_rope", False, {}))
    # Instance-level kernels (need the built nn.Module).
    # chalk signatures: install_qwen35_mlp_fp8(model, *, down=True),
    # install_fp8_base(model, *, attn=True, mlp=True, min_k=256),
    # install_qwen35_embedding(model).
    if _truthy("FLASH_MLP_FP8"):
        # FLASH_MLP_FP8_DOWN selects whether the down-projection is fused too (chalk default True).
        down = os.environ.get("FLASH_MLP_FP8_DOWN")
        kwargs = {} if down is None else {"down": _truthy("FLASH_MLP_FP8_DOWN")}
        selected.append(("install_qwen35_mlp_fp8", True, kwargs))
    if _truthy("FLASH_FP8_BASE"):
        # Per-kernel scope knobs mirror chalk's install_fp8_base kwargs; only forward the ones the
        # operator set so chalk's own defaults (attn=True, mlp=True, min_k=256) apply otherwise.
        kwargs = {}
        if os.environ.get("FLASH_FP8_BASE_ATTN") is not None:
            kwargs["attn"] = _truthy("FLASH_FP8_BASE_ATTN")
        if os.environ.get("FLASH_FP8_BASE_MLP") is not None:
            kwargs["mlp"] = _truthy("FLASH_FP8_BASE_MLP")
        if os.environ.get("FLASH_FP8_BASE_MIN_K") is not None:
            kwargs["min_k"] = _int_env("FLASH_FP8_BASE_MIN_K", 256)
        selected.append(("install_fp8_base", True, kwargs))
    if _truthy("FLASH_EMBED_KERNEL"):
        selected.append(("install_qwen35_embedding", True, {}))
    return selected


def install_chalk_kernels(model=None) -> dict:
    """Install the chalk kernels the operator selected via ``FLASH_*`` flags.

    chalk reads NO env vars (install-on-call), so the WORKER decides which installers to call:
    each enabled ``FLASH_*`` flag maps to one chalk installer (see :func:`_selected_installers`).
    With no flags set this calls nothing and returns ``{}``.

    Call once per training run. Pass ``model`` (the built ``nn.Module``) to enable the
    instance-level kernels (FP8 base, embedding, FP8 MLP); the class/function-level kernels
    (LoRA delta, fused MLP/QKV, RoPE) install with ``model=None`` too, so this can also be
    called *before* the trainer builds the model.

    Returns a ``{installer_name: result}`` map (``{}`` when no flag is set or chalk isn't
    installed).
    """
    selected = _selected_installers()
    if not selected:
        # No FLASH_* kernel flag set -> the operator opted into nothing. Don't even import chalk.
        return {}

    try:
        import chalk.transformers as ck
    except ImportError:
        # A FLASH_* flag is set but chalk isn't installed (control plane, or a worker without a
        # chalk spec in FLASH_CHALK_SPEC/extra_pip). Documented as always safe — warn and no-op.
        log.warning(
            "FLASH_* chalk kernel(s) requested but freesolo-chalk is not installed "
            "(set FLASH_CHALK_SPEC to an installable spec on the worker); skipping."
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
        fn = getattr(ck, name, None)
        if fn is None:
            log.warning("chalk has no installer %s (skipping)", name)
            continue
        if needs_model and model is None:
            continue  # instance-level kernel; needs the built model (installed in the post-build call)
        try:
            results[name] = fn(model, **kwargs) if needs_model else fn(**kwargs)
        except Exception as e:  # never block training on an optional kernel
            log.warning("chalk %s failed (ignored): %s", name, e)
            results[name] = f"error: {e}"

    enabled = {k: v for k, v in results.items() if v not in (False, None) and not str(v).startswith("error")}
    if enabled:
        log.info("chalk kernels active: %s", ", ".join(enabled))
    return results
