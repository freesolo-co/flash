"""Static lookup facts for the cost model: GPU price/VRAM/compute + cheapest-fit
selection, model size/quant, and reward-grader latency. Pure tables + accessors."""

from __future__ import annotations

import re

from flash.catalog import MODELS
from flash.providers.base import GPU_INFO, POLICY_NAMES, GpuClass, canonical_gpu, providers_for

# ===== GPU facts =====
GPU_COMPUTE_TFLOPS: dict[str, float] = {
    "RTX A4000": 77.0,
    "RTX 2000 Ada": 89.0,
    "RTX A4500": 89.0,
    "RTX 4000 Ada": 90.0,
    "RTX A5000": 89.0,
    "RTX 3090": 71.0,
    "L4": 60.0,
    "RTX Pro 4000": 95.0,
    "RTX 4090": 165.0,
    "RTX 5090": 210.0,
    "RTX A6000": 155.0,
    "A40": 150.0,
    "RTX 6000 Ada": 182.0,
    "L40S": 181.0,
    "A100 SXM 40GB": 312.0,
    "A100 PCIe": 312.0,
    "A100 SXM": 312.0,
    "H100 NVL": 835.0,
    "H100": 990.0,
    "RTX Pro 6000": 250.0,
    "RTX Pro 6000 WK": 250.0,
}
_DEFAULT_TFLOPS = 100.0


def gpu_tflops(name: str) -> float:
    """Peak bf16 tensor TFLOPS for a managed GPU class."""
    return GPU_COMPUTE_TFLOPS.get(name, _DEFAULT_TFLOPS)


def gpu_hourly_usd(name: str) -> float:
    """Static fallback (on-demand list) $/hr for a class."""
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    return info.hourly_usd


# Realized (spot/queue) $/hr a class is actually billed at -- a conservative single per-class
# value, usually below the on-demand list. Deliberately not method-specific: pricing GRPO at
# its full pricier-instance rate would over-quote the total. An unobserved class falls back to
# the list price (no rate invented).
REALIZED_HOURLY_USD: dict[str, float] = {
    "RTX 3090": 0.239,
    "RTX 4090": 0.426,
    "RTX 5090": 0.871,
    "RTX A5000": 0.304,
    "RTX 6000 Ada": 0.601,
    "A100 PCIe": 1.035,
    "A100 SXM": 1.133,
    "H100": 10.037,
}


def realized_hourly_usd(name: str) -> float:
    """Market (spot/queue) $/hr a class is billed at; list price if not yet observed."""
    return REALIZED_HOURLY_USD.get(name, gpu_hourly_usd(name))


def gpu_vram_gb(name: str) -> int:
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    return info.vram_gb


def pick_gpu(
    required_vram_gb: int,
    *,
    pin: str | None = None,
    provider: str | None = None,
    allow_unvalidated: bool = False,
) -> str:
    """Cheapest GPU class that fits ``required_vram_gb`` -- mirrors ``allocator.allocate``.

    Ranks the static registry by (hourly_usd, vram_gb, name), over validated classes by
    default (``allow_unvalidated`` widens the pool). A concrete ``pin`` is canonicalized (an
    unknown one raises) and honored only if it fits, else selection escalates to the cheapest
    fitting class; policy sentinels (auto/cheapest/empty) fall through to cheapest-fit.
    ``provider`` restricts candidates to classes that provider can provision (``providers_for``);
    this filter holds even with ``allow_unvalidated``.
    """

    def _selectable(g: GpuClass) -> bool:
        # Provisionability gate (always) + validation gate (relaxed by allow_unvalidated).
        # allow_unvalidated never lets a class be priced on a substrate that can't serve it.
        if provider not in (None, "auto") and provider not in providers_for(g.name):
            return False
        if allow_unvalidated:
            return True
        if provider in (None, "auto"):
            return g.validated
        return provider in g.validated_on

    candidates = [
        g for g in GPU_INFO.values() if g.vram_gb >= required_vram_gb and _selectable(g)
    ]
    pin_key = (pin or "").strip().lower()
    if pin_key and pin_key not in POLICY_NAMES:
        canonical = canonical_gpu(pin)  # raises UnsupportedGpuError for an unknown pin
        pinned = [g for g in candidates if g.name == canonical]
        if pinned:
            candidates = pinned  # honor the pin when it fits; else escalate to cheapest-fit
    if not candidates:
        raise ValueError(
            f"no GPU class fits >= {required_vram_gb} GB (allow_unvalidated={allow_unvalidated})"
        )
    best = min(candidates, key=lambda g: (g.hourly_usd, g.vram_gb, g.name))
    return best.name


# ===== Model-size facts =====
_DOWNLOAD_BYTES_PER_PARAM = 2.0
DEFAULT_PARAMS_B = 4.0  # when a model's size can't be read (open-model policy, offline probe)


def _params_str(model_id: str) -> str | None:
    info = MODELS.get(model_id)
    return info.params if info is not None else None


def _params_b_from_str(s: str | None) -> float | None:
    """Leading param count (billions) from a catalog/id string. Case-insensitive on the size
    suffix; expands ``NxMB`` MoE to total (Mixtral-8x7B -> 56B); understands an ``M`` suffix."""
    if not s:
        return None
    moe = re.search(r"([0-9]+)\s*[xX]\s*([0-9]+(?:\.[0-9]+)?)\s*[Bb]\b", s)
    if moe:
        return float(moe.group(1)) * float(moe.group(2))
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[Bb]\b", s)
    if m:
        return float(m.group(1))
    mil = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[Mm]\b", s)
    if mil:
        return float(mil.group(1)) / 1000.0
    return None


def total_params_b(model_id: str, params_str: str | None = None) -> float:
    """Total parameter count in billions (the full checkpoint)."""
    s = params_str if params_str is not None else _params_str(model_id)
    val = _params_b_from_str(s)
    if val is None:
        val = _params_b_from_str(model_id)  # unlisted: parse the id (e.g. "...-35B-A3B")
    return float(val) if val else DEFAULT_PARAMS_B


def active_params_b(model_id: str, params_str: str | None = None) -> float:
    """Active parameters per token (billions). For MoE, the ``A<n>B`` figure; else == total."""
    s = params_str if params_str is not None else _params_str(model_id)
    for hay in (model_id, s or ""):
        m = re.search(r"[Aa](\d+(?:\.\d+)?)\s*[Bb]\b", hay)
        if m:
            return float(m.group(1))
    return total_params_b(model_id, params_str)


def model_quant(model_id: str) -> str:
    """Quantization of the curated entry (``"bf16"`` or ``"4bit-qlora"``); bf16 default."""
    info = MODELS.get(model_id)
    return (getattr(info, "quant", None) or "bf16") if info is not None else "bf16"


def download_weight_gb(model_id: str, params_str: str | None = None) -> float:
    """Approximate GB pulled from the HF hub at cold start (full bf16 checkpoint)."""
    return total_params_b(model_id, params_str) * _DOWNLOAD_BYTES_PER_PARAM


# ===== Reward-grader latency (GRPO) =====
# A SINGLE average grader latency (seconds to score one completion), applied to every
# environment. We deliberately don't classify the env from its slug: that isn't generalizable
# (an unknown env gets mis-tiered), and real graders span ~0.01s (regex / exact-match / math)
# to ~3s (LLM-as-judge, sandboxed code). One average means a heavier-than-average grader is
# under-quoted (we charge less) and a lighter one over-quoted slightly -- we prefer the
# under-quote. A run can still pin its own value via RunConfig.reward_seconds_per_completion.
AVG_REWARD_SECONDS_PER_COMPLETION = 0.3


def reward_seconds_per_completion(override: float | None = None) -> float:
    """Per-completion reward latency (s): the explicit override, else the single average."""
    if override is not None:
        return max(0.0, override)
    return AVG_REWARD_SECONDS_PER_COMPLETION
