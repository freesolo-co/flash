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


# Realized $/hr a class is actually billed at -- the empirical effective rate (measured cost /
# measured wall) over the real runs, NOT the on-demand list. A calibrated price INPUT, not an
# output adjustment. Deliberately a single conservative per-class value (at/below the dataset
# median, not method-specific): pricing GRPO at its full pricier-instance rate over-quotes the
# total. ``fit_constants`` returns the exact medians to refresh these from.
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
    pin_must_fit: bool = True,
) -> str:
    """Cheapest GPU class that fits ``required_vram_gb`` -- mirrors ``allocator.allocate``.

    Ranks the static registry by (hourly_usd, vram_gb, name), over validated classes by
    default (``allow_unvalidated`` widens the pool). A concrete ``pin`` is canonicalized (an
    unknown one raises) and honored only if it fits; policy sentinels (auto/cheapest/empty)
    fall through to cheapest-fit. ``provider`` restricts candidates to classes that provider
    can provision (``providers_for``); this filter holds even with ``allow_unvalidated``.

    ``pin_must_fit`` is True for a forward estimate (a too-small pin escalates). It is False
    only when GRADING a measured run, where the pin is the card the run demonstrably ran on:
    the pin is then honored even if the offline VRAM heuristic over-estimates and would drop
    it -- otherwise the measured bill is priced on a different GPU. Only a concrete known pin
    is force-honored; the provider/validation gates still apply.
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
            candidates = pinned
        elif not pin_must_fit and _selectable(GPU_INFO[canonical]):
            # Grading a measured run: force the recorded class back in past the VRAM-fit gate
            # only (provider/validation gates above still apply).
            return GPU_INFO[canonical].name
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
# Seconds to grade one completion, by reward "weight".
REWARD_TIERS: dict[str, float] = {
    "trivial": 0.01,  # exact-match / regex / numeric check
    "light": 0.15,  # parsing + multi-field scoring, classification
    "medium": 0.6,  # light tool/eval, multi-step verifier
    "heavy": 3.0,  # sandboxed code execution, LLM-as-judge, web/tool rollout
}

# Environment-slug keyword -> tier (first match wins).
_ENV_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("code", "swe", "exec", "contest", "leetcode"), "heavy"),
    (("judge", "chat", "ticket", "support", "rubric", "dialog"), "heavy"),
    (("search", "browse", "web", "tool", "agent", "linkd"), "medium"),
    (("sentiment", "classif", "extract", "ner", "label"), "light"),
    (("math", "gsm", "arith", "hendrycks", "aime", "bench", "mmlu"), "trivial"),
)

DEFAULT_REWARD_SECONDS = 0.3  # unknown env: assume a light-to-medium grader


def reward_seconds_per_completion(environment: str | None, override: float | None = None) -> float:
    """Per-completion reward latency (s): explicit override, else inferred from the env."""
    if override is not None:
        return max(0.0, override)
    if not environment:
        return DEFAULT_REWARD_SECONDS
    slug = environment.lower()
    for keywords, tier in _ENV_KEYWORDS:
        if any(k in slug for k in keywords):
            return REWARD_TIERS[tier]
    return DEFAULT_REWARD_SECONDS
