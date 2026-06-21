"""Static lookup facts for the cost model: GPU price/VRAM/compute + cheapest-fit
selection, model size/quant, and reward-grader latency. Pure tables + accessors."""

from __future__ import annotations

import re

from flash.catalog import MODELS
from flash.providers.base import GPU_INFO, GpuClass, canonical_gpu, providers_for

# Policy sentinels that mean "no pin -- pick the cheapest fitting class" (base.POLICY_NAMES
# is gone with the pinning subsystem; the cost picker only needs these two words).
_POLICY_GPUS = {"auto", "cheapest"}

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
# value, the spot/queue discount below the on-demand list (RTX 5090 lists $0.99 but bills ~$0.87,
# A100 PCIe lists $1.39 but bills ~$1.04). ``realized_hourly_usd`` CLAMPS each entry to the
# registry list price, so the "never above list" invariant holds by construction -- a stale or
# tight-market sample that crept above list (e.g. RTX A5000's $0.304 vs its $0.27 list) can never
# over-quote that class, and new entries are future-proofed the same way. Deliberately not
# method-specific: pricing GRPO at its full pricier-instance rate would over-quote the total. A
# class WITHOUT a clean observed rate falls back to its list price (no rate invented). H100 is
# intentionally omitted: the only H100 sample on record was a single 0.8B GRPO run that billed
# ~$10/hr against a $3.29 list (an anomalous surge/tight-market Vast offer, on a model that
# wouldn't even pick an H100) -- one outlier isn't a stable realized rate, so H100 falls back to
# list ($3.29) until a clean multi-run rate is measured.
REALIZED_HOURLY_USD: dict[str, float] = {
    "RTX 3090": 0.239,
    "RTX 4090": 0.426,
    "RTX 5090": 0.871,
    "RTX A5000": 0.304,
    "RTX 6000 Ada": 0.601,
    "A100 PCIe": 1.035,
    "A100 SXM": 1.133,
}


def realized_hourly_usd(name: str) -> float:
    """Market (spot/queue) $/hr a class is billed at; list price if not yet observed.

    Clamped to the on-demand list price so a realized entry can never over-quote vs list
    (conservative: the estimator never quotes above the on-demand rate)."""
    list_price = gpu_hourly_usd(name)
    return min(REALIZED_HOURLY_USD.get(name, list_price), list_price)


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
) -> str:
    """Cheapest GPU class that fits ``required_vram_gb`` -- mirrors ``allocator.allocate``.

    Ranks by (hourly_usd, vram_gb, name). There is NO validation gate: every fitting class is
    eligible (validated or not), so the estimate considers the truly cheapest card -- matching
    the allocator, which always picks the cheapest fitting class across all providers. A concrete
    ``pin`` is canonicalized (unknown raises) and honored only if it fits, else selection
    escalates to cheapest-fit; policy sentinels (auto/cheapest) auto-select. ``provider``
    restricts candidates to what it can provision.
    """

    def _selectable(g: GpuClass) -> bool:
        # Provisionability filter only -- no validation gate (all fitting classes are eligible).
        return provider in (None, "auto") or provider in providers_for(g.name)

    candidates = [
        g for g in GPU_INFO.values() if g.vram_gb >= required_vram_gb and _selectable(g)
    ]
    pin_key = (pin or "").strip().lower()
    if pin_key and pin_key not in _POLICY_GPUS:
        canonical = canonical_gpu(pin)  # raises UnsupportedGpuError for an unknown pin
        pinned = [g for g in candidates if g.name == canonical]
        if pinned:
            candidates = pinned  # honor the pin when it fits; else escalate to cheapest-fit
    if not candidates:
        raise ValueError(f"no GPU class fits >= {required_vram_gb} GB")
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
# A SINGLE average grader latency (s/completion) for every environment -- we don't classify the
# env from its slug (not generalizable). Real graders span ~0.01s (regex/math) to ~3s (LLM
# judge / code); ~1s is a middle-of-the-road average across that range. A run can pin its own
# via RunConfig.reward_seconds_per_completion.
AVG_REWARD_SECONDS_PER_COMPLETION = 1.0


def reward_seconds_per_completion(override: float | None = None) -> float:
    """Per-completion reward latency (s): the explicit override, else the single average."""
    if override is not None:
        return max(0.0, override)
    return AVG_REWARD_SECONDS_PER_COMPLETION
