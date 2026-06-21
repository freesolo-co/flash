"""Static lookup facts for the cost model: GPU price/VRAM/compute + cheapest-fit
selection, model size/quant, and reward-grader latency. Pure tables + accessors."""

from __future__ import annotations

from flash.catalog import MODELS
from flash.providers.base import GPU_INFO, GpuClass, providers_for

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


# Realized (spot/queue) $/hr per class -- the discount below on-demand list (RTX 5090 lists
# $0.99, bills ~$0.87). ``realized_hourly_usd`` CLAMPS to the list price so it can never
# over-quote; a class with no clean observed rate falls back to list.
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
    """Market (spot/queue) $/hr, clamped to the list price; the list price when not observed."""
    list_price = gpu_hourly_usd(name)
    return min(REALIZED_HOURLY_USD.get(name, list_price), list_price)


def gpu_vram_gb(name: str) -> int:
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    return info.vram_gb


def pick_gpu(required_vram_gb: int, *, provider: str | None = None) -> str:
    """Cheapest GPU class that fits ``required_vram_gb``, ranked by the REALIZED (market) $/hr it
    is BILLED at (ties: vram, name) -- so selection is consistent with the bill and approximates
    the allocator, which provisions the cheapest live offer. No pin and no validation gate -- every
    fitting class is eligible. ``provider`` restricts candidates to what it can provision.
    """

    def _selectable(g: GpuClass) -> bool:
        return provider in (None, "auto") or provider in providers_for(g.name)

    candidates = [g for g in GPU_INFO.values() if g.vram_gb >= required_vram_gb and _selectable(g)]
    if not candidates:
        raise ValueError(f"no GPU class fits >= {required_vram_gb} GB")
    best = min(candidates, key=lambda g: (realized_hourly_usd(g.name), g.vram_gb, g.name))
    return best.name


# ===== Model-size facts (catalog-only; five dense text models, no MoE/open-model sizing) =====
def total_params_b(model_id: str) -> float:
    """Total parameter count (billions) for a catalog model -- the curated ``params_b`` stat."""
    info = MODELS.get(model_id)
    if info is None:
        raise ValueError(
            f"unknown model {model_id!r}; cost estimation supports catalog models only "
            f"({', '.join(MODELS)})"
        )
    return info.params_b


def model_quant(model_id: str) -> str:
    """Quantization of the catalog entry (``"bf16"`` or ``"4bit-qlora"``); bf16 default."""
    info = MODELS.get(model_id)
    return (info.quant or "bf16") if info is not None else "bf16"


def download_weight_gb(model_id: str) -> float:
    """GB pulled from the HF hub at cold start (full bf16 checkpoint, 2 bytes/param)."""
    return total_params_b(model_id) * 2.0


# ===== Reward-grader latency (GRPO) =====
# A single average grader latency (s/completion) for every env. Graders span ~0.01s (regex/math)
# to ~3s (LLM judge/code); ~1s is a middle-of-the-road default (a run can override it).
AVG_REWARD_SECONDS_PER_COMPLETION = 1.0


def reward_seconds_per_completion(override: float | None = None) -> float:
    """Per-completion reward latency (s): the explicit override, else the single average."""
    if override is not None:
        return max(0.0, override)
    return AVG_REWARD_SECONDS_PER_COMPLETION
