"""Static lookup facts for the cost model: GPU price/VRAM/compute + cheapest-fit
selection, model size/quant, and reward-grader latency. Pure tables + accessors."""

from __future__ import annotations

from flash.catalog import MODELS
from flash.providers.base import GPU_INFO, GpuClass, providers_for

GPU_COMPUTE_TFLOPS: dict[str, float] = {
    "RTX 4090": 165.0,
    "RTX 5090": 210.0,
    "RTX A6000": 155.0,
    "A100 PCIe": 312.0,
    "A100 SXM": 312.0,
    "H100": 990.0,
    # H200 is a Hopper part (same SMs/tensor cores as H100, more HBM) -> same bf16 dense TFLOPS.
    "H200": 990.0,
    "RTX Pro 6000": 250.0,
    # B200 (Blackwell datacenter, sm100): NVIDIA spec 2.25 PFLOPS bf16 dense tensor (no sparsity),
    # listed like H100's 990 dense number so the cost estimator doesn't fall back to the 100-TFLOPS
    # default and wildly over-estimate the 35B's train time.
    "B200": 2250.0,
}
_DEFAULT_TFLOPS = 100.0


def gpu_tflops(name: str) -> float:
    """Peak bf16 tensor TFLOPS for a managed GPU class."""
    return GPU_COMPUTE_TFLOPS.get(name, _DEFAULT_TFLOPS)


def gpu_hourly_usd(name: str, provider: str | None = None) -> float:
    """Representative $/hr for a class, on ``provider`` when given.

    The nominal ``GpuClass.hourly_usd`` is the RunPod rate, which is WRONG for a provider-specific
    quote (e.g. a Lambda RTX A6000 is $1.09/hr, not RunPod's $0.49). When ``provider`` is ``lambda``
    or ``vast`` and the class is offered there, price it through that provider's pricing module (live
    with a static fallback); otherwise (runpod/auto/None) use the nominal rate.
    """
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    p = (provider or "").strip().lower()
    if p == "lambda" and info.lambda_name:
        from flash.providers.lambdalabs.pricing import hourly_rate

        return hourly_rate(name)
    if p == "vast" and info.vast_name:
        # Vast is a live verified-datacenter market — its rates differ materially from RunPod's static
        # ones, so a provider="vast" quote must price through the Vast pricing module (live + static
        # fallback), not fall back to GpuClass.hourly_usd (the RunPod rate).
        from flash.providers.vast.pricing import hourly_rate

        return hourly_rate(name)
    return info.hourly_usd


def gpu_vram_gb(name: str) -> int:
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    return info.vram_gb


def pick_gpu(required_vram_gb: int, *, provider: str | None = None) -> str:
    """Cheapest GPU class that fits ``required_vram_gb``, ranked by static $/hr.

    No pin; every fitting class is eligible, validated or not. NOTE this is intentionally
    gate-free: the submit-time allocator restricts to the validated pool, so the
    actually-provisioned class can be pricier than the one priced here. ``provider`` restricts
    candidates to what it can provision.
    """

    def _selectable(g: GpuClass) -> bool:
        return provider in (None, "auto") or provider in providers_for(g.name)

    candidates = [g for g in GPU_INFO.values() if g.vram_gb >= required_vram_gb and _selectable(g)]
    if not candidates:
        raise ValueError(f"no GPU class fits >= {required_vram_gb} GB")
    # Rank by the rate on the REQUESTED provider so a provider-specific quote picks that provider's
    # cheapest fit (not the cheapest by the RunPod nominal rate).
    best = min(candidates, key=lambda g: (gpu_hourly_usd(g.name, provider=provider), g.vram_gb, g.name))
    return best.name


# Model-size facts (catalog-only; five dense text models, no MoE/open-model sizing)
def total_params_b(model_id: str) -> float:
    """Total parameter count (billions) for a catalog model -- the curated ``params_b`` stat."""
    info = MODELS.get(model_id)
    if info is None:
        raise ValueError(
            f"unknown model {model_id!r}; cost estimation supports catalog models only "
            f"({', '.join(MODELS)})"
        )
    return info.params_b


def active_params_b(model_id: str) -> float:
    """Parameters ACTIVE per token (billions) — the per-token FLOPs/step-time size.

    For an MoE this is the curated ``active_params_b`` (a token routes through only a subset of
    experts); for a dense model (``active_params_b`` unset / 0) it falls back to the total
    ``params_b``. Use this for compute (FLOPs) terms; use ``total_params_b`` for memory/size terms
    (VRAM, disk, download), which always size the full checkpoint."""
    info = MODELS.get(model_id)
    if info is None:
        raise ValueError(
            f"unknown model {model_id!r}; cost estimation supports catalog models only "
            f"({', '.join(MODELS)})"
        )
    return info.active_params_b or info.params_b


def model_quant(model_id: str) -> str:
    """Quantization of the catalog entry; ``"bf16"`` for the whole catalog today (bf16 default)."""
    info = MODELS.get(model_id)
    return (info.quant or "bf16") if info is not None else "bf16"


def download_weight_gb(model_id: str) -> float:
    """GB pulled from the HF hub at cold start (full bf16 checkpoint, 2 bytes/param)."""
    return total_params_b(model_id) * 2.0


# A single average grader latency (s/completion) for every env. Graders span ~0.01s (regex/math)
# to ~3s (LLM judge/code); ~1s is a middle-of-the-road default (a run can override it).
AVG_REWARD_SECONDS_PER_COMPLETION = 1.0


def reward_seconds_per_completion(override: float | None = None) -> float:
    """Per-completion reward latency (s): the explicit override, else the single average."""
    if override is not None:
        return max(0.0, override)
    return AVG_REWARD_SECONDS_PER_COMPLETION
