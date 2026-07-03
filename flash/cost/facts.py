"""Static lookup facts for the cost model: GPU price/VRAM/compute + cheapest-fit
selection, model size/quant, and reward-grader latency. Pure tables + accessors."""

from __future__ import annotations

from flash.catalog import MODELS
from flash.providers.base import GPU_INFO, GpuClass, providers_for

GPU_COMPUTE_TFLOPS: dict[str, float] = {
    # A10: 125 TFLOPS dense bf16 tensor (NVIDIA spec); Lambda-only 24 GB class, else defaults to 100.
    "A10": 125.0,
    "RTX 4090": 165.0,
    "RTX 5090": 210.0,
    "A100 PCIe": 312.0,
    "A100 SXM": 312.0,
    # A100 SXM 40GB: same SMs/tensor cores as the 80GB A100 SXM, less HBM only (Lambda-only class).
    # Without this, removing A6000 leaves the 33-40 GB Lambda pick falling back to _DEFAULT_TFLOPS.
    "A100 SXM 40GB": 312.0,
    "H100": 990.0,
    # H200: same SMs/tensor cores as H100, more HBM only.
    "H200": 990.0,
    "RTX Pro 6000": 250.0,
    # B200: 2.25 PFLOPS bf16 dense (NVIDIA spec); prevents ~10x cost over-estimate vs _DEFAULT_TFLOPS.
    "B200": 2250.0,
}
_DEFAULT_TFLOPS = 100.0


def gpu_tflops(name: str) -> float:
    """Peak bf16 tensor TFLOPS for a managed GPU class."""
    return GPU_COMPUTE_TFLOPS.get(name, _DEFAULT_TFLOPS)


def gpu_hourly_usd(name: str, provider: str | None = None) -> float:
    """Representative $/hr for a class; uses provider-specific rate when provider='lambda'."""
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    p = (provider or "").strip().lower()
    if p == "lambda" and info.lambda_name:
        from flash.providers.lambdalabs.pricing import hourly_rate

        return hourly_rate(name)
    return info.hourly_usd


def gpu_vram_gb(name: str) -> int:
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    return info.vram_gb


def pick_gpu(required_vram_gb: int, *, provider: str | None = None) -> str:
    """Cheapest GPU class (by $/hr) that fits required_vram_gb; gate-free (submit-time allocator restricts to validated pool)."""

    def _selectable(g: GpuClass) -> bool:
        return provider in (None, "auto") or provider in providers_for(g.name)

    candidates = [g for g in GPU_INFO.values() if g.vram_gb >= required_vram_gb and _selectable(g)]
    if not candidates:
        raise ValueError(f"no GPU class fits >= {required_vram_gb} GB")
    best = min(
        candidates, key=lambda g: (gpu_hourly_usd(g.name, provider=provider), g.vram_gb, g.name)
    )
    return best.name


def total_params_b(model_id: str) -> float:
    """Total parameter count (billions) for a catalog model."""
    info = MODELS.get(model_id)
    if info is None:
        raise ValueError(
            f"unknown model {model_id!r}; cost estimation supports catalog models only "
            f"({', '.join(MODELS)})"
        )
    return info.params_b


def active_params_b(model_id: str) -> float:
    """Active params per token (billions); falls back to total for dense models. Use for FLOPs, not VRAM."""
    info = MODELS.get(model_id)
    if info is None:
        raise ValueError(
            f"unknown model {model_id!r}; cost estimation supports catalog models only "
            f"({', '.join(MODELS)})"
        )
    return info.active_params_b or info.params_b


def model_quant(model_id: str) -> str:
    """Quantization of the catalog entry; defaults to 'bf16'."""
    info = MODELS.get(model_id)
    return (info.quant or "bf16") if info is not None else "bf16"


def download_weight_gb(model_id: str) -> float:
    """Full bf16 checkpoint size in GB (2 bytes/param)."""
    return total_params_b(model_id) * 2.0


# ~1s mid-range default across grader types (regex ~0.01s to LLM judge ~3s).
AVG_REWARD_SECONDS_PER_COMPLETION = 1.0


def reward_seconds_per_completion(override: float | None = None) -> float:
    """Per-completion reward latency (s): the explicit override, else the single average."""
    if override is not None:
        return max(0.0, override)
    return AVG_REWARD_SECONDS_PER_COMPLETION


# Fireworks serverless pricing for the on-policy-distillation GLM teacher, $/1M tokens as
# (input, output). Static like gpu_hourly_usd so cost estimation is deterministic/offline and needs
# NO FIREWORKS_API_KEY. Source: https://fireworks.ai/models/fireworks/glm-5p2 lists
# $1.40 / $0.14 / $4.40 (input / cached input / output) per 1M. opd echo-scores completions
# (max_tokens=0), so only INPUT tokens are billed (the teacher never generates) — but the table keeps
# both so a mispriced entry is obvious. glm-5p1 shares the GLM-5 serverless rate.
TEACHER_USD_PER_1M: dict[str, tuple[float, float]] = {
    "accounts/fireworks/models/glm-5p2": (1.40, 4.40),
    "accounts/fireworks/models/glm-5p1": (1.40, 4.40),
}
# Representative default = the GLM-5 serverless rate. The recipe's default teacher is glm-5p2, so an
# opd run that OMITS [train] teacher_model (priced through this fallback) still quotes the real rate.
_DEFAULT_TEACHER_USD_PER_1M = (1.40, 4.40)
# Fireworks echo-scoring round-trip per completion (wall time, concurrency-bound like reward grading).
AVG_TEACHER_SECONDS_PER_COMPLETION = 2.0


def teacher_price_per_1m(teacher_model: str) -> tuple[float, float]:
    """(input, output) $/1M tokens for a teacher model; falls back to a representative default."""
    return TEACHER_USD_PER_1M.get(teacher_model or "", _DEFAULT_TEACHER_USD_PER_1M)


def teacher_token_cost_usd(
    input_tokens: float, output_tokens: float = 0.0, teacher_model: str = ""
) -> float:
    """External teacher-API dollar cost for a token count. Deterministic; no network."""
    inp, outp = teacher_price_per_1m(teacher_model)
    return (max(0.0, input_tokens) * inp + max(0.0, output_tokens) * outp) / 1_000_000.0


def teacher_seconds_per_completion(override: float | None = None) -> float:
    """Per-completion teacher-scoring latency (s): the explicit override, else the average."""
    if override is not None:
        return max(0.0, override)
    return AVG_TEACHER_SECONDS_PER_COMPLETION
