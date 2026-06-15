"""Coarse VRAM-fit estimation for one-consumer-GPU LoRA jobs.

Used by the open-model policy (``model_policy = "allow"``) to sanity-check that an
unlisted HF model can plausibly run on the requested GPU before provisioning it.

These are deliberately coarse heuristics (documented ±20%): they exist to catch
*provably impossible* configurations (70B bf16 on a 24 GB card) and to warn on tight
fits — not to guarantee success. Calibrated against the measured catalog entries
(Qwen3-0.6B/4B/8B, Qwen3.5 dense, Qwen3-30B-A3B QLoRA).
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _gpu_vram_table() -> dict[str, int]:
    try:
        from autoslm.providers.base import GPU_INFO

        return {name: info.vram_gb for name, info in GPU_INFO.items()}
    except Exception:
        return {"RTX 4090": 24, "RTX 5090": 32}


GPU_VRAM_GB = _gpu_vram_table()

_BYTES_PER_PARAM = {
    "bf16": 2.0,
    "fp16": 2.0,
    "4bit-qlora": 0.55,  # NF4 weights + quantization constants
}

# Fixed overheads (GB): CUDA context + activations w/ gradient checkpointing +
# LoRA params/grads/Adam states (tiny at rank<=64) + fragmentation headroom.
_BASE_OVERHEAD_GB = 4.0
# Colocated GRPO extras: vLLM KV cache + sampler scratch (with sleep mode the engine
# weights are offloaded between steps, so we don't double-count full weights).
_GRPO_KV_OVERHEAD_GB = 3.0


@dataclass(frozen=True)
class VramEstimate:
    params_b: float | None
    algorithm: str
    quant: str
    est_gb: float | None
    gpu: str
    gpu_gb: int
    verdict: str  # "fits" | "tight" | "too_big" | "unknown"

    def describe(self) -> str:
        if self.est_gb is None:
            return f"{self.gpu}: VRAM need unknown (could not read model size)"
        return (
            f"{self.gpu} ({self.gpu_gb} GB): estimated ~{self.est_gb:.0f} GB needed "
            f"({self.params_b:.1f}B params, {self.quant}, {self.algorithm}) -> {self.verdict}"
        )


def estimate_vram_gb(params_b: float, algorithm: str, quant: str = "bf16") -> float:
    """Estimated peak VRAM (GB) for a LoRA job on one GPU.

    sft:  weights + activations/overhead
    grpo: trainer weights + (sleep-mode) colocated vLLM KV/scratch + overhead
    """
    bpp = _BYTES_PER_PARAM.get(quant, 2.0)
    weights = params_b * bpp
    algo = "grpo" if algorithm in ("grpo", "rl") else algorithm
    est = weights + _BASE_OVERHEAD_GB
    if algo == "grpo":
        # Sleep-mode colocate: vLLM weights offloaded between steps, but KV cache,
        # sampler scratch, and the wake-phase overlap still add real pressure.
        est += _GRPO_KV_OVERHEAD_GB + 0.5 * weights
    return est


def fetch_hf_params_b(model_id: str) -> float | None:
    """Total params (billions) from the HF API safetensors metadata (no download)."""
    if os.environ.get("AUTOSLM_SKIP_NET"):
        return None
    try:
        from huggingface_hub import HfApi

        info = HfApi(token=os.environ.get("HUGGINGFACE_TOKEN")).model_info(
            model_id, expand=["safetensors"]
        )
        total = getattr(getattr(info, "safetensors", None), "total", None)
        if total:
            return float(total) / 1e9
    except Exception:
        # Best-effort size probe (network/HF-metadata may be unavailable); fall through
        # to None so callers report "size unknown" rather than failing.
        pass
    return None


def check_fit(
    model_id: str,
    algorithm: str,
    gpu: str,
    quant: str = "bf16",
    params_b: float | None = None,
) -> VramEstimate:
    """Estimate whether ``model_id`` plausibly trains on ``gpu``; never raises."""
    gpu_gb = GPU_VRAM_GB.get(gpu, 32)
    if params_b is None:
        params_b = fetch_hf_params_b(model_id)
    if params_b is None:
        return VramEstimate(None, algorithm, quant, None, gpu, gpu_gb, "unknown")
    est = estimate_vram_gb(params_b, algorithm, quant)
    if est > gpu_gb * 1.15:
        verdict = "too_big"
    elif est > gpu_gb * 0.85:
        verdict = "tight"
    else:
        verdict = "fits"
    return VramEstimate(params_b, algorithm, quant, est, gpu, gpu_gb, verdict)
