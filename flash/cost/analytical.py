"""Analytical cost model: wall-clock hours x GPU $/hr."""

from __future__ import annotations

import math

from flash.providers.allocator import required_vram_gb, vram_headroom

from .facts import (
    active_params_b,
    download_weight_gb,
    gpu_hourly_usd,
    gpu_tflops,
    gpu_vram_gb,
    model_quant,
    pick_gpu,
    reward_seconds_per_completion,
    total_params_b,
)
from .types import CostEstimate, RunConfig

SFT_FLOPS_PER_TOKEN_PER_PARAM = 6.0  # forward (2) + backward (4)
GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM = 2.0  # autoregressive rollout forward
GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM = 8.0  # policy fwd+bwd (6) + frozen-ref fwd (2)

# MFU calibrated against real RunPod wall clock; LoRA+small batches sit well below pretraining MFU.
MFU_TRAIN = 0.35  # GRPO policy/reference update
MFU_SFT_TRAIN = 0.25  # SFT fwd/bwd
MFU_DECODE = 0.12  # batched vLLM rollout (memory-bandwidth-bound)

# Reward grading runs in parallel; wall = ceil(completions / slots) waves x latency.
REWARD_CONCURRENCY = 16.0

# Cold-start constants; model-load dominates short jobs (not boot/deps).
WORKER_BOOT_S = 120.0
DEPS_INSTALL_S = 90.0
MODEL_LOAD_BASE_S = 235.0  # fixed deserialize + GPU placement + CUDA init
VLLM_INIT_S = 120.0
DOWNLOAD_RATE_GBPS = 0.4  # effective HF snapshot download (hf_transfer)

DEFAULT_WALL_CAP_S = 24 * 3600  # spec gpu.max_wall_seconds default


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours = seconds / 3600
    return f"{hours:.0f}h" if abs(hours - round(hours)) < 1e-9 else f"{hours:.1f}h"


def setup_seconds(config: RunConfig) -> float:
    """Cold-start wall time: boot + deps + model load + vLLM init (GRPO only)."""
    model_load = MODEL_LOAD_BASE_S + download_weight_gb(config.model_id) / DOWNLOAD_RATE_GBPS
    s = WORKER_BOOT_S + DEPS_INSTALL_S + model_load
    if config.is_grpo:
        s += VLLM_INIT_S
    return s


def seconds_per_step(config: RunConfig, gpu: str) -> float:
    """Steady-state wall time for one optimizer step on ``gpu``."""
    n = config.normalized()
    params = active_params_b(n.model_id) * 1e9
    peak = gpu_tflops(gpu) * 1e12  # FLOP/s

    if not n.is_grpo:
        flops = SFT_FLOPS_PER_TOKEN_PER_PARAM * params * (n.batch_size * n.seq_len)
        return flops / (peak * MFU_SFT_TRAIN)

    completions = n.batch_size * n.group_size
    gen_tokens = completions * n.completion_len
    gen_s = (GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * MFU_DECODE)
    update_s = (GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * MFU_TRAIN)
    latency = reward_seconds_per_completion(n.reward_seconds_per_completion)
    reward_s = math.ceil(completions / REWARD_CONCURRENCY) * latency  # ceil: a partial wave still costs one latency
    return gen_s + reward_s + update_s


def select_gpu(config: RunConfig) -> tuple[str, int]:
    """Return (cheapest fitting GPU class, required VRAM GB). pick_gpu is gate-free; live allocator may pick pricier."""
    total_params_b(config.model_id)  # catalog-only gate: reject unknown model before HF sizing
    need = required_vram_gb(
        config.model_id,
        config.method,
        train=config.train_knobs(),
        thinking=config.thinking,
    )
    gpu = pick_gpu(need, provider=config.provider)
    return gpu, need


def _notes(config: RunConfig, raw_train_s: float, wall_capped: bool, cap_s: float) -> tuple[str, ...]:
    n = config.normalized()
    notes: list[str] = []
    if (quant := model_quant(n.model_id)) != "bf16":
        notes.append(f"{quant}: smaller VRAM footprint -> cheaper GPU class fits")
    if n.is_grpo:
        comps = n.batch_size * n.group_size
        rsec = reward_seconds_per_completion(n.reward_seconds_per_completion)
        notes.append(
            f"GRPO step = vLLM rollout of {n.batch_size}x{n.group_size}={comps} completions "
            f"@ {n.completion_len} tok + reward ({rsec:.2f}s/completion"
            + (f", env {n.environment}" if n.environment else "")
            + ") + policy+reference update"
        )
    notes.append(f"GPU sized with {vram_headroom() - 1:.0%} VRAM headroom; static GPU $/hr")
    if wall_capped:
        per_seed = "" if config.setup_repeats == 1 else "per-seed "
        notes.append(
            f"training clamped to fit the {_fmt_duration(cap_s)} {per_seed}wall cap "
            f"(after setup; uncapped: {_fmt_duration(raw_train_s)})"
        )
    return tuple(notes)


def estimate_cost(config: RunConfig, *, wall_cap_s: float = DEFAULT_WALL_CAP_S) -> CostEstimate:
    """Deterministic pre-flight cost calculation."""
    gpu, need = select_gpu(config)
    hourly = gpu_hourly_usd(gpu, provider=config.provider)
    cap_s = max(60.0, float(config.max_wall_seconds)) if config.max_wall_seconds is not None else wall_cap_s  # mirror runner floor

    # Each seed is its own job with its own cold start and wall cap.
    seeds = config.setup_repeats
    setup_per_seed = setup_seconds(config)
    sps = seconds_per_step(config, gpu)
    raw_train_per_seed = (config.steps / seeds) * sps

    wall_capped = (setup_per_seed + raw_train_per_seed) > cap_s
    setup_per_seed = min(setup_per_seed, cap_s)
    train_per_seed = max(0.0, cap_s - setup_per_seed) if wall_capped else raw_train_per_seed

    setup, train = setup_per_seed * seeds, train_per_seed * seeds
    wall = setup + train

    return CostEstimate(
        model_id=config.model_id,
        method=config.method,
        steps=config.steps,
        gpu=gpu,
        provider=config.provider,
        gpu_vram_gb=gpu_vram_gb(gpu),
        required_vram_gb=need,
        gpu_hourly_usd=hourly,
        setup_seconds=setup,
        seconds_per_step=sps,
        train_seconds=train,
        wall_clock_seconds=wall,
        wall_capped=wall_capped,
        total_usd=wall / 3600.0 * hourly,
        notes=_notes(config, raw_train_per_seed, wall_capped, cap_s),
    )
