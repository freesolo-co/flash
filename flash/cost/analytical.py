"""The analytical cost model: total = wall-clock hours x GPU $/hr, where wall = cold-start
setup + steps x per-step time (a FLOPs/MFU estimate). GRPO splits each step into a vLLM
rollout + reward grading + policy/reference update."""

from __future__ import annotations

import math

from flash.providers.allocator import required_vram_gb, vram_headroom

from .facts import (
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

# FLOPs per token per active-parameter.
SFT_FLOPS_PER_TOKEN_PER_PARAM = 6.0  # forward (2) + backward (4)
GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM = 2.0  # autoregressive rollout forward
GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM = 8.0  # policy fwd+bwd (6) + frozen-ref fwd (2)

# Model-FLOPs utilization (fraction of peak sustained), calibrated against real RunPod
# wall clock. LoRA + small batches sit well below dense-pretraining MFU.
MFU_TRAIN = 0.35  # GRPO policy/reference update
MFU_SFT_TRAIN = 0.25  # SFT fwd/bwd (smaller effective batch, long sequences)
MFU_DECODE = 0.12  # batched vLLM rollout (decode is memory-bandwidth-bound)

# Reward grading is CONCURRENT: a step's completions score in parallel slots, so the reward
# wall is ceil(completions / slots) waves x latency, not completions x latency.
REWARD_CONCURRENCY = 16.0

# Cold-start overhead (seconds): container boot + deps + model load (+ vLLM init for GRPO).
#
# Calibrated against a real fresh-worker run (0.8B SFT, RTX 3090 @ $0.239/hr) whose billed wall
# was ~708s for only ~26 priced steps -- i.e. cold start, not training, dominated. A fresh worker
# spent ~12.5 min in `sft_model_load` alone (download + checkpoint deserialize + GPU placement +
# framework/CUDA init), so the MODEL-LOAD term -- not boot/deps -- is the dominant cost of a short
# job. MODEL_LOAD_BASE_S is the fixed (size-independent) load/init overhead; the download term on
# top of it scales with checkpoint size, so bigger models pay a longer cold start.
WORKER_BOOT_S = 120.0  # container pull + start
DEPS_INSTALL_S = 90.0  # pip/uv resolve + install
MODEL_LOAD_BASE_S = 235.0  # fixed checkpoint deserialize + GPU placement + framework/CUDA init
VLLM_INIT_S = 120.0
DOWNLOAD_RATE_GBPS = 0.4  # effective HF snapshot download (hf_transfer), on top of the base load

DEFAULT_WALL_CAP_S = 24 * 3600  # spec gpu.max_wall_seconds default


def _fmt_duration(seconds: float) -> str:
    """Human duration for notes: seconds < 1m, minutes < 1h, else whole/1-decimal hours."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours = seconds / 3600
    return f"{hours:.0f}h" if abs(hours - round(hours)) < 1e-9 else f"{hours:.1f}h"


def setup_seconds(config: RunConfig) -> float:
    """Cold-start wall time billed before the first optimizer step: container boot + deps + model
    load (a fixed deserialize/placement/init base + a size-scaled download), plus vLLM init for
    GRPO. The model-load term dominates a short job's bill (see the constants above)."""
    model_load = MODEL_LOAD_BASE_S + download_weight_gb(config.model_id) / DOWNLOAD_RATE_GBPS
    s = WORKER_BOOT_S + DEPS_INSTALL_S + model_load
    if config.is_grpo:
        s += VLLM_INIT_S
    return s


def seconds_per_step(config: RunConfig, gpu: str) -> float:
    """Steady-state wall time for one optimizer step on ``gpu``."""
    n = config.normalized()
    params = total_params_b(n.model_id) * 1e9
    peak = gpu_tflops(gpu) * 1e12  # FLOP/s

    if not n.is_grpo:
        flops = SFT_FLOPS_PER_TOKEN_PER_PARAM * params * (n.batch_size * n.seq_len)
        return flops / (peak * MFU_SFT_TRAIN)

    # GRPO step = rollout (G completions/prompt) + concurrent reward grading + policy/ref update.
    completions = n.batch_size * n.group_size
    gen_tokens = completions * n.completion_len
    gen_s = (GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * MFU_DECODE)
    update_s = (GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * MFU_TRAIN)
    latency = reward_seconds_per_completion(n.reward_seconds_per_completion)
    reward_s = math.ceil(completions / REWARD_CONCURRENCY) * latency  # ceil: a partial wave still costs one latency
    return gen_s + reward_s + update_s


def select_gpu(config: RunConfig) -> tuple[str, int]:
    """(chosen GPU class, required VRAM GB): the cheapest fitting class for the cost.

    Uses ``pick_gpu``, which (unlike the submit-time allocator) intentionally stays gate-free —
    it considers every fitting class, validated or not — so the estimate reflects the cheapest
    card that *could* run the job. The live allocator restricts to the validated pool, so the
    actually-provisioned class can be pricier than this. Catalog sizing is offline/deterministic."""
    total_params_b(config.model_id)  # catalog-only: reject a non-catalog model before any (HF) sizing
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
    hourly = gpu_hourly_usd(gpu)
    # Mirror the runner's max(60, max_wall_seconds) floor so a sub-60s cap isn't underpriced.
    cap_s = max(60.0, float(config.max_wall_seconds)) if config.max_wall_seconds is not None else wall_cap_s

    # Each seed is its own job (own cold start + own wall cap): price one seed, clamp, x seeds.
    seeds = config.setup_repeats
    setup_per_seed = setup_seconds(config)
    sps = seconds_per_step(config, gpu)
    raw_train_per_seed = (config.steps / seeds) * sps

    # The cap is on total per-seed wall; setup is billed too, so clamp training to fit it.
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
