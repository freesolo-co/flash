"""The analytical cost model: total = training-only GPU hours x GPU $/hr.

Elapsed wall clock still includes cold-start setup + steps x per-step time, but setup/cold-start
is reported as non-billable. GRPO splits each step into a vLLM rollout + reward grading +
policy/reference update.
"""

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
    teacher_seconds_per_completion,
    teacher_token_cost_usd,
    total_params_b,
)
from .types import CostEstimate, RunConfig

# FLOPs per token per active-parameter.
SFT_FLOPS_PER_TOKEN_PER_PARAM = 6.0  # forward (2) + backward (4)
GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM = 2.0  # autoregressive rollout forward
GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM = 8.0  # policy fwd+bwd (6) + frozen-ref fwd (2)
# OPD's update is policy fwd+bwd ONLY (6): the teacher's per-token logprobs come from the
# Fireworks API, so there is NO local frozen-reference forward (GRPO's extra 2).
OPD_UPDATE_FLOPS_PER_TOKEN_PER_PARAM = 6.0

# Model-FLOPs utilization (fraction of peak sustained), calibrated against real RunPod
# wall clock. LoRA + small batches sit well below dense-pretraining MFU.
MFU_TRAIN = 0.35  # GRPO policy/reference update
MFU_SFT_TRAIN = 0.25  # SFT fwd/bwd (smaller effective batch, long sequences)
MFU_DECODE = 0.12  # batched vLLM rollout (decode is memory-bandwidth-bound)

# Reward grading is CONCURRENT: a step's completions score in parallel slots, so the reward
# wall is ceil(completions / slots) waves x latency, not completions x latency.
REWARD_CONCURRENCY = 16.0
# OPD teacher scoring is likewise concurrent (parallel Fireworks calls per step).
TEACHER_CONCURRENCY = 16.0

# Cold-start overhead (seconds): container boot + deps + model load (+ vLLM init for GRPO).
#
# Calibrated against a real fresh-worker run (0.8B SFT, RTX 3090 @ $0.239/hr) whose elapsed wall
# was ~708s for only ~26 priced steps -- i.e. cold start, not training, dominated. A fresh worker
# spent ~12.5 min in `sft_model_load` alone (download + checkpoint deserialize + GPU placement +
# framework/CUDA init), so the MODEL-LOAD term -- not boot/deps -- dominates a short job's elapsed
# time. MODEL_LOAD_BASE_S is the fixed (size-independent) load/init overhead; the download term on
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
    """Cold-start wall time before the first optimizer step: container boot + deps + model load
    (a fixed deserialize/placement/init base + a size-scaled download), plus vLLM init for GRPO.
    This elapsed setup time is reported but not included in customer-facing cost."""
    model_load = MODEL_LOAD_BASE_S + download_weight_gb(config.model_id) / DOWNLOAD_RATE_GBPS
    s = WORKER_BOOT_S + DEPS_INSTALL_S + model_load
    if config.is_grpo:
        s += VLLM_INIT_S
    return s


def seconds_per_step(config: RunConfig, gpu: str) -> float:
    """Steady-state wall time for one optimizer step on ``gpu``."""
    n = config.normalized()
    # Per-token FLOPs scale with the ACTIVE params (an MoE token routes through only a subset of
    # experts); for a dense model this equals the total. Memory/size terms below keep total_params_b.
    params = active_params_b(n.model_id) * 1e9
    peak = gpu_tflops(gpu) * 1e12  # FLOP/s

    if n.is_opd:
        # OPD step = on-policy student rollout (like GRPO) + remote teacher scoring
        # (concurrent Fireworks round-trips, replaces reward grading) + policy update (fwd+bwd
        # only, NO local reference forward — the teacher is the API).
        completions = n.batch_size * n.group_size
        gen_tokens = completions * n.completion_len
        gen_s = (GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * MFU_DECODE)
        update_s = (OPD_UPDATE_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * MFU_TRAIN)
        teacher_lat = teacher_seconds_per_completion(config.reward_seconds_per_completion)
        teacher_s = math.ceil(completions / TEACHER_CONCURRENCY) * teacher_lat
        return gen_s + teacher_s + update_s

    if not n.is_grpo:
        flops = SFT_FLOPS_PER_TOKEN_PER_PARAM * params * (n.batch_size * n.seq_len)
        return flops / (peak * MFU_SFT_TRAIN)

    # GRPO step = rollout (G completions/prompt) + concurrent reward grading + policy/ref update.
    completions = n.batch_size * n.group_size
    gen_tokens = completions * n.completion_len
    gen_s = (GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * MFU_DECODE)
    update_s = (GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * MFU_TRAIN)
    latency = reward_seconds_per_completion(n.reward_seconds_per_completion)
    reward_s = (
        math.ceil(completions / REWARD_CONCURRENCY) * latency
    )  # ceil: a partial wave still costs one latency
    return gen_s + reward_s + update_s


def sft_seconds_for_tokens(config: RunConfig, gpu: str, train_tokens: float) -> float:
    """SFT steady-state wall time for an actual token count on ``gpu``."""
    n = config.normalized()
    params = active_params_b(n.model_id) * 1e9
    peak = gpu_tflops(gpu) * 1e12
    flops = SFT_FLOPS_PER_TOKEN_PER_PARAM * params * train_tokens
    return flops / (peak * MFU_SFT_TRAIN)


def select_gpu(config: RunConfig) -> tuple[str, int]:
    """(chosen GPU class, required VRAM GB): the cheapest fitting class for the cost.

    Uses ``pick_gpu``, which (unlike the submit-time allocator) intentionally stays gate-free —
    it considers every fitting class, validated or not — so the estimate reflects the cheapest
    card that *could* run the job. The live allocator restricts to the validated pool, so the
    actually-provisioned class can be pricier than this. Catalog sizing is offline/deterministic."""
    total_params_b(
        config.model_id
    )  # catalog-only: reject a non-catalog model before any (HF) sizing
    need = required_vram_gb(
        config.model_id,
        config.method,
        train=config.train_knobs(),
        thinking=config.thinking,
    )
    gpu = pick_gpu(need, provider=config.provider)
    return gpu, need


def _notes(
    config: RunConfig, raw_train_s: float, wall_capped: bool, cap_s: float
) -> tuple[str, ...]:
    n = config.normalized()
    notes: list[str] = []
    if (quant := model_quant(n.model_id)) != "bf16":
        notes.append(f"{quant}: smaller VRAM footprint -> cheaper GPU class fits")
    if n.is_opd:
        comps = n.batch_size * n.group_size
        tsec = teacher_seconds_per_completion(n.reward_seconds_per_completion)
        notes.append(
            f"opd step = student rollout of {n.batch_size}x{n.group_size}={comps} completions "
            f"@ {n.completion_len} tok + GLM teacher scoring ({tsec:.2f}s/completion) + policy "
            "update (no local reference forward)"
        )
    elif n.is_grpo:
        comps = n.batch_size * n.group_size
        rsec = reward_seconds_per_completion(n.reward_seconds_per_completion)
        notes.append(
            f"GRPO step = vLLM rollout of {n.batch_size}x{n.group_size}={comps} completions "
            f"@ {n.completion_len} tok + reward ({rsec:.2f}s/completion"
            + (f", env {n.environment}" if n.environment else "")
            + ") + policy+reference update"
        )
    elif n.train_tokens is not None:
        notes.append(f"SFT priced on {n.train_tokens:,} actual train tokens")
    notes.append(f"GPU sized with {vram_headroom() - 1:.0%} VRAM headroom; static GPU $/hr")
    if wall_capped:
        notes.append(
            f"training clamped to fit the {_fmt_duration(cap_s)} wall cap "
            f"(after setup; uncapped: {_fmt_duration(raw_train_s)})"
        )
    return tuple(notes)


def estimate_cost(config: RunConfig, *, wall_cap_s: float = DEFAULT_WALL_CAP_S) -> CostEstimate:
    """Deterministic pre-flight cost calculation."""
    gpu, need = select_gpu(config)
    hourly = gpu_hourly_usd(gpu, provider=config.provider)
    # Mirror the runner's max(60, max_wall_seconds) floor so elapsed wall time is not undercounted.
    cap_s = (
        max(60.0, float(config.max_wall_seconds))
        if config.max_wall_seconds is not None
        else wall_cap_s
    )

    setup = setup_seconds(config)
    sps = seconds_per_step(config, gpu)
    raw_train = config.steps * sps
    if not config.is_grpo and config.train_tokens is not None:
        raw_train = sft_seconds_for_tokens(config, gpu, config.train_tokens)
        sps = raw_train / config.steps

    # The cap is on total elapsed wall; setup is reported but not billed, so only training
    # contributes to total_usd.
    wall_capped = (setup + raw_train) > cap_s
    setup = min(setup, cap_s)
    train = max(0.0, cap_s - setup) if wall_capped else raw_train
    wall = setup + train

    # OPD: add the external Fireworks teacher token spend. The teacher echo-scores every
    # sampled completion (input ~ prompt+completion per completion), so bill INPUT tokens over the
    # EFFECTIVE (wall-capped) step count — not the uncapped `steps` — so a wall-capped run's teacher
    # bill tracks the GPU time it is actually billed for.
    teacher_api_usd = 0.0
    if config.is_opd:
        n = config.normalized()
        effective_steps = (train / sps) if sps > 0 else config.steps
        completions_per_step = n.batch_size * n.group_size
        teacher_input_tokens = effective_steps * completions_per_step * n.seq_len
        teacher_api_usd = teacher_token_cost_usd(teacher_input_tokens, 0.0)

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
        total_usd=train / 3600.0 * hourly + teacher_api_usd,
        teacher_api_usd=teacher_api_usd,
        notes=_notes(config, raw_train, wall_capped, cap_s),
    )
