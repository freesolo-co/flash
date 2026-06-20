"""The analytical cost model -- the deterministic ground truth.

Total cost = wall-clock hours x GPU $/hr. Wall clock = cold-start setup + steps x per-step
time, where per-step time is a first-principles FLOPs estimate (a small multiple of
active-params x tokens, over the GPU's peak bf16 throughput x a calibrated MFU). GRPO splits
each step into a vLLM rollout (decode-bound) + reward grading + policy/reference update.
"""

from __future__ import annotations

import math

from flash.providers.allocator import required_vram_gb, vram_headroom
from flash.providers.base import providers_for, unvalidated_allowed

from .facts import (
    active_params_b,
    download_weight_gb,
    gpu_tflops,
    gpu_vram_gb,
    model_quant,
    pick_gpu,
    realized_hourly_usd,
    reward_seconds_per_completion,
    total_params_b,
)
from .types import CostEstimate, RunConfig

# --- compute model (FLOPs per token per active-parameter) ---
SFT_FLOPS_PER_TOKEN_PER_PARAM = 6.0  # forward (2) + backward (4)
GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM = 2.0  # autoregressive rollout forward
GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM = 8.0  # policy fwd+bwd (6) + frozen-ref fwd (2)

# Model-FLOPs utilization: fraction of peak the run sustains. Calibrated (with cold-start
# below) against real RunPod/Vast wall-clock measurements. LoRA + small batches sit well
# below dense-pretraining MFU.
MFU_TRAIN = 0.35  # GRPO policy/reference update
MFU_SFT_TRAIN = 0.25  # SFT fwd/bwd (smaller effective batch, long sequences -> less efficient)
MFU_DECODE = 0.12  # batched vLLM rollout (decode is memory-bandwidth-bound)

# Reward grading runs CONCURRENTLY: a step's completions are scored in parallel slots, so the
# reward wall is ceil(completions / slots) waves x per-completion latency, NOT completions x
# latency (which over-counts a heavy grader ~100x and saturates the wall cap).
REWARD_CONCURRENCY = 16.0

# --- cold-start overhead (seconds): container boot + deps + model download (+ vLLM init for
# GRPO). Jointly calibrated with MFU. The per-run implied cold-start spans ~470-840s, the
# dominant source of irreducible per-run error -- we fit the central value, not the noise.
WORKER_BOOT_S = 180.0
DEPS_INSTALL_S = 120.0
VLLM_INIT_S = 120.0  # GRPO only: vLLM engine + tokenizer load
DOWNLOAD_RATE_GBPS = 0.4  # effective HF snapshot download (hf_transfer)

DEFAULT_WALL_CAP_S = 24 * 3600  # spec ``gpu.max_wall_seconds`` default


def _fmt_duration(seconds: float) -> str:
    """Human duration for notes: sub-hour caps render in minutes (a 60s cap is ``1m``, not the
    confusing ``0h`` that ``{s/3600:.0f}h`` would print), whole hours stay clean (``24h``),
    fractional multi-hour spans show one decimal (``1.5h``)."""
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    hours = seconds / 3600
    return f"{hours:.0f}h" if abs(hours - round(hours)) < 1e-9 else f"{hours:.1f}h"


def setup_seconds(config: RunConfig) -> float:
    """Cold-start wall time billed before the first optimizer step."""
    s = WORKER_BOOT_S + DEPS_INSTALL_S
    s += download_weight_gb(config.model_id) / DOWNLOAD_RATE_GBPS
    if config.is_grpo:
        s += VLLM_INIT_S
    return s


def seconds_per_step(config: RunConfig, gpu: str) -> float:
    """Steady-state wall time for one optimizer step on ``gpu``."""
    n = config.normalized()
    active = active_params_b(n.model_id) * 1e9
    peak = gpu_tflops(gpu) * 1e12  # FLOP/s

    if not n.is_grpo:
        tokens = n.batch_size * n.seq_len
        flops = SFT_FLOPS_PER_TOKEN_PER_PARAM * active * tokens
        return flops / (peak * MFU_SFT_TRAIN)

    # GRPO step = rollout (generate G completions/prompt) + reward (score each) + update.
    completions = n.batch_size * n.group_size
    gen_tokens = completions * n.completion_len
    gen_s = (GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM * active * gen_tokens) / (peak * MFU_DECODE)
    update_s = (GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM * active * gen_tokens) / (peak * MFU_TRAIN)
    # ceil() waves: a partial final wave still occupies the slots for a whole latency.
    latency = reward_seconds_per_completion(n.reward_seconds_per_completion)
    reward_s = math.ceil(completions / REWARD_CONCURRENCY) * latency
    return gen_s + reward_s + update_s


def _offline_open_model_vram_gb(config: RunConfig) -> int | None:
    """VRAM for an UNLISTED model whose size can't be read from HF (offline/no-creds).

    Offline (``FLASH_SKIP_NET``), the allocator returns a flat 24 GB tier for any open model
    even when the id encodes a big size. Here we mirror the allocator's open-model branch
    (heavier GRPO phase + headroom) seeded with the parsed param count instead. Returns None
    (defer to the allocator) for a catalog'd model or one whose size IS readable from HF.
    """
    import os

    from flash.catalog import MODELS
    from flash.engine.vram import estimate_vram_gb, fetch_hf_params_b, grpo_seq_escalation_gb

    # Only the advertised offline mode sizes from the parsed id; a transient HF failure must
    # not silently switch strategy (the allocator still uses its flat-24 fallback at submit).
    if not os.environ.get("FLASH_SKIP_NET"):
        return None
    if MODELS.get(config.model_id) is not None:
        return None
    if fetch_hf_params_b(config.model_id) is not None:
        return None

    params_b = total_params_b(config.model_id)
    knobs = config.train_knobs()
    est = estimate_vram_gb(
        params_b,
        "grpo",
        seq_len=knobs.get("max_length", 1024),
        max_tokens=knobs.get("max_tokens"),
        lora_rank=knobs.get("lora_rank", 32),
        batch_size=knobs.get("batch_size", 1),
        group_size=knobs.get("group_size", 8),
        thinking=config.thinking,
    )
    need = math.ceil(est * vram_headroom())
    if config.is_grpo:
        need += grpo_seq_escalation_gb(params_b, knobs.get("max_length", 1024))
    return need


def select_gpu(config: RunConfig) -> tuple[str, int]:
    """(chosen GPU class, required VRAM GB) for the run, offline/deterministic."""
    need = _offline_open_model_vram_gb(config)
    if need is None:
        need = required_vram_gb(
            config.model_id, config.method, train=config.train_knobs(), thinking=config.thinking
        )
    # ``allow_unvalidated`` is tri-state; resolve a None the same way the submit-time allocator
    # does (``unvalidated_allowed``) so the estimate's pool matches what would be allocated.
    gpu = pick_gpu(
        need,
        pin=config.gpu,
        provider=config.provider,
        allow_unvalidated=unvalidated_allowed(config.allow_unvalidated),
    )
    return gpu, need


def _notes(
    config: RunConfig,
    gpu: str,
    raw_train_s: float,
    wall_capped: bool,
    *,
    cap_s: float = DEFAULT_WALL_CAP_S,
) -> tuple[str, ...]:
    n = config.normalized()
    notes: list[str] = []
    total = total_params_b(n.model_id)
    active = active_params_b(n.model_id)
    if active < total:
        notes.append(f"MoE: {active:.0f}B active of {total:.0f}B total params drive compute cost")
    quant = model_quant(n.model_id)
    if quant != "bf16":
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
    # ``vram_headroom()`` is a multiplier (e.g. 1.1) -> report the fraction (10%), not 110%.
    notes.append(f"GPU sized with {vram_headroom() - 1:.0%} VRAM headroom; market (spot/queue) $/hr")
    if wall_capped:
        per_seed = "" if config.setup_repeats == 1 else "per-seed "
        notes.append(
            f"train clamped to the {_fmt_duration(cap_s)} {per_seed}wall cap "
            f"(uncapped: {_fmt_duration(raw_train_s)})"
        )
    return tuple(notes)


def estimate_cost(config: RunConfig, *, wall_cap_s: float = DEFAULT_WALL_CAP_S) -> CostEstimate:
    """Deterministic pre-flight cost estimate -- the analytical ground truth."""
    gpu, need = select_gpu(config)
    hourly = realized_hourly_usd(gpu)  # market (spot/queue) rate runs are billed at
    # The runner enforces max(60, spec.gpu.max_wall_seconds); mirror that floor so a sub-60s cap
    # isn't priced below what the run actually bills (e.g. a zero-dollar estimate).
    cap_s = (
        max(60.0, float(config.max_wall_seconds))
        if config.max_wall_seconds is not None
        else wall_cap_s
    )

    # A multi-seed run is N independent jobs, each cold-starting and capped on its OWN wall.
    # Price one seed, clamp THAT to the cap, then multiply -- not capping the aggregate once.
    seeds = config.setup_repeats
    setup_per_seed = setup_seconds(config)
    sps = seconds_per_step(config, gpu)
    raw_train_per_seed = (config.steps / seeds) * sps

    # The cap is on total per-seed wall; setup is billed too, so clamp training to fit (a cap
    # shorter than setup leaves zero training budget and clamps the billed setup to the cap).
    wall_capped = (setup_per_seed + raw_train_per_seed) > cap_s
    setup_per_seed = min(setup_per_seed, cap_s)
    train_per_seed = max(0.0, cap_s - setup_per_seed) if wall_capped else raw_train_per_seed

    setup = setup_per_seed * seeds
    train = train_per_seed * seeds
    wall = setup + train
    total_usd = wall / 3600.0 * hourly

    # Report the provider implied by the CHOSEN hardware. Under the "auto" sentinel the
    # selected class may be provisionable on only one substrate (e.g. a Vast-only "H100 NVL"),
    # in which case the estimate's "chosen hardware" provider is concretely that substrate, not
    # "auto". A genuinely multi-substrate class stays "auto" (the allocator picks at submit
    # time); an explicit pin is always kept as-is.
    provider = config.provider
    if provider == "auto":
        provs = providers_for(gpu)
        if len(provs) == 1:
            provider = provs[0]

    return CostEstimate(
        model_id=config.model_id,
        method=config.method,
        steps=config.steps,
        gpu=gpu,
        provider=provider,
        gpu_vram_gb=gpu_vram_gb(gpu),
        required_vram_gb=need,
        gpu_hourly_usd=hourly,
        setup_seconds=setup,
        seconds_per_step=sps,
        train_seconds=train,
        wall_clock_seconds=wall,
        wall_capped=wall_capped,
        total_usd=total_usd,
        notes=_notes(config, gpu, raw_train_per_seed, wall_capped, cap_s=cap_s),
    )
