"""The analytical cost model -- the deterministic ground truth.

Total cost = wall-clock hours x GPU $/hr. Wall clock = cold-start setup + steps x
per-step time. Per-step time is a first-principles FLOPs estimate (compute = a small
multiple of active-params x tokens, divided by the GPU's peak bf16 throughput times a
calibrated model-FLOPs-utilization), so the model is transparent and every term is
named. GRPO splits each step into a vLLM rollout (generation, decode-bound) plus the
policy/reference update (train-bound) -- which is why GRPO is several times costlier
than SFT per step. Hardware selection reuses ``flash``'s own VRAM matrix + cheapest-
fit rule, so the chosen GPU class matches what the allocator would pick.

These constants are calibrated so end-to-end figures land in a realistic band (cents
for a small SFT smoke; a few-to-tens of dollars for a large GRPO run) rather than to
any single measured run. The model is the *reference* the LLM estimator is graded
against in ``experiment.py``.
"""

from __future__ import annotations

import math

from flash.providers.allocator import required_vram_gb, vram_headroom
from flash.providers.base import unvalidated_allowed

from .config import RunConfig
from .estimate import CostEstimate
from .hardware import gpu_hourly_usd, gpu_tflops, gpu_vram_gb, pick_gpu
from .model_specs import active_params_b, download_weight_gb, model_quant, total_params_b
from .rewards import reward_seconds_per_completion

# --- compute model (FLOPs per token per active-parameter) ---
SFT_FLOPS_PER_TOKEN_PER_PARAM = 6.0  # forward (2) + backward (4)
GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM = 2.0  # autoregressive rollout forward
GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM = 8.0  # policy fwd+bwd (6) + frozen-ref fwd (2)

# Model-FLOPs utilization: fraction of peak the run actually sustains.
MFU_TRAIN = 0.40  # LoRA fwd/bwd/optimizer on a single card
MFU_DECODE = 0.18  # batched vLLM rollout (decode is memory-bound; batching helps)

# --- cold-start overhead (seconds) ---
WORKER_BOOT_S = 180.0  # provision + container start + framework import
DEPS_INSTALL_S = 150.0  # per-run pip deps not already baked into the image
VLLM_INIT_S = 90.0  # GRPO only: vLLM engine + tokenizer load
DOWNLOAD_RATE_GBPS = 0.6  # effective HF snapshot download (hf_transfer)

# Default total wall-clock cap (spec ``gpu.max_wall_seconds`` default = 24h).
DEFAULT_WALL_CAP_S = 24 * 3600


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
        return flops / (peak * MFU_TRAIN)

    # GRPO step = rollout (generate G completions/prompt) + reward (score every
    # completion through the verifiers env) + policy/reference update.
    completions = n.batch_size * n.group_size  # prompts_per_step x group_size
    gen_tokens = completions * n.completion_len
    gen_s = (GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM * active * gen_tokens) / (peak * MFU_DECODE)
    update_s = (GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM * active * gen_tokens) / (peak * MFU_TRAIN)
    reward_s = completions * reward_seconds_per_completion(
        n.environment, n.reward_seconds_per_completion
    )
    return gen_s + reward_s + update_s


def _offline_open_model_vram_gb(config: RunConfig) -> int | None:
    """VRAM for an UNLISTED model whose size can't be read from HF (offline/no-creds).

    ``model_required_vram_gb`` sizes unlisted models from HF metadata and, when that's
    unreadable, returns a flat 24 GB tier -- so offline (``FLASH_SKIP_NET``) every
    open model lands on a 24 GB card even when its id clearly encodes a big size
    (``vendor/foo-70B``). ``model_specs`` parses that size, so here we mirror the
    allocator's OWN open-model branch (GRPO is the heavier phase; same long-context
    escalation + headroom) but seeded with the parsed param count instead of the
    fallback. Returns ``None`` (defer to the allocator) when the model is catalog'd or
    its size IS readable from HF -- so this only ever replaces the flat-24 fallback.
    """
    import os

    from flash.catalog import MODELS
    from flash.engine.vram import (
        estimate_vram_gb,
        fetch_hf_params_b,
        grpo_seq_escalation_gb,
    )

    # Only the advertised offline mode (FLASH_SKIP_NET) sizes from the parsed id. A transient
    # HF/network failure must NOT silently switch strategy: the allocator still uses its flat 24 GB
    # open-model fallback at submit, so preflight GPU/cost would otherwise diverge from the real run.
    if not os.environ.get("FLASH_SKIP_NET"):
        return None

    if MODELS.get(config.model_id) is not None:
        return None  # catalog model: the allocator sizes it from curated facts
    if fetch_hf_params_b(config.model_id) is not None:
        return None  # HF metadata readable: trust the allocator's HF-sized estimate

    params_b = total_params_b(config.model_id)  # parsed from the id (e.g. "...-70B")
    knobs = config.train_knobs()
    # Mirror model_required_vram_gb's open-model branch: size against the heavier GRPO
    # phase regardless of the requested algorithm, with the run's actual knobs + headroom.
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
    # ``allow_unvalidated`` is tri-state. ``None`` means "unspecified", which the submit-time
    # allocator resolves via ``unvalidated_allowed`` (the FLASH_GPU_ALLOW_UNVALIDATED env
    # default) -- so resolve it the SAME way here instead of treating None as validated-only,
    # or a run the env widened would be priced against the narrower validated pool.
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
        rsec = reward_seconds_per_completion(n.environment, n.reward_seconds_per_completion)
        notes.append(
            f"GRPO step = vLLM rollout of {n.batch_size}x{n.group_size}={comps} completions "
            f"@ {n.completion_len} tok + reward ({rsec:.2f}s/completion"
            + (f", env {n.environment}" if n.environment else "")
            + ") + policy+reference update"
        )
    notes.append(f"GPU sized with {vram_headroom():.0%} VRAM headroom; static fallback $/hr")
    if wall_capped:
        per_seed = "" if config.setup_repeats == 1 else "per-seed "
        notes.append(
            f"train clamped to the {cap_s / 3600:.0f}h {per_seed}wall cap "
            f"(uncapped: {raw_train_s / 3600:.1f}h)"
        )
    return tuple(notes)


def estimate_cost(config: RunConfig, *, wall_cap_s: float = DEFAULT_WALL_CAP_S) -> CostEstimate:
    """Deterministic pre-flight cost estimate -- the experiment's ground truth."""
    gpu, need = select_gpu(config)
    hourly = gpu_hourly_usd(gpu)
    # The wall cap is ``gpu.max_wall_seconds`` from the spec (default 24h); fall back to the
    # caller's ``wall_cap_s`` (the module default) when the config doesn't pin one.
    cap_s = float(config.max_wall_seconds) if config.max_wall_seconds is not None else wall_cap_s

    # A multi-seed run is N independent jobs (runner.py: "one allocation per seed"), each of
    # which cold-starts, runs its own steps, and has ``max_wall_seconds`` applied to ITS OWN
    # wall clock. So price one seed (setup + its share of the steps), clamp THAT to the cap,
    # then multiply by the seed count -- summing per-seed capped wall, not capping the
    # aggregate once (which under-/over-prices a multi-seed run whose per-seed cap binds).
    seeds = config.setup_repeats
    setup_per_seed = setup_seconds(config)
    sps = seconds_per_step(config, gpu)
    raw_train_per_seed = (config.steps / seeds) * sps

    # The cap is on TOTAL per-seed wall clock; setup is billed too, so clamp training to fit.
    wall_capped = (setup_per_seed + raw_train_per_seed) > cap_s
    train_per_seed = max(0.0, cap_s - setup_per_seed) if wall_capped else raw_train_per_seed

    setup = setup_per_seed * seeds
    train = train_per_seed * seeds
    wall = setup + train
    total_usd = wall / 3600.0 * hourly

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
        total_usd=total_usd,
        notes=_notes(config, gpu, raw_train_per_seed, wall_capped, cap_s=cap_s),
    )
