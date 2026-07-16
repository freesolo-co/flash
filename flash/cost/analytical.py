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

# --- MoE (mixture-of-experts) per-step correction ----------------------------------------------
# For an MoE model (active params << total, e.g. Qwen3.6-35B-A3B: ~3B active / 35B total) the wall
# time per step is NOT the tiny active-param FLOPs the dense model predicts. Routing, all-expert
# coordination, and grouped-GEMM under-utilization at small batch make the step scale with TOTAL
# params. Pricing 35B-A3B on active params under-quoted real RunPod runs by 13-27x (SFT ~1.2s vs
# ~42s realized/step; GRPO ~1.4s vs ~24s). So an MoE prices on TOTAL params at a reduced effective
# MFU + a per-step overhead + a one-time compile. DENSE models (active == total) are unaffected --
# they keep the original active-param path and the MFUs above.
MFU_SFT_TRAIN_MOE = 0.10  # MoE SFT fwd/bwd priced on total params
MFU_TRAIN_MOE = 0.09  # MoE GRPO/OPD policy update priced on total params
MOE_STEP_OVERHEAD_S = 2.0  # routing/dispatch/kernel-launch overhead an MoE pays every step
# One-time kernel/graph compile amortized into the first training step (fused Triton kernels;
# + vLLM cudagraph capture for rollout methods). MoE-only; the prior model omitted it entirely.
COMPILE_MOE_SFT_S = 35.0
COMPILE_MOE_ROLLOUT_S = 48.0  # GRPO / OPD (adds vLLM cudagraph capture)

# Reward grading is CONCURRENT: a step's completions score in parallel slots, so the reward
# wall is ceil(completions / slots) waves x latency, not completions x latency.
REWARD_CONCURRENCY = 16.0

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
DOWNLOAD_RATE_GBPS = 0.4  # effective hf snapshot download (hf_transfer), on top of the base load

# synchronous required saves serialize the lora state and publish it. sft/grpo make two hf commits
# per save (deployable adapter plus resume checkpoint); opd publishes a single deployable adapter.
# price a per-commit floor plus a conservative model-size/rank serialize term. this changes wall/cost
# only, never optimizer steps.
REQUIRED_SAVE_COMMIT_FLOOR_S = 7.5
REQUIRED_SAVE_S_PER_MODEL_B_AT_RANK32 = 1.5
_REQUIRED_SAVE_COMMITS = {"sft": 2, "grpo": 2, "opd": 1}

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
    (a fixed deserialize/placement/init base + a size-scaled download), plus vLLM init for rollouts.
    This elapsed setup time is reported but not included in customer-facing cost."""
    model_load = (
        MODEL_LOAD_BASE_S
        + download_weight_gb(config.model_id, config.model_revision) / DOWNLOAD_RATE_GBPS
    )
    s = WORKER_BOOT_S + DEPS_INSTALL_S + model_load
    if config.has_rollout:
        s += VLLM_INIT_S
    return s


def _opd_step_shape(n: RunConfig) -> tuple[int, int]:
    """(completions per step, prompt+completion tokens per step) for one OPD step, from a NORMALIZED
    config. completions = batch x group; each is billed over the FULL n.seq_len (prompt+completion,
    not completion-only) since the loss forward runs model(prompt_ids + student_ids)."""
    completions = n.batch_size * n.group_size
    return completions, completions * n.seq_len


def required_save_overhead_seconds(config: RunConfig) -> float:
    """Conservative synchronous required-save wall time for exact save_at_steps (sft/grpo/opd)."""
    commits = _REQUIRED_SAVE_COMMITS.get(config.method, 0)
    if not commits or not config.save_at_steps:
        return 0.0
    n = config.normalized()
    serialize_s = (
        REQUIRED_SAVE_S_PER_MODEL_B_AT_RANK32
        * total_params_b(n.model_id, n.model_revision)
        * (n.lora_rank / 32.0)
    )
    per_save = commits * REQUIRED_SAVE_COMMIT_FLOOR_S + serialize_s
    return len(n.save_at_steps) * per_save


def _is_moe(model_id: str) -> bool:
    """True when the model routes each token through a subset of experts (active < total params)."""
    return active_params_b(model_id) < total_params_b(model_id)


def compile_seconds(config: RunConfig, gpu: str) -> float:
    """One-time kernel/graph compile folded into the first training step. MoE-only; 0 for dense
    (whose original timing did not model it). Larger for rollout methods (add vLLM cudagraph
    capture)."""
    _ = gpu
    if not _is_moe(config.model_id):
        return 0.0
    return COMPILE_MOE_SFT_S if config.method == "sft" else COMPILE_MOE_ROLLOUT_S


def seconds_per_step(config: RunConfig, gpu: str) -> float:
    """Steady-state wall time for one optimizer step on ``gpu``."""
    n = config.normalized()
    peak = gpu_tflops(gpu) * 1e12  # FLOP/s
    # An MoE's per-step wall scales with TOTAL params (routing + all-expert coordination + grouped
    # GEMM under-utilization), not the tiny active-param FLOPs; dense models keep active (== total).
    moe = _is_moe(n.model_id)
    params = (total_params_b(n.model_id) if moe else active_params_b(n.model_id)) * 1e9
    overhead = MOE_STEP_OVERHEAD_S if moe else 0.0
    sft_mfu = MFU_SFT_TRAIN_MOE if moe else MFU_SFT_TRAIN
    update_mfu = MFU_TRAIN_MOE if moe else MFU_TRAIN

    if n.is_opd:
        # OPD step = on-policy student rollout (like GRPO) + remote teacher scoring (CONCURRENT
        # Fireworks round-trips, replaces reward grading) + policy update (fwd+bwd only, NO local
        # reference forward — the teacher is the API). Bill local compute on the FULL prompt+completion
        # sequence (see _opd_step_shape), not completion-only, or long-prompt opd is underquoted.
        completions, seq_tokens = _opd_step_shape(n)
        gen_s = (GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM * params * seq_tokens) / (peak * MFU_DECODE)
        update_s = (OPD_UPDATE_FLOPS_PER_TOKEN_PER_PARAM * params * seq_tokens) / (peak * update_mfu)
        teacher_lat = teacher_seconds_per_completion()
        # run_opd's primary path scores a step's completions CONCURRENTLY over Fireworks with a fan-out
        # cap of the step's OWN completion count (prompts_per_step * group_size, opd.py Phase 2), so
        # every completion in a step is scored in ONE parallel wave — the teacher wall is a single
        # latency, NOT the full serial sum (that describes only the CPU-test fallback that can't
        # batch-generate). The teacher endpoint's rate limit is the real ceiling on this fan-out.
        teacher_s = teacher_lat
        return overhead + gen_s + teacher_s + update_s

    if not n.is_grpo:
        flops = SFT_FLOPS_PER_TOKEN_PER_PARAM * params * (n.batch_size * n.seq_len)
        return overhead + flops / (peak * sft_mfu)

    # GRPO step = rollout (G completions/prompt) + concurrent reward grading + policy/ref update.
    completions = n.batch_size * n.group_size
    gen_tokens = completions * n.completion_len
    gen_s = (GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * MFU_DECODE)
    update_s = (GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * update_mfu)
    latency = reward_seconds_per_completion(n.reward_seconds_per_completion)
    reward_s = (
        math.ceil(completions / REWARD_CONCURRENCY) * latency
    )  # ceil: a partial wave still costs one latency
    return overhead + gen_s + reward_s + update_s

def sft_seconds_for_tokens(config: RunConfig, gpu: str, train_tokens: float) -> float:
    """SFT steady-state wall time for an actual token count on ``gpu``."""
    n = config.normalized()
    # MoE prices on total params at a reduced MFU (see seconds_per_step); dense keeps active.
    moe = _is_moe(n.model_id)
    params = (total_params_b(n.model_id) if moe else active_params_b(n.model_id)) * 1e9
    mfu = MFU_SFT_TRAIN_MOE if moe else MFU_SFT_TRAIN
    peak = gpu_tflops(gpu) * 1e12
    flops = SFT_FLOPS_PER_TOKEN_PER_PARAM * params * train_tokens
    return flops / (peak * mfu)


def select_gpu(config: RunConfig, *, max_wall_seconds: float = 0.0) -> tuple[str, int]:
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
        model_revision=config.model_revision,
    )
    gpu = pick_gpu(need, provider=config.provider, max_wall_seconds=max_wall_seconds)
    return gpu, need


def _notes(
    config: RunConfig, raw_train_s: float, wall_capped: bool, cap_s: float
) -> tuple[str, ...]:
    n = config.normalized()
    notes: list[str] = []
    if (quant := model_quant(n.model_id)) != "bf16":
        notes.append(f"{quant}: smaller VRAM footprint -> cheaper GPU class fits")
    if n.is_opd:
        comps, _ = _opd_step_shape(n)
        tsec = teacher_seconds_per_completion()
        from flash.engine.recipe import resolve_teacher

        teacher_name = resolve_teacher(n.teacher_model).display_name
        notes.append(
            f"opd step = student rollout of {n.batch_size}x{n.group_size}={comps} completions "
            f"@ {n.completion_len} tok + {teacher_name} teacher scoring ({tsec:.2f}s/completion) + policy "
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
    required_save_s = required_save_overhead_seconds(n)
    if required_save_s:
        notes.append(
            f"{len(n.save_at_steps)} synchronous required save(s) add "
            f"~{_fmt_duration(required_save_s)}"
        )
    if _is_moe(n.model_id):
        notes.append(
            "MoE model: per-step time priced on total params (routing + all-expert coordination), "
            "not just active-param FLOPs, plus a one-time kernel compile"
        )
    notes.append(f"GPU sized with {vram_headroom() - 1:.0%} VRAM headroom; static GPU $/hr")
    if wall_capped:
        notes.append(
            f"training clamped to fit the {_fmt_duration(cap_s)} wall cap "
            f"(after setup; uncapped: {_fmt_duration(raw_train_s)})"
        )
    return tuple(notes)


def estimate_cost(config: RunConfig, *, wall_cap_s: float = DEFAULT_WALL_CAP_S) -> CostEstimate:
    """Deterministic pre-flight cost calculation."""
    # Billing cap: mirror the runner's max(60, max_wall_seconds) floor so a sub-60s cap isn't underpriced.
    cap_s = max(60.0, float(config.max_wall_seconds)) if config.max_wall_seconds is not None else wall_cap_s
    # Vast market duration filter: price against offers that outlast the run, using the SAME semantics
    # ``usable_offers`` applies at LAUNCH (not the 60s-floored billing cap_s) — a non-positive wall means
    # NO filter, a positive one is floored at 60s by usable_offers itself:
    #   None -> the 24h spec default the run runs under (== DEFAULT_WALL_CAP_S);
    #   > 0  -> that wall;   <= 0 -> 0.0 (no filter, exactly like launch).
    if config.max_wall_seconds is None:
        market_wall_s = wall_cap_s
    elif config.max_wall_seconds > 0:
        market_wall_s = float(config.max_wall_seconds)
    else:
        market_wall_s = 0.0
    if config.exact_type:
        # exact_type is provisioned at LAUNCH through allocate (disk-aware, live rate). quote the same
        # path for BOTH auto and a pinned provider so the estimate matches the actual launch rate, and a
        # disk floor lifts the quote off the wrong class. provider="" lets allocate pick when unpinned.
        from flash.providers.allocator import allocate

        allocation = allocate(
            config.model_id,
            config.method,
            train=config.train_knobs(),
            thinking=config.thinking,
            max_wall_seconds=market_wall_s,
            disk_gb=config.disk_gb,
            provider=("" if config.provider == "auto" else config.provider),
            exact_type=config.exact_type,
            model_revision=config.model_revision,
        )
        gpu = allocation.gpu
        quote_provider = allocation.provider
        hourly = allocation.hourly_usd
        need = allocation.min_vram_gb
    else:
        gpu, need = select_gpu(config, max_wall_seconds=market_wall_s)
        quote_provider = config.provider
        # quote the same vram-floored vast market pick_gpu selected under (min_vram_gb=need): without the
        # floor the rate lookup searches from the smallest managed class, letting cheap small-card offers
        # crowd a high-vram selection off the limited page -> it silently falls back to the static rate.
        hourly = gpu_hourly_usd(
            gpu,
            provider=quote_provider,
            max_wall_seconds=market_wall_s,
            min_vram_gb=need,
            exact_type="",  # this branch is only reached when exact_type is empty
        )

    setup = setup_seconds(config)
    sps = seconds_per_step(config, gpu)
    required_save_s = required_save_overhead_seconds(config)
    # A one-time kernel/graph compile is paid once on the first step (MoE-only; 0 for dense). It is
    # training GPU time, so it belongs in the (billed) train term, not setup.
    compile_s = compile_seconds(config, gpu)
    raw_train = compile_s + config.steps * sps + required_save_s
    if not config.is_grpo and config.train_tokens is not None:
        raw_train = compile_s + sft_seconds_for_tokens(config, gpu, config.train_tokens) + required_save_s
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
        _, tokens_per_step = _opd_step_shape(n)
        teacher_input_tokens = effective_steps * tokens_per_step
        teacher_api_usd = teacher_token_cost_usd(teacher_input_tokens, 0.0, config.teacher_model)

    return CostEstimate(
        model_id=config.model_id,
        method=config.method,
        steps=config.steps,
        gpu=gpu,
        provider=quote_provider,
        gpu_vram_gb=gpu_vram_gb(gpu),
        required_vram_gb=need,
        gpu_hourly_usd=hourly,
        setup_seconds=setup,
        seconds_per_step=sps,
        train_seconds=train,
        wall_clock_seconds=wall,
        wall_capped=wall_capped,
        # total_usd is the customer gpu charge. the platform-owned teacher spend is itemized
        # only as a diagnostic and is not passed through to the customer.
        total_usd=train / 3600.0 * hourly,
        teacher_api_usd=teacher_api_usd,
        notes=_notes(config, raw_train, wall_capped, cap_s),
    )
