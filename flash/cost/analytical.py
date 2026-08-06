"""The analytical cost model: total = training-only GPU hours x GPU $/hr.

Elapsed wall clock still includes cold-start setup + steps x per-step time, but setup/cold-start
is reported as non-billable. GRPO splits each step into a vLLM rollout + reward grading +
policy/reference update.
"""

from __future__ import annotations

import math

from flash.opd_limits import OPD_TEACHER_SCORING_CONCURRENCY, opd_teacher_request_multiplier
from flash.providers.allocator import geometry_safe_gpu_cap, required_vram_gb, vram_headroom

from .facts import (
    GPU_COMPUTE_TFLOPS,
    active_params_b,
    download_weight_gb,
    effective_train_tflops,
    gpu_vram_gb,
    has_nvlink,
    model_quant,
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
# parasail api, so there is no local frozen-reference forward (grpo's extra 2).
OPD_UPDATE_FLOPS_PER_TOKEN_PER_PARAM = 6.0

# Model-FLOPs utilization (fraction of peak sustained), calibrated against real RunPod
# wall clock. LoRA + small batches sit well below dense-pretraining MFU.
MFU_TRAIN = 0.35  # GRPO policy/reference update
MFU_SFT_TRAIN = 0.25  # SFT fwd/bwd (smaller effective batch, long sequences)
MFU_DECODE = 0.12  # batched vLLM rollout (decode is memory-bandwidth-bound)

# --- rollout step floor ------------------------------------------------------------------------
# Per-step work that the FLOPs terms above do not model at all. verl's own timing_s/* keys
# decompose a real grpo step to 100.0% (a2_revtext, Qwen3.5-0.8B, RTX 4090, 8 steps):
#
#   update_actor    62.1%  -> update_s
#   old_log_prob    17.9%  -> NOTHING
#   gen             15.5%  -> gen_s
#   update_weights   3.0%  -> NOTHING
#   save_checkpoint  1.5%  -> NOTHING
#   reward           0.0%  -> reward_s
#
# ~22% of every step had no term. grpo recomputes log-probs under the current policy before each
# update, and syncs trained weights into the vllm rollout engine to keep the next rollout
# on-policy; both are real gpu work proportional to the step, not overhead.
#
# This was invisible for as long as it was because the fictitious reward wall stood in for it:
# 32 completions x the old 1.0 s/completion default = 32.00s of a 32.88s prediction, against a
# MEASURED reward phase of 0.000s. Two large errors in opposite directions, so the total looked
# calibrated while both halves were wrong. Removing the fiction WITHOUT adding this term scores
# 49.8x geometric bias -- far worse than leaving both alone.
#
# MEASURED per card as median(real_step - gpu_bound) over a 45-arm campaign, then scored on 11
# held-out arms (ratio realized/predicted, band 0.70-1.43):
#
#   flash as-is                     geo 3.262x   0/11 in band
#   measured reward only            geo 49.771x  0/11
#   floor only                      geo 0.842x   9/11
#   measured reward + floor         geo 1.050x   8/11   (44/56 on the full set vs 31/56 floor-only)
#
# ONE CONSTANT, not per card and not scaled, because every richer form was measured and lost:
#
#   form                                    held-out (11 arms)   note
#   flat, one constant                      8/11                 shipped
#   flat, per card (6 constants)            8/11                 ties, and INVERTS b200 vs h200
#   flat per card * (completions / 32)      5/11
#   linear in completions, per card         5/11
#   a * completions * params_B + c          6/11                 R^2 0.64
#   k * modelled_gpu_seconds                2/11
#
# The per-card table looks far better in aggregate (44/56 vs 36/56) but 45 of those 56 arms are
# the ones its constants were fitted on. On arms it has never seen it only ties. It is fitting
# each card's model and completion-count mix, not the hardware: H200's 152.5s comes from 4 arms,
# and B200's 81.2s would quote B200 71s per step FASTER than H200 for an identical run, which is
# backwards -- B200 training is H100/H200-class on portable kernels at a higher $/hr.
#
# The floor is not proportional to anything the model already computes: it is ~98x the modelled
# gpu-bound seconds, so any proportional form amplifies a tiny noisy denominator (2/11).
#
# It does grow with completions (g32 77s, g64 147s, g256 231s) and model size (0.8B 71s, 2B 80s,
# 4B 110s), which is old_log_prob's shape -- a forward pass over completions x parameters. But
# the fit arms cluster at 32 and 256 completions while the held-out arms run 16-64, so a slope
# fitted on a 32-vs-256 contrast is not evidence about the gap it would be extrapolating into,
# and every form that tried scored worse than the constant.
#
# This is an empirical aggregate and will drift as hardware, verl, or the engine change. The
# principled fix models old_log_prob and update_weights as real FLOPs terms.
STEP_FLOOR_SECONDS = 78.8


def step_floor_seconds(gpu: str) -> float:
    """Unmodelled per-step work (old_log_prob, weight sync, checkpointing).

    Takes ``gpu`` because the phases are real GPU work and a future FLOPs-based model will need
    it, but returns one constant today: a per-card table was measured and does not beat this one
    out of sample (see above).
    """
    return STEP_FLOOR_SECONDS


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

# Single-turn grpo scores a step's completions SERIALLY, so the reward wall is the full
# completions x latency sum. this is not an accident, it is a deliberate contract: the verl reward
# bridge takes a global lock around the env call so the flash env sees sequential calls
# (worker/rl_train.py). the serialization protects envs whose scorers are not thread-safe, so it
# cannot be removed to match a faster model.
#
# a concurrency divisor here was a flat 16x understatement of the reward term, and reward time lands
# in the FIXED half of a step -- the half no gpu choice can shorten. under-counting it makes a
# latency-bound judge env look gpu-bound, which over-weights raw flops and rents a fast card whose
# speed the run cannot use. multi-turn envs do score concurrently, but multi-turn is unknowable at
# quote time (RunConfig has no such field, and deciding it means LOADING user env code inside a
# pricing function that is deliberately offline). serial is therefore the conservative default: for
# a concurrent env it over-states fixed time, which can only steer the allocator to a cheaper card,
# never to a more expensive one it cannot exploit.

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


def _opd_step_shape(n: RunConfig) -> tuple[int, float]:
    """(completions per step, prompt+completion tokens per step) for one OPD step, from a NORMALIZED
    config. completions = batch x group; each is billed over the FULL prompt+completion length
    (not completion-only) since the loss forward runs model(prompt_ids + student_ids)."""
    completions = n.batch_size * n.group_size
    return completions, completions * _sequence_tokens(n)


def _sequence_tokens(n: RunConfig) -> float:
    """Prompt+completion tokens ONE rollout costs, measured when a profile exists.

    ``n.seq_len`` is ``max_context_tokens``: a capacity ceiling the engine is configured with, not
    the work a step performs. Billing it assumes every rollout fills the context, which measurement
    contradicts -- realized generation ran 0.323x of a 2048-token cap on the reference sample.

    Both halves must be measured together. Substituting a measured completion length while leaving
    the prompt at the context ceiling would price a short completion onto a full-context prompt,
    which is not a shape any rollout has.
    """
    if n.measured_completion_tokens is None or n.measured_prompt_tokens is None:
        return float(n.seq_len)
    measured = n.measured_prompt_tokens + n.measured_completion_tokens
    # the ceiling still binds: a measured mean above it would mean the profile and the run
    # disagree about the engine's context, and the engine wins.
    return min(float(n.seq_len), measured)


def _completion_tokens(n: RunConfig) -> float:
    """Tokens ONE rollout generates, measured when a profile exists, else the declared cap."""
    if n.measured_completion_tokens is None:
        return float(n.completion_len)
    return min(float(n.completion_len), n.measured_completion_tokens)


def _describe_rollout_tokens(n: RunConfig) -> str:
    """How the rollout length reaching the quote should be shown to the user.

    A measured quote and a cap-based quote can differ several-fold, so the note has to say which
    one it is. Reporting a measured mean as though it were the configured cap would make the
    cheaper number look like a pricing change rather than a measurement.
    """
    if n.measured_completion_tokens is None:
        return f"{n.completion_len} tok"
    return f"{_completion_tokens(n):.0f} tok measured (cap {n.completion_len})"


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


# collective overhead means n cards never deliver n times one card's throughput: fsdp all-gathers
# parameters and reduce-scatters gradients every layer, and the share of a step spent in collectives
# grows with the card count. this is the realized fraction of linear scaling per ADDED card, applied
# geometrically. the allocator must never rank a combination on a speedup the interconnect will not
# deliver, because every card in it bills whether or not it contributes.
#
# BOTH constants are MEASURED, one identical 2-card fsdp benchmark per interconnect (same global
# batch in both arms, both arms on one pod, slowest rank defining the step):
#   nvlink 2x A100-SXM4-80GB: 0.39491 s -> 0.22343 s = 1.7675x, per-card 0.884
#   pcie   2x L40S          : 0.57692 s -> 0.40596 s = 1.4212x, per-card 0.711
# a single constant cannot cover both. the previous global 0.85 was ~4% conservative on nvlink but
# ~20% OPTIMISTIC on pcie, and optimistic is the direction that actually misprices a run: it lets a
# 2-card pcie combination win a ranking on scaling it does not deliver, then bills both cards for
# the longer wall time. each constant is set to its own measured per-card efficiency.
MULTI_CARD_SCALING_NVLINK = 0.88
MULTI_CARD_SCALING_PCIE = 0.71


def multi_card_scaling(gpu: str) -> float:
    """Realized fraction of linear scaling per added card of class ``gpu``."""
    return MULTI_CARD_SCALING_NVLINK if has_nvlink(gpu) else MULTI_CARD_SCALING_PCIE


def multi_card_speedup(gpu_count: int, gpu: str) -> float:
    """Throughput multiplier for sharding the gpu-bound half of a step over ``gpu_count`` cards.

    Both measurements are 2-card. Beyond that the geometric form is an extrapolation, and it errs
    conservative by construction: real fabrics degrade faster than geometrically as the collective
    fan-out grows, so this never credits a wide combination with more than it can deliver.

    Clamped to be non-decreasing in ``gpu_count``. Below a scaling factor of ~0.72 the raw
    geometric curve turns back down (at 0.71: 3 cards 1.512x but 4 cards 1.432x), which would model
    a wider combination as SLOWER than a narrower one and let the allocator reject cards that do
    add throughput. Adding a card cannot reduce aggregate throughput on any real fabric, so the
    honest reading of the extrapolation is that scaling FLATTENS, not that it reverses.
    """
    n = max(1, int(gpu_count))
    scaling = multi_card_scaling(gpu)
    return max(k * (scaling ** (k - 1)) for k in range(1, n + 1))


def step_seconds_split(config: RunConfig, gpu: str) -> tuple[float, float]:
    """(gpu-bound, gpu-independent) seconds for one optimizer step on ``gpu``.

    Sharding a step across cards divides the gpu-bound half and leaves the rest untouched: remote
    teacher scoring and reward grading are waits on services no card count speeds up, and an MoE pays
    its routing overhead once per step regardless. Ranking hardware needs the halves apart, because a
    latency-bound job gets no benefit from a faster card while a compute-bound one gets all of it.
    ``seconds_per_step`` is simply their sum, so this is the single source of the step model.
    """
    n = config.normalized()
    peak = effective_train_tflops(gpu) * 1e12  # FLOP/s (realized training throughput; see facts)
    # An MoE's per-step wall scales with TOTAL params (routing + all-expert coordination + grouped
    # GEMM under-utilization), not the tiny active-param FLOPs; dense models keep active (== total).
    moe = _is_moe(n.model_id)
    params = (total_params_b(n.model_id) if moe else active_params_b(n.model_id)) * 1e9
    overhead = MOE_STEP_OVERHEAD_S if moe else 0.0
    sft_mfu = MFU_SFT_TRAIN_MOE if moe else MFU_SFT_TRAIN
    update_mfu = MFU_TRAIN_MOE if moe else MFU_TRAIN

    if n.is_opd:
        # opd step = on-policy student rollout (like grpo) + remote teacher scoring (concurrent
        # parasail round-trips, replaces reward grading) + policy update (fwd+bwd only, no local
        # reference forward - the teacher is the api). bill local compute on the full prompt+completion
        # sequence (see _opd_step_shape), not completion-only, or long-prompt opd is underquoted.
        completions, seq_tokens = _opd_step_shape(n)
        gen_s = (GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM * params * seq_tokens) / (peak * MFU_DECODE)
        update_s = (OPD_UPDATE_FLOPS_PER_TOKEN_PER_PARAM * params * seq_tokens) / (
            peak * update_mfu
        )
        teacher_lat = teacher_seconds_per_completion()
        request_multiplier = opd_teacher_request_multiplier(
            multi_turn=n.opd_multi_turn,
            max_turns=n.opd_max_turns,
        )
        score_items = completions * request_multiplier
        # conservative retry-and-turn wave policy: every potentially scored request consumes one slot,
        # including bounded no-signal replacement attempts and every bounded multi-turn assistant turn.
        # the worker and broker enforce the same shared concurrency ceiling.
        teacher_waves = math.ceil(score_items / OPD_TEACHER_SCORING_CONCURRENCY)
        teacher_s = teacher_waves * teacher_lat
        # opd samples on-policy and syncs weights to the rollout engine exactly as grpo does, so it
        # pays the same unmodelled per-step floor. it has no frozen-reference forward, but
        # old_log_prob and the weight sync are rollout properties, not grpo-specific ones.
        floor_s = step_floor_seconds(gpu)
        # the teacher is a remote api: its latency is identical on every card, so it is the part of an
        # opd step that a faster or more numerous gpu cannot shorten.
        return gen_s + update_s + floor_s, overhead + teacher_s

    if not n.is_grpo:
        flops = SFT_FLOPS_PER_TOKEN_PER_PARAM * params * (n.batch_size * n.seq_len)
        return flops / (peak * sft_mfu), overhead

    # GRPO step = rollout (G completions/prompt) + serial reward grading + policy/ref update.
    completions = n.batch_size * n.group_size
    # measured realized generation when a rollout profile exists, else max_completion_tokens. this
    # count feeds BOTH terms below, so quoting the cap multiplies its error through the two largest
    # parts of a grpo step.
    gen_tokens = completions * _completion_tokens(n)
    gen_s = (GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * MFU_DECODE)
    update_s = (GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * update_mfu)
    latency = reward_seconds_per_completion(n.reward_seconds_per_completion)
    # every completion is scored, one at a time (see the serial-scoring note above).
    reward_s = completions * latency
    # old_log_prob + weight sync + checkpointing: real gpu work with no flops term of its own.
    # gpu-bound, not fixed -- it is compute on this card, so a faster card shortens it and
    # sharding divides it, unlike reward grading which is a wait on off-gpu python.
    floor_s = step_floor_seconds(gpu)
    # reward grading runs off-gpu, so like the opd teacher it is fixed wall time no card choice
    # changes. a grpo step dominated by it is latency-bound, not compute-bound.
    return gen_s + update_s + floor_s, overhead + reward_s


def seconds_per_step(config: RunConfig, gpu: str) -> float:
    """Steady-state wall time for one optimizer step on ``gpu``."""
    gpu_bound, fixed = step_seconds_split(config, gpu)
    return gpu_bound + fixed


def step_cost_key(config: RunConfig):
    """``(gpu, hourly_rate) -> dollars per optimizer step``, or None if ``config`` can't be priced.

    The single ranking basis shared by every GPU-selection path (submit-time allocation, the
    parse-time provisional shown in the schema, and the cost estimate). Renting the cheapest card
    is not the same as running the job for the least money: a card bills for the time it takes, so
    the right basis is rate x duration. An A10 at $0.75/hr is nominally cheaper than an H100 at
    $3.29 but sustains a fraction of its FLOPs, so the same run costs about three times as much on
    it. Ranking per STEP prices both halves at once, and since the step count is identical across
    candidates it orders them exactly as total job cost does -- without needing the run's length.

    Returns None when the model is outside the cost catalog, so callers fall back to $/hr for
    EVERY candidate rather than ranking a mix of two incomparable bases.
    """
    try:
        step_seconds_split(config, "H100")  # probe: raises for a non-catalog model
    except Exception:
        return None

    def cost_key(gpu: str, hourly_usd: float) -> float:
        if gpu not in GPU_COMPUTE_TFLOPS:
            # no measured/spec throughput for this class, so its step time would be computed from a
            # placeholder. ranking on that invents a speed difference the hardware may not have;
            # return a constant instead, which leaves these classes ordered by the $/hr tie-break.
            return 0.0
        try:
            gpu_bound, fixed = step_seconds_split(config, gpu)
        except Exception:
            return 0.0  # unpriceable class falls back to the $/hr tie-break, never fails selection
        return hourly_usd * (gpu_bound + fixed) / 3600.0

    return cost_key


def sft_seconds_for_tokens(config: RunConfig, gpu: str, train_tokens: float) -> float:
    """SFT steady-state wall time for an actual token count on ``gpu``."""
    n = config.normalized()
    # MoE prices on total params at a reduced MFU (see seconds_per_step); dense keeps active.
    moe = _is_moe(n.model_id)
    params = (total_params_b(n.model_id) if moe else active_params_b(n.model_id)) * 1e9
    mfu = MFU_SFT_TRAIN_MOE if moe else MFU_SFT_TRAIN
    peak = effective_train_tflops(gpu) * 1e12
    flops = SFT_FLOPS_PER_TOKEN_PER_PARAM * params * train_tokens
    return flops / (peak * mfu)


def _offline_gpu_shape(
    config: RunConfig, *, max_wall_seconds: float = 0.0
) -> tuple[str, int, int, str, float]:
    """Offline structural quote: (gpu, need, count, provider, per-card rate).

    Quote preparation must not query live capacity: a sold-out market or transient lookup failure is
    exactly what the lifecycle retry machinery exists to survive, and consuming it here prevents the
    run/status from being created at all. Rank the managed 1/2/4/8-card shapes on the same cost model
    as allocation, then replace this provisional quote with the selected live candidate immediately
    before provisioning.
    """
    total_params_b(config.model_id)  # catalog-only; no HF/network sizing in `--cost`
    need = required_vram_gb(
        config.model_id,
        config.method,
        train=config.train_knobs(),
        thinking=config.thinking,
        model_revision=config.model_revision,
    )
    from flash.providers.base import (
        GPU_INFO,
        canonical_gpu,
        combined_vram_gb,
        providers_for,
        rentable_gpu_counts,
    )

    provider = config.provider if config.provider != "auto" else "auto"
    if config.gpu_type:
        names = (canonical_gpu(config.gpu_type),)
    else:
        names = tuple(
            info.name for info in GPU_INFO.values() if info.enum_member and info.validated
        )
    safe_gpu_count = geometry_safe_gpu_cap(
        config.model_id, config.gpu_count, model_revision=config.model_revision
    )
    ranked = []
    for gpu in names:
        info = GPU_INFO[gpu]
        if provider != "auto" and provider not in providers_for(gpu):
            continue
        for count in rentable_gpu_counts(safe_gpu_count):
            if combined_vram_gb(info.vram_gb, count) < need:
                continue
            # Provisional quoting is structural and must not touch a live market. Vast pricing is
            # offer-backed (therefore capacity-backed), and Lambda's catalog can blip too. Use the
            # provider's offline static rate here; the lifecycle replaces it from the selected candidate
            # before provisioning, so the persisted/charged quote still carries the exact live rate.
            if provider == "lambda":
                from flash.providers.lambdalabs.pricing import static_hourly_rate

                hourly = static_hourly_rate(gpu)
            else:
                hourly = info.hourly_usd
            gpu_bound, fixed = step_seconds_split(config, gpu)
            step_seconds = gpu_bound / multi_card_speedup(count, gpu) + fixed
            ranked.append(
                (
                    hourly * count * step_seconds,
                    count,
                    combined_vram_gb(info.vram_gb, count),
                    info.vram_gb,
                    gpu,
                    hourly,
                )
            )
    if not ranked:
        if config.gpu_type:
            info = GPU_INFO[canonical_gpu(config.gpu_type)]
            raise ValueError(
                f"exact GPU {info.name!r} cannot fit this run: it requires at least {need} GB"
            )
        shape = f" across up to {safe_gpu_count} cards" if safe_gpu_count > 1 else ""
        raise ValueError(f"no GPU class fits >= {need} GB{shape}")
    _cost, count, _combined, _per_card, gpu, hourly = min(ranked)
    return gpu, need, count, provider, hourly


def select_gpu(config: RunConfig, *, max_wall_seconds: float = 0.0) -> tuple[str, int]:
    """(chosen GPU class, required VRAM GB) from the offline structural quote."""
    gpu, need, _count, _provider, _hourly = _offline_gpu_shape(
        config, max_wall_seconds=max_wall_seconds
    )
    return gpu, need


def _offline_profile_shape(config: RunConfig) -> tuple[str, int, int, str, float]:
    """Offline structural quote for a cpu-only profile job: the cheapest rentable single card.

    Mirrors ``_offline_gpu_shape``'s offline-only rule, but not its ranking: the profile job's wall
    is a fixed cap, so no card finishes it sooner and rate alone decides.
    """
    from flash.providers.allocator import profile_required_vram_gb
    from flash.providers.base import GPU_INFO, canonical_gpu, providers_for

    need = profile_required_vram_gb()
    provider = config.provider if config.provider != "auto" else "auto"
    names = (
        (canonical_gpu(config.gpu_type),)
        if config.gpu_type
        else tuple(info.name for info in GPU_INFO.values() if info.enum_member and info.validated)
    )
    ranked = []
    for gpu in names:
        info = GPU_INFO[gpu]
        if provider != "auto" and provider not in providers_for(gpu):
            continue
        if provider == "lambda":
            from flash.providers.lambdalabs.pricing import static_hourly_rate

            hourly = static_hourly_rate(gpu)
        else:
            hourly = info.hourly_usd
        ranked.append((hourly, info.vram_gb, gpu))
    if not ranked:
        raise ValueError("no GPU class can host the workload profile job")
    hourly, _vram, gpu = min(ranked)
    return gpu, need, 1, provider, hourly


def _quote_shape(
    config: RunConfig, allocation, market_wall_s: float, *, profile: bool = False
) -> tuple[str, int, int, str, float]:
    """The (gpu, need, count, provider, per-card rate) a quote bills against.

    ``allocation`` is the exact live candidate the lifecycle selected; without one the shape is the
    offline structural pick, which must never touch a live market (see ``_offline_gpu_shape``).
    """
    if allocation is None:
        return (
            _offline_profile_shape(config)
            if profile
            else _offline_gpu_shape(config, max_wall_seconds=market_wall_s)
        )
    need = int(
        getattr(allocation, "min_vram_gb", 0)
        or required_vram_gb(
            config.model_id,
            config.method,
            train=config.train_knobs(),
            thinking=config.thinking,
            model_revision=config.model_revision,
        )
    )
    return (
        allocation.gpu,
        need,
        int(getattr(allocation, "gpu_count", 1) or 1),
        allocation.provider,
        float(allocation.hourly_usd),
    )


def estimate_profile_cost(config: RunConfig, *, allocation=None) -> CostEstimate:
    """Price a bounded workload-profile job from its wall cap, not from the workload it measures.

    A profile job exists to produce the exact workload evidence a training quote needs, so it cannot
    be priced through the training estimator without a circular dependency. It runs no optimizer
    steps and loads no model weights: the charge is the rented shape held for at most its wall cap,
    which is a ceiling the real run comes in under.
    """
    wall_s = max(60.0, float(config.max_wall_seconds or 0.0))
    gpu, need, billed_gpu_count, quote_provider, hourly = _quote_shape(
        config, allocation, wall_s, profile=True
    )
    return CostEstimate(
        model_id=config.model_id,
        method=config.method,
        steps=config.steps,
        gpu=gpu,
        provider=quote_provider,
        gpu_vram_gb=gpu_vram_gb(gpu),
        required_vram_gb=need,
        gpu_hourly_usd=hourly,
        setup_seconds=0.0,
        seconds_per_step=wall_s,
        train_seconds=wall_s,
        wall_clock_seconds=wall_s,
        wall_capped=True,
        gpu_count=billed_gpu_count,
        total_usd=wall_s / 3600.0 * hourly * billed_gpu_count,
        notes=(
            f"workload profile job: billed at most its {_fmt_duration(wall_s)} wall cap "
            "(no optimizer steps)",
        ),
    )


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
            f"@ {_describe_rollout_tokens(n)} + {teacher_name} teacher scoring "
            f"({tsec:.2f}s/request, {OPD_TEACHER_SCORING_CONCURRENCY} concurrent) + policy "
            "update (no local reference forward)"
        )
    elif n.is_grpo:
        comps = n.batch_size * n.group_size
        rsec = reward_seconds_per_completion(n.reward_seconds_per_completion)
        notes.append(
            f"GRPO step = vLLM rollout of {n.batch_size}x{n.group_size}={comps} completions "
            f"@ {_describe_rollout_tokens(n)} + reward ({rsec:.2f}s/completion"
            + (f", env {n.environment}" if n.environment else "")
            + ") + policy+reference update"
        )
    elif n.train_tokens is not None:
        profile_shape = (
            f" across {n.sft_packed_blocks:,} packed block(s)"
            if n.sft_packed_blocks is not None
            else ""
        )
        if n.supervised_train_tokens is not None and n.sft_packing_mode:
            notes.append(
                f"SFT priced on {n.train_tokens:,} exact compute tokens"
                f" ({n.supervised_train_tokens:,} supervised, {n.sft_packing_mode}{profile_shape})"
            )
        else:
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


def estimate_cost(
    config: RunConfig, *, wall_cap_s: float = DEFAULT_WALL_CAP_S, allocation=None
) -> CostEstimate:
    """Deterministic pre-flight cost calculation.

    ``allocation`` is the exact live candidate selected by the retrying lifecycle. Preparation omits
    it and stays offline; immediately before provisioning the persisted quote is replaced from this
    candidate so successful billing uses the real provider, class, count, and per-card rate.
    """
    # Billing cap: mirror the runner's max(60, max_wall_seconds) floor so a sub-60s cap isn't underpriced.
    cap_s = (
        max(60.0, float(config.max_wall_seconds))
        if config.max_wall_seconds is not None
        else wall_cap_s
    )
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
    if allocation is not None:
        gpu = allocation.gpu
        quote_provider = allocation.provider
        hourly = float(allocation.hourly_usd)
        need = int(
            getattr(allocation, "min_vram_gb", 0)
            or required_vram_gb(
                config.model_id,
                config.method,
                train=config.train_knobs(),
                thinking=config.thinking,
                model_revision=config.model_revision,
            )
        )
        billed_gpu_count = int(getattr(allocation, "gpu_count", 1) or 1)
    else:
        # Preparation and `flash train --cost` must stay independent of live capacity. A provider
        # lookup blip here would consume the first allocation failure before a run/status exists, so
        # the lifecycle could never retry it. This provisional structural quote is replaced from the
        # exact selected candidate immediately before provisioning.
        gpu, need, billed_gpu_count, quote_provider, hourly = _offline_gpu_shape(
            config, max_wall_seconds=market_wall_s
        )

    setup = setup_seconds(config)
    gpu_bound_s, fixed_s = step_seconds_split(config, gpu)
    speedup = multi_card_speedup(billed_gpu_count, gpu)
    sps = gpu_bound_s / speedup + fixed_s
    required_save_s = required_save_overhead_seconds(config)
    # A one-time kernel/graph compile is paid once on the first step (MoE-only; 0 for dense). It is
    # training GPU time, so it belongs in the (billed) train term, not setup. Required saves are also
    # synchronous fixed overhead; neither is divided by the number of cards.
    compile_s = compile_seconds(config, gpu)
    raw_train = compile_s + config.steps * sps + required_save_s
    if not config.is_grpo and config.train_tokens is not None:
        raw_train = (
            compile_s
            + sft_seconds_for_tokens(config, gpu, config.train_tokens) / speedup
            + required_save_s
        )
    sps = raw_train / config.steps

    # The cap is on total elapsed wall; setup is reported but not billed, so only training
    # contributes to total_usd.
    wall_capped = (setup + raw_train) > cap_s
    setup = min(setup, cap_s)
    train = max(0.0, cap_s - setup) if wall_capped else raw_train
    wall = setup + train

    # opd: add the external parasail teacher token spend. the teacher echo-scores every
    # sampled completion (input ~ prompt+completion per completion), so bill input tokens over the
    # effective (wall-capped) step count - not the uncapped `steps` - so a wall-capped run's teacher
    # bill tracks the GPU time it is actually billed for.
    teacher_api_usd = 0.0
    if config.is_opd:
        n = config.normalized()
        effective_steps = (train / sps) if sps > 0 else config.steps
        completions_per_step, tokens_per_step = _opd_step_shape(n)
        request_multiplier = opd_teacher_request_multiplier(
            multi_turn=n.opd_multi_turn,
            max_turns=n.opd_max_turns,
        )
        teacher_input_tokens = effective_steps * tokens_per_step * request_multiplier
        teacher_output_tokens = effective_steps * completions_per_step * request_multiplier
        teacher_api_usd = teacher_token_cost_usd(
            teacher_input_tokens,
            teacher_output_tokens,
            config.teacher_model,
        )

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
        gpu_count=billed_gpu_count,
        # total_usd is the customer gpu charge. the platform-owned teacher spend is itemized
        # only as a diagnostic and is not passed through to the customer. an n-card job occupies
        # n cards for the billed training wall, so the charge scales linearly with gpu_count.
        total_usd=train / 3600.0 * hourly * billed_gpu_count,
        teacher_api_usd=teacher_api_usd,
        notes=_notes(config, raw_train, wall_capped, cap_s),
    )
