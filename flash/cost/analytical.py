"""Estimate training cost from GPU time and hourly rate.

Elapsed wall clock includes non-billable setup. GRPO steps include rollout, reward grading, and
policy/reference updates.
"""

from __future__ import annotations

import math

from flash.opd_limits import OPD_TEACHER_SCORING_CONCURRENCY, opd_teacher_request_multiplier
from flash.providers.allocator import geometry_safe_gpu_cap, required_vram_gb, vram_headroom

from .facts import (
    GPU_COMPUTE_TFLOPS,
    _catalog_model_info,
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
# empirically fitted over 45 RunPod arms as fixed per-step overhead plus rollout-batch work.
# the intercept and per-completion slope cover old_log_prob, weight sync, checkpointing, and
# small-model overhead that peak-FLOPs scaling misses. this aggregate will drift with hardware,
# verl, and engine changes.
#
# this floor and the deleted per-completion reward wall are coupled: the wall's fictitious 1.0
# s/completion had been cancelling this missing overhead, so removing it WITHOUT this term scores
# 49.8x geometric bias. do not drop one without re-fitting the other.
STEP_FLOOR_BASE_SECONDS = 62.7
STEP_FLOOR_SECONDS_PER_COMPLETION = 0.805

# --- per-card offset ---------------------------------------------------------------------------
# empirically fitted intercept offsets capture measured card-level variance; the slope stays shared.
# offsets are shrunk by n/(n+2), require three arms, and use replicate-group holdouts.
# unmeasured cards use the pooled floor rather than a hardware guess.
STEP_FLOOR_MIN_ARMS_FOR_OFFSET = 3
STEP_FLOOR_CARD_OFFSET_SECONDS = {
    "A100 PCIe": -12.6,  # n=6, completion counts 32/256
    "B200": 42.5,  # n=4, 32/256 -- tied to H200, see below
    "H100": -10.4,  # n=24, 16/32/256
    "H200": 42.5,  # n=4, 32/256
    "RTX 4090": 14.8,  # n=10, 16/32/64/96
    "RTX 5090": -30.0,  # n=4, 32/96
}
# B200 and H200 share the slower offset as a conservative never-cheaper policy, despite matched A/B
# evidence that B200 is faster. untying changes the contract asserted by
# test_b200_not_cheaper_or_faster_than_h200_for_grpo, not just this fit. evidence:
# /home/azureuser/benchmark/b200-vs-h200-20260806/.


def step_floor_seconds(gpu: str, completions: int) -> float:
    """Per-step seconds the FLOPs terms do not account for.

    An unmeasured ``gpu`` gets the pooled floor, never a guess: the per-card offsets below are
    measured corrections for cards the campaign covered, not a model of hardware.
    """
    offset = STEP_FLOOR_CARD_OFFSET_SECONDS.get(gpu, 0.0)
    floor = STEP_FLOOR_BASE_SECONDS + STEP_FLOOR_SECONDS_PER_COMPLETION * max(0, completions)
    return max(0.0, floor + offset)


# --- how much of the floor a second card actually removes ---------------------------------------
# a single-card verl timing decomposition empirically assigns 79.9% to shardable old_log_prob.
# weight sync and checkpointing do not shrink with card count, so sharding the whole floor
# under-quotes wide runs. the multi-card behavior is mechanical, not fitted, and should be replaced
# by matched multi-card measurements.
STEP_FLOOR_SHARDED_FRACTION = 0.799


def _step_floor_seconds_for(config: RunConfig, gpu: str) -> float:
    """Step-floor seconds ``config`` pays on ``gpu``; 0 for sft, which runs no rollout."""
    n = config.normalized()
    if n.is_opd:
        completions, _seq_tokens = _opd_step_shape(n)
    elif n.is_grpo:
        completions = n.batch_size * n.group_size
    else:
        return 0.0
    return step_floor_seconds(gpu, completions)


# --- MoE (mixture-of-experts) per-step correction ----------------------------------------------
# measured MoE wall time scales with total params because routing and grouped GEMMs dominate at
# small batch. price MoE on total params with reduced MFU, per-step overhead, and one-time compile;
# dense models retain the active-param path.
MFU_SFT_TRAIN_MOE = 0.10  # MoE SFT fwd/bwd priced on total params
MFU_TRAIN_MOE = 0.09  # MoE GRPO/OPD policy update priced on total params
MOE_STEP_OVERHEAD_S = 2.0  # routing/dispatch/kernel-launch overhead an MoE pays every step
# One-time kernel/graph compile amortized into the first training step (fused Triton kernels;
# + vLLM cudagraph capture for rollout methods). MoE-only; the prior model omitted it entirely.
COMPILE_MOE_SFT_S = 35.0
COMPILE_MOE_ROLLOUT_S = 48.0  # GRPO / OPD (adds vLLM cudagraph capture)

# single-turn grpo scoring is serial under the global env lock in worker/rl_train.py, protecting
# non-thread-safe scorers. reward latency is therefore fixed wall time, not gpu work. multi-turn
# concurrency is unknown to this offline quote, so serial scoring is the conservative default.

# cold-start seconds are empirically calibrated from a fresh worker. model load dominates short
# jobs: MODEL_LOAD_BASE_S covers fixed deserialize/init work; download scales with checkpoint size.
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
    """Return measured prompt plus completion tokens, or the context cap.

    ``n.seq_len`` is capacity, not work. Prompt and completion must be measured together to avoid
    pricing a short completion against a full-context prompt.
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
    """Describe whether rollout length is measured or cap-based.

    The two can differ several-fold, so the note must not present a measured mean as the cap.
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
    """True when the model routes each token through a subset of experts (active < total params).

    Routing is curated architecture metadata, so this is a catalog read and takes no revision: a
    pinned commit of an entry does not change whether it is an MoE. (The uncataloged branch that
    compared active against total params is gone -- both of those raise for an unknown id now, so it
    could only ever have raised rather than answered.)
    """
    info = _catalog_model_info(model_id)
    return bool(info.active_params_b and info.active_params_b < info.params_b)


def compile_seconds(config: RunConfig, gpu: str) -> float:
    """One-time kernel/graph compile folded into the first training step. MoE-only; 0 for dense
    (whose original timing did not model it). Larger for rollout methods (add vLLM cudagraph
    capture)."""
    _ = gpu
    if not _is_moe(config.model_id):
        return 0.0
    return COMPILE_MOE_SFT_S if config.method == "sft" else COMPILE_MOE_ROLLOUT_S


# fsdp collective overhead prevents linear multi-card scaling. these empirical two-card efficiencies
# are split by interconnect so the allocator never credits pcie with nvlink scaling; they will drift
# with hardware and kernels.
MULTI_CARD_SCALING_NVLINK = 0.88
MULTI_CARD_SCALING_PCIE = 0.71


def multi_card_scaling(gpu: str, provider: str = "") -> float:
    """Realized fraction of linear scaling per added card of class ``gpu``."""
    return MULTI_CARD_SCALING_NVLINK if has_nvlink(gpu, provider) else MULTI_CARD_SCALING_PCIE


def multi_card_speedup(gpu_count: int, gpu: str, provider: str = "") -> float:
    """Return the multi-card throughput multiplier for gpu-bound work.

    The two-card measurements are extrapolated geometrically and clamped non-decreasing: additional
    cards may flatten scaling, but must not reduce aggregate throughput.
    """
    n = max(1, int(gpu_count))
    scaling = multi_card_scaling(gpu, provider)
    return max(k * (scaling ** (k - 1)) for k in range(1, n + 1))


# --- sft shards by SEQUENCE, not by data ---------------------------------------------------------
# sft_train.py pins Ulysses sequence parallelism, while grpo/opd use fsdp data parallelism; their
# collective costs are not interchangeable. no matched multi-card sft arm exists, so these separate
# constants conservatively reuse the dp values until an sft-specific measurement replaces them.
# verl reference: workers/engine/fsdp/transformer_impl.py get_data_parallel_size.
MULTI_CARD_SCALING_SP_NVLINK = 0.88
MULTI_CARD_SCALING_SP_PCIE = 0.71


def sequence_parallel_speedup(gpu_count: int, gpu: str, provider: str = "") -> float:
    """Return SFT sequence-parallel throughput across cards.

    Keep separate constants from fsdp data parallelism; see MULTI_CARD_SCALING_SP_NVLINK.
    """
    n = max(1, int(gpu_count))
    scaling = (
        MULTI_CARD_SCALING_SP_NVLINK if has_nvlink(gpu, provider) else MULTI_CARD_SCALING_SP_PCIE
    )
    return max(k * (scaling ** (k - 1)) for k in range(1, n + 1))


def method_card_speedup(config: RunConfig, gpu_count: int, gpu: str, provider: str = "") -> float:
    """Return throughput for the run's sequence- or data-parallel strategy.

    ``provider`` must reflect the rented substrate because interconnect changes scaling. Live
    allocation paths learn it after building ``config``; pinned ``config.provider`` is the fallback.
    """
    n = config.normalized()
    resolved = (provider or "").strip().lower()
    if not resolved:
        resolved = n.provider if n.provider != "auto" else ""
    if n.method == "sft":
        return sequence_parallel_speedup(gpu_count, gpu, resolved)
    return multi_card_speedup(gpu_count, gpu, resolved)


def sharded_step_seconds(config: RunConfig, gpu: str, gpu_count: int, provider: str = "") -> float:
    """Return one step's wall seconds on the selected multi-card shape.

    This is the only card-count division point. It splits the rollout floor so weight-copy and
    checkpoint phases remain unsharded, then applies method- and provider-specific speedup.
    """
    gpu_bound, fixed = step_seconds_split(config, gpu)
    speedup = method_card_speedup(config, gpu_count, gpu, provider)
    floor_s = _step_floor_seconds_for(config, gpu)
    # the floor is inside gpu_bound; split it so only the phases that actually shard get divided.
    shardable = gpu_bound - floor_s * (1.0 - STEP_FLOOR_SHARDED_FRACTION)
    unshardable = floor_s * (1.0 - STEP_FLOOR_SHARDED_FRACTION)
    return shardable / speedup + unshardable + fixed


def step_seconds_split(config: RunConfig, gpu: str) -> tuple[float, float]:
    """Return ``(gpu-bound, gpu-independent)`` seconds for one optimizer step.

    Remote scoring and reward grading stay fixed across cards; hardware ranking shards only the
    gpu-bound half. ``seconds_per_step`` is their sum.
    """
    n = config.normalized()
    peak = effective_train_tflops(gpu) * 1e12  # FLOP/s (realized training throughput; see facts)
    # An MoE's per-step wall scales with TOTAL params (routing + all-expert coordination + grouped
    # GEMM under-utilization), not the tiny active-param FLOPs; dense models keep active (== total).
    moe = _is_moe(n.model_id)
    params = (
        total_params_b(n.model_id, n.model_revision)
        if moe
        else active_params_b(n.model_id, n.model_revision)
    ) * 1e9
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
        # unvalidated extrapolation: the floor was fitted only on grpo. opd uses the same rollout
        # engine, weight sync, and checkpointing, but no matched opd campaign has confirmed these
        # constants; its larger floor share increases the uncertainty.
        floor_s = step_floor_seconds(gpu, completions)
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
    # old_log_prob, weight sync, and checkpointing are gpu work without a flops term. only the
    # STEP_FLOOR_SHARDED_FRACTION part shards; grpo/opd callers must use sharded_step_seconds().
    floor_s = step_floor_seconds(gpu, completions)
    # reward grading runs off-gpu, so like the opd teacher it is fixed wall time no card choice
    # changes. a grpo step dominated by it is latency-bound, not compute-bound.
    return gen_s + update_s + floor_s, overhead + reward_s


def seconds_per_step(config: RunConfig, gpu: str) -> float:
    """Steady-state wall time for one optimizer step on ``gpu``."""
    gpu_bound, fixed = step_seconds_split(config, gpu)
    return gpu_bound + fixed


def step_cost_key(config: RunConfig):
    """Return ``(gpu, hourly_rate) -> dollars per step``, or None if unpriceable.

    Every selection path ranks rate times duration, not hourly rate alone. Returning None makes all
    candidates fall back to $/hr instead of mixing incomparable bases.
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
    params = (
        total_params_b(n.model_id, n.model_revision)
        if moe
        else active_params_b(n.model_id, n.model_revision)
    ) * 1e9
    mfu = MFU_SFT_TRAIN_MOE if moe else MFU_SFT_TRAIN
    peak = effective_train_tflops(gpu) * 1e12
    flops = SFT_FLOPS_PER_TOKEN_PER_PARAM * params * train_tokens
    return flops / (peak * mfu)


def _offline_gpu_shape(
    config: RunConfig, *, max_wall_seconds: float = 0.0
) -> tuple[str, int, int, str, float]:
    """Return an offline structural GPU quote.

    Preparation must not consume live-capacity failures before run creation. Rank rentable shapes
    offline, then replace the quote with the selected live candidate before provisioning.
    """
    # Fail closed on a model that cannot be sized at all. Curated entries answer from the catalog
    # with no network call; a PINNED revision still resolves that commit's real geometry, so the
    # revision has to be passed or the check sizes a different set of weights than the worker loads.
    total_params_b(config.model_id, config.model_revision)
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
    elif provider == "auto":
        # `auto` ranks the RunPod pool: it is the default substrate, and the allocator only reaches a
        # lambda-only class after the cheaper runpod classes exhaust, so quoting one here would name
        # hardware auto would not have picked. `enum_member` IS the runpod membership flag.
        names = tuple(
            info.name for info in GPU_INFO.values() if info.enum_member and info.validated
        )
    else:
        # a pinned provider must rank that provider's whole validated pool. filtering on
        # `enum_member` silently means "on runpod", so lambda-only classes (A10, A100 SXM 40GB) never
        # entered a `provider=lambda` quote and the estimate - and the submit-time affordability
        # precheck - overstated cost against a cheaper shape `allocate()` would really pick. the
        # `providers_for` filter below narrows this pool to the classes the provider can provision.
        names = tuple(info.name for info in GPU_INFO.values() if info.validated)
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
            # `provider` is "auto" here only when the pool being ranked IS the runpod pool (see the
            # name selection above), so an empty provider never reaches the multiplier as a
            # substrate whose interconnect is unknown.
            step_seconds = sharded_step_seconds(
                config, gpu, count, "" if provider == "auto" else provider
            )
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
    """Price a workload profile from its wall cap.

    Pricing through the workload it exists to measure would be circular. It runs no optimizer steps;
    charge only the rented shape held up to the cap.
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
    """Calculate deterministic pre-flight cost.

    Preparation stays offline; a live ``allocation`` replaces the quote before provisioning so
    billing uses its provider, class, count, and rate.
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
    # sft shards by sequence and grpo/opd by data, so the multiplier is method-specific. the quote
    # provider is passed explicitly: with a live allocation it is the substrate actually selected,
    # which `config` does not carry -- an auto run reaches here with provider "auto" and would
    # otherwise be credited nvlink scaling on a vast combination that cannot deliver it.
    speedup = method_card_speedup(config, billed_gpu_count, gpu, quote_provider)
    sps = sharded_step_seconds(config, gpu, billed_gpu_count, quote_provider)
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
