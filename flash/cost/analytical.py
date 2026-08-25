"""Estimate training cost from GPU time and hourly rate.

Elapsed wall clock includes non-billable setup. Customer estimates exclude the identifiable SFT
framework-init block and add no GRPO reward-function latency term.
"""

from __future__ import annotations

import math

from flash.cost.facts import (
    GPU_COMPUTE_TFLOPS,
    _catalog_model_info,
    active_params_b,
    download_weight_gb,
    effective_train_tflops,
    gpu_vram_gb,
    has_nvlink,
    model_quant,
    teacher_seconds_per_completion,
    teacher_token_cost_usd,
    total_params_b,
)
from flash.cost.types import CostEstimate, RunConfig
from flash.engine.plan.steps import rl_data_parallel_cards, sft_data_parallel_cards
from flash.providers.core.allocator import geometry_safe_gpu_cap, required_vram_gb, vram_headroom
from flash.teacher.limits import OPD_TEACHER_SCORING_CONCURRENCY, opd_teacher_request_multiplier

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
# every calibration arm reported timing_s/reward=0.0000, so this fitted floor contains no
# measurable reward-function wait. it covers only recurring gpu-side work and small-model
# inefficiency that the peak-flops terms miss. reward latency therefore has no estimate term.
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

# --- sft recurring per-step floor ---------------------------------------------------------------
# the measured sft train wall includes a one-time verl/model/lora/fsdp startup block and recurring
# per-step publish/sync work. customer policy excludes the one-time framework setup from the quote,
# while the recurring per-step work remains billable training. setup_seconds reports cold start and
# model initialization separately for observability and never contributes to total_usd.
SFT_STEP_FLOOR_SECONDS = 0.8


def sft_step_floor_seconds(steps: int) -> float:
    """Recurring sft training seconds outside the flops term."""
    return SFT_STEP_FLOOR_SECONDS * max(0, steps)


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
    return completions, completions * float(n.seq_len)


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


def method_card_speedup(config: RunConfig, gpu_count: int, gpu: str, provider: str = "") -> float:
    """Return multi-card throughput for the run.

    Every algorithm now shards by DATA. SFT used to pin Ulysses sequence parallelism and therefore
    carried its own scaling constants, but sequence parallelism is incorrect for the catalog's GDN
    hybrids (the linear-attention recurrence and causal conv carry state along the sequence, and
    verl passes no state across ranks), so ``sft_train_runner`` pins it off. One fsdp constant now
    describes all three -- and the sp pair it replaces was numerically identical to it, so no quote
    moves. See ``sft_data_parallel_cards``.

    ``provider`` must reflect the rented substrate because interconnect changes scaling. Live
    allocation paths learn it after building ``config``; pinned ``config.provider`` is the fallback.

    Credit SFT only the ranks that will actually execute. ``gpu_count`` here is the BILLED shape,
    and sharding by data bounds the executed width by BOTH the batch and the row count: an unpacked
    profile pins ``batch_size`` to 1, so a 2-card rental trains on one rank, and a batch-compatible
    width that does not divide the rows is narrowed again so the sampler cannot drop the remainder.
    Quoting the billed width would promise throughput the run cannot deliver and understate wall
    time against the run's cap. The cards are still billed -- that is the point of the
    ``[sft][warn]`` line the worker prints.

    ``sft_retained_examples`` is the rows the trainer iterates. It must be carried explicitly
    rather than derived from ``sft_packed_blocks``, which is ``ceil(rows / examples_per_update)``
    and reconstructs 10 rows at a batch of 8 as 16 -- an over-credit, i.e. the failure this clamp
    exists to prevent.
    """
    n = config.normalized()
    resolved = (provider or "").strip().lower()
    if not resolved:
        resolved = n.provider if n.provider != "auto" else ""
    return multi_card_speedup(executed_gpu_count(config, gpu_count), gpu, resolved)


def executed_gpu_count(config: RunConfig, gpu_count: int) -> int:
    """Ranks this run launches on ``gpu_count`` cards, which a small batch can bound below it.

    THE definition of "how wide does this actually run", shared by the throughput model above and
    the offline shape search below. They must not answer it separately: the quote reporting a shape
    the allocator then rejects tells a user a run is feasible and priced, and then refuses it at
    submit. Every algorithm shards by data, so every width is bounded by the work one step holds --
    rows for sft, sequences (prompts times group) for grpo and opd. Mirrors
    ``allocator._executed_gpu_count``; the two are one rule stated on each side of the quote.
    """
    n = config.normalized()
    if n.method == "sft":
        return sft_data_parallel_cards(gpu_count, n.batch_size or 1, n.sft_retained_examples or 0)
    prompts = int(n.batch_size or 0)
    if prompts <= 0:
        # unknown batch does not narrow: see `_executed_rl_gpu_count`.
        return gpu_count
    return rl_data_parallel_cards(gpu_count, prompts * int(n.group_size or 1))


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

    Remote OPD scoring stays fixed across cards; hardware ranking shards only the gpu-bound half.
    ``seconds_per_step`` is their sum.
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

    # grpo step = rollout (g completions/prompt) + policy/reference update. reward-function
    # latency is deliberately absent: all floor-calibration arms measured timing_s/reward=0.
    completions = n.batch_size * n.group_size
    gen_tokens = completions * float(n.completion_len)
    gen_s = (GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * MFU_DECODE)
    update_s = (GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * update_mfu)
    # old_log_prob, weight sync, checkpointing, and small-model inefficiency are covered by the
    # fitted floor. reward latency is not. only STEP_FLOOR_SHARDED_FRACTION shards; callers must
    # use sharded_step_seconds().
    floor_s = step_floor_seconds(gpu, completions)
    return gen_s + update_s + floor_s, overhead


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


def _wider_shape_remedy(config: RunConfig, need: float, names: tuple[str, ...]) -> str:
    """The `--gpus N` clause this quote's fit failure carries; see ``base.wider_shape_remedy``.

    ``names`` is the pool the quote already ranked, so the remedy is searched over exactly the
    classes that were considered -- reusing the caller's provider filtering instead of
    reconstructing it here and risking a suggestion for a class it never had.

    A class is dropped when every provider that would serve it here names the card count in the
    SKU, since this path is offline by contract and cannot confirm such a shape is sold. Quoting an
    unverifiable width is worse than a bare dead end: `--cost` is consulted precisely to avoid a
    doomed launch. A pinned provider narrows that question to itself -- H100 is on RunPod, but a
    lambda-pinned quote may not borrow RunPod's freedom to rent any count.
    """
    from flash.providers.core.base import (
        GPU_INFO,
        providers_for,
        wider_shape_remedy,
    )
    from flash.providers.core.fit_errors import rents_arbitrary_card_counts
    from flash.providers.core.sharding import MAX_COMBINATION_CARDS

    def _in_play(gpu: str) -> tuple[str, ...]:
        carriers = providers_for(gpu)
        if config.provider == "auto":
            return carriers
        return tuple(name for name in carriers if name == config.provider)

    # the authored ceiling limited the ranking above; the geometry cap at the MAXIMUM rentable
    # width is what bounds a suggestion.
    return wider_shape_remedy(
        (GPU_INFO[gpu].vram_gb for gpu in names if rents_arbitrary_card_counts(_in_play(gpu))),
        need,
        ceiling=geometry_safe_gpu_cap(
            config.model_id, MAX_COMBINATION_CARDS, model_revision=config.model_revision
        ),
        # `gpu_count` is now optional: none means the author never named a width, so no count has
        # been "already tried" and the search must exclude nothing. 0 is that empty exclusion --
        # passing none compares int > none and crashes the quote.
        above=config.gpu_count or 0,
        # the same width rule the ranking loop above rejected shapes with, so the remedy cannot
        # promise a count the retry will not launch on.
        executed_width=lambda count: executed_gpu_count(config, count),
    )


def _catalog_check_remedy(config: RunConfig, need: float, names: tuple[str, ...]) -> str:
    """The width to ASK a fixed-count provider for, when no width can be promised offline.

    ``_wider_shape_remedy`` drops classes whose providers name the count in the SKU, which leaves a
    Lambda- or Vast-pinned exact quote with no remedy at all -- so it fell through to knob advice
    telling the user to shrink a run that already fits at a wider count. `live_capacity` means the
    count is confirmed dynamically, not that the SKU is absent: Lambda resolves `gpu_4x_h100_pcie`
    against its own catalog. This mirrors the allocator's ``_catalog_check_hint`` so the same
    shortfall reads the same whether it surfaced from `--cost` or from submit.

    Still a check and never a promise: nothing offline proved the wider SKU is purchasable.

    Withheld when the run would not LAUNCH on the width found, mirroring ``_catalog_check_hint``:
    ``smallest_fitting_gpu_count`` credits rented cards, so for an sft run the batch caps at fewer
    ranks it names a count that buys idle cards. The mirror has to hold in both directions or
    `--cost` promises a width submit rejects.
    """
    from flash.providers.core.base import smallest_fitting_gpu_count
    from flash.providers.core.sharding import MAX_COMBINATION_CARDS

    width = smallest_fitting_gpu_count(
        need,
        max_gpu_count=geometry_safe_gpu_cap(
            config.model_id, MAX_COMBINATION_CARDS, model_revision=config.model_revision
        ),
        gpu_names=names,
        executed_width=lambda count: executed_gpu_count(config, count),
    )
    if width is None or width <= (config.gpu_count or 0):
        return ""
    pinned = names[0] if len(names) == 1 else "multi-card"
    return (
        f". Their catalog may list a {width}-card {pinned} instance -- raise the card ceiling "
        f"with `--gpus {width}` to check it against their catalog"
    )


def _quote_gpu_ceiling(
    config: RunConfig, need: float, names: tuple[str, ...], *, ceiling: int | None, auto_cap: int
) -> int:
    """The widest count this quote ranks over: the authored ceiling, or the smallest that fits.

    An authored ceiling is the user's own `[gpu] count`, narrowed only by the model's geometry cap.
    Auto-sizing instead searches for the smallest fitting count, with the SAME executed-width rule
    the ranking loop applies -- without it the ceiling is chosen on rented cards and can land below
    the shape that actually fits, because sft's executed width is not monotonic in the rented count
    (batch 3 over 3 rows launches 1 rank on 2 cards but 3 on 4). A ceiling of 2 would then hide the
    4-card shape and the quote would reject a job submit accepts.

    Falls back to ``auto_cap`` when nothing fits, so the caller reports the shortfall against the
    widest shape rather than silently ranking a narrow one.
    """
    if ceiling is not None:
        return geometry_safe_gpu_cap(config.model_id, ceiling, model_revision=config.model_revision)
    from flash.providers.core.base import smallest_fitting_gpu_count

    return (
        smallest_fitting_gpu_count(
            need,
            max_gpu_count=auto_cap,
            gpu_names=names,
            executed_width=lambda count: executed_gpu_count(config, count),
        )
        or auto_cap
    )


def _offline_preferred_gpu_shape(config: RunConfig) -> tuple[str, int, int, str, float]:
    """quote the first structurally usable preference, then cost-rank unnamed fallbacks."""
    from dataclasses import replace

    from flash.providers.core.registry import PROVIDER_NAMES, available_providers

    # quote only what this plane can actually rent. `allocate()` starts from the configured set, so
    # a preference naming a provider this plane cannot provision is ignored there -- quoting it
    # anyway prices a shape the run will never get, and the affordability check runs on that
    # estimate, so a balance sufficient for the real allocation can be refused.
    configured = available_providers()
    # an unconfigured plane (no credentials anywhere) has nothing to filter against; fall back to the
    # registered set so the quote keeps its historical structural answer instead of going empty.
    eligible = configured or PROVIDER_NAMES
    for provider in config.providers:
        if provider not in eligible:
            continue
        try:
            return _offline_gpu_shape(replace(config, provider=provider, providers=()))
        except ValueError:
            # a soft preference that cannot carry this shape contributes no candidate. keep walking
            # instead of turning its authored position into a hard pin.
            continue
    unnamed = tuple(name for name in eligible if name not in config.providers)
    fallback_quotes = []
    for provider in unnamed:
        try:
            fallback_quotes.append(
                _offline_gpu_shape(replace(config, provider=provider, providers=()))
            )
        except ValueError:
            continue
    if fallback_quotes:
        return min(
            fallback_quotes,
            key=lambda quote: (
                quote[4] * quote[2] * sharded_step_seconds(config, quote[0], quote[2], quote[3])
            ),
        )
    # every eligible provider has now been tried and none has a structural fit. dropping the
    # restriction here to reach the unpinned `provider="auto"` diagnostic would rank the registered
    # runpod pool regardless of credentials, quoting a shape this plane cannot rent -- the same
    # defect the eligibility filter above exists to prevent, reintroduced at the last step. that
    # quote would then pass affordability and only fail once live allocation runs, after the run is
    # recorded. re-raise the eligible set's own failure instead, which names the real constraint.
    return _offline_gpu_shape(replace(config, provider=eligible[0], providers=()))


def _offline_gpu_shape(config: RunConfig) -> tuple[str, int, int, str, float]:
    """Return an offline structural GPU quote.

    Preparation must not consume live-capacity failures before run creation. Rank rentable shapes
    offline, then replace the quote with the selected live candidate before provisioning.
    """
    if config.providers:
        return _offline_preferred_gpu_shape(config)
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
    from flash.providers.core.base import (
        GPU_INFO,
        authored_gpu_ceiling,
        canonical_gpu,
        providers_for,
        rentable_gpu_counts,
    )
    from flash.providers.core.sharding import (
        MAX_COMBINATION_CARDS,
        combined_vram_gb,
    )

    provider = config.provider if config.provider != "auto" else "auto"
    if config.gpu_type:
        # rank every class the author declared acceptable, not just the head. `allocate()` cost-ranks
        # the whole ordered set, so quoting the head alone prices a shape the run may never be given
        # -- an authored ["B200", "H100"] quotes 3x the H100 the allocator would actually rent, and
        # the submit-time affordability precheck refuses an affordable run on that inflated number.
        names = tuple(
            dict.fromkeys(
                canonical_gpu(name) for name in (config.gpu_type, *config.gpu_type_fallbacks)
            )
        )
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
    # narrow to what the pinned provider can actually provision BEFORE sizing. the ranking loop
    # below filters per candidate, which is too late for three decisions taken up front: the
    # auto-sized count, the no-fit message, and the `--gpus` remedy would all reason over classes
    # this provider cannot rent. measured: a vast-pinned 119 GB run sized 1 card against another
    # provider's H200, ranked empty, and reported "more than any 8-card combination (1177.6 GB
    # max)" -- a number larger than the requirement it claimed could not be met, while 2x80 GB vast
    # cards would have fit.
    if provider != "auto":
        names = tuple(name for name in names if provider in providers_for(name))
    auto_cap = geometry_safe_gpu_cap(
        config.model_id, MAX_COMBINATION_CARDS, model_revision=config.model_revision
    )
    ceiling = authored_gpu_ceiling(config.gpu_type, config.gpu_count)
    safe_gpu_count = _quote_gpu_ceiling(config, need, names, ceiling=ceiling, auto_cap=auto_cap)
    ranked = []
    for gpu in names:
        info = GPU_INFO[gpu]
        for count in rentable_gpu_counts(safe_gpu_count):
            # credit only the cards that JOIN the run, matching the allocator's `_fits`. quoting the
            # billed count here made the two disagree: a 27B at 128k over 10 rows was quoted 4x H200
            # (460 GB credited against a 422 GB need) while the allocator launches 2 ranks -- 234 GB
            # -- and rejects it, so the run was priced as feasible and then refused at submit.
            launched = executed_gpu_count(config, count)
            if combined_vram_gb(info.vram_gb, launched) < need:
                continue
            # customer quoting is structural and must not touch a live market. vast pricing is
            # offer-backed (therefore capacity-backed), and lambda's catalog can blip too. use the
            # provider's offline static rate here; allocation may later select a different live rate,
            # but that operational price never changes the accepted customer quote.
            if provider == "lambda":
                from flash.providers.lambda_.client.pricing import static_hourly_rate

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
                    combined_vram_gb(info.vram_gb, launched),
                    info.vram_gb,
                    gpu,
                    hourly,
                )
            )
    if not ranked:
        _raise_no_fitting_shape(
            config,
            need,
            names,
            provider=provider,
            ceiling=ceiling,
            safe_gpu_count=safe_gpu_count,
            auto_cap=auto_cap,
        )
    _cost, count, _combined, _per_card, gpu, hourly = min(ranked)
    return gpu, need, count, provider, hourly


def _raise_no_fitting_shape(
    config: RunConfig,
    need: float,
    names: tuple[str, ...],
    *,
    provider: str,
    ceiling: int | None,
    safe_gpu_count: int,
    auto_cap: int,
) -> None:
    """Reject a run no rentable shape fits, naming what the quote was allowed to rank."""
    from flash.providers.core.base import GPU_INFO, canonical_gpu
    from flash.providers.core.fit_errors import vram_fit_error_message, vram_knob_advice

    # a pinned class is blocked by the class itself, so name it -- the pool-wide message would
    # report the widest validated shape, which is not hardware this quote was ever allowed to
    # use. `_wider_shape_remedy` searches only `names`, already narrowed to the pin, so the
    # `--gpus N` clause it appends can never name a shape the pin forbids. an unpinned run
    # falls through to the pool-wide message, which reports the count that would fit.
    remedy = _wider_shape_remedy(config, need, names)
    if config.gpu_type:
        # name every acceptable class, not just the head: with fallbacks authored, reporting the
        # head alone tells the user to fix a class that may not be the one that failed, and
        # hides that the whole declared set was ranked and rejected.
        declared = tuple(
            GPU_INFO[canonical_gpu(name)].name
            for name in (config.gpu_type, *config.gpu_type_fallbacks)
        )
        label = (
            repr(declared[0])
            if len(declared) == 1
            else "none of " + ", ".join(repr(name) for name in declared)
        )
        raise ValueError(
            f"exact GPU {label} cannot fit this run: it requires at least {need} GB"
            + (
                remedy
                or _catalog_check_remedy(config, need, names)
                or f". {vram_knob_advice(config.method).capitalize()}."
            )
        )
    raise ValueError(
        vram_fit_error_message(
            config.method,
            need,
            requested_gpu_count=ceiling,
            effective_gpu_count=safe_gpu_count,
            max_gpu_count=auto_cap,
            gpu_names=names,
            providers=None if provider == "auto" else (provider,),
            # same rule the ranking loop rejected shapes with, so the advice cannot name a
            # width the retry will not launch on.
            executed_width=lambda count: executed_gpu_count(config, count),
            # an offline quote does not know the configured fleet, so it cannot claim that
            # dropping a provider pin would make a wider shape purchasable.
            widenable_without_pin=None,
        )
    )


def _allocation_quote_shape(config: RunConfig, allocation) -> tuple[str, int, int, str, float]:
    """The (gpu, need, count, provider, per-card rate) billed against a SELECTED live candidate.

    Both quote paths bill an allocation the same way; they differ only in the offline shape they
    fall back to when the lifecycle has not selected one yet, so that choice stays with the caller.
    """
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
        from flash.engine.plan.recipe import resolve_teacher

        teacher_name = resolve_teacher(n.teacher_model).display_name
        notes.append(
            f"opd step = student rollout of {n.batch_size}x{n.group_size}={comps} completions "
            f"@ {n.completion_len} tok + {teacher_name} teacher scoring "
            f"({tsec:.2f}s/request, {OPD_TEACHER_SCORING_CONCURRENCY} concurrent) + policy "
            "update (no local reference forward)"
        )
    elif n.is_grpo:
        comps = n.batch_size * n.group_size
        notes.append(
            f"GRPO step = vLLM rollout of {n.batch_size}x{n.group_size}={comps} completions "
            f"@ {n.completion_len} tok + policy+reference update"
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
    if allocation is not None:
        gpu, need, billed_gpu_count, quote_provider, hourly = _allocation_quote_shape(
            config, allocation
        )
    else:
        # Preparation and `flash train --cost` must stay independent of live capacity. A provider
        # lookup blip here would consume the first allocation failure before a run/status exists, so
        # the lifecycle could never retry it. This provisional structural quote is replaced from the
        # exact selected candidate immediately before provisioning.
        gpu, need, billed_gpu_count, quote_provider, hourly = _offline_gpu_shape(config)

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
    # setup_seconds is excluded for every method. the separately identifiable sft framework-init
    # block is excluded here too; grpo/opd retain their aggregate recurring-work floor because its
    # calibration cannot safely split synchronization, checkpointing, and any in-train init.
    sft_floor_s = sft_step_floor_seconds(config.steps) if config.method == "sft" else 0.0
    raw_train = compile_s + config.steps * sps + required_save_s + sft_floor_s
    if not config.is_grpo and config.train_tokens is not None:
        raw_train = (
            compile_s
            + sft_seconds_for_tokens(config, gpu, config.train_tokens) / speedup
            + required_save_s
            + sft_floor_s
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
        # the width the run LAUNCHES on, which a small batch can bound below the billed count.
        # billing follows the rented cards; the vram a shape offers follows the ranks that join.
        executed_gpu_count=executed_gpu_count(config, billed_gpu_count),
        # total_usd is the customer gpu charge. the platform-owned teacher spend is itemized
        # only as a diagnostic and is not passed through to the customer. an n-card job occupies
        # n cards for the billed training wall, so the charge scales linearly with gpu_count.
        total_usd=train / 3600.0 * hourly * billed_gpu_count,
        teacher_api_usd=teacher_api_usd,
        notes=_notes(config, raw_train, wall_capped, cap_s),
    )
