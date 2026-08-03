"""The analytical cost model: total = training-only GPU hours x GPU $/hr.

Elapsed wall clock still includes cold-start setup + steps x per-step time, but setup/cold-start
is reported as non-billable. GRPO splits each step into a vLLM rollout + reward grading +
policy/reference update.
"""

from __future__ import annotations

import math

from flash.providers.allocator import required_vram_gb, vram_headroom

from .facts import (
    GPU_COMPUTE_TFLOPS,
    active_params_b,
    download_weight_gb,
    effective_train_tflops,
    gpu_hourly_usd,
    gpu_vram_gb,
    has_nvlink,
    model_quant,
    pick_gpu,
    reward_seconds_per_completion,
    rollout_is_resident,
    rollout_step_seconds,
    run_block_seconds,
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

# mirrors _TEXT_TEACHER_BATCH_SIZE in flash/engine/worker/opd_train.py. the cost model must not
# import the worker (it has to price a run without the training stack installed), so the value is
# duplicated here and pinned by a test that reads the worker's constant and asserts they agree.
OPD_TEACHER_BATCH_SIZE = 8

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

# SFT-ONLY run-level startup, paid once inside the billed train wall: verl launch, model load, lora
# wrap and fsdp init all land after setup_seconds is stamped (lifecycle.py:1203, before the training
# subprocess starts) and before the first optimizer step. sft quoted this at zero, so it underquoted
# EVERY sft arm: shipped/realized median 0.46x, 14 of 16 arms below 0.8x.
#
# MEASURED as the intercept of train_wall regressed on step count, with card, model, batch and
# sequence shape held fixed. the raw pooled fit is not identifiable at this sample size (replicates
# at one step count spread 39-61%), but the spread is not noise: it is BIMODAL, and pod class --
# read off the heartbeat's driver version, independent of anything fitted -- separates it. within a
# class the spread collapses to 2-3s, and each class puts the intercept well above zero:
#   4090 drv 580.126.20  0.868 s/step + 68.4s   (n=5, within-class spread 3.0s at 32 steps)
#   4090 drv 570.195.03  1.173 s/step + 105.3s  (n=4, spread 2.4s)
#   4090 drv 570.172.08  1.648 s/step + 123.0s  (n=2, SAME physical gpu -> zero pod variance)
#   H100 (own fit, no part in choosing this value)      + 74.4s
# the class model beat the pooled fit on 8 of 9 arms, INCLUDING replicates launched after the
# hypothesis was formed, so this is a holdout result rather than a post-hoc rationalization.
#
# the value is the smallest per-class intercept, not the mean: the quote cannot see the pod class,
# so the floor is the only figure justified in EVERY class. this still underquotes the slow classes
# -- deliberately, since overquoting a class the user might not land on is the worse error.
#
# grpo and opd get their own run-level block (facts.run_block_seconds) rather than this one, which
# is fitted on sft arms. an earlier revision of this comment excluded them outright, reasoning that
# their per-step floor "already prices this" so a run-level term would double-count. that reasoning
# was wrong, and step sweeps falsified it: the per-step floor IS this block, divided by the step
# count of the arms it was fitted on. the double-count is avoided by moving the block out of the
# per-step term, not by refusing to model it -- see facts._RUN_BLOCK_S.
# the flat block previously here is falsified by realized sft arms, and so is the bare flops term next to it.
# scored fit-free -- each config bounded by the fastest run of that same config, monotone-enveloped
# in tokens -- the shipped pair quotes 13 of 13 configs BELOW a run that actually happened, at geo
# 0.401x. a whole-run quote cannot sit under a realized instance of that whole run, so this needs no
# regression to reject and no step leverage to believe.
#
# WHICH mechanism is at fault was nearly unidentifiable: every sft arm in the corpus runs the same
# (batch_size=8, seq_len=1024) shape, and at one shape tokens = steps x 8192 exactly, so a
# per-token rate error and a per-step overhead are the same regressor. they make opposite
# predictions only when the shape changes -- 4x the batch costs 4x the step under a rate, but
# AMORTIZES a per-step overhead 4x.
#
# the discriminator is the batch-32 cold-start anchor below (test_cold_start_calibrated_to_real_
# short_sft_run): a real 0.8B/26-step/RTX 4090 run whose train wall came in under its 449.5 s setup.
# extrapolating the 4090 group's measured 6.1x rate error to batch 32 predicts 762 s and BLOWS that
# bound; a per-step overhead predicts 333 s and holds it. so the missing term is per-step, and the
# earlier rate-multiplier form is falsified rather than merely unsupported.
#
# that is the same mechanism the MoE path already prices (MOE_STEP_OVERHEAD_S, "routing/dispatch/
# kernel-launch overhead an MoE pays every step"). dense sft simply had none, so its step was billed
# as pure matmul while dataloading, packing, the optimizer and fsdp collectives all sit in the
# realized wall. the two stack: an MoE sft step pays both.
#
# every coefficient here is MEASURED, not grid-searched. within a (card, model) group the card and
# params are fixed, so the slope of realized wall vs steps IS the per-step cost and the intercept IS
# the block:
#   RTX 4090 / 0.8B   block  81.7 s   7.30 s/step measured
#   H100     / 4B     block 178.6 s   9.93 s/step measured
# K and the exponent solve exactly from those two blocks, which are y-intercepts of wall vs steps and
# so do not depend on how the slope is modelled.
#
# the SLOPE is the part that needed care. charging the shortfall as a flat per-step overhead fits
# both groups, but only by carrying two constants (6.23 and 8.08 s/step) with no account of why they
# differ -- card and model change TOGETHER between the two groups, so "card-invariant" was assumed,
# never measured. and a card-invariant term is not harmless here: at 7 s/step it makes over half a 4B
# step stop responding to card speed, which flips the A10 ahead of a faster RTX 4090 on job cost.
# breaking card ranking is the exact failure this module exists to prevent, so that form is rejected.
#
# what actually happens physically is that a step this small cannot fill the GPU. below a saturation
# size the step is launch- and occupancy-bound, so it costs what SFT_SATURATION_TOKENS would cost
# rather than what it carries. that is ONE constant, and it is over-determined: solved independently
# from each group it gives 55,764 and 44,016 tokens, a 1.27x spread across a 5x param and 3x card
# span. it also reproduces the flat form's one real success -- under saturation the step time is
# constant in batch size, which is why it satisfies the batch-32 anchor below -- while remaining
# proportional to params and INVERSELY proportional to card speed, so ranking survives.
#
# the anchor is a real 0.8B/26-step/RTX 4090 run at batch 32 whose train wall came in under its
# 449.5 s setup (test_cold_start_calibrated_to_real_short_sft_run). every sft arm in the corpus that
# solved this constant ran one (batch_size=8, seq_len=1024) shape, and under the OLD basis that
# billed the context capacity, such a step was billed at exactly 8192 tokens.
#
# that is where 8192 comes from, and it is a derivation rather than a fit: the constant is solved
# against a corpus whose every member has the same billed step size, so the saturation point cannot
# come out anywhere except that size. the previous 49152 was solved the same way but against the
# capacity-inflated basis (batch x max_context_tokens), which is why it landed 6x higher. once
# _sft_step_shape bills realized tokens (~1024 for these arms), a 49152 floor multiplies every small
# step by 48x, and the whole-run envelope check went to geo 3.093x / spread 6.01x with 0 of 12
# configs quoted below a run that actually happened. the floor and the capacity basis were
# COMPENSATING ERRORS: the inflated basis sat above the floor and hid it.
#
# scored on that same fit-free lower envelope, 8192 is the only value that passes: geo 1.133x,
# spread 1.80x, 3 of 12 under 1.0 (49152 -> 3.093x/6.01x/0; 16384 -> 1.564x/2.51x/1;
# 4096 -> 0.895x/2.09x/8). the batch-32 anchor holds either way and holds on real data: a measured
# batch-32 sweep on this exact card ran 6.695 s/step (n=3 cell mean, raw wall/steps), so 26 steps is
# 174 s against 449.5 s of setup. the 762 s rate extrapolation the 49152 floor was introduced to
# rescue was 4.4x higher than what the hardware actually did.
#
# NOT fitted from the batch sweep: a 9-arm matched replicate sweep on this card measured within-cell
# spread W = 1.71x against a between-cell batch effect B = 1.69x, so W >= B and the batch dimension
# of the step is not resolvable with that design (a preregistered rule, fixed before the data
# landed). a second 9-arm replicate sweep reproduced that verdict: W = 2.120x against B = 1.900x.
# this constant therefore stays a single saturation size and is deliberately NOT given a
# batch-dependent form. the batch-32 cell does run slower per step than batch 8 (6.526 vs 3.435 s/step
# cell means), but with W >= B twice over, that gap is not separable from pod variance on this design.
# resolving it needs ~n=12 per cell.
#
# scored fit-free on the monotone lower envelope (each config bounded by the fastest run at >= its
# own token count), adding this block took the model from geo 0.401x / spread 4.71x / 13 of 13
# configs quoted BELOW a run that actually happened, to geo 1.033x / spread 1.90x at the time it was
# fitted. those two figures are the block's own before/after and were measured under the capacity
# basis; the current envelope for the model as a whole is geo 1.198x / spread 1.74x (see
# SFT_SATURATION_TOKENS). spread is the figure that matters for a selection model -- a uniform bias
# cancels in a ranking, variation across configs does not.
#
# the envelope's target is NOT 1.0x. its bound is the MINIMUM realized wall of a config while the
# model quotes an EXPECTED one, and across 20 sft configs with >= 2 matched observations the measured
# mean/min ratio is geo 1.174x. a correctly-centred model therefore scores ~1.17x there by
# construction, so 1.198x is on target rather than a 20% over-quote.
#
# this block is measured at 104.7s on the canonical cell against the 81.7s the constants below
# produce (0.78x). it is left as fitted: the one other sft cell with step leverage (H100 / 4B, 5 runs
# over a 128x span) fits a 322.2s block against the model's 178.6s, so both measured cells say the
# block is if anything understated, and correcting it moves every ranking crossover LATER, never
# earlier. refitting it needs per-card leverage the corpus does not yet have on more than two cells.
#
# reusing the already-calibrated per-card rollout block (facts.run_block_seconds) was tried first,
# since it would have added no new constant at all. it overshoots (geo 1.855x, worst 4.74x) because
# that block prices a vLLM engine build and first weight sync, which an sft run never performs.
SFT_RUN_STARTUP_K = 85.9
SFT_RUN_STARTUP_PARAMS_EXP = 0.473

# sft uses verl remove-padding, so max_context_tokens is capacity, not realized work. in the current
# calibration corpus, twelve completed runs measured 1000.2-1009.0 tokens per step at batch 8 across
# context caps 1024, 2048 and 4096, while a matched batch sweep measured 1006.2, 2018.0 and 4015.1
# tokens per step at batch 8, 16 and 32. the batch dimension is therefore real and nearly linear; the
# context dimension is not. dividing these corpus arms by batch gives 125.0-126.1 tokens per example,
# so 128 is a conservative fallback for quotes that cannot inspect the dataset, not a general property
# of sft workloads. when post-run train_tokens exists, _sft_step_shape uses its measured average
# instead. the fallback is capped only for an authored context below 128, where the tokenizer cannot
# realize the full prior; raising capacity above 128 never raises billed work.
SFT_TYPICAL_TOKENS_PER_EXAMPLE = 128

# the smallest step a card is billed for. sized so the quoted step reproduces the MARGINAL cost of
# one step -- the slope of train_wall against step count -- because the quote adds run_startup_seconds
# separately and charging an AVERAGE s/step here would bill the run block twice.
#
# measured on the canonical cell (RTX 4090 / Qwen3.5-0.8B / batch 8 / ctx 1024), 16 completed runs
# spanning 2 to 256 steps (128x leverage), least squares on the per-step-count means:
#   train_wall = 104.7s + 1.237 s/step
# the average s/step at 64 steps is 3.435, which is 2.8x the marginal rate: that gap IS the block,
# amortized. an earlier revision of this constant was derived from that average and so double-counted
# the block, which is what put it at 24576.
#
# 9450 tokens is the value at which the quoted canonical step equals the measured 1.237 s/step
# marginal. 8192 (the corpus's own billed step size) lands at 0.87x of it and is within the
# replicate noise; 9450 is used because it is the measurement rather than a coincidence of shape.
SFT_SATURATION_TOKENS = 9450

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
    """(completions per step, billed tokens per step) for one OPD step, from a NORMALIZED config.

    Billed on ``n.completion_len``, the per-completion generation cap, exactly as GRPO bills its
    ``gen_tokens``. This term used to bill ``n.seq_len``, which is ``max_context_tokens`` -- the
    engine's context CAPACITY, not the tokens a step realizes. Capacity is a memory-sizing bound that
    a user may raise without changing the work done, so billing it made the quote track a config knob
    instead of the workload, and it over-quoted every arm whose context exceeded its completion cap.

    MEASURED, 65 held-out OPD arms across 5 (card, model, completion, context) cells: billing
    ``seq_len`` ran a geometric bias of 2.452x with 11/65 arms inside the band and a worst cell of
    7.23x; billing ``completion_len`` gives 1.520x, 20/65, worst cell 5.40x. A third candidate that
    split the prompt side onto the update term (prompt+cap on update, cap on generation) scored
    1.590x/19-of-65, i.e. no better than the simple cap, so the extra term is not carried.

    Known residual, deliberately NOT patched here. 1.520x is still outside this corpus's 1.194x
    run-to-run noise floor, so a second OPD error remains. It is not in this token basis: the wall
    is dominated by the card-independent fixed term (the FLOPs term is 2-15% of the OPD prediction),
    and the teacher is charged as ``teacher_lat * ceil(completions / OPD_TEACHER_BATCH_SIZE)``
    serial round trips while the measured blocking teacher wall
    (``opd_phase_teacher_wait_seconds``) is a median 2.26% of the train wall -- the worker overlaps
    the teacher with a prefetch pipeline instead of blocking on it. Fixing that is a change to the
    fixed term, not to the token shape, and it needs its own fit.
    """
    completions = n.batch_size * n.group_size
    return completions, completions * n.completion_len


def _sft_step_shape(n: RunConfig) -> tuple[int, int]:
    """(examples per step, realized tokens per step) for one sft step from a normalized config.

    when actual ``train_tokens`` are available, their per-step average is the strongest basis and is
    rounded up to a whole token. pre-flight quotes cannot inspect or tokenize user data, so they use
    ``SFT_TYPICAL_TOKENS_PER_EXAMPLE`` times the effective batch, capped only when the authored context
    is shorter than that prior.

    this term used to bill ``n.seq_len``, which is ``max_context_tokens`` -- the engine's context
    capacity, not the tokens a step realizes. verl removes padding, so a user may raise capacity
    without changing the work done; billing the capacity made the quote track a config knob instead
    of the workload.

    in the current 12-run calibration corpus, batch 8 realized 1000.2-1009.0 tokens per step across
    context caps 1024, 2048 and 4096. the old basis billed 8192, 16384 and 32768 tokens for that
    flat work; the 128-token prior bills 1024, only 1.015-1.024x the observed work. a matched sweep at
    context 1024 realized 1006.2, 2018.0 and 4015.1 tokens per step for batch 8, 16 and 32, so the
    batch multiplier remains while the context-cap multiplier is removed.
    """
    examples = n.batch_size
    if n.train_tokens is not None:
        return examples, max(1, math.ceil(n.train_tokens / n.steps))
    tokens_per_example = min(n.seq_len, SFT_TYPICAL_TOKENS_PER_EXAMPLE)
    return examples, examples * tokens_per_example


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


def run_startup_seconds(config: RunConfig, gpu: str) -> float:
    """One-time startup inside the billed train wall, paid once before the first optimizer step.

    SFT and the rollout methods pay different blocks for different reasons, so they are fitted
    separately: sft's is a verl launch plus model load and fsdp wrap (SFT_RUN_STARTUP_K), while a
    rollout run also builds the vLLM engine and does its first weight sync
    (``facts.run_block_seconds``, keyed on card because the block is card-shaped).

    ``gpu`` is only consulted for rollout methods; sft's block has no per-class table (see
    SFT_RUN_STARTUP_K -- reusing the rollout block overshoots, because it prices an engine build
    sft never performs). sft's block scales with model size instead: a flat constant is the same
    68.4 s for a 0.8B and a 27B run, while the realized arms need 81 s and 437 s respectively.
    """
    if config.method == "sft":
        params_b = total_params_b(config.normalized().model_id)
        return SFT_RUN_STARTUP_K * (params_b**SFT_RUN_STARTUP_PARAMS_EXP)
    if not config.has_rollout:
        return 0.0
    return run_block_seconds(gpu, resident=rollout_is_resident(config.normalized().model_id))


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

    It is also unfalsifiable in-tree, and for a STRUCTURAL reason rather than a contingent one.
    Measuring a speedup requires the same model on the same card class at two card counts, and no
    catalog model admits that pairing: a run is only submittable at n>1 when the model does NOT fit
    one card of that class (the allocator treats gpu.count as a ceiling, keeps single-card candidates
    that fit alone, downgrades to 1, and ``_validate_effective_spec`` then rejects the mismatch). So
    the two halves of the comparison are mutually exclusive by construction. Swept live for
    35B-A3B at bs 8 / ctx 4096: A100 PCIe, A100 SXM, H100 and RTX Pro 6000 all raise
    ``UnsupportedGpuError`` at n=1 (the model needs 100 GB, those cards have 80) while accepting
    n=2; H200 and B200 accept n=1 and silently downgrade n=2 back to 1. Zero cards do both.

    Multi-card arms DO now reach step timings -- v2/v3_mc_35b_sft_n2 completed on A100 PCIe n=2 at
    64 and 256 steps, giving a measured 5.443 s/step and a 519.6 s block -- so the earlier "no arm
    ever reached a step timing" is no longer the obstacle. What those arms cannot supply is the
    RATIO, because there is no matched single-card arm to divide by, and the nearest single-card
    arms (H200, 35B-A3B) differ in both card and context length. The absolute walls are also
    unusable as a check on their own: the MoE params basis over-quotes these same arms 1.9-3.8x
    even at n=1, so an absolute comparison scores that error, not this curve.

    Three further runtime defects, none of them cost defects, keep multi-card fragile:

    1. The allocator proposes card counts verl rejects. The gate is KV heads, not attention heads,
       and it is a disjunction: ``kv % sp == 0 or sp % kv == 0``. Under grouped-query attention the
       KV count is far smaller than the attention-head count, so it binds much earlier -- 27B has 24
       attention heads (``24 % 3 == 0`` passes) and only 4 KV heads, and dies on ``sp=3``. Computing
       both branches across the catalog (every model has 2 or 4 KV heads) leaves legal sp of {1,2,4},
       so a 3-card combination is unusable for EVERY catalog model while the allocator proposes it
       for all of them. Confirmed against the live allocator rather than only from the head counts:
       sweeping 27B and 35B-A3B over {sft, grpo, opd} x {A100 SXM, H100, A100 PCIe, RTX 5090} at
       ``max_gpu_count=4``, every grpo and opd shape returned exactly 3 cards. The illegal count is
       the modal multi-card outcome for the RL methods, not a corner case.
    2. Counts that do divide still die in NCCL, because Ray is never told the container's GPU count
       and both ranks bind device 0.
    3. Multi-card SFT dies before its first step on a fused-CE label shape under Ulysses SP.

    All three are still live as of this change: ``ulysses_sequence_parallel_size`` is set to the raw
    card count with no divisibility guard (sft_train.py:1292/1553, rl_train.py:391, opd_train.py:
    2501/2806), and ``ray_kwargs.ray_init`` passes num_cpus but never num_gpus (rl_train.py:454,
    opd_train.py:1622). So the 2-card CONSTANTS are measured (one fsdp benchmark per interconnect)
    while the end-to-end CURVE is not, and this docstring is the place that says so. Fixing any of
    them is trainer work, not cost work -- but note that fixing the gpu.count ceiling is also what
    would make this curve measurable at all, since that is what forbids the matched pairing above.

    What this module can and does guarantee meanwhile is SELF-CONSISTENCY: the ranker that picks a
    combination and the quote that prices it now share one call (``seconds_per_step``), so whatever
    this curve is worth, both halves of the cost model are worth exactly the same.

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
    """(shardable, non-shardable) seconds for one optimizer step on ``gpu``.

    Sharding a step across cards divides the first half and leaves the rest untouched: remote
    teacher scoring and reward grading are waits on services no card count speeds up, and an MoE pays
    its routing overhead once per step regardless. Ranking hardware needs the halves apart, because a
    latency-bound job gets no benefit from a faster card while a compute-bound one gets all of it.
    ``seconds_per_step`` is simply their sum, so this is the single source of the step model.

    The second half is what ADDING CARDS cannot shorten, which is not the same as being identical on
    every card. The rollout floor below is the case that distinguishes them: it varies by card class
    but does not shard, because engine init and weight sync do not run faster when split across more
    ranks (a wider job syncs more of them). Anything that scales with the arithmetic goes in the
    first half; anything a second card cannot help goes in the second.
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
        # OPD step = on-policy student rollout (like GRPO) + remote teacher scoring (SERIAL batched
        # Fireworks round-trips, replaces reward grading) + policy update (fwd+bwd only, NO local
        # reference forward — the teacher is the API). Bill local compute on the FULL prompt+completion
        # sequence (see _opd_step_shape), not completion-only, or long-prompt opd is underquoted.
        completions, seq_tokens = _opd_step_shape(n)
        gen_s = (GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM * params * seq_tokens) / (peak * MFU_DECODE)
        update_s = (OPD_UPDATE_FLOPS_PER_TOKEN_PER_PARAM * params * seq_tokens) / (
            peak * update_mfu
        )
        teacher_lat = teacher_seconds_per_completion()
        # a step's completions are echo-scored in batches of OPD_TEACHER_BATCH_SIZE, and those
        # batches are SERIAL: _TextTeacherBatcher (opd_train.py) runs a single daemon thread whose
        # loop takes one batch, blocks in _score_batch, and only then takes the next. _score_batch
        # issues exactly one score_many call = one echo POST (teacher.py). so the teacher wall is
        # ceil(completions / batch) round trips, NOT one — it grows linearly with step size, and the
        # two coincide only for a step that fits in a single batch.
        #
        # teacher_lat is an assumed average (facts.AVG_TEACHER_SECONDS_PER_COMPLETION), not a
        # measured per-batch latency, and an 8-item echo batch does not cost the same wall as a
        # 1-item one. the ROUND-TRIP COUNT below is read off the worker; the per-trip coefficient
        # remains a declared assumption. opd records only a teacher success count and no teacher
        # wall time, so nothing in-tree can calibrate it yet.
        teacher_trips = -(-completions // OPD_TEACHER_BATCH_SIZE)  # ceil, ints only
        teacher_s = teacher_lat * teacher_trips
        # the teacher is a remote api: its latency is identical on every card, so it is the part of an
        # opd step that a faster or more numerous gpu cannot shorten.
        #
        # opd carries the rollout wall because it runs the same verl rollout path grpo does -- vllm
        # generation and the actor->rollout weight sync both happen here, and it generates the same
        # per-completion sequences, so the slope applies. opd arms are now IN the fitted corpus (a
        # 4090 step sweep at 8/16/48), so this is measured on opd rather than inferred from grpo.
        #
        # the run-level engine build is NOT charged here -- it is once per run, and lands in
        # run_startup_seconds. see facts.rollout_step_seconds for why no per-card constant remains.
        return gen_s + update_s, overhead + teacher_s + rollout_step_seconds(completions)

    if not n.is_grpo:
        # NO rollout wall here, neither term. sft is a plain fwd/bwd loop: no vllm engine to enter and
        # leave, no actor->rollout weight sync, and no sampled completions for the slope to price. the
        # mechanism does not exist here. every arm the wall was fitted on is a rollout arm, so charging
        # it would extrapolate ~50s/step onto a loop it was never measured against.
        #
        # 11 sft arms have since been measured (2..256 steps, RTX 4090 and H100, 0.8B and 4B) and
        # they do NOT contradict the exclusion above: there is no rollout in an sft loop and nothing
        # in the timings looks like one. what they DO show is that a separate fixed block exists.
        # train_wall wraps run_verl_training and the watcher.stop() upload drain (sft_train.py
        # 1450-1478) while setup_seconds is stamped at :1203, BEFORE that subprocess launches -- so
        # verl startup, model load, lora wrap and fsdp init land inside train_wall and are quoted
        # here at zero.
        #
        # that block is deliberately NOT added, because its size is not measured. fitting
        # wall = fixed + slope*steps looked convincing (two families agreeing with the quote below to
        # within 5%) and did not survive leave-one-out: dropping one of five 4090 arms moved the
        # slope 1.021 -> 1.992 s/step and the intercept 70.9s -> 26.6s.
        #
        # replicates were then run to settle it. six arms identical in card, model, step count and
        # seed -- differing only in run_name -- spread 99.2/148.1/150.4s at 32 steps (39% of the
        # mean) and 291.9/404.1/545.0s at 256 steps (61%), and pooling them gives slope
        # 0.632..1.990 s/step (a factor of 3.2) with intercept 35.5..130.2s.
        #
        # that spread is NOT run-to-run pod noise, which is what it was first read as. the six arms
        # landed on three different host classes (nvidia driver builds 570.172.08 / 570.195.03 /
        # 580.126.20) and the wall tracks the class: across 11 replicate groups campaign-wide,
        # the 8 whose replicates shared a driver build spread 1.2..16.4% (median 5.2%) while the 3
        # that straddled builds spread 38.6..81.4%. driver is near-perfectly confounded with
        # datacenter in these arms, but the cases that separate them rule datacenter out -- same
        # build across different sites stays tight (grpo 453.2s@US-CA-2 vs 477.5s@US-NE-1, and the
        # 32-step sft pair 148.1s@US-IL-1 vs 150.4s@EUR-NO-1, 1.6% apart) while same site across
        # different builds does not. so the confound is host class, and it is systematic, not noise.
        #
        # controlling for it, the fit IS stable: within build 580.126.20 (5 arms, 3 step counts, 8
        # pairings) slope is 0.657..0.894 s/step and the fixed block 63.0..78.2s -- the intercept
        # range collapses from 94.7s pooled to 15.2s. the other builds fit the same shape with a
        # slower slope and a bigger block (570.195.03: 1.132..1.808 s/step, 92.6..114.2s).
        #
        # a fixed block of roughly 63..123s therefore exists and is measured, but it is per host
        # class, and this module has no host-class input: the caller knows the card, not which
        # driver build the pod will land on.
        #
        # that argues against quoting the MEAN, not against quoting anything. the alternative on
        # offer is not "a right value vs a wrong one" -- it is 0s, which is wrong for every class by
        # more than any in-range value is wrong for any single one. so the block IS quoted, once per
        # run, at the bottom of the measured range (SFT_RUN_STARTUP_K, charged in estimate_cost
        # rather than here because it is per run, not per step). the floor is the one value that
        # cannot overquote any class observed, and it strictly beats zero everywhere: over all 16
        # sft arms, quoted/realized moves from a median 0.46x to 0.83x, 14 of 16 arms land closer to
        # their measured wall, arms underquoting by >20% fall from 14 to 6, and no arm crosses into
        # overquoting (0 of 16 above 1.25x, before and after). what remains is a deliberate residual
        # underquote on the slower classes, which is the safe direction -- a user who lands on a
        # fast pod is never billed for a slow one.
        #
        # the per-step SFT slope is a different question and stays unchanged: pooled it is still
        # unidentifiable (0.632..1.990 s/step across pairings), and nothing above resolves it.
        # a step below SFT_SATURATION_TOKENS cannot fill the card, so it costs what a saturated step
        # costs. this stays in the gpu-bound half: it is still arithmetic on this card, just at an
        # occupancy floor, so a faster card still shortens it and extra cards still shard it.
        _, realized_step_tokens = _sft_step_shape(n)
        step_tokens = max(realized_step_tokens, SFT_SATURATION_TOKENS)
        flops = SFT_FLOPS_PER_TOKEN_PER_PARAM * params * step_tokens
        return flops / (peak * sft_mfu), overhead

    # GRPO step = rollout (G completions/prompt) + serial reward grading + policy/ref update.
    completions = n.batch_size * n.group_size
    gen_tokens = completions * n.completion_len
    gen_s = (GRPO_GEN_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * MFU_DECODE)
    update_s = (GRPO_UPDATE_FLOPS_PER_TOKEN_PER_PARAM * params * gen_tokens) / (peak * update_mfu)
    latency = reward_seconds_per_completion(n.reward_seconds_per_completion)
    # every completion is scored, one at a time (see the serial-scoring note above).
    reward_s = completions * latency
    # reward grading runs off-gpu, so like the opd teacher it is fixed wall time no card choice
    # changes. a grpo step dominated by it is latency-bound, not compute-bound.
    #
    # reward and rollout are BOTH per-completion but they are separate terms on purpose. they were
    # conflated before: the rollout slope (0.66s) is close enough to the reward default (1.0s) that
    # one term could stand in for both, right up until a caller passed a MEASURED reward latency --
    # real graders here are ~0.0003s -- and the rollout cost vanished with it. see facts.py.
    #
    # the engine build a sleep_unsupported model skips is once per RUN, so the resident case is
    # handled in run_startup_seconds, not here. this slope is per-sequence sampling work that a
    # resident engine still pays -- see facts.rollout_step_seconds.
    return gen_s + update_s, overhead + reward_s + rollout_step_seconds(completions)


def seconds_per_step(config: RunConfig, gpu: str, gpu_count: int = 1) -> float:
    """Steady-state wall time for one optimizer step on ``gpu_count`` cards of class ``gpu``.

    Sharding divides only the gpu-bound half (see ``step_seconds_split``); the non-shardable floor
    is paid whole on every card count. This is the SAME arithmetic the allocator ranks combinations
    with (``_step_cost_ranker``), and it has to be, because the two answer one question: the ranker
    picks the combination and this prices it. Before they were shared, the quote applied no speedup
    at all while still billing every card, so a 2-card 4090 run was ranked at 1.41x of one card and
    then quoted at 2.00x with an identical wall -- the model contradicting itself by the full
    speedup factor.
    """
    gpu_bound, fixed = step_seconds_split(config, gpu)
    return gpu_bound / multi_card_speedup(gpu_count, gpu) + fixed


def step_cost_key(config: RunConfig):
    """``(gpu, hourly_rate) -> dollars per optimizer step``, or None if ``config`` can't be priced.

    The single ranking basis shared by every GPU-selection path (submit-time allocation, the
    parse-time provisional shown in the schema, and the cost estimate). Renting the cheapest card
    is not the same as running the job for the least money: a card bills for the time it takes, so
    the right basis is rate x duration. An A10 at $0.75/hr is nominally cheaper than an H100 at
    $3.29 but sustains a fraction of its FLOPs, so the same run costs about three times as much on
    it. Ranking per STEP prices both halves at once.

    The run's LENGTH is load-bearing here, which it was not before. A per-step key orders candidates
    exactly as total job cost does only while every term is per-step -- then the shared step count
    is a positive constant factor and cancels. Once-per-run terms break that: they are paid once no
    matter how long the run is, so a key that adds one whole gives it the weight of a single step,
    over-weighting it by a factor of `steps`. The rollout block makes this concrete -- it spans ~7x
    across classes (98s on RTX 5090 to 762s on H200), so mis-weighting it reorders the fleet. Above
    8 steps the un-amortized key ranks RTX 5090 cheapest when RTX 4090 actually is, picking a card
    that costs up to 1.40x more to run the job; over a 792-configuration grid it chose wrong on 133,
    while the amortized key below chose wrong on 0.

    So EVERY per-run term is divided by the step count -- the rollout/sft block, the one-time MoE
    compile, and the required-save overhead. Charging only some of them is not a partial fix but a
    different wrong answer: an early revision amortized the block alone and still inverted 4 of 324
    MoE-and-save configurations, because a term being card-FREE in seconds does not make it neutral
    in dollars. Card-free seconds are billed at a per-card rate, so omitting them under-weights
    exactly the cheap-card advantage this key exists to price.

    With all four terms present the key is exactly ``job_cost / steps``, and dividing every candidate
    by the SAME positive number is order-preserving -- which restores, by construction, the
    cancellation this docstring used to claim unconditionally. It also matches what ``estimate_cost``
    already does when it reports a per-step figure (``sps = raw_train / config.steps``); before this
    the ranker and the quote disagreed about the same seconds.

    ``config.steps`` is therefore read for its VALUE, not just carried. Callers that rank before the
    true horizon is known pass the corpus median -- see ``providers.base.run_config_for_ranking``.

    ``gpu_count`` shards the gpu-bound half of a step (see ``multi_card_speedup``) and nothing else:
    the per-run block is paid once by the job however many cards it spans. Multi-card ranking runs
    through this same closure rather than a parallel formula in the allocator -- when it did not,
    the two branches priced different things and a combination's block silently vanished from the
    key while the quote still billed it.

    Returns None when the model is outside the cost catalog, so callers fall back to $/hr for
    EVERY candidate rather than ranking a mix of two incomparable bases.
    """
    try:
        step_seconds_split(config, "H100")  # probe: raises for a non-catalog model
    except Exception:
        return None

    steps = max(1, config.steps)
    # card-free, so hoisted out of the per-candidate path; still amortized, for the dollars reason
    # in the docstring above.
    save_s = required_save_overhead_seconds(config)

    def cost_key(gpu: str, hourly_usd: float, gpu_count: int = 1) -> float:
        if gpu not in GPU_COMPUTE_TFLOPS:
            # no measured/spec throughput for this class, so its step time would be computed from a
            # placeholder. ranking on that invents a speed difference the hardware may not have;
            # return a constant instead, which leaves these classes ordered by the $/hr tie-break.
            return 0.0
        try:
            gpu_bound, fixed = step_seconds_split(config, gpu)
        except Exception:
            return 0.0  # unpriceable class falls back to the $/hr tie-break, never fails selection
        per_run = compile_seconds(config, gpu) + run_startup_seconds(config, gpu) + save_s
        step_s = gpu_bound / multi_card_speedup(gpu_count, gpu) + fixed
        # hourly_usd is PER CARD (Candidate keeps it that way), so an n-card combination occupies
        # n cards for the whole wall -- the same multiplication estimate_cost applies to total_usd.
        return max(1, gpu_count) * hourly_usd * (per_run / steps + step_s) / 3600.0

    return cost_key


def sft_seconds_for_tokens(
    config: RunConfig, gpu: str, train_tokens: float, gpu_count: int = 1
) -> float:
    """SFT steady-state wall time for an actual token count on ``gpu_count`` cards of ``gpu``.

    Unlike a step, this is pure arithmetic with no non-shardable floor in it, so the WHOLE quantity
    divides by the multi-card speedup rather than half of it.

    The budget is charged per STEP at the SFT_SATURATION_TOKENS floor rather than as one bulk flops
    figure, because a step carrying fewer tokens than that floor still costs a saturated step (see
    that constant). Pricing the budget in bulk would let a token-budgeted quote come out an order of
    magnitude under the same run priced from its realized shape by ``step_seconds_split``.
    """
    n = config.normalized()
    # MoE prices on total params at a reduced MFU (see seconds_per_step); dense keeps active.
    moe = _is_moe(n.model_id)
    params = (total_params_b(n.model_id) if moe else active_params_b(n.model_id)) * 1e9
    mfu = MFU_SFT_TRAIN_MOE if moe else MFU_SFT_TRAIN
    peak = effective_train_tflops(gpu) * 1e12
    _, realized_step_tokens = _sft_step_shape(n)
    step_tokens = max(1, realized_step_tokens)
    steps = max(1.0, math.ceil(train_tokens / step_tokens))
    billed_tokens = steps * max(step_tokens, SFT_SATURATION_TOKENS)
    flops = SFT_FLOPS_PER_TOKEN_PER_PARAM * params * billed_tokens
    return flops / (peak * mfu) / multi_card_speedup(gpu_count, gpu)


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
    # rank on job cost, not rental rate, so the quote names the class the allocator will pick.
    gpu = pick_gpu(
        need,
        provider=config.provider,
        max_wall_seconds=max_wall_seconds,
        cost_key=step_cost_key(config),
    )
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
    if config.gpu_type:
        # gpu_type is provisioned at launch through allocate (disk-aware, live rate). quote the same
        # path with either an automatic or pinned provider so the estimate matches the actual launch rate,
        # and a disk floor lifts the quote off the wrong class.
        from flash.providers.allocator import allocate

        allocation = allocate(
            config.model_id,
            config.method,
            train=config.train_knobs(),
            thinking=config.thinking,
            max_wall_seconds=market_wall_s,
            disk_gb=config.disk_gb,
            provider=("" if config.provider == "auto" else config.provider),
            gpu_type=config.gpu_type,
            model_revision=config.model_revision,
            # quote the same card ceiling launch allocates under, so a multi-card run is estimated on
            # the combination it will actually rent rather than on a single-class search that ignores it.
            max_gpu_count=config.gpu_count,
        )
        gpu = allocation.gpu
        quote_provider = allocation.provider
        hourly = allocation.hourly_usd
        need = allocation.min_vram_gb
        # max_gpu_count is a CEILING, so the allocator may fit the run on fewer cards than requested
        # (e.g. 2x of a class when 4 was allowed). bill the count it actually chose, not the ceiling.
        billed_gpu_count = getattr(allocation, "gpu_count", 1) or 1
    else:
        gpu, need = select_gpu(config, max_wall_seconds=market_wall_s)
        quote_provider = config.provider
        # no gpu_type pin means no allocate() call and so no combination search; this branch picks a
        # single class that fits alone, and the run occupies the requested count of it.
        billed_gpu_count = config.gpu_count
        # quote the same vram-floored vast market pick_gpu selected under (min_vram_gb=need): without the
        # floor the rate lookup searches from the smallest managed class, letting cheap small-card offers
        # crowd a high-vram selection off the limited page -> it silently falls back to the static rate.
        hourly = gpu_hourly_usd(
            gpu,
            provider=quote_provider,
            max_wall_seconds=market_wall_s,
            min_vram_gb=need,
            gpu_type="",  # this branch is only reached when gpu_type is empty
        )

    setup = setup_seconds(config)
    # price the step on the card count actually billed, not on one card. total_usd multiplies by
    # billed_gpu_count, so quoting a single-card wall here would charge n cards for a run the model
    # says takes the same time on 1 -- always ranking one card cheapest, by construction rather than
    # by measurement. the speedup applies to the gpu-bound half only (see seconds_per_step).
    sps = seconds_per_step(config, gpu, billed_gpu_count)
    required_save_s = required_save_overhead_seconds(config)
    # A one-time kernel/graph compile is paid once on the first step (MoE-only; 0 for dense). It is
    # training GPU time, so it belongs in the (billed) train term, not setup.
    compile_s = compile_seconds(config, gpu)
    # Same argument: launch, model load and framework init run inside the billed train wall and are
    # quoted per RUN, not per step. sft pays a verl/fsdp block; grpo and opd pay a larger, card-shaped
    # one because they also build the vLLM engine and do the first weight sync.
    startup_s = run_startup_seconds(config, gpu)
    raw_train = compile_s + startup_s + config.steps * sps + required_save_s
    if not config.is_grpo and config.train_tokens is not None:
        raw_train = (
            compile_s
            + startup_s
            + sft_seconds_for_tokens(config, gpu, config.train_tokens, billed_gpu_count)
            + required_save_s
        )
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
        gpu_count=billed_gpu_count,
        # total_usd is the customer gpu charge. the platform-owned teacher spend is itemized
        # only as a diagnostic and is not passed through to the customer. an n-card job occupies
        # n cards for the billed training wall, so the charge scales linearly with gpu_count.
        total_usd=train / 3600.0 * hourly * billed_gpu_count,
        teacher_api_usd=teacher_api_usd,
        notes=_notes(config, raw_train, wall_capped, cap_s),
    )
