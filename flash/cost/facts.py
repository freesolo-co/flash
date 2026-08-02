"""Catalog facts and provider-aware pricing accessors for the cost model."""

from __future__ import annotations

from flash.catalog import MODELS, ModelInfo
from flash.providers.base import GPU_INFO, GpuClass, providers_for

GPU_COMPUTE_TFLOPS: dict[str, float] = {
    # A10: 125 TFLOPS dense bf16 tensor (NVIDIA spec); Lambda-only 24 GB class, else defaults to 100.
    "A10": 125.0,
    # MEASURED on a rented RunPod RTX 4090 (sm89, torch 2.10+cu128): 162.2-162.7 TFLOPS sustained
    # on large square bf16 matmuls, against the 165 vendor spec -- within 1.5%, so the spec number
    # is a fair proxy for this class and is kept.
    "RTX 4090": 165.0,
    "RTX 5090": 210.0,
    # MEASURED on a rented RunPod A100 80GB PCIe (sm80, torch 2.10+cu128): 242-259 TFLOPS sustained
    # across repeated trials at full clocks (1410 MHz) and 31C, i.e. neither thermal- nor
    # power-limited. The 312 figure is the vendor dense-bf16 spec, which real matmul kernels do not
    # reach; using it overstated A100 throughput by ~20% and made the class look faster (and so
    # cheaper per job) than it is. Set to the measured sustained rate.
    "A100 PCIe": 250.0,
    # MEASURED on a rented RunPod A100-SXM4-80GB (sm80, torch 2.10+cu128): 263-265 TFLOPS across
    # repeated trials. Same 108 SMs as the PCIe part but a 400 W envelope against 300 W, which is
    # what the ~6% gain over the measured PCIe rate reflects. Both are far below the shared 312
    # vendor spec.
    "A100 SXM": 264.0,
    # A100 SXM 40GB: same SMs/tensor cores as the 80GB A100 SXM, less HBM only, so it inherits the
    # measured SXM rate. Without an entry, 33-40 GB Lambda/Vast quotes fall back to _DEFAULT_TFLOPS.
    "A100 SXM 40GB": 264.0,
    # MEASURED on a rented RunPod H100 PCIe (sm90, torch 2.10+cu128): 490-504 TFLOPS sustained
    # across repeated trials at full clocks (1755 MHz), 310 W board. The 990 figure is the SXM
    # part's spec; the PCIe board this class actually provisions delivers about half of it. Using
    # 990 made H100 look twice as fast as it is and let it win nearly every ranking decision on
    # throughput it cannot deliver. Set to the measured sustained rate.
    "H100": 500.0,
    # H200: same SM/tensor-core configuration as H100 with more HBM and a higher power envelope.
    # Scaled from the H100 measurement rather than the SXM spec. Not directly measured.
    "H200": 550.0,
    "RTX Pro 6000": 250.0,
    # B200: 2.25 PFLOPS bf16 dense (NVIDIA spec); prevents ~10x cost over-estimate vs _DEFAULT_TFLOPS.
    "B200": 2250.0,
}
_DEFAULT_TFLOPS = 100.0

# classes whose cards talk to each other over nvlink rather than the pcie root complex. this is the
# single largest input to multi-card scaling: the same 2-card benchmark measured 1.7675x on an
# nvlink pair and 1.4212x on a pcie pair, so one global scaling constant cannot describe both.
# membership is by form factor -- sxm datacenter parts carry nvlink, pcie boards and every geforce
# part do not (the 4090 dropped nvlink entirely). anything absent is treated as pcie, which is the
# conservative side: it under-credits a combination rather than ranking it on bandwidth it lacks.
#
# classify by the board a MULTI-CARD run actually lands on. multi-card provisioning is runpod-only
# (vast and lambda have no gpu_count path), so what matters per class is runpod's pin, not the
# cheapest board the class can serve a single-gpu run on.
_NVLINK_CLASSES: frozenset[str] = frozenset(
    {
        # MEASURED: 2x A100-SXM4-80GB on RunPod reached 1.7675x (see MULTI_CARD_SCALING_NVLINK).
        "A100 SXM",
        # same sxm4 board and nvlink fabric as the 80gb part, less hbm only.
        "A100 SXM 40GB",
        # runpod pins H100 to NVIDIA_H100_80GB_HBM3 (providers/base.py) -- the sxm part, with
        # nvlink. the pcie and NVL boards in runpod's ADA_80_PRO pool are explicitly NEGATED by
        # providers/runpod/gpus.py, so a multi-card H100 combination lands on sxm silicon. the
        # measured-tflops comment on the H100 entry above refers to a single-gpu PCIe rental and
        # says nothing about which board a multi-card run gets.
        "H100",
        # sxm parts with nvlink/nvswitch by form factor. INFERRED, not measured -- no multi-card
        # benchmark has been run on either class.
        "H200",
        "B200",
    }
)


def gpu_tflops(name: str) -> float:
    """Peak bf16 tensor TFLOPS for a managed GPU class."""
    return GPU_COMPUTE_TFLOPS.get(name, _DEFAULT_TFLOPS)


def has_nvlink(name: str) -> bool:
    """Whether cards of class ``name`` are interconnected by nvlink rather than pcie.

    Unknown classes report False: a class nobody has classified is far more likely to be a pcie
    board than an sxm one, and guessing wrong in that direction only under-credits scaling.
    """
    return name in _NVLINK_CLASSES


# realized TRAINING throughput sits well below peak when a class's training kernels don't reach it.
# b200 (sm100) has no arch-tuned training kernels yet and falls back to the same portable paths as
# h200 (sm90), so its 2.25 pflops dense-bf16 peak does not materialize for training -- realized
# throughput is h200-class (and frequently lower for rl/grpo). cap b200 at h200 so the analytical
# cost model does not rank it as faster/cheaper than h200 on peak flops alone (it is not, and is
# often slower). this is a conservative floor: it never prices b200 above h200-equivalent training
# time, so it cannot over-charge, and it removes the "b200 looks cheapest" inversion at the source.
# refine per-workload once real b200 training samples exist. the vram/serving paths keep the true
# peak via gpu_tflops; only the training-time model uses this.
_TRAIN_TFLOPS_CAP: dict[str, float] = {
    # h200-class realized training throughput, not the 2250 dense-bf16 peak. tracks the H200 entry,
    # which is itself now anchored to a measured H100 PCIe rate rather than a vendor spec.
    "B200": 550.0,
}


def effective_train_tflops(name: str) -> float:
    """Realized bf16 TRAINING throughput for the analytical cost model.

    Equal to ``gpu_tflops`` (peak) except where a class's training kernels fall short of peak: b200
    training falls back to portable kernels, so it is capped at h200-class throughput rather than
    its raw 2.25 pflops peak."""
    peak = gpu_tflops(name)
    cap = _TRAIN_TFLOPS_CAP.get(name)
    return min(peak, cap) if cap is not None else peak


# --- per-step rollout wall ----------------------------------------------------------------------
# MEASURED. Every optimizer step pays wall cost the FLOPs model cannot see: vLLM engine entry/exit,
# the actor->rollout weight sync, ray actor dispatch, per-sequence sampling and detokenize, and
# dataloader/collate work. None of it is arithmetic, so no MFU or TFLOPS number can produce it, and
# the previous model had no term for it at all -- it quoted a step as (compute + reward wait).
#
# The effect dominates real runs. Across a 56-arm RunPod campaign (Qwen3.5 0.8B/2B/4B, 8 classes,
# batch/group/context swept, 16-256 completions/step) the compute term was 0.3-9.3s while realized
# steps took 43-392s, so 95-99% of a real step is wall the model did not price. Under leave-one-out
# cross-validation over that corpus -- every arm predicted by a fit that never saw it -- this lands
# geo-bias 1.01x, typical error 1.22x, 50/56 arms inside the 0.70-1.43x band.
#
# LOOCV rather than a single holdout because the obvious holdout is not honest: the wave-M arms that
# would serve as one include the only two A100 SXM arms in existence, which are also the only source
# for that class's constant. Scoring a table on rows it was fitted on flatters it. LOOCV has no such
# row, and it uses the whole corpus for the completion-count axis the two candidate models disagree
# on. On the 9 wave-M arms that ARE genuinely out-of-sample, this scores geo-bias 0.93x / 7 in band.
#
# Every arm here is a ROLLOUT (grpo) arm on RUNPOD. Two limits follow, both deliberate:
#   - sft is excluded at the call site (see analytical.step_seconds_split), on mechanism not
#     measurement: sft runs no vllm engine and no weight sync, so the wall this table measures has
#     no source there. sft keeps its pre-existing compute-only quote.
#   - the floor is keyed on card alone, not (card, provider). A matched lambda/vast replication was
#     attempted and returned zero usable timings (8/8 arms died in provisioning, step-0 wedge, or a
#     trainer crash), so a provider term is unmeasured rather than measured-and-rejected. Card-only
#     is the conservative choice: it is a strict improvement over pricing no floor at all on every
#     provider, and if a provider offset does exist it shows up as a class being uniformly off on
#     that provider, which is the same signal that would justify adding an entry.
#
# The wall has TWO terms, and separating them is the whole subtlety here.
#
# An earlier revision of this table fitted a per-card CONSTANT on top of the existing quote. That
# quote still contained ``completions * AVG_REWARD_SECONDS_PER_COMPLETION``, whose default is 1.0s.
# 30 of the 32 arms in the first campaign ran exactly 32 completions, so that term was a near
# constant 32s across the entire fit and the "floor" silently absorbed it. It scored well only
# because the corpus never varied the axis the two terms disagree on.
#
# Widening to 56 arms spanning 16-256 completions separates them, and the wall is plainly NOT
# completion-independent -- per-card medians of (realized - compute), reward excluded:
#
#     card         g=32    g=64    g=96    g=256
#     H100         77.8                    230.7
#     A100 PCIe    71.8                    215.6
#     B200         77.7                    383.2
#     H200        133.3                    351.5
#     RTX 4090     79.6   146.6   216.1
#     RTX 5090     43.9           60.7
#
# So it is modelled as ``const_card + ROLLOUT_SECONDS_PER_COMPLETION * completions``. The slope is
# pooled across cards (it is a rollout-path property -- sampling, detokenize, per-sequence engine
# bookkeeping) while each class keeps its own constant (engine entry/exit and weight sync, which do
# not scale with the batch). Per-card slopes were rejected: several classes have 2-4 arms, which
# cannot separate slope from intercept, and fitting per-card slopes on that produced NEGATIVE
# extrapolated step times.
#
# Why this matters beyond accuracy: at 0.81s/completion the slope is within 20% of the fictitious
# 1.0s reward default, which is exactly why the constant-only revision looked right. It was pricing
# per-completion ROLLOUT cost through the reward term. That is not a harmless mislabel: feed the
# constant-only model a measured reward latency (real graders here are ~0.0003s) and the quote
# collapses from geo-bias 1.017x (50/56 in band) to 0.517x (15/56) -- the more accurate the reward
# number, the worse the estimate. Splitting the terms removes that coupling.
#
# To be exact about who can supply that number today: NOBODY on the submit path. RunConfig accepts
# an override, but both construction sites (cost/spec.py, providers/base.py) leave it None, so the
# persisted quote always uses the default below. engine/reward_profile.py measures the real grader
# on the worker AFTER submission and only records it in run notes. The collapse above is therefore
# a sensitivity result from the offline harness, not a live regression -- but it is the failure the
# split makes structurally impossible, which is what makes wiring that measurement through safe to
# do later. Until it is wired, see the judge-gap note on AVG_REWARD_SECONDS_PER_COMPLETION.
#
# Honest scoring note: leave-one-out over the 56 arms puts two-term at 1.007x/50-56 against
# const-only's 1.017x/50-56, so pooled this is a TIE on accuracy, not a win. It is kept for the
# robustness above and for the high-completion bucket the first fit could not see (g>=129:
# 1.183x -> 1.022x). Scored by LOOCV because the natural holdout contains the only two A100 SXM
# arms, which are also the sole source of that class's constant.
ROLLOUT_SECONDS_PER_COMPLETION = 0.66

# Per-RUN rollout block: vllm engine build, the first actor->rollout weight sync, and trainer/replay
# init. Paid ONCE before the first step, not on every step. It lands after ``setup_seconds`` is
# stamped, so like sft's SFT_RUN_STARTUP_K it is billed inside the train wall.
#
# This table replaces a per-STEP constant, and the replacement is the fix, not a re-tuning. That
# constant was fitted per arm as median(realized - compute - slope*completions), and the corpus was
# ~90% 6-8 step arms, so each arm's share of a once-per-run block entered the median ALREADY DIVIDED
# by its own step count. The constant was therefore never a per-step quantity: it was this block
# scaled by 1/steps. Two independent checks:
#   - BLOCK/7 reproduces the old constant within a few seconds on three classes fitted separately.
#   - After charging the block once, the RESIDUAL per-step constant is ~0 on every class in the
#     fleet (0.00 H100, 0.00 RTX 5090, -0.00 B200, -0.00 H200, 0.00 A100 SXM, -0.24 A100 PCIe,
#     0.66 RTX 4090, -1.72 RTX Pro 6000) against old constants of 17.7 to 126.4 s/step. There is
#     nothing left for a per-step constant to explain, which is why one is no longer charged.
#
# Separating an intercept from a slope needs leverage: a class whose arms all sit at one step count
# cannot do it, and a 6->8 step span is collinear enough to produce absurdities (a 2.26 s/step slope
# on H100, a negative intercept on a 4090 pair). Every class below was measured across at least
# three distinct step counts for that reason; the campaign that added steps 4 and 24 to the
# short-span classes is what made A100 SXM, H200 and RTX Pro 6000 identifiable at all.
#
# SCORED BY LEAVE-ONE-OUT over all 88 rollout arms, refitting BOTH forms on the other 87 with each
# one's own estimator -- comparing a whole-corpus-fitted incumbent against a held-out challenger
# would have biased the result toward the change. Geo-bias 1.427x -> 1.295x, in band 42/88 -> 53/88,
# 7 of 8 classes improved. The gain is concentrated exactly where the mechanism predicts, outside
# the 6-8 step band the old constant was fitted in:
#   steps <= 8 (n=77): 1.344x -> 1.313x   in band 41 -> 45
#   steps 9-24 (n=7) : 2.016x -> 1.105x   in band  1 -> 7
#   steps > 24 (n=4) : 2.011x -> 1.290x   in band  0 -> 1
# For scale: replicate arms differing only in run_name spread 1.194x (median over 17 groups), so
# 1.295x is at the measurement floor while 1.427x is well above it. Per-class differences smaller
# than that floor are not distinguishable and are not claimed either way.
#
# This is load-bearing for card SELECTION, not just for the quote. Charging a per-run block per step
# multiplies a ~7x class spread by `steps` and reorders the fleet: above 8 steps the old model ranks
# RTX 5090 cheapest when RTX 4090 actually is, and the card it picks costs up to 1.34x more to run
# the job. See ``analytical.step_cost_key`` for how the block enters ranking without that error.
#
# Every arm is a ROLLOUT (grpo/opd) arm on RUNPOD. Two limits follow, both deliberate:
#   - sft is excluded at the call site (see analytical.run_startup_seconds), on mechanism not
#     measurement: sft runs no vllm engine and no weight sync, so the block this table measures has
#     no source there. sft keeps its own SFT_RUN_STARTUP_K.
#   - it is keyed on card alone, not (card, provider). A matched lambda/vast replication returned
#     zero usable timings (8/8 arms died in provisioning, step-0 wedge, or a trainer crash), so a
#     provider term is unmeasured rather than measured-and-rejected. Card-only is the conservative
#     choice, and a real provider offset would surface as a class being uniformly off on that
#     provider -- the same signal that would justify adding an entry.
_RUN_BLOCK_S: dict[str, float] = {
    "A100 PCIe": 251.2,
    # ~2.2x its own PCIe sibling, which is why the two variants cannot share one entry.
    "A100 SXM": 543.4,
    # Same SMs and tensor cores as the 80GB board, less HBM only -- and this block is engine-build
    # and weight-sync bound, not capacity bound. So it inherits the 80GB measurement for the same
    # reason the TFLOPS table does. Left out, it fell through to the pooled default and mis-priced
    # every lambda/vast quote in that band.
    "A100 SXM 40GB": 543.4,
    # Re-estimated on step leverage (was 410.3, fitted at a single step count). B200 is the ONLY
    # class the step-identification sweep moved: its 410.3 exceeded the entire realized 4-step run
    # on that card (272.8 s), and a block is a strict subset of the run it starts. 1.50x is past the
    # 1.194x replicate noise floor, and all three estimators agree it is >15% high (least squares
    # 235.7, robust median 302.8, feasibility 180.9). The robust median is taken rather than the
    # least-squares point because on B200's own arms 235.7 overshoots to 0.823x while 302.8 lands at
    # 0.932x and gains an arm into band (4/6 -> 5/6).
    "B200": 302.8,
    "H100": 339.0,
    # H200 is the outlier: ~2.2x the H100 block on the same sm90 kernels. Consistent across its arms,
    # so it is a property of the class as provisioned, not a bad sample.
    #
    # Its 4-step arm also under-runs this block (549.6 s), but it is NOT corrected, and the contrast
    # with B200 is the reason: the three estimators disagree here (672.7 / 769.4 / 457.7, only 1 of 3
    # below), the class carries 2.17x spread across its own arms, and the shipped value already
    # scores 1.016x on them. Every candidate replacement is worse (457.7 -> 0.726x). One low arm on a
    # wide-spread class is a leverage point, not a measurement.
    "H200": 762.4,
    "RTX 4090": 404.6,
    "RTX 5090": 98.4,
    "RTX Pro 6000": 260.5,
}
# Unmeasured classes take the pooled median across the whole campaign rather than zero: charging no
# block at all is the failure this table exists to fix, and it under-quotes by 3-4x. A class here is
# quoted low, so add a measured entry above once a class has arms at three or more step counts.
_DEFAULT_RUN_BLOCK_S = 344.8


def rollout_is_resident(model_id: str) -> bool:
    """Whether ``model_id``'s rollout engine stays resident instead of sleeping between steps.

    Reads the same ``sleep_unsupported`` catalog flag the worker does
    (``engine.worker.backend_common.rollout_sleep_unsupported``), rather than importing it, to keep
    the cost model off the worker import path. The flag is the single source of truth for both.
    """
    from flash.catalog import MODELS

    info = MODELS.get(model_id)
    return bool(info is not None and getattr(info, "sleep_unsupported", False))


def rollout_step_seconds(completions: int = 0) -> float:
    """Per-step rollout wall that no amount of FLOPs shortens: ``slope * completions``.

    This is per-sequence sampling, detokenization and dispatch work. It is not arithmetic, so it
    does not shrink on a faster card and (see ``providers.allocator``) it does not shard across more
    of them.

    There is deliberately NO per-card constant term here. One used to be charged, but refitting with
    the once-per-run block separated out (``_RUN_BLOCK_S``) drives the residual per-step constant to
    ~0 on every class in the fleet -- the largest magnitude left anywhere is 1.72 s/step, against
    old constants of 17.7 to 126.4. The constant was the run block divided by its arms' step count,
    not a per-step quantity, so charging both terms would double-count the same seconds.

    Nor is the slope keyed on card, and that is measured rather than assumed: it is per-sequence
    host-side and sampling work whose per-class fits agree within the corpus noise floor, so a
    per-card table would be fitting run-to-run pod variance.

    No ``resident`` parameter, for the same reason: what a resident engine skips is the engine
    build/teardown cycle, which now lives entirely in the run block. It does not skip per-sequence
    sampling, so this term is charged in full either way. Zeroing it for resident models would be
    the opposite of the old error, not a correction to it.

    Known under-quote, MULTI-TURN. ``completions`` is the episode count, but a multi-turn env issues
    one vLLM ``generate`` per assistant turn (``worker/grpo_multiturn.py``), so turns 2..N escape the
    slope and a long-episode run is under-quoted by roughly (turns - 1) x slope x completions. It is
    not fixed here because the turn shape does not reach this layer: ``max_turns`` is read on the
    worker via ``getattr(env, "max_turns", ...)`` off the user's loaded environment object, and
    neither it nor ``multi_turn`` exists on ``RunConfig`` or in the spec. A blanket multiplier was
    rejected as the wrong trade -- it would inflate every single-turn run, which is the entire fitted
    corpus, to hedge a shape this layer cannot observe. Threading the turn shape into the pricing
    config is the fix, and it is a spec-path change.
    """
    return ROLLOUT_SECONDS_PER_COMPLETION * max(0, completions)


def run_block_seconds(name: str, resident: bool = False) -> float:
    """One-time rollout setup inside the billed train wall on ``name`` (0 for a resident engine).

    ``resident=True`` zeroes this for a model the catalog flags ``sleep_unsupported``. Those pin the
    rollout engine resident instead of sleeping it (catalog.py, backend_common.py), so the engine
    build and teardown this block is mostly made of never runs. Charging it anyway extrapolates well
    past the fit: every fitted arm is a sleep-capable 0.8B/2B/4B model, so a resident rollout has
    never been measured.

    Being exact about what zero gives up: the block is engine build AND the first actor->rollout
    weight sync, and that sync still happens on a resident engine -- the resident overrides touch
    only ``free_cache_engine``/``enable_sleep_mode``, both sleep knobs, and an on-policy run must
    push the policy into vLLM either way. So zero under-quotes by one sync. The block is NOT split,
    because the corpus cannot split it: every fitted arm ran sleep-enabled, so no measurement
    separates the two components, and a guessed split ratio would be a fabricated number wearing a
    measured one's clothes. Zero is the bounded, honest error: a colocate sync is a same-device
    weight copy (~70GB bf16 for the 35B, tens of ms at HBM bandwidth plus collective overhead),
    small against a 250-760s block, and it errs toward under-quoting one model rather than
    over-quoting it by a whole engine build. Replace this with a measured resident block once a
    resident arm is calibrated.
    """
    if resident:
        return 0.0
    return _RUN_BLOCK_S.get(name, _DEFAULT_RUN_BLOCK_S)


def gpu_hourly_usd(
    name: str,
    provider: str | None = None,
    max_wall_seconds: float = 0.0,
    min_vram_gb: int = 0,
    gpu_type: str = "",
) -> float:
    """Representative $/hr for a class, on ``provider`` when given.

    When ``provider`` is ``lambda`` or ``vast`` and the class is offered there, price it through that
    provider's pricing module (live with a static fallback); otherwise use the RunPod static rate.

    ``max_wall_seconds`` (>0) is threaded into the Vast live market so a duration-bound quote prices
    against offers that outlast the run, not a short-lived one filtered out at launch.

    ``min_vram_gb`` (>0) floors the Vast market search at the job's required VRAM — the SAME floor
    ``pick_gpu`` selected under — so a high-VRAM class isn't crowded off the price-sorted page and
    misquoted on the static fallback (selection/quote parity).
    """
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    p = (provider or "").strip().lower()
    if p == "lambda" and info.lambda_name:
        from flash.providers import get_provider

        return get_provider("lambda").hourly_rate(name)
    if p == "vast" and info.vast_name:
        # Vast is a live market whose rates differ materially from RunPod's static ones, so price a
        # provider="vast" quote through the Vast pricing module (live + static fallback).
        from flash.providers.vast.pricing import hourly_rate

        return hourly_rate(
            name,
            max_wall_seconds=max_wall_seconds,
            min_vram_gb=min_vram_gb,
            gpu_type=gpu_type,
        )
    return info.hourly_usd


def gpu_vram_gb(name: str) -> int:
    info = GPU_INFO.get(name)
    if info is None:
        raise KeyError(f"unknown GPU class {name!r}")
    return info.vram_gb


def pick_gpu(
    required_vram_gb: int,
    *,
    provider: str | None = None,
    max_wall_seconds: float = 0.0,
    cost_key=None,
) -> str:
    """Cheapest GPU class that fits ``required_vram_gb``.

    No pin; every fitting class is eligible, validated or not. NOTE this is intentionally
    gate-free: the submit-time allocator restricts to the validated pool, so the
    actually-provisioned class can be pricier than the one priced here. ``provider`` restricts
    candidates to what it can provision. ``max_wall_seconds`` (>0) prices the Vast market against
    offers that outlast the run, so a long-run quote doesn't SELECT a class on the strength of a
    short-lived offer that won't survive to launch.

    ``cost_key`` is ``(gpu_name, hourly_rate) -> comparable``: pass it to rank on what the JOB
    costs rather than what the CARD costs, so a faster class wins when it finishes soon enough to
    pay for its higher rate. It is injected rather than imported because the step model lives in
    ``analytical``, which imports this module -- building the key there keeps the dependency
    one-way. Omitted (the default) ranks on $/hr, which is correct for callers that have no run to
    price and is the honest fallback for a model outside the cost catalog.
    """

    def _selectable(g: GpuClass) -> bool:
        return provider in (None, "auto") or provider in providers_for(g.name)

    candidates = [g for g in GPU_INFO.values() if g.vram_gb >= required_vram_gb and _selectable(g)]
    if not candidates:
        raise ValueError(f"no GPU class fits >= {required_vram_gb} GB")
    # Rank by the rate on the REQUESTED provider, not the RunPod nominal. For Vast, fetch the live offer
    # map ONCE (a duration-bound query bypasses the per-call cache, so pricing per candidate would fire N
    # identical market fetches) and, when reachable, restrict to classes that actually have a rentable
    # offer under the wall cap and rank by LIVE price — else a cheaper class with no surviving offer gets
    # selected/quoted on a static rate the launch path would never rent. Static fallback when offline.
    if (provider or "").strip().lower() == "vast":
        from flash.providers.vast.pricing import live_offer_rates

        live = live_offer_rates(max_wall_seconds=max_wall_seconds, min_vram_gb=required_vram_gb)
        rentable = [g for g in candidates if g.name in live] if live else []
        if rentable:
            candidates = rentable

            def _rate(g: GpuClass) -> float:
                return live[g.name]
        else:

            def _rate(g: GpuClass) -> float:
                return g.hourly_usd
    else:

        def _rate(g: GpuClass) -> float:
            return gpu_hourly_usd(g.name, provider=provider, max_wall_seconds=max_wall_seconds)

    if cost_key is not None:
        # rank on job cost, tie-breaking on rate then the same vram/name order as the $/hr path so
        # the two bases stay comparable when a run is unpriceable in only one of them.
        best = min(
            candidates, key=lambda g: (cost_key(g.name, _rate(g)), _rate(g), g.vram_gb, g.name)
        )
    else:
        best = min(candidates, key=lambda g: (_rate(g), g.vram_gb, g.name))
    return best.name


def _catalog_model_info(model_id: str) -> ModelInfo:
    info = MODELS.get(model_id)
    if info is None:
        raise ValueError(
            f"unknown model {model_id!r}; cost estimation supports catalog models only "
            f"({', '.join(MODELS)})"
        )
    return info


def total_params_b(model_id: str, revision: str = "") -> float:
    """Total parameter count (billions) for a catalog model.

    when a revision is pinned, size the pinned commit (validated against the catalog, fail-closed)
    so setup/save cost tracks the weights the worker actually loads, not the default-revision count.
    """
    info = _catalog_model_info(model_id)
    if revision:
        from flash.engine.vram import _validated_revision_geometry

        params_b, _vocab = _validated_revision_geometry(model_id, revision, info)
        return params_b
    return info.params_b


def active_params_b(model_id: str) -> float:
    """Active params per token (billions); falls back to total for dense models. Use for FLOPs, not VRAM."""
    info = _catalog_model_info(model_id)
    return info.active_params_b or info.params_b


def model_quant(model_id: str) -> str:
    """Quantization of the catalog entry; defaults to 'bf16'."""
    info = MODELS.get(model_id)
    return (info.quant or "bf16") if info is not None else "bf16"


def download_weight_gb(model_id: str, revision: str = "") -> float:
    """Full bf16 checkpoint size in GB (2 bytes/param)."""
    return total_params_b(model_id, revision) * 2.0


# Per-completion GRADING latency only -- the wall spent inside the env's scorer, nothing else.
#
# This was 1.0s, chosen as a mid-range across grader types (regex ~0.01s to LLM judge ~3s). That
# value was doing a second, undeclared job: it was the only per-completion term in the model, so it
# also stood in for rollout cost, which is ~0.81s/completion (see ROLLOUT_SECONDS_PER_COMPLETION).
# The two are now separate, and leaving this at 1.0 would charge that rollout cost twice -- it
# over-quotes the 56-arm corpus at geo-bias 1.462x, with only 26/56 arms in band.
#
# 0.05s is a grading-only default. It is NOT the corpus minimum: every env measured here is a fast
# programmatic grader (~0.0003s), and fitting to that would publish "grading is free". It sits two
# orders above the measured programmatic graders and two below a judge.
#
# Known gap, stated rather than papered over: no submit-time path passes an override (see the note
# above), so an llm-judge env at ~3s/completion is quoted at 0.05 and under-priced by ~1510s per
# step at the default 64x8 shape. Restoring 1.0 does not fix that -- it still under-prices the same
# judge by ~1024s -- while re-charging rollout cost twice on every programmatic env, which is what
# put the corpus at 1.462x with only 26/56 in band. The real fix is to feed reward_profile's
# measurement into the quote; that crosses into the submit path and belongs in its own change.
AVG_REWARD_SECONDS_PER_COMPLETION = 0.05


def reward_seconds_per_completion(override: float | None = None) -> float:
    """Per-completion reward latency (s): the explicit override, else the single average."""
    if override is not None:
        return max(0.0, override)
    return AVG_REWARD_SECONDS_PER_COMPLETION


# Fireworks echo-scoring round-trip per completion (wall time, concurrency-bound like reward grading).
AVG_TEACHER_SECONDS_PER_COMPLETION = 2.0


def teacher_price_per_1m(teacher_model: str) -> tuple[float, float]:
    """(input, output) $/1M tokens for a teacher model.

    Routes through resolve_teacher, the single OPD-teacher resolver, whose recipe.TEACHER_MODELS is
    the one source of teacher prices. ``teacher_model`` is the Fireworks model id chosen via ``[train]
    teacher_model``, or "" for the default GLM 5.2 teacher. Teacher pricing is static, offline, and
    credential-free. OPD echo-scores completions (max_tokens=0), so only the input column is billed,
    but both are returned. An unsupported value falls back defensively to the default rate."""
    from flash.engine.recipe import resolve_teacher

    try:
        return resolve_teacher(teacher_model).usd_per_1m
    except ValueError:
        return resolve_teacher("").usd_per_1m


def teacher_token_cost_usd(
    input_tokens: float, output_tokens: float = 0.0, teacher_model: str = ""
) -> float:
    """External teacher-API dollar cost for a token count. Deterministic; no network."""
    inp, outp = teacher_price_per_1m(teacher_model)
    return (max(0.0, input_tokens) * inp + max(0.0, output_tokens) * outp) / 1_000_000.0


def teacher_seconds_per_completion(override: float | None = None) -> float:
    """Per-completion teacher-scoring latency (s): the explicit override, else the average."""
    if override is not None:
        return max(0.0, override)
    return AVG_TEACHER_SECONDS_PER_COMPLETION
