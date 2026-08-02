"""The analytical cost model: total = training-only GPU hours x GPU $/hr.

Elapsed wall clock still includes cold-start setup + steps x per-step time, but setup/cold-start
is reported as non-billable. GRPO splits each step into a vLLM rollout + reward grading +
policy/reference update.
"""

from __future__ import annotations

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
    step_floor_seconds,
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
        # per-completion sequences, so both the constant and the slope apply. that is the mechanism
        # the wall measures, but note every arm it was FITTED on is grpo, so the size of the opd wall
        # is inferred from a shared mechanism rather than measured on opd arms.
        #
        # opd honours the resident flag for the same reason grpo does: opd_train pins the rollout
        # resident off the same catalog flag, so the engine cycle is absent on both drivers.
        return gen_s + update_s, overhead + teacher_s + step_floor_seconds(
            gpu, completions, resident=rollout_is_resident(n.model_id)
        )

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
        # slope 1.021 -> 1.992 s/step and the intercept 70.9s -> 26.6s. the data is non-monotonic in
        # steps -- 2 and 32 steps both took 96.2s, 128 steps took MORE wall than 256 -- so pod
        # variance currently exceeds the step-count signal. replicates are needed before any
        # coefficient here changes. the underquote is real; its magnitude is not yet known.
        flops = SFT_FLOPS_PER_TOKEN_PER_PARAM * params * (n.batch_size * n.seq_len)
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
    # conflated before: the rollout slope (0.81s) is close enough to the reward default (1.0s) that
    # one term could stand in for both, right up until a caller passed a MEASURED reward latency --
    # real graders here are ~0.0003s -- and the rollout cost vanished with it. see facts.py.
    #
    # a sleep_unsupported model keeps its rollout engine RESIDENT, so it never pays the entry/exit
    # cycle the constant is made of -- see step_floor_seconds. the slope still applies.
    return gen_s + update_s, overhead + reward_s + step_floor_seconds(
        gpu, completions, resident=rollout_is_resident(n.model_id)
    )


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
    sps = seconds_per_step(config, gpu)
    required_save_s = required_save_overhead_seconds(config)
    # A one-time kernel/graph compile is paid once on the first step (MoE-only; 0 for dense). It is
    # training GPU time, so it belongs in the (billed) train term, not setup.
    compile_s = compile_seconds(config, gpu)
    raw_train = compile_s + config.steps * sps + required_save_s
    if not config.is_grpo and config.train_tokens is not None:
        raw_train = (
            compile_s + sft_seconds_for_tokens(config, gpu, config.train_tokens) + required_save_s
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
