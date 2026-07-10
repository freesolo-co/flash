"""On-policy distillation training path (algorithm="opd") for the fine-tuning worker.

The student (a Qwen3.5 / MiniCPM catalog model) samples completions on-policy; a Fireworks-hosted
GLM-5.2 teacher scores those completions token-by-token over the API; a groupwise reverse-KL loss is
backpropagated through the student LoRA.

Cross-tokenizer bridge (teacher GLM vocab != student Qwen3.5 / MiniCPM vocab): the teacher and student
tokenize the same completion string differently, so their per-token distributions can't be compared
directly. We align by SHARED DECODED-TEXT SPANS — the coarsest common refinement of the two
tokenizations (a group boundary is any character offset that begins a token in BOTH tokenizers) — and
apply reverse-KL per span using only the REALIZED-token logprobs on each side (the collinear-ai
*spider* / Tinker method). No top-k candidates, no surface->vocab projection, so it is exact across
arbitrary tokenizer mismatch and covers every student token. See ``tokenizer_align``.

There is NO local reference model: the teacher lives behind the API. Sampling uses a colocated vLLM
engine loaded with the student LoRA and refreshed after each optimizer step, matching GRPO's rollout
shape. All heavy imports (torch/transformers/peft/vllm) are inside functions, so importing this module
is CPU/offline-safe.
"""

from __future__ import annotations

import contextlib
import os
import random
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed

from flash.engine.chalk_kernels import active_kernels, install_chalk_kernels
from flash.engine.recipe import RECIPE
from flash.engine.steps import on_policy_steps
from flash.engine.structured_outputs import describe_structured_outputs, parse_structured_outputs
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.opd_gkd import (
    _generation_eos_ids,
    _teacher_prompt_text,
    student_tokens_with_offsets,
)
from flash.engine.worker.opd_gkd import (
    _rollout_terminated as _rollout_terminated,
)
from flash.engine.worker.opd_gkd import (
    _trim_trailing_stop as _trim_trailing_stop,
)
from flash.engine.worker.opd_knobs import OpdKnobs as OpdKnobs
from flash.engine.worker.opd_knobs import _resolve_opd_knobs
from flash.engine.worker.opd_loss import (
    _EOS_RUNAWAY_HI as _EOS_RUNAWAY_HI,
)
from flash.engine.worker.opd_loss import (
    _EOS_RUNAWAY_LO as _EOS_RUNAWAY_LO,
)
from flash.engine.worker.opd_loss import (
    _EOS_TRUNC_EMA_DECAY,
    _bump_model_counter,
    _drop_fully_forced_groups,
    _eos_reinforce_term,
    _forward_logits,
    _gkd_loss_from_logits_rows,
    _prepare_gkd_groups,
    _PreparedLoss,
    _primary_eos_id,
    _runaway_eos_scale,
)
from flash.engine.worker.opd_loss import (
    _gkd_loss_from_logps as _gkd_loss_from_logps,
)
from flash.engine.worker.opd_loss import (
    _PreparedGkdGroups as _PreparedGkdGroups,
)
from flash.engine.worker.opd_publish import _publish_opd_deployable, _save_adapter
from flash.engine.worker.opd_rollout import (
    SampleResult,
    _generate_many_vllm,
    _GenResult,
    _Pending,
    _resolve_no_loss_sample,
    _sample_skip_reason,
)
from flash.engine.worker.opd_rollout import (
    _gen_from_vllm_output as _gen_from_vllm_output,
)
from flash.engine.worker.opd_scoring import _score_one, _ScoreResult
from flash.engine.worker.opd_setup import (
    _format_skip_counts,
    _opd_progress,
    _student_model,
    _thinking_prefill_text,
)
from flash.engine.worker.opd_sizing import (
    OPD_LOSS_MICROBATCH_SIZE as OPD_LOSS_MICROBATCH_SIZE,
)
from flash.engine.worker.opd_sizing import (
    OPD_ROLLOUT_PIPELINE_MAX_CHUNKS as OPD_ROLLOUT_PIPELINE_MAX_CHUNKS,
)
from flash.engine.worker.opd_sizing import (
    OPD_ROLLOUT_PIPELINE_TARGET_CHUNK_SIZE as OPD_ROLLOUT_PIPELINE_TARGET_CHUNK_SIZE,
)
from flash.engine.worker.opd_sizing import (
    OPD_TEACHER_BATCH_SIZE as OPD_TEACHER_BATCH_SIZE,
)
from flash.engine.worker.opd_sizing import (
    _opd_loss_microbatch_size,
    _opd_rollout_chunk_size,
    _opd_rollout_pipeline_chunks,
    _opd_rollout_pipeline_max_chunks,
    _opd_rollout_pipeline_target_chunk_size,
    _opd_teacher_batch_size,
    _opd_teacher_workers,
)
from flash.engine.worker.opd_vllm import OpdVllmOutput as OpdVllmOutput
from flash.engine.worker.opd_vllm import OpdVllmRolloutEngine
from flash.engine.worker.opd_vllm import opd_lora_rank as _opd_lora_rank
from flash.engine.worker.opd_vllm import opd_vllm_kwargs as _opd_vllm_kwargs
from flash.engine.worker.perf import (
    RetriableInfraError,
    _sdpa_cudnn_ctx,
    free_gpu,
    gpu_diagnostics,
    grad_checkpointing_on,
    grpo_use_reentrant,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)
from flash.engine.worker.teacher import TeacherError
from flash.engine.worker.tokenizer_align import groupwise_alignment, groupwise_coverage


def run_opd():
    import torch
    from transformers import AutoTokenizer

    from flash.engine.worker.teacher import TeacherClient

    env = _w.require_active_env()
    if getattr(env, "is_tool_env", False):
        # Tool envs need TRL's native tool-call loop (rl.py hands the tool schemas + callables to the
        # trainer); opd's owned vLLM rollout loop can't drive that, so it stays unsupported. Pure
        # multi-turn (episode) envs ARE supported below via the per-turn rollout path.
        raise RuntimeError(
            "opd does not support tool-calling environments yet: it cannot drive TRL's tool-call "
            "loop. Use grpo for a tool env, or a single-turn / pure multi-turn env for opd."
        )
    # Multi-turn (episode) envs distil EACH assistant turn as an independent on-policy sample
    # conditioned on the transcript so far (see _run_multi_turn_step / rollout_one_records). A
    # single-turn env keeps the original one-generate-per-prompt path.
    multi_turn = bool(getattr(env, "multi_turn", False))
    t_start = time.time()
    _w.heartbeat("opd_start", gpu=gpu_diagnostics())
    knobs = _resolve_opd_knobs()
    warm_start = _w.JOB_SPEC.train.init_from_adapter if _w.JOB_SPEC else ""
    print(
        f"[opd] gkd (groupwise reverse-KL) teacher={knobs.teacher_model} "
        f"epochs={knobs.epochs} warm_start={warm_start or 'none'} "
        f"mode={'multi-turn' if multi_turn else 'single-turn'} "
        f"eos_loss_coef={knobs.eos_loss_coef}"
    )

    # The GLM teacher key is a platform-owned credential the control plane injects into the worker
    # env (like HF_TOKEN); users never supply it. Read it like any other flash-used key.
    api_key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "no FIREWORKS_API_KEY (the GLM teacher key) in the opd worker env. It is platform-"
            "managed and injected by the control plane, so this means the deployment has no "
            "FIREWORKS_API_KEY configured in its environment."
        )
    teacher = TeacherClient(api_key, knobs.teacher_base_url, knobs.teacher_model)

    wait_for_gpu(_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None)
    setup_perf_backends()
    model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else RECIPE.hf_model_id
    # Tokenizer only (a few small files) up front -- the prompt-budget filter below needs it. The FULL
    # base-weight prefetch (tens of GB) is deferred until AFTER the filter confirms a non-empty pool, so
    # a dataset whose every prompt is over-budget fails fast without paying for the download (codex[bot]).
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Thinking-mode student prompts open a reasoning block (e.g. Qwen's <think>) after the generation
    # prompt; the teacher must condition on the SAME prefill or its logprobs score a different prefix
    # than the sampled tokens (see _thinking_prefill_text). Compute once — the template is fixed.
    thinking_prefill = _thinking_prefill_text(tok)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _attn = optimal_attn_impl()
    mik = {"dtype": torch.bfloat16}
    if _attn:
        mik["attn_implementation"] = _attn
    # --- Build the on-policy prompt pool BEFORE loading the student ------------------------------
    # Fetch the dataset, seed the RNGs, and pre-filter to the prompts that fit the context budget
    # HERE — ahead of _student_model, which for a VL warm-start downloads the base and MERGES the SFT
    # into it. A dataset whose every prompt is over-budget is a deterministic failure; detecting it
    # now fails fast, before a paid worker pays for the base download + SFT merge (only tok/env/knobs
    # are needed, none of which depend on the loaded model).
    train = env.dataset()
    if not train:
        raise RuntimeError(
            "opd: the environment dataset is empty — no prompts to sample on-policy. Check the "
            "environment's dataset()/train split before provisioning a GPU."
        )
    _max_examples = getattr(_w.JOB_SPEC.train, "max_examples", None) if _w.JOB_SPEC else None
    max_examples = int(_max_examples or 0) if _max_examples is not None else 0
    if max_examples > 0:
        train = train[:max_examples]
    rng = random.Random(_w.SEED)
    rng.shuffle(train)
    ppl_step = knobs.prompts_per_step
    group = knobs.group_size
    # Prompt budget mirrors GRPO: DROP (not truncate) prompts over the context budget, so the student
    # never conditions on a truncated prompt the teacher didn't see. Use the configured context
    # budget when set, else the recipe prompt cap.
    if knobs.max_length:
        prompt_budget = knobs.max_length - knobs.max_completion
        if prompt_budget < 1:
            # A non-positive remainder means the total context budget is no larger than the
            # completion budget: there is no room for any prompt, so every sample would run
            # generate+loss past the configured context. Reject loudly instead of clamping to a
            # 1-token budget that silently admits over-budget runs.
            raise RuntimeError(
                f"opd: [train] max_context_tokens ({knobs.max_length}) leaves no prompt budget "
                f"after max_completion_tokens ({knobs.max_completion}); set "
                f"max_context_tokens > max_completion_tokens."
            )
    else:
        prompt_budget = RECIPE.opd.max_prompt_len

    def _render_prompt_ids(messages):
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=_w.THINKING
        )
        return tok(text, add_special_tokens=False).input_ids

    # Build the on-policy pool from ONLY the prompts that fit the budget (GRPO-style pre-filter). The
    # step loop visits a deterministic examples[(step*ppl_step+i) % len] slice, so a whole-dataset
    # `any(fits)` precheck could pass while every prompt in the visited slice is over-budget and
    # dropped — burning the GPU allocation with no trained step. Filtering the pool here (before the
    # student is even loaded) guarantees every visited prompt fits and fails fast otherwise. One
    # tokenize per prompt, once, at startup.
    # Rendering+tokenizing a large train split runs AFTER the last setup heartbeat and BEFORE the
    # model-load liveness starts. Pollers reset setup grace only on NON-liveness (progress) heartbeats,
    # so a healthy worker scanning a big split could exceed the grace and be reaped as stalled
    # (codex[bot]). Drive a real progress heartbeat off a monotonic scan counter while filtering.
    _scanned = 0
    with liveness_heartbeat("opd_filtering_prompts", progress=lambda: _scanned):
        examples = []
        for ex in train:
            # Render ONCE here and CACHE (messages + ids) alongside ex. env.prompt_messages can be
            # stateful/randomized, so re-rendering at train time could yield a DIFFERENT prompt than the
            # one admitted by this budget filter — an over-budget re-render would then be dropped, so a
            # pool that PASSED this filter could still yield no usable samples after paying for GPU/model
            # setup. Reusing this exact render below guarantees every visited prompt fits (codex[bot]).
            msgs = env.prompt_messages(ex)
            ids = _render_prompt_ids(msgs)
            if len(ids) <= prompt_budget:
                examples.append((ex, msgs, ids))
            _scanned += 1
    n_over_budget = len(train) - len(examples)
    if not examples:
        raise RuntimeError(
            f"opd: every prompt exceeds the {prompt_budget}-token budget "
            f"(max_context_tokens={knobs.max_length or 'unset'}, "
            f"max_completion={knobs.max_completion}). Raise [train].max_context_tokens or shorten "
            "prompts — failing before the training loop instead of dropping every prompt for "
            "every step and burning the GPU allocation."
        )
    if n_over_budget:
        print(
            f"[opd] filtered {n_over_budget}/{len(train)} prompts over the "
            f"{prompt_budget}-token budget; pool = {len(examples)}"
        )
    if ppl_step > len(examples):
        print(
            f"[opd] lowering prompts_per_step from {ppl_step} to {len(examples)}: "
            "only that many prompt(s) fit after filtering"
        )
        ppl_step = len(examples)
    steps = on_policy_steps(
        epochs=knobs.epochs,
        prompt_count=len(examples),
        prompts_per_step=ppl_step,
    )
    print(
        f"[opd] epochs={knobs.epochs} over {len(examples)} retained prompt(s) at "
        f"{ppl_step} prompts/step -> steps={steps}"
    )

    # Now that a non-empty on-policy pool is confirmed, prefetch the full base weights (deferred from
    # setup so an all-over-budget dataset fails before this download). Still inside the setup phase
    # (opt_steps==0 -> wide poller grace), same as when it ran earlier.
    download_seconds = _w.prefetch_model(model_id)

    # Seed torch/CUDA BEFORE constructing the student LoRA: get_peft_model samples the LoRA A matrix
    # (init_lora_weights=True) from the torch default generator, so seeding must precede _student_model
    # for the fixed Flash seed to reproduce the same adapter init run-to-run (the fresh-LoRA and VL
    # warm-start paths both build a fresh LoRA). The colocated vLLM rollout engine receives the same
    # seed below. The prompt shuffle above uses a SEPARATE random.Random(_w.SEED), so its ordering is
    # unaffected by where torch is seeded.
    torch.manual_seed(_w.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_w.SEED)

    setup_seconds = time.time() - t_start
    _w.heartbeat("opd_model_load", setup_seconds=setup_seconds, gpu=gpu_diagnostics())
    with liveness_heartbeat("opd_initializing"):
        model, rollout_model_source = _student_model(model_id, mik, device)
        # Apply chalk standalone kernels to the student, exactly as sft/rl do after building their
        # trainer. The HF/PEFT student remains the GKD loss target, so without this the default
        # Qwen3.5/3.6 catalog model silently falls back to eager GDN/RMSNorm/RoPE/LoRA kernels and a
        # long distillation runs much slower than the rest of the stack (codex[bot]).
        # No-op ({}) when freesolo-chalk isn't installed or the arch is unsupported.
        _chalk_report = install_chalk_kernels(model)
        _chalk_active = active_kernels(_chalk_report)
        if _chalk_active:
            print(f"[opd] chalk kernels active: {', '.join(_chalk_active)}")
        # Engine length gates whether gradient checkpointing is needed for the loss forward.
        seq_cap = knobs.max_length or (RECIPE.opd.max_prompt_len + knobs.max_completion)
        if grad_checkpointing_on(model_id, seq_cap):
            # GDN/MoE models MUST use reentrant recompute (parity with sft.py / rl.py). The default
            # non-reentrant path asserts recomputed-activation metadata equality and dies on the FIRST
            # backward for GatedDeltaNet (the fused chunk-scan + chalk Triton kernels save shape-/data-
            # dependent tensors laid out differently on recompute) -> torch.utils.checkpoint
            # CheckpointError; its fragmenting alloc pattern also OOMs under non-expandable segments.
            # See grpo_use_reentrant (#429/#432 fixed this for GRPO; OPD's custom loop never got it).
            _reentrant = grpo_use_reentrant(model_id)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": _reentrant}
            )
            model.enable_input_require_grads()
            print(f"[opd] gradient checkpointing enabled (use_reentrant={_reentrant})")
    # The HF/PEFT model only handles differentiable loss forwards; the colocated vLLM engine owns
    # KV-cached student rollout generation.
    model.config.use_cache = False

    # vLLM's EngineCore starts in a separate process and cannot reuse CUDA blocks PyTorch is only
    # caching. Release load/merge leftovers before sizing and launching the rollout engine; otherwise
    # large warm-started OPD jobs can fail with an opaque EngineCore startup error while torch reports
    # tens of GiB reserved but unallocated.
    free_gpu()

    vllm_kwargs = _opd_vllm_kwargs(model_id, knobs, seq_cap)
    lora_rank = _opd_lora_rank(
        model, getattr(_w.JOB_SPEC.train, "lora_rank", 32) if _w.JOB_SPEC else 32
    )
    print(
        f"[opd] rollout backend: colocated vLLM model={rollout_model_source} "
        f"ctx={seq_cap} lora_rank={lora_rank} util={vllm_kwargs['gpu_memory_utilization']:.2f} "
        f"rollout_batch={vllm_kwargs.get('rollout_batch_size') or 'auto'}"
    )
    _so_spec = parse_structured_outputs(knobs.structured_outputs)
    if _so_spec:
        print(
            f"[opd] structured outputs: every student rollout constrained to "
            f"{describe_structured_outputs(_so_spec)}"
        )
    t_vllm_init = time.time()
    with liveness_heartbeat("opd_vllm_initializing"):
        vllm_rollout = OpdVllmRolloutEngine(
            model_source=rollout_model_source,
            max_model_len=seq_cap,
            temperature=knobs.temperature,
            top_p=knobs.top_p,
            stop_sequences=tuple(str(s) for s in knobs.stop_sequences),
            structured_outputs=_so_spec,
            lora_rank=lora_rank,
            seed=_w.SEED,
            **vllm_kwargs,
        )
        vllm_rollout.sync_from_model(model)
    vllm_init_seconds = time.time() - t_vllm_init
    print(f"[opd] vLLM rollout initialized in {vllm_init_seconds:.1f}s")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=knobs.learning_rate
    )

    # Initialize W&B if configured ([wandb] config + WANDB_API_KEY). wandb_report_to() CREATES the run
    # as a side effect -- sft/rl get this for free by passing report_to into the HF Trainer, but opd's
    # custom loop has no Trainer, so without an explicit init here wandb.run stays None: no dashboard,
    # and the wandb_run_info() already threaded into train_meta below returns {} (codex[bot]). We log
    # loss/coverage per optimizer step; the worker exit path (__init__.py) calls wandb_finish.
    _wandb_on = bool(_w.wandb_report_to())

    resume_ckpt = _w.hf_resume_checkpoint()
    if resume_ckpt:
        print("[opd] resume-from-checkpoint is not yet supported for opd; starting fresh")

    out_dir = f"/tmp/opd_seed{_w.SEED}"
    adapter_dir = f"{out_dir}/adapter"
    os.makedirs(out_dir, exist_ok=True)

    loss_curve: list[float] = []
    coverage_curve: list[float] = []
    generated_tokens = 0
    teacher_input_tokens = 0
    # Rollouts skipped for hitting the cap without EOS (mid-output truncations we refuse to distil),
    # and a running mean of alignment granularity — both surfaced in train_meta for diagnosis.
    truncated_rollouts = 0
    # EMA of the per-rollout truncation rate — the runaway signal that scales terminal-EOS reinforcement
    # (see _runaway_eos_scale). Starts at 0 (assume the student stops fine); rises if rollouts run away.
    trunc_ema = 0.0
    granularity_sum = 0.0
    granularity_n = 0
    # Running mean of the reinforced terminal log P(eos): should RISE over a run as the student learns
    # to stop, the counterpart to truncated_rollouts FALLING. Surfaced in train_meta.
    eos_logprob_sum = 0.0
    eos_logprob_n = 0
    opt_steps = (
        0  # optimizer steps actually applied (< steps if any iteration had no teacher signal)
    )
    _reset_peak = getattr(_w, "_reset_peak_gpu", None)
    if _reset_peak:
        _reset_peak()

    # Monotonic count of student samples ATTEMPTED (generate+score), across all steps. Fed to
    # liveness_heartbeat as its progress signal so the liveness thread emits a REAL (non-liveness)
    # opd_step heartbeat whenever a sample advances — parity with sft/rl. Without a progress callback
    # opd's liveness thread emitted only liveness=True pings that share the opd_step upload-throttle
    # slot and could suppress the main thread's per-sample progress uploads (codex[bot]).
    samples_seen = 0
    # No-signal accounting: distinguish a run that trained nothing because the TEACHER was down
    # (transient) from one where scoring succeeded but never aligned — the former is retriable infra.
    teacher_ok = 0
    teacher_transient = 0
    teacher_error = 0
    skip_counts: Counter[str] = Counter()
    no_signal_resamples = 0
    no_signal_skipped_steps = 0

    # OPD rollouts always use the colocated vLLM engine. The HF model remains the authoritative
    # training target for the GKD loss forward/backward; vLLM only samples the current LoRA.

    # Multi-turn per-turn rollout wiring (built once; unused for single-turn). Each episode is driven
    # by rollout_one_records, which reuses the SAME env turn loop + tokenizer-sensitive glue/seam logic
    # as GRPO's rollout_one (flash/engine/multiturn_rollout.py) but yields per-turn records instead of a
    # flat masked sequence. engine_max_len bounds the transcript to the same seq cap the VRAM estimate
    # and grad-checkpointing gate were sized for (seq_cap above), so every per-turn loss forward
    # (prompt+turn) fits; per_turn_max_tokens caps a single turn.
    mt_max_turns = 0
    mt_engine_max_len = 0
    generate_fn = env_glue_fn = render_fn = None
    episodes_seen = 0
    mt_turn_records = 0  # total per-turn records produced across all episodes (multi-turn only)
    if multi_turn:
        from flash.engine.multiturn_rollout import (
            make_env_glue,
            render_message_ids,
            rollout_one_records,
        )

        mt_max_turns = int(getattr(env, "max_turns", 10) or 10)
        mt_engine_max_len = knobs.max_length or (RECIPE.opd.max_prompt_len + knobs.max_completion)
        env_glue_fn = make_env_glue(tok, thinking=_w.THINKING)

        def render_fn(messages, add_generation_prompt):
            return render_message_ids(tok, messages, add_generation_prompt, thinking=_w.THINKING)

        def generate_fn(prefix_ids, max_new):
            """One turn's generation through the colocated vLLM engine."""
            return _generate_many_vllm(
                vllm_rollout, tok, [prefix_ids], knobs, max_tokens=max(1, int(max_new))
            )[0]

    def _account(r, *, backward: bool = True):
        """Consume one sample's SampleResult: tally teacher health / truncations, backprop a distilled
        sample (scaled 1/accum_target for gradient accumulation), advance the step aggregates, and
        refresh the stall clock. Called after vLLM generation and teacher scoring; ``samples_seen`` is
        advanced by the caller once per generated rollout so its timing is unchanged."""
        nonlocal teacher_ok, teacher_transient, teacher_error, truncated_rollouts, step_loss, step_cov
        nonlocal granularity_sum, granularity_n, generated_tokens, teacher_input_tokens, nseq
        nonlocal eos_logprob_sum, eos_logprob_n, trunc_ema
        if r.teacher_status == "ok":
            teacher_ok += 1
        elif r.teacher_status == "transient":
            teacher_transient += 1
        elif r.teacher_status == "error":
            teacher_error += 1
        if r.truncated:
            truncated_rollouts += 1
        # Update the runaway EMA once per accounted rollout (truncated or not) so terminal-EOS
        # reinforcement tracks the student's live termination health (see _runaway_eos_scale).
        trunc_ema += (1.0 - _EOS_TRUNC_EMA_DECAY) * ((1.0 if r.truncated else 0.0) - trunc_ema)
        if r.loss is None:
            reason = _sample_skip_reason(r)
            skip_counts[reason] += 1
            step_skip_counts[reason] += 1
            # Refresh the stall clock even when a sample yields no teacher signal (rationale in
            # _opd_progress) — else an all-skip stretch emits only ignored liveness pings.
            _opd_progress(opt_steps, nseq)
            return
        if backward:
            (r.loss / accum_target).backward()
        step_loss += float(r.loss.detach())
        step_cov += r.coverage
        granularity_sum += r.group_granularity
        granularity_n += 1
        generated_tokens += r.gen_tokens
        teacher_input_tokens += r.teacher_tokens
        if r.eos_logprob is not None:
            eos_logprob_sum += r.eos_logprob
            eos_logprob_n += 1
        nseq += 1
        # Non-liveness progress ping WITHIN the step (rationale in _opd_progress).
        _opd_progress(opt_steps, nseq)

    t_train = time.time()
    opd_phase_seconds: Counter[str] = Counter()
    opd_phase_counts: Counter[str] = Counter()
    # Drive the loop by optimizer UPDATES, not raw iterations. A no-signal iteration (empty
    # completions / a flaky teacher) skips optimizer.step() below, so `for step in range(steps)` could
    # exit with opt_steps < steps -- shipping an under-trained adapter as the served DEFAULT while the
    # run is billed the full submit-time `steps` quote (codex[bot]). Loop until `steps` real updates
    # land. A no-signal attempt is retried a few times with a fresh rollout/data slice before the
    # optimizer step is abandoned, so sporadic empty teacher responses or degenerate samples do not
    # waste a requested optimizer update. Bound total attempts so a persistently degraded teacher cannot
    # spin unboundedly -- the post-loop guard then turns a shortfall into a RETRY, not a silent
    # under-trained publish.
    step = 0
    max_iters = 3 * steps + 10
    max_no_signal_attempts = 3
    teacher_batch_size = _opd_teacher_batch_size(knobs.prompts_per_step * knobs.group_size)
    loss_microbatch_size = _opd_loss_microbatch_size(
        model_id, knobs.prompts_per_step * knobs.group_size
    )
    max_teacher_workers = _opd_teacher_workers(
        knobs.prompts_per_step * knobs.group_size, teacher_batch_size
    )
    # fields= carries opt_steps on the liveness thread's opd_step pings: opd_step is upload-throttled,
    # so a stepless liveness ping could win the slot and overwrite the main thread's stepped heartbeat,
    # leaving actual_steps_run to floor a cancelled run to 1 step (codex[bot]).
    # _sdpa_cudnn_ctx(_attn) forces the cuDNN SDPA backend on Blackwell (sm10x/sm120), where
    # optimal_attn_impl() returns "sdpa": sft/rl wrap their forwards the same way (sft.py, rl.py:596),
    # but opd only set attn_implementation at LOAD, so both the on-policy generate and the gkd loss
    # forward ran under the default SDPA dispatch and silently lost the cuDNN kernel (codex[bot]).
    # No-op (nullcontext) on non-Blackwell GPUs / when _attn isn't "sdpa".
    def _samples_progress():
        return samples_seen

    with (
        liveness_heartbeat(
            "opd_step", progress=_samples_progress, fields=lambda: {"step": opt_steps}
        ),
        _sdpa_cudnn_ctx(_attn),
        ThreadPoolExecutor(max_workers=max_teacher_workers) as teacher_pool,
    ):
        while opt_steps < steps and step < max_iters:
            optimizer.zero_grad(set_to_none=True)
            for no_signal_attempt in range(1, max_no_signal_attempts + 1):
                it = step  # data-slice + display index for THIS rollout attempt
                step += 1  # advance up front so the nseq==0 retry path can't spin forever
                batch = [examples[(it * ppl_step + i) % len(examples)] for i in range(ppl_step)]
                accum_target = max(1, ppl_step * group)
                step_loss = 0.0
                step_cov = 0.0
                nseq = 0
                step_skip_counts: Counter[str] = Counter()
                # Step pipeline. vLLM produces on-policy rollouts, scorable completions are immediately
                # handed to the resident teacher pool, and completed teacher futures are converted into
                # differentiable GKD losses as soon as they finish. This preserves one optimizer update
                # per OPD step, but removes the old "wait for every teacher response before any loss"
                # barrier.
                teacher_futures: dict[Future, list[_Pending]] = {}

                def _queue_teacher_batch(
                    pendings: list[_Pending],
                    *,
                    _teacher_futures: dict[Future, list[_Pending]] = teacher_futures,
                ) -> None:
                    def _score_many_timed(batch: list[_Pending]):
                        started = time.perf_counter()
                        scores = _score_many(
                            teacher,
                            batch,
                            thinking_prefill=thinking_prefill,
                        )
                        return scores, time.perf_counter() - started

                    for i in range(0, len(pendings), teacher_batch_size):
                        batch = pendings[i : i + teacher_batch_size]
                        if batch:
                            _teacher_futures[teacher_pool.submit(_score_many_timed, batch)] = batch

                def _queue_or_account(
                    p: _Pending, *, _teacher_futures: dict[Future, list[_Pending]] = teacher_futures
                ) -> None:
                    if p.gen.skip or p.gen.truncated:
                        _account(_resolve_no_loss_sample(p.gen, None))
                        return
                    _queue_teacher_batch([p])

                def _cancel_pending_teacher_futures(
                    *, _teacher_futures: dict[Future, list[_Pending]] = teacher_futures
                ) -> None:
                    for other in _teacher_futures:
                        other.cancel()

                def _consume_teacher_future(
                    fut: Future,
                    *,
                    _teacher_futures: dict[Future, list[_Pending]] = teacher_futures,
                    _accum_target=accum_target,
                ) -> None:
                    batch = _teacher_futures.pop(fut)
                    try:
                        scores, teacher_rpc_seconds = fut.result()
                    except TeacherError:
                        _cancel_pending_teacher_futures()
                        raise
                    opd_phase_seconds["teacher_rpc_sum"] += teacher_rpc_seconds
                    opd_phase_counts["teacher_batches"] += 1
                    if len(scores) != len(batch):
                        raise RuntimeError(
                            f"opd teacher batch returned {len(scores)} score(s) for {len(batch)} sample(s)"
                        )
                    scored_samples = []
                    for p, score in zip(batch, scores, strict=True):
                        p.score = score
                        scored_samples.append((p.gen, score, p.prompt_ids))
                    loss_started = time.perf_counter()
                    resolved = _resolve_samples_batched(
                        model,
                        tok,
                        device,
                        scored_samples,
                        knobs,
                        loss_microbatch_size,
                        backward_scale=1.0 / _accum_target,
                        runaway_rate=trunc_ema,
                    )
                    opd_phase_seconds["loss_backward"] += time.perf_counter() - loss_started
                    opd_phase_counts["loss_batches"] += 1
                    for r in resolved:
                        _account(r, backward=False)

                def _drain_ready_teacher_futures(
                    *, _teacher_futures: dict[Future, list[_Pending]] = teacher_futures
                ) -> None:
                    for fut in list(_teacher_futures):
                        if fut.done():
                            _consume_teacher_future(fut)

                if multi_turn:
                    # Multi-turn Phase 1: drive an EPISODE per (prompt x group), each yielding one
                    # per-turn record. Every assistant turn becomes a _Pending distilled independently
                    # against its transcript-so-far prefix.
                    def _on_turn(_s=opt_steps, _n=nseq):
                        nonlocal samples_seen
                        samples_seen += 1
                        _opd_progress(_s, _n)

                    for _ex, _prompt_messages, _prompt_ids in batch:
                        for _g in range(group):
                            records = rollout_one_records(
                                example=_ex,
                                active_env=env,
                                render=render_fn,
                                generate=generate_fn,
                                env_glue=env_glue_fn,
                                max_turns=mt_max_turns,
                                per_turn_max_tokens=knobs.max_completion,
                                engine_max_len=mt_engine_max_len,
                                on_turn_generated=_on_turn,
                            )
                            episodes_seen += 1
                            mt_turn_records += len(records)
                            for rec in records:
                                _queue_or_account(
                                    _Pending(
                                        gen=rec["gen"],
                                        prompt_ids=rec["prefix_ids"],
                                        prompt_messages=rec["context_messages"],
                                    )
                                )
                            _drain_ready_teacher_futures()
                else:
                    contexts: list[tuple[list[int], object]] = []
                    prompts: list[list[int]] = []
                    for _ex, prompt_messages, prompt_ids in batch:
                        for _g in range(group):
                            contexts.append((prompt_ids, prompt_messages))
                            prompts.append(prompt_ids)
                    chunk_size = _opd_rollout_chunk_size(len(prompts))
                    for start in range(0, len(prompts), chunk_size):
                        end = start + chunk_size
                        chunk_prompts = prompts[start:end]
                        chunk_contexts = contexts[start:end]
                        with liveness_heartbeat(
                            "opd_step",
                            progress=_samples_progress,
                            fields=lambda _step=opt_steps: {"step": _step},
                            keepalive=True,
                        ):
                            rollout_started = time.perf_counter()
                            gens = _generate_many_vllm(
                                vllm_rollout,
                                tok,
                                chunk_prompts,
                                knobs,
                                max_tokens=knobs.max_completion,
                            )
                            opd_phase_seconds["rollout_generate"] += (
                                time.perf_counter() - rollout_started
                            )
                            opd_phase_counts["rollout_generate_calls"] += 1
                        scorable: list[_Pending] = []
                        for gen, (prompt_ids, prompt_messages) in zip(
                            gens, chunk_contexts, strict=True
                        ):
                            p = _Pending(
                                gen=gen, prompt_ids=prompt_ids, prompt_messages=prompt_messages
                            )
                            if gen.skip or gen.truncated:
                                _account(_resolve_no_loss_sample(gen, None))
                            else:
                                scorable.append(p)
                            samples_seen += 1  # advances the liveness-thread progress signal
                            _opd_progress(opt_steps, nseq)
                        _queue_teacher_batch(scorable)
                        _drain_ready_teacher_futures()

                _opd_progress(opt_steps, nseq)  # refresh entering the (bounded) network phase
                wait_started = time.perf_counter()
                for fut in as_completed(list(teacher_futures)):
                    opd_phase_seconds["teacher_wait"] += time.perf_counter() - wait_started
                    if fut in teacher_futures:
                        _consume_teacher_future(fut)
                    wait_started = time.perf_counter()
                _opd_progress(opt_steps, nseq)  # refresh leaving the network phase

                if nseq:
                    break
                reasons = _format_skip_counts(step_skip_counts)
                if no_signal_attempt < max_no_signal_attempts and step < max_iters:
                    no_signal_resamples += 1
                    print(
                        f"[opd] step {it}: no usable teacher signal ({reasons}); "
                        f"resampling rollout {no_signal_attempt}/{max_no_signal_attempts}"
                    )
                    continue
                no_signal_skipped_steps += 1
                print(f"[opd] step {it}: no usable teacher signal this step (skipped; {reasons})")
                break
            if nseq == 0:
                continue
            # Each seq's grad was scaled by 1/accum_target; if some seqs were skipped (teacher call
            # failed / empty completion), rescale to a true 1/nseq mean so a partial step isn't a
            # silently smaller update.
            if nseq != accum_target:
                scale = accum_target / nseq
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
            optimizer_started = time.perf_counter()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            opd_phase_seconds["optimizer_step"] += time.perf_counter() - optimizer_started
            opd_phase_counts["optimizer_steps"] += 1
            opt_steps += 1
            # On-policy means the next rollout must sample from the just-updated student LoRA.
            # The final trained adapter is saved from the HF/PEFT model below, so skip a useless
            # post-final vLLM sync that can fail after all optimizer updates have already landed.
            if opt_steps < steps:
                with liveness_heartbeat(
                    "opd_vllm_sync", progress=lambda s=opt_steps: s, progress_step=True
                ):
                    sync_started = time.perf_counter()
                    vllm_rollout.sync_from_model(model)
                    opd_phase_seconds["vllm_sync"] += time.perf_counter() - sync_started
                    opd_phase_counts["vllm_syncs"] += 1
            avg_loss = step_loss / nseq
            avg_cov = step_cov / nseq
            loss_curve.append(avg_loss)
            coverage_curve.append(avg_cov)
            _w.heartbeat(
                "opd_step",
                # opt_steps (just incremented) == optimizer updates applied, so it is >=1 here: this
                # POST-update ping tightens the stall window (step-gated in the poller) and matches
                # the opt_steps-based checkpoint naming below. force=True so it is NOT throttled away by
                # a mid-step progress ping (carrying the PREVIOUS opt_steps) that just claimed the
                # opd_step upload slot -- otherwise a cancellation here would be billed from the stale
                # pre-update step even though this update landed (codex[bot]).
                step=opt_steps,
                loss=avg_loss,
                coverage=avg_cov,
                gpu=gpu_diagnostics(include_torch=False),
                force=True,
            )
            if _wandb_on:
                # Best-effort: a W&B network hiccup must never abort a paid training run.
                with contextlib.suppress(Exception):
                    import wandb

                    wandb.log({"opd/loss": avg_loss, "opd/coverage": avg_cov}, step=opt_steps)
            if it % 10 == 0:
                print(
                    f"[opd] step {it + 1}/{steps} loss={avg_loss:.4f} "
                    f"coverage={avg_cov:.0%} seqs={nseq}"
                )
            # Checkpoint on OPTIMIZER-step count, not the loop index: a `step-N` artifact must
            # contain N real updates (skipped no-signal iterations don't advance opt_steps), else a
            # warm-start/deploy from step-N would use fewer updates than its name implies.
            if knobs.save_every and opt_steps % knobs.save_every == 0:
                _save_adapter(model, tok, adapter_dir)
                # Best-effort: a mid-run publish failure (e.g. a transient upload error) must not abort
                # the loop after real optimizer steps — the finalize publish is strict.
                _publish_opd_deployable(adapter_dir, opt_steps, as_default=False, best_effort=True)

    train_wall = time.time() - t_train
    if not loss_curve:
        if teacher_ok == 0 and teacher_transient > 0:
            # No sample ever got a teacher score and every failure was a RETRYABLE outage (5xx /
            # timeout / rate-limit): a Fireworks outage that happened to span the whole run. Raise a
            # retriable infra error so the supervisor RETRIES (the run isn't broken), instead of the
            # plain RuntimeError below, which it treats as permanent (codex[bot]).
            raise RetriableInfraError(
                f"opd produced no trained step: all {teacher_transient} teacher scoring calls "
                "failed transiently (0 succeeded) — a Fireworks outage/rate-limit spanning the run. "
                "Retrying."
            )
        raise RuntimeError(
            "opd produced no trained step — every teacher scoring call failed or aligned to "
            "zero positions. Check FIREWORKS_API_KEY and the teacher model id. Failing loudly "
            "instead of reporting a no-op run as done."
        )

    if opt_steps < steps:
        # Real updates landed, but skips kept the loop from reaching the requested `steps` optimizer
        # updates within the iteration budget. Publishing now would serve an under-trained adapter as
        # the DEFAULT while billing the full `steps` quote, so gate the served default (the intermediate
        # non-default checkpoints already published stay available) and either retry or fail loudly.
        diag = (
            f"opd reached only {opt_steps}/{steps} optimizer updates within {max_iters} iterations "
            f"({n_over_budget} prompts pre-filtered over budget, {truncated_rollouts} non-terminated "
            f"rollouts, {teacher_transient} transient teacher failures, {teacher_error} teacher errors, "
            f"{no_signal_resamples} no-signal resamples, {no_signal_skipped_steps} no-signal skipped "
            f"steps, skip reasons: {_format_skip_counts(skip_counts)})"
        )
        if teacher_transient > 0:
            # SOME scoring calls hit a RETRYABLE teacher outage: a healthier teacher next attempt may
            # complete the remaining updates, so RETRY rather than ship short (codex[bot]).
            raise RetriableInfraError(
                f"{diag} — retrying rather than publishing an under-trained adapter billed as full steps."
            )
        # No transient teacher failures: the shortfall is DETERMINISTIC (over-budget prompts /
        # non-terminated or empty rollouts / zero teacher↔student alignment). OPD has no resume, so a
        # retry restarts from step 0 with the same seed/model/data and reproduces the identical skips —
        # burning GPU on an unfixable run and masking the real config/model problem. Fail PERMANENTLY
        # with the skip diagnostics so the user fixes the env/model instead (codex[bot]).
        raise RuntimeError(
            f"{diag} — the shortfall is deterministic (no transient teacher failures), so a retry would "
            "repeat it (opd has no resume). Fix the setup: shorten prompts, enable stop_sequences or a "
            "warm-start so rollouts terminate, or verify the teacher tokenizer aligns to the student."
        )

    _save_adapter(model, tok, adapter_dir)
    # Ship the deployable adapter: the continued (or fresh) LoRA deploys as-is on the catalog base.
    # Name the final checkpoint by real optimizer steps applied, not the planned `steps` count.
    _publish_opd_deployable(adapter_dir, opt_steps, as_default=True)
    # step=opt_steps on this (unthrottled) final ping AND on the opd_train_done ping below keeps the
    # persisted heartbeat's step at the true completed count. Without it, a cancel landing after the
    # adapter publish but before DONE is persisted reads a STEPLESS opd_trained/opd_train_done as the
    # last heartbeat, and actual_steps_run floors a fully-trained run to 0 (opd_trained isn't a training
    # stage) -- re-pricing paid work as $0 (codex[bot]).
    _w.heartbeat("opd_trained", step=opt_steps, train_wall=train_wall, gpu=gpu_diagnostics())

    _w.write_train_meta(
        phase="opd",
        step=opt_steps,
        adapter_dir=adapter_dir,
        model_id=model_id,
        train_wall=train_wall,
        setup_seconds=setup_seconds,
        train_tokens=0,
        generated_tokens=generated_tokens,
        notes={
            "steps": steps,
            "epochs": knobs.epochs,
            "retained_prompts": len(examples),
            # Optimizer steps actually applied; < steps if any iteration had no usable teacher
            # signal (skipped). loss_curve length == opt_steps, so reporting stays honest.
            "opt_steps": opt_steps,
            "dropped_long_prompts": n_over_budget,  # prompts pre-filtered for exceeding the context budget
            "method": "gkd",
            "init_from_adapter": warm_start or None,
            "teacher_model": knobs.teacher_model,
            "download_seconds": download_seconds,
            "chalk_kernels": _chalk_active or None,
            "thinking": _w.THINKING,
            "loss_curve": loss_curve,
            "mean_coverage": (sum(coverage_curve) / len(coverage_curve)) if coverage_curve else 0.0,
            # Rollouts that didn't terminate naturally (max_new_tokens cap OR max_time cut, no EOS/stop)
            # — skipped, not distilled (see _rollout_terminated). A high count means the student is
            # generating runaway/non-terminating filters (cold start, untrained stop token); the
            # terminal-EOS reinforcement below (opd_eos_loss_coef) and warm-starting from SFT both help.
            "truncated_rollouts": truncated_rollouts,
            # Terminal-EOS behaviour-cloning (see _eos_reinforce_term): the coefficient, how many
            # distilled samples got the stop signal, and the mean reinforced log P(eos). The last
            # should RISE as truncated_rollouts falls — visible confirmation the student is learning
            # to stop. mean_eos_logprob is None when reinforcement is disabled (coef 0).
            "eos_loss_coef": knobs.eos_loss_coef,
            "eos_reinforced_samples": eos_logprob_n,
            "mean_eos_logprob": (eos_logprob_sum / eos_logprob_n) if eos_logprob_n else None,
            "teacher_transient_failures": teacher_transient,
            "teacher_errors": teacher_error,
            "no_signal_resamples": no_signal_resamples,
            "no_signal_skipped_steps": no_signal_skipped_steps,
            "skip_reasons": dict(sorted(skip_counts.items())),
            # Real alignment-health signal (mean student-tokens-per-group); mean_coverage is ~1.0 even
            # for a degenerate collapsed alignment, so it can't flag that failure mode.
            "mean_align_granularity": (granularity_sum / granularity_n) if granularity_n else 0.0,
            "teacher_input_tokens": teacher_input_tokens,
            "temperature": knobs.temperature,
            "group_size": group,
            "prompts_per_step": ppl_step,
            "max_completion_len": knobs.max_completion,
            "rollout_backend": "vllm",
            "vllm_model": getattr(vllm_rollout, "model_source", None),
            "vllm_max_model_len": getattr(vllm_rollout, "max_model_len", None),
            "vllm_gpu_memory_utilization": getattr(vllm_rollout, "gpu_memory_utilization", None),
            "vllm_kv_cache_dtype": vllm_kwargs.get("kv_cache_dtype"),
            "vllm_rollout_batch_size": vllm_kwargs.get("rollout_batch_size"),
            "vllm_max_num_batched_tokens": vllm_kwargs.get("max_num_batched_tokens"),
            "vllm_enforce_eager": vllm_kwargs.get("enforce_eager"),
            "vllm_compilation_config": vllm_kwargs.get("compilation_config"),
            "vllm_init_seconds": vllm_init_seconds,
            "vllm_lora_syncs": getattr(vllm_rollout, "sync_count", None),
            "opd_phase_rollout_generate_seconds": float(opd_phase_seconds["rollout_generate"]),
            "opd_phase_teacher_rpc_sum_seconds": float(opd_phase_seconds["teacher_rpc_sum"]),
            "opd_phase_teacher_wait_seconds": float(opd_phase_seconds["teacher_wait"]),
            "opd_phase_loss_backward_seconds": float(opd_phase_seconds["loss_backward"]),
            "opd_phase_optimizer_step_seconds": float(opd_phase_seconds["optimizer_step"]),
            "opd_phase_vllm_sync_seconds": float(opd_phase_seconds["vllm_sync"]),
            "opd_phase_rollout_generate_calls": int(opd_phase_counts["rollout_generate_calls"]),
            "opd_phase_teacher_batches": int(opd_phase_counts["teacher_batches"]),
            "opd_phase_loss_batches": int(opd_phase_counts["loss_batches"]),
            "opd_phase_optimizer_steps": int(opd_phase_counts["optimizer_steps"]),
            "opd_phase_vllm_syncs": int(opd_phase_counts["vllm_syncs"]),
            "opd_rollout_pipeline_chunks": (
                _opd_rollout_pipeline_chunks(ppl_step * group) if not multi_turn else None
            ),
            "opd_rollout_chunk_size": (
                _opd_rollout_chunk_size(ppl_step * group) if not multi_turn else None
            ),
            "opd_rollout_pipeline_target_chunk_size": (
                _opd_rollout_pipeline_target_chunk_size(ppl_step * group)
                if not multi_turn
                else None
            ),
            "opd_rollout_pipeline_max_chunks": (
                _opd_rollout_pipeline_max_chunks(ppl_step * group) if not multi_turn else None
            ),
            "opd_teacher_workers": max_teacher_workers,
            "opd_teacher_batch_size": teacher_batch_size,
            "opd_loss_microbatch_size": loss_microbatch_size,
            "opd_full_logits_batches": getattr(model, "_flash_opd_full_logits_batches", 0),
            # Multi-turn: each assistant turn is distilled independently against its transcript-so-far
            # prefix, so a "sample" is a TURN, not a whole episode. Report the mode + turn ceiling and
            # the mean turns/episode so a run that collapsed to one-turn episodes (env never replied /
            # student never continued) is visible. Single-turn: mode="single-turn", episodes==0.
            "multi_turn": multi_turn,
            "max_turns": mt_max_turns if multi_turn else None,
            "episodes": episodes_seen if multi_turn else None,
            "mean_turns_per_episode": (
                (mt_turn_records / episodes_seen) if (multi_turn and episodes_seen) else None
            ),
            **_w.wandb_run_info(),
        },
    )
    vllm_rollout.close()
    free_gpu(model)


def _score_many(
    teacher, pendings: list[_Pending], *, thinking_prefill, max_attempts: int = 2
) -> list[_ScoreResult]:
    """[THREAD POOL — network only] Batch teacher echo-scoring for one chunk of scorable samples."""
    if not pendings:
        return []
    prompts = [
        (_teacher_prompt_text(p.prompt_messages, thinking_prefill), p.gen.completion_text)
        for p in pendings
    ]
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            scored = teacher.score_many(prompts)
        except AttributeError:
            return [
                _score_one(
                    teacher,
                    p.gen,
                    prompt_messages=p.prompt_messages,
                    thinking_prefill=thinking_prefill,
                    max_attempts=max_attempts,
                )
                for p in pendings
            ]
        except TeacherError as e:
            if e.permanent:
                raise
            if attempt < attempts:
                print(
                    f"[opd] teacher batch failed (transient, retrying batch {attempt}/{attempts}, "
                    f"samples={len(pendings)}): {e}"
                )
                continue
            print(
                f"[opd] teacher batch failed (transient, skipping batch, samples={len(pendings)}): {e}"
            )
            return [_ScoreResult(status="transient", error=str(e)) for _ in pendings]
        except Exception as e:
            if attempt < attempts:
                print(
                    f"[opd] teacher batch failed (retrying batch {attempt}/{attempts}, "
                    f"samples={len(pendings)}): {e}"
                )
                continue
            print(f"[opd] teacher batch failed (skipping batch, samples={len(pendings)}): {e}")
            return [_ScoreResult(status="error", error=str(e)) for _ in pendings]
        if len(scored) != len(pendings):
            return [
                _ScoreResult(
                    status="error",
                    error=f"teacher batch returned {len(scored)} score(s) for {len(pendings)} sample(s)",
                )
                for _ in pendings
            ]
        return [_ScoreResult(teacher_toks=toks, status="ok") for toks in scored]
    return [_ScoreResult(status="error", error="teacher batch attempts exhausted") for _ in pendings]


def _resolve_samples_batched(
    model,
    tok,
    device,
    samples: list[tuple[_GenResult, _ScoreResult, object]],
    knobs,
    microbatch: int,
    *,
    backward_scale: float | None = None,
    runaway_rate: float = 1.0,
) -> list[SampleResult]:
    import torch

    if not samples:
        return []
    results: list[SampleResult | None] = [None] * len(samples)
    prepared: list[_PreparedLoss] = []
    # Terminal-EOS supervision config (see _eos_reinforce_term). Resolved once — the model/tokenizer
    # don't change across the chunk. getattr defaults keep bare test stand-ins (SimpleNamespace knobs)
    # and eos-less fake tokenizers working with reinforcement OFF.
    eos_coef = float(getattr(knobs, "eos_loss_coef", 0.0) or 0.0)
    # Controller: reinforce the terminal EOS only in proportion to the student's CURRENT failure to
    # terminate (see _runaway_eos_scale). A cleanly-terminating run -> ~0 push -> the shared eos logit
    # can't ratchet into an empty-collapse; genuine runaway still drives full reinforcement. Default
    # runaway_rate=1.0 keeps bare callers (tests) at full strength = pre-controller behaviour.
    eos_coef *= _runaway_eos_scale(runaway_rate)
    eos_stop_sequences = tuple(getattr(knobs, "stop_sequences", ()) or ())
    eos_ids = _generation_eos_ids(model, tok) if eos_coef > 0 else frozenset()
    eos_primary = _primary_eos_id(tok, eos_ids) if eos_coef > 0 else None
    for idx, (gen, score, prompt_ids) in enumerate(samples):
        if gen.truncated or gen.skip or score is None or score.status != "ok":
            results[idx] = _resolve_no_loss_sample(gen, score)
            continue
        teacher_tokens = len(prompt_ids) + gen.gen_tokens
        student_ids, student_toks = student_tokens_with_offsets(
            tok, gen.completion_ids, gen.completion_text
        )
        if not student_ids:
            results[idx] = SampleResult(
                teacher_status="ok",
                gen_tokens=gen.gen_tokens,
                teacher_tokens=teacher_tokens,
                skip_reason="student_tokens_empty",
            )
            continue
        groups = groupwise_alignment(student_toks, score.teacher_toks)
        # Drop fully grammar-forced groups, THEN let the loss's per-token mean (_gkd_loss_from_logps)
        # re-normalize over the SURVIVING tokens: masking removes the spurious forced-position teacher
        # signal without shrinking the content-token gradient. A fully-forced completion drops to zero
        # groups -> alignment_empty (below), and such samples are excluded from the step's 1/nseq mean
        # (_account skips r.loss is None), so the batch stays normalized too.
        groups = _drop_fully_forced_groups(groups, gen.forced or ())
        coverage = groupwise_coverage(groups, student_toks)
        n_align = sum(1 for st in student_toks if st.end > st.start)
        group_granularity = (n_align / len(groups)) if groups else 0.0
        prepared_groups = _prepare_gkd_groups(groups)
        if prepared_groups is None:
            results[idx] = SampleResult(
                teacher_status="ok",
                coverage=coverage,
                gen_tokens=gen.gen_tokens,
                teacher_tokens=teacher_tokens,
                group_granularity=group_granularity,
                skip_reason="alignment_empty",
            )
            continue
        prepared.append(
            _PreparedLoss(
                idx=idx,
                prompt_ids=prompt_ids,
                student_ids=student_ids,
                groups=prepared_groups,
                coverage=coverage,
                gen_tokens=gen.gen_tokens,
                teacher_tokens=teacher_tokens,
                group_granularity=group_granularity,
            )
        )

    if prepared:
        model.train()
        model.config.use_cache = False
        pad_id = int(getattr(tok, "pad_token_id", 0) or 0)
        mb = max(1, int(microbatch))
        for start in range(0, len(prepared), mb):
            chunk = prepared[start : start + mb]
            _bump_model_counter(model, "_flash_opd_full_logits_batches")
            seqs = [list(p.prompt_ids) + list(p.student_ids) for p in chunk]
            max_len = max(len(seq) for seq in seqs)
            input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros(
                (len(seqs), max_len), dtype=torch.long, device=device
            )
            for row, seq in enumerate(seqs):
                input_ids[row, : len(seq)] = torch.tensor(
                    seq, dtype=torch.long, device=device
                )
                attention_mask[row, : len(seq)] = 1
            logits = _forward_logits(model, input_ids, attention_mask)
            losses = []
            for row, p in enumerate(chunk):
                prompt_len = len(p.prompt_ids)
                comp_len = len(p.student_ids)
                rows = logits[row, prompt_len - 1 : prompt_len - 1 + comp_len]
                loss = _gkd_loss_from_logits_rows(
                    rows, p.student_ids, p.groups, kl_coef=knobs.kl_coef
                )
                if loss is None:
                    results[p.idx] = SampleResult(
                        teacher_status="ok",
                        coverage=p.coverage,
                        gen_tokens=p.gen_tokens,
                        teacher_tokens=p.teacher_tokens,
                        group_granularity=p.group_granularity,
                        skip_reason="alignment_empty",
                    )
                else:
                    # Behaviour-clone the terminal stop the reverse-KL alignment cannot reach. Added
                    # to the distilled samples only (loss is not None), so no-signal skip accounting is
                    # unchanged. logits[row] is this sample's per-position logits [max_len, V]; the term
                    # self-limits, so a student that already stops cleanly contributes ~0.
                    eos_logprob = None
                    if eos_coef > 0:
                        eos_out = _eos_reinforce_term(
                            logits[row],
                            prompt_len,
                            p.student_ids,
                            eos_ids,
                            eos_primary,
                            eos_stop_sequences,
                            eos_coef,
                        )
                        if eos_out is not None:
                            eos_term, eos_logprob = eos_out
                            loss = loss + eos_term
                    loss_for_result = loss
                    if backward_scale is not None:
                        losses.append(loss)
                        loss_for_result = loss.detach()
                    results[p.idx] = SampleResult(
                        loss=loss_for_result,
                        teacher_status="ok",
                        coverage=p.coverage,
                        gen_tokens=p.gen_tokens,
                        teacher_tokens=p.teacher_tokens,
                        group_granularity=p.group_granularity,
                        eos_logprob=eos_logprob,
                    )
            if losses:
                (sum(losses) * float(backward_scale)).backward()

    return [r if r is not None else SampleResult(skip_reason="teacher_error") for r in results]
