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

There is NO local reference model and NO colocated vLLM engine: sampling is HF ``generate`` on the
resident student (so the VRAM profile matches SFT), and the teacher lives behind the API. All heavy
imports (torch/transformers/peft) are inside functions, so importing this module is CPU/offline-safe.
"""

from __future__ import annotations

import contextlib
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from flash.engine.chalk_kernels import active_kernels, install_chalk_kernels
from flash.engine.recipe import RECIPE
from flash.engine.vram import opd_completion_len
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.opd_gkd import (
    _generation_eos_ids,
    _rollout_terminated,
    _teacher_prompt_text,
    _to_cpu_ids,
    _trim_trailing_stop,
    gkd_loss,
    student_tokens_with_offsets,
)
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

# Upper bound on teacher-scoring HTTP calls fired concurrently within one optimizer step. Generation
# (GPU) and the gkd loss forward/backward stay serial on the main thread; only the (dominant) teacher
# Fireworks round-trips for a step's samples are overlapped, so this caps the fan-out into the API.
_TEACHER_SCORE_MAX_WORKERS = 8  # bound concurrent GLM/Fireworks scoring calls per step


@dataclass(frozen=True)
class OpdKnobs:
    """Resolved opd knobs from the JobSpec's [train] table (falling back to RECIPE.opd), returned by
    ``_resolve_opd_knobs``. A typed container replacing the old stringly-typed dict; field names match
    the former dict keys one-for-one. The defaults are placeholders only — ``_resolve_opd_knobs``
    always sets every field explicitly — kept so partial construction stays ergonomic for tests."""

    teacher_model: str = ""
    teacher_base_url: str = ""
    steps: int = 0
    learning_rate: float = 0.0
    temperature: float = 0.0
    top_p: float = 1.0
    max_completion: int = 0
    prompts_per_step: int = 0
    group_size: int = 0
    # gkd reverse-KL scale; reuses the existing [train] kl_penalty_coef knob (default 1.0).
    kl_coef: float = 1.0
    save_every: int = 0
    max_length: int = 0
    # Student on-policy sampling stops at these delimiters (parity with GRPO), so the teacher never
    # scores/trains on text past the intended answer boundary.
    stop_sequences: tuple = ()


def _resolve_opd_knobs() -> OpdKnobs:
    """Resolve every opd knob from the JobSpec's [train] table, falling back to RECIPE.opd."""
    d = RECIPE.opd
    t = _w.JOB_SPEC.train if _w.JOB_SPEC else None

    def opt(name, default):
        v = getattr(t, name, None) if t else None
        return v if v is not None else default

    # kl_penalty_coef IS the gkd distillation objective's scale: gkd_loss multiplies every span
    # coefficient by it, so kl_coef=0 makes EVERY backward a zero gradient while the loop still counts
    # opt_steps and would publish/charge a fully-untrained adapter. The shared schema allows 0 for GRPO
    # (a valid "no KL penalty"), but for OPD it must be positive, so reject an explicit 0 here (a plain
    # `or` would also wrongly treat 0.0 as unset). Omitting the field uses the positive recipe default.
    _kl = opt("kl_penalty_coef", None)
    kl_coef = float(_kl if _kl is not None else d.kl_coef)
    if kl_coef == 0.0:
        raise RuntimeError(
            "opd: [train] kl_penalty_coef must be > 0 — it scales the gkd distillation objective, so "
            "0 makes every optimizer step a no-op (zero gradient) yet still counts toward `steps` and "
            "publishes an untrained adapter. Omit the field to use the default, or set a positive value."
        )
    return OpdKnobs(
        teacher_model=opt("teacher_model", "") or d.teacher_model,
        teacher_base_url=d.teacher_base_url,
        steps=int(opt("steps", 0) or d.num_steps),
        learning_rate=float(opt("learning_rate", 0) or d.learning_rate),
        temperature=float(
            opt("temperature", None)
            if (t and t.temperature is not None)
            else d.sampling_temperature
        ),
        top_p=d.sampling_top_p,
        max_completion=opd_completion_len(opt("max_tokens", 0), _w.THINKING),
        prompts_per_step=int(opt("batch_size", 0) or d.prompts_per_step),
        group_size=int(opt("group_size", 0) or d.group_size),
        kl_coef=kl_coef,
        save_every=int(opt("save_every", 0) or 20),
        max_length=int(opt("max_length", 0) or 0),
        stop_sequences=tuple(getattr(t, "stop_sequences", ()) or ()),
    )


def _thinking_prefill_text(tok) -> str:
    """The trailing text a thinking-mode chat template opens after the generation prompt (Qwen's
    ``<think>\\n``), i.e. the delta between the enable_thinking=True and =False renders. Returns "" when
    thinking is off or the template ignores enable_thinking (the two renders match), so callers can
    unconditionally append it to the teacher prompt for student/teacher conditioning parity."""
    if not _w.THINKING:
        return ""
    probe = [{"role": "user", "content": ""}]
    with contextlib.suppress(Exception):
        base = tok.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        think = tok.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        if think == base:
            return ""  # template ignores enable_thinking -> plain "Assistant: " already matches
        # The thinking render opens a reasoning block the non-thinking render doesn't, but it is NOT
        # always a pure suffix (the old think.startswith(base) test). Compute the longest common PREFIX
        # and SUFFIX of the two renders; the thinking render's UNIQUE MIDDLE is the opener.
        p = 0
        m = min(len(base), len(think))
        while p < m and base[p] == think[p]:
            p += 1
        s = 0
        while s < len(base) - p and s < len(think) - p and base[-1 - s] == think[-1 - s]:
            s += 1
        think_mid = think[p : len(think) - s]
        base_mid = base[p : len(base) - s]
        # CLOSED-BLOCK hybrid recovery, checked BEFORE the think_mid early-return: enable_thinking=False
        # force-CLOSES the block (base's unique middle is a closing tag "</think>...") while the shared
        # prefix already opened "<think>", which the student pre-fills. Recover the OPEN-block opener from
        # the think render so the teacher conditions on the same open block instead of base's closed one.
        # This MUST run before `if think_mid` because a base that closes the block right after the opener
        # leaves a non-empty WHITESPACE remainder in think_mid (base "<think></think>" vs think
        # "<think>\n" -> think_mid "\n"), which the early-return would otherwise hand back in place of the
        # real "<think>\n" opener (codex[bot]). The base "<think>\n\n</think>" / think "<think>\n" shape
        # (think_mid EMPTY) is the SAME recovery. lstrip absorbs intra-block whitespace before the closing
        # tag so detection still fires; we return think[cut:] (the thinking-side opener), so the strip only
        # affects DETECTION, not the returned opener. If the opener isn't in the shared prefix, fall
        # through: return the think_mid delta ("" only when the model opens <think> inside the completion).
        base_mid_tag = base_mid.lstrip()
        if base_mid_tag.startswith("</") and ">" in base_mid_tag:
            open_tag = "<" + base_mid_tag[2 : base_mid_tag.index(">") + 1]  # "</think>..." -> "<think>"
            cut = think.rfind(open_tag, 0, p)
            if cut != -1:
                return think[cut:]  # e.g. "<think>\n"
        if think_mid:
            return think_mid  # opener appended, or inserted before shared trailing template text
    return ""


def _student_model(model_id, mik, device):
    """Build the trainable student LoRA. Warm-starts from ``train.init_from_adapter`` when set —
    continuing a prior run's adapter (e.g. an SFT checkpoint), the same path GRPO uses via
    ``_init_adapter_model`` — otherwise a fresh LoRA on the base. This makes an SFT->opd pipeline a
    genuine continuation (the opd stage keeps the SFT behavior) rather than silently restarting from
    base.

    Two warm-start shapes, both from ``_init_adapter_model`` (parity with GRPO):
      - non-VL (e.g. MiniCPM): a trainable PeftModel that CONTINUES the SFT adapter in place
        (``init_peft is None``); the saved adapter already carries the SFT.
      - VL (Qwen3.5/3.6): ``_init_adapter_model`` MERGES the SFT into the base and returns the merged
        dir + a FRESH LoRA config. We train that fresh LoRA on the merged base; ``run_opd`` then
        recombines SFT⊕opd at publish (``recombined_warmstart_adapter_dir``) so the DEPLOYED adapter
        reproduces base+SFT+opd on the unmodified catalog base — exactly GRPO's VL warm-start path.
    """
    init_model, init_peft = _w._init_adapter_model(model_id)
    if init_peft is None:
        # init_model is already a trainable PeftModel continuing the prior (e.g. SFT) adapter.
        return init_model.to(device)
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    # init_model is the base id (fresh run) or the VL SFT-merged dir; both load as a causal LM — the
    # merged dir preserves the catalog arch + remote code, so this matches the base load the fresh
    # path already uses on Qwen3.5. The fresh LoRA excludes vision modules via make_lora.
    base = AutoModelForCausalLM.from_pretrained(init_model, trust_remote_code=True, **mik).to(
        device
    )
    return get_peft_model(base, init_peft)


def _opd_progress(step: int, done: int) -> None:
    """Emit the in-loop non-liveness ``opd_step`` progress ping. Pollers ignore liveness heartbeats, so
    a long teacher-bound step (the bounded-concurrency teacher scoring of a large batch/group, or a
    slow/retrying Fireworks endpoint that stalls even the overlapped calls; also the serial on-policy
    generation preceding it) — or a stretch where every sample skips (empty completions, no teacher
    signal) — would otherwise emit only
    liveness pings and trip the training stall window. Report ``step`` = optimizer updates COMPLETED so
    far (opt_steps), NOT the loop index: opd_step is step-gated in the poller (_poll.STEP_GATED_STAGES),
    so while the FIRST optimizer step is still accumulating (opt_steps==0) these pings keep the WIDE
    setup grace and must not flip a still-running first step into the tight training window; once a real
    update has landed (opt_steps>=1) they tighten it as intended, and the throttle bounds the HF upload
    rate."""
    _w.heartbeat("opd_step", step=step, samples_done=done)


def run_opd():
    import torch
    from transformers import AutoTokenizer

    from flash.engine.worker.teacher import TeacherClient

    env = _w.require_active_env()
    if getattr(env, "is_tool_env", False):
        # Tool envs need TRL's native tool-call loop (rl.py hands the tool schemas + callables to the
        # trainer); opd's HF-generate rollout can't drive that, so it stays unsupported. Pure
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
        f"steps={knobs.steps} warm_start={warm_start or 'none'} "
        f"mode={'multi-turn' if multi_turn else 'single-turn'}"
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
    rng = random.Random(_w.SEED)
    rng.shuffle(train)
    steps = knobs.steps
    ppl_step = knobs.prompts_per_step
    group = knobs.group_size
    # Prompt budget mirrors GRPO: DROP (not truncate) prompts over the context budget, so the student
    # never conditions on a truncated prompt the teacher didn't see. Use the configured max_length
    # when set, else the recipe prompt cap.
    if knobs.max_length:
        prompt_budget = knobs.max_length - knobs.max_completion
        if prompt_budget < 1:
            # A non-positive remainder means max_length <= max_tokens: there is no room for any
            # prompt, so every sample would run generate+loss past the configured context. Reject
            # loudly instead of clamping to a 1-token budget that silently admits over-budget runs.
            raise RuntimeError(
                f"opd: [train] max_length ({knobs.max_length}) leaves no prompt budget after "
                f"max_tokens ({knobs.max_completion}); set max_length > max_tokens."
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
            f"(max_length={knobs.max_length or 'unset'}, "
            f"max_completion={knobs.max_completion}). Raise [train].max_length or shorten "
            "prompts — failing before the training loop instead of dropping every prompt for "
            "every step and burning the GPU allocation."
        )
    if n_over_budget:
        print(
            f"[opd] filtered {n_over_budget}/{len(train)} prompts over the "
            f"{prompt_budget}-token budget; pool = {len(examples)}"
        )

    # Now that a non-empty on-policy pool is confirmed, prefetch the full base weights (deferred from
    # setup so an all-over-budget dataset fails before this download). Still inside the setup phase
    # (opt_steps==0 -> wide poller grace), same as when it ran earlier.
    download_seconds = _w.prefetch_model(model_id)

    # Seed torch/CUDA BEFORE constructing the student LoRA: get_peft_model samples the LoRA A matrix
    # (init_lora_weights=True) from the torch default generator, so seeding must precede _student_model
    # for the fixed Flash seed to reproduce the same adapter init run-to-run (the fresh-LoRA and VL
    # warm-start paths both build a fresh LoRA). It also makes the later model.generate(do_sample=True)
    # completions reproducible. The prompt shuffle above uses a SEPARATE random.Random(_w.SEED), so its
    # ordering is unaffected by where torch is seeded.
    torch.manual_seed(_w.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_w.SEED)

    setup_seconds = time.time() - t_start
    _w.heartbeat("opd_model_load", setup_seconds=setup_seconds, gpu=gpu_diagnostics())
    with liveness_heartbeat("opd_initializing"):
        model = _student_model(model_id, mik, device)
        # Apply chalk standalone kernels to the student, exactly as sft/rl do after building their
        # trainer. The student drives BOTH on-policy generation and the loss forward, so without this
        # the default Qwen3.5/3.6 catalog model silently falls back to eager GDN/RMSNorm/RoPE/LoRA
        # kernels and a long distillation runs much slower than the rest of the stack (codex[bot]).
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
    model.config.use_cache = (
        True  # generation needs the KV cache; re-disabled per loss forward below
    )

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

    gen_cfg = {
        "do_sample": knobs.temperature > 0,
        "temperature": max(knobs.temperature, 1e-5),
        "top_p": knobs.top_p,
        "max_new_tokens": knobs.max_completion,
        "pad_token_id": tok.pad_token_id,
        # Bound generation WALL-TIME so a degenerate or near-OOM-thrashing generate cannot silently
        # eat the training stall window: a single _train_one blocks the loop while it runs (the
        # per-sample opd_step heartbeat only fires AFTER it returns), so an unbounded generate that
        # stops making progress emits no non-liveness ping and the poller reaps the whole attempt as
        # "stalled" (observed on the OPD-16 linkd e2e: one step wedged >1500s at 93% VRAM). HF checks
        # max_time between token steps, so a thrashing (still-stepping) generate is cut here and the
        # sample returns partial/empty -> the heartbeat resumes. Scale with the completion budget
        # (thinking mode needs longer) but keep it well under the poller's ~1500s training window.
        "max_time": min(900.0, max(180.0, float(knobs.max_completion) * 0.75)),
    }
    if knobs.stop_sequences:
        # HF stops generation at any of these strings (needs the tokenizer to match them on decode).
        gen_cfg["stop_strings"] = list(knobs.stop_sequences)
        gen_cfg["tokenizer"] = tok
    loss_curve: list[float] = []
    coverage_curve: list[float] = []
    generated_tokens = 0
    teacher_input_tokens = 0
    # Rollouts skipped for hitting the cap without EOS (mid-output truncations we refuse to distil),
    # and a running mean of alignment granularity — both surfaced in train_meta for diagnosis.
    truncated_rollouts = 0
    granularity_sum = 0.0
    granularity_n = 0
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

    # Concurrent teacher scoring needs a REAL student that samples on-policy (model.generate) so the
    # step's rollouts can be produced up front (serial GPU) and then scored all at once (Phase 1/2/3
    # below). Every HF/Peft catalog model has generate(); CPU unit-test stand-ins without it drive the
    # equivalent per-sample _train_one path (the serial fallback), keeping _train_one the single stub
    # point those tests inject. The model does not change across steps, so decide once.
    _parallel_scoring = hasattr(model, "generate") and hasattr(model, "eval")
    if multi_turn and not _parallel_scoring:
        # Multi-turn distillation IS the on-policy rollout: it must sample each turn with the real
        # student (model.generate), so a stand-in without generate has nothing to roll out. The serial
        # _train_one fallback is single-turn only.
        raise RuntimeError(
            "opd multi-turn requires a student that can sample on-policy (model.generate); the "
            "loaded model exposes none."
        )

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
            return render_message_ids(
                tok, messages, add_generation_prompt, thinking=_w.THINKING
            )

        def generate_fn(prefix_ids, max_new):
            """One turn's generation: HF model.generate over the transcript-so-far ``prefix_ids``,
            bounded to ``max_new`` (the per-turn cap the driver already shrank to the remaining engine
            budget), reusing _generate_one so the termination / trailing-stop-trim / empty / U+FFFD
            gates apply per turn exactly as they do single-turn."""
            turn_cfg = dict(gen_cfg)
            turn_cfg["max_new_tokens"] = max(1, int(max_new))
            prompt_tensor = torch.tensor([prefix_ids], device=device)
            return _generate_one(
                model=model,
                tok=tok,
                device=device,
                prompt_tensor=prompt_tensor,
                gen_cfg=turn_cfg,
                knobs=knobs,
            )

    def _account(r):
        """Consume one sample's SampleResult: tally teacher health / truncations, backprop a distilled
        sample (scaled 1/accum_target for gradient accumulation), advance the step aggregates, and
        refresh the stall clock. Shared by the concurrent-scoring path's Phase 3 and the serial
        fallback so both keep the pre-parallelization per-sample bookkeeping byte-for-byte identical.
        ``samples_seen`` is advanced by the caller (once per GENERATION) so its timing is unchanged."""
        nonlocal teacher_ok, teacher_transient, truncated_rollouts, step_loss, step_cov
        nonlocal granularity_sum, granularity_n, generated_tokens, teacher_input_tokens, nseq
        if r.teacher_status == "ok":
            teacher_ok += 1
        elif r.teacher_status == "transient":
            teacher_transient += 1
        if r.truncated:
            truncated_rollouts += 1
        if r.loss is None:
            # Refresh the stall clock even when a sample yields no teacher signal (rationale in
            # _opd_progress) — else an all-skip stretch emits only ignored liveness pings.
            _opd_progress(opt_steps, nseq)
            return
        (r.loss / accum_target).backward()
        step_loss += float(r.loss.detach())
        step_cov += r.coverage
        granularity_sum += r.group_granularity
        granularity_n += 1
        generated_tokens += r.gen_tokens
        teacher_input_tokens += r.teacher_tokens
        nseq += 1
        # Non-liveness progress ping WITHIN the step (rationale in _opd_progress).
        _opd_progress(opt_steps, nseq)

    t_train = time.time()
    # Drive the loop by optimizer UPDATES, not raw iterations. A no-signal iteration (empty
    # completions / a flaky teacher) skips optimizer.step() below, so `for step in range(steps)` could
    # exit with opt_steps < steps -- shipping an under-trained adapter as the served DEFAULT while the
    # run is billed the full submit-time `steps` quote (codex[bot]). Loop until `steps` real updates
    # land, visiting a fresh data slice each iteration (`it` advances on a skip so a bad batch is not
    # re-tried). Bound the iterations so a persistently degraded teacher cannot spin unboundedly --
    # the post-loop guard then turns a shortfall into a RETRY, not a silent under-trained publish.
    step = 0
    max_iters = 2 * steps + 10
    # fields= carries opt_steps on the liveness thread's opd_step pings: opd_step is upload-throttled,
    # so a stepless liveness ping could win the slot and overwrite the main thread's stepped heartbeat,
    # leaving actual_steps_run to floor a cancelled run to 1 step (codex[bot]).
    # _sdpa_cudnn_ctx(_attn) forces the cuDNN SDPA backend on Blackwell (sm10x/sm120), where
    # optimal_attn_impl() returns "sdpa": sft/rl wrap their forwards the same way (sft.py, rl.py:596),
    # but opd only set attn_implementation at LOAD, so both the on-policy generate and the gkd loss
    # forward ran under the default SDPA dispatch and silently lost the cuDNN kernel (codex[bot]).
    # No-op (nullcontext) on non-Blackwell GPUs / when _attn isn't "sdpa".
    with (
        liveness_heartbeat(
            "opd_step", progress=lambda: samples_seen, fields=lambda: {"step": opt_steps}
        ),
        _sdpa_cudnn_ctx(_attn),
    ):
        while opt_steps < steps and step < max_iters:
            it = step  # data-slice + display index for THIS iteration
            step += 1  # advance up front so the nseq==0 `continue` below can't spin the while loop
            batch = [examples[(it * ppl_step + i) % len(examples)] for i in range(ppl_step)]
            accum_target = max(1, ppl_step * group)
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_cov = 0.0
            nseq = 0
            if _parallel_scoring:
                # Concurrent-scoring step in three phases. Behavior-preserving vs. the old per-sample
                # loop: SAME samples in the SAME generation order (RNG untouched), SAME per-sample
                # losses, SAME gradient accumulation — only the teacher's Fireworks HTTP round-trips
                # (the dominant per-step cost, previously issued one-at-a-time) are overlapped.
                #
                # Phase 1 (serial, GPU eval): sample every completion on-policy, IN ORDER, caching the
                # prompt context each one needs downstream. The cached messages/ids from the budget
                # filter drive BOTH the student prompt AND the teacher prompt, so student sampling and
                # teacher scoring stay in sync and every visited prompt fits (codex[bot]).
                pending: list[_Pending] = []
                if multi_turn:
                    # Multi-turn Phase 1: drive an EPISODE per (prompt x group), each yielding one
                    # per-turn record. Every assistant turn becomes a _Pending distilled independently
                    # against its transcript-so-far prefix — so a group-of-3 over a 4-turn episode
                    # produces up to 12 turn-samples, all scored + lossed by the SAME Phase 2/3 below.
                    # Bind opt_steps/nseq as defaults (nseq is 0 all of Phase 1; opt_steps is fixed
                    # until Phase 3's optimizer.step) to satisfy the loop-var-capture lint — same
                    # pattern as the serial fallback's on_generated below. Called synchronously per
                    # turn inside rollout_one_records, so these are the right current-iteration values.
                    def _on_turn(_s=opt_steps, _n=nseq):
                        # Advance the liveness progress signal and refresh the stall clock after each
                        # turn's (serial, up-to-~900s) generation, before the next turn / teacher call.
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
                            pending.extend(
                                _Pending(
                                    gen=rec["gen"],
                                    prompt_ids=rec["prefix_ids"],
                                    prompt_messages=rec["context_messages"],
                                )
                                for rec in records
                            )
                else:
                    for _ex, prompt_messages, prompt_ids in batch:
                        prompt_tensor = torch.tensor([prompt_ids], device=device)
                        for _g in range(group):
                            gen = _generate_one(
                                model=model,
                                tok=tok,
                                device=device,
                                prompt_tensor=prompt_tensor,
                                gen_cfg=gen_cfg,
                                knobs=knobs,
                            )
                            pending.append(
                                _Pending(
                                    gen=gen, prompt_ids=prompt_ids, prompt_messages=prompt_messages
                                )
                            )
                            samples_seen += 1  # advances the liveness-thread progress signal
                            # Feed the stall clock during the (serial, up-to-~900s-each) generations;
                            # nseq is still 0 here since no loss has landed yet this step.
                            _opd_progress(opt_steps, nseq)
                # Phase 2 (concurrent, NETWORK only): fire the teacher scoring calls for every scorable
                # rollout at once, bounded by _TEACHER_SCORE_MAX_WORKERS. _score_one touches only the
                # stateless teacher HTTP client + Python strings (no torch/model/shared mutable state),
                # so the round-trips overlap safely; truncated/empty/U+FFFD rollouts are never scored.
                scorable = [i for i, p in enumerate(pending) if not (p.gen.skip or p.gen.truncated)]
                _opd_progress(opt_steps, nseq)  # refresh entering the (bounded) network phase
                if scorable:
                    permanent_err = None
                    with ThreadPoolExecutor(
                        max_workers=min(len(scorable), _TEACHER_SCORE_MAX_WORKERS)
                    ) as pool:
                        fut_to_idx = {
                            pool.submit(
                                _score_one,
                                teacher,
                                pending[i].gen,
                                prompt_messages=pending[i].prompt_messages,
                                thinking_prefill=thinking_prefill,
                            ): i
                            for i in scorable
                        }
                        for fut in as_completed(fut_to_idx):
                            i = fut_to_idx[fut]
                            try:
                                pending[i].score = fut.result()
                            except TeacherError as e:
                                # _score_one only propagates a PERMANENT teacher error (bad key / model
                                # id / malformed echo); transient/other failures return a status. Capture
                                # the first and re-raise AFTER the pool drains so the run aborts exactly
                                # as the serial path did over one un-scorable sample (codex[bot]).
                                if permanent_err is None:
                                    permanent_err = e
                    if permanent_err is not None:
                        raise permanent_err
                _opd_progress(opt_steps, nseq)  # refresh leaving the network phase
                # Phase 3 (serial, GPU train): resolve each rollout IN ORDER into the exact SampleResult
                # the old serial path produced, then run the gkd loss forward + backward serially on the
                # main thread (via _account) so gradient accumulation is unchanged.
                for p in pending:
                    _account(
                        _resolve_sample(model, tok, device, p.gen, p.score, p.prompt_ids, knobs)
                    )
            else:
                # Serial fallback: the student stand-in cannot generate on-policy (a CPU unit-test fake
                # without .generate), so there is nothing to overlap — drive the original per-sample
                # path so _train_one stays the single injection point those tests stub.
                for _ex, prompt_messages, prompt_ids in batch:
                    # Reuse the messages + ids CACHED by the budget filter instead of re-rendering. A
                    # fresh env.prompt_messages(ex) here can (for a stateful/randomized env) both desync
                    # student sampling from teacher scoring AND exceed the budget — dropping a prompt the
                    # filter admitted and starving the step of signal after GPU/model setup. The cached
                    # ids already passed the filter and drive BOTH the student prompt AND the teacher
                    # prompt, so they stay in sync and every visited prompt fits (codex[bot]).
                    prompt_tensor = torch.tensor([prompt_ids], device=device)
                    for _g in range(group):
                        r = _train_one(
                            model=model,
                            tok=tok,
                            teacher=teacher,
                            device=device,
                            prompt_ids=prompt_ids,
                            prompt_tensor=prompt_tensor,
                            prompt_messages=prompt_messages,
                            gen_cfg=gen_cfg,
                            knobs=knobs,
                            thinking_prefill=thinking_prefill,
                            # Refresh the stall clock between generation and teacher scoring (see
                            # _train_one and _opd_progress). Bind opt_steps/nseq as lambda defaults
                            # (called synchronously in this iteration, so the current values are the
                            # right ones) to satisfy the loop-var-capture lint.
                            on_generated=lambda s=opt_steps, n=nseq: _opd_progress(s, n),
                        )
                        samples_seen += 1  # advances the liveness-thread progress signal
                        _account(r)
            if nseq == 0:
                print(f"[opd] step {it}: no usable teacher signal this step (skipped)")
                continue
            # Each seq's grad was scaled by 1/accum_target; if some seqs were skipped (teacher call
            # failed / empty completion), rescale to a true 1/nseq mean so a partial step isn't a
            # silently smaller update.
            if nseq != accum_target:
                scale = accum_target / nseq
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            opt_steps += 1
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
                # Best-effort: a mid-run recombine/publish failure (e.g. transiently evicted SFT dir)
                # must not abort the loop after real optimizer steps — the finalize publish is strict.
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
            f"rollouts, {teacher_transient} transient teacher failures)"
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
    # Ship the deployable adapter (VL warm-start: recombine SFT⊕opd so it reproduces base+SFT+opd on
    # the catalog base; no-op for text/fresh). Name the final checkpoint by real optimizer steps
    # applied, not the planned `steps` count.
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
            # generating runaway/non-terminating filters (cold start, untrained stop token) and
            # warm-starting from SFT — which encodes termination — would help.
            "truncated_rollouts": truncated_rollouts,
            # Real alignment-health signal (mean student-tokens-per-group); mean_coverage is ~1.0 even
            # for a degenerate collapsed alignment, so it can't flag that failure mode.
            "mean_align_granularity": (granularity_sum / granularity_n) if granularity_n else 0.0,
            "teacher_input_tokens": teacher_input_tokens,
            "temperature": knobs.temperature,
            "group_size": group,
            "prompts_per_step": ppl_step,
            "max_completion_len": knobs.max_completion,
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
    free_gpu(model)


@dataclass(frozen=True)
class SampleResult:
    """One student sample's outcome, returned by ``_train_one`` for ``run_opd`` to aggregate.

    ``loss`` is the groupwise reverse-KL loss tensor when the sample was distilled, else ``None``
    (the sample was skipped — truncated rollout, empty completion, or no teacher signal). The stats
    describe what happened so the caller can count teacher health / truncations and, on a no-loss run,
    decide whether it is a retriable infra failure."""

    loss: object = None  # torch scalar tensor when distilled, else None (module is torch-free)
    # "ok" once teacher.score returns, "transient" on a retryable teacher outage, else None (teacher
    # not reached). run_opd uses this to decide whether a no-signal run is a retriable infra failure.
    teacher_status: str | None = None
    # A rollout that didn't terminate naturally — cap hit OR max_time cut, no EOS/stop (skipped, not
    # distilled — see _rollout_terminated).
    truncated: bool = False
    coverage: float = 0.0
    gen_tokens: int = 0
    teacher_tokens: int = 0
    # Mean student-tokens-per-alignment-group; a real health signal where coverage is not.
    group_granularity: float = 0.0


@dataclass(frozen=True)
class _GenResult:
    """One student rollout produced by ``_generate_one`` (serial, GPU eval): the sampled completion
    plus the skip/truncated verdicts the old serial ``_train_one`` computed inline. ``truncated``
    (max_new_tokens cap hit OR max_time cut, no EOS/stop) and ``skip`` (empty or U+FFFD completion)
    mean the rollout is dropped BEFORE teacher scoring; otherwise ``completion_ids``/``completion_text``
    carry the trimmed on-policy answer to score + distil. Torch-free (completion_ids is a CPU list) so
    it can be handed to the model-free scoring thread pool."""

    completion_ids: object = None
    completion_text: str = ""
    gen_tokens: int = 0
    truncated: bool = False
    skip: bool = False


@dataclass(frozen=True)
class _ScoreResult:
    """Teacher-scoring outcome from ``_score_one`` (RUN IN THE THREAD POOL). ``status`` is "ok"
    (``teacher_toks`` populated), "transient" (retryable teacher outage -> sample skipped + counted),
    or "error" (any other exception -> sample skipped, teacher uncounted). A PERMANENT ``TeacherError``
    is NOT represented here — ``_score_one`` re-raises it so the run aborts, exactly as before."""

    teacher_toks: object = None
    status: str = "ok"


@dataclass
class _Pending:
    """A Phase-1 rollout awaiting concurrent scoring (Phase 2) then loss (Phase 3), carrying the prompt
    context both need. Mutable: ``score`` is filled in by the thread pool for scorable rollouts."""

    gen: _GenResult
    prompt_ids: object
    prompt_messages: object
    score: object = None


def _generate_one(*, model, tok, device, prompt_tensor, gen_cfg, knobs) -> _GenResult:
    """[SERIAL, GPU eval] Sample one student completion on-policy and apply the pre-scoring gates:
    reject a rollout that did not terminate naturally (``truncated``), an empty completion or one
    carrying U+FFFD (``skip``), and trim trailing stop delimiters. Touches only the torch model +
    tokenizer; the returned ``_GenResult`` is torch-free so it can be handed to the (model-free)
    scoring thread pool. Mirrors the generation half of the old serial ``_train_one`` exactly."""
    import torch

    # On-policy: the student samples; the teacher echo-scores that exact completion.
    model.eval()
    model.config.use_cache = True
    with torch.no_grad():
        gen = model.generate(prompt_tensor, **gen_cfg)
    completion_ids = _to_cpu_ids(
        gen[0, prompt_tensor.shape[1] :]
    )  # one GPU->CPU copy, reused below
    completion_text = tok.decode(completion_ids, skip_special_tokens=True)  # teacher/alignment text
    # Stop detection + trimming run on a decode that KEEPS special tokens: a [train] stop_sequence can
    # be a tokenizer special token (e.g. <|im_end|>) that skip_special_tokens=True strips, so the clean
    # text no longer ends with the delimiter and the rollout is misread as truncated / never trimmed —
    # burning every usable sample for that config (codex[bot]). HF stop_strings halts AT the delimiter
    # (no EOS can trail it when stops are configured), so this raw tail is unambiguous.
    stop_text = tok.decode(completion_ids, skip_special_tokens=False)
    # Skip a rollout that did NOT terminate naturally (no EOS, no stop delimiter) BEFORE scoring/
    # distilling it: a max_new_tokens cap hit OR a gen_cfg.max_time cut leaves a filter cut off
    # mid-output, which OPD would otherwise echo-score and reinforce, teaching a runaway it can never
    # learn to end (OPD can't supervise the stop token). Checked on the RAW rollout text, before
    # _trim_trailing_stop removes the delimiter — so a stop-terminated sample AT the cap is KEPT, not
    # discarded (codex[bot]). _generation_eos_ids gathers EVERY halting id (tokenizer + generation_
    # config, either of which may be a list) so a model that stops on a secondary eos isn't misread as
    # truncated; a fake/EOS-less tokenizer yields an empty set -> fail-open in the helper.
    if not _rollout_terminated(
        completion_ids, stop_text, _generation_eos_ids(model, tok), knobs.stop_sequences
    ):
        return _GenResult(truncated=True, gen_tokens=len(completion_ids))
    # `stop_sequences` halt generation on-policy (gen_cfg.stop_strings), but HF emits the delimiter
    # before stopping — trim it from BOTH ids and text (token-level) so the teacher scores/distils
    # only the answer, and ids/text stay consistent for gkd_loss + token counting.
    if knobs.stop_sequences:
        completion_ids, completion_text = _trim_trailing_stop(
            tok, completion_ids, stop_text, knobs.stop_sequences
        )
    gen_tokens = len(completion_ids)
    if not completion_text.strip():
        return _GenResult(skip=True, gen_tokens=gen_tokens)
    # A completion carrying U+FFFD (the Unicode replacement char) decoded a PARTIAL/invalid UTF-8 byte
    # sequence — the student emitted a lone byte-level BPE token that is not a whole character. Echo-
    # scoring it is impossible: the replacement char does not re-tokenize to the same char span, so
    # teacher.score fails its char-for-char tiling check and raises a PERMANENT TeacherError that would
    # abort the ENTIRE run over one un-scorable on-policy sample. On-policy sampling (esp. at temperature
    # 1.0) occasionally emits these, so skip the rollout like a truncated/empty one and keep training;
    # the shortfall guard still fails the run if too many samples end up unusable.
    if "\ufffd" in completion_text:  # U+FFFD replacement char
        return _GenResult(skip=True, gen_tokens=gen_tokens)
    return _GenResult(
        completion_ids=completion_ids, completion_text=completion_text, gen_tokens=gen_tokens
    )


def _score_one(teacher, gen_result, *, prompt_messages, thinking_prefill) -> _ScoreResult:
    """[THREAD POOL — network only] Build the teacher prompt and echo-score the completion over the
    API. MUST NOT touch the torch model or any shared mutable state — it reads only the completion
    string + prompt messages and calls the stateless teacher HTTP client, so it is safe to run
    concurrently for every scorable sample in a step. ``thinking_prefill`` is appended to the teacher
    prompt so it conditions on the same trailing context the student sampled after in thinking mode.
    Error semantics match the old serial ``_train_one`` exactly (same print messages): a PERMANENT
    ``TeacherError`` propagates (the run aborts), a transient one -> status "transient", any other
    exception -> status "error"; both leave the sample skipped with no teacher signal."""
    teacher_prompt = _teacher_prompt_text(prompt_messages, thinking_prefill)
    try:
        teacher_toks = teacher.score(teacher_prompt, gen_result.completion_text)
    except TeacherError as e:
        if e.permanent:  # bad key / model id / malformed -> abort now, don't burn the whole run
            raise
        print(f"[opd] teacher score failed (transient, skipping sample): {e}")
        return _ScoreResult(status="transient")
    except Exception as e:
        print(f"[opd] teacher score failed (skipping sample): {e}")
        return _ScoreResult(status="error")
    return _ScoreResult(teacher_toks=teacher_toks, status="ok")


def _loss_one(model, tok, device, gen_result, score_result, prompt_ids, knobs) -> SampleResult:
    """[SERIAL, GPU train] Turn one scored rollout into the differentiable groupwise reverse-KL loss.
    Runs the model in train mode with use_cache=False and reproduces the loss half of the old serial
    ``_train_one``: student token offsets, groupwise alignment / coverage / granularity, and gkd_loss.
    Returns a ``SampleResult`` with identical fields — loss=None (teacher_status "ok") when the
    completion yields no alignable student tokens, else the loss tensor plus per-sequence stats."""
    teacher_tokens = len(prompt_ids) + gen_result.gen_tokens
    student_ids, student_toks = student_tokens_with_offsets(
        tok, gen_result.completion_ids, gen_result.completion_text
    )
    if not student_ids:
        return SampleResult(
            teacher_status="ok", gen_tokens=gen_result.gen_tokens, teacher_tokens=teacher_tokens
        )

    model.train()
    model.config.use_cache = False
    # gkd — groupwise reverse-KL (spider/Tinker); covers every token from the realized logprobs.
    groups = groupwise_alignment(student_toks, score_result.teacher_toks)
    # Coverage = alignable (non-zero-width) student tokens that landed in a group / alignable total,
    # so it stays in [0, 1] (a zero-width eos/partial-byte token riding along in a group no longer
    # inflates it past 100%).
    coverage = groupwise_coverage(groups, student_toks)
    # mean_coverage is structurally ~1.0 for a CORRECT fine-grained alignment AND for a degenerate
    # collapsed-into-one-giant-span alignment, so it can't detect the latter. Mean student-tokens-per-
    # group does: ~1.0 == each token its own group (healthy); large == coarse spans smearing one
    # teacher logprob across many student tokens.
    _n_align = sum(1 for st in student_toks if st.end > st.start)
    group_granularity = (_n_align / len(groups)) if groups else 0.0
    return SampleResult(
        loss=gkd_loss(model, prompt_ids, student_ids, groups, device, kl_coef=knobs.kl_coef),
        teacher_status="ok",
        coverage=coverage,
        gen_tokens=gen_result.gen_tokens,
        teacher_tokens=teacher_tokens,
        group_granularity=group_granularity,
    )


def _resolve_sample(model, tok, device, gen, score, prompt_ids, knobs) -> SampleResult:
    """Map a generated (and, when scorable, teacher-scored) rollout to its ``SampleResult`` — the single
    truncated/skip/transient/error/ok decision shared by ``run_opd``'s concurrent Phase 3 and the serial
    ``_train_one``, so the two paths can never drift apart on skip semantics. ``score`` is consulted only
    for a rollout that was actually scored (not truncated, not skip); callers pass ``None`` for the rest."""
    if gen.truncated:
        # Didn't terminate naturally (cap/max_time cut) — skipped, not distilled.
        return SampleResult(truncated=True, gen_tokens=gen.gen_tokens)
    if gen.skip:  # empty completion or U+FFFD — skipped before scoring
        return SampleResult(gen_tokens=gen.gen_tokens)
    if score.status == "transient":  # retryable teacher outage — skipped + counted
        return SampleResult(teacher_status="transient", gen_tokens=gen.gen_tokens)
    if score.status != "ok":  # any other teacher exception — skipped, teacher uncounted
        return SampleResult(gen_tokens=gen.gen_tokens)
    return _loss_one(model, tok, device, gen, score, prompt_ids, knobs)


def _train_one(
    *,
    model,
    tok,
    teacher,
    device,
    prompt_ids,
    prompt_tensor,
    prompt_messages,
    gen_cfg,
    knobs,
    thinking_prefill="",
    on_generated=None,
) -> SampleResult:
    """Sample one student completion on-policy, score it with the teacher, and return a
    ``SampleResult`` carrying the groupwise reverse-KL loss (or None) plus per-sequence stats. This is
    the SERIAL composition of the three phase helpers — ``_generate_one`` -> ``_score_one`` ->
    ``_loss_one`` — that ``run_opd`` runs as OVERLAPPED phases across a whole step (generate every
    rollout serially on the GPU, score them all concurrently over the API, then run each loss serially).
    Kept as the single-sample entry point for the unit tests and the fallback for a student that cannot
    batch-generate. ``thinking_prefill`` is appended to the teacher prompt so it conditions on the same
    trailing context the student sampled after in thinking mode. ``on_generated`` (optional) is called
    AFTER generation and BEFORE teacher scoring to refresh the stall clock: both the max_time-bounded
    generate and the retrying teacher call block for a long time and the caller's per-sample progress
    ping only fires AFTER scoring returns, so without a mid-sample refresh a slow generation followed by
    a teacher outage can span >1200s with no non-liveness heartbeat and be reaped as stalled before the
    transient-teacher handling runs."""
    gen = _generate_one(
        model=model, tok=tok, device=device, prompt_tensor=prompt_tensor, gen_cfg=gen_cfg, knobs=knobs
    )
    if gen.truncated or gen.skip:  # dropped before scoring (see _resolve_sample); score unused
        return _resolve_sample(model, tok, device, gen, None, prompt_ids, knobs)
    # Refresh the stall clock between the (bounded, up to ~900s) generation and the retrying teacher
    # call (up to four ~90s timeouts); both block and the caller only pings AFTER scoring returns.
    if on_generated is not None:
        on_generated()
    score = _score_one(
        teacher, gen, prompt_messages=prompt_messages, thinking_prefill=thinking_prefill
    )
    return _resolve_sample(model, tok, device, gen, score, prompt_ids, knobs)

def _save_adapter(model, tok, adapter_dir: str) -> None:
    """Persist the LoRA adapter + tokenizer for deploy (identical layout to SFT)."""
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)


def _publish_opd_deployable(
    adapter_dir: str, step: int, *, as_default: bool, best_effort: bool = False
) -> None:
    """Publish the step-``step`` deployable adapter (and, when ``as_default``, the ``<prefix>/adapter``
    served default), recombining SFT⊕opd for a VL warm-start so the deployed adapter reproduces
    base+SFT+opd on the unmodified catalog base. For a VL warm-start the trained ``adapter_dir`` is
    SFT-less (a fresh LoRA on the SFT-merged base); ``recombined_warmstart_adapter_dir`` stacks the
    original SFT LoRA back in. No-op recombine (deploys ``adapter_dir`` as-is) for the continued-
    adapter (non-VL) and fresh-LoRA paths, which already carry the SFT. Mirrors GRPO finalize (rl.py).

    ``best_effort`` (mid-run per-step publish): swallow a recombine/publish failure and KEEP training —
    a transient or evicted SFT dir during a save_every publish must not terminate run_opd after real
    optimizer steps (GRPO's per-step checkpoint callback is likewise best-effort). At finalize
    (``best_effort=False``) a recombine failure is FATAL: shipping the SFT-less adapter as the served
    default is exactly the broken deploy the recombine guards against."""
    try:
        recombined = _w.recombined_warmstart_adapter_dir(adapter_dir)
    except Exception as e:
        if not best_effort:
            raise
        print(
            f"[opd] deployable recombine failed at step {step}; "
            f"skipping this publish, training continues: {e}"
        )
        return
    deploy_dir = recombined or adapter_dir
    try:
        if as_default:
            _w.hf_upload_folder(deploy_dir, "adapter", required=True)
        _w.publish_deployable_checkpoint(deploy_dir, step)
    except Exception as e:
        if not best_effort:
            raise
        print(f"[opd] deployable publish failed at step {step}; skipping, training continues: {e}")
    finally:
        if recombined:
            import shutil

            shutil.rmtree(recombined, ignore_errors=True)
