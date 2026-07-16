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
shape. Length-capped rollouts bypass teacher scoring, contribute no loss, and are counted in run
metadata. All heavy imports (torch/transformers/peft/vllm) are inside functions, so importing this
module is CPU/offline-safe.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from flash.engine.chalk_kernels import active_kernels, install_chalk_kernels
from flash.engine.recipe import RECIPE, resolve_teacher
from flash.engine.steps import (
    final_save_due,
    on_policy_steps,
    resolve_update_horizon,
    save_step_due,
    validate_save_steps,
)
from flash.engine.structured_outputs import (
    describe_structured_outputs,
    parse_structured_outputs,
    reasoning_parser_for,
)
from flash.engine.vram import opd_completion_len
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.hf import RequiredSaveError, _deployable_adapter_on_hf
from flash.engine.worker.opd_gkd import (
    _generation_eos_ids,
    _rollout_terminated,
    _teacher_prompt_text,
    _trim_trailing_stop,
    student_tokens_with_offsets,
)
from flash.engine.worker.opd_vllm import (
    OpdVllmOutput,
    OpdVllmRolloutEngine,
)
from flash.engine.worker.opd_vllm import (
    opd_lora_rank as _opd_lora_rank,
)
from flash.engine.worker.opd_vllm import (
    opd_vllm_kwargs as _opd_vllm_kwargs,
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
from flash.engine.worker.rng import (
    backend_seed,
    capture_training_rng_state,
    restore_training_rng_state,
    rollout_request_seed,
    seed_training_rngs,
)
from flash.engine.worker.teacher import TeacherError
from flash.engine.worker.tokenizer_align import groupwise_alignment, groupwise_coverage
from flash.opd_retry_contract import (
    OPD_RESUME_STATE_VERSION,
    opd_resume_checkpoint_complete,
    validate_opd_resume_state_metadata,
)

OPD_ROLLOUT_PIPELINE_TARGET_CHUNK_SIZE = 16
OPD_ROLLOUT_PIPELINE_MAX_CHUNKS = 8
OPD_TEACHER_BATCH_SIZE = 8
OPD_LOSS_MICROBATCH_SIZE = 4


def _cumulative_train_wall_seconds(baseline: float, started_at: float, now: float) -> float:
    """return cumulative training-loop wall time across replacement attempts."""
    elapsed = max(0.0, float(now) - float(started_at))
    total = float(baseline) + elapsed
    if not math.isfinite(total) or total < 0:
        raise RuntimeError("opd cumulative training wall time is invalid")
    return total


def _require_restored_opd_state(state: dict | None) -> dict:
    """fail terminally when a downloaded checkpoint cannot be restored."""
    if state is None:
        raise RuntimeError(
            "opd resume checkpoint is present but could not be restored; "
            "refusing to train fresh after possible prior optimizer mutation"
        )
    return state


def _reconcile_required_opd_deployable(
    checkpoint_dir: str, opt_steps: int, save_at_steps: tuple[int, ...]
) -> None:
    """Publish a missed required companion directly from the restored checkpoint."""
    if opt_steps not in save_at_steps or _deployable_adapter_on_hf(opt_steps):
        return
    _publish_opd_deployable(
        checkpoint_dir,
        opt_steps,
        as_default=False,
        save_required=True,
    )


def _opd_prompt_pool_fingerprint(examples: list[tuple[object, object, list[int]]]) -> str:
    """hash the exact ordered filtered prompt pool without exposing prompt content."""
    digest = hashlib.sha256()
    for example, messages, prompt_ids in examples:
        encoded = json.dumps(
            [example, messages, [int(token) for token in prompt_ids]],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _apply_opd_optimizer_update(
    *, model, optimizer, torch, opt_steps: int, nseq: int, accum_target: int
) -> int:
    """Apply one OPD update with the mutation marker immediately before optimizer mutation."""
    if nseq <= 0 or accum_target <= 0:
        raise ValueError("opd optimizer update requires positive sequence counts")
    if nseq != accum_target:
        scale = accum_target / nseq
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    if opt_steps == 0:
        _w.publish_opd_optimizer_start_marker()
    optimizer.step()
    return opt_steps + 1


def _opd_rollout_pipeline_target_chunk_size(total_prompts: int) -> int:
    total = max(1, int(total_prompts))
    return max(1, min(total, OPD_ROLLOUT_PIPELINE_TARGET_CHUNK_SIZE))


def _opd_rollout_pipeline_max_chunks(total_prompts: int) -> int:
    total = max(1, int(total_prompts))
    return max(1, min(total, OPD_ROLLOUT_PIPELINE_MAX_CHUNKS))


def _opd_rollout_pipeline_chunks(total_prompts: int) -> int:
    """Number of single-turn rollout chunks in one OPD step.

    Small steps still split once so teacher scoring can overlap later vLLM generation. Larger steps
    target moderately sized vLLM batches, which starts remote teacher scoring earlier without reducing
    rollout generation to one request per prompt.
    """
    total = max(1, int(total_prompts))
    if total < 8:
        return 1
    target_chunk = _opd_rollout_pipeline_target_chunk_size(total)
    max_chunks = _opd_rollout_pipeline_max_chunks(total)
    chunks = (total + target_chunk - 1) // target_chunk
    if max_chunks == 1:
        return 1
    return max(2, min(max_chunks, chunks))


def _opd_rollout_chunk_size(total_prompts: int) -> int:
    """Split a single OPD step into a small number of rollout chunks so remote teacher scoring for
    earlier chunks overlaps with vLLM generation for later chunks without collapsing vLLM batching."""
    total = max(1, int(total_prompts))
    chunks = _opd_rollout_pipeline_chunks(total)
    return max(1, (total + chunks - 1) // chunks)


def _opd_teacher_batch_size(total_samples: int) -> int:
    total = max(1, int(total_samples))
    return max(1, min(total, OPD_TEACHER_BATCH_SIZE))


def _opd_teacher_workers(total_samples: int, batch_size: int) -> int:
    total = max(1, int(total_samples))
    return max(1, (total + max(1, int(batch_size)) - 1) // max(1, int(batch_size)))


def _opd_loss_microbatch_size(model_id: str, total_samples: int) -> int:
    total = max(1, int(total_samples))
    try:
        from flash.engine.vram import resolve_params_b

        params_b = float(resolve_params_b(model_id) or 0.0)
    except Exception:
        params_b = 0.0
    # The dense vocab logits from GKD are the peak. Batch small/medium dense models, but keep 35B-class
    # OPD serial by default so the B200 path does not trade speed for OOM risk.
    default = OPD_LOSS_MICROBATCH_SIZE if params_b and params_b <= 10.0 else 1
    return max(1, min(total, default))


@dataclass(frozen=True)
class OpdKnobs:
    """Resolved opd knobs from the JobSpec's [train] table (falling back to RECIPE.opd), returned by
    ``_resolve_opd_knobs``. A typed container replacing the old stringly-typed dict; field names match
    the former dict keys one-for-one. The defaults are placeholders only — ``_resolve_opd_knobs``
    always sets every field explicitly — kept so partial construction stays ergonomic for tests."""

    teacher_model: str = ""
    teacher_base_url: str = ""
    epochs: int = RECIPE.opd.num_epochs
    learning_rate: float = 0.0
    temperature: float = 0.0
    top_p: float = 1.0
    max_completion: int = 0
    prompts_per_step: int = 0
    group_size: int = 0
    # gkd reverse-KL scale; reuses the existing [train] kl_penalty_coef knob (default 1.0).
    kl_coef: float = 1.0
    save_every: int = 0
    max_steps: int = 0
    save_at_steps: tuple[int, ...] = ()
    max_length: int = 0
    # Student on-policy sampling stops at these delimiters (parity with GRPO), so the teacher never
    # scores/trains on text past the intended answer boundary.
    stop_sequences: tuple = ()
    # Canonical StructuredOutputsParams kwargs JSON from [train] structured_outputs ("" = off);
    # constrains every student rollout (parity with GRPO). Teacher echo-scoring needs no schema —
    # it scores the student's already-constrained tokens.
    structured_outputs: str = ""


def _resolve_opd_knobs() -> OpdKnobs:
    """Resolve every opd knob from the JobSpec's [train] table, falling back to RECIPE.opd."""
    d = RECIPE.opd
    t = _w.JOB_SPEC.train if _w.JOB_SPEC else None

    def opt(name, default):
        v = getattr(t, name, None) if t else None
        return v if v is not None else default

    # kl_penalty_coef IS the gkd distillation objective's scale: the OPD loss multiplies every span
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
    # Resolve the managed teacher from [train].teacher_model (the resolved Fireworks model id, "" =>
    # the GLM 5.2 default). Parse already validated + canonicalized it, but JobSpec.from_dict is a
    # tolerant deserializer, so re-validate at this boundary (like the kl_coef guard above): resolve
    # is idempotent for a canonical model id, and a spec that reaches the worker with an unsupported
    # teacher fails loudly here rather than as an opaque Fireworks 404 mid-run. base_url is shared by
    # every allow-listed teacher (one Fireworks endpoint + one managed key), so it stays d.teacher_base_url.
    try:
        teacher = resolve_teacher(opt("teacher_model", ""))
    except ValueError as e:
        raise RuntimeError(f"opd: {e}") from e
    return OpdKnobs(
        teacher_model=teacher.model_id,
        teacher_base_url=d.teacher_base_url,
        epochs=int(t.epochs) if t and t.epochs is not None else d.num_epochs,
        learning_rate=float(opt("learning_rate", 0) or d.learning_rate),
        temperature=float(
            opt("temperature", None)
            if (t and t.temperature is not None)
            else d.sampling_temperature
        ),
        top_p=d.sampling_top_p,
        max_completion=opd_completion_len(opt("max_completion_tokens", 0), _w.THINKING),
        prompts_per_step=int(opt("batch_size", 0) or d.prompts_per_step),
        group_size=int(opt("group_size", 0) or d.group_size),
        kl_coef=kl_coef,
        save_every=int(opt("save_every", 0) or 20),
        max_steps=int(opt("max_steps", 0) or 0),
        save_at_steps=tuple(getattr(t, "save_at_steps", ()) or ()),
        max_length=int(opt("max_context_tokens", 0) or 0),
        stop_sequences=tuple(getattr(t, "stop_sequences", ()) or ()),
        structured_outputs=str(getattr(t, "structured_outputs", "") or ""),
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
            open_tag = (
                "<" + base_mid_tag[2 : base_mid_tag.index(">") + 1]
            )  # "</think>..." -> "<think>"
            cut = think.rfind(open_tag, 0, p)
            if cut != -1:
                return think[cut:]  # e.g. "<think>\n"
        if think_mid:
            return think_mid  # opener appended, or inserted before shared trailing template text
    return ""


def _student_model(model_id, model_init_kwargs, device):
    """Build the trainable student LoRA and return ``(model, rollout_model_source)``.

    Warm-starts from ``train.init_from_adapter`` when set — continuing a prior run's adapter (e.g. an
    SFT checkpoint), the same path GRPO uses via ``_init_adapter_model`` — otherwise a fresh LoRA on
    the base. This makes an SFT->opd pipeline a genuine continuation (the opd stage keeps the SFT
    behavior) rather than silently restarting from base.

    Warm-start (``init_peft is None``) returns a trainable PeftModel that CONTINUES the prior adapter
    in place — VL and non-VL alike — so the saved/deployed adapter is that same rank-r adapter on the
    catalog base (no merge, no recombine). Only the FRESH-LoRA path (``init_peft`` is a config) builds
    a new adapter here, loading the full multimodal model for VL so all-linear LoRA sees every target.
    """
    init_model, init_peft = _w._init_adapter_model(model_id)
    job_spec = getattr(_w, "JOB_SPEC", None)
    model_revision = getattr(job_spec, "model_revision", "") if job_spec else ""
    if init_peft is None:
        # init_model is already a trainable PeftModel continuing the prior (e.g. SFT) adapter.
        return init_model.to(device), model_id
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    loader_revision = str(model_init_kwargs.get("revision") or "").strip()
    if loader_revision and model_revision and loader_revision != model_revision:
        raise ValueError("OPD architecture probe revision must match from_pretrained revision")
    effective_revision = model_revision or loader_revision
    model_cls = AutoModelForCausalLM
    if _w.is_vl_checkpoint(model_id, revision=effective_revision):
        # VL checkpoints are trained/served on the full multimodal tree, including visual linears, so
        # a fresh LoRA must target that same tree (parity with the warm-start / serving module set).
        from transformers import AutoModelForImageTextToText

        model_cls = AutoModelForImageTextToText

    # init_model is the base id (fresh run). VL runs load the full multimodal model so all-linear LoRA
    # sees every target; non-VL runs keep the lighter causal-LM loader.
    from flash.engine.worker.hf import model_revision_kwargs

    base = model_cls.from_pretrained(
        init_model,
        trust_remote_code=True,
        **model_init_kwargs,
        **{
            key: value
            for key, value in model_revision_kwargs(model_revision).items()
            if key not in model_init_kwargs
        },
    ).to(device)
    return get_peft_model(base, init_peft), str(init_model)


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


def _format_skip_counts(counts: Counter[str]) -> str:
    """Compact human-readable summary for OPD no-signal diagnostics."""

    if not counts:
        return "unknown=0"
    return ", ".join(f"{k}={counts[k]}" for k in sorted(counts))


def run_opd():
    import torch

    from flash.engine.worker.hf import load_tokenizer, model_revision_kwargs
    from flash.engine.worker.teacher import TeacherClient

    seed_training_rngs(_w.SEED)
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

    wait_for_gpu(
        _w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None,
        exact_type=_w.JOB_SPEC.gpu.exact_type if _w.JOB_SPEC else "",
    )
    setup_perf_backends()
    model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else RECIPE.hf_model_id
    model_revision = getattr(_w.JOB_SPEC, "model_revision", "") if _w.JOB_SPEC else ""
    # Tokenizer only (a few small files) up front -- the prompt-budget filter below needs it. The FULL
    # base-weight prefetch (tens of GB) is deferred until AFTER the filter confirms a non-empty pool, so
    # a dataset whose every prompt is over-budget fails fast without paying for the download (codex[bot]).
    tok = load_tokenizer(model_id, revision=model_revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Thinking-mode student prompts open a reasoning block (e.g. Qwen's <think>) after the generation
    # prompt; the teacher must condition on the SAME prefill or its logprobs score a different prefix
    # than the sampled tokens (see _thinking_prefill_text). Compute once — the template is fixed.
    thinking_prefill = _thinking_prefill_text(tok)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _attn = optimal_attn_impl()
    model_init_kwargs = {
        "dtype": torch.bfloat16,
        **model_revision_kwargs(model_revision),
    }
    if _attn:
        model_init_kwargs["attn_implementation"] = _attn
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
    prompts_per_step = knobs.prompts_per_step
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
    # step loop visits a deterministic examples[(step*prompts_per_step+i) % len] slice, so a whole-dataset
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
    prompt_pool_fingerprint = _opd_prompt_pool_fingerprint(examples)
    if n_over_budget:
        print(
            f"[opd] filtered {n_over_budget}/{len(train)} prompts over the "
            f"{prompt_budget}-token budget; pool = {len(examples)}"
        )
    if prompts_per_step > len(examples):
        print(
            f"[opd] lowering prompts_per_step from {prompts_per_step} to {len(examples)}: "
            "only that many prompt(s) fit after filtering"
        )
        prompts_per_step = len(examples)
    derived_steps = on_policy_steps(
        epochs=knobs.epochs,
        prompt_count=len(examples),
        prompts_per_step=prompts_per_step,
    )
    steps = resolve_update_horizon(derived_steps, knobs.max_steps)
    validate_save_steps(knobs.save_at_steps, steps)
    print(
        f"[opd] epochs={knobs.epochs} over {len(examples)} retained prompt(s) at "
        f"{prompts_per_step} prompts/step -> derived_steps={derived_steps} "
        f"update_horizon={steps}"
    )

    # Now that a non-empty on-policy pool is confirmed, prefetch the full base weights (deferred from
    # setup so an all-over-budget dataset fails before this download). Still inside the setup phase
    # (opt_steps==0 -> wide poller grace), same as when it ran earlier.
    download_seconds = (
        _w.prefetch_model(model_id, revision=model_revision)
        if model_revision
        else _w.prefetch_model(model_id)
    )

    # reseed before stochastic student and adapter construction.
    seed_training_rngs(_w.SEED)

    setup_seconds = time.time() - t_start
    _w.heartbeat("opd_model_load", setup_seconds=setup_seconds, gpu=gpu_diagnostics())
    with liveness_heartbeat("opd_initializing"):
        model, rollout_model_source = _student_model(model_id, model_init_kwargs, device)
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
        if grad_checkpointing_on(model_id, seq_cap, revision=model_revision):
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

    # resolve the model/tokenizer halt set once for vllm sampling and termination classification.
    generation_eos_ids = _generation_eos_ids(model, tok)
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
    _reasoning_parser = reasoning_parser_for(thinking=_w.THINKING, structured_outputs=_so_spec)
    if _so_spec:
        print(
            f"[opd] structured outputs: every student rollout constrained to "
            f"{describe_structured_outputs(_so_spec)}"
            + (
                f" (applied only after </think> via reasoning_parser={_reasoning_parser})"
                if _reasoning_parser
                else ""
            )
        )
    t_vllm_init = time.time()
    with liveness_heartbeat("opd_vllm_initializing"):
        vllm_rollout = OpdVllmRolloutEngine(
            model_source=rollout_model_source,
            model_revision=model_revision,
            max_model_len=seq_cap,
            temperature=knobs.temperature,
            top_p=knobs.top_p,
            stop_sequences=tuple(str(s) for s in knobs.stop_sequences),
            eos_token_ids=tuple(sorted(generation_eos_ids)),
            structured_outputs=_so_spec,
            reasoning_parser=_reasoning_parser,
            lora_rank=lora_rank,
            seed=backend_seed(_w.SEED),
            **vllm_kwargs,
        )
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

    # full-state resume matches sft/rl, which resume through trainer.train(resume_from_checkpoint).
    # download errors fail closed to a retry so a replacement never trains fresh after possible mutation;
    # confirmed checkpoint absence still means this is a legitimate fresh start.
    resume_revision = _w.OPD_RESUME_REVISION or None
    resume_ckpt = _w.hf_resume_checkpoint(
        fail_closed=bool(resume_revision), revision=resume_revision
    )
    resume_state = None
    if resume_ckpt:
        resume_state = _require_restored_opd_state(
            _restore_opd_full_state(
                resume_ckpt,
                model=model,
                optimizer=optimizer,
                torch=torch,
                max_opt_steps=steps,
                prompt_pool_fingerprint=prompt_pool_fingerprint,
            )
        )
        _reconcile_required_opd_deployable(
            resume_ckpt,
            int(resume_state["opt_steps"]),
            knobs.save_at_steps,
        )
        print(
            f"[opd] resumed from {resume_ckpt}: opt_steps={resume_state['opt_steps']} "
            f"step={resume_state['step']}"
        )

    # synchronize exactly once before the first rollout. resumed runs reconcile the restored required
    # checkpoint companion first, so an initial sync failure cannot strand a missing deployable.
    with liveness_heartbeat("opd_vllm_initializing"):
        vllm_rollout.sync_from_model(model)

    def _resumed(key, default):
        """Restored accumulator value, or the fresh-start default when not resuming."""
        if resume_state is not None and key in resume_state:
            return resume_state[key]
        return default

    out_dir = f"/tmp/opd_seed{_w.SEED}"
    adapter_dir = f"{out_dir}/adapter"
    os.makedirs(out_dir, exist_ok=True)

    loss_curve: list[float] = list(_resumed("loss_curve", []))
    coverage_curve: list[float] = list(_resumed("coverage_curve", []))
    generated_tokens = int(_resumed("generated_tokens", 0))
    teacher_input_tokens = int(_resumed("teacher_input_tokens", 0))
    # length-capped rollouts and alignment granularity are surfaced in train_meta for diagnosis.
    truncated_rollouts = int(_resumed("truncated_rollouts", 0))
    granularity_sum = float(_resumed("granularity_sum", 0.0))
    granularity_n = int(_resumed("granularity_n", 0))
    # optimizer steps actually applied (< steps if any iteration had no teacher signal). on resume this
    # is the restored count and loss_curve/coverage_curve above carry the matching per-step history, so
    # the honest-metrics invariant (loss_curve length == opt_steps) holds across the restart.
    opt_steps = int(_resumed("opt_steps", 0))
    _reset_peak = getattr(_w, "_reset_peak_gpu", None)
    if _reset_peak:
        _reset_peak()

    # Monotonic count of student samples ATTEMPTED (generate+score), across all steps. Fed to
    # liveness_heartbeat as its progress signal so the liveness thread emits a REAL (non-liveness)
    # opd_step heartbeat whenever a sample advances — parity with sft/rl. Without a progress callback
    # opd's liveness thread emitted only liveness=True pings that share the opd_step upload-throttle
    # slot and could suppress the main thread's per-sample progress uploads (codex[bot]).
    samples_seen = int(_resumed("samples_seen", 0))
    rollout_seed_ordinal = int(_resumed("rollout_seed_ordinal", 0))
    # No-signal accounting: distinguish a run that trained nothing because the TEACHER was down
    # (transient) from one where scoring succeeded but never aligned — the former is retriable infra.
    teacher_ok = int(_resumed("teacher_ok", 0))
    teacher_transient = int(_resumed("teacher_transient", 0))
    # restored failures remain cumulative; only new failures make this attempt's shortfall retriable.
    teacher_transient_baseline = teacher_transient
    teacher_error = int(_resumed("teacher_error", 0))
    skip_counts: Counter[str] = Counter(_resumed("skip_counts", {}))
    no_signal_resamples = int(_resumed("no_signal_resamples", 0))
    no_signal_skipped_steps = int(_resumed("no_signal_skipped_steps", 0))

    def _generate_with_rollout_seeds(prompt_ids_batch, *, max_tokens):
        nonlocal rollout_seed_ordinal
        request_seeds = [
            rollout_request_seed(_w.SEED, rollout_seed_ordinal + offset)
            for offset in range(len(prompt_ids_batch))
        ]
        generations = _generate_many_vllm(
            vllm_rollout,
            tok,
            prompt_ids_batch,
            knobs,
            generation_eos_ids,
            max_tokens=max_tokens,
            request_seeds=request_seeds,
        )
        if len(generations) != len(prompt_ids_batch):
            raise RuntimeError(
                f"opd vLLM returned {len(generations)} generations for "
                f"{len(prompt_ids_batch)} requests"
            )
        rollout_seed_ordinal += len(generations)
        return generations

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
    episodes_seen = int(_resumed("episodes_seen", 0))
    mt_turn_records = int(
        _resumed("mt_turn_records", 0)
    )  # total per-turn records produced across all episodes (multi-turn only)
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
            return _generate_with_rollout_seeds([prefix_ids], max_tokens=max(1, int(max_new)))[0]

    def _account(r):
        """Consume one sample's SampleResult: tally teacher health / truncations, advance the step
        aggregates, and refresh the stall clock. Called after vLLM generation and teacher scoring;
        ``samples_seen`` is advanced by the caller once per generated rollout so its timing is
        unchanged."""
        nonlocal \
            teacher_ok, \
            teacher_transient, \
            teacher_error, \
            truncated_rollouts, \
            step_loss, \
            step_cov
        nonlocal granularity_sum, granularity_n, generated_tokens, teacher_input_tokens, nseq
        if r.teacher_status == "ok":
            teacher_ok += 1
        elif r.teacher_status == "transient":
            teacher_transient += 1
        elif r.teacher_status == "error":
            teacher_error += 1
        if r.truncated:
            truncated_rollouts += 1
        if r.loss is None:
            reason = _sample_skip_reason(r)
            skip_counts[reason] += 1
            step_skip_counts[reason] += 1
            # Refresh the stall clock even when a sample yields no teacher signal (rationale in
            # _opd_progress) — else an all-skip stretch emits only ignored liveness pings.
            _opd_progress(opt_steps, nseq)
            return
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
    train_wall_baseline = float(_resumed("train_wall_seconds", 0.0))
    opd_phase_seconds: Counter[str] = Counter(_resumed("opd_phase_seconds", {}))
    opd_phase_counts: Counter[str] = Counter(_resumed("opd_phase_counts", {}))
    # Drive the loop by optimizer UPDATES, not raw iterations. A no-signal iteration (empty
    # completions / a flaky teacher) skips optimizer.step() below, so `for step in range(steps)` could
    # exit with opt_steps < steps -- shipping an under-trained adapter as the served DEFAULT while the
    # run is billed the full submit-time `steps` quote (codex[bot]). Loop until `steps` real updates
    # land. A no-signal attempt is retried a few times with a fresh rollout/data slice before the
    # optimizer step is abandoned, so sporadic empty teacher responses or degenerate samples do not
    # waste a requested optimizer update. Bound total attempts so a persistently degraded teacher cannot
    # spin unboundedly -- the post-loop guard then turns a shortfall into a RETRY, not a silent
    # under-trained publish.
    # data cursor: the batch at each iteration is a pure function of `step` (examples is a seeded
    # deterministic shuffle rebuilt identically every worker start), so restoring this int restores the
    # exact data position -- the dataset itself is never persisted. max_iters stays absolute.
    step = int(_resumed("step", 0))
    final_adapter_staged = False
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
                batch = [
                    examples[(it * prompts_per_step + i) % len(examples)]
                    for i in range(prompts_per_step)
                ]
                accum_target = max(1, prompts_per_step * group)
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

                def _queue_or_account(p: _Pending) -> None:
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
                    resolved = _resolve_samples_batched(
                        model,
                        tok,
                        device,
                        scored_samples,
                        knobs,
                        loss_microbatch_size,
                        backward_scale=1.0 / _accum_target,
                    )
                    for r in resolved:
                        _account(r)

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
                            gens = _generate_with_rollout_seeds(
                                chunk_prompts, max_tokens=knobs.max_completion
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
            # each seq's grad was scaled by 1/accum_target; the mutation boundary rescales a partial
            # step, clips, publishes the first-update marker, mutates the optimizer, then increments.
            optimizer_started = time.perf_counter()
            opt_steps = _apply_opd_optimizer_update(
                model=model,
                optimizer=optimizer,
                torch=torch,
                opt_steps=opt_steps,
                nseq=nseq,
                accum_target=accum_target,
            )
            opd_phase_seconds["optimizer_step"] += time.perf_counter() - optimizer_started
            opd_phase_counts["optimizer_steps"] += 1
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
            # checkpoint on optimizer-step count, not the loop index: a `step-n` artifact must
            # contain n real updates. the final optimizer state is always a required recovery boundary,
            # even when the configured deployable schedule does not select that step.
            checkpoint_due = save_step_due(
                opt_steps, knobs.save_at_steps, knobs.save_every
            )
            is_final_step = opt_steps == steps
            if checkpoint_due or is_final_step:
                is_required_save = opt_steps in knobs.save_at_steps
                publish_periodic = checkpoint_due and not is_required_save and not is_final_step
                if is_required_save or publish_periodic:
                    _save_adapter(model, tok, adapter_dir)
                    if is_final_step:
                        final_adapter_staged = True
                checkpoint_accounting = {
                    "loss_curve": list(loss_curve),
                    "coverage_curve": list(coverage_curve),
                    "generated_tokens": int(generated_tokens),
                    "teacher_input_tokens": int(teacher_input_tokens),
                    "truncated_rollouts": int(truncated_rollouts),
                    "granularity_sum": float(granularity_sum),
                    "granularity_n": int(granularity_n),
                    "samples_seen": int(samples_seen),
                    "teacher_ok": int(teacher_ok),
                    "teacher_transient": int(teacher_transient),
                    "teacher_error": int(teacher_error),
                    "skip_counts": dict(skip_counts),
                    "no_signal_resamples": int(no_signal_resamples),
                    "no_signal_skipped_steps": int(no_signal_skipped_steps),
                    "episodes_seen": int(episodes_seen),
                    "mt_turn_records": int(mt_turn_records),
                    "opd_phase_seconds": dict(opd_phase_seconds),
                    "opd_phase_counts": dict(opd_phase_counts),
                    "train_wall_seconds": _cumulative_train_wall_seconds(
                        train_wall_baseline, t_train, time.time()
                    ),
                }

                def _publish_step_deployable(
                    _opt_steps=opt_steps, _required=is_required_save
                ) -> None:
                    _publish_opd_deployable(
                        adapter_dir,
                        _opt_steps,
                        as_default=False,
                        best_effort=not _required,
                        save_required=_required,
                    )

                # exact boundaries keep full state and their required companion in one transaction.
                # nonfinal periodic publications stay ordered before sync but are fully best-effort.
                # final periodic or otherwise unscheduled steps publish only full state here; finalization
                # stages and publishes the deployable once after recovery state is known durable.
                if is_required_save:
                    _save_opd_resume_checkpoint(
                        model=model,
                        tok=tok,
                        optimizer=optimizer,
                        torch=torch,
                        out_dir=out_dir,
                        opt_steps=opt_steps,
                        step=step,
                        accounting=checkpoint_accounting,
                        rollout_seed_ordinal=rollout_seed_ordinal,
                        prompt_pool_fingerprint=prompt_pool_fingerprint,
                        required=True,
                        after_upload=_publish_step_deployable,
                    )
                elif is_final_step:
                    _save_opd_resume_checkpoint(
                        model=model,
                        tok=tok,
                        optimizer=optimizer,
                        torch=torch,
                        out_dir=out_dir,
                        opt_steps=opt_steps,
                        step=step,
                        accounting=checkpoint_accounting,
                        rollout_seed_ordinal=rollout_seed_ordinal,
                        prompt_pool_fingerprint=prompt_pool_fingerprint,
                        required=True,
                    )
                else:
                    _publish_step_deployable()
                    _save_opd_resume_checkpoint(
                        model=model,
                        tok=tok,
                        optimizer=optimizer,
                        torch=torch,
                        out_dir=out_dir,
                        opt_steps=opt_steps,
                        step=step,
                        accounting=checkpoint_accounting,
                        rollout_seed_ordinal=rollout_seed_ordinal,
                        prompt_pool_fingerprint=prompt_pool_fingerprint,
                        required=False,
                    )
            # on-policy means the next rollout must sample from the just-updated student lora. a due
            # optimizer-boundary checkpoint must already be durable before this fallible sync. skip the
            # final sync because no rollout remains and the final adapter is saved from the hf model.
            if opt_steps < steps:
                with liveness_heartbeat(
                    "opd_vllm_sync", progress=lambda s=opt_steps: s, progress_step=True
                ):
                    sync_started = time.perf_counter()
                    vllm_rollout.sync_from_model(model)
                    opd_phase_seconds["vllm_sync"] += time.perf_counter() - sync_started
                    opd_phase_counts["vllm_syncs"] += 1

    train_wall = _cumulative_train_wall_seconds(train_wall_baseline, t_train, time.time())
    if not loss_curve:
        # use attempt-local failures for retry classification, not the cumulative reporting counter.
        if teacher_ok == 0 and (teacher_transient - teacher_transient_baseline) > 0:
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
        # use attempt-local failures for retry classification, not the cumulative reporting counter.
        if (teacher_transient - teacher_transient_baseline) > 0:
            # SOME scoring calls hit a RETRYABLE teacher outage: a healthier teacher next attempt may
            # complete the remaining updates, so RETRY rather than ship short (codex[bot]).
            raise RetriableInfraError(
                f"{diag} — retrying rather than publishing an under-trained adapter billed as full steps."
            )
        # no transient teacher failures: the shortfall is deterministic (over-budget prompts,
        # non-terminated or empty rollouts, or zero teacher-to-student alignment). some updates may have
        # landed, but retrying the same seed, model, data, and restored cursor reproduces the insufficient
        # successful-update rate. fail permanently with the skip diagnostics so the user fixes the setup.
        raise RuntimeError(
            f"{diag}; the shortfall is deterministic (no transient teacher failures), so resuming the "
            "same seed, model, data, and rollout position would repeat the insufficient successful-update "
            "rate. Fix the setup: shorten prompts, enable stop_sequences or a warm-start so rollouts "
            "terminate, or verify the teacher tokenizer aligns to the student."
        )

    if not final_adapter_staged:
        _save_adapter(model, tok, adapter_dir)
    # keep the served default while suppressing unrequested final checkpoints.
    _publish_opd_deployable(
        adapter_dir,
        opt_steps,
        as_default=True,
        publish_checkpoint=final_save_due(opt_steps, knobs.save_at_steps),
    )
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
            # rollouts that hit the generation length cap without eos/stop are not teacher-scored or
            # distilled.
            "truncated_rollouts": truncated_rollouts,
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
            "prompts_per_step": prompts_per_step,
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
            "opd_phase_optimizer_step_seconds": float(opd_phase_seconds["optimizer_step"]),
            "opd_phase_vllm_sync_seconds": float(opd_phase_seconds["vllm_sync"]),
            "opd_phase_rollout_generate_calls": int(opd_phase_counts["rollout_generate_calls"]),
            "opd_phase_teacher_batches": int(opd_phase_counts["teacher_batches"]),
            "opd_phase_optimizer_steps": int(opd_phase_counts["optimizer_steps"]),
            "opd_phase_vllm_syncs": int(opd_phase_counts["vllm_syncs"]),
            "opd_rollout_pipeline_chunks": (
                _opd_rollout_pipeline_chunks(prompts_per_step * group) if not multi_turn else None
            ),
            "opd_rollout_chunk_size": (
                _opd_rollout_chunk_size(prompts_per_step * group) if not multi_turn else None
            ),
            "opd_rollout_pipeline_target_chunk_size": (
                _opd_rollout_pipeline_target_chunk_size(prompts_per_step * group)
                if not multi_turn
                else None
            ),
            "opd_rollout_pipeline_max_chunks": (
                _opd_rollout_pipeline_max_chunks(prompts_per_step * group)
                if not multi_turn
                else None
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


@dataclass(frozen=True)
class SampleResult:
    """One student sample's outcome, returned by the rollout/loss pipeline for ``run_opd`` to aggregate.

    ``loss`` is the groupwise reverse-KL loss for a distilled sample. Otherwise it is ``None``. The
    stats describe what happened so the caller can count teacher health / truncations and, on a
    no-loss run, decide whether it is a retriable infra failure."""

    loss: object = None  # torch scalar tensor when distilled, else None (module is torch-free)
    # "ok" once teacher.score returns, "transient" on a retryable teacher outage, else None (teacher
    # not reached). run_opd uses this to decide whether a no-signal run is a retriable infra failure.
    teacher_status: str | None = None
    # a rollout that did not terminate naturally is never teacher-distilled.
    truncated: bool = False
    coverage: float = 0.0
    gen_tokens: int = 0
    teacher_tokens: int = 0
    # Mean student-tokens-per-alignment-group; a real health signal where coverage is not.
    group_granularity: float = 0.0
    # Machine-readable reason when loss is None. Used for skipped-step diagnostics and train_meta.
    skip_reason: str = ""


@dataclass(frozen=True)
class _GenResult:
    """One student rollout after OPD's pre-scoring gates: the sampled completion plus skip/truncated
    verdicts. ``truncated`` samples bypass teacher scoring; ``skip`` covers empty or U+FFFD completions.
    Otherwise ``completion_ids`` / ``completion_text`` carry the
    trimmed on-policy answer to score + distil. Torch-free (completion_ids is a CPU list) so it can be
    handed to the model-free scoring thread pool."""

    completion_ids: object = None
    completion_text: str = ""
    gen_tokens: int = 0
    truncated: bool = False
    skip: bool = False
    skip_reason: str = ""
    finish_reason: str | None = None
    stop_reason: object = None
    terminal_eos_id: int | None = None
    # Grammar-forced mask (parallel to completion_ids): True where guided decoding left one legal
    # token. Threaded from OpdVllmOutput.forced (sliced in lockstep with stop-trimming); () = none.
    forced: tuple = ()


@dataclass(frozen=True)
class _ScoreResult:
    """Teacher-scoring outcome from ``_score_one`` (RUN IN THE THREAD POOL). ``status`` is "ok"
    (``teacher_toks`` populated), "transient" (retryable teacher outage -> sample skipped + counted),
    or "error" (any other exception -> sample skipped, teacher uncounted). A PERMANENT ``TeacherError``
    is NOT represented here — ``_score_one`` re-raises it so the run aborts, exactly as before."""

    teacher_toks: object = None
    status: str = "ok"
    error: str = ""


def _sample_skip_reason(r: SampleResult) -> str:
    """Classify a no-loss OPD sample for skipped-step diagnostics."""

    if r.skip_reason:
        return r.skip_reason
    if r.truncated:
        return "truncated_rollout"
    if r.teacher_status == "transient":
        return "teacher_transient"
    if r.teacher_status == "error":
        return "teacher_error"
    if r.teacher_status == "ok":
        return "alignment_empty"
    return "pre_teacher_skip"


@dataclass
class _Pending:
    """A rollout awaiting concurrent teacher scoring and then loss/backward, carrying the prompt context
    both need. Mutable: ``score`` is filled in by the thread pool for scorable rollouts."""

    gen: _GenResult
    prompt_ids: object
    prompt_messages: object
    score: object = None


def _termination_cause(
    out: OpdVllmOutput, completion_ids, stop_text: str, eos_ids, stop_sequences
) -> tuple[str, int | None]:
    """Classify why generation stopped and return the terminal eos id when present."""
    reason = str(out.finish_reason or "").lower()
    stop_reason = out.stop_reason
    terminal_eos_id = next(
        (int(token_id) for token_id in reversed(completion_ids) if int(token_id) in eos_ids), None
    )
    if terminal_eos_id is None and isinstance(stop_reason, int) and stop_reason in eos_ids:
        terminal_eos_id = int(stop_reason)

    # precedence is deliberate: a configured stop sequence wins over eos, which wins over length.
    # vllm can report a generic "stop" while also returning a token id or echoed stop delimiter.
    stop_sequence = bool(
        stop_sequences
        and (
            any(s and stop_text.endswith(s) for s in stop_sequences)
            or (isinstance(stop_reason, str) and stop_reason in stop_sequences)
        )
    )
    if stop_sequence:
        return "stop_sequence", terminal_eos_id
    if terminal_eos_id is not None or reason == "eos":
        return "eos", terminal_eos_id
    if reason == "length":
        return "length", None
    if reason == "stop" or out.terminated:
        return "eos", terminal_eos_id
    if _rollout_terminated(completion_ids, stop_text, eos_ids, stop_sequences):
        return "eos", terminal_eos_id
    return "unknown", None


def _gen_from_vllm_output(out: OpdVllmOutput, tok, knobs, eos_ids) -> _GenResult:
    """Apply OPD's pre-scoring gates to one vLLM completion."""
    completion_ids = [int(t) for t in out.token_ids]
    forced = tuple(bool(f) for f in (getattr(out, "forced", ()) or ()))
    decode = getattr(tok, "decode", None)
    completion_text = out.text or (
        decode(completion_ids, skip_special_tokens=True) if decode else ""
    )
    stop_text = decode(completion_ids, skip_special_tokens=False) if decode else completion_text
    cause, terminal_eos_id = _termination_cause(
        out, completion_ids, stop_text, eos_ids, knobs.stop_sequences
    )
    metadata = {
        "finish_reason": out.finish_reason,
        "stop_reason": out.stop_reason,
        "terminal_eos_id": terminal_eos_id,
    }
    if cause in {"length", "unknown"}:
        return _GenResult(
            completion_ids=completion_ids,
            completion_text=completion_text,
            truncated=True,
            gen_tokens=len(completion_ids),
            skip_reason="truncated_rollout",
            forced=forced,
            **metadata,
        )
    # vLLM may strip stop strings unless include_stop_str_in_output is supported. Trim when the
    # delimiter is present; otherwise keep the already-stripped ids/text.
    if cause == "stop_sequence":
        completion_ids, completion_text = _trim_trailing_stop(
            tok, completion_ids, stop_text, knobs.stop_sequences
        )
    gen_tokens = len(completion_ids)
    if not completion_text.strip():
        return _GenResult(
            skip=True, gen_tokens=gen_tokens, skip_reason="empty_completion", **metadata
        )
    if "\ufffd" in completion_text:
        return _GenResult(
            skip=True, gen_tokens=gen_tokens, skip_reason="replacement_char", **metadata
        )
    return _GenResult(
        completion_ids=completion_ids,
        completion_text=completion_text,
        gen_tokens=gen_tokens,
        # Trimming drops a trailing stop suffix, so the kept ids are a prefix -> slice forced to match.
        forced=forced[: len(completion_ids)],
        **metadata,
    )


def _generate_many_vllm(
    rollout: OpdVllmRolloutEngine,
    tok,
    prompt_ids_batch: list[list[int]],
    knobs,
    eos_ids,
    *,
    max_tokens: int,
    request_seeds: list[int] | None = None,
) -> list[_GenResult]:
    return [
        _gen_from_vllm_output(out, tok, knobs, eos_ids)
        for out in rollout.generate(
            prompt_ids_batch, max_tokens=max_tokens, request_seeds=request_seeds
        )
    ]


def _score_one(
    teacher, gen_result, *, prompt_messages, thinking_prefill, max_attempts: int = 2
) -> _ScoreResult:
    """[THREAD POOL — network only] Build the teacher prompt and echo-score the completion over the
    API. MUST NOT touch the torch model or any shared mutable state — it reads only the completion
    string + prompt messages and calls the stateless teacher HTTP client, so it is safe to run
    concurrently for every scorable sample in a step. ``thinking_prefill`` is appended to the teacher
    prompt so it conditions on the same trailing context the student sampled after in thinking mode.
    Error semantics are the shared OPD teacher semantics: a PERMANENT ``TeacherError`` propagates (the
    run aborts), a transient one -> status "transient", any other exception -> status "error"; both
    leave the sample skipped with no teacher signal."""
    teacher_prompt = _teacher_prompt_text(prompt_messages, thinking_prefill)
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        try:
            teacher_toks = teacher.score(teacher_prompt, gen_result.completion_text)
        except TeacherError as e:
            if e.permanent:  # bad key / model id / malformed -> abort now, don't burn the whole run
                raise
            if attempt < attempts:
                print(
                    f"[opd] teacher score failed (transient, retrying sample {attempt}/{attempts}): {e}"
                )
                continue
            print(f"[opd] teacher score failed (transient, skipping sample): {e}")
            return _ScoreResult(status="transient", error=str(e))
        except Exception as e:
            if attempt < attempts:
                print(f"[opd] teacher score failed (retrying sample {attempt}/{attempts}): {e}")
                continue
            print(f"[opd] teacher score failed (skipping sample): {e}")
            return _ScoreResult(status="error", error=str(e))
        return _ScoreResult(teacher_toks=teacher_toks, status="ok")
    return _ScoreResult(status="error", error="teacher scoring attempts exhausted")


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
    return [
        _ScoreResult(status="error", error="teacher batch attempts exhausted") for _ in pendings
    ]


@dataclass(frozen=True)
class _PreparedGkdGroups:
    token_indices: tuple[int, ...]
    group_lengths: tuple[int, ...]
    teacher_logsums: tuple[float, ...]


def _drop_fully_forced_groups(groups, forced):
    """Remove alignment groups whose student tokens were ALL grammar-forced: the student had no
    choice there, so the teacher's (unconstrained) logprob over that span is spurious signal.
    Dropping the whole ``(student_idx, teacher_logsum)`` tuple keeps both sides of the reverse-KL
    balanced. ``forced`` is parallel to the student tokens (== completion_ids); empty -> no-op."""
    if not forced:
        return groups
    return [
        (s_idx, tsum)
        for (s_idx, tsum) in groups
        if not (s_idx and all(i < len(forced) and forced[i] for i in s_idx))
    ]


def _prepare_gkd_groups(groups) -> _PreparedGkdGroups | None:
    token_indices: list[int] = []
    group_lengths: list[int] = []
    teacher_logsums: list[float] = []
    for s_idx, teacher_logsum in groups:
        if not s_idx:
            continue
        token_indices.extend(int(i) for i in s_idx)
        group_lengths.append(len(s_idx))
        teacher_logsums.append(float(teacher_logsum))
    if not token_indices:
        return None
    return _PreparedGkdGroups(
        token_indices=tuple(token_indices),
        group_lengths=tuple(group_lengths),
        teacher_logsums=tuple(teacher_logsums),
    )


@dataclass(frozen=True)
class _PreparedLoss:
    idx: int
    prompt_ids: object
    student_ids: object
    groups: _PreparedGkdGroups | None
    coverage: float
    gen_tokens: int
    teacher_tokens: int
    group_granularity: float


def _gkd_loss_from_logits_rows(rows, student_ids, groups, kl_coef=1.0):
    import torch
    import torch.nn.functional as F

    if not student_ids or not groups:
        return None
    prepared = groups if isinstance(groups, _PreparedGkdGroups) else _prepare_gkd_groups(groups)
    if prepared is None:
        return None
    rows = rows.float()
    ids_t = torch.tensor(student_ids, device=rows.device)
    sp_t = -F.cross_entropy(rows, ids_t, reduction="none")
    return _gkd_loss_from_logps(sp_t, prepared, kl_coef=kl_coef)


def _gkd_loss_from_logps(sp_t, groups, kl_coef=1.0):
    import torch

    if sp_t is None or not groups:
        return None
    prepared = groups if isinstance(groups, _PreparedGkdGroups) else _prepare_gkd_groups(groups)
    if prepared is None:
        return None
    sp_det = sp_t.detach()
    flat_idx_t = torch.tensor(prepared.token_indices, device=sp_t.device)
    group_lengths_t = torch.tensor(prepared.group_lengths, device=sp_t.device)
    group_ids_t = torch.repeat_interleave(
        torch.arange(len(prepared.group_lengths), device=sp_t.device), group_lengths_t
    )
    student_group_logsum = sp_det.new_zeros(len(prepared.group_lengths))
    student_group_logsum.index_add_(0, group_ids_t, sp_det.index_select(0, flat_idx_t))
    teacher_logsum_t = torch.tensor(prepared.teacher_logsums, dtype=sp_t.dtype, device=sp_t.device)
    coeffs = (
        kl_coef * (student_group_logsum - teacher_logsum_t) / group_lengths_t.to(dtype=sp_t.dtype)
    )
    coeff_vec = coeffs.index_select(0, group_ids_t)
    sp_sel = sp_t.index_select(0, flat_idx_t)
    return (coeff_vec * sp_sel).mean()


def _forward_logits(
    model, input_ids, attention_mask=None, *, position_ids=None, logits_to_keep=None
):
    kwargs = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        kwargs["position_ids"] = position_ids
    if logits_to_keep is not None:
        kwargs["logits_to_keep"] = logits_to_keep
    try:
        return model(input_ids, **kwargs).logits
    except TypeError:
        if position_ids is not None or logits_to_keep is not None:
            raise
        if attention_mask is not None:
            return model(input_ids).logits
        raise


def _bump_model_counter(model, name: str, inc: int = 1) -> None:
    with contextlib.suppress(Exception):
        setattr(model, name, int(getattr(model, name, 0) or 0) + int(inc))


def _resolve_samples_batched(
    model,
    tok,
    device,
    samples: list[tuple[_GenResult, _ScoreResult | None, object]],
    knobs,
    microbatch: int,
    *,
    backward_scale: float | None = None,
) -> list[SampleResult]:
    import torch

    if not samples:
        return []
    results: list[SampleResult | None] = [None] * len(samples)
    prepared: list[_PreparedLoss] = []
    for idx, (gen, score, prompt_ids) in enumerate(samples):
        if gen.truncated:
            results[idx] = _resolve_no_loss_sample(gen, score)
            continue
        if gen.skip or score is None or score.status != "ok":
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
        # groups -> alignment_empty (below), and such samples are excluded from the step's 1/nseq mean.
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
            attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
            for row, seq in enumerate(seqs):
                input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
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
                    continue
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
                )
            if losses:
                (sum(losses) * float(backward_scale)).backward()

    return [r if r is not None else SampleResult(skip_reason="teacher_error") for r in results]


def _resolve_no_loss_sample(gen, score) -> SampleResult:
    """Map a generated rollout with no usable teacher score to its skipped ``SampleResult``.

    Scored samples must go through ``_resolve_samples_batched`` so OPD has exactly one loss path.
    """
    if gen.truncated:
        return SampleResult(
            truncated=True,
            gen_tokens=gen.gen_tokens,
            skip_reason=gen.skip_reason or "truncated_rollout",
        )
    if gen.skip:  # empty completion or U+FFFD — skipped before scoring
        return SampleResult(
            gen_tokens=gen.gen_tokens,
            skip_reason=gen.skip_reason or "pre_teacher_skip",
        )
    if score is None:
        return SampleResult(
            teacher_status="error",
            gen_tokens=gen.gen_tokens,
            skip_reason="teacher_missing_score",
        )
    if score.status == "transient":  # retryable teacher outage — skipped + counted
        return SampleResult(
            teacher_status="transient",
            gen_tokens=gen.gen_tokens,
            skip_reason="teacher_transient",
        )
    if score.status != "ok":  # any other teacher exception — skipped, teacher uncounted
        return SampleResult(
            teacher_status="error",
            gen_tokens=gen.gen_tokens,
            skip_reason="teacher_error",
        )
    raise RuntimeError("opd scored samples must be resolved through _resolve_samples_batched")


def _is_lora_adapter_key(key: object) -> bool:
    if not isinstance(key, str):
        return False
    adapter_parts = {
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
    }
    return any(part in adapter_parts for part in key.split("."))


def _save_opd_resume_checkpoint(
    *,
    model,
    tok,
    optimizer,
    torch,
    out_dir: str,
    opt_steps: int,
    step: int,
    accounting: dict,
    rollout_seed_ordinal: int,
    prompt_pool_fingerprint: str,
    required: bool = False,
    after_upload=None,
) -> None:
    """Persist and upload one latest-only OPD full-state resume checkpoint."""
    import shutil

    local_dir = os.path.join(out_dir, "resume_ckpt")
    try:
        shutil.rmtree(local_dir, ignore_errors=True)
        os.makedirs(local_dir, exist_ok=True)
        model.save_pretrained(local_dir)
        tok.save_pretrained(local_dir)
        torch.save(optimizer.state_dict(), os.path.join(local_dir, "optimizer.pt"))
        torch.save(capture_training_rng_state(torch), os.path.join(local_dir, "rng_state.pth"))
        payload = dict(accounting)
        payload.update(
            {
                "contract_version": OPD_RESUME_STATE_VERSION,
                "seed": int(_w.SEED),
                "opt_steps": int(opt_steps),
                "step": int(step),
                "rollout_seed_ordinal": int(rollout_seed_ordinal),
                "prompt_pool_fingerprint": str(prompt_pool_fingerprint),
            }
        )
        state_path = os.path.join(local_dir, "opd_state.json")
        with open(state_path, "w") as f:
            json.dump(payload, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:
        if required:
            raise RetriableInfraError(
                f"required opd full-state checkpoint construction failed at step {opt_steps}"
            ) from e
        print(f"[opd] resume checkpoint save failed at opt_steps={opt_steps}; training continues: {e}")
        return

    try:
        if after_upload is None:
            uploaded = _w.upload_resume_checkpoint(opt_steps, local_dir)
        else:
            uploaded = _w.upload_resume_checkpoint(opt_steps, local_dir, after_upload=after_upload)
    except (RequiredSaveError, RetriableInfraError) as e:
        if required:
            raise
        print(
            f"[opd] resume checkpoint upload failed at opt_steps={opt_steps}; training continues: {e}"
        )
        return
    except Exception as e:
        if required:
            raise RetriableInfraError(
                f"required opd full-state checkpoint upload failed at step {opt_steps}"
            ) from e
        print(
            f"[opd] resume checkpoint upload failed at opt_steps={opt_steps}; training continues: {e}"
        )
        return
    if uploaded:
        return
    if required:
        raise RetriableInfraError(
            f"required opd full-state checkpoint upload failed at step {opt_steps}"
        )
    print(
        f"[opd] resume checkpoint upload failed at opt_steps={opt_steps}; training continues"
    )


def _restore_opd_full_state(
    ckpt_dir: str,
    *,
    model,
    optimizer,
    torch,
    prompt_pool_fingerprint: str,
    max_opt_steps: int | None = None,
) -> dict | None:
    """Restore adapter + adam moments + rng + accounting from a resume checkpoint dir, in place.

    Returns the parsed accounting state (including ``opt_steps`` and ``step``) on success, or None when
    the dir is not a usable resume checkpoint for THIS run: incomplete files, wrong schema version, a
    different seed, or a byte-level load failure. The caller treats None-on-a-downloaded-checkpoint as
    fail-closed (it refuses to train fresh after a possible prior optimizer mutation). LoRA weights are
    copied IN PLACE into the already-constructed peft model so the optimizer's parameter references
    (created over ``model.parameters()``) stay valid and the loaded adam moments keep applying to them.
    """
    try:
        basenames = os.listdir(ckpt_dir)
    except OSError:
        return None
    if not opd_resume_checkpoint_complete(basenames):
        return None
    try:
        with open(os.path.join(ckpt_dir, "opd_state.json")) as f:
            state = json.load(f)
    except (OSError, ValueError):
        return None
    _base = os.path.basename(ckpt_dir.rstrip("/"))
    if not _base.startswith("checkpoint-"):
        return None
    _suffix = _base[len("checkpoint-") :]
    if not _suffix.isdigit():
        return None
    _dir_step = int(_suffix)
    if _dir_step <= 0 or _suffix != str(_dir_step):
        return None
    # validate every dependency-free json invariant before touching adapter or optimizer state.
    try:
        state = validate_opd_resume_state_metadata(
            state,
            expected_seed=int(_w.SEED),
            checkpoint_step=_dir_step,
        )
    except ValueError:
        return None
    if state["prompt_pool_fingerprint"] != prompt_pool_fingerprint:
        return None
    opt_steps = state["opt_steps"]
    if max_opt_steps is not None and opt_steps > max_opt_steps:
        return None
    try:
        from peft import load_peft_weights, set_peft_model_state_dict

        param_device = next(p for p in model.parameters() if p.requires_grad).device
        _peft_load = set_peft_model_state_dict(
            model, load_peft_weights(ckpt_dir, device=str(param_device))
        )
        _missing = getattr(_peft_load, "missing_keys", None) or []
        _unexpected = getattr(_peft_load, "unexpected_keys", None) or []
        _missing_adapter = [key for key in _missing if _is_lora_adapter_key(key)]
        if _missing_adapter or _unexpected:
            print(
                "[opd] resume adapter incompatible: "
                f"missing={_missing_adapter} unexpected={_unexpected}"
            )
            return None
        # weights_only=false: these are this run's own trusted artifacts (adam moments, and an rng blob
        # carrying python/numpy state), not plain tensor files, so torch 2.6+'s weights_only default
        # would reject them.
        optimizer.load_state_dict(
            torch.load(
                os.path.join(ckpt_dir, "optimizer.pt"),
                map_location=param_device,
                weights_only=False,
            )
        )
        rng = torch.load(
            os.path.join(ckpt_dir, "rng_state.pth"), map_location="cpu", weights_only=False
        )
    except MemoryError:
        raise
    except Exception as e:
        cuda_oom_type = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if isinstance(cuda_oom_type, type) and isinstance(e, cuda_oom_type):
            raise
        print(f"[opd] resume checkpoint at {ckpt_dir} failed to load: {e}")
        return None
    # rng is part of the semantic continuation contract. a downloaded replacement checkpoint whose
    # generator state cannot be restored is unusable and must never degrade to fresh training.
    if not restore_training_rng_state(torch, rng):
        print(f"[opd] resume checkpoint at {ckpt_dir} has incompatible rng state")
        return None
    return state


def _save_adapter(model, tok, adapter_dir: str) -> None:
    """Persist the LoRA adapter + tokenizer for deploy (identical layout to SFT)."""
    spec = getattr(_w, "JOB_SPEC", None)
    if spec is None:
        raise RuntimeError("OPD adapter save requires a JobSpec")
    _w.stamp_adapter_provenance(model, spec.model, spec.model_revision)
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    _w.write_base_model_provenance(adapter_dir, spec.model, spec.model_revision)


def _publish_opd_deployable(
    adapter_dir: str,
    step: int,
    *,
    as_default: bool,
    best_effort: bool = False,
    publish_checkpoint: bool = True,
    save_required: bool = False,
) -> None:
    """Publish the served default and, when requested, the step checkpoint."""
    try:
        if as_default:
            _w.hf_upload_folder(adapter_dir, "adapter", required=True)
        if publish_checkpoint:
            if save_required:
                _w.publish_deployable_checkpoint(adapter_dir, step, required=True)
            else:
                _w.publish_deployable_checkpoint(adapter_dir, step)
    except Exception as e:
        if not best_effort:
            raise
        print(f"[opd] deployable publish failed at step {step}; skipping, training continues: {e}")
