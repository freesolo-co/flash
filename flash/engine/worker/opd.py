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
arbitrary tokenizer mismatch and covers every student token. See ``tokenizer_align`` and
docs/on-policy-distillation.md.

There is NO local reference model and NO colocated vLLM engine: sampling is HF ``generate`` on the
resident student (so the VRAM profile matches SFT), and the teacher lives behind the API. All heavy
imports (torch/transformers/peft) are inside functions, so importing this module is CPU/offline-safe.
"""

from __future__ import annotations

import os
import random
import time

from flash.engine.recipe import RECIPE
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.perf import (
    free_gpu,
    gpu_diagnostics,
    grad_checkpointing_on,
    optimal_attn_impl,
    setup_perf_backends,
    wait_for_gpu,
)
from flash.engine.worker.teacher import TeacherError
from flash.engine.worker.tokenizer_align import (
    StudentToken,
    groupwise_alignment,
    groupwise_coverage,
)


def _resolve_opd_knobs():
    """Resolve every opd knob from the JobSpec's [train] table, falling back to RECIPE.opd."""
    d = RECIPE.opd
    t = _w.JOB_SPEC.train if _w.JOB_SPEC else None

    def opt(name, default):
        v = getattr(t, name, None) if t else None
        return v if v is not None else default

    max_completion = int(
        opt("max_tokens", 0)
        or (d.max_completion_len_thinking if _w.THINKING else d.max_completion_len)
    )
    # Honor an explicit kl_penalty_coef=0.0 (a plain `or` would treat 0.0 as unset).
    _kl = opt("kl_penalty_coef", None)
    kl_coef = float(_kl if _kl is not None else d.kl_coef)
    return {
        "teacher_model": opt("teacher_model", "") or d.teacher_model,
        "teacher_base_url": d.teacher_base_url,
        "steps": int(opt("steps", 0) or d.num_steps),
        "learning_rate": float(opt("learning_rate", 0) or d.learning_rate),
        "temperature": float(
            opt("temperature", None)
            if (t and t.temperature is not None)
            else d.sampling_temperature
        ),
        "top_p": d.sampling_top_p,
        "max_completion": max_completion,
        "prompts_per_step": int(opt("batch_size", 0) or d.prompts_per_step),
        "group_size": int(opt("group_size", 0) or d.group_size),
        # gkd reverse-KL scale; reuses the existing [train] kl_penalty_coef knob (default 1.0).
        "kl_coef": kl_coef,
        "save_every": int(opt("save_every", 0) or 20),
        "max_length": int(opt("max_length", 0) or 0),
        # Student on-policy sampling stops at these delimiters (parity with GRPO), so the teacher
        # never scores/trains on text past the intended answer boundary.
        "stop_sequences": tuple(getattr(t, "stop_sequences", ()) or ()),
    }


def _teacher_prompt_text(prompt_messages: list[dict]) -> str:
    """Render the prompt for the teacher's scoring context (template-agnostic; ends at 'Assistant: ')."""
    parts = []
    for m in prompt_messages:
        role = str(m.get("role", "user")).capitalize()
        parts.append(f"{role}: {m.get('content', '')!s}")
    parts.append("Assistant: ")
    return "\n".join(parts)


def student_tokens_with_offsets(tok, completion_ids, completion_text: str):
    """Char spans for the ORIGINAL sampled token ids, indexed into ``completion_text`` (the exact
    string the teacher echo-scored). Using the sampled ids — not a re-tokenization of the decoded
    text — keeps the loss on the true on-policy tokens. Offsets are built by incrementally decoding
    the id prefix and measuring its length (clamped monotonic in ``[0, len]``, so a byte-level token
    that splits a multi-byte char never yields a negative/backward span); a special token (e.g. eos)
    decodes to nothing, so it gets a zero-width span and is naturally excluded from the alignment."""
    ids = [int(t) for t in completion_ids]
    toks: list[StudentToken] = []
    prev = 0
    n = len(completion_text)
    for i in range(len(ids)):
        end = min(n, max(prev, len(tok.decode(ids[: i + 1], skip_special_tokens=True))))
        toks.append(StudentToken(token_id=ids[i], start=prev, end=end))
        prev = end
    return ids, toks


def gkd_loss(model, prompt_ids, student_ids, groups, device, kl_coef=1.0):
    """Groupwise reverse-KL on-policy distillation (the collinear-ai *spider* / Tinker method).

    Per aligned text-span group, the per-token loss coefficient is
    ``(log P_student(span).detach() - log P_teacher(span)) / |span|`` — the REINFORCE surrogate whose
    gradient IS the reverse-KL gradient ``E_student[∇ log π · (log π - log p*)]`` (Thinking Machines,
    *On-Policy Distillation*). It needs only the REALIZED-token logprobs (no top-k, no vocabulary
    projection), so it is exact across arbitrary tokenizer mismatches and covers every token.
    ``kl_coef`` scales the objective (``[train] kl_penalty_coef``). Scalar loss or None.
    """
    import torch

    if not student_ids or not groups:
        return None
    input_ids = torch.tensor([prompt_ids + student_ids], device=device)
    logits = model(input_ids).logits[0]  # [T, V], model dtype
    P = len(prompt_ids)
    # Differentiable student logprob of each REALIZED completion token (logits at P+j-1 predict token
    # j), computed WITHOUT materializing a full [T, V] fp32 log_softmax: we select only the C
    # completion rows, cast those to fp32, and use logit_realized - logsumexp. This keeps the
    # vocab-projection memory at [C, V] instead of an extra [T, V] fp32 buffer (avoids OOM on tight
    # placements with large-vocab students).
    pos = torch.arange(P - 1, P - 1 + len(student_ids), device=logits.device)
    ids_t = torch.tensor(student_ids, device=logits.device)
    rows = logits.index_select(0, pos).float()  # [C, V]
    sp_t = rows.gather(1, ids_t.unsqueeze(1)).squeeze(1) - torch.logsumexp(rows, dim=-1)  # [C]
    sp = [sp_t[j] for j in range(len(student_ids))]
    terms = []
    for s_idx, teacher_logsum in groups:
        if not s_idx:  # defensive: a teacher-only span carries no student token to supervise
            continue
        student_logsum_det = float(sum(sp[j].detach() for j in s_idx))
        # coeff > 0 where the student is MORE confident than the teacher on the span (push down);
        # coeff < 0 where the teacher is more confident (push up). Gradient = reverse-KL gradient.
        coeff = kl_coef * (student_logsum_det - teacher_logsum) / len(s_idx)
        terms.extend(coeff * sp[j] for j in s_idx)
    if not terms:
        return None
    return torch.stack(terms).mean()


def _student_model(model_id, mik, device):
    """Build the trainable student LoRA. Warm-starts from ``train.init_from_adapter`` when set —
    continuing a prior run's adapter (e.g. an SFT checkpoint), the same path GRPO uses via
    ``_init_adapter_model`` — otherwise a fresh LoRA on the base. This makes an SFT->opd pipeline a
    genuine continuation (the opd stage keeps the SFT behavior) rather than silently restarting from
    base. VL merge-warm-start (which needs SFT (+) opd recombination before deploy) is not yet wired
    for opd, so it is refused loudly instead of shipping a broken deploy."""
    init_model, init_peft = _w._init_adapter_model(model_id)
    if init_peft is None:
        # init_model is already a trainable PeftModel continuing the prior (e.g. SFT) adapter.
        return init_model.to(device)
    if getattr(_w, "_VL_WARMSTART_SFT_DIR", None) is not None:
        raise RuntimeError(
            "opd warm-start from a VL SFT adapter is not yet supported (the saved adapter would need "
            "SFT+opd recombination before deploy). Use a text-model SFT adapter, or omit "
            "train.init_from_adapter to train a fresh LoRA on the base."
        )
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(init_model, trust_remote_code=True, **mik).to(
        device
    )
    return get_peft_model(base, init_peft)


def run_opd():
    import torch
    from transformers import AutoTokenizer

    from flash.engine.worker.teacher import TeacherClient

    env = _w.require_active_env()
    t_start = time.time()
    _w.heartbeat("opd_start", gpu=gpu_diagnostics())
    knobs = _resolve_opd_knobs()
    warm_start = _w.JOB_SPEC.train.init_from_adapter if _w.JOB_SPEC else ""
    print(
        f"[opd] gkd (groupwise reverse-KL) teacher={knobs['teacher_model']} "
        f"steps={knobs['steps']} warm_start={warm_start or 'none'}"
    )

    api_key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "opd requires the FIREWORKS_API_KEY runtime secret (the GLM teacher); it was not "
            "delivered to the worker. Declare it under [environment] secrets and export it locally."
        )
    teacher = TeacherClient(api_key, knobs["teacher_base_url"], knobs["teacher_model"])

    wait_for_gpu(_w.JOB_SPEC.gpu.type if _w.JOB_SPEC else None)
    setup_perf_backends()
    model_id = _w.JOB_SPEC.model if _w.JOB_SPEC else RECIPE.hf_model_id
    download_seconds = _w.prefetch_model(model_id)
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _attn = optimal_attn_impl()
    mik = {"dtype": torch.bfloat16}
    if _attn:
        mik["attn_implementation"] = _attn
    setup_seconds = time.time() - t_start
    _w.heartbeat("opd_model_load", setup_seconds=setup_seconds, gpu=gpu_diagnostics())
    with liveness_heartbeat("opd_initializing"):
        model = _student_model(model_id, mik, device)
        # Engine length gates whether gradient checkpointing is needed for the loss forward.
        seq_cap = knobs["max_length"] or (RECIPE.opd.max_prompt_len + knobs["max_completion"])
        if grad_checkpointing_on(model_id, seq_cap):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            print("[opd] gradient checkpointing enabled")
    model.config.use_cache = (
        True  # generation needs the KV cache; re-disabled per loss forward below
    )

    # Build the on-policy prompt pool from the environment (same rendering as GRPO).
    train = env.dataset()
    if not train:
        raise RuntimeError(
            "opd: the environment dataset is empty — no prompts to sample on-policy. Check the "
            "environment's dataset()/train split before provisioning a GPU."
        )
    rng = random.Random(_w.SEED)
    rng.shuffle(train)
    examples = train
    # Seed torch/CUDA too (not just the Python shuffle RNG): the student samples via
    # model.generate(do_sample=True), so the fixed Flash seed must reproduce the same completions
    # (and therefore the same trained adapter) run-to-run.
    torch.manual_seed(_w.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_w.SEED)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=knobs["learning_rate"]
    )
    steps = knobs["steps"]
    ppl_step = knobs["prompts_per_step"]
    group = knobs["group_size"]
    # Prompt budget mirrors GRPO: DROP (not truncate) prompts over the context budget, so the student
    # never conditions on a truncated prompt the teacher didn't see. Use the configured max_length
    # when set, else the recipe prompt cap.
    prompt_budget = max(
        1,
        (knobs["max_length"] - knobs["max_completion"])
        if knobs["max_length"]
        else RECIPE.opd.max_prompt_len,
    )
    dropped_long = 0
    resume_ckpt = _w.hf_resume_checkpoint()
    if resume_ckpt:
        print("[opd] resume-from-checkpoint is not yet supported for opd; starting fresh")

    out_dir = f"/tmp/opd_seed{_w.SEED}"
    adapter_dir = f"{out_dir}/adapter"
    os.makedirs(out_dir, exist_ok=True)

    gen_cfg = {
        "do_sample": knobs["temperature"] > 0,
        "temperature": max(knobs["temperature"], 1e-5),
        "top_p": knobs["top_p"],
        "max_new_tokens": knobs["max_completion"],
        "pad_token_id": tok.pad_token_id,
    }
    if knobs["stop_sequences"]:
        # HF stops generation at any of these strings (needs the tokenizer to match them on decode).
        gen_cfg["stop_strings"] = list(knobs["stop_sequences"])
        gen_cfg["tokenizer"] = tok
    loss_curve: list[float] = []
    coverage_curve: list[float] = []
    generated_tokens = 0
    teacher_input_tokens = 0
    opt_steps = 0  # optimizer steps actually applied (< steps if any iteration had no teacher signal)
    _reset_peak = getattr(_w, "_reset_peak_gpu", None)
    if _reset_peak:
        _reset_peak()

    t_train = time.time()
    with liveness_heartbeat("opd_step"):
        for step in range(steps):
            batch = [examples[(step * ppl_step + i) % len(examples)] for i in range(ppl_step)]
            accum_target = max(1, ppl_step * group)
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_cov = 0.0
            nseq = 0
            for ex in batch:
                prompt_messages = env.prompt_messages(ex)
                # Render the student prompt from the SAME messages the teacher conditions on;
                # env.prompt_messages can be stateful/randomized, so re-deriving it via render_prompt
                # (which calls prompt_messages again) would desync sampling from teacher scoring.
                prompt_text = tok.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=_w.THINKING,
                )
                prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
                # Drop over-budget prompts (like GRPO) rather than truncating: a truncated student
                # prompt would no longer match the teacher's full-prompt conditioning.
                if len(prompt_ids) > prompt_budget:
                    dropped_long += 1
                    continue
                prompt_tensor = torch.tensor([prompt_ids], device=device)
                for _g in range(group):
                    loss = _train_one(
                        model=model,
                        tok=tok,
                        teacher=teacher,
                        device=device,
                        prompt_ids=prompt_ids,
                        prompt_tensor=prompt_tensor,
                        prompt_messages=prompt_messages,
                        gen_cfg=gen_cfg,
                        knobs=knobs,
                        torch=torch,
                    )
                    if loss is None:
                        continue
                    (loss / accum_target).backward()
                    step_loss += float(loss.detach())
                    step_cov += _train_one.last_coverage
                    generated_tokens += _train_one.last_gen_tokens
                    teacher_input_tokens += _train_one.last_teacher_tokens
                    nseq += 1
            if nseq == 0:
                print(f"[opd] step {step}: no usable teacher signal this step (skipped)")
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
                step=step + 1,
                loss=avg_loss,
                coverage=avg_cov,
                gpu=gpu_diagnostics(include_torch=False),
            )
            if step % 10 == 0:
                print(
                    f"[opd] step {step + 1}/{steps} loss={avg_loss:.4f} "
                    f"coverage={avg_cov:.0%} seqs={nseq}"
                )
            if knobs["save_every"] and (step + 1) % knobs["save_every"] == 0:
                _save_adapter(model, tok, adapter_dir)
                _w.publish_deployable_checkpoint(adapter_dir, step + 1)

    train_wall = time.time() - t_train
    if not loss_curve:
        raise RuntimeError(
            "opd produced no trained step — every teacher scoring call failed or aligned to "
            "zero positions. Check FIREWORKS_API_KEY and the teacher model id. Failing loudly "
            "instead of reporting a no-op run as done."
        )

    _save_adapter(model, tok, adapter_dir)
    _w.hf_upload_folder(adapter_dir, "adapter", required=True)
    _w.publish_deployable_checkpoint(adapter_dir, steps)
    _w.heartbeat("opd_trained", train_wall=train_wall, gpu=gpu_diagnostics())

    _w.write_train_meta(
        phase="opd",
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
            "dropped_long_prompts": dropped_long,  # prompts skipped for exceeding the context budget
            "method": "gkd",
            "init_from_adapter": warm_start or None,
            "teacher_model": knobs["teacher_model"],
            "download_seconds": download_seconds,
            "thinking": _w.THINKING,
            "loss_curve": loss_curve,
            "mean_coverage": (sum(coverage_curve) / len(coverage_curve)) if coverage_curve else 0.0,
            "teacher_input_tokens": teacher_input_tokens,
            "temperature": knobs["temperature"],
            "group_size": group,
            "prompts_per_step": ppl_step,
            "max_completion_len": knobs["max_completion"],
            **_w.wandb_run_info(),
        },
    )
    free_gpu(model)


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
    torch,
):
    """Sample one student completion on-policy, score it with the teacher, and return the groupwise
    reverse-KL loss (or None). Side-channels per-sequence stats on function attributes to keep the
    caller compact."""
    _train_one.last_coverage = 0.0
    _train_one.last_gen_tokens = 0
    _train_one.last_teacher_tokens = 0

    # On-policy: the student samples; the teacher echo-scores that exact completion.
    model.eval()
    model.config.use_cache = True
    with torch.no_grad():
        gen = model.generate(prompt_tensor, **gen_cfg)
    completion_ids = gen[0, prompt_tensor.shape[1] :]
    completion_text = tok.decode(completion_ids, skip_special_tokens=True)
    _train_one.last_gen_tokens = int(completion_ids.shape[0])
    if not completion_text.strip():
        return None
    # NB: `stop_sequences` are wired into `generate` (gen_cfg.stop_strings), so the student halts at
    # the delimiter on-policy. We score the completion the student actually produced verbatim —
    # ids and text stay consistent (no post-hoc trimming that would desync gkd_loss / token counts).

    teacher_prompt = _teacher_prompt_text(prompt_messages)
    try:
        teacher_toks = teacher.score(teacher_prompt, completion_text)
    except TeacherError as e:
        if e.permanent:  # bad key / model id / malformed -> abort now, don't burn the whole run
            raise
        print(f"[opd] teacher score failed (transient, skipping sample): {e}")
        return None
    except Exception as e:
        print(f"[opd] teacher score failed (skipping sample): {e}")
        return None
    _train_one.last_teacher_tokens = len(prompt_ids) + _train_one.last_gen_tokens

    student_ids, student_toks = student_tokens_with_offsets(tok, completion_ids, completion_text)
    if not student_ids:
        return None

    model.train()
    model.config.use_cache = False
    # gkd — groupwise reverse-KL (spider/Tinker); covers every token from the realized logprobs.
    groups = groupwise_alignment(student_toks, teacher_toks)
    # Coverage = alignable (non-zero-width) student tokens that landed in a group / alignable total,
    # so it stays in [0, 1] (a zero-width eos/partial-byte token riding along in a group no longer
    # inflates it past 100%).
    _train_one.last_coverage = groupwise_coverage(groups, student_toks)
    return gkd_loss(model, prompt_ids, student_ids, groups, device, kl_coef=knobs["kl_coef"])


def _save_adapter(model, tok, adapter_dir: str) -> None:
    """Persist the LoRA adapter + tokenizer for deploy (identical layout to SFT)."""
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
