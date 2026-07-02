"""On-policy distillation training path (algorithm="opd") for the fine-tuning worker.

The student (a Qwen catalog model) samples completions on-policy; a Fireworks-hosted GLM teacher
scores those completions token-by-token over the API; a per-token distillation loss is
backpropagated through the student LoRA. Four cross-tokenizer strategies bridge the teacher/student
vocabulary mismatch (see ``tokenizer_align`` and docs/on-policy-distillation.md):

- ``gkd`` (default): groupwise reverse-KL over shared decoded-text spans (the collinear-ai *spider* /
  Tinker method) — uses only realized-token logprobs, so it covers every token exactly.
- ``align``: sparse top-k forward-KL, teacher candidates projected onto the student vocab.
- ``uld``: sorted-distribution (optimal-transport) matching, vocabulary-agnostic.
- ``seqkd``: off-policy sequence KD — the teacher generates targets, student trains with CE.

There is NO local reference model and NO colocated vLLM engine: sampling is HF ``generate`` on the
resident student (so the VRAM profile matches SFT), and the teacher lives behind the API. All heavy
imports (torch/transformers/peft) are inside functions, so importing this module is CPU/offline-safe.
"""

from __future__ import annotations

import functools
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
from flash.engine.worker.tokenizer_align import (
    StudentToken,
    align_targets,
    coverage,
    groupwise_alignment,
    groupwise_coverage,
    uld_targets,
)


def _resolve_opd_knobs():
    """Resolve every opd knob from the JobSpec's [train] table, falling back to RECIPE.opd."""
    d = RECIPE.opd
    t = _w.JOB_SPEC.train if _w.JOB_SPEC else None

    def opt(name, default):
        v = getattr(t, name, None) if t else None
        return v if v is not None else default

    strategy = (opt("tokenizer_alignment", "") or d.tokenizer_alignment).lower()
    max_completion = int(
        opt("max_tokens", 0)
        or (d.max_completion_len_thinking if _w.THINKING else d.max_completion_len)
    )
    return {
        "teacher_model": opt("teacher_model", "") or d.teacher_model,
        "teacher_base_url": d.teacher_base_url,
        "strategy": strategy,
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
        "top_logprobs": int(opt("teacher_top_logprobs", 0) or d.teacher_top_logprobs),
        "kd_temperature": d.kd_temperature,
        # gkd reverse-KL scale; reuses the existing [train] kl_penalty_coef knob (default 1.0).
        "kl_coef": float(opt("kl_penalty_coef", None) or d.kl_coef),
        "save_every": int(opt("save_every", 0) or 20),
        "max_length": int(opt("max_length", 0) or 0),
    }


def _teacher_prompt_text(prompt_messages: list[dict]) -> str:
    """Render the prompt for the teacher's scoring context (template-agnostic; ends at 'Assistant: ')."""
    parts = []
    for m in prompt_messages:
        role = str(m.get("role", "user")).capitalize()
        parts.append(f"{role}: {m.get('content', '')!s}")
    parts.append("Assistant: ")
    return "\n".join(parts)


def student_tokens_with_offsets(tok, completion_text: str):
    """Tokenize the completion with character offsets. Returns ``(ids, [StudentToken, ...])`` sharing
    the SAME token sequence used for the loss forward, so student offsets and ids stay consistent."""
    enc = tok(completion_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = enc["offset_mapping"]
    toks = [
        StudentToken(token_id=i, start=int(a), end=int(b))
        for i, (a, b) in zip(ids, offsets, strict=True)
    ]
    return ids, toks


def _forward_logprobs(model, input_ids):
    """Log-softmax over the vocab at every position (fp32 for stability). ``input_ids``: [1, T]."""
    import torch

    out = model(input_ids)
    return torch.log_softmax(out.logits[0].float(), dim=-1)  # [T, V]


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
    logp = _forward_logprobs(model, input_ids)  # [T, V]
    P = len(prompt_ids)
    # Differentiable student logprob of each REALIZED completion token: logits at P+j-1 predict token j.
    sp = [logp[P + j - 1, student_ids[j]] for j in range(len(student_ids))]
    terms = []
    for s_idx, teacher_logsum in groups:
        student_logsum_det = float(sum(sp[j].detach() for j in s_idx))
        # coeff > 0 where the student is MORE confident than the teacher on the span (push down);
        # coeff < 0 where the teacher is more confident (push up). Gradient = reverse-KL gradient.
        coeff = kl_coef * (student_logsum_det - teacher_logsum) / len(s_idx)
        terms.extend(coeff * sp[j] for j in s_idx)
    if not terms:
        return None
    return torch.stack(terms).mean()


def align_loss(model, prompt_ids, student_ids, targets, device):
    """Sparse top-k forward-KL: at each aligned position push the student toward the teacher's
    projected candidate distribution. Returns a scalar loss (grad) or None if nothing aligned."""
    import torch

    if not student_ids or not any(t for t in targets):
        return None
    input_ids = torch.tensor([prompt_ids + student_ids], device=device)
    logp = _forward_logprobs(model, input_ids)  # [T, V]
    P = len(prompt_ids)
    terms = []
    for j, target in enumerate(targets):
        if not target:
            continue
        pos = P + j - 1  # logits at pos predict the (P+j)-th token = the j-th completion token
        ids = torch.tensor(list(target.keys()), device=device)
        w = torch.tensor(list(target.values()), device=device, dtype=logp.dtype)
        terms.append(-(w * logp[pos].index_select(0, ids)).sum())
    if not terms:
        return None
    return torch.stack(terms).mean()


def uld_loss(model, prompt_ids, student_ids, uld_tgts, device, top_k):
    """Universal Logit Distillation: L1 between the sorted teacher top-k and the student's own sorted
    top-k probabilities at each aligned position (no token identity). Scalar loss or None."""
    import torch

    if not student_ids or not any(t for t in uld_tgts):
        return None
    input_ids = torch.tensor([prompt_ids + student_ids], device=device)
    out = model(input_ids)
    probs = torch.softmax(out.logits[0].float(), dim=-1)  # [T, V]
    P = len(prompt_ids)
    terms = []
    for j, tvec in enumerate(uld_tgts):
        if not tvec:
            continue
        pos = P + j - 1
        k = max(len(tvec), top_k)
        s_top = torch.topk(probs[pos], min(k, probs.shape[-1])).values  # sorted desc, grad-carrying
        s_top = s_top / (s_top.sum() + 1e-8)
        t_top = torch.tensor(tvec, device=device, dtype=s_top.dtype)
        m = max(s_top.shape[0], t_top.shape[0])
        s_pad = torch.nn.functional.pad(s_top, (0, m - s_top.shape[0]))
        t_pad = torch.nn.functional.pad(t_top, (0, m - t_top.shape[0]))
        terms.append((s_pad - t_pad).abs().sum())
    if not terms:
        return None
    return torch.stack(terms).mean()


def seqkd_loss(model, prompt_ids, target_ids, device):
    """Off-policy sequence KD: completion-only cross-entropy on the teacher-generated target."""
    import torch

    if not target_ids:
        return None
    input_ids = torch.tensor([prompt_ids + target_ids], device=device)
    out = model(input_ids)
    logits = out.logits[0][:-1].float()  # predict tokens [1:]
    labels = torch.tensor([-100] * len(prompt_ids) + list(target_ids), device=device)[1:]
    return torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)


def run_opd():
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from flash.engine.worker.teacher import TeacherClient

    env = _w.require_active_env()
    t_start = time.time()
    _w.heartbeat("opd_start", gpu=gpu_diagnostics())
    knobs = _resolve_opd_knobs()
    strategy = knobs["strategy"]
    print(f"[opd] strategy={strategy} teacher={knobs['teacher_model']} steps={knobs['steps']}")

    api_key = os.environ.get("FIREWORKS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "opd requires the FIREWORKS_API_KEY runtime secret (the GLM teacher); it was not "
            "delivered to the worker. Declare it under [environment] secrets and export it locally."
        )
    teacher = TeacherClient(
        api_key,
        knobs["teacher_base_url"],
        knobs["teacher_model"],
        top_logprobs=knobs["top_logprobs"],
    )

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
        base = AutoModelForCausalLM.from_pretrained(model_id, **mik).to(device)
        model = get_peft_model(base, _w.make_lora(model_id))
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
    rng = random.Random(_w.SEED)
    rng.shuffle(train)
    examples = train

    @functools.lru_cache(maxsize=65536)
    def first_token_id(surface: str):
        if not surface:
            return None
        ids = tok(surface, add_special_tokens=False).input_ids
        return int(ids[0]) if ids else None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=knobs["learning_rate"]
    )
    steps = knobs["steps"]
    ppl_step = knobs["prompts_per_step"]
    group = knobs["group_size"]
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
    loss_curve: list[float] = []
    coverage_curve: list[float] = []
    generated_tokens = 0
    teacher_input_tokens = 0
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
                prompt_text = _w.render_prompt(tok, ex)
                prompt_ids = tok(prompt_text, add_special_tokens=False).input_ids
                prompt_tensor = torch.tensor([prompt_ids], device=device)
                for _g in range(group):
                    loss = _train_one(
                        model=model,
                        tok=tok,
                        teacher=teacher,
                        strategy=strategy,
                        device=device,
                        prompt_ids=prompt_ids,
                        prompt_tensor=prompt_tensor,
                        prompt_messages=prompt_messages,
                        gen_cfg=gen_cfg,
                        knobs=knobs,
                        first_token_id=first_token_id,
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
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
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
            "zero positions. Check FIREWORKS_API_KEY, the teacher model id, and the tokenizer "
            "alignment strategy. Failing loudly instead of reporting a no-op run as done."
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
            "strategy": strategy,
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
    strategy,
    device,
    prompt_ids,
    prompt_tensor,
    prompt_messages,
    gen_cfg,
    knobs,
    first_token_id,
    torch,
):
    """Sample one student completion, score it with the teacher, and return the strategy's loss (or
    None). Side-channels per-sequence stats on function attributes to keep the caller compact."""
    _train_one.last_coverage = 0.0
    _train_one.last_gen_tokens = 0
    _train_one.last_teacher_tokens = 0

    if strategy == "seqkd":
        # Off-policy: the teacher generates the target; the student imitates it.
        try:
            target_text = teacher.generate(
                prompt_messages, max_tokens=knobs["max_completion"], temperature=0.7
            )
        except Exception as e:
            print(f"[opd] teacher generate failed: {e}")
            return None
        target_ids = tok(target_text, add_special_tokens=False).input_ids
        if not target_ids:
            return None
        _train_one.last_teacher_tokens = len(target_ids)
        model.train()
        model.config.use_cache = False
        return seqkd_loss(model, prompt_ids, target_ids, device)

    # On-policy: the student samples; the teacher scores that completion.
    model.eval()
    model.config.use_cache = True
    with torch.no_grad():
        gen = model.generate(prompt_tensor, **gen_cfg)
    completion_ids = gen[0, prompt_tensor.shape[1] :]
    completion_text = tok.decode(completion_ids, skip_special_tokens=True)
    _train_one.last_gen_tokens = int(completion_ids.shape[0])
    if not completion_text.strip():
        return None

    teacher_prompt = _teacher_prompt_text(prompt_messages)
    try:
        teacher_toks = teacher.score(teacher_prompt, completion_text)
    except Exception as e:
        print(f"[opd] teacher score failed: {e}")
        return None
    _train_one.last_teacher_tokens = len(prompt_ids) + _train_one.last_gen_tokens

    student_ids, student_toks = student_tokens_with_offsets(tok, completion_text)
    if not student_ids:
        return None

    model.train()
    model.config.use_cache = False
    if strategy == "align":
        tgts = align_targets(
            student_toks, teacher_toks, first_token_id, kd_temperature=knobs["kd_temperature"]
        )
        _train_one.last_coverage = coverage(tgts)
        return align_loss(model, prompt_ids, student_ids, tgts, device)
    if strategy == "uld":
        tgts = uld_targets(student_toks, teacher_toks, kd_temperature=knobs["kd_temperature"])
        _train_one.last_coverage = coverage(tgts)
        return uld_loss(model, prompt_ids, student_ids, tgts, device, knobs["top_logprobs"])
    # default: gkd — groupwise reverse-KL (spider/Tinker), covers every token from realized logprobs.
    groups = groupwise_alignment(student_toks, teacher_toks)
    _train_one.last_coverage = groupwise_coverage(groups, len(student_ids))
    return gkd_loss(model, prompt_ids, student_ids, groups, device, kl_coef=knobs["kl_coef"])


def _save_adapter(model, tok, adapter_dir: str) -> None:
    """Persist the LoRA adapter + tokenizer for deploy (identical layout to SFT)."""
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
