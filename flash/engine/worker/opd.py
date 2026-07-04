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

import contextlib
import os
import random
import time

from flash.engine.chalk_kernels import active_kernels, install_chalk_kernels
from flash.engine.recipe import RECIPE
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.heartbeat import liveness_heartbeat
from flash.engine.worker.perf import (
    RetriableInfraError,
    _sdpa_cudnn_ctx,
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


def _teacher_prompt_text(prompt_messages: list[dict], thinking_prefill: str = "") -> str:
    """Render the prompt for the teacher's scoring context (template-agnostic; ends at 'Assistant: ').

    ``thinking_prefill`` is the extra trailing text a thinking-mode student template opens after the
    generation prompt (e.g. Qwen's ``<think>\\n``). The student samples its on-policy completion AFTER
    that prefill, so the teacher must condition on the SAME trailing context; otherwise every
    thinking-mode token is scored against a prompt that never opened the reasoning block, and the gkd
    logprobs are conditioned on a different prefix than the sampled tokens (codex[bot]). Empty (the
    default) when thinking is off or the template ignores it -- the plain ``Assistant: `` already
    matches."""
    parts = []
    for m in prompt_messages:
        role = str(m.get("role", "user")).capitalize()
        parts.append(f"{role}: {m.get('content', '')!s}")
    parts.append("Assistant: " + thinking_prefill)
    return "\n".join(parts)


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
        if think_mid:
            return think_mid  # opener appended, or inserted before shared trailing template text
        # think adds nothing UNIQUE yet the renders differ (checked above): a CLOSED-BLOCK hybrid where
        # enable_thinking=False force-CLOSES the block (base has an extra "</think>..." the open render
        # lacks) so the shared "<think>" opener the student pre-fills was eaten into the common prefix.
        # Recover it: if base's unique middle is a closing tag ("</...>"), the opener is its opening form
        # ("<...>") that ends the common prefix — return it plus the thinking-side trailing text so the
        # teacher conditions on the same OPEN block instead of the empty prefix the delta alone gives
        # (codex[bot]/cursor). "" only when the model opens <think> itself inside the completion.
        base_mid = base[p : len(base) - s]
        if base_mid.startswith("</") and ">" in base_mid:
            open_tag = "<" + base_mid[2 : base_mid.index(">") + 1]  # "</think>..." -> "<think>"
            cut = think.rfind(open_tag, 0, p)
            if cut != -1:
                return think[cut:]  # e.g. "<think>\n"
    return ""


def _to_cpu_ids(completion_ids):
    """Move the sampled ids off the GPU in ONE transfer and reuse the CPU list for decode/trim/offsets.

    ``model.generate`` returns the ids on-device; iterating that tensor and calling ``int(t)`` per token
    does one scalar CUDA->CPU sync per token -- thousands of tiny syncs per sample once ``[train]
    .max_tokens`` is raised (reported by codex[bot]). A single ``detach().cpu().tolist()`` collapses that
    to one bulk copy. Falls back to a plain-list normalization when handed a list (tests / the already-
    trimmed path re-pass a list)."""
    if hasattr(completion_ids, "detach"):
        return completion_ids.detach().cpu().tolist()
    return [int(t) for t in completion_ids]


def _is_truncated_rollout(completion_ids, max_completion, eos_id) -> bool:
    """True when a rollout hit the ``max_completion`` cap WITHOUT emitting EOS — i.e. a completion cut
    off mid-output rather than one that terminated.

    OPD cannot supervise the stop token (the teacher's and student's EOS differ and are both zero-width
    in the text-span alignment, so EOS gets no gradient), so distilling a cap-hit fragment reinforces
    non-terminating output that reverse-KL can never teach the student to end — a driver of the eval's
    unterminated-JSON parse failures. A ``stop_sequence`` halt or a natural EOS both leave the rollout
    shorter than the cap or containing EOS, so neither trips this."""
    return len(completion_ids) >= int(max_completion) and (
        eos_id is None or eos_id not in completion_ids
    )


def student_tokens_with_offsets(tok, completion_ids, completion_text: str):
    """Char spans for the ORIGINAL sampled token ids, indexed into ``completion_text`` (the exact
    string the teacher echo-scored). Using the sampled ids — not a re-tokenization of the decoded
    text — keeps the loss on the true on-policy tokens. Offsets are built by incrementally decoding
    the id prefix and measuring its length (clamped monotonic in ``[0, len]``, so a byte-level token
    that splits a multi-byte char never yields a negative/backward span); a special token (e.g. eos)
    decodes to nothing, so it gets a zero-width span and is naturally excluded from the alignment.

    Split multi-byte chars (byte-level tokenizers): a char whose bytes span two+ ids decodes to the
    Unicode replacement char ``U+FFFD`` until its FINAL byte-id arrives. Measuring each half-id's
    decoded length independently would give one half the whole char and the other a zero-width span —
    silently dropping a real byte-token from the alignment and undercounting that char's student
    logprob. Instead we GROW the window while the decoded prefix still ends in ``U+FFFD`` and assign
    the SHARED completed-char span to every byte-id in it, so both halves share a ``start`` and land in
    the same alignment group. A normal (already-complete) token is a window of one — identical to the
    old per-token length measurement."""
    ids = [int(t) for t in completion_ids]
    toks: list[StudentToken] = []
    prev = 0
    n = len(completion_text)
    i = 0
    while i < len(ids):
        j = i
        # Grow the window over a split multi-byte char, decoding ONLY the window from the current
        # boundary (ids[i:j+1]) -- never the whole prefix ids[:j+1] -- so total decoding is O(len), not
        # O(len^2) (reported by codex[bot]; the quadratic bites once max_tokens raises completions to
        # 1000s of tokens). A byte-level tokenizer renders an INCOMPLETE char as a trailing U+FFFD that
        # is NOT the real char at completion_text[prev:]; a genuine U+FFFD the model emitted DOES match
        # there (startswith from prev), so we stop and keep it as its own span, not over-merged.
        while j + 1 < len(ids):
            dec = tok.decode(ids[i : j + 1], skip_special_tokens=True)
            if not dec.endswith("\ufffd") or completion_text.startswith(dec, prev):
                break
            j += 1
        # end = where THIS window's decoded text ends in completion_text. Anchor to completion_text
        # (the ground-truth full decode) via find() from prev rather than prev + len(decode(window)):
        # a SentencePiece/LLaMA tokenizer decodes a word token IN ISOLATION without its leading
        # word-boundary space (decode([▁world]) == "world", not " world"), so prev + len(window) would
        # undercount the span by one char and drift EVERY following offset -- misaligning teacher spans
        # onto the wrong sampled ids (codex[bot]). find() locates where the window text actually sits
        # (skipping any dropped leading whitespace, which is absorbed into this token's start) and the
        # start stays pinned at prev so the spans remain contiguous. For a byte-level tokenizer (Qwen,
        # GPT) the window already carries its space, find() returns prev, and end == the old value. An
        # empty decode (special/eos) -> find() returns prev -> zero-width span, unchanged.
        window_text = tok.decode(ids[i : j + 1], skip_special_tokens=True)
        hit = completion_text.find(window_text, prev)
        end = (hit + len(window_text)) if hit != -1 else prev + len(window_text)
        end = min(n, max(prev, end))
        toks.extend(StudentToken(token_id=ids[k], start=prev, end=end) for k in range(i, j + 1))
        prev = end
        i = j + 1
    return ids, toks


def _trim_trailing_stop(tok, completion_ids, completion_text: str, stops):
    """Drop a trailing stop delimiter from BOTH the decoded text and the sampled ids (token-level).

    HF ``stop_strings`` halts only AFTER the delimiter text is emitted, so a run with e.g.
    ``[train] stop_sequences=["</answer>"]`` would otherwise score/distil the delimiter the user asked
    to stop at. Trimming both sides keeps ids and text consistent (no gkd_loss / token-count desync,
    the failure mode of a text-only trim). Returns ``(ids, text)`` (a list + str)."""
    ids = [int(t) for t in completion_ids]
    # Pick the LONGEST configured stop that is a trailing match (the earliest stop boundary in the
    # text). Overlapping delimiters like ["\n", "\n\n"] would otherwise trim only the first-listed
    # shorter suffix off a "\n\n" tail, leaving one newline for the teacher to score/distil; taking
    # the longest match removes the whole delimiter in one shot regardless of config order.
    stop = max(
        (s for s in stops if s and completion_text.endswith(s)),
        key=len,
        default="",
    )
    if not stop:
        return ids, completion_text
    keep_len = len(completion_text) - len(stop)
    # Locate the kept prefix by scanning from the END: the delimiter is short, so only a handful of
    # trailing tokens are dropped -> O(dropped * completion) work. Decoding growing prefixes from the
    # START (ids[:1], ids[:2], ...) instead is O(completion^2) and can dominate CPU ahead of teacher
    # scoring on long completions once [train].max_tokens is raised.
    kept = len(ids)
    while kept > 0 and len(tok.decode(ids[:kept], skip_special_tokens=True)) > keep_len:
        kept -= 1
    # Trim the TEXT to exactly what the kept ids decode to — NOT completion_text[:keep_len].
    # When the stop starts INSIDE the final sampled token (that token decodes to e.g.
    # "B</answer>" while the stop is "</answer>"), that whole token is excluded from `kept`, so
    # slicing the raw text at keep_len would keep a "B" the kept ids can no longer represent —
    # desyncing the teacher-scored text from the student ids (gkd_loss / token-count skew).
    # Decoding the kept ids keeps both sides identical (drops the fused char, never distils the
    # delimiter); in the common case (the stop is its own clean token) it equals
    # completion_text[:keep_len].
    return ids[:kept], tok.decode(ids[:kept], skip_special_tokens=True)


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
        # Keep the detached group logprob sum ON-DEVICE — NO float(): float() forces a CUDA->CPU sync
        # per alignment group, i.e. thousands of tiny device syncs on a long sample (reported by
        # codex[bot]). teacher_logsum is a Python float, so (device tensor - float) stays a 0-dim
        # device tensor and coeff * sp[j] never leaves the GPU; the only sync is the final .mean().
        student_logsum_det = sum(sp[j].detach() for j in s_idx)
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


def run_opd():
    import torch
    from transformers import AutoTokenizer

    from flash.engine.worker.teacher import TeacherClient

    env = _w.require_active_env()
    if getattr(env, "multi_turn", False) or getattr(env, "is_tool_env", False):
        # opd distills a SINGLE model.generate() over env.prompt_messages() and never drives the
        # turn loop (env.new_rollout_state / env.env_reply) or hands tool schemas to generation — GRPO
        # does, in rl.py. On a multi-turn / tool-calling env opd would silently distill only the FIRST
        # assistant turn, so fail fast at setup (before any GPU work) rather than train a wrong
        # objective. (Reported by codex[bot]; drop this guard once opd implements the rollout path.)
        raise RuntimeError(
            "opd does not support multi-turn or tool-calling environments yet: it samples one "
            "completion per prompt and cannot drive the turn/tool loop, so it would distill only "
            "the first assistant turn. Use grpo for this environment, or a single-turn env for opd."
        )
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
    steps = knobs["steps"]
    ppl_step = knobs["prompts_per_step"]
    group = knobs["group_size"]
    # Prompt budget mirrors GRPO: DROP (not truncate) prompts over the context budget, so the student
    # never conditions on a truncated prompt the teacher didn't see. Use the configured max_length
    # when set, else the recipe prompt cap.
    if knobs["max_length"]:
        prompt_budget = knobs["max_length"] - knobs["max_completion"]
        if prompt_budget < 1:
            # A non-positive remainder means max_length <= max_tokens: there is no room for any
            # prompt, so every sample would run generate+loss past the configured context. Reject
            # loudly instead of clamping to a 1-token budget that silently admits over-budget runs.
            raise RuntimeError(
                f"opd: [train] max_length ({knobs['max_length']}) leaves no prompt budget after "
                f"max_tokens ({knobs['max_completion']}); set max_length > max_tokens."
            )
    else:
        prompt_budget = RECIPE.opd.max_prompt_len
    dropped_long = 0

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
            if len(_render_prompt_ids(env.prompt_messages(ex))) <= prompt_budget:
                examples.append(ex)
            _scanned += 1
    n_over_budget = len(train) - len(examples)
    if not examples:
        raise RuntimeError(
            f"opd: every prompt exceeds the {prompt_budget}-token budget "
            f"(max_length={knobs['max_length'] or 'unset'}, "
            f"max_completion={knobs['max_completion']}). Raise [train].max_length or shorten "
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
        seq_cap = knobs["max_length"] or (RECIPE.opd.max_prompt_len + knobs["max_completion"])
        if grad_checkpointing_on(model_id, seq_cap):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            print("[opd] gradient checkpointing enabled")
    model.config.use_cache = (
        True  # generation needs the KV cache; re-disabled per loss forward below
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=knobs["learning_rate"]
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
        "do_sample": knobs["temperature"] > 0,
        "temperature": max(knobs["temperature"], 1e-5),
        "top_p": knobs["top_p"],
        "max_new_tokens": knobs["max_completion"],
        "pad_token_id": tok.pad_token_id,
        # Bound generation WALL-TIME so a degenerate or near-OOM-thrashing generate cannot silently
        # eat the training stall window: a single _train_one blocks the loop while it runs (the
        # per-sample opd_step heartbeat only fires AFTER it returns), so an unbounded generate that
        # stops making progress emits no non-liveness ping and the poller reaps the whole attempt as
        # "stalled" (observed on the OPD-16 linkd e2e: one step wedged >1500s at 93% VRAM). HF checks
        # max_time between token steps, so a thrashing (still-stepping) generate is cut here and the
        # sample returns partial/empty -> the heartbeat resumes. Scale with the completion budget
        # (thinking mode needs longer) but keep it well under the poller's ~1500s training window.
        "max_time": min(900.0, max(180.0, float(knobs["max_completion"]) * 0.75)),
    }
    if knobs["stop_sequences"]:
        # HF stops generation at any of these strings (needs the tokenizer to match them on decode).
        gen_cfg["stop_strings"] = list(knobs["stop_sequences"])
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
            for ex in batch:
                prompt_messages = env.prompt_messages(ex)
                # Render the student prompt from the SAME messages the teacher conditions on;
                # env.prompt_messages can be stateful/randomized, so re-deriving it via render_prompt
                # (which calls prompt_messages again) would desync sampling from teacher scoring.
                prompt_ids = _render_prompt_ids(prompt_messages)
                # Drop over-budget prompts (like GRPO) rather than truncating: a truncated student
                # prompt would no longer match the teacher's full-prompt conditioning.
                if len(prompt_ids) > prompt_budget:
                    dropped_long += 1
                    # Refresh the stall clock on the drop (see the no-signal skip below): a
                    # randomized env can re-render every prompt over budget for a whole step, which
                    # would otherwise emit no non-liveness ping and leave the stall clock unrefreshed.
                    _w.heartbeat("opd_step", step=opt_steps, samples_done=nseq)
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
                        thinking_prefill=thinking_prefill,
                        # Refresh the stall clock between generation and teacher scoring (see
                        # _train_one): opt_steps step-gates the ping so it keeps the wide setup grace
                        # during the first step and tightens correctly once a real update lands.
                        # Bind opt_steps/nseq as defaults (called synchronously inside this iteration,
                        # so the current values are the right ones) to satisfy the loop-var-capture lint.
                        on_generated=lambda s=opt_steps, n=nseq: _w.heartbeat(
                            "opd_step", step=s, samples_done=n
                        ),
                    )
                    samples_seen += 1  # advances the liveness-thread progress signal
                    _t_status = getattr(_train_one, "last_teacher_status", None)
                    if _t_status == "ok":
                        teacher_ok += 1
                    elif _t_status == "transient":
                        teacher_transient += 1
                    if getattr(_train_one, "last_truncated", False):
                        truncated_rollouts += 1
                    if loss is None:
                        # Refresh the stall clock even when a sample yields no teacher signal. The
                        # success ping below is the only NON-liveness opd_step heartbeat, so a step
                        # where every sample skips (empty completions, or slow/retrying Fireworks)
                        # would otherwise emit only liveness pings — which the pollers ignore — and a
                        # prolonged all-skip stretch on a later step (opt_steps>=1, tight window)
                        # could be reaped as stalled. Report opt_steps (same step-gating as the
                        # success ping) so a still-accumulating first step keeps the wide setup grace.
                        _w.heartbeat("opd_step", step=opt_steps, samples_done=nseq)
                        continue
                    (loss / accum_target).backward()
                    step_loss += float(loss.detach())
                    step_cov += _train_one.last_coverage
                    granularity_sum += _train_one.last_group_granularity
                    granularity_n += 1
                    generated_tokens += _train_one.last_gen_tokens
                    teacher_input_tokens += _train_one.last_teacher_tokens
                    nseq += 1
                    # Non-liveness progress ping WITHIN the step: the pollers ignore liveness
                    # heartbeats, so a long teacher-bound step (serial scoring of a large
                    # batch/group, or slow/retrying Fireworks) would otherwise trip the training
                    # stall window. Report opt_steps (optimizer updates COMPLETED so far), not the
                    # loop index: opd_step is step-gated in the poller (_poll.STEP_GATED_STAGES), so
                    # while the FIRST optimizer step is still accumulating (opt_steps==0) these pings
                    # keep the WIDE setup grace — they must not flip a still-running first step into
                    # the tight training window. Once a real update has landed (opt_steps>=1) the
                    # pings tighten it as intended, and the opd_step throttle bounds the HF upload rate.
                    _w.heartbeat("opd_step", step=opt_steps, samples_done=nseq)
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
            if knobs["save_every"] and opt_steps % knobs["save_every"] == 0:
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
        # Real updates landed, but skips (over-budget prompts / empty completions / an intermittent
        # teacher) kept the loop from reaching the requested `steps` optimizer updates within the
        # iteration budget. Publishing now would serve an under-trained adapter as the DEFAULT while
        # billing the full `steps` quote, so RETRY (a healthier teacher / less degenerate sampling may
        # complete it next attempt) rather than silently shipping short (codex[bot]). The intermediate
        # non-default checkpoints already published stay available; only the served default is gated.
        raise RetriableInfraError(
            f"opd reached only {opt_steps}/{steps} optimizer updates within {max_iters} iterations "
            f"({dropped_long} prompts dropped over budget, {teacher_transient} transient teacher "
            "failures) — retrying rather than publishing an under-trained adapter billed as full steps."
        )

    _save_adapter(model, tok, adapter_dir)
    # Ship the deployable adapter (VL warm-start: recombine SFT⊕opd so it reproduces base+SFT+opd on
    # the catalog base; no-op for text/fresh). Name the final checkpoint by real optimizer steps
    # applied, not the planned `steps` count.
    _publish_opd_deployable(adapter_dir, opt_steps, as_default=True)
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
            "chalk_kernels": _chalk_active or None,
            "thinking": _w.THINKING,
            "loss_curve": loss_curve,
            "mean_coverage": (sum(coverage_curve) / len(coverage_curve)) if coverage_curve else 0.0,
            # Rollouts cut off at the cap without EOS — skipped, not distilled (see _is_truncated_rollout).
            # A high count means the student is generating runaway/non-terminating filters (cold start,
            # untrained stop token) and warm-starting from SFT — which encodes termination — would help.
            "truncated_rollouts": truncated_rollouts,
            # Real alignment-health signal (mean student-tokens-per-group); mean_coverage is ~1.0 even
            # for a degenerate collapsed alignment, so it can't flag that failure mode.
            "mean_align_granularity": (granularity_sum / granularity_n) if granularity_n else 0.0,
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
    thinking_prefill="",
    on_generated=None,
):
    """Sample one student completion on-policy, score it with the teacher, and return the groupwise
    reverse-KL loss (or None). Side-channels per-sequence stats on function attributes to keep the
    caller compact. ``thinking_prefill`` is appended to the teacher prompt so it conditions on the
    same trailing context the student sampled after in thinking mode. ``on_generated`` (optional) is
    called AFTER generation and BEFORE teacher scoring to refresh the stall clock: both the
    max_time-bounded generate and the retrying teacher call block for a long time and the caller's
    per-sample progress ping only fires AFTER scoring returns, so without a mid-sample refresh a slow
    generation followed by a teacher outage can span >1200s with no non-liveness heartbeat and be
    reaped as stalled before the transient-teacher handling runs."""
    _train_one.last_coverage = 0.0
    _train_one.last_gen_tokens = 0
    _train_one.last_teacher_tokens = 0
    # A cap-hit rollout truncated mid-output (skipped, not distilled — see _is_truncated_rollout).
    _train_one.last_truncated = False
    # Mean student-tokens-per-alignment-group; a real health signal where mean_coverage is not.
    _train_one.last_group_granularity = 0.0
    # "ok" once teacher.score returns, "transient" on a retryable teacher outage, else None (teacher
    # not reached). run_opd uses this to decide whether a no-signal run is a retriable infra failure.
    _train_one.last_teacher_status = None

    # On-policy: the student samples; the teacher echo-scores that exact completion.
    model.eval()
    model.config.use_cache = True
    with torch.no_grad():
        gen = model.generate(prompt_tensor, **gen_cfg)
    completion_ids = _to_cpu_ids(
        gen[0, prompt_tensor.shape[1] :]
    )  # one GPU->CPU copy, reused below
    completion_text = tok.decode(completion_ids, skip_special_tokens=True)
    # Drop a cap-hit truncation BEFORE scoring/distilling it: a filter cut off mid-output would
    # otherwise be echo-scored and reinforced, teaching the student a runaway it can never be taught to
    # end (OPD can't supervise the stop token). Detect on the RAW rollout, before any stop_sequence
    # trim (a stop-halted rollout is shorter than the cap, so it never trips this). Skip + count.
    if _is_truncated_rollout(completion_ids, knobs["max_completion"], tok.eos_token_id):
        _train_one.last_truncated = True
        _train_one.last_gen_tokens = len(completion_ids)
        return None
    # `stop_sequences` halt generation on-policy (gen_cfg.stop_strings), but HF emits the delimiter
    # before stopping — trim it from BOTH ids and text (token-level) so the teacher scores/distils
    # only the answer, and ids/text stay consistent for gkd_loss + token counting.
    if knobs["stop_sequences"]:
        completion_ids, completion_text = _trim_trailing_stop(
            tok, completion_ids, completion_text, knobs["stop_sequences"]
        )
    _train_one.last_gen_tokens = len(completion_ids)
    if not completion_text.strip():
        return None

    # Refresh the stall clock between the (gen_cfg.max_time-bounded, up to ~900s) generation and the
    # retrying teacher call (up to four ~90s timeouts): both block, and the caller only emits its
    # per-sample opd_step ping AFTER scoring returns, so a slow generation + a teacher outage could
    # otherwise span >1200s with no non-liveness heartbeat and be reaped as stalled once opt_steps>=1
    # (codex[bot]). Splitting the gap here keeps each blocking phase under the training stall window.
    if on_generated is not None:
        on_generated()

    teacher_prompt = _teacher_prompt_text(prompt_messages, thinking_prefill)
    try:
        teacher_toks = teacher.score(teacher_prompt, completion_text)
    except TeacherError as e:
        if e.permanent:  # bad key / model id / malformed -> abort now, don't burn the whole run
            raise
        _train_one.last_teacher_status = (
            "transient"  # retryable outage -> may make the run retriable
        )
        print(f"[opd] teacher score failed (transient, skipping sample): {e}")
        return None
    except Exception as e:
        print(f"[opd] teacher score failed (skipping sample): {e}")
        return None
    _train_one.last_teacher_status = "ok"
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
    # mean_coverage is structurally ~1.0 for a CORRECT fine-grained alignment AND for a degenerate
    # collapsed-into-one-giant-span alignment, so it can't detect the latter. Mean student-tokens-per-
    # group does: ~1.0 == each token its own group (healthy); large == coarse spans smearing one
    # teacher logprob across many student tokens.
    _n_align = sum(1 for st in student_toks if st.end > st.start)
    _train_one.last_group_granularity = (_n_align / len(groups)) if groups else 0.0
    return gkd_loss(model, prompt_ids, student_ids, groups, device, kl_coef=knobs["kl_coef"])


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
