"""Pure token-alignment + groupwise reverse-KL math for on-policy distillation (algorithm="opd").

Extracted from ``opd.py`` to keep that module under the size bar and make the cross-tokenizer
alignment + GKD loss an independently unit-testable unit. Every function here is PURE: it takes the
tokenizer / ids / model as arguments and touches NO worker (``_w``) globals, recipe, or run state.
See ``opd.py`` for the orchestration (``run_opd`` / ``_train_one``) and docs/on-policy-distillation.md
for the method. ``opd.py`` re-imports these names, so existing ``from ...opd import <fn>`` call sites
(and tests) are unaffected.
"""

from __future__ import annotations

from flash.engine.worker.tokenizer_align import StudentToken


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


def _rollout_terminated(completion_ids, completion_text, eos_id, stop_sequences) -> bool:
    """True iff the rollout ended NATURALLY — the student emitted EOS, or (when stop_sequences are
    configured) the decoded text ends with a stop delimiter.

    HF ``generate`` halts on four conditions — EOS, a ``stop_strings`` match, the ``max_new_tokens``
    cap, or the ``gen_cfg.max_time`` wall-clock bound — and only the first two are natural completions.
    A cap hit OR a max_time cut leaves the output cut off mid-JSON. OPD cannot supervise the stop token
    (the teacher's and student's EOS differ and are both zero-width in the text-span alignment, so EOS
    gets no gradient), so distilling such a fragment reinforces non-terminating output that reverse-KL
    can never teach the student to end — a driver of the eval's unterminated-JSON parse failures. The
    caller skips anything that isn't terminated (codex[bot]).

    EOS is checked on the IDS (``completion_text`` is decoded with ``skip_special_tokens=True``, so EOS
    isn't visible there). The stop delimiter IS in the raw text — HF emits it before halting — matched
    with the same trailing-``endswith`` semantics ``_trim_trailing_stop`` uses to remove it, so a
    stop-terminated rollout is recognised even when the delimiter lands as the final token AT the cap
    (which a length-only check wrongly discarded). Fail OPEN only when NO termination signal exists at
    all (no eos_token_id AND no stop_sequences): we then can't tell a finished answer from a cut-off one,
    so we distil rather than skip every rollout. Real tokenizers always define eos_token_id, so in
    production a cap/max_time cut without EOS or a stop delimiter correctly returns False (skip)."""
    if eos_id is not None and eos_id in completion_ids:
        return True
    if stop_sequences and any(s and completion_text.endswith(s) for s in stop_sequences):
        return True
    return eos_id is None and not stop_sequences


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
        # Decode the window ONCE, then re-decode only when it actually grows over a split char (halves
        # the decode calls on the common single-token path vs a separate probe decode + a final decode).
        window_text = tok.decode(ids[i : j + 1], skip_special_tokens=True)
        while (
            j + 1 < len(ids)
            and window_text.endswith("\ufffd")
            and not completion_text.startswith(window_text, prev)
        ):
            j += 1
            window_text = tok.decode(ids[i : j + 1], skip_special_tokens=True)
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
    terms = []
    for s_idx, teacher_logsum in groups:
        if not s_idx:  # defensive: a teacher-only span carries no student token to supervise
            continue
        # Keep the detached group logprob sum ON-DEVICE — NO float(): float() forces a CUDA->CPU sync
        # per alignment group, i.e. thousands of tiny device syncs on a long sample (reported by
        # codex[bot]). teacher_logsum is a Python float, so (device tensor - float) stays a 0-dim
        # device tensor and coeff * sp_t[j] never leaves the GPU; the only sync is the final .mean().
        student_logsum_det = sum(sp_t[j].detach() for j in s_idx)
        # coeff > 0 where the student is MORE confident than the teacher on the span (push down);
        # coeff < 0 where the teacher is more confident (push up). Gradient = reverse-KL gradient.
        coeff = kl_coef * (student_logsum_det - teacher_logsum) / len(s_idx)
        terms.extend(coeff * sp_t[j] for j in s_idx)
    if not terms:
        return None
    return torch.stack(terms).mean()
