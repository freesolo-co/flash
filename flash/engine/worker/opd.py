"""On-policy distillation entry point (algorithm="opd") and the knob/prompt helpers it shares.

``run_opd`` delegates to ``opd_train.run_opd_train``, which owns the whole training path: rollout,
teacher scoring, and the groupwise reverse-KL backward. What stays here is the part the verl worker
imports rather than reimplements — ``OpdKnobs`` / ``_resolve_opd_knobs`` (the [train] table's
resolved view), ``_thinking_prefill_text`` (the reasoning-block opener the teacher prompt must
match), and ``_drop_fully_forced_groups`` (grammar-forced spans carry no student choice, so they
carry no signal).

On the distillation objective itself, see ``opd_train`` for the implementation and
``tokenizer_align`` for the cross-tokenizer bridge: teacher and student tokenize the same completion
differently, so their distributions are compared over SHARED DECODED-TEXT SPANS using only
realized-token logprobs, which is exact across arbitrary tokenizer mismatch.

This module holds no heavy imports, so importing it stays CPU/offline-safe.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from flash.engine.recipe import RECIPE, resolve_teacher
from flash.engine.vram import opd_completion_len
from flash.engine.worker._pkg import W as _w


@dataclass(frozen=True)
class OpdKnobs:
    """Resolved opd knobs from the JobSpec's [train] table (falling back to RECIPE.opd), returned by
    ``_resolve_opd_knobs``. A typed container replacing the old stringly-typed dict; field names match
    the former dict keys one-for-one. The defaults are placeholders only — ``_resolve_opd_knobs``
    always sets every field explicitly — kept so partial construction stays ergonomic for tests."""

    teacher_model: str = ""
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
    # resolve the managed alias again at the worker boundary because jobspec.from_dict is tolerant.
    # the provider model id is derived only after this closed-catalog check succeeds.
    try:
        teacher = resolve_teacher(opt("teacher_model", ""))
    except ValueError as e:
        raise RuntimeError(f"opd: {e}") from e
    return OpdKnobs(
        teacher_model=teacher.model_id,
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
        # CLOSED-BLOCK hybrid recovery, checked BEFORE the think_mid early-return:
        # enable_thinking=False force-CLOSES the block (base's unique middle is a closing tag
        # "</think>...") while the shared prefix already opened "<think>", which the student
        # pre-fills. Recover the OPEN-block opener from the think render so the teacher conditions
        # on the same open block instead of base's closed one. This MUST run before `if think_mid`
        # because a base that closes the block right after the opener leaves a non-empty WHITESPACE
        # remainder in think_mid (base "<think></think>" vs think "<think>\n" -> think_mid "\n"),
        # which the early-return would otherwise hand back in place of the real "<think>\n" opener.
        # The base "<think>\n\n</think>" / think "<think>\n" shape (think_mid EMPTY) is the SAME
        # recovery. lstrip absorbs intra-block whitespace before the closing tag so detection still
        # fires; we return think[cut:] (the thinking-side opener), so the strip only affects
        # DETECTION, not the returned opener. If the opener isn't in the shared prefix, fall
        # through: return the think_mid delta ("" only when the model opens <think> inside the
        # completion).
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


def run_opd():
    """Run OPD. verl is the only backend; this module keeps the knob and prompt helpers it shares."""
    from flash.engine.worker.opd_train import run_opd_train

    run_opd_train()
