"""On-policy distillation entry point (algorithm="opd") and the knob/prompt helpers it shares.

Training lives in `opd_train`; this module keeps CPU-safe knobs and prompt helpers. Cross-tokenizer
scoring compares shared decoded-text spans in `tokenizer_align`.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import flash.engine.worker.runtime.state as _worker_state
from flash.engine.plan.recipe import RECIPE, resolve_teacher
from flash.engine.plan.vram import opd_completion_len


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
    t = _worker_state.JOB_SPEC.train if _worker_state.JOB_SPEC else None

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
        max_completion=opd_completion_len(opt("max_completion_tokens", 0), _worker_state.THINKING),
        # reads prompts_per_step, NOT batch_size: opd rejects batch_size at parse time, so an opd
        # spec carries the optimizer batch only under this key. reading the old name found None on
        # every run and trained the recipe default no matter what the user authored.
        prompts_per_step=int(opt("prompts_per_step", 0) or d.prompts_per_step),
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
    """return the exact terminal Qwen reasoning opener, or fail closed."""
    if not _worker_state.THINKING:
        return ""
    expected = "<think>\n"
    probe = [{"role": "user", "content": ""}]
    with contextlib.suppress(Exception):
        from flash.content.thinking import messages_for_chat_template

        probe = messages_for_chat_template(probe)
        base = tok.apply_chat_template(
            probe,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            preserve_thinking=False,
        )
        think = tok.apply_chat_template(
            probe,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            preserve_thinking=False,
        )
        if think == base or not think.endswith(expected):
            return ""

        think_prefix = think[: -len(expected)]
        base_prefix = base
        trimmed_base = base.rstrip()
        if trimmed_base.endswith("</think>"):
            close_start = len(trimmed_base) - len("</think>")
            open_start = trimmed_base.rfind("<think>", 0, close_start)
            if open_start < 0 or trimmed_base[open_start + len("<think>") : close_start].strip():
                return ""
            base_prefix = trimmed_base[:open_start]

        suffix_len = 0
        for left, right in zip(reversed(base_prefix), reversed(think_prefix), strict=False):
            if left != right:
                break
            suffix_len += 1
        if suffix_len and base_prefix[-suffix_len:].strip():
            return expected
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
    from flash.engine.worker.train.entry.opd_train import run_opd_train

    run_opd_train()
