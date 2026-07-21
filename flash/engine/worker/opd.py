"""Pure OPD configuration and groupwise reverse-KL helpers.

The out-of-process verl orchestration lives in ``opd_verl``. This module intentionally retains only
CPU-safe semantics shared by the parent bridge and parity tests.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

from flash.engine.recipe import RECIPE, resolve_teacher
from flash.engine.vram import opd_completion_len
from flash.engine.worker._pkg import W as _w
from flash.engine.worker.opd_gkd import (
    _generation_eos_ids,
    _rollout_terminated,
    _teacher_prompt_text,
    _trim_trailing_stop,
    student_tokens_with_offsets,
)

__all__ = [
    "_generation_eos_ids",
    "_rollout_terminated",
    "_teacher_prompt_text",
    "_trim_trailing_stop",
    "student_tokens_with_offsets",
]


@dataclass(frozen=True)
class OpdKnobs:
    teacher_model: str = ""
    teacher_base_url: str = ""
    epochs: int = RECIPE.opd.num_epochs
    learning_rate: float = 0.0
    temperature: float = 0.0
    top_p: float = 1.0
    max_completion: int = 0
    prompts_per_step: int = 0
    group_size: int = 0
    kl_coef: float = 1.0
    save_every: int = 0
    max_steps: int = 0
    save_at_steps: tuple[int, ...] = ()
    max_length: int = 0
    stop_sequences: tuple = ()
    structured_outputs: str = ""


def _resolve_opd_knobs() -> OpdKnobs:
    """Resolve the managed OPD surface from the current job spec."""
    defaults = RECIPE.opd
    train = _w.JOB_SPEC.train if _w.JOB_SPEC else None

    def opt(name, default):
        value = getattr(train, name, None) if train else None
        return value if value is not None else default

    raw_kl = opt("kl_penalty_coef", None)
    kl_coef = float(raw_kl if raw_kl is not None else defaults.kl_coef)
    if kl_coef <= 0.0:
        raise RuntimeError(
            "opd: [train] kl_penalty_coef must be > 0 because it directly scales the "
            "groupwise reverse-kl objective"
        )
    try:
        teacher = resolve_teacher(opt("teacher_model", ""))
    except ValueError as error:
        raise RuntimeError(f"opd: {error}") from error
    return OpdKnobs(
        teacher_model=teacher.model_id,
        teacher_base_url=defaults.teacher_base_url,
        epochs=int(train.epochs) if train and train.epochs is not None else defaults.num_epochs,
        learning_rate=float(opt("learning_rate", 0) or defaults.learning_rate),
        temperature=float(
            opt("temperature", None)
            if train and train.temperature is not None
            else defaults.sampling_temperature
        ),
        top_p=defaults.sampling_top_p,
        max_completion=opd_completion_len(opt("max_completion_tokens", 0), _w.THINKING),
        prompts_per_step=int(opt("batch_size", 0) or defaults.prompts_per_step),
        group_size=int(opt("group_size", 0) or defaults.group_size),
        kl_coef=kl_coef,
        save_every=int(opt("save_every", 0) or 20),
        max_steps=int(opt("max_steps", 0) or 0),
        save_at_steps=tuple(getattr(train, "save_at_steps", ()) or ()),
        max_length=int(opt("max_context_tokens", 0) or 0),
        stop_sequences=tuple(getattr(train, "stop_sequences", ()) or ()),
        structured_outputs=str(getattr(train, "structured_outputs", "") or ""),
    )


def _thinking_prefill_text(tokenizer) -> str:
    """Return the reasoning opener inserted by the student's thinking chat template."""
    if not _w.THINKING:
        return ""
    probe = [{"role": "user", "content": ""}]
    with contextlib.suppress(Exception):
        base = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        think = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        if think == base:
            return ""
        prefix = 0
        limit = min(len(base), len(think))
        while prefix < limit and base[prefix] == think[prefix]:
            prefix += 1
        suffix = 0
        while (
            suffix < len(base) - prefix
            and suffix < len(think) - prefix
            and base[-1 - suffix] == think[-1 - suffix]
        ):
            suffix += 1
        think_mid = think[prefix : len(think) - suffix]
        base_mid = base[prefix : len(base) - suffix]
        base_mid_tag = base_mid.lstrip()
        if base_mid_tag.startswith("</") and ">" in base_mid_tag:
            open_tag = "<" + base_mid_tag[2 : base_mid_tag.index(">") + 1]
            cut = think.rfind(open_tag, 0, prefix)
            if cut != -1:
                return think[cut:]
        if think_mid:
            return think_mid
    return ""


@dataclass(frozen=True)
class _PreparedGkdGroups:
    token_indices: tuple[int, ...]
    group_lengths: tuple[int, ...]
    teacher_logsums: tuple[float, ...]


def _drop_fully_forced_groups(groups, forced):
    """Remove alignment groups for which structured decoding forced every student token."""
    if not forced:
        return groups
    return [
        (student_indices, teacher_logsum)
        for student_indices, teacher_logsum in groups
        if not (
            student_indices
            and all(index < len(forced) and forced[index] for index in student_indices)
        )
    ]


def _prepare_gkd_groups(groups) -> _PreparedGkdGroups | None:
    token_indices: list[int] = []
    group_lengths: list[int] = []
    teacher_logsums: list[float] = []
    for student_indices, teacher_logsum in groups:
        if not student_indices:
            continue
        token_indices.extend(int(index) for index in student_indices)
        group_lengths.append(len(student_indices))
        teacher_logsums.append(float(teacher_logsum))
    if not token_indices:
        return None
    return _PreparedGkdGroups(
        token_indices=tuple(token_indices),
        group_lengths=tuple(group_lengths),
        teacher_logsums=tuple(teacher_logsums),
    )


def _gkd_loss_from_logps(student_logps, groups, kl_coef=1.0):
    """Compute flash's realized-token groupwise reverse-KL surrogate."""
    import torch

    if student_logps is None or not groups:
        return None
    prepared = groups if isinstance(groups, _PreparedGkdGroups) else _prepare_gkd_groups(groups)
    if prepared is None:
        return None
    detached = student_logps.detach()
    flat_indices = torch.tensor(prepared.token_indices, device=student_logps.device)
    group_lengths = torch.tensor(prepared.group_lengths, device=student_logps.device)
    group_ids = torch.repeat_interleave(
        torch.arange(len(prepared.group_lengths), device=student_logps.device), group_lengths
    )
    student_group_logsums = detached.new_zeros(len(prepared.group_lengths))
    student_group_logsums.index_add_(
        0, group_ids, detached.index_select(0, flat_indices)
    )
    teacher_logsums = torch.tensor(
        prepared.teacher_logsums, dtype=student_logps.dtype, device=student_logps.device
    )
    coefficients = (
        float(kl_coef)
        * (student_group_logsums - teacher_logsums)
        / group_lengths.to(dtype=student_logps.dtype)
    )
    selected_logps = student_logps.index_select(0, flat_indices)
    return (coefficients.index_select(0, group_ids) * selected_logps).mean()


def run_opd():
    """Compatibility entrypoint that dispatches to the verl implementation."""
    from flash.engine.worker.opd_verl import run_opd_verl

    return run_opd_verl(_w.JOB_SPEC)
