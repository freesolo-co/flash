"""Cross-tokenizer alignment for on-policy distillation (teacher GLM -> student Qwen).

The teacher and student have different tokenizers and vocabularies, so the teacher's per-token
distribution cannot be compared to the student's directly. These pure helpers turn the teacher's
character-anchored per-token top-k (from an echo-scored completion) into a per-student-token target
the distillation loss can consume, using the two strategies that need a position alignment:

- ``align``: project the teacher's top-k next-token candidates onto the STUDENT vocabulary (by
  student-tokenizing each candidate surface string and taking its first token id) to form a sparse
  target distribution — a cross-tokenizer top-k forward-KL target.
- ``uld``: Universal Logit Distillation — keep only the SORTED teacher top-k probabilities (no token
  identity), to be compared against the student's own sorted top-k (a truncated Wasserstein/L1).

Both align teacher<->student positions by CHARACTER OFFSET into the completion string: a student
token gets a target only where a teacher token starts at the same character (i.e. the two tokenizers
agree on that boundary); other positions are masked (``None``). No torch/network imports here, so the
module is import-safe and unit-testable on a CPU box. The ``seqkd`` strategy needs no alignment (it is
plain completion-only cross-entropy on teacher-generated text) and lives in the worker.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class TeacherToken:
    """One teacher token over the completion string. ``start``/``end`` are character offsets rebased
    to the completion (0 = first completion char). ``top`` is ``(surface, logprob)`` alternatives,
    including the realized token."""

    text: str
    logprob: float
    top: tuple[tuple[str, float], ...]
    start: int
    end: int


@dataclass(frozen=True)
class StudentToken:
    """One student completion token: its vocab id and character span in the completion string."""

    token_id: int
    start: int
    end: int


def _softmax_from_logprobs(logprobs: list[float], temperature: float) -> list[float]:
    """Temperature-softmax over natural-log probabilities. Numerically stable; empty -> []."""
    if not logprobs:
        return []
    t = temperature if temperature and temperature > 0 else 1.0
    scaled = [lp / t for lp in logprobs]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def _teacher_by_start(teacher_toks: list[TeacherToken]) -> dict[int, TeacherToken]:
    """Map completion char-offset -> the FIRST teacher token that starts there."""
    by_start: dict[int, TeacherToken] = {}
    for tt in teacher_toks:
        by_start.setdefault(tt.start, tt)
    return by_start


def align_targets(
    student_toks: list[StudentToken],
    teacher_toks: list[TeacherToken],
    first_token_id: Callable[[str], int | None],
    *,
    kd_temperature: float = 1.0,
) -> list[dict[int, float] | None]:
    """Per student completion token -> a sparse target distribution over the STUDENT vocab, or None.

    For each student token whose start offset coincides with a teacher token boundary, project that
    teacher token's top-k candidates onto the student vocabulary (``first_token_id(surface)``) and
    renormalise into a distribution. Positions with no coincident teacher boundary (the tokenizers
    disagree there) or no mappable candidate are masked (None) and excluded from the loss.
    """
    by_start = _teacher_by_start(teacher_toks)
    out: list[dict[int, float] | None] = []
    for st in student_toks:
        tt = by_start.get(st.start)
        if tt is None or not tt.top:
            out.append(None)
            continue
        probs = _softmax_from_logprobs([lp for _, lp in tt.top], kd_temperature)
        acc: dict[int, float] = {}
        for (surface, _), p in zip(tt.top, probs, strict=True):
            vid = first_token_id(surface)
            if vid is None:
                continue
            acc[vid] = acc.get(vid, 0.0) + p
        if not acc:
            out.append(None)
            continue
        z = sum(acc.values()) or 1.0
        out.append({vid: p / z for vid, p in acc.items()})
    return out


def uld_targets(
    student_toks: list[StudentToken],
    teacher_toks: list[TeacherToken],
    *,
    kd_temperature: float = 1.0,
) -> list[list[float] | None]:
    """Per student completion token -> the teacher's SORTED (descending) top-k probability vector,
    normalised over the top-k, or None where no teacher boundary coincides. Vocabulary-agnostic: the
    loss compares this against the student's own sorted top-k, so no token identity is needed."""
    by_start = _teacher_by_start(teacher_toks)
    out: list[list[float] | None] = []
    for st in student_toks:
        tt = by_start.get(st.start)
        if tt is None or not tt.top:
            out.append(None)
            continue
        probs = _softmax_from_logprobs([lp for _, lp in tt.top], kd_temperature)
        out.append(sorted(probs, reverse=True))
    return out


def groupwise_alignment(
    student_toks: list[StudentToken], teacher_toks: list[TeacherToken]
) -> list[tuple[list[int], float]]:
    """Align student & teacher tokens into matching decoded-text spans, for groupwise reverse-KL.

    This is the collinear-ai *spider* / Tinker-cookbook realignment (``_build_alignment_groups`` +
    ``_compute_groupwise_reverse_kl``): the coarsest common refinement of the two tokenizations —
    a group boundary is any character position that starts a token in BOTH tokenizers. Between
    consecutive shared boundaries, the student tokens and teacher tokens covering that span form one
    group (both sides possibly a different number of tokens). Unlike ``align``/``uld``, this covers
    EVERY student token (no masking): where the tokenizers disagree locally, the span just grows
    until they next agree. Because both offset sets index the same completion string, matching
    character spans is equivalent to matching decoded text.

    Returns ``[(student_indices, teacher_logprob_sum), ...]`` — for each group the student token
    indices in it and the summed teacher logprob over that span (``log P_teacher(span)``). The loss
    (``opd.gkd_loss``) pairs this with the differentiable ``log P_student(span)`` for reverse-KL.
    """
    if not student_toks or not teacher_toks:
        return []
    s_starts = {st.start for st in student_toks}
    t_starts = {tt.start for tt in teacher_toks}
    begin = min(student_toks[0].start, teacher_toks[0].start)
    # Group boundaries: char positions that begin a token in both tokenizers (always include the
    # start, so the first group is anchored even if the two first-token offsets differ).
    boundaries = sorted((s_starts & t_starts) | {begin})
    end = max(max(st.end for st in student_toks), max(tt.end for tt in teacher_toks))
    edges = [*boundaries, end]
    groups: list[tuple[list[int], float]] = []
    for k in range(len(boundaries)):
        lo, hi = edges[k], edges[k + 1]
        s_idx = [i for i, st in enumerate(student_toks) if lo <= st.start < hi]
        t_lp = [tt.logprob for tt in teacher_toks if lo <= tt.start < hi]
        if s_idx and t_lp:  # both sides must be non-empty for a valid group
            groups.append((s_idx, float(sum(t_lp))))
    return groups


def groupwise_coverage(groups: list[tuple[list[int], float]], n_student_tokens: int) -> float:
    """Fraction of student completion tokens that landed in a valid (both-sided) alignment group."""
    if not n_student_tokens:
        return 0.0
    covered = sum(len(s_idx) for s_idx, _ in groups)
    return covered / n_student_tokens


def coverage(targets: list[dict[int, float] | None] | list[list[float] | None]) -> float:
    """Fraction of student completion positions that received a (non-masked) teacher target."""
    if not targets:
        return 0.0
    return sum(1 for t in targets if t is not None) / len(targets)
