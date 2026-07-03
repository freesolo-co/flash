"""Cross-tokenizer alignment for on-policy distillation (teacher GLM-5.2 -> student Qwen3.5 / MiniCPM).

The teacher and student have different tokenizers and vocabularies, so they segment the SAME
completion string at different byte boundaries and their per-token distributions can't be compared
directly. ``groupwise_alignment`` bridges this by matching SHARED DECODED-TEXT SPANS: the coarsest
common refinement of the two tokenizations (a group boundary is any character offset that begins a
token in BOTH tokenizers). Between consecutive shared boundaries, the student tokens and teacher
tokens covering that span form one group. This is the collinear-ai *spider* / Tinker-cookbook
realignment (``_build_alignment_groups`` + ``_compute_groupwise_reverse_kl``), and it uses only the
REALIZED-token logprobs on each side — no top-k candidates, no vocabulary projection — so it is exact
across arbitrary tokenizer mismatch and covers every student token. No torch/network imports here, so
the module is import-safe and unit-testable on a CPU box.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TeacherToken:
    """One teacher token over the completion string. ``start``/``end`` are character offsets rebased
    to the completion (0 = first completion char). ``logprob`` is the teacher's log-probability of
    the realized token."""

    text: str
    logprob: float
    start: int
    end: int


@dataclass(frozen=True)
class StudentToken:
    """One student completion token: its vocab id and character span in the completion string."""

    token_id: int
    start: int
    end: int


def groupwise_alignment(
    student_toks: list[StudentToken], teacher_toks: list[TeacherToken]
) -> list[tuple[list[int], float]]:
    """Align student & teacher tokens into matching decoded-text spans, for groupwise reverse-KL.

    This is the collinear-ai *spider* / Tinker-cookbook realignment (``_build_alignment_groups`` +
    ``_compute_groupwise_reverse_kl``): the coarsest common refinement of the two tokenizations —
    a group boundary is any character position that starts a token in BOTH tokenizers. Between
    consecutive shared boundaries, the student tokens and teacher tokens covering that span form one
    group (both sides possibly a different number of tokens). This covers EVERY student token (no
    masking): where the tokenizers disagree locally, the span just grows until they next agree.
    Because both offset sets index the same completion string, matching character spans is equivalent
    to matching decoded text.

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
    pending: list[int] = []  # student tokens whose span had no teacher token yet
    for k in range(len(boundaries)):
        lo, hi = edges[k], edges[k + 1]
        s_idx = [i for i, st in enumerate(student_toks) if lo <= st.start < hi]
        t_lp = [tt.logprob for tt in teacher_toks if lo <= tt.start < hi]
        if t_lp:  # a teacher-bearing span closes the group, absorbing any carried student tokens
            groups.append((pending + s_idx, float(sum(t_lp))))
            pending = []
        else:  # student-only span: carry its tokens to the next teacher-bearing span (never dropped)
            pending += s_idx
    if pending and groups:  # trailing student-only tokens attach to the last group
        idx, tsum = groups[-1]
        groups[-1] = (idx + pending, tsum)
    return groups


def groupwise_coverage(groups: list[tuple[list[int], float]], n_student_tokens: int) -> float:
    """Fraction of student completion tokens that landed in a valid (both-sided) alignment group."""
    if not n_student_tokens:
        return 0.0
    covered = sum(len(s_idx) for s_idx, _ in groups)
    return covered / n_student_tokens
