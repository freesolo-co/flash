"""OPD groupwise reverse-KL (gkd) loss primitives + terminal-EOS reinforcement / runaway control.

Extracted from ``opd`` (which re-imports these and keeps the ``_resolve_samples_batched`` driver
that calls them) to isolate the differentiable loss math. All heavy imports (torch) stay inside
functions, so importing this module is CPU/offline-safe.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass


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
    groups: _PreparedGkdGroups
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
    teacher_logsum_t = torch.tensor(
        prepared.teacher_logsums, dtype=sp_t.dtype, device=sp_t.device
    )
    coeffs = kl_coef * (
        student_group_logsum - teacher_logsum_t
    ) / group_lengths_t.to(dtype=sp_t.dtype)
    coeff_vec = coeffs.index_select(0, group_ids_t)
    sp_sel = sp_t.index_select(0, flat_idx_t)
    return (coeff_vec * sp_sel).mean()


def _forward_logits(model, input_ids, attention_mask=None):
    kwargs = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    try:
        return model(input_ids, **kwargs).logits
    except TypeError:
        if attention_mask is not None:
            return model(input_ids).logits
        raise


def _bump_model_counter(model, name: str, inc: int = 1) -> None:
    with contextlib.suppress(Exception):
        setattr(model, name, int(getattr(model, name, 0) or 0) + int(inc))


def _primary_eos_id(tok, eos_ids: frozenset) -> int | None:
    """The single eos id to behaviour-clone in ``_eos_reinforce_term``'s stripped-eos case: the
    tokenizer's ``eos_token_id`` (the chat terminator — ``<|im_end|>`` for Qwen chat models), falling
    back to any member of the model's generation-halting set. ``None`` when no eos is defined (a bare
    test stand-in), which disables reinforcement."""
    tid = getattr(tok, "eos_token_id", None)
    if isinstance(tid, int) and not isinstance(tid, bool):
        return tid
    if isinstance(tid, (list, tuple)) and tid:
        for x in tid:
            if isinstance(x, int) and not isinstance(x, bool):
                return x
    return min(eos_ids) if eos_ids else None


def _eos_reinforce_term(
    sample_logits, prompt_len: int, student_ids, eos_ids, eos_primary_id, stop_sequences, eos_coef
):
    """Positive supervision for the terminal stop token the cross-tokenizer alignment drops.

    OPD's reverse-KL is computed over shared DECODED-TEXT spans (``groupwise_alignment``); a special/eos
    token is zero-width (decodes to nothing) so it lands in no alignment group and receives NO gradient
    — the student is never taught to STOP. Distilling only the content tokens toward a strong, verbose
    teacher (GLM-5.2) then erodes the student's termination and rollouts run away to the length cap. This
    adds a plain cross-entropy on log P(eos) at the position the rollout naturally terminated. It has a
    BOUNDED gradient (softmax - onehot, ‖·‖ ≤ √2, unlike the reverse-KL surrogate whose magnitude grows
    as P(eos)->0) so it never dominates grad-clipping, and it SELF-LIMITS (vanishes as P(eos)->1), so a
    student that already stops cleanly — the models that currently train fine — is unaffected.

    Two shapes, because vLLM may or may not keep the terminal eos in ``token_ids``:
      * eos KEPT (``student_ids`` ends in an eos id) — reinforce that token's own predictive row.
      * eos STRIPPED (content-only ids) — reinforce the FIRST post-content row toward the primary eos.
        The forward covers ``prompt+content``, so that row (the last real position's next-token
        prediction) already exists; no extra token is appended. Skipped when ``stop_sequences`` are
        configured (the rollout terminates on the delimiter — ordinary text the reverse-KL already
        trains — not on eos).

    Returns ``(term, logp)`` — the differentiable scalar loss addend (``-eos_coef * log P(eos)``) plus
    the detached log-prob for logging — or ``None`` when there is no eos to reinforce.
    """
    import torch

    if eos_coef <= 0:
        return None
    comp_len = len(student_ids)
    if comp_len == 0:
        return None
    last_id = int(student_ids[-1])
    if eos_ids and last_id in eos_ids:
        # rows[-1]: the row that predicted the sampled terminal eos.
        row_index = prompt_len - 1 + comp_len - 1
        target = last_id
    elif not stop_sequences and eos_primary_id is not None:
        # First post-content position: log P(eos | prompt + content).
        row_index = prompt_len - 1 + comp_len
        target = int(eos_primary_id)
    else:
        return None
    if row_index < 0 or row_index >= sample_logits.shape[0]:
        return None
    logits_row = sample_logits[row_index].float()
    logp = logits_row[target] - torch.logsumexp(logits_row, dim=-1)
    return (-float(eos_coef)) * logp, float(logp.detach())


# Terminal-EOS reinforcement is a PROPORTIONAL CONTROLLER, not a constant push. #482 added the EOS
# behaviour-cloning term on every distilled (cleanly-terminated) rollout unconditionally; because it
# raises the SHARED eos output logit — which governs P(eos) at EVERY position, not just the terminal
# one — and the terminal-row "self-limit" (vanishing gradient as P(eos)->1) never engages while P(eos)
# plateaus below 1, it ratchets the eos logit up over a long run until the student emits eos FIRST:
# empty completions -> finish_reason='stop' -> empty serve (the very bug #458 fixed, reintroduced by
# #482's mechanism). Seen at 500-ex scale: 342 reinforced samples "correcting" only 5 truncated
# rollouts, then an all-empty collapse by step ~55/63. Fix: scale the coef by how much the student is
# CURRENTLY failing to terminate (an EMA of the per-rollout truncation rate), so a run that stops
# reliably gets ~zero EOS push and cannot ratchet, while genuine runaway still drives full reinforcement.
_EOS_RUNAWAY_LO = 0.02  # truncation rate <= this: student stops fine -> NO reinforcement (no ratchet)
_EOS_RUNAWAY_HI = 0.15  # truncation rate >= this: full opd_eos_loss_coef (clear runaway to correct)
_EOS_TRUNC_EMA_DECAY = 0.97  # per-rollout EMA (~30-rollout window): responsive yet smoothed


def _runaway_eos_scale(runaway_rate: float) -> float:
    """Fraction of ``opd_eos_loss_coef`` to apply given the CURRENT runaway rate (EMA of per-rollout
    truncation). 0 while the student terminates reliably (rate <= _EOS_RUNAWAY_LO) so the terminal-EOS
    reinforcement cannot ratchet the shared eos logit into an empty-collapse; ramps linearly to full by
    _EOS_RUNAWAY_HI. Reinforce EXACTLY in proportion to the runaway the term exists to correct."""
    if runaway_rate <= _EOS_RUNAWAY_LO:
        return 0.0
    if runaway_rate >= _EOS_RUNAWAY_HI:
        return 1.0
    return (runaway_rate - _EOS_RUNAWAY_LO) / (_EOS_RUNAWAY_HI - _EOS_RUNAWAY_LO)
