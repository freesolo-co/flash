"""Optional local losses for forward_backward_custom; they run on the client, never the GPU.

Option scoring reads, for each datum, the logprobs of K option tokens (for example the letters
A-D) at the position right after the prompt:

    Datum(prompt, target_tokens=[option_ids], target_positions=[len(prompt) - 1])

Each datum then scores K target entries, and forward_backward_custom hands loss_fn one [1, K]
row per datum. The loss renormalizes those rows over the options (softmax over the options
alone), matching a letter-logit readout.
"""

from __future__ import annotations

import math
import numbers


def option_ce_brier_loss(option_logprobs, targets, *, brier_weight=1.0):
    """Mean over data of -sum(t * log q) + brier_weight * sum((q - t) ** 2).

    option_logprobs holds one tensor per datum with K option logprobs ([K] or [1, K]); q is
    their softmax over the K options. targets holds, per datum, the correct option index
    (one-hot) or K non-negative probabilities that sum to 1 (soft targets). Returns a scalar
    tensor and a metrics dict with the mean cross-entropy, Brier score and argmax accuracy.
    """
    import torch

    if isinstance(brier_weight, bool) or not isinstance(brier_weight, int | float):
        raise ValueError("brier_weight must be a finite non-negative number")
    if not math.isfinite(brier_weight) or brier_weight < 0:
        raise ValueError("brier_weight must be a finite non-negative number")
    if len(option_logprobs) != len(targets) or not option_logprobs:
        raise ValueError("option_logprobs and targets need one nonempty entry per datum")
    ce_terms, brier_terms, correct = [], [], 0
    for row, target in zip(option_logprobs, targets, strict=True):
        if row.ndim == 2 and row.shape[0] == 1:
            row = row[0]
        if row.ndim != 1 or row.shape[0] < 2:
            raise ValueError("each datum needs one row of at least 2 option logprobs")
        expected = _target_distribution(target, row.shape[0], row)
        log_q = torch.log_softmax(row, dim=-1)
        q = log_q.exp()
        ce_terms.append(-(expected * log_q).sum())
        brier_terms.append(((q - expected) ** 2).sum())
        correct += int(int(q.argmax()) == int(expected.argmax()))
    ce = torch.stack(ce_terms).mean()
    brier = torch.stack(brier_terms).mean()
    metrics = {
        "option_ce": ce.item(),
        "option_brier": brier.item(),
        "option_accuracy": correct / len(targets),
    }
    return ce + brier_weight * brier, metrics


def option_ce_brier_loss_fn(targets, *, brier_weight=1.0):
    """A forward_backward_custom loss_fn over per-datum option logprobs and fixed targets."""

    def loss_fn(_data, logprobs):
        return option_ce_brier_loss(logprobs, targets, brier_weight=brier_weight)

    return loss_fn


def _target_distribution(target, options, like):
    import torch

    if isinstance(target, bool):
        raise ValueError("an option target must be an index or a probability vector")
    if isinstance(target, numbers.Integral):
        if not 0 <= target < options:
            raise ValueError(f"option target index {int(target)} is outside [0, {options - 1}]")
        expected = torch.zeros(options, dtype=like.dtype, device=like.device)
        expected[target] = 1.0
        return expected
    expected = torch.as_tensor(target, dtype=like.dtype, device=like.device).detach()
    if (
        expected.shape != (options,)
        or not torch.isfinite(expected).all()
        or (expected < 0).any()
        or abs(float(expected.sum()) - 1.0) > 1e-4
    ):
        raise ValueError(f"soft option targets must be {options} non-negative probabilities")
    return expected
