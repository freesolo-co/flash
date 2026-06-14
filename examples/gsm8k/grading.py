"""Shared GSM8K grader — the SINGLE source of truth for scoring.

Used identically for the SFT target check, the RL train-time reward, and eval, so
a model is always scored the same way across phases.
"""

from __future__ import annotations

import re

# GSM8K gold answers look like: "...<reasoning>...\n#### 42"
_GSM8K_GOLD_RE = re.compile(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)")
_BOXED_RE = re.compile(r"\\boxed\{\s*(-?[0-9][0-9,]*\.?[0-9]*)\s*\}")
# Any number (last-number fallback). Allows $, commas, decimals, negatives.
_NUMBER_RE = re.compile(r"-?\$?\s*[0-9][0-9,]*\.?[0-9]*")


def _normalize_number(s: str) -> str | None:
    """Normalize a numeric string to a canonical comparable form.

    Strips $, commas, surrounding whitespace; drops trailing .0; returns None
    if it cannot be parsed as a number.
    """
    if s is None:
        return None
    s = s.strip().replace(",", "").replace("$", "").strip()
    if s in ("", "-", "."):
        return None
    try:
        val = float(s)
    except ValueError:
        return None
    # Represent integers without trailing .0 so "42" == "42.0"
    if val == int(val):
        return str(int(val))
    # Round to avoid float noise; keep up to 6 decimals
    return str(round(val, 6))


def extract_gold_answer(gold_solution: str) -> str | None:
    """Extract the canonical final answer from a GSM8K gold solution string."""
    m = _GSM8K_GOLD_RE.search(gold_solution)
    if m:
        return _normalize_number(m.group(1))
    # Fallback: last number in the gold text
    nums = _NUMBER_RE.findall(gold_solution)
    return _normalize_number(nums[-1]) if nums else None


def extract_pred_answer(completion: str) -> str | None:
    """Extract the model's final numeric answer from a generated completion.

    Priority: \\boxed{...} > #### ... > last number in text.
    """
    if completion is None:
        return None
    m = _BOXED_RE.search(completion)
    if m:
        return _normalize_number(m.group(1))
    m = _GSM8K_GOLD_RE.search(completion)
    if m:
        return _normalize_number(m.group(1))
    nums = _NUMBER_RE.findall(completion)
    return _normalize_number(nums[-1]) if nums else None


def is_correct(completion: str, gold: str) -> bool:
    """True iff the prediction's final number matches the gold final number.

    `gold` may be a raw GSM8K solution (with ####) or an already-extracted answer.
    """
    gold_ans = None
    if "####" not in gold and len(gold) <= 16:
        gold_ans = _normalize_number(gold)
    if gold_ans is None:
        gold_ans = extract_gold_answer(gold)
    pred_ans = extract_pred_answer(completion)
    if pred_ans is None or gold_ans is None:
        return False
    return pred_ans == gold_ans


def has_format(completion: str) -> bool:
    """Rule-based format check: did the model present a final answer cleanly?"""
    if completion is None:
        return False
    return bool(_BOXED_RE.search(completion) or _GSM8K_GOLD_RE.search(completion))


def reward(completion: str, gold: str, correct_w: float = 1.0, format_w: float = 0.1) -> float:
    """Verifiable RL reward shared by both arms."""
    r = 0.0
    if is_correct(completion, gold):
        r += correct_w
    if has_format(completion):
        r += format_w
    return r


def accuracy(completions: list[str], golds: list[str]) -> float:
    assert len(completions) == len(golds)
    if not completions:
        return 0.0
    n = sum(1 for c, g in zip(completions, golds, strict=True) if is_correct(c, g))
    return n / len(completions)
