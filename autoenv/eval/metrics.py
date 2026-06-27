"""Independent, paper-faithful metric registry.

These metrics score ``(gold_output, model_response)`` pairs directly — they are the
**headline** number, deliberately separate from the env reward the agent optimized, so a
reward-hacked run can't inflate the score (the run pipeline cross-checks the two and flags a
large divergence). Every metric returns a per-example score in ``[0, 1]``; a dataset metric
is the mean (``aggregate``). All pure and dependency-free so they run in CI.

Add a metric by registering a ``(gold, response) -> float`` callable in ``METRICS``; the
manifest's ``[metric] name`` must be a key here (the gate's metric check enforces it).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
# First signed integer/decimal in a string (the conventional final-answer extraction for math).
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — the standard QA normalisation."""
    text = _PUNCT.sub(" ", (text or "").lower())
    return _WS.sub(" ", text).strip()


def exact_match(gold: str, response: str) -> float:
    """1.0 iff the response equals the gold answer exactly (raw strings, trimmed)."""
    return 1.0 if (gold or "").strip() == (response or "").strip() else 0.0


def normalized_match(gold: str, response: str) -> float:
    """1.0 iff response == gold after lowercasing/punctuation/whitespace normalisation."""
    return 1.0 if _normalize(gold) == _normalize(response) else 0.0


def contains(gold: str, response: str) -> float:
    """1.0 iff the normalized gold answer appears anywhere in the normalized response.

    Mirrors the starter env's ``exact_match_reward`` substring check — lenient, useful when
    the model wraps the answer in prose.
    """
    g = _normalize(gold)
    return 1.0 if g and g in _normalize(response) else 0.0


def numeric_match(gold: str, response: str) -> float:
    """1.0 iff the LAST number in the response equals the gold's number (math final-answer).

    The convention for GSM8K-style tasks: the answer is the final number emitted. Returns 0.0
    when either side has no number.
    """
    g = _NUMBER.findall(gold or "")
    r = _NUMBER.findall(response or "")
    if not g or not r:
        return 0.0
    try:
        return 1.0 if float(g[-1]) == float(r[-1]) else 0.0
    except ValueError:
        return 0.0


def token_f1(gold: str, response: str) -> float:
    """Token-overlap F1 over normalized whitespace tokens (SQuAD-style partial credit)."""
    g_tokens = _normalize(gold).split()
    r_tokens = _normalize(response).split()
    if not g_tokens or not r_tokens:
        return 1.0 if not g_tokens and not r_tokens else 0.0
    common = Counter(g_tokens) & Counter(r_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(r_tokens)
    recall = overlap / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


# Registry. ``accuracy`` is an alias for exact classification match.
METRICS: dict[str, Callable[[str, str], float]] = {
    "exact_match": exact_match,
    "accuracy": exact_match,
    "normalized_match": normalized_match,
    "contains": contains,
    "numeric_match": numeric_match,
    "token_f1": token_f1,
}


def score_one(name: str, gold: str, response: str) -> float:
    """Per-example score in [0, 1] for metric ``name``."""
    try:
        fn = METRICS[name]
    except KeyError as exc:
        raise KeyError(f"unknown metric {name!r}; known: {', '.join(sorted(METRICS))}") from exc
    return float(fn(gold, response))


def aggregate(name: str, pairs: Iterable[tuple[str, str]]) -> float:
    """Mean per-example score over ``(gold, response)`` pairs. Empty -> 0.0."""
    scores = [score_one(name, gold, resp) for gold, resp in pairs]
    return sum(scores) / len(scores) if scores else 0.0
