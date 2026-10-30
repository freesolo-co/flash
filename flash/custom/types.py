"""Explicit next-token alignment and bounded custom-training wire inputs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


def integer(value, name, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the allowed range")
    return float(value)


def checkpoint_name(name):
    if not isinstance(name, str) or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name) is None:
        raise ValueError(
            "checkpoint name must contain 1-64 letters, digits, underscores or hyphens"
        )
    return name


@dataclass(frozen=True)
class Datum:
    """tokens[i] predicts tokens[i+1]; weights[i] weights that target's negative logprob.

    No implicit averaging: the weighted loss is a sum across all supplied tokens and rows.
    Set prompt-target weights to zero, and divide by your chosen denominator yourself.
    """

    tokens: list[int]
    weights: list[float] | None = None

    def to_dict(self):
        result = {"tokens": list(self.tokens)}
        if self.weights is not None:
            result["weights"] = list(self.weights)
        return result


def parse_batch(raw, *, vocab_size, max_tokens, max_batch, weighted=False):
    if not isinstance(raw, list) or not 1 <= len(raw) <= max_batch:
        raise ValueError(f"data must contain 1-{max_batch} rows")
    rows = []
    for row in raw:
        if not isinstance(row, dict) or set(row) - {"tokens", "weights"}:
            raise ValueError("each datum must contain tokens and optional weights")
        tokens = row.get("tokens")
        if not isinstance(tokens, list) or not 2 <= len(tokens) <= max_tokens:
            raise ValueError(f"each datum needs 2-{max_tokens} tokens")
        tokens = [integer(t, "token", 0, vocab_size - 1) for t in tokens]
        weights = row.get("weights")
        if weighted and weights is None:
            raise ValueError("forward_backward requires explicit weights for every target token")
        if weights is not None:
            if not isinstance(weights, list) or len(weights) != len(tokens) - 1:
                raise ValueError("weights must have len(tokens)-1 entries")
            weights = [number(w, "weight") for w in weights]
        rows.append(Datum(tokens, weights))
    return rows
