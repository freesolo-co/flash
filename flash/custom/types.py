"""Explicit next-token alignment and bounded custom-training wire inputs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

MAX_BODY_BYTES = 4 * 1024 * 1024
CUSTOM_TRAINING_CAPABILITY = "custom_training_v1"
# Target columns per position and target cells per request; a teacher top-20 fits both.
MAX_TARGETS_PER_POSITION = 64
MAX_TARGET_ENTRIES = 100_000


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

    target_tokens optionally replaces the implicit next-token targets with a rectangular
    [len(tokens)-1, K] matrix: row i lists K tokens scored at the position that predicts
    tokens[i+1], and weights then has the same [len(tokens)-1, K] shape. Logprobs come back
    with the shape of the targets. Weighting a teacher's top-k tokens by its probabilities
    gives a soft-target (forward KL) loss; give unused cells weight zero.

    No implicit averaging: the weighted loss is a sum across all supplied tokens and rows.
    Set prompt-target weights to zero, and divide by your chosen denominator yourself.
    """

    tokens: list[int]
    weights: list[float] | list[list[float]] | None = None
    target_tokens: list[list[int]] | None = None

    def to_dict(self):
        result = {"tokens": list(self.tokens)}
        if self.weights is not None:
            result["weights"] = [list(w) if isinstance(w, list) else w for w in self.weights]
        if self.target_tokens is not None:
            result["target_tokens"] = [list(ids) for ids in self.target_tokens]
        return result


def _matrix(value, name, rows, convert):
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{name} must have len(tokens)-1 rows")
    width = len(value[0]) if value and isinstance(value[0], list) else -1
    if not 1 <= width <= MAX_TARGETS_PER_POSITION or any(
        not isinstance(row, list) or len(row) != width for row in value
    ):
        raise ValueError(
            f"{name} must be a rectangular matrix with 1-{MAX_TARGETS_PER_POSITION} columns"
        )
    return [[convert(item) for item in row] for row in value]


def parse_batch(raw, *, vocab_size, max_tokens, max_batch, weighted=False):
    if not isinstance(raw, list) or not 1 <= len(raw) <= max_batch:
        raise ValueError(f"data must contain 1-{max_batch} rows")
    rows, entries = [], 0
    for row in raw:
        if not isinstance(row, dict) or set(row) - {"tokens", "weights", "target_tokens"}:
            raise ValueError(
                "each datum must contain tokens and optional weights and target_tokens"
            )
        tokens = row.get("tokens")
        if not isinstance(tokens, list) or not 2 <= len(tokens) <= max_tokens:
            raise ValueError(f"each datum needs 2-{max_tokens} tokens")
        tokens = [integer(t, "token", 0, vocab_size - 1) for t in tokens]
        positions = len(tokens) - 1
        weights = row.get("weights")
        if weighted and weights is None:
            raise ValueError("forward_backward requires explicit weights for every target token")
        target_tokens = row.get("target_tokens")
        if target_tokens is not None:
            target_tokens = _matrix(
                target_tokens,
                "target_tokens",
                positions,
                lambda t: integer(t, "target token", 0, vocab_size - 1),
            )
            entries += positions * len(target_tokens[0])
            if entries > MAX_TARGET_ENTRIES:
                raise ValueError(
                    f"target_tokens exceed {MAX_TARGET_ENTRIES} entries; split the batch"
                )
            if weights is not None:
                weights = _matrix(weights, "weights", positions, lambda w: number(w, "weight"))
                if len(weights[0]) != len(target_tokens[0]):
                    raise ValueError("weights must have the shape of target_tokens")
        elif weights is not None:
            if not isinstance(weights, list) or len(weights) != positions:
                raise ValueError("weights must have len(tokens)-1 entries")
            weights = [number(w, "weight") for w in weights]
        rows.append(Datum(tokens, weights, target_tokens))
    return rows
