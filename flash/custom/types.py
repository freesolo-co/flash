"""Explicit next-token alignment and bounded custom-training wire inputs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

MAX_BODY_BYTES = 4 * 1024 * 1024
CUSTOM_TRAINING_CAPABILITY = "custom_training_v1"
# Candidate targets per position and per request; a teacher top-20 fits within both.
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

    target_tokens[i] optionally lists further candidate tokens scored at the same position,
    and target_weights[i] weights their negative logprobs. Weighting a teacher's top-k tokens
    by its probabilities gives a soft-target (forward KL) loss. An empty list scores nothing.

    No implicit averaging: the weighted loss is a sum across all supplied tokens and rows.
    Set prompt-target weights to zero, and divide by your chosen denominator yourself.
    """

    tokens: list[int]
    weights: list[float] | None = None
    target_tokens: list[list[int]] | None = None
    target_weights: list[list[float]] | None = None

    def to_dict(self):
        result = {"tokens": list(self.tokens)}
        if self.weights is not None:
            result["weights"] = list(self.weights)
        if self.target_tokens is not None:
            result["target_tokens"] = [list(ids) for ids in self.target_tokens]
        if self.target_weights is not None:
            result["target_weights"] = [list(values) for values in self.target_weights]
        return result


def _targets(row, positions, vocab_size):
    tokens, weights = row.get("target_tokens"), row.get("target_weights")
    if tokens is None:
        if weights is not None:
            raise ValueError("target_weights require target_tokens")
        return None, None
    if not isinstance(tokens, list) or len(tokens) != positions:
        raise ValueError("target_tokens must have len(tokens)-1 entries")
    if weights is not None and (not isinstance(weights, list) or len(weights) != positions):
        raise ValueError("target_weights must have len(tokens)-1 entries")
    parsed_tokens, parsed_weights = [], []
    for index, ids in enumerate(tokens):
        if not isinstance(ids, list) or len(ids) > MAX_TARGETS_PER_POSITION:
            raise ValueError(
                f"each position takes at most {MAX_TARGETS_PER_POSITION} target tokens"
            )
        ids = [integer(t, "target token", 0, vocab_size - 1) for t in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("target tokens must be distinct at each position")
        parsed_tokens.append(ids)
        if weights is not None:
            values = weights[index]
            if not isinstance(values, list) or len(values) != len(ids):
                raise ValueError("target_weights must match target_tokens at every position")
            parsed_weights.append([number(w, "target weight") for w in values])
    return parsed_tokens, parsed_weights if weights is not None else None


def parse_batch(raw, *, vocab_size, max_tokens, max_batch, weighted=False):
    if not isinstance(raw, list) or not 1 <= len(raw) <= max_batch:
        raise ValueError(f"data must contain 1-{max_batch} rows")
    rows, entries = [], 0
    for row in raw:
        if not isinstance(row, dict) or set(row) - {
            "tokens",
            "weights",
            "target_tokens",
            "target_weights",
        }:
            raise ValueError(
                "each datum must contain tokens and optional weights, target_tokens "
                "and target_weights"
            )
        tokens = row.get("tokens")
        if not isinstance(tokens, list) or not 2 <= len(tokens) <= max_tokens:
            raise ValueError(f"each datum needs 2-{max_tokens} tokens")
        tokens = [integer(t, "token", 0, vocab_size - 1) for t in tokens]
        weights = row.get("weights")
        target_tokens, target_weights = _targets(row, len(tokens) - 1, vocab_size)
        if target_tokens is not None:
            entries += sum(len(ids) for ids in target_tokens)
            if entries > MAX_TARGET_ENTRIES:
                raise ValueError(
                    f"target tokens exceed {MAX_TARGET_ENTRIES} entries; split the batch"
                )
        if weighted and weights is None and target_weights is None:
            raise ValueError(
                "forward_backward requires explicit weights or target_weights for every row"
            )
        if weighted and target_tokens is not None and target_weights is None:
            raise ValueError("forward_backward requires target_weights for target_tokens")
        if weights is not None:
            if not isinstance(weights, list) or len(weights) != len(tokens) - 1:
                raise ValueError("weights must have len(tokens)-1 entries")
            weights = [number(w, "weight") for w in weights]
        rows.append(Datum(tokens, weights, target_tokens, target_weights))
    return rows
