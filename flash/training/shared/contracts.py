"""Validated, token-only requests for a trusted shared model runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass


class AdmissionError(ValueError):
    """A request exceeds the pool or tenant's configured capacity."""


class ConflictError(ValueError):
    """An idempotency key or model version conflicts with accepted work."""


class UnavailableError(RuntimeError):
    """The pool has stopped accepting work."""


@dataclass(frozen=True)
class Principal:
    tenant: str
    owner: str

    def __post_init__(self):
        for value in (self.tenant, self.owner):
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError("invalid authenticated principal")


@dataclass(frozen=True)
class Limits:
    adapters: int = 8
    adapters_per_tenant: int = 2
    queued_requests: int = 64
    queued_requests_per_tenant: int = 16
    queued_tokens_per_tenant: int = 16384
    request_tokens: int = 4096
    context_tokens: int = 4096

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError("pool limits must be positive integers")
        if self.adapters_per_tenant > self.adapters:
            raise ValueError("tenant adapter limit exceeds pool capacity")


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in [{low}, {high}]")
    return float(value)


def fields(payload, required, optional=()):
    if not isinstance(payload, dict) or set(payload) - set(required) - set(optional):
        raise ValueError("unknown request fields")
    if set(required) - set(payload):
        raise ValueError("missing request fields")


def session_config(payload):
    fields(payload, ("algorithm", "seed", "learning_rate"))
    if payload["algorithm"] not in ("sft", "grpo", "opd"):
        raise ValueError("algorithm must be sft, grpo, or opd")
    return {
        "algorithm": payload["algorithm"],
        "seed": integer(payload["seed"], "seed", 0, 2**31 - 1),
        "learning_rate": number(payload["learning_rate"], "learning_rate", 1e-9, 0.1),
    }


def token_ids(value, limit, vocab_size):
    if not isinstance(value, list) or not 1 <= len(value) <= limit:
        raise ValueError("token sequence exceeds context limit or is empty")
    return [integer(v, "token", 0, vocab_size - 1) for v in value]


def vector(value, length, name, low, high):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must align with target tokens")
    return [number(v, name, low, high) for v in value]


def train_request(payload, algorithm, limits, vocab_size):
    fields(payload, ("version", "rows"), ("clip", "normalizer"))
    integer(payload["version"], "version", 0, 2**53)
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows or len(rows) > limits.request_tokens:
        raise ValueError("rows must be a nonempty bounded list")
    count = 0
    for row in rows:
        extra = {
            "sft": (),
            "grpo": ("old_logprobs", "advantages"),
            "opd": ("teacher_logprobs", "teacher_groups"),
        }[algorithm]
        fields(row, ("tokens", "mask", *extra))
        tokens = token_ids(row["tokens"], limits.context_tokens, vocab_size)
        if len(tokens) < 2:
            raise ValueError("training requires at least two tokens")
        width = len(tokens) - 1
        mask = row["mask"]
        if (
            not isinstance(mask, list)
            or len(mask) != width
            or any(type(v) is not bool for v in mask)
        ):
            raise ValueError("mask must contain one boolean per next-token target")
        if not any(mask):
            raise ValueError("each row must have supervised targets")
        if algorithm == "grpo":
            vector(row["old_logprobs"], width, "old_logprobs", -1e6, 0)
            vector(row["advantages"], width, "advantages", -1e6, 1e6)
        if algorithm == "opd":
            vector(row["teacher_logprobs"], width, "teacher_logprobs", -1e6, 0)
            groups = row["teacher_groups"]
            if not isinstance(groups, list) or len(groups) != width:
                raise ValueError("teacher groups must align with target tokens")
            for group in groups:
                integer(group, "teacher group", -1, width)
            selected = [i for i in range(width) if mask[i] and groups[i] >= 0]
            if not selected:
                raise ValueError("OPD row has no aligned teacher signal")
            for group in {groups[i] for i in selected}:
                values = {row["teacher_logprobs"][i] for i in selected if groups[i] == group}
                if len(values) != 1:
                    raise ValueError("teacher logsum must agree within each alignment group")
        count += len(tokens)
    if count > limits.request_tokens:
        raise AdmissionError("training request exceeds token budget")
    if algorithm != "grpo" and set(payload) & {"clip", "normalizer"}:
        raise ValueError("clip and normalizer apply only to GRPO")
    if algorithm == "grpo":
        number(payload.get("clip", 0.2), "clip", 0, 1)
        integer(
            payload.get("normalizer", limits.context_tokens), "normalizer", 1, limits.context_tokens
        )
    return count


def command_tokens(operation, payload, algorithm, limits, vocab_size):
    if operation == "train":
        return train_request(payload, algorithm, limits, vocab_size)
    if operation == "sample":
        fields(payload, ("version", "prompt", "max_tokens", "seed"))
        integer(payload["version"], "version", 0, 2**53)
        integer(payload["seed"], "seed", 0, 2**31 - 1)
        prompt = token_ids(payload["prompt"], limits.context_tokens, vocab_size)
        maximum = integer(payload["max_tokens"], "max_tokens", 1, limits.context_tokens)
        if len(prompt) + maximum > min(limits.context_tokens, limits.request_tokens):
            raise AdmissionError("sample request exceeds token budget")
        return len(prompt) + maximum
    if operation in ("checkpoint", "close"):
        fields(payload, ())
        return 0
    raise ValueError("unsupported shared-training operation")
