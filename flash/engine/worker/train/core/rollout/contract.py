"""V1 request identity and token semantics shared by generation and training.

Log probabilities are natural logs of the sampled token given the preceding token
prefix, recorded during generation. V1 names raw and processed distributions
separately; an adapter must never silently translate between them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import asdict, dataclass, fields

PROTOCOL_VERSION = 1
LOGPROB_DEFINITIONS = frozenset({"raw_logprobs", "processed_logprobs"})


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"rollout {name} must be an integer >= {minimum}")
    return value


def text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"rollout {name} must be a nonempty string")
    return value


def exact_fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("rollout contract has an invalid field set")


@dataclass(frozen=True)
class PolicyIdentity:
    run_id: str
    base_revision: str
    tokenizer_digest: str
    template_digest: str
    policy_version: int
    adapter_revision: str

    def __post_init__(self):
        text(self.run_id, "run_id")
        if not isinstance(self.base_revision, str) or not re.fullmatch(
            r"[0-9a-f]{40}", self.base_revision
        ):
            raise ValueError("rollout base_revision must be an immutable commit")
        for name in ("tokenizer_digest", "template_digest"):
            if not isinstance(getattr(self, name), str) or not re.fullmatch(
                r"[0-9a-f]{64}", getattr(self, name)
            ):
                raise ValueError(f"rollout {name} must be a SHA256 digest")
        integer(self.policy_version, "policy_version")
        text(self.adapter_revision, "adapter_revision")

    @classmethod
    def decode(cls, value):
        exact_fields(value, (item.name for item in fields(cls)))
        return cls(**value)

    def encode(self):
        return asdict(self)


@dataclass(frozen=True)
class RolloutRequest:
    protocol_version: int
    policy: PolicyIdentity
    optimizer_step: int
    sample_index: int
    rollout_ordinal: int
    attempt_id: str
    turn_index: int
    request_id: str
    prompt_digest: str
    logprob_definition: str

    def __post_init__(self):
        if type(self.protocol_version) is not int or self.protocol_version != PROTOCOL_VERSION:
            raise ValueError("unsupported rollout protocol version")
        if not isinstance(self.policy, PolicyIdentity):
            raise ValueError("rollout policy must be a PolicyIdentity")
        for name in ("optimizer_step", "sample_index", "rollout_ordinal", "turn_index"):
            integer(getattr(self, name), name, 1 if name == "optimizer_step" else 0)
        for name in ("attempt_id", "request_id", "prompt_digest"):
            text(getattr(self, name), name)
        if self.logprob_definition not in LOGPROB_DEFINITIONS:
            raise ValueError("unsupported rollout logprob definition")

    @classmethod
    def decode(cls, value):
        exact_fields(value, (item.name for item in fields(cls)))
        return cls(**dict(value, policy=PolicyIdentity.decode(value["policy"])))

    def encode(self):
        return asdict(self)


def validate_tokens(token_ids, logprobs):
    for token in token_ids:
        integer(token, "token id")
    if logprobs is None or len(token_ids) != len(logprobs):
        raise ValueError("rollout logprobs must align exactly with tokens")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value > 0
        for value in logprobs
    ):
        raise ValueError("rollout logprobs must be finite nonpositive natural logs")


def token_digest(prompt, response, logprobs, mask):
    """Bind queue tensors using their transport precision (float32 logprobs)."""
    validate_tokens(response, logprobs)
    for token in prompt:
        integer(token, "prompt token id")
    if len(mask) != len(response) or any(type(x) is not int or x not in (0, 1) for x in mask):
        raise ValueError("rollout loss mask must be token aligned and binary")
    logs = [struct.unpack("!f", struct.pack("!f", value))[0] for value in logprobs]
    return digest([prompt, response, logs, mask])


def validate_result_header(request, result):
    exact_fields(result, ("request", "load_ack", "termination", "sampling", "token_digest"))
    if RolloutRequest.decode(result["request"]) != request:
        raise ValueError("rollout result does not match its requested policy or attempt")
    text(result["load_ack"], "load acknowledgement")
    if result["termination"] not in {"stop", "length", "cancelled"}:
        raise ValueError("unknown rollout termination reason")
    if not isinstance(result["sampling"], dict) or not result["sampling"]:
        raise ValueError("rollout result is missing effective sampling parameters")
    if result["termination"] == "cancelled":
        raise ValueError("cancelled rollout cannot be accepted for training")


def validate_result(request, result, token_ids, logprobs):
    validate_result_header(request, result)
    validate_tokens(token_ids, logprobs)
    if result["token_digest"] != digest([token_ids, logprobs]):
        raise ValueError("rollout result token/probability digest mismatch")
