"""dependency-light sampling validation and vllm choice normalization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .errors import RuntimeConfigurationError, RuntimeNotReadyError

MIN_SEED = -(2**63)
MAX_SEED = 2**63 - 1


def validate_choice_count(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= 4:
        raise RuntimeConfigurationError("n must be an integer from 1 through 4")
    return value


def validate_seed(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not MIN_SEED <= value <= MAX_SEED:
        raise RuntimeConfigurationError("seed must be a signed 64-bit integer or null")
    if value == -1:
        raise RuntimeConfigurationError("seed=-1 is reserved and is not supported")
    return value


def validate_penalty(value: Any, name: str) -> float:
    if type(value) not in {int, float}:
        raise RuntimeConfigurationError(f"{name} must be a finite number from -2 through 2")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise RuntimeConfigurationError(
            f"{name} must be a finite number from -2 through 2"
        ) from exc
    if not math.isfinite(normalized) or not -2 <= normalized <= 2:
        raise RuntimeConfigurationError(f"{name} must be a finite number from -2 through 2")
    return normalized


def validate_logprobs(value: Any) -> bool:
    if type(value) is not bool:
        raise RuntimeConfigurationError("logprobs must be a boolean")
    return value


def validate_top_logprobs(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= 20:
        raise RuntimeConfigurationError("top_logprobs must be an integer from 0 through 20")
    return value


def validate_sampling_relationships(
    *, n: int, temperature: float, logprobs: bool, top_logprobs: int
) -> None:
    if n > 1 and temperature == 0:
        raise RuntimeConfigurationError("n greater than 1 requires temperature greater than zero")
    if top_logprobs > 0 and not logprobs:
        raise RuntimeConfigurationError("positive top_logprobs requires logprobs=true")


def choice_index(output: Any, *, n: int) -> int:
    value = getattr(output, "index", None)
    if type(value) is not int or not 0 <= value < n:
        raise RuntimeNotReadyError("vllm returned an invalid choice index")
    return value


def indexed_outputs(request_output: Any, *, n: int) -> dict[int, Any]:
    outputs = getattr(request_output, "outputs", None)
    if not isinstance(outputs, list | tuple) or not outputs:
        raise RuntimeNotReadyError("vllm returned no output choices")
    indexed: dict[int, Any] = {}
    for output in outputs:
        index = choice_index(output, n=n)
        if index in indexed:
            raise RuntimeNotReadyError("vllm returned a duplicate choice index")
        indexed[index] = output
    return indexed


def complete_indexed_outputs(request_output: Any, *, n: int) -> dict[int, Any]:
    indexed = indexed_outputs(request_output, n=n)
    if set(indexed) != set(range(n)):
        raise RuntimeNotReadyError("vllm returned an incomplete choice set")
    return indexed


def normalize_token_logprobs(
    token_ids: Sequence[Any], raw_logprobs: Any, *, top_logprobs: int
) -> list[dict[str, Any]] | None:
    if raw_logprobs is None:
        return None
    if not isinstance(raw_logprobs, list | tuple) or len(raw_logprobs) != len(token_ids):
        raise RuntimeNotReadyError("vllm returned malformed token logprobs")
    if type(top_logprobs) is not int or top_logprobs < 0:
        raise RuntimeNotReadyError("requested top_logprobs count is invalid")
    return [
        _normalize_token_position(int(token_id), candidates, top_logprobs=top_logprobs)
        for token_id, candidates in zip(token_ids, raw_logprobs, strict=True)
    ]


def _normalize_token_position(
    token_id: int, candidates: Any, *, top_logprobs: int
) -> dict[str, Any]:
    if not isinstance(candidates, Mapping) or not candidates:
        raise RuntimeNotReadyError("vllm returned malformed token logprobs")
    normalized: list[tuple[int, dict[str, Any]]] = []
    selected: dict[str, Any] | None = None
    for candidate_id, candidate in candidates.items():
        if type(candidate_id) is not int:
            raise RuntimeNotReadyError("vllm returned a non-integer logprob token id")
        record = _normalize_logprob_candidate(candidate_id, candidate)
        normalized.append((candidate_id, record))
        if candidate_id == token_id:
            selected = record
    if selected is None:
        raise RuntimeNotReadyError("vllm logprobs omitted the selected token")
    return {
        **selected,
        "top_logprobs": [record for _, record in normalized[:top_logprobs]],
    }


def _normalize_logprob_candidate(token_id: int, candidate: Any) -> dict[str, Any]:
    raw_value = getattr(candidate, "logprob", None)
    if type(raw_value) not in {int, float}:
        raise RuntimeNotReadyError("vllm returned an invalid logprob value")
    value = float(cast("int | float", raw_value))
    if not math.isfinite(value):
        raise RuntimeNotReadyError("vllm returned a non-finite logprob value")
    decoded = getattr(candidate, "decoded_token", None)
    if not isinstance(decoded, str):
        decoded = str(token_id)
    return {
        "token": decoded,
        "logprob": value,
        "bytes": list(decoded.encode("utf-8")),
    }
