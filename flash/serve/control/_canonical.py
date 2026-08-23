"""canonical json normalization and fingerprints for control records."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence


def _normalize_json_value(value: object, path: str) -> object:
    if value is None or type(value) in {bool, str, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite json numbers")
        if value == 0:
            return 0
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, entry in value.items():
            if type(key) is not str:
                raise ValueError(f"{path} keys must be strings")
            normalized[key] = _normalize_json_value(entry, f"{path}.{key}")
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _normalize_json_value(entry, f"{path}[{index}]") for index, entry in enumerate(value)
        ]
    raise ValueError(f"{path} must contain only json-safe values")


def canonical_json(value: object) -> str:
    """return deterministic json for an already normalized public value."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """normalize a json-safe mapping without retaining caller-owned containers."""

    normalized = _normalize_json_value(value, "mapping")
    if type(normalized) is not dict:
        raise ValueError("value must be a mapping")
    return normalized


def canonical_mapping_fingerprint(value: Mapping[str, object]) -> str:
    """return the sha-256 fingerprint of a normalized json-safe mapping."""

    payload = canonical_json(canonical_mapping(value)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
