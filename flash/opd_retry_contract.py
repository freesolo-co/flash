"""Private fail-closed contract for detecting possible OPD optimizer mutation."""

from __future__ import annotations

import json
import re
from typing import Any

OPD_RETRY_CONTRACT_STATUS_KEY = "opd_retry_contract_version"
OPD_RETRY_CONTRACT_VERSION = 1
OPD_OPTIMIZER_START_CONTRACT = "flash.opd.optimizer-start"
OPD_OPTIMIZER_START_PHASE = "opd"
OPD_RETRY_ARTIFACT_PREFIX = "_opd_retry"
OPD_OPTIMIZER_START_FILENAME = "optimizer-start.v1.json"
MAX_BOUNDED_NONNEGATIVE_INTEGER = (1 << 63) - 1

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PAYLOAD_KEYS = frozenset({"attempt", "contract", "phase", "run_id", "seed", "version"})


def bounded_nonnegative_integer(value: object, *, field: str) -> int:
    """Return a bounded nonnegative integer, rejecting bool and coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a bounded nonnegative integer")
    if value < 0 or value > MAX_BOUNDED_NONNEGATIVE_INTEGER:
        raise ValueError(f"{field} must be a bounded nonnegative integer")
    return value


def require_opd_retry_contract_version(value: object) -> int:
    """Require the only supported private OPD retry contract version."""
    version = bounded_nonnegative_integer(value, field="opd retry contract version")
    if version != OPD_RETRY_CONTRACT_VERSION:
        raise ValueError(f"unsupported opd retry contract version: {version}")
    return version


def _run_id(value: object) -> str:
    if not isinstance(value, str) or _RUN_ID_RE.fullmatch(value) is None:
        raise ValueError("run_id is invalid")
    return value


def opd_optimizer_start_marker_path(run_id: str, attempt: int) -> str:
    """Return the exact private HF path for one attempt's mutation marker."""
    return (
        f"{OPD_RETRY_ARTIFACT_PREFIX}/{_run_id(run_id)}/attempts/attempt-"
        f"{bounded_nonnegative_integer(attempt, field='attempt')}/{OPD_OPTIMIZER_START_FILENAME}"
    )


def opd_optimizer_start_payload(
    *, run_id: str, attempt: int, seed: int, version: int = OPD_RETRY_CONTRACT_VERSION
) -> dict[str, Any]:
    """Build the exact optimizer-start marker payload."""
    return {
        "attempt": bounded_nonnegative_integer(attempt, field="attempt"),
        "contract": OPD_OPTIMIZER_START_CONTRACT,
        "phase": OPD_OPTIMIZER_START_PHASE,
        "run_id": _run_id(run_id),
        "seed": bounded_nonnegative_integer(seed, field="seed"),
        "version": require_opd_retry_contract_version(version),
    }


def canonical_opd_optimizer_start_json(
    *, run_id: str, attempt: int, seed: int, version: int = OPD_RETRY_CONTRACT_VERSION
) -> bytes:
    """Encode the exact payload as deterministic canonical JSON bytes."""
    payload = opd_optimizer_start_payload(
        run_id=run_id,
        attempt=attempt,
        seed=seed,
        version=version,
    )
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate marker key: {key}")
        obj[key] = value
    return obj


def decode_opd_optimizer_start_json(
    raw: bytes | str,
    *,
    run_id: str,
    attempt: int,
    seed: int,
    version: int = OPD_RETRY_CONTRACT_VERSION,
) -> dict[str, Any]:
    """Strictly decode and identity-check one canonical optimizer-start marker."""
    expected = opd_optimizer_start_payload(
        run_id=run_id,
        attempt=attempt,
        seed=seed,
        version=version,
    )
    if isinstance(raw, str):
        encoded = raw.encode("utf-8")
    elif isinstance(raw, bytes):
        encoded = raw
    else:
        raise ValueError("optimizer-start marker must be utf-8 json")
    try:
        decoded = json.loads(encoded.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("optimizer-start marker is not valid utf-8 json") from exc
    if not isinstance(decoded, dict) or frozenset(decoded) != _PAYLOAD_KEYS:
        raise ValueError("optimizer-start marker schema is invalid")
    if decoded != expected:
        raise ValueError("optimizer-start marker identity is invalid")
    canonical = canonical_opd_optimizer_start_json(
        run_id=run_id,
        attempt=attempt,
        seed=seed,
        version=version,
    )
    if encoded != canonical:
        raise ValueError("optimizer-start marker json is not canonical")
    return decoded
