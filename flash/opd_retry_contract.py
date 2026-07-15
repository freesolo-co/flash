"""Private fail-closed contract for detecting possible OPD optimizer mutation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from typing import Any

from flash.adapter_artifacts import ADAPTER_WEIGHT_FILES

OPD_RETRY_CONTRACT_STATUS_KEY = "opd_retry_contract_version"
OPD_RETRY_CONTRACT_VERSION = 1
OPD_OPTIMIZER_START_CONTRACT = "flash.opd.optimizer-start"
OPD_OPTIMIZER_START_PHASE = "opd"
OPD_RETRY_ARTIFACT_PREFIX = "_opd_retry"
OPD_OPTIMIZER_START_FILENAME = "optimizer-start.v1.json"
OPD_RESUME_REVISION_ENV = "FLASH_OPD_RESUME_REVISION"
OPD_RESUME_STATE_VERSION = 2
MAX_BOUNDED_NONNEGATIVE_INTEGER = (1 << 63) - 1

# a full-state opd resume checkpoint (the custom loop's counterpart to the hf trainer's checkpoint-n)
# is complete only with the adapter config, an adapter weight file, the optimizer moments, the rng
# blob, and the loop accounting. both the worker restore and the runner replacement gate key off this
# exact set so "resumable" means the same thing on the write and read sides.
OPD_RESUME_STATE_REQUIRED_FILES = (
    "adapter_config.json",
    "optimizer.pt",
    "rng_state.pth",
    "opd_state.json",
)

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_OPD_PROMPT_FINGERPRINT_RE = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_KEYS = frozenset({"attempt", "contract", "phase", "run_id", "seed", "version"})
_OPD_RESUME_ACCOUNTING_SCHEMA = {
    "generated_tokens": "nonneg_int",
    "teacher_input_tokens": "nonneg_int",
    "truncated_rollouts": "nonneg_int",
    "granularity_n": "nonneg_int",
    "samples_seen": "nonneg_int",
    "teacher_ok": "nonneg_int",
    "teacher_transient": "nonneg_int",
    "teacher_error": "nonneg_int",
    "no_signal_resamples": "nonneg_int",
    "no_signal_skipped_steps": "nonneg_int",
    "episodes_seen": "nonneg_int",
    "mt_turn_records": "nonneg_int",
    "granularity_sum": "nonneg_number",
    "train_wall_seconds": "nonneg_number",
    "loss_curve": "list",
    "coverage_curve": "list",
    "skip_counts": "dict",
    "opd_phase_seconds": "dict",
    "opd_phase_counts": "dict",
}


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


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"opd resume state {field} must be a nonnegative integer")
    return value


def _finite_number(value: object, *, field: str, nonnegative: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"opd resume state {field} must be a finite number")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"opd resume state {field} must be a finite number")
    if nonnegative and value < 0:
        raise ValueError(f"opd resume state {field} must be nonnegative")
    return value


def validate_opd_resume_state_metadata(
    state: object, *, expected_seed: int, checkpoint_step: int
) -> dict[str, Any]:
    """Validate the dependency-free JSON contract for one OPD resume checkpoint."""
    if not isinstance(state, dict):
        raise ValueError("opd resume state must be a json object")
    contract_version = _nonnegative_integer(
        state.get("contract_version"), field="contract_version"
    )
    if contract_version != OPD_RESUME_STATE_VERSION:
        raise ValueError(
            f"opd resume state contract_version must equal {OPD_RESUME_STATE_VERSION}"
        )
    expected = _nonnegative_integer(expected_seed, field="expected seed")
    seed = _nonnegative_integer(state.get("seed"), field="seed")
    if seed != expected:
        raise ValueError(f"opd resume state seed {seed} does not match expected seed {expected}")
    selected_step = _nonnegative_integer(checkpoint_step, field="checkpoint step")
    if selected_step == 0:
        raise ValueError("opd resume state checkpoint step must be positive")
    opt_steps = _nonnegative_integer(state.get("opt_steps"), field="opt_steps")
    if opt_steps == 0:
        raise ValueError("opd resume state opt_steps must be positive")
    if opt_steps != selected_step:
        raise ValueError(
            f"opd resume state opt_steps {opt_steps} does not match checkpoint-{selected_step}"
        )
    step = _nonnegative_integer(state.get("step"), field="step")
    if step < opt_steps:
        raise ValueError("opd resume state step must be at least opt_steps")
    _nonnegative_integer(state.get("rollout_seed_ordinal"), field="rollout_seed_ordinal")
    fingerprint = state.get("prompt_pool_fingerprint")
    if not isinstance(fingerprint, str) or _OPD_PROMPT_FINGERPRINT_RE.fullmatch(fingerprint) is None:
        raise ValueError("opd resume state prompt_pool_fingerprint must be 64 lowercase hex characters")

    for field, field_type in _OPD_RESUME_ACCOUNTING_SCHEMA.items():
        value = state.get(field)
        if field_type == "nonneg_int":
            _nonnegative_integer(value, field=field)
        elif field_type == "nonneg_number":
            _finite_number(value, field=field, nonnegative=True)
        elif field_type == "list" and not isinstance(value, list):
            raise ValueError(f"opd resume state {field} must be a list")
        elif field_type == "dict" and not isinstance(value, dict):
            raise ValueError(f"opd resume state {field} must be an object")

    for field in ("loss_curve", "coverage_curve"):
        curve = state[field]
        if len(curve) != opt_steps:
            raise ValueError(f"opd resume state {field} length must equal opt_steps")
        for index, value in enumerate(curve):
            _finite_number(value, field=f"{field}[{index}]")

    for key, value in state["skip_counts"].items():
        if not isinstance(key, str):
            raise ValueError("opd resume state skip_counts keys must be strings")
        _nonnegative_integer(value, field=f"skip_counts[{key!r}]")
    for key, value in state["opd_phase_seconds"].items():
        if not isinstance(key, str):
            raise ValueError("opd resume state opd_phase_seconds keys must be strings")
        _finite_number(value, field=f"opd_phase_seconds[{key!r}]", nonnegative=True)
    for key, value in state["opd_phase_counts"].items():
        if not isinstance(key, str):
            raise ValueError("opd resume state opd_phase_counts keys must be strings")
        _nonnegative_integer(value, field=f"opd_phase_counts[{key!r}]")
    return dict(state)


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


def opd_resume_checkpoint_complete(basenames: Iterable[str]) -> bool:
    """True iff `basenames` (the direct children of one checkpoint-N dir) form a complete resume state.

    Complete means every required state file plus at least one adapter weight file is present. A partial
    upload (e.g. adapter written but optimizer.pt missing) is NOT resumable, so both the restore path and
    the replacement gate treat it as absent and fail closed.
    """
    names = set(basenames)
    if not all(f in names for f in OPD_RESUME_STATE_REQUIRED_FILES):
        return False
    return any(weight in names for weight in ADAPTER_WEIGHT_FILES)
