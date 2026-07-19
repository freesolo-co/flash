"""Durable hints for reusing idle RunPod training endpoints.

The reuse contract requires identical account, GPU, image, volume, base model, immutable model
revision, and context length. Environment, dataset, seed, run id, phase, and training
hyperparameters are deliberately excluded because each queued job reloads its model and receives its
environment and data in the job payload. Execution timeout is checked separately when claiming.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from flash._logging import get_logger
from flash.providers._deadline import deadline_kwargs
from flash.providers.base import canonical_gpu
from flash.providers.runpod import api as runpod_api
from flash.providers.runpod.train import worker_image_for_gpu
from flash.runner import _STATE_DIR

logger = get_logger(__name__)

MAX_WARM_ENDPOINTS = 8
WARM_DIR = os.path.join(_STATE_DIR, "runpod_warm")
_REGISTRY_PATH = "endpoints.json"
_LOCK_PATH = "endpoints.lock"
_THREAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class WarmEndpoint:
    endpoint_id: str
    name: str
    key_fingerprint: str
    signature: str
    execution_timeout_ms: int
    released_at: float

    @classmethod
    def from_dict(cls, value: dict) -> WarmEndpoint:
        fields = {
            "endpoint_id",
            "name",
            "key_fingerprint",
            "signature",
            "execution_timeout_ms",
            "released_at",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("warm endpoint record fields are invalid")
        for field in ("endpoint_id", "name", "key_fingerprint", "signature"):
            item = value[field]
            if not isinstance(item, str) or not item:
                raise ValueError(f"warm endpoint {field} is invalid")
        timeout = value["execution_timeout_ms"]
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError("warm endpoint execution timeout is invalid")
        released_at = value["released_at"]
        if not isinstance(released_at, float) or not math.isfinite(released_at):
            raise ValueError("warm endpoint release timestamp is invalid")
        return cls(**value)


def reuse_signature(
    spec,
    key_fingerprint: str,
    *,
    worker_image: str | None = None,
    has_volume: bool | None = None,
) -> str:
    """Return the stable endpoint compatibility signature for one RunPod account."""
    gpu_type = canonical_gpu(spec.gpu.type)
    image = worker_image or worker_image_for_gpu(gpu_type, allow_default=True)
    volume_attached = bool(spec.gpu.network_volume) if has_volume is None else has_volume
    # environment intentionally stays out of this signature. cross-environment reuse depends on
    # per-job worker environment isolation, which is enforced by a separate worker-isolation change.
    contract = {
        "provider": "runpod",
        "account_fp": key_fingerprint,
        "gpu_type": gpu_type,
        "gpu_count": 1,
        "disk_gb": spec.gpu.disk_gb,
        "network_volume": spec.gpu.network_volume if volume_attached else None,
        "network_volume_gb": spec.gpu.network_volume_gb if volume_attached else None,
        "worker_image": image,
        "base_model": spec.model,
        "base_model_revision": spec.model_revision,
        "context_length": spec.train.max_context_tokens,
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


@contextmanager
def _locked_registry() -> Iterator[None]:
    os.makedirs(WARM_DIR, exist_ok=True)
    lock_path = os.path.join(WARM_DIR, _LOCK_PATH)
    with _THREAD_LOCK, open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _load_records() -> dict[str, WarmEndpoint]:
    path = os.path.join(WARM_DIR, _REGISTRY_PATH)
    try:
        with open(path) as registry_file:
            raw = json.load(registry_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    records: dict[str, WarmEndpoint] = {}
    for endpoint_id, value in raw.items():
        if not isinstance(endpoint_id, str):
            continue
        try:
            record = WarmEndpoint.from_dict(value)
        except (TypeError, ValueError):
            continue
        if record.endpoint_id == endpoint_id:
            records[endpoint_id] = record
    return records


def _write_records(records: dict[str, WarmEndpoint]) -> None:
    path = os.path.join(WARM_DIR, _REGISTRY_PATH)
    data = {endpoint_id: asdict(record) for endpoint_id, record in sorted(records.items())}
    fd, tmp = tempfile.mkstemp(dir=WARM_DIR, prefix="endpoints.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as registry_file:
            json.dump(data, registry_file, indent=2, sort_keys=True)
            registry_file.flush()
            os.fsync(registry_file.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(WARM_DIR, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def _health_state(health: object) -> str:
    if not isinstance(health, dict):
        return "stale"
    workers = health.get("workers")
    jobs_info = health.get("jobs")
    if not isinstance(workers, dict) or not isinstance(jobs_info, dict):
        return "stale"
    if (
        (jobs_info.get("inQueue") or 0) != 0
        or (jobs_info.get("inProgress") or 0) != 0
        or (workers.get("unhealthy") or 0) != 0
    ):
        return "busy"
    return "reusable"


def _prune_dead_records(records: dict[str, WarmEndpoint]) -> bool:
    changed = False
    for endpoint_id, record in list(records.items()):
        try:
            health = runpod_api.endpoint_health_for_fingerprint(
                endpoint_id,
                record.key_fingerprint,
            )
        except Exception:
            records.pop(endpoint_id, None)
            changed = True
            continue
        if _health_state(health) == "stale":
            records.pop(endpoint_id, None)
            changed = True
    return changed


def register(record: WarmEndpoint) -> bool:
    """Add or replace one endpoint, reconciling dead records only when the hard cap is full."""
    with _locked_registry():
        records = _load_records()
        if record.endpoint_id not in records and len(records) >= MAX_WARM_ENDPOINTS:
            if _prune_dead_records(records):
                _write_records(records)
            if len(records) >= MAX_WARM_ENDPOINTS:
                return False
        records[record.endpoint_id] = record
        _write_records(records)
    return True


def candidates(signature: str, min_execution_timeout_ms: int) -> list[WarmEndpoint]:
    """Return compatible warm endpoints, newest release first."""
    with _locked_registry():
        records = _load_records()
    return sorted(
        (
            record
            for record in records.values()
            if record.signature == signature
            and record.execution_timeout_ms >= min_execution_timeout_ms
        ),
        key=lambda record: record.released_at,
        reverse=True,
    )


def _all_candidates() -> list[WarmEndpoint]:
    with _locked_registry():
        records = _load_records()
    return sorted(records.values(), key=lambda record: record.released_at, reverse=True)


def claim(endpoint_id: str) -> bool:
    """Atomically remove one endpoint so only one concurrent caller can reuse it."""
    with _locked_registry():
        records = _load_records()
        if endpoint_id not in records:
            return False
        records.pop(endpoint_id)
        _write_records(records)
    return True


def prune(endpoint_id: str) -> None:
    """Remove one stale endpoint hint if present."""
    with _locked_registry():
        records = _load_records()
        if endpoint_id not in records:
            return
        records.pop(endpoint_id)
        _write_records(records)


def acquire(
    spec,
    min_execution_timeout_ms: int,
    deadline_at: float | None,
    *,
    worker_image: str | None = None,
    has_volume: bool | None = None,
) -> WarmEndpoint | None:
    """Claim and return the first compatible healthy endpoint across all stored accounts."""
    try:
        records = _all_candidates()
    except Exception:
        logger.warning("warm endpoint registry unavailable; creating a fresh endpoint", exc_info=True)
        return None
    for record in records:
        signature = reuse_signature(
            spec,
            record.key_fingerprint,
            worker_image=worker_image,
            has_volume=has_volume,
        )
        if (
            record.signature != signature
            or record.execution_timeout_ms < min_execution_timeout_ms
        ):
            continue
        try:
            health = runpod_api.endpoint_health_for_fingerprint(
                record.endpoint_id,
                record.key_fingerprint,
                **deadline_kwargs(runpod_api.endpoint_health_for_fingerprint, deadline_at),
            )
        except Exception:
            with contextlib.suppress(Exception):
                prune(record.endpoint_id)
            continue
        state = _health_state(health)
        if state == "busy":
            continue
        if state == "stale":
            with contextlib.suppress(Exception):
                prune(record.endpoint_id)
            continue
        try:
            claimed = claim(record.endpoint_id)
        except Exception:
            logger.warning("warm endpoint claim failed; creating a fresh endpoint", exc_info=True)
            return None
        if not claimed:
            continue
        logger.info("reusing warm RunPod endpoint %s (%s)", record.name, record.endpoint_id)
        return record
    return None
