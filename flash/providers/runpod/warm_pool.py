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
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from flash.providers.base import canonical_gpu
from flash.providers.runpod.train import worker_image_for_gpu
from flash.runner import _STATE_DIR

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
    gpu_type: str
    execution_timeout_ms: int
    released_at: float


def reuse_signature(spec, key_fingerprint: str) -> str:
    """Return the stable endpoint compatibility signature for one RunPod account."""
    gpu_type = canonical_gpu(spec.gpu.type)
    contract = {
        "provider": "runpod",
        "account_fp": key_fingerprint,
        "gpu_type": gpu_type,
        "gpu_count": 1,
        "disk_gb": spec.gpu.disk_gb,
        "network_volume": spec.gpu.network_volume,
        "network_volume_gb": spec.gpu.network_volume_gb,
        "worker_image": worker_image_for_gpu(gpu_type, allow_default=True),
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
        if not isinstance(endpoint_id, str) or not isinstance(value, dict):
            continue
        try:
            record = WarmEndpoint(**value)
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


def register(record: WarmEndpoint) -> bool:
    """Add or replace one endpoint, returning false when the hard pool cap rejects it."""
    with _locked_registry():
        records = _load_records()
        if record.endpoint_id not in records and len(records) >= MAX_WARM_ENDPOINTS:
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
