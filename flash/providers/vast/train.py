"""Vast.ai train submission: build the instance payload + submit a run."""

from __future__ import annotations

from flash.providers.runpod.train import (
    WORKER_DEPS,
    WORKER_SYSTEM_DEPS,
    build_worker_env,
    resolve_worker_deps,
)
from flash.providers.vast.jobs import build_payload, submit_run_vast

__all__ = [
    "WORKER_DEPS",
    "WORKER_SYSTEM_DEPS",
    "build_payload",
    "build_worker_env",
    "resolve_worker_deps",
    "submit_run_vast",
]
