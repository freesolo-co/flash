"""Vast.ai train submission: build the instance payload + submit a durable run.

The worker stack/env is substrate-neutral, so the per-run worker env and dependency
resolution are shared with RunPod (``providers/runpod/train.py``); this module owns the
Vast-specific submission entrypoint and the instance payload shape. Provisioning,
polling, and teardown live in ``providers/vast/durable.py``.
"""

from __future__ import annotations

# Shared, substrate-neutral worker stack (single source of truth on RunPod's module).
from autoslm.providers.runpod.train import (
    WORKER_DEPS,
    WORKER_SYSTEM_DEPS,
    build_worker_env,
    resolve_worker_deps,
)
from autoslm.providers.vast.durable import build_payload, submit_train_durable_vast

__all__ = [
    "WORKER_DEPS",
    "WORKER_SYSTEM_DEPS",
    "build_payload",
    "build_worker_env",
    "resolve_worker_deps",
    "submit_train_durable_vast",
]
