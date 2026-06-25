"""Lambda Cloud train submission: build the instance payload + submit a run.

The worker stack/env is substrate-neutral, so the per-run worker env and dependency resolution are
shared with RunPod (``providers/runpod/train.py``); this module owns the Lambda-specific submission
entrypoint and the instance payload shape. Provisioning, polling, and teardown live in
``providers/lambdalabs/jobs``.
"""

from __future__ import annotations

# Shared, substrate-neutral worker stack (single source of truth on RunPod's module).
from flash.providers.lambdalabs.jobs import build_payload, submit_run_lambda
from flash.providers.runpod.train import (
    WORKER_DEPS,
    WORKER_SYSTEM_DEPS,
    build_worker_env,
    resolve_worker_deps,
)

__all__ = [
    "WORKER_DEPS",
    "WORKER_SYSTEM_DEPS",
    "build_payload",
    "build_worker_env",
    "resolve_worker_deps",
    "submit_run_lambda",
]
