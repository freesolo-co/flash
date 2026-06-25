"""Hyperstack train submission: build the VM payload + submit a run.

The worker stack/env is substrate-neutral, so the per-run worker env and dependency resolution are
shared with RunPod; this module owns the Hyperstack-specific submission entrypoint. Provisioning,
polling, and teardown live in ``providers/hyperstack/jobs``.
"""

from __future__ import annotations

from flash.providers.hyperstack.jobs import build_payload, submit_run_hyperstack
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
    "submit_run_hyperstack",
]
