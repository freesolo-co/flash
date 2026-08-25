"""RunPod fleet cleanup and orphan reconciliation."""

from __future__ import annotations

import time
from collections.abc import Callable

from flash._internal.logging import get_logger
from flash.providers._lifecycle.instances.instance import label_matches_run
from flash.providers._lifecycle.instances.poll import preload_box_reap_due
from flash.providers.runpod.client import api as runpod_api
from flash.providers.runpod.client import pods as runpod_pods
from flash.providers.runpod.execution.identity import (
    payload_secret_name_from_pod_label,
    pod_run_prefix,
)

logger = get_logger(__name__)


def payload_secret_name(label: str) -> str:
    return payload_secret_name_from_pod_label(label)


def _pod_payload_secret_name(pod: runpod_pods.RunpodPod) -> str:
    name = payload_secret_name(pod.name)
    if pod.payload_secret_name != name:
        raise runpod_api.RunpodApiError(
            f"runpod Pod {pod.id} payload secret identity is incomplete or conflicting"
        )
    return name


def run_pods_remaining(run_id: str) -> list[str]:
    """Return exact run-labeled Pod ids, failing closed on any account listing failure."""
    by_fingerprint, failed = runpod_pods.list_pods_by_key()
    if failed:
        raise runpod_api.RunpodApiError(
            "runpod Pod fleet listing was incomplete; run cleanup is unconfirmed"
        )
    prefix = pod_run_prefix(run_id)
    return [
        pod.id
        for pods in by_fingerprint.values()
        for pod in pods
        if label_matches_run(pod.name, prefix)
    ]


def destroy_run_pods(run_id: str) -> list[str]:
    """Terminate every exact run-labeled Pod and its attempt payload secret."""
    by_fingerprint, failed = runpod_pods.list_pods_by_key()
    if failed:
        raise runpod_api.RunpodApiError(
            "runpod Pod fleet listing was incomplete; refusing partial run cleanup"
        )
    prefix = pod_run_prefix(run_id)
    deleted = []
    deadline_at = time.time() + 120.0
    for fingerprint, pods in by_fingerprint.items():
        for pod in pods:
            if not label_matches_run(pod.name, prefix):
                continue
            _account, secrets = runpod_pods.list_secrets_for_fingerprint(
                fingerprint,
                name=_pod_payload_secret_name(pod),
                deadline_at=deadline_at,
            )
            runpod_pods.delete_pod_for_fingerprint(pod.id, fingerprint, deadline_at=deadline_at)
            deleted.append(pod.id)
            for secret in secrets:
                runpod_pods.delete_secret_for_fingerprint(
                    fingerprint,
                    secret.id,
                    secret.name,
                    deadline_at=deadline_at,
                )
    return deleted


def sweep_orphan_pods(
    *,
    active_labels: set[str] | Callable[[], set[str]] | None,
    known_labels: set[str] | Callable[[], set[str]] | None,
) -> list[str]:
    """Terminate inactive control-plane-owned Pods only after a complete fleet observation."""
    by_fingerprint, failed = runpod_pods.list_pods_by_key()
    if failed:
        logger.warning("runpod orphan sweep skipped; one or more accounts failed to list")
        return []
    try:
        active_raw = active_labels() if callable(active_labels) else active_labels
        known_raw = known_labels() if callable(known_labels) else known_labels
    except Exception as exc:
        logger.warning("runpod orphan sweep skipped; could not resolve run sets: %s", exc)
        return []
    active = {pod_run_prefix(value) for value in (active_raw or set())}
    known = (
        None if known_labels is None else {pod_run_prefix(value) for value in (known_raw or set())}
    )

    def owned(label: str, prefixes: set[str]) -> bool:
        return any(label_matches_run(label, prefix) for prefix in prefixes)

    deadline_at = time.time() + 120.0
    secret_inventory = {}
    try:
        for fingerprint in by_fingerprint:
            _account, secrets = runpod_pods.list_secrets_for_fingerprint(
                fingerprint, deadline_at=deadline_at
            )
            secret_inventory[fingerprint] = secrets
    except Exception as exc:
        logger.warning("runpod orphan sweep skipped; payload secret listing failed: %s", exc)
        return []
    deleted = []
    for fingerprint, pods in by_fingerprint.items():
        for pod in pods:
            if owned(pod.name, active):
                continue
            expired_preload = preload_box_reap_due(pod.name, time.time())
            if known is not None and not owned(pod.name, known) and not expired_preload:
                continue
            try:
                secret_name = _pod_payload_secret_name(pod)
                runpod_pods.delete_pod_for_fingerprint(pod.id, fingerprint, deadline_at=deadline_at)
                for secret in secret_inventory[fingerprint]:
                    if secret.name != secret_name:
                        continue
                    runpod_pods.delete_secret_for_fingerprint(
                        fingerprint,
                        secret.id,
                        secret.name,
                        deadline_at=deadline_at,
                    )
                deleted.append(pod.id)
            except Exception as exc:
                logger.warning("runpod orphan cleanup failed for Pod %s: %s", pod.id, exc)
    return deleted
