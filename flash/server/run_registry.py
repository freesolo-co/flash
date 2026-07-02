"""Best-effort reporting of managed Flash runs/checkpoints to the Freesolo backend."""

from __future__ import annotations

import logging
import urllib.request
from datetime import UTC, datetime
from typing import Any

from ._internal_client import org_id_of, post_internal_json

_LOG = logging.getLogger("flash.server.runs")
_RUN_PATH = "/api/flash/runs/internal"
_CHECKPOINT_PATH = "/api/flash/runs/checkpoints/internal"


def _iso_from_epoch(value: float | int | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _post(path: str, body: dict[str, Any]) -> bool:
    return post_internal_json(
        path,
        body,
        subject=f"report {path}",
        logger=_LOG,
        urlopen=urllib.request.urlopen,
    )


def _context_from_status(status: Any) -> dict[str, Any]:
    platform = getattr(status, "platform_context", None)
    if isinstance(platform, dict):
        return platform
    billing = getattr(status, "billing_context", None)
    if isinstance(billing, dict):
        return billing
    return {}


def _spec_from_status(status: Any) -> dict[str, Any]:
    spec = getattr(status, "spec", None)
    return spec if isinstance(spec, dict) else {}


def _managed_environment_slug(spec: dict[str, Any]) -> str | None:
    env = spec.get("environment") if isinstance(spec.get("environment"), dict) else {}
    env_id = env.get("id")
    if not isinstance(env_id, str) or not env_id.strip():
        return None
    try:
        from flash.envs.adapter import is_managed_environment_slug

        return env_id.strip() if is_managed_environment_slug(env_id.strip()) else None
    except Exception:
        return None


def record_training_run(*, status: Any, key: dict[str, Any] | None = None) -> bool:
    context = {**_context_from_status(status), **(key or {})}
    org_id = org_id_of(context)
    if not org_id:
        return False

    spec = _spec_from_status(status)
    gpu = spec.get("gpu") if isinstance(spec.get("gpu"), dict) else {}
    body = {
        "orgId": org_id,
        "runId": status.run_id,
        "status": status.state,
        "userId": context.get("user_id"),
        "apiKeyId": context.get("api_key_id"),
        "environmentSlug": _managed_environment_slug(spec),
        "model": spec.get("model") if isinstance(spec.get("model"), str) else None,
        "algorithm": spec.get("algorithm") if isinstance(spec.get("algorithm"), str) else None,
        "phase": spec.get("phase") if isinstance(spec.get("phase"), str) else None,
        "gpuType": gpu.get("type") if isinstance(gpu.get("type"), str) else None,
        "costUsd": status.cost_usd,
        "realizedCostUsd": status.realized_cost_usd,
        "adapterRef": status.to_dict().get("adapter_ref"),
        "artifactsDir": status.artifacts_dir,
        "error": status.error,
        "spec": spec,
        "deployment": status.deployment,
        "lastHeartbeat": status.last_heartbeat,
        "gpuStatus": status.gpu_status,
        "createdAt": _iso_from_epoch(status.created_at),
        "updatedAt": _iso_from_epoch(status.updated_at),
        "metadata": {"source": "flash.control_plane"},
    }
    return _post(_RUN_PATH, body)


def record_training_checkpoint(
    *,
    spec: Any,
    metrics: dict[str, Any],
    artifact_path: str,
) -> bool:
    try:
        from flash.runner import adapter_ref, get_status

        status = get_status(spec.run_id)
        ref = adapter_ref(spec)
    except Exception:
        return False
    context = _context_from_status(status)
    org_id = org_id_of(context)
    if not org_id:
        return False
    body = {
        "orgId": org_id,
        "runId": spec.run_id,
        "checkpointId": "final",
        "phase": getattr(spec, "phase", None),
        "adapterRef": ref,
        "artifactPath": artifact_path,
        "metrics": metrics,
        "metadata": {"source": "flash.control_plane"},
        "updatedAt": _iso_from_epoch(getattr(status, "updated_at", None)),
    }
    return _post(_CHECKPOINT_PATH, body)
