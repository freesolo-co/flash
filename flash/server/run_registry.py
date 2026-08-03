"""Best-effort reporting of managed Flash runs/checkpoints to the Freesolo backend."""

from __future__ import annotations

import logging
import urllib.request
from datetime import UTC, datetime
from typing import Any

from flash.spec import require_project_id

from ._internal_client import org_id_of, post_internal_json

_LOG = logging.getLogger("flash.server.runs")
_RUN_PATH = "/api/flash/runs/internal"
_CHECKPOINT_PATH = "/api/flash/runs/checkpoints/internal"
_EVENT_PATH = "/api/flash/events/internal"


def _iso_from_epoch(value: float | None) -> str | None:
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


def _upload_suppressed(spec: dict[str, Any]) -> bool:
    """True when this run opted out of dashboard mirroring via `upload = false`.

    Every reporter in this module funnels through here, so one check covers runs, checkpoint
    metrics, deployments (which ride inside the run report's `deployment` field), and export
    events. Only an explicit `False` suppresses: a spec that predates the field reports exactly
    as it did before, and a malformed value can never silently hide a run that asked to be shown.

    One run-scoped report lives outside this module and is gated at its own call site:
    `record_environment_use` in routes/runs.py, which posts the run id at submit time. Any new
    backend post that carries a run id belongs behind one of these two checks.
    """
    return spec.get("upload") is False


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
    if _upload_suppressed(spec):
        return False
    try:
        project_id = require_project_id(spec.get("project"))
    except (TypeError, ValueError):
        return False
    gpu = spec.get("gpu") if isinstance(spec.get("gpu"), dict) else {}
    body = {
        "orgId": org_id,
        "runId": status.run_id,
        "status": status.state,
        "userId": context.get("user_id"),
        "apiKeyId": context.get("api_key_id"),
        "projectId": project_id,
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


def record_model_exported(
    *,
    status: Any,
    key: dict[str, Any] | None = None,
    repository: str,
    url: str,
    step: int | None = None,
) -> bool:
    """Report an adapter export as a platform product-analytics event.

    An export is an hf-to-hf copy that never otherwise touches the platform
    backend, so it is reported explicitly through the (allowlisted) internal
    product-event endpoint. Best-effort like every other internal reporter.
    """
    context = {**_context_from_status(status), **(key or {})}
    org_id = org_id_of(context)
    if not org_id:
        return False
    spec = _spec_from_status(status)
    if _upload_suppressed(spec):
        return False
    try:
        project_id = require_project_id(spec.get("project"))
    except (TypeError, ValueError):
        return False
    body = {
        "orgId": org_id,
        "userId": context.get("user_id"),
        "event": "flash_model_exported",
        "properties": {
            "project": project_id,
            "run_id": status.run_id,
            "repository": repository,
            "url": url,
            "step": step,
            "model": spec.get("model") if isinstance(spec.get("model"), str) else None,
        },
    }
    return _post(_EVENT_PATH, body)


def record_training_checkpoint(
    *,
    spec: Any,
    metrics: dict[str, Any],
    artifact_path: str,
) -> bool:
    try:
        from flash.engine.accounting import sanitize_worker_metrics
        from flash.runner import adapter_ref, get_status

        metrics = sanitize_worker_metrics(metrics)
        status = get_status(spec.run_id)
        ref = adapter_ref(spec)
    except Exception:
        return False
    context = _context_from_status(status)
    org_id = org_id_of(context)
    if not org_id:
        return False
    persisted_spec = _spec_from_status(status)
    if _upload_suppressed(persisted_spec):
        return False
    try:
        project_id = require_project_id(persisted_spec.get("project"))
    except (TypeError, ValueError):
        return False
    body = {
        "orgId": org_id,
        "projectId": project_id,
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
