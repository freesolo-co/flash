"""Best-effort reporting of managed Flash runs/checkpoints to the Freesolo backend."""

from __future__ import annotations

import logging
import urllib.request
from datetime import UTC, datetime
from typing import Any

from flash.core.spec import attributed_gpu_type, require_project_id
from flash.schema import format_checkpoint_ref, parse_checkpoint_ref
from flash.server.platform.internal_client import org_id_of, post_internal_json

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


def _managed_environment_slug(spec: dict[str, Any]) -> str | None:
    env = spec.get("environment") if isinstance(spec.get("environment"), dict) else {}
    env_id = env.get("id")
    if not isinstance(env_id, str) or not env_id.strip():
        return None
    try:
        from flash.envs.loading.adapter import is_managed_environment_slug

        return env_id.strip() if is_managed_environment_slug(env_id.strip()) else None
    except Exception:
        return None


_DEPLOYMENT_FIELDS = frozenset(
    {
        "state",
        "checkpoint_id",
        "endpoint",
        "openai_base_url",
        "deployment_generation",
        "requested_at",
        "updated_at",
        "verified_at",
        "last_deploy_failed_at",
        "error",
        "last_deploy_error",
    }
)


def _checkpoint_id(status: Any) -> str | None:
    deployment = status.deployment if isinstance(status.deployment, dict) else {}
    candidate = deployment.get("checkpoint_id")
    parsed = parse_checkpoint_ref(candidate) if isinstance(candidate, str) else None
    return candidate if parsed is not None and parsed[0] == status.run_id else None


def _deployment_projection(status: Any) -> dict[str, Any] | None:
    deployment = status.deployment if isinstance(status.deployment, dict) else None
    checkpoint_id = _checkpoint_id(status)
    if deployment is None or checkpoint_id is None:
        return None
    projected = {**deployment, "checkpoint_id": checkpoint_id}
    projected["endpoint"] = deployment.get("endpoint") or deployment.get("endpoint_name")
    return {
        key: value
        for key, value in projected.items()
        if key in _DEPLOYMENT_FIELDS and value is not None
    }


def _matching_persisted_status(status: Any) -> dict[str, Any] | None:
    """Load the exact durable snapshot represented by this ordered report."""
    sequence = getattr(status, "report_sequence", None)
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        return None
    try:
        from flash.runner.lifecycle.status import _load_status_json

        raw = _load_status_json(status.run_id)
    except Exception:
        return None
    if raw.get("run_id") != status.run_id or raw.get("report_sequence") != sequence:
        return None
    return raw


def _canonical_remote_attempt(remote: object) -> int | None:
    try:
        from flash.runner.accounting.reconciliation import _canonical_cleanup_remote

        canonical = _canonical_cleanup_remote(remote)
    except Exception:
        return None
    if canonical is None:
        return None
    attempt = canonical.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 0:
        return None
    return attempt


def _valid_attempt(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _validated_terminal_source(raw: dict[str, Any]) -> bool:
    verified_attempt = raw.get("source_verified_attempt")
    if (
        isinstance(verified_attempt, bool)
        or not isinstance(verified_attempt, int)
        or verified_attempt < 0
    ):
        return False
    try:
        from flash.snapshot.archive import parse_descriptor

        parse_descriptor(raw.get("source_snapshot"))
    except Exception:
        return False
    return True


def _lifecycle_projection(status: Any, public_status: dict[str, Any]) -> dict[str, bool]:
    lifecycle = {
        "started": False,
        "progressed": False,
        "artifactsComplete": False,
        "cleanupComplete": False,
    }
    raw = _matching_persisted_status(status)
    if raw is None:
        return lifecycle

    remote_attempt = _canonical_remote_attempt(raw.get("remote") or raw.get("realized_cost_remote"))
    started_attempt = _valid_attempt(raw.get("lifecycle_started_attempt"))
    if started_attempt is None:
        started_attempt = remote_attempt
    progressed_attempt = _valid_attempt(raw.get("lifecycle_progressed_attempt"))
    lifecycle["started"] = started_attempt is not None
    lifecycle["progressed"] = started_attempt is not None and progressed_attempt is not None

    adapter_ref = public_status.get("adapter_ref")
    artifacts_dir = raw.get("artifacts_dir")
    lifecycle["artifactsComplete"] = (
        raw.get("state") in {"done", "deployed"}
        and _validated_terminal_source(raw)
        and isinstance(adapter_ref, str)
        and bool(adapter_ref.strip())
        and isinstance(artifacts_dir, str)
        and bool(artifacts_dir.strip())
    )

    cleanup_remotes = raw.get("cleanup_remotes", [])
    lifecycle["cleanupComplete"] = (
        raw.get("state") in {"done", "failed", "cancelled", "dry_run", "deployed"}
        and raw.get("remote") is None
        and isinstance(cleanup_remotes, list)
        and not cleanup_remotes
    )
    return lifecycle


def record_training_run(*, status: Any, key: dict[str, Any] | None = None) -> bool:
    context = {**_context_from_status(status), **(key or {})}
    org_id = org_id_of(context)
    if not org_id:
        return False

    spec = _spec_from_status(status)
    try:
        project_id = require_project_id(spec.get("project"))
    except (TypeError, ValueError):
        return False
    public_status = status.to_dict()
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
        "gpuType": attributed_gpu_type(status) or None,
        "costUsd": status.cost_usd,
        "realizedCostUsd": status.realized_cost_usd,
        "checkpointId": _checkpoint_id(status),
        "artifactsDir": status.artifacts_dir,
        "error": status.error,
        "spec": spec,
        "deployment": _deployment_projection(status),
        "lastHeartbeat": public_status.get("last_heartbeat"),
        "gpuStatus": status.gpu_status,
        "lifecycle": _lifecycle_projection(status, public_status),
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
        from flash.engine.result.accounting import sanitize_worker_metrics
        from flash.runner.lifecycle.status import get_status

        metrics = sanitize_worker_metrics(metrics)
        status = get_status(spec.run_id)
    except Exception:
        return False
    context = _context_from_status(status)
    org_id = org_id_of(context)
    if not org_id:
        return False
    persisted_spec = _spec_from_status(status)
    try:
        project_id = require_project_id(persisted_spec.get("project"))
    except (TypeError, ValueError):
        return False
    checkpoint_id = format_checkpoint_ref(spec.run_id, None)
    body = {
        "orgId": org_id,
        "projectId": project_id,
        "runId": spec.run_id,
        "checkpointId": checkpoint_id,
        "phase": getattr(spec, "phase", None),
        "artifactPath": artifact_path,
        "metrics": metrics,
        "metadata": {"source": "flash.control_plane"},
        "updatedAt": _iso_from_epoch(getattr(status, "updated_at", None)),
    }
    return _post(_CHECKPOINT_PATH, body)
