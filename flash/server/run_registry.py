"""Best-effort reporting of managed Flash runs/checkpoints to the Freesolo backend."""

from __future__ import annotations

import logging
import urllib.request
from datetime import UTC, datetime
from typing import Any

from flash.serve.model_checkpoints import INTERPOLATED_CHECKPOINT_INTENT_SCHEMA

from ._internal_client import org_id_of, post_internal_json

_LOG = logging.getLogger("flash.server.runs")
_RUN_PATH = "/api/flash/runs/internal"
_CHECKPOINT_PATH = "/api/flash/runs/checkpoints/internal"


def _checkpoint_accepted(body: Any) -> bool:
    return isinstance(body, dict) and body.get("checkpointAccepted") is True


def _iso_from_epoch(value: float | int | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=UTC).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _post(
    path: str,
    body: dict[str, Any],
    *,
    response_validator=None,
) -> bool:
    return post_internal_json(
        path,
        body,
        subject=f"report {path}",
        logger=_LOG,
        urlopen=urllib.request.urlopen,
        response_validator=response_validator,
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


def _checkpoint_from_status(status: Any) -> dict[str, Any] | None:
    checkpoint = getattr(status, "checkpoint", None)
    if not isinstance(checkpoint, dict):
        return None
    return {
        "schema": INTERPOLATED_CHECKPOINT_INTENT_SCHEMA,
        "version": checkpoint.get("version"),
        "modelId": checkpoint.get("model_id"),
        "baseModel": checkpoint.get("base_model"),
        "modelRepoId": checkpoint.get("model_repo_id"),
        "modelRevision": checkpoint.get("model_revision"),
        "tokenizerRepoId": checkpoint.get("tokenizer_repo_id"),
        "tokenizerRevision": checkpoint.get("tokenizer_revision"),
        "deploymentToken": checkpoint.get("deployment_token"),
        "payloadHash": checkpoint.get("payload_hash"),
        "outputFingerprint": checkpoint.get("output_fingerprint"),
        "interpolationOutputFingerprint": checkpoint.get(
            "interpolation_output_fingerprint"
        ),
        "activationState": checkpoint.get("activation_state"),
        "activationAttempts": checkpoint.get("activation_attempts"),
        "activationError": checkpoint.get("activation_error"),
        "activationUpdatedAt": checkpoint.get("activation_updated_at"),
        "activationNextRetryAt": checkpoint.get("activation_next_retry_at"),
        "activatedAt": checkpoint.get("activated_at"),
        "metadata": checkpoint.get("metadata") or {},
    }


def record_training_run(*, status: Any, key: dict[str, Any] | None = None) -> bool:
    context = {**_context_from_status(status), **(key or {})}
    org_id = org_id_of(context)
    if not org_id:
        return False

    spec = _spec_from_status(status)
    gpu = spec.get("gpu") if isinstance(spec.get("gpu"), dict) else {}
    checkpoint = _checkpoint_from_status(status)
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
        "checkpoint": checkpoint,
        "lastHeartbeat": status.last_heartbeat,
        "gpuStatus": status.gpu_status,
        "createdAt": _iso_from_epoch(status.created_at),
        "updatedAt": _iso_from_epoch(status.updated_at),
        "metadata": {"source": "flash.control_plane"},
    }
    return _post(
        _RUN_PATH,
        body,
        response_validator=_checkpoint_accepted if checkpoint is not None else None,
    )


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
    return _post(
        _CHECKPOINT_PATH,
        body,
        response_validator=_checkpoint_accepted,
    )
