"""idempotent reconciliation for durable interpolated-checkpoint activation intents."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any

from flash.serve.deploy import ServingError
from flash.serve.model_checkpoints import (
    checkpoint_registration_payload,
    checkpoint_request,
    validate_active_checkpoint_response,
)

_RETRY_BASE_SECONDS = 5.0
_RETRY_MAX_SECONDS = 15.0 * 60.0
_TRANSIENT_STATUSES = {202, 502, 503, 504}
_PERMANENT_STATUSES = {409, 422}


def retry_delay_seconds(deployment_token: str, attempt: int) -> float:
    exponent = max(0, int(attempt) - 1)
    base = min(_RETRY_MAX_SECONDS, _RETRY_BASE_SECONDS * (2**min(exponent, 16)))
    digest = hashlib.sha256(f"{deployment_token}:{attempt}".encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return min(_RETRY_MAX_SECONDS, base * (0.8 + 0.4 * unit))


def _response_json(response: Any) -> dict[str, Any]:
    try:
        body = response.json()
    except Exception as exc:
        raise ServingError("checkpoint serving returned invalid json") from exc
    if not isinstance(body, dict):
        raise ServingError("checkpoint serving returned a non-object response")
    return body


def _persist_retry(run_id: str, checkpoint: dict[str, Any], detail: str) -> None:
    from flash.runner import record_checkpoint_state

    attempt = max(1, int(checkpoint.get("activation_attempts") or 0))
    record_checkpoint_state(
        run_id,
        deployment_token=checkpoint["deployment_token"],
        activation_state="retry_wait",
        activation_error=detail[:1000],
        activation_next_retry_at=time.time()
        + retry_delay_seconds(checkpoint["deployment_token"], attempt),
    )


def _persist_failed(run_id: str, checkpoint: dict[str, Any], detail: str) -> None:
    from flash.runner import record_checkpoint_state

    record_checkpoint_state(
        run_id,
        deployment_token=checkpoint["deployment_token"],
        activation_state="failed",
        activation_error=detail[:1000],
    )


def _persist_active(run_id: str, checkpoint: dict[str, Any], body: dict[str, Any]) -> None:
    from flash.runner import record_checkpoint_state

    record_checkpoint_state(
        run_id,
        deployment_token=checkpoint["deployment_token"],
        activation_state="active",
        activated_at=time.time(),
    )
    reconcile_backend_mirrors(run_id)


def _authoritative_get(checkpoint: dict[str, Any]):
    return checkpoint_request(
        "GET",
        f"/model-checkpoints/{checkpoint['model_id']}",
        params={"expected_deployment_token": checkpoint["deployment_token"]},
        timeout=60.0,
    )


def reconcile_checkpoint(run_id: str) -> bool:
    from flash.runner import get_status, record_checkpoint_state

    status = get_status(run_id)
    checkpoint = status.checkpoint
    if status.state not in {"done", "deployed"} or not isinstance(checkpoint, dict):
        return False
    state = checkpoint.get("activation_state")
    if state == "active":
        if checkpoint.get("backend_mirror_state") != "synced":
            reconcile_backend_mirrors(run_id)
        return False
    if state not in {"pending", "activating", "retry_wait"}:
        return False
    if state == "retry_wait" and float(checkpoint.get("activation_next_retry_at") or 0) > time.time():
        return False
    token = checkpoint["deployment_token"]
    if not record_checkpoint_state(
        run_id,
        deployment_token=token,
        activation_state="activating",
        activation_error=None,
        activation_next_retry_at=None,
        increment_attempts=True,
    ):
        return False
    checkpoint = get_status(run_id).checkpoint
    try:
        response = _authoritative_get(checkpoint)
        if response.status_code == 200:
            body = validate_active_checkpoint_response(_response_json(response), checkpoint)
            _persist_active(run_id, checkpoint, body)
            return True
        if response.status_code not in {404, 409}:
            if response.status_code in _TRANSIENT_STATUSES or response.status_code >= 500:
                _persist_retry(run_id, checkpoint, f"GET HTTP {response.status_code}")
            else:
                _persist_failed(run_id, checkpoint, f"GET HTTP {response.status_code}")
            return False

        response = checkpoint_request(
            "POST",
            "/model-checkpoints",
            json=checkpoint_registration_payload(checkpoint),
        )
        if response.status_code in {200, 201}:
            validate_active_checkpoint_response(_response_json(response), checkpoint)
            readback = _authoritative_get(checkpoint)
            if readback.status_code != 200:
                detail = f"readback HTTP {readback.status_code}"
                if readback.status_code == 409:
                    _persist_failed(run_id, checkpoint, detail)
                elif (
                    readback.status_code == 404
                    or readback.status_code in _TRANSIENT_STATUSES
                    or readback.status_code >= 500
                ):
                    _persist_retry(run_id, checkpoint, detail)
                else:
                    _persist_failed(run_id, checkpoint, detail)
                return False
            body = validate_active_checkpoint_response(_response_json(readback), checkpoint)
            _persist_active(run_id, checkpoint, body)
            return True
        if response.status_code in _PERMANENT_STATUSES:
            _persist_failed(run_id, checkpoint, f"POST HTTP {response.status_code}: {response.text[:500]}")
            return False
        if response.status_code in _TRANSIENT_STATUSES or response.status_code >= 500:
            _persist_retry(run_id, checkpoint, f"POST HTTP {response.status_code}")
            return False
        _persist_failed(run_id, checkpoint, f"POST HTTP {response.status_code}: {response.text[:500]}")
        return False
    except ServingError as exc:
        if exc.status_code in _PERMANENT_STATUSES:
            _persist_failed(run_id, checkpoint, str(exc))
        else:
            _persist_retry(run_id, checkpoint, str(exc))
        return False
    except Exception as exc:
        _persist_retry(run_id, checkpoint, str(exc))
        return False


def deactivate_checkpoint(run_id: str) -> bool:
    from flash.runner import get_status, record_checkpoint_state

    status = get_status(run_id)
    checkpoint = status.checkpoint
    if not isinstance(checkpoint, dict):
        return False
    state = checkpoint.get("activation_state")
    if state == "disabled":
        return True
    if state != "active":
        raise ServingError(f"checkpoint is not active (state={state!r})", status_code=409)
    token = checkpoint["deployment_token"]
    response = checkpoint_request(
        "DELETE",
        f"/model-checkpoints/{checkpoint['model_id']}",
        params={"expected_deployment_token": token},
        timeout=60.0,
    )
    if response.status_code not in {200, 204}:
        raise ServingError(
            f"checkpoint deactivation failed with HTTP {response.status_code}: {response.text[:500]}",
            status_code=response.status_code,
        )
    readback = _authoritative_get(checkpoint)
    if readback.status_code not in {404, 409}:
        if readback.status_code == 200:
            validate_active_checkpoint_response(_response_json(readback), checkpoint)
        raise ServingError(
            f"checkpoint deactivation was not authoritatively confirmed (HTTP {readback.status_code})"
        )
    changed = record_checkpoint_state(
        run_id,
        deployment_token=token,
        activation_state="disabled",
        activation_error=None,
    )
    if changed:
        reconcile_backend_mirrors(run_id)
    return bool(changed)


def reconcile_backend_mirrors(run_id: str) -> bool:
    from flash.runner import JobSpec, artifacts_dir, get_status, record_checkpoint_mirror
    from flash.server.run_registry import record_training_checkpoint, record_training_run

    status = get_status(run_id)
    if not isinstance(status.checkpoint, dict):
        return False
    checkpoint = status.checkpoint
    snapshot = {
        "expected_deployment_token": checkpoint["deployment_token"],
        "expected_activation_state": checkpoint["activation_state"],
        "expected_activation_updated_at": checkpoint["activation_updated_at"],
    }
    try:
        spec = JobSpec.from_dict(status.spec)
        metrics_path = os.path.join(artifacts_dir(spec), "metrics.json")
        with open(metrics_path) as f:
            metrics = json.load(f)
        run_ok = record_training_run(status=status)
        metrics_ok = record_training_checkpoint(
            spec=spec, metrics=metrics, artifact_path=artifacts_dir(spec)
        )
        if not run_ok or not metrics_ok:
            raise RuntimeError("backend checkpoint mirror was not accepted")
    except Exception as exc:
        record_checkpoint_mirror(
            run_id,
            synced=False,
            error=str(exc)[:1000],
            **snapshot,
        )
        return False
    return record_checkpoint_mirror(
        run_id,
        synced=True,
        mirrored_at=time.time(),
        **snapshot,
    )


def reconcile_checkpoints_once() -> int:
    from flash.runner import list_runs

    reconciled = 0
    for status in list_runs():
        checkpoint = status.checkpoint
        if not isinstance(checkpoint, dict):
            continue
        if status.state not in {"done", "deployed"}:
            continue
        # backend mirror convergence is independent of serving activation convergence.
        if checkpoint.get("backend_mirror_state") != "synced":
            reconcile_backend_mirrors(status.run_id)
        if checkpoint.get("activation_state") == "active":
            continue
        if reconcile_checkpoint(status.run_id):
            reconciled += 1
    return reconciled


def request_checkpoint_reconciliation(run_id: str) -> None:
    threading.Thread(target=reconcile_checkpoint, args=(run_id,), daemon=True).start()
