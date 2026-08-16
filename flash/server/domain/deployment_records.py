"""Shaping the persisted deployment record and deciding whether a write actually landed.

Two callers share this: the service (which queues and finishes a request) and the lifecycle
executor (which runs the rest of the attempt in the background). Keeping the shaping here means
there is exactly one definition of what a state transition looks like on disk.
"""

from __future__ import annotations

import time

from flash.serve.urls import public_deployment


def deployment_state(deployment: dict, state: str, **fields) -> dict:
    return {**deployment, **fields, "state": state, "updated_at": time.time()}


def public_deployment_view(deployment: dict) -> dict:
    """The record as an API caller may see it: internal bookkeeping stripped, identity filled in."""
    out = public_deployment(deployment)
    run_id = out.get("run_id")
    out.update(
        {
            "run_id": run_id,
            "checkpoint_step": out.get("checkpoint_step"),
            "adapter_revision": out.get("adapter_revision"),
            "state": out.get("state"),
            "verified_at": out.get("verified_at"),
            "openai_model": out.get("openai_model") or run_id,
        }
    )
    return out


def deployment_failure_persisted(status, failed: dict) -> bool:
    """Whether `failed` is what the store now holds, so the transition may be reported.

    An exact match is the simple case. The store may also have rebased the failure onto the
    predecessor it preserved, which is still this failure landing -- recognised by the predecessor's
    fields surviving intact alongside this attempt's error stamp.
    """
    if status.deployment == failed:
        return True
    previous = failed.get("previous_deployment")
    deployment = status.deployment
    failure_fields = {"last_deploy_error", "last_deploy_failed_at"}
    expected_error = failed.get("error") or "deployment failed"
    return bool(
        isinstance(previous, dict)
        and isinstance(deployment, dict)
        and all(
            deployment.get(key) == value
            for key, value in previous.items()
            if key not in failure_fields
        )
        and deployment.get("last_deploy_error") == expected_error
        and deployment.get("last_deploy_failed_at") is not None
    )


def deployment_attempt_is_stale(
    deployment: dict, stale_seconds: float, *, busy_states, now: float | None = None
) -> bool:
    """Whether a busy record is old enough that a new deploy may take it over.

    An unreadable timestamp counts as stale: a record nothing can date is one nothing can prove is
    still live, and leaving it busy would answer every retry with a conflict until it aged out.
    """
    if deployment.get("state") not in busy_states:
        return False
    raw = deployment.get("updated_at") or deployment.get("requested_at")
    try:
        stamp = float(raw)
    except (TypeError, ValueError):
        return True
    return (time.time() if now is None else now) - stamp >= stale_seconds
