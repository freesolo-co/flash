"""Deployment state serialization and persisted transition reporting."""

from __future__ import annotations

import os
import time

from flash.serve.contract.urls import public_deployment


def _deployment_state(deployment: dict, state: str, **fields) -> dict:
    return {**deployment, **fields, "state": state, "updated_at": time.time()}


def _public_deployment(deployment: dict) -> dict:
    out = public_deployment(deployment)
    run_id = out.get("run_id")
    out.update(
        {
            "run_id": run_id,
            "checkpoint_step": out.get("checkpoint_step"),
            "checkpoint_id": out.get("checkpoint_id"),
            "state": out.get("state"),
            "verified_at": out.get("verified_at"),
            "openai_model": out.get("openai_model") or out.get("checkpoint_id"),
        }
    )
    return out


def _enqueue_deployment_report(status) -> None:
    from flash.runner.lifecycle.reporting import _report_status, _report_status_async

    if os.environ.get("FLASH_DEPLOY_SYNC") == "1":
        _report_status(status)
    else:
        _report_status_async(status)


def _report_persisted_transition(previous, current, *, persisted: bool) -> None:
    if not persisted or (
        previous.state == current.state and previous.deployments == current.deployments
    ):
        return
    _enqueue_deployment_report(current)


def _deployment_failure_persisted(status, failed: dict) -> bool:
    return status.deployments.get(failed["checkpoint_id"]) == failed
