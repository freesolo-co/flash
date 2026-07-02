"""Run lifecycle endpoints: create, list, status, logs, worker output, cancel, checkpoints."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from flash.runner import cancel_run, new_run_id, runs_file_path
from flash.server import app as _app
from flash.server import db
from flash.server._deps import _parse_spec, _require_bool, _runtime_secrets, owned_run, require_key
from flash.spec import JobSpec

_LOG = logging.getLogger("flash.server.runs")

router = APIRouter()


def _precheck_budget_or_block(spec: JobSpec, org_id: str) -> None:
    """Reject a run up front when its org can't afford the flash.cost estimate.

    Runs at submit, BEFORE any GPU is allocated, so a run that can't be billed never trains to
    completion only to fail the charge at the end. Blocks ONLY on a definitive 402 (insufficient
    balance / no billing record). Fails open on everything else -- internal reporting disabled, a
    cost-estimate error, or an unreachable/5xx billing service must not halt training over a
    transient blip (the completion charge is the backstop). Verify-only: no money moves here.
    """
    from flash.server._internal_client import internal_key as _internal_key

    key = _internal_key()
    if not key:
        # internal reporting is off -> no completion billing either, so there is nothing to gate.
        return
    try:
        from flash.cost.spec import estimate_for_spec

        estimate_usd = float(estimate_for_spec(spec).total_usd)
    except Exception:
        _LOG.warning("budget precheck skipped for %s: cost estimate failed", spec.run_id, exc_info=True)
        return
    try:
        from flash.server.billing import precheck_training_run

        precheck_training_run(internal_key=key, org_id=org_id, estimate_usd=estimate_usd)
    except Exception as exc:
        from flash.server.billing import BillingError

        if isinstance(exc, BillingError) and exc.status_code == 402:
            raise HTTPException(status_code=402, detail=exc.detail) from exc
        # backend unreachable / 5xx / unexpected -> fail open, never block training on infra noise.
        _LOG.warning("budget precheck skipped for %s (billing service error): %s", spec.run_id, exc)


@router.post("/v1/runs")
def create_run(payload: dict, key: Annotated[dict, Depends(require_key)]):
    spec = _parse_spec(payload, run_id=new_run_id())
    dry_run = _require_bool(payload, "dry_run", False)
    runtime_secrets = _runtime_secrets(payload, spec, require_environment_secrets=not dry_run)
    # Bill with operator key at completion; never persist the user's API key.
    bill_on_completion = not dry_run and key.get("auth_kind") != "internal"
    billing_context = None
    if bill_on_completion:
        org_id = str(key.get("org_id") or "").strip()
        if not org_id:
            raise HTTPException(
                status_code=400,
                detail="org id is required to bill a completed training run",
            )
        billing_context = {"org_id": org_id}
        # gate BEFORE recording/submitting: reject an unaffordable run before any GPU is allocated.
        _precheck_budget_or_block(spec, org_id)
    try:
        db.record_run(spec.run_id, key["id"])
        submit_kwargs = {"dry_run": dry_run, "background": True, "owner_key_id": key["id"]}
        if runtime_secrets:
            submit_kwargs["runtime_secrets"] = runtime_secrets
        if billing_context:
            submit_kwargs["billing_context"] = billing_context
        platform_context = {
            field: value
            for field, value in {
                "org_id": key.get("org_id"),
                "user_id": key.get("user_id"),
                "api_key_id": key.get("api_key_id"),
            }.items()
            if value
        }
        if platform_context:
            submit_kwargs["platform_context"] = platform_context
        status = _app.submit_job(spec, **submit_kwargs)
    except Exception as exc:
        db.delete_run(spec.run_id)  # idempotent: a no-op if record_run never landed
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Platform reporting is best-effort; must NEVER roll back an already-submitted run.
    try:
        from flash.envs.adapter import is_managed_environment_slug
        from flash.server.environment_registry import record_environment_use

        if is_managed_environment_slug(spec.environment.id):
            record_environment_use(slug=spec.environment.id, run_id=spec.run_id, key=key)
    except Exception:
        _LOG.warning(
            "platform reporting failed for %s (run already submitted)",
            spec.run_id,
            exc_info=True,
        )
    return status.to_dict()


@router.get("/v1/runs")
def list_runs(key: Annotated[dict, Depends(require_key)]):
    out = []
    for row in db.runs_for_key(key["id"]):
        try:
            out.append(_app.get_status(row["run_id"]).to_dict())
        except FileNotFoundError:
            continue
    return {"runs": out}


@router.get("/v1/runs/{run_id}")
def run_status(run_id: str, key: Annotated[dict, Depends(require_key)]):
    status = owned_run(run_id, key)
    return status.to_dict()


@router.get("/v1/runs/{run_id}/logs")
def run_logs(run_id: str, key: Annotated[dict, Depends(require_key)], offset: int = 0):
    status = owned_run(run_id, key)
    log_path = runs_file_path(run_id, ".log")
    chunk, end = "", max(0, offset)
    if os.path.exists(log_path):
        with open(log_path) as f:
            try:
                f.seek(end)
                chunk = f.read()
            except (ValueError, OSError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"invalid log offset {offset}: {exc}"
                ) from exc
            end = f.tell()
    return {
        "run_id": run_id,
        "logs": chunk,
        "offset": end,
        "state": status.state,
        "last_heartbeat": status.last_heartbeat,
        "gpu_status": status.gpu_status,
    }


@router.get("/v1/runs/{run_id}/worker")
def run_worker_output(run_id: str, key: Annotated[dict, Depends(require_key)]):
    status = owned_run(run_id, key)
    return {"run_id": run_id, "worker": _app._worker_artifacts(JobSpec.from_dict(status.spec))}


@router.post("/v1/runs/{run_id}/cancel")
def cancel(run_id: str, key: Annotated[dict, Depends(require_key)]):
    owned_run(run_id, key)
    return cancel_run(run_id).to_dict()


@router.get("/v1/runs/{run_id}/checkpoints")
def run_checkpoints(run_id: str, key: Annotated[dict, Depends(require_key)]):
    """List a run's deployable per-step RL checkpoints."""
    status = owned_run(run_id, key)
    spec = JobSpec.from_dict(status.spec)
    checkpoints = _app.list_checkpoints(spec)
    with contextlib.suppress(Exception):
        from flash.server.checkpoints import register_checkpoints_best_effort

        register_checkpoints_best_effort(status)
    return {"run_id": run_id, "checkpoints": checkpoints}
