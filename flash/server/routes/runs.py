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


def _precheck_budget_or_block(*, run_id: str, estimate_usd: float, org_id: str) -> None:
    """Reject an unaffordable prepared run before recording or allocating it."""
    from flash.server._internal_client import internal_key as _internal_key

    key = _internal_key()
    if not key:
        return
    try:
        from flash.server.billing import precheck_training_run

        precheck_training_run(internal_key=key, org_id=org_id, estimate_usd=estimate_usd)
    except Exception as exc:
        from flash.server.billing import BillingError

        if isinstance(exc, BillingError) and exc.status_code == 402:
            raise HTTPException(status_code=402, detail=exc.detail) from exc
        _LOG.warning("budget precheck skipped for %s (billing service error): %s", run_id, exc)


@router.post("/v1/runs")
def create_run(payload: dict, key: Annotated[dict, Depends(require_key)]):
    spec = _parse_spec(payload, run_id=new_run_id())
    dry_run = _require_bool(payload, "dry_run", False)
    runtime_secrets = _runtime_secrets(payload, spec, require_environment_secrets=not dry_run)
    affordability_org_id = str(key.get("org_id") or "").strip()
    bill_on_completion = not dry_run and key.get("auth_kind") != "internal"
    billing_context = None
    if bill_on_completion:
        if not affordability_org_id:
            raise HTTPException(
                status_code=400,
                detail="org id is required to bill a completed training run",
            )
        billing_context = {"org_id": affordability_org_id}
    platform_context = {
        field: value
        for field, value in {
            "org_id": key.get("org_id"),
            "user_id": key.get("user_id"),
            "api_key_id": key.get("api_key_id"),
        }.items()
        if value
    }
    run_id = spec.run_id
    try:
        try:
            prepared = _app.prepare_job(
                spec,
                billing_context=billing_context,
                platform_context=platform_context or None,
                owner_key_id=key["id"],
            )
        except Exception as exc:
            source_ref = spec.train.init_from_adapter
            if source_ref:
                _LOG.warning(
                    "warm-start preparation failed for %s from %s",
                    run_id,
                    source_ref,
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"train.init_from_adapter source {source_ref!r} could not be prepared; "
                        "verify that the source adapter is complete, compatible, and unchanged"
                    ),
                ) from exc
            raise
        run_id = prepared.public_spec.run_id
        if affordability_org_id:
            _precheck_budget_or_block(
                run_id=run_id,
                estimate_usd=prepared.estimated_cost_usd,
                org_id=affordability_org_id,
            )
        db.record_run(run_id, key["id"])
        submit_kwargs = {
            "dry_run": dry_run,
            "background": True,
            "owner_key_id": key["id"],
            "prepared_job": prepared,
        }
        if runtime_secrets:
            submit_kwargs["runtime_secrets"] = runtime_secrets
        if billing_context:
            submit_kwargs["billing_context"] = billing_context
        if platform_context:
            submit_kwargs["platform_context"] = platform_context
        status = _app.submit_job(prepared.public_spec, **submit_kwargs)
    except Exception as exc:
        db.delete_run(run_id)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        from flash.envs.adapter import is_managed_environment_slug
        from flash.server.environment_registry import record_environment_use

        if is_managed_environment_slug(prepared.public_spec.environment.id):
            record_environment_use(
                slug=prepared.public_spec.environment.id,
                run_id=run_id,
                key=key,
            )
    except Exception:
        _LOG.warning(
            "platform reporting failed for %s (run already submitted)", run_id, exc_info=True
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
