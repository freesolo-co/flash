"""Run lifecycle endpoints: create, list, status, logs, worker output, cancel, checkpoints.

Service functions that the test-suite monkeypatches (``submit_job``, ``get_status``,
``list_checkpoints``, ``_worker_artifacts``) are resolved through the ``flash.server.app``
module (``_app.<name>``) at call time, so patching ``app.<name>`` is honored here.
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from flash.runner import cancel_run, new_run_id, runs_file_path
from flash.server import app as _app
from flash.server import db
from flash.server._deps import _parse_spec, _runtime_secrets, owned_run, require_key
from flash.spec import JobSpec

_LOG = logging.getLogger("flash.server.runs")

router = APIRouter()


@router.post("/v1/runs")
def create_run(payload: dict, key: Annotated[dict, Depends(require_key)]):
    spec = _parse_spec(payload, run_id=new_run_id())
    # Validate ``dry_run`` is an actual JSON boolean — never ``bool(...)`` a truthy non-bool
    # (e.g. the string "false" would coerce to True and silently flip a real run into dry-run).
    dry_run_raw = payload.get("dry_run", False)
    if not isinstance(dry_run_raw, bool):
        raise HTTPException(status_code=400, detail="dry_run must be a boolean")
    dry_run = dry_run_raw
    runtime_secrets = _runtime_secrets(payload, spec, require_environment_secrets=not dry_run)
    # External user-key runs are charged only after training succeeds. Persist the org id
    # (non-secret) so the background runner can bill with the operator internal key at
    # completion; never persist the submitting user's API key.
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
    try:
        db.record_run(spec.run_id, key["id"])
        submit_kwargs = {"dry_run": dry_run, "background": True}
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
    # Freesolo platform reporting is best-effort and runs AFTER the run is already submitted, so it
    # must NEVER roll back ownership or 400 the request — a reporting failure (import error /
    # unexpected runtime error; the network path already swallows internally) would otherwise
    # delete an already-submitted run and report failure to the caller. Swallow it instead.
    try:
        # submit_job already reports the freshly-created status to the backend via
        # _report_status -> record_training_run, and the status carries platform_context
        # (org_id/user_id/api_key_id derived from `key`), so a second explicit
        # record_training_run(status, key) here would just re-POST the same creation record.
        # Don't duplicate it.
        from flash.envs.adapter import is_managed_environment_slug
        from flash.server.environment_registry import record_environment_use

        if is_managed_environment_slug(spec.environment.id):
            record_environment_use(slug=spec.environment.id, run_id=spec.run_id, key=key)
    except Exception:
        # Best-effort: log to the structured server logger (not stdout) and never re-raise — the
        # run is already submitted, so a reporting failure must not 400 the caller.
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
            # A client-supplied/stale `offset` is only guaranteed valid as a cookie from a prior
            # `.tell()` on this exact file; an arbitrary one can raise ValueError/OSError on a text
            # stream. That's a bad-request, not a 500 — surface it as a clear 400.
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
    # The full train-subprocess stdout/traceback, pulled from the run's HF artifact repo with
    # the operator token — the real worker output the offset-paged .log can't carry. Kept off
    # the hot /logs poll path (it hits HF) so streaming `--follow` stays fast; `--logs` calls
    # this once. Best-effort: {} when nothing's been uploaded yet.
    status = owned_run(run_id, key)
    return {"run_id": run_id, "worker": _app._worker_artifacts(JobSpec.from_dict(status.spec))}


@router.post("/v1/runs/{run_id}/cancel")
def cancel(run_id: str, key: Annotated[dict, Depends(require_key)]):
    owned_run(run_id, key)
    return cancel_run(run_id).to_dict()


@router.get("/v1/runs/{run_id}/checkpoints")
def run_checkpoints(run_id: str, key: Annotated[dict, Depends(require_key)]):
    """List a run's deployable per-step RL checkpoints (each `flash deploy --step N`-able).

    Reads the snapshots the worker streamed to HF, and best-effort mirrors them to the
    backend store so a listing also persists them."""
    status = owned_run(run_id, key)
    spec = JobSpec.from_dict(status.spec)
    checkpoints = _app.list_checkpoints(spec)
    with contextlib.suppress(Exception):
        from flash.server.checkpoints import register_checkpoints_best_effort

        register_checkpoints_best_effort(status)
    return {"run_id": run_id, "checkpoints": checkpoints}
