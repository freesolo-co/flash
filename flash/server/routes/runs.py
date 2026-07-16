"""Run lifecycle endpoints: create, list, status, logs, worker output, cancel, checkpoints."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from flash.runner import (
    DeploymentRevocationError,
    DeploymentStatePersistenceError,
    cancel_run,
    new_run_id,
    runs_file_path,
)
from flash.schema import train_schema_metadata
from flash.server import app as _app
from flash.server import db
from flash.server._deps import _parse_spec, _require_bool, _runtime_secrets, owned_run, require_key
from flash.spec import JobSpec

_LOG = logging.getLogger("flash.server.runs")

router = APIRouter()

_MAX_SCHEMA_FIELDS = 256
_MAX_SCHEMA_TEXT = 128


def _client_train_schema(payload: dict) -> dict | None:
    raw = payload.get("client_train_schema")
    if not isinstance(raw, dict) or set(raw) != {"version", "fields", "authored_keys"}:
        return None
    version = raw.get("version")
    fields = raw.get("fields")
    authored_keys = raw.get("authored_keys")
    if not isinstance(version, str) or not version or len(version) > _MAX_SCHEMA_TEXT:
        return None
    if not isinstance(fields, dict) or len(fields) > _MAX_SCHEMA_FIELDS:
        return None
    if not isinstance(authored_keys, list) or len(authored_keys) > _MAX_SCHEMA_FIELDS:
        return None
    if any(
        not isinstance(key, str)
        or not key
        or len(key) > _MAX_SCHEMA_TEXT
        or not isinstance(value, str)
        or not value
        or len(value) > _MAX_SCHEMA_TEXT
        for key, value in fields.items()
    ):
        return None
    if any(
        not isinstance(key, str) or not key or len(key) > _MAX_SCHEMA_TEXT for key in authored_keys
    ):
        return None
    if len(authored_keys) != len(set(authored_keys)) or not set(authored_keys) <= set(fields):
        return None
    server_fields = train_schema_metadata()
    shared = sorted(set(fields) & set(server_fields))
    return {
        "version": version,
        "fields": dict(fields),
        "authored_keys": tuple(authored_keys),
        "compatibility": {
            "status": "agreement" if fields == server_fields else "disagreement",
            "client_only": sorted(set(fields) - set(server_fields)),
            "server_only": sorted(set(server_fields) - set(fields)),
            "introduced_in_differences": [
                {"key": key, "client": fields[key], "server": server_fields[key]}
                for key in shared
                if fields[key] != server_fields[key]
            ],
        },
    }


def _precheck_budget_or_block(*, run_id: str, estimate_usd: float, org_id: str) -> None:
    """Reject an unaffordable prepared run before recording or allocating it."""
    from flash.server._internal_client import internal_key as _internal_key

    key = _internal_key()
    if not key:
        # internal reporting is off -> no completion billing either, so there is nothing to gate.
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
    dry_run = _require_bool(payload, "dry_run", False)
    schema = _client_train_schema(payload)
    submitted = payload.get("spec")
    submitted_train = submitted.get("train") if isinstance(submitted, dict) else None
    try:
        spec = _parse_spec(payload, run_id=new_run_id())
    except HTTPException as exc:
        server_fields = train_schema_metadata()
        unsupported = (
            sorted(
                name
                for name in schema["authored_keys"]
                if name in submitted_train and name not in server_fields
            )
            if exc.status_code == 400 and schema and isinstance(submitted_train, dict)
            else []
        )
        if unsupported:
            declared = ", ".join(
                f"{name} (minimum released Flash version {schema['fields'][name]})"
                for name in unsupported
            )
            client_only = ", ".join(schema["compatibility"]["client_only"]) or "none"
            detail = (
                f"{exc.detail}. Unsupported authored [train] key(s): {declared}; "
                f"client/server [train] schemas disagree (client Flash {schema['version']}; "
                f"client-only keys: {client_only})"
            )
            raise HTTPException(status_code=400, detail=detail) from exc
        raise
    runtime_secrets = _runtime_secrets(payload, spec)
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
        if (
            spec.train.init_from_adapter
            and schema is not None
            and "lora_alpha" in schema["authored_keys"]
            and spec.train.lora_alpha != prepared.worker_spec.train.lora_alpha
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"train.lora_alpha={spec.train.lora_alpha} does not match the "
                    "train.init_from_adapter source adapter "
                    f"lora_alpha={prepared.worker_spec.train.lora_alpha}; omit train.lora_alpha "
                    "because source adapter alpha metadata is authoritative"
                ),
            )
        run_id = prepared.public_spec.run_id
        if bill_on_completion:
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
    response = status.to_dict()
    if dry_run and schema is not None:
        response["train_schema_compatibility"] = schema["compatibility"]
    return response


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
    try:
        return cancel_run(run_id).to_dict()
    except DeploymentRevocationError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "code": "deployment_revocation_failed",
                "run_id": run_id,
                "retryable": True,
                "message": str(exc),
            },
        ) from exc
    except DeploymentStatePersistenceError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "deployment_state_persistence_failed",
                "run_id": run_id,
                "retryable": True,
                "backend_outcome": exc.backend_outcome,
                "message": str(exc),
            },
        ) from exc


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
