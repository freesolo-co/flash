"""Run lifecycle endpoints: create, list, status, logs, worker output, cancel, checkpoints."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

import flash.runner as _runner
from flash.content.multimodal import preflight_validate_image_opd
from flash.runner import (
    DeploymentRevocationError,
    DeploymentStatePersistenceError,
    cancel_run,
    new_run_id,
    runs_file_path,
)
from flash.schema import train_schema_metadata
from flash.serve.preflight import ServingPreflightError
from flash.server import app as _app
from flash.server.platform import db
from flash.server.platform.deps import (
    _parse_spec,
    _require_bool,
    _runtime_secrets,
    owned_run,
    readable_run,
    require_key,
)

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


def _schema_disagreement_detail(
    exc: HTTPException, schema: dict | None, submitted_train: object
) -> str | None:
    """Explain a 400 as a client/server ``[train]`` schema disagreement, or ``None`` if it is not one.

    A spec rejected for naming a key this server has never heard of is an out-of-date server, not a
    malformed request, and the bare parse error would send the user hunting a typo that isn't there.
    Only keys the CLIENT says it authored count: a key the client did not send cannot be the reason
    its spec failed to parse.
    """
    if exc.status_code != 400 or not schema or not isinstance(submitted_train, dict):
        return None
    server_fields = train_schema_metadata()
    unsupported = sorted(
        name
        for name in schema["authored_keys"]
        if name in submitted_train and name not in server_fields
    )
    if not unsupported:
        return None
    declared = ", ".join(
        f"{name} (minimum released Flash version {schema['fields'][name]})" for name in unsupported
    )
    client_only = ", ".join(schema["compatibility"]["client_only"]) or "none"
    return (
        f"{exc.detail}. Unsupported authored [train] key(s): {declared}; "
        f"client/server [train] schemas disagree (client Flash {schema['version']}; "
        f"client-only keys: {client_only})"
    )


def _precheck_budget_or_block(*, run_id: str, estimate_usd: float, org_id: str) -> bool:
    """Reject an unaffordable prepared run before recording or allocating it.

    Returns whether affordability was actually VERIFIED. The two fail-open paths below deliberately
    let the run through on a billing-infra problem, but a caller that reports the outcome must be
    able to tell that apart from a real pass: an unverified run can still be rejected 402 later.
    """
    from flash.server.platform.internal_client import internal_key as _internal_key

    key = _internal_key()
    if not key:
        # internal reporting is off -> no completion billing either, so there is nothing to gate.
        return False
    try:
        from flash.server.billing.charges import precheck_training_run

        precheck_training_run(internal_key=key, org_id=org_id, estimate_usd=estimate_usd)
    except Exception as exc:
        from flash.server.billing.charges import BillingError

        if isinstance(exc, BillingError) and exc.status_code == 402:
            raise HTTPException(status_code=402, detail=exc.detail) from exc
        _LOG.warning("budget precheck skipped for %s (billing service error): %s", run_id, exc)
        return False
    return True


@router.post("/v1/runs")
def create_run(
    payload: dict,
    key: Annotated[dict, Depends(require_key)],
    authorization: Annotated[str | None, Header()] = None,
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
):
    dry_run = _require_bool(payload, "dry_run", False)
    schema = _client_train_schema(payload)
    submitted = payload.get("spec")
    submitted_train = submitted.get("train") if isinstance(submitted, dict) else None
    try:
        spec = _parse_spec(payload, run_id=new_run_id())
    except HTTPException as exc:
        detail = _schema_disagreement_detail(exc, schema, submitted_train)
        if detail is None:
            raise
        raise HTTPException(status_code=400, detail=detail) from exc
    from flash.server.domain.projects import require_project_access

    project_id = require_project_access(
        project_id=spec.project,
        key=key,
        authorization=authorization,
        org_id=x_freesolo_org_id,
    )
    reporting_key = {**key, "org_id": key.get("org_id") or x_freesolo_org_id}
    from flash.envs.adapter import canonical_managed_environment_slug

    try:
        environment_slug = canonical_managed_environment_slug(spec.environment.id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if environment_slug is not None:
        from flash.server.domain import envs as managed_envs
        from flash.server.domain.environment_registry import require_environment_project

        try:
            environment_slug = managed_envs.canonical_env_id(environment_slug)
        except managed_envs.EnvPublishError as exc:
            raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
        require_environment_project(
            slug=environment_slug,
            project_id=project_id,
            key=reporting_key,
            repair_missing=True,
        )
    runtime_secrets = _runtime_secrets(payload, spec)
    affordability_org_id = str(key.get("org_id") or "").strip()
    billable_key = key.get("auth_kind") != "internal"
    bill_on_completion = not dry_run and billable_key
    billing_context = None
    # the org requirement is a property of the KEY, not of the mode: a dry run whose org cannot be
    # resolved would otherwise skip the affordability check and answer 200, while submitting the very
    # same spec for real is rejected 400 here -- the preview contradicting the launch it previews.
    if billable_key and not affordability_org_id:
        raise HTTPException(
            status_code=400,
            detail="org id is required to bill a completed training run",
        )
    # a workload profile is a separate job that really runs, so it bills on its own completion
    # whenever the key is billable -- including under `--dry-run`. dry-run previews the TRAINING
    # run; it does not make work that actually executed free.
    profile_billing_context = {"org_id": affordability_org_id} if billable_key else None
    if bill_on_completion:
        billing_context = profile_billing_context
    platform_context = {
        field: value
        for field, value in {
            "org_id": key.get("org_id") or x_freesolo_org_id,
            "user_id": key.get("user_id"),
            "api_key_id": key.get("api_key_id"),
            "project_id": project_id,
        }.items()
        if value
    }
    run_id = spec.run_id
    affordability_verified = False
    try:
        try:
            prepared = _app.prepare_job(
                spec,
                billing_context=billing_context,
                platform_context=platform_context or None,
                owner_key_id=key["id"],
            )
        except ServingPreflightError:
            raise
        # resolve the class off the module rather than binding it at import: flash.runner is
        # reloaded (tests, and any reimport), which rebinds the class to a new object, and a
        # bound name would silently stop matching the error the runner actually raises.
        except _runner.WarmStartPreparationError as exc:
            # only failures raised while resolving the warm-start source reach here, so the adapter
            # really is the cause. the reason itself stays out of the response on purpose: it can
            # name internal storage paths and source-run internals. it is logged instead.
            source_ref = spec.train.init_from_adapter
            _LOG.warning(
                "warm-start preparation failed for %s from %s", run_id, source_ref, exc_info=True
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"train.init_from_adapter source {source_ref!r} could not be prepared; "
                    "verify that the source adapter is complete, compatible, and unchanged"
                ),
            ) from exc
        except _runner.WorkloadProfilePending as exc:
            pending = exc.prepared_job
            state = exc.state
            # whether THIS request launched and billed the profile, as opposed to joining one that
            # was already running. only a winning claim launches, so anything else must not be told
            # it was charged. defaults false: a path that never reaches the claim launched nothing.
            launched = False
            if isinstance(pending, _runner.PreparedJob):
                profile_run_id = pending.public_spec.run_id
                # claim before spending anything on it. the id is deterministic in the workload, so
                # another key may already own this exact profile -- in which case it is already
                # running and this submitter neither launches it again nor pays for it a second
                # time. losing the claim is ordinary reuse, not an error; it just means waiting.
                spent_at = getattr(exc, "spent_at", None)
                if spent_at is None:
                    claimed = db.claim_profile_run(profile_run_id, key["id"])
                else:
                    # the id is already claimed by a run that is spent, so a fresh claim cannot be
                    # inserted. take the existing one over against the spent run's own timestamp,
                    # so of several submitters watching the same dead profile exactly one relaunches
                    # it and the rest are told to wait on the one that is now running.
                    claimed = db.reclaim_spent_profile_run(
                        profile_run_id, key["id"], spent_at=spent_at
                    )
                if claimed:
                    profile_submit_kwargs = {
                        "background": True,
                        "owner_key_id": key["id"],
                        "prepared_job": pending,
                    }
                    if runtime_secrets:
                        profile_submit_kwargs["runtime_secrets"] = runtime_secrets
                    if profile_billing_context:
                        profile_submit_kwargs["billing_context"] = profile_billing_context
                    if platform_context:
                        profile_submit_kwargs["platform_context"] = platform_context
                    try:
                        if billable_key:
                            _precheck_budget_or_block(
                                run_id=profile_run_id,
                                estimate_usd=pending.estimated_cost_usd,
                                org_id=affordability_org_id,
                            )
                        _app.submit_job(pending.public_spec, **profile_submit_kwargs)
                    except Exception:
                        # delete the deterministic claim when launch creates no run, or the id wedges
                        # forever. if ``submit_job`` already persisted status, keep the claim so
                        # ownership and normal lifecycle cleanup remain intact.
                        if not os.path.exists(runs_file_path(profile_run_id, ".json")):
                            db.delete_run(profile_run_id)
                        raise
                    # set only after submit_job returns: a launch that raised deleted the row and
                    # charged nothing, so reporting it as launched would name a charge that was
                    # rolled back.
                    launched = True
                # launched here, lost the claim to a submitter who launched it, or
                # lost a takeover to one who is relaunching it. in every case the
                # id now belongs to a live attempt, so the state has to name that
                # attempt: reporting the spent state a takeover loser read would
                # name a run that no longer exists under this id, and leaving a
                # plain claim loser on the synthetic "required" tells a user whose
                # profile is queued and running that it has not started --
                # `required` is a marker this route invents, not a state any run
                # is ever in.
                state = "queued"
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "workload_profile_pending",
                    "profile_run_id": exc.profile_run_id,
                    "state": state,
                    # whether THIS key can read that run. the id is deterministic, so a submitter
                    # can be waiting on a profile another key launched -- telling them to poll a
                    # run id that answers 404 for them would read as the server inventing an id.
                    "owned": db.run_owner(exc.profile_run_id) == key["id"],
                    # whether this request started and billed the profile. an owner polling a
                    # profile it launched earlier joins rather than launches, and must not be told
                    # it was charged again.
                    "launched": launched,
                },
            ) from exc
        run_id = prepared.public_spec.run_id
        # validate the spec BEFORE charging affordability against it. submit_job runs these same
        # read-only gates, but it runs them after this point, so an unsupported spec would be told
        # "insufficient balance" (402) for a run it can never launch at any balance -- sending the
        # user to top up instead of to the real defect. both are pure and raise ValueError, which the
        # handler below turns into the 400 submit_job would have produced.
        preflight_validate_image_opd(prepared.worker_spec)
        # run the affordability check for dry runs too. it is verify-only (moves no money), so a
        # `--dry-run` that passes now also proves the org can cover the estimate, instead of the run
        # being validated here and rejected 402 only on real submission.
        if bill_on_completion or (dry_run and billable_key):
            affordability_verified = _precheck_budget_or_block(
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
        from flash.server.domain.environment_registry import record_environment_use

        if environment_slug is not None:
            record_environment_use(
                slug=environment_slug,
                project_id=project_id,
                run_id=run_id,
                key=reporting_key,
            )
    except Exception:
        _LOG.warning(
            "platform reporting failed for %s (run already submitted)", run_id, exc_info=True
        )
    response = status.to_dict()
    if dry_run and schema is not None:
        response["train_schema_compatibility"] = schema["compatibility"]
    if dry_run:
        # say whether the affordability check actually RAN. it fails open on a billing-infra
        # problem, so a silent 200 would claim the dry run validated cost when it did not -- and
        # the identical spec can still be rejected 402 once the backend recovers.
        response["affordability_verified"] = affordability_verified
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
def run_logs(
    run_id: str,
    key: Annotated[dict, Depends(require_key)],
    offset: int = 0,
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
):
    status = readable_run(run_id, key, x_freesolo_org_id)
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
def run_worker_output(
    run_id: str,
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
):
    status = readable_run(run_id, key, x_freesolo_org_id)
    # hf_repo + run_id (adapter prefix) are platform-managed and stripped from the public spec;
    # the worker artifact repo lives under the internal carrier (see _internal_spec_from_status).
    from flash.runner import _internal_spec_from_status

    return {
        "run_id": run_id,
        "worker": _app._worker_artifacts(_internal_spec_from_status(status)),
    }


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
    # checkpoint listing keys off hf_repo + run_id, both platform-managed and stripped from the
    # public spec; resolve them from the internal carrier (see _internal_spec_from_status).
    from flash.runner import _internal_spec_from_status

    spec = _internal_spec_from_status(status)
    checkpoints = _app.list_checkpoints(spec)
    with contextlib.suppress(Exception):
        from flash.server.domain.checkpoints import register_checkpoints_best_effort

        register_checkpoints_best_effort(status)
    return {"run_id": run_id, "checkpoints": checkpoints}
