"""Run lifecycle endpoints: create, list, status, logs, worker output, cancel, checkpoints."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from flash.engine.profiling.image_tokens import ImageGeometryUnavailable
from flash.runner.lifecycle import status as runner_status
from flash.runner.lifecycle.preparation import WarmStartPreparationError
from flash.runner.lifecycle.state import new_run_id, runs_file_path
from flash.runner.lifecycle.submit import SourceSnapshotPublicationError
from flash.runner.supervise.deploy import (
    DeploymentRevocationError,
    DeploymentStatePersistenceError,
    cancel_run,
)
from flash.schema import train_schema_metadata
from flash.serve.control._canonical import canonical_mapping_fingerprint
from flash.serve.deployment.preflight import ServingPreflightError
from flash.server.asgi import app as _app
from flash.server.domain.teacher.broker import TeacherBrokerConfigurationError
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
_IDEMPOTENCY_KEY_RE = re.compile(r"^[A-Za-z0-9._~-]{16,128}$")


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


def _require_idempotency_key(value: str | None) -> str:
    if not isinstance(value, str) or _IDEMPOTENCY_KEY_RE.fullmatch(value) is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Idempotency-Key must be 16 to 128 characters using only "
                "A-Z, a-z, 0-9, period, underscore, tilde, or hyphen"
            ),
        )
    return value


def _effective_org_id(key: dict, org_id_header: str | None) -> str:
    return str(key.get("org_id") or org_id_header or "").strip()


def _request_fingerprint(
    *,
    spec,
    dry_run: bool,
    schema: dict | None,
    effective_org_id: str,
    runtime_secrets: dict[str, str],
) -> str:
    accepted_schema = None
    if schema is not None:
        accepted_schema = {
            "version": schema["version"],
            "fields": schema["fields"],
            "authored_keys": sorted(schema["authored_keys"]),
        }
    return canonical_mapping_fingerprint(
        {
            "spec": spec.to_dict(),
            "dry_run": dry_run,
            "client_train_schema": accepted_schema,
            "effective_org_id": effective_org_id,
            "runtime_secrets": runtime_secrets,
        }
    )


def _submission_lock_name(key_id: int, idempotency_key: str) -> str:
    payload = f"flash-run-submission-v1\0{key_id}\0{idempotency_key}".encode()
    return hashlib.sha256(payload).hexdigest()


def _submitted_instance_providers() -> tuple[str, ...]:
    from flash.providers.core.registry import INSTANCE_PROVIDERS, available_providers

    return tuple(sorted(name for name in available_providers() if name in INSTANCE_PROVIDERS))


def _idempotency_error(
    code: str, *, retryable: bool, run_id: str | None = None, reason: str | None = None
):
    detail = {"code": code, "retryable": retryable}
    if run_id is not None:
        detail["run_id"] = run_id
    if reason is not None:
        detail["reason"] = reason
    return HTTPException(status_code=409, detail=detail)


def _cleanup_failure_http_error(run_id: str) -> HTTPException:
    return HTTPException(
        status_code=500,
        detail={
            "code": "submission_cleanup_failed",
            "retryable": False,
            "run_id": run_id,
        },
    )


def _dispose_unrecoverable_secret_claim(run_id: str) -> None:
    disposal_failed = False
    try:
        db.dispose_run_submission(run_id, reason="runtime_secrets_unrecoverable")
    except Exception:
        disposal_failed = True
    if disposal_failed:
        raise _cleanup_failure_http_error(run_id) from None
    raise _idempotency_error(
        "submission_disposed",
        run_id=run_id,
        retryable=False,
        reason="runtime_secrets_unrecoverable",
    )


def _replay_submission(
    claim: dict, request_fingerprint: str, *, schema: dict | None
) -> dict | None:
    run_id = claim["run_id"]
    if claim["request_fingerprint"] != request_fingerprint:
        raise _idempotency_error("idempotency_key_reused", run_id=run_id, retryable=False)
    if claim["phase"] == "disposed":
        raise _idempotency_error(
            "submission_disposed",
            run_id=run_id,
            retryable=False,
            reason=claim.get("disposed_reason") or "submission_disposed",
        )
    if claim["phase"] == "claimed":
        if claim["had_runtime_secrets"]:
            _dispose_unrecoverable_secret_claim(run_id)
        # no durable dispatch fence exists. even if the authoritative queued status landed, no
        # supervisor was allowed to start, so resume the same run id instead of replaying inert work.
        return None
    try:
        status = _app.get_status(run_id)
    except FileNotFoundError:
        if claim["phase"] == "bound":
            if claim["had_runtime_secrets"]:
                _dispose_unrecoverable_secret_claim(run_id)
            raise _idempotency_error(
                "submission_state_unavailable", run_id=run_id, retryable=False
            ) from None
        return None
    response = status.to_dict()
    if claim["dry_run"]:
        verified = claim.get("affordability_verified")
        if verified is not None:
            response["affordability_verified"] = bool(verified)
        if schema is not None:
            response["train_schema_compatibility"] = schema["compatibility"]
    return response


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


@dataclass(frozen=True)
class _SubmissionContext:
    """Billing and platform attribution derived from the key and the requested mode."""

    affordability_org_id: str
    billable_key: bool
    bill_on_completion: bool
    billing_context: dict | None
    platform_context: dict


def _submission_context(
    *, key: dict, dry_run: bool, project_id: str, org_id_header: str | None
) -> _SubmissionContext:
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
    if bill_on_completion:
        billing_context = {"org_id": affordability_org_id}
    platform_context = {
        field: value
        for field, value in {
            "org_id": key.get("org_id") or org_id_header,
            "user_id": key.get("user_id"),
            "api_key_id": key.get("api_key_id"),
            "project_id": project_id,
        }.items()
        if value
    }
    return _SubmissionContext(
        affordability_org_id=affordability_org_id,
        billable_key=billable_key,
        bill_on_completion=bill_on_completion,
        billing_context=billing_context,
        platform_context=platform_context,
    )


def _resolve_managed_environment(spec, *, project_id: str, reporting_key: dict) -> str | None:
    """Return the canonical managed environment slug for ``spec``, or ``None`` when unmanaged."""
    from flash.envs.loading.adapter import canonical_managed_environment_slug

    try:
        environment_slug = canonical_managed_environment_slug(spec.environment.id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if environment_slug is None:
        return None
    from flash.server.domain.registry import envs as managed_envs
    from flash.server.domain.registry.environment_registry import require_environment_project

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
    return environment_slug


def _record_environment_use(
    environment_slug: str | None, *, project_id: str, run_id: str, reporting_key: dict
) -> None:
    """Report the environment use for an already-submitted run; never fails the request."""
    try:
        from flash.server.domain.registry.environment_registry import record_environment_use

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


def _redact_runtime_secret_text(value: str, secret_values: tuple[str, ...]) -> str:
    """Drop a contaminated diagnostic while preserving text that contains no request secret."""
    if not any(secret and secret in value for secret in secret_values):
        return value
    fallback = "submission failed"
    return "" if any(secret and secret in fallback for secret in secret_values) else fallback


def _redact_runtime_secret_detail(value, secret_values: tuple[str, ...]):
    """Redact strings recursively while preserving structured HTTP error details."""
    if isinstance(value, str):
        return _redact_runtime_secret_text(value, secret_values)
    if isinstance(value, dict):
        return {
            _redact_runtime_secret_detail(key, secret_values): _redact_runtime_secret_detail(
                item, secret_values
            )
            for key, item in value.items()
        }
    if isinstance(value, list | tuple):
        return type(value)(_redact_runtime_secret_detail(item, secret_values) for item in value)
    return value


def _redact_runtime_secret_headers(
    headers: Mapping[str, str] | None, secret_values: tuple[str, ...]
) -> dict[str, str] | None:
    """Preserve safe downstream headers and omit each contaminated header completely."""
    if headers is None:
        return None
    return {
        name: value
        for name, value in headers.items()
        if not any(secret and (secret in name or secret in value) for secret in secret_values)
    }


def _submit_failure_http_error(
    exc: Exception, *, runtime_secret_values: tuple[str, ...] = ()
) -> HTTPException:
    """Classify a failed submission without exposing request-only credentials.

    Everything reaching here is a bad request by default. Submit-time errors may opt into 503 with
    a truthy ``plane_fault`` attribute when the submitter cannot fix the failure by changing the spec.
    """
    if isinstance(exc, HTTPException):
        return HTTPException(
            status_code=exc.status_code,
            detail=_redact_runtime_secret_detail(exc.detail, runtime_secret_values),
            headers=_redact_runtime_secret_headers(exc.headers, runtime_secret_values),
        )
    detail = _redact_runtime_secret_text(str(exc), runtime_secret_values)
    if isinstance(exc, SourceSnapshotPublicationError):
        return HTTPException(status_code=503, detail=detail)
    if (
        isinstance(exc, (ImageGeometryUnavailable, TeacherBrokerConfigurationError))
        and exc.plane_fault
    ):
        return HTTPException(status_code=503, detail=detail)
    return HTTPException(status_code=400, detail=detail)


def _dispose_failed_submission(
    run_id: str,
    *,
    dry_run: bool,
    runtime_secret_values: tuple[str, ...],
    environment_slug: str | None,
    project_id: str,
    reporting_key: dict,
) -> None:
    """Settle a failed submission while preserving replay and recovery invariants."""
    if not os.path.exists(runs_file_path(run_id, ".json")):
        db.remove_run_submission_claim(run_id)
    elif dry_run:
        # a half-persisted dry run must remain invisible to recovery, but its idempotency key stays
        # consumed as a tombstone so replay can never turn the same request into paid work.
        db.dispose_run_submission(run_id, reason="dry_run_status_incomplete")
    elif runtime_secret_values:
        claim = db.run_submission_claim_for_run(run_id)
        if claim is None or claim["phase"] != "bound":
            # the durable dispatch fence never landed, so no supervisor was allowed to start. the
            # request-only secrets cannot be resumed and this remains a pre-dispatch tombstone.
            db.dispose_run_submission(run_id, reason="runtime_secrets_unrecoverable")
            return
        # the dispatch fence landed before the submitter failed. keep the owner and bound claim so
        # replay and startup recovery reconcile or terminate it rather than disposing dispatched work
        # as an abandoned pre-submission claim.
        try:
            runner_status._update(
                run_id,
                "failed",
                error=(
                    "submission failed after its runtime secrets were dispatched; "
                    "the run was stopped because recovery cannot restore them - resubmit"
                ),
            )
        except Exception:
            raise RuntimeError("could not terminalize dispatched secretful submission") from None
    else:
        # a retained non-secret run can recover safely. its phase already records whether the durable
        # dispatch fence landed, so cleanup must not promote a pre-fence claim to bound.
        _record_environment_use(
            environment_slug, project_id=project_id, run_id=run_id, reporting_key=reporting_key
        )


def _parse_submission_spec(payload: dict, *, run_id: str, schema: dict | None):
    submitted = payload.get("spec")
    submitted_train = submitted.get("train") if isinstance(submitted, dict) else None
    try:
        return _parse_spec(payload, run_id=run_id)
    except HTTPException as exc:
        detail = _schema_disagreement_detail(exc, schema, submitted_train)
        if detail is None:
            raise
        raise HTTPException(status_code=400, detail=detail) from exc


def _prepare_submission(spec, *, ctx: _SubmissionContext, key: dict):
    try:
        return _app.prepare_job(
            spec,
            billing_context=ctx.billing_context,
            platform_context=ctx.platform_context or None,
            owner_key_id=key["id"],
        )
    except ServingPreflightError:
        raise
    # resolve the class off the module because runner reloads rebind the exception class.
    except WarmStartPreparationError as exc:
        source_ref = spec.train.init_from_adapter
        _LOG.warning("warm-start preparation failed for %s (%s)", spec.run_id, type(exc).__name__)
        raise HTTPException(
            status_code=400,
            detail=(
                f"train.init_from_adapter source {source_ref!r} could not be prepared; "
                "verify that the source adapter is complete, compatible, and unchanged"
            ),
        ) from exc


def _submit_claimed_run(
    *,
    spec,
    runtime_secrets: dict[str, str],
    dry_run: bool,
    schema: dict | None,
    key: dict,
    authorization: str | None,
    org_id_header: str | None,
) -> dict:
    from flash.server.domain.registry.projects import require_project_access

    project_id = ""
    reporting_key = {**key, "org_id": key.get("org_id") or org_id_header}
    environment_slug = None
    affordability_verified = False
    submission_error = None
    runtime_secret_values: tuple[str, ...] = ()
    try:
        project_id = require_project_access(
            project_id=spec.project,
            key=key,
            authorization=authorization,
            org_id=org_id_header,
        )
        environment_slug = _resolve_managed_environment(
            spec, project_id=project_id, reporting_key=reporting_key
        )
        ctx = _submission_context(
            key=key, dry_run=dry_run, project_id=project_id, org_id_header=org_id_header
        )
        prepared = _prepare_submission(spec, ctx=ctx, key=key)
        if ctx.bill_on_completion or (dry_run and ctx.billable_key):
            affordability_verified = _precheck_budget_or_block(
                run_id=spec.run_id,
                estimate_usd=prepared.estimated_cost_usd,
                org_id=ctx.affordability_org_id,
            )
        submit_kwargs = {
            "dry_run": dry_run,
            "background": True,
            "owner_key_id": key["id"],
            "prepared_job": prepared,
            "status_persisted_fence": lambda: db.bind_run_submission(
                spec.run_id,
                affordability_verified=affordability_verified if dry_run else None,
            ),
        }
        if runtime_secrets:
            submit_kwargs["runtime_secrets"] = runtime_secrets
        if ctx.billing_context:
            submit_kwargs["billing_context"] = ctx.billing_context
        if ctx.platform_context:
            submit_kwargs["platform_context"] = ctx.platform_context
        status = _app.submit_job(prepared.public_spec, **submit_kwargs)
    except Exception as exc:
        runtime_secret_values = tuple(runtime_secrets.values())
        submission_error = _submit_failure_http_error(
            exc, runtime_secret_values=runtime_secret_values
        )
    if submission_error is not None:
        cleanup_error = None
        try:
            _dispose_failed_submission(
                spec.run_id,
                dry_run=dry_run,
                runtime_secret_values=runtime_secret_values,
                environment_slug=environment_slug,
                project_id=project_id,
                reporting_key=reporting_key,
            )
        except Exception as exc:
            cleanup_error = _submit_failure_http_error(
                exc, runtime_secret_values=runtime_secret_values
            )
        if cleanup_error is not None:
            raise _cleanup_failure_http_error(spec.run_id) from None
        raise submission_error from None
    _record_environment_use(
        environment_slug, project_id=project_id, run_id=spec.run_id, reporting_key=reporting_key
    )
    response = status.to_dict()
    if dry_run and schema is not None:
        response["train_schema_compatibility"] = schema["compatibility"]
    if dry_run:
        response["affordability_verified"] = affordability_verified
    return response


@router.post("/v1/runs")
def create_run(
    payload: dict,
    key: Annotated[dict, Depends(require_key)],
    authorization: Annotated[str | None, Header()] = None,
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
):
    idempotency_key = _require_idempotency_key(idempotency_key)
    dry_run = _require_bool(payload, "dry_run", False)
    schema = _client_train_schema(payload)
    spec = _parse_submission_spec(payload, run_id=new_run_id(), schema=schema)
    runtime_secrets = _runtime_secrets(payload, spec)
    fingerprint = _request_fingerprint(
        spec=spec,
        dry_run=dry_run,
        schema=schema,
        effective_org_id=_effective_org_id(key, x_freesolo_org_id),
        runtime_secrets=runtime_secrets,
    )
    from flash.server.platform.locks import submission_lock

    lock = submission_lock(_submission_lock_name(key["id"], idempotency_key))
    if not lock.acquire(blocking=False):
        raise _idempotency_error("submission_in_progress", retryable=True)
    try:
        claim = db.run_submission_claim(key["id"], idempotency_key)
        if claim is not None:
            replay = _replay_submission(claim, fingerprint, schema=schema)
            if replay is not None:
                return replay
            spec = _parse_submission_spec(payload, run_id=claim["run_id"], schema=schema)
        else:
            claim_error = None
            try:
                db.claim_run_submission(
                    run_id=spec.run_id,
                    key_id=key["id"],
                    idempotency_key=idempotency_key,
                    request_fingerprint=fingerprint,
                    dry_run=dry_run,
                    had_runtime_secrets=bool(runtime_secrets),
                    submitted_instance_providers=_submitted_instance_providers(),
                )
            except Exception as exc:
                claim_error = _submit_failure_http_error(
                    exc, runtime_secret_values=tuple(runtime_secrets.values())
                )
            if claim_error is not None:
                raise claim_error
        return _submit_claimed_run(
            spec=spec,
            runtime_secrets=runtime_secrets,
            dry_run=dry_run,
            schema=schema,
            key=key,
            authorization=authorization,
            org_id_header=x_freesolo_org_id,
        )
    finally:
        lock.release()


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
    from flash.runner.results.verified_revisions import read_verified_checkpoints

    status = owned_run(run_id, key)
    return {
        **status.to_dict(),
        "verified_checkpoints": sorted(read_verified_checkpoints(run_id)),
    }


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
    public_status = status.to_dict()
    return {
        "run_id": run_id,
        "logs": chunk,
        "offset": end,
        "state": status.state,
        "last_heartbeat": public_status.get("last_heartbeat"),
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
    from flash.runner.lifecycle.state import _internal_spec_from_status

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
    from flash.runner.lifecycle.state import _internal_spec_from_status

    spec = _internal_spec_from_status(status)
    checkpoints = _app.list_checkpoints(spec)
    with contextlib.suppress(Exception):
        from flash.server.domain.registry.checkpoints import register_checkpoints_best_effort

        register_checkpoints_best_effort(status)
    return {"run_id": run_id, "checkpoints": checkpoints}
