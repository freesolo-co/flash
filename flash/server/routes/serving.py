"""Serving endpoints: deploy / undeploy an adapter, list deployments, and chat.

Service functions are resolved through ``flash.server.app`` at call time so test-suite patches on
``app.<name>`` are honored here.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event, Lock
from typing import Annotated
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from jsonschema import SchemaError, ValidationError, validate

from flash.runner import (
    adapter_prefix,
    complete_deployment_cleanup,
    mark_checkpoint_deployed,
    mark_deployed,
    mark_deployment_attempt_queued,
    mark_deployment_failed,
    mark_deployment_intent,
    mark_deployment_pre_intent_failed,
    mark_undeployed,
)
from flash.runner.checkpoints import checkpoint_adapter_prefix
from flash.serve.deploy import (
    READBACK_ATTEMPTS,
    READBACK_DELAY_SECONDS,
    AdapterConfigMissing,
    ServingError,
)
from flash.serve.urls import public_deployment
from flash.server import app as _app
from flash.server import db
from flash.server._deps import _require_bool, owned_run, require_key
from flash.server._internal_client import run_org_id
from flash.spec import JobSpec

router = APIRouter()
logger = logging.getLogger(__name__)

_DEPLOYMENT_BUSY_STATES = {"deploying"}
_DEPLOYMENT_STALE_SECONDS = 30 * 60
_RECOVERY_READ_ATTEMPTS = 3
_RECOVERY_RETRY_DELAY_SECONDS = 1.0
_SMOKE_PROMPT = "Deployment smoke test: answer in one short sentence. What is 2+2?"
# process-local exact owners; restart clears them so persisted attempts recover as orphaned.
_ACTIVE_DEPLOYMENT_WORKERS: set[tuple[str, str]] = set()
_ACTIVE_DEPLOYMENT_WORKERS_LOCK = Lock()


def _deployment_worker_key(run_id: str, mutation_id: str) -> tuple[str, str]:
    return run_id, mutation_id


def _mark_deployment_worker_active(run_id: str, mutation_id: str) -> None:
    with _ACTIVE_DEPLOYMENT_WORKERS_LOCK:
        _ACTIVE_DEPLOYMENT_WORKERS.add(_deployment_worker_key(run_id, mutation_id))


def _clear_deployment_worker_active(run_id: str, mutation_id: str) -> None:
    with _ACTIVE_DEPLOYMENT_WORKERS_LOCK:
        _ACTIVE_DEPLOYMENT_WORKERS.discard(_deployment_worker_key(run_id, mutation_id))


def _deployment_worker_is_active(run_id: str, mutation_id: str) -> bool:
    with _ACTIVE_DEPLOYMENT_WORKERS_LOCK:
        return _deployment_worker_key(run_id, mutation_id) in _ACTIVE_DEPLOYMENT_WORKERS


def _deployment_state(deployment: dict, state: str, **fields) -> dict:
    return {**deployment, **fields, "state": state, "updated_at": time.time()}


def _public_deployment(deployment: dict) -> dict:
    return public_deployment(deployment)


def _deployment_attempt_is_stale(deployment: dict, *, now: float | None = None) -> bool:
    if deployment.get("state") not in _DEPLOYMENT_BUSY_STATES:
        return False
    raw = deployment.get("updated_at") or deployment.get("requested_at")
    try:
        stamp = float(raw)
    except (TypeError, ValueError):
        return True
    return (time.time() if now is None else now) - stamp >= _DEPLOYMENT_STALE_SECONDS


def _read_adapter_for_recovery(run_id: str) -> dict | None:
    last_error: Exception | None = None
    for attempt in range(_RECOVERY_READ_ATTEMPTS):
        if attempt:
            time.sleep(_RECOVERY_RETRY_DELAY_SECONDS * attempt)
        try:
            return _app.read_adapter_record(run_id)
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _recover_deployment(run_id: str) -> bool:
    with _app._deploy_lock(run_id):
        try:
            status = _app.get_status(run_id)
        except FileNotFoundError:
            return False
        cleanup = getattr(status, "deployment_cleanup", None)
        if isinstance(cleanup, dict):
            try:
                reconciled = _app.reconcile_owned_adapter_cleanup(run_id, cleanup)
            except Exception as exc:
                logger.warning("deployment cleanup retry deferred for %s: %s", run_id, exc)
                return False
            if reconciled:
                complete_deployment_cleanup(run_id, cleanup)
            return reconciled
        deployment_attempt = getattr(status, "deployment_attempt", None)
        if isinstance(deployment_attempt, dict):
            queued = deployment_attempt.get("deployment") or {}
            mutation_id = str(queued.get("mutation_id") or "")
            if mutation_id and _deployment_worker_is_active(run_id, mutation_id):
                return False
            failed = _deployment_state(
                queued,
                "failed",
                error="deployment lifecycle interrupted before registry intent was persisted",
                detail="deployment interrupted; retry `flash deploy`",
            )
            mark_deployment_pre_intent_failed(run_id, deployment_attempt, failed)
            return True
        deployment = status.deployment or {}
        if deployment.get("state") != "deploying":
            return False
        mutation_id = deployment.get("mutation_id")
        if mutation_id and _deployment_worker_is_active(run_id, str(mutation_id)):
            return False
        desired = deployment.get("desired_record")
        target = deployment.get("target_revision")
        prior = deployment.get("prior_revision")
        try:
            requested_at = float(deployment["requested_at"])
            if not math.isfinite(requested_at):
                raise ValueError("requested_at must be finite")
        except (KeyError, TypeError, ValueError):
            mark_deployment_failed(
                run_id,
                _deployment_state(
                    deployment,
                    "failed",
                    error="deployment attempt metadata is malformed",
                    detail="deployment interrupted; retry `flash deploy`",
                ),
            )
            return True
        if not isinstance(desired, dict) or not isinstance(target, int) or not mutation_id:
            mark_deployment_failed(
                run_id,
                _deployment_state(
                    deployment,
                    "failed",
                    error="deployment lifecycle interrupted before registry intent was persisted",
                    detail="deployment interrupted; retry `flash deploy`",
                ),
            )
            return True
        try:
            current = _read_adapter_for_recovery(run_id)
        except Exception as exc:
            logger.warning("startup deployment recovery readback deferred for %s: %s", run_id, exc)
            return False
        if _app.record_matches(current, desired, target):
            _resume_registered_deployment(run_id, status.spec, deployment)
            return True
        current_revision = current.get("registry_revision") if isinstance(current, dict) else None
        current_mutation = current.get("mutation_id") if isinstance(current, dict) else None
        owned_disabled = (
            isinstance(current, dict)
            and current_revision == target + 1
            and current_mutation == mutation_id
            and current.get("status") == "disabled"
        )
        unchanged = current is None or current_revision == prior
        if owned_disabled:
            reason = "deployment mutation was disabled before recovery"
        elif unchanged:
            reason = "deployment registry mutation did not commit before restart"
        else:
            reason = "deployment was superseded while the control plane was offline"
        mark_deployment_failed(
            run_id,
            _deployment_state(
                deployment,
                "failed",
                error=reason,
                detail="deployment interrupted; retry `flash deploy`",
            ),
        )
        return True


def recover_deployments(*, max_workers: int = 4, stop_event: Event | None = None) -> int:
    """Reconcile every pre-restart busy deployment with bounded, stoppable workers."""
    run_ids = iter(row["run_id"] for row in db.all_runs())
    run_ids_lock = Lock()

    def worker() -> int:
        recovered = 0
        while stop_event is None or not stop_event.is_set():
            with run_ids_lock:
                try:
                    run_id = next(run_ids)
                except StopIteration:
                    return recovered
            try:
                recovered += int(_recover_deployment(run_id))
            except Exception:
                logger.exception("startup deployment recovery task failed for %s", run_id)
        return recovered

    worker_count = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(worker) for _ in range(worker_count)]
        return sum(future.result() for future in futures)


def _answer_after_reasoning(content: str, *, thinking: bool, require_closed: bool = False) -> str:
    if not thinking:
        return content.strip()
    closed = content.rfind("</think>")
    if closed < 0 and "<think>" in content:
        raise ServingError("smoke generation opened a <think> block but did not close it")
    if require_closed and closed < 0:
        raise ServingError(
            "structured smoke generation for a thinking adapter never closed its reasoning with "
            "</think>, so the deferred grammar was never exercised"
        )
    return (content[closed + len("</think>") :] if closed >= 0 else content).strip()


def _validate_structured_smoke(answer: str, structured_outputs: str) -> None:
    from flash.engine.structured_outputs import parse_structured_outputs

    constraint = parse_structured_outputs(structured_outputs)
    if not constraint:
        return
    try:
        if "json" in constraint:
            validate(instance=json.loads(answer), schema=constraint["json"])
        elif constraint.get("json_object") is True:
            if not isinstance(json.loads(answer), dict):
                raise ServingError("structured smoke output is valid JSON but not a JSON object")
        elif "choice" in constraint:
            if answer not in constraint["choice"]:
                raise ServingError(
                    f"structured smoke output {answer!r} is not one of {constraint['choice']!r}"
                )
        elif "regex" in constraint and re.fullmatch(str(constraint["regex"]), answer) is None:
            raise ServingError("structured smoke output does not match the configured regex")
    except json.JSONDecodeError as exc:
        raise ServingError(f"structured smoke output is not valid JSON: {exc}") from exc
    except (ValidationError, SchemaError) as exc:
        raise ServingError(
            f"structured smoke output violates the configured JSON schema: {exc}"
        ) from exc


def _run_deployment_smoke(
    run_id: str,
    spec: JobSpec,
    *,
    expected_checkpoint: str,
    expected_registry_revision: int,
    expected_mutation_id: str,
) -> dict:
    started = time.monotonic()
    result = _app.serve_chat(
        run_id=run_id,
        messages=[{"role": "user", "content": _SMOKE_PROMPT}],
        temperature=0.0,
        max_tokens=256,
        thinking=spec.thinking,
        expected_checkpoint=expected_checkpoint,
        expected_registry_revision=expected_registry_revision,
        expected_mutation_id=expected_mutation_id,
    )
    latency = time.monotonic() - started
    choice = (result.get("choices") or [{}])[0]
    content = str((choice.get("message") or {}).get("content") or "")
    finish = choice.get("finish_reason")
    if finish == "length":
        raise ServingError("smoke generation was truncated at the maximum token length")
    from flash.engine.structured_outputs import parse_structured_outputs

    structured = bool(parse_structured_outputs(spec.train.structured_outputs))
    answer = _answer_after_reasoning(content, thinking=spec.thinking, require_closed=structured)
    if not answer:
        raise ServingError(
            "smoke generation returned no answer content "
            f"(finish_reason={finish!r}) after {latency:.1f}s"
        )
    _validate_structured_smoke(answer, spec.train.structured_outputs)
    return {
        "verified_at": time.time(),
        "verify_latency_s": latency,
        "verify_finish_reason": finish,
        "thinking_tag": "<think>" in content or "</think>" in content,
        "verify_sample": answer[:160],
    }


def _attempt_owned(run_id: str, mutation_id: str) -> bool:
    status = _app.get_status(run_id)
    active = status.deployment or {}
    if active.get("state") == "deploying" and active.get("mutation_id") == mutation_id:
        return True
    pending = getattr(status, "deployment_attempt", None) or {}
    queued = pending.get("deployment") or {}
    return queued.get("mutation_id") == mutation_id


def _smoke_with_retries(run_id: str, spec: JobSpec, deployment: dict) -> dict:
    desired = deployment["desired_record"]
    target = int(deployment["target_revision"])
    mutation_id = str(deployment["mutation_id"])
    last_error: Exception | None = None
    for attempt in range(READBACK_ATTEMPTS):
        if attempt:
            time.sleep(READBACK_DELAY_SECONDS * attempt)
        try:
            return _run_deployment_smoke(
                run_id,
                spec,
                expected_checkpoint=str(desired["checkpoint"]),
                expected_registry_revision=target,
                expected_mutation_id=mutation_id,
            )
        except Exception as exc:
            last_error = exc
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
            response_text = str(getattr(response, "text", "") or "")
            message = f"{exc} {response_text}".lower()
            retryable = isinstance(exc, httpx.RequestError)
            retryable = retryable or (status_code is not None and status_code >= 500)
            retryable = retryable or (status_code == 404 and "unknown" in message)
            if status_code == 409 and "deployment mismatch" in message:
                try:
                    retryable = _app.record_matches(
                        _app.read_adapter_record(run_id), desired, target
                    )
                except Exception:
                    retryable = False
            if not retryable:
                raise
    assert last_error is not None
    raise last_error


def _finalize_registered_deployment(
    run_id: str,
    spec: JobSpec,
    deployment: dict,
    *,
    checkpoint_step: int | None,
    is_checkpoint: bool,
    prev_state: str,
    verify: bool,
) -> None:
    mutation_id = str(deployment["mutation_id"])
    if not _attempt_owned(run_id, mutation_id):
        return
    desired = deployment["desired_record"]
    target = int(deployment["target_revision"])
    if verify:
        smoke = _smoke_with_retries(run_id, spec, deployment)
    else:
        smoke = {"detail": "registered; smoke skipped"}
    if not _attempt_owned(run_id, mutation_id):
        return
    try:
        current = _read_adapter_for_recovery(run_id)
    except Exception as exc:
        logger.warning("final deployment readback deferred for %s: %s", run_id, exc)
        return
    if not _app.record_matches(current, desired, target):
        raise ServingError("serving registry changed during smoke")
    ready = _deployment_state(deployment, "ready", **smoke)
    public_ready = _public_deployment(ready)
    if is_checkpoint:
        state_guard = prev_state if prev_state in _app._DEPLOYABLE_STATES else None
        mark_checkpoint_deployed(
            run_id,
            public_ready,
            expect_state=state_guard,
            expect_mutation_id=mutation_id,
            expect_deployment_state="deploying",
        )
    else:
        mark_deployed(
            run_id,
            public_ready,
            expect_state=prev_state,
            expect_mutation_id=mutation_id,
            expect_deployment_state="deploying",
        )


def _disable_failed_attempt(run_id: str, deployment: dict) -> bool:
    """Return whether the failed attempt is conclusively no longer active."""
    try:
        _app.disable_owned_adapter(
            run_id,
            int(deployment["target_revision"]),
            str(deployment["mutation_id"]),
        )
    except _app.DeploymentSuperseded:
        return True
    except Exception as exc:
        logger.warning("failed to disable owned deployment %s: %s", run_id, exc)
        return False
    return True


def _resume_registered_deployment(run_id: str, spec_dict: dict, deployment: dict) -> None:
    spec = JobSpec.from_dict(spec_dict)
    try:
        _finalize_registered_deployment(
            run_id,
            spec,
            deployment,
            checkpoint_step=deployment.get("checkpoint_step"),
            is_checkpoint=deployment.get("checkpoint_step") is not None,
            prev_state=str(deployment.get("prev_state") or _app.get_status(run_id).state),
            verify=bool(deployment.get("verify", True)),
        )
    except Exception as exc:
        if not _attempt_owned(run_id, str(deployment["mutation_id"])):
            return
        if not _disable_failed_attempt(run_id, deployment):
            return
        mark_deployment_failed(
            run_id,
            _deployment_state(
                deployment,
                "failed",
                error=str(exc),
                detail="deployment failed; retry `flash deploy` after fixing the error",
            ),
        )


def _finish_deployment_unlocked(
    *,
    run_id: str,
    spec_dict: dict,
    checkpoint_step: int | None,
    is_checkpoint: bool,
    deploy_kwargs: dict,
    deployment_attempt: dict,
    prev_state: str,
    verify: bool,
) -> None:
    spec = JobSpec.from_dict(spec_dict)
    deployment = deployment_attempt["deployment"]
    mutation_id = str(deployment["mutation_id"])
    intent_persisted = False

    def persist_intent(
        prior_revision: int | None,
        desired: dict,
        target_revision: int,
        persisted_mutation_id: str,
        repo_revision: str,
        *,
        prior_mutation_id: str | None = None,
    ) -> None:
        nonlocal intent_persisted
        if persisted_mutation_id != mutation_id:
            raise ServingError("deployment mutation identity changed before registry mutation")
        intent = _deployment_state(
            deployment,
            "deploying",
            detail="persisting deployment intent",
            desired_record=desired,
            prior_revision=prior_revision,
            prior_mutation_id=prior_mutation_id,
            target_revision=target_revision,
            mutation_id=mutation_id,
            repo_revision=repo_revision,
            prev_state=prev_state,
        )
        with _app._deploy_lock(run_id):
            marked = mark_deployment_intent(run_id, intent, expect_attempt=deployment_attempt)
            active = marked.deployment or {}
            if active != intent or getattr(marked, "deployment_attempt", None) is not None:
                raise ServingError("deployment ownership changed before registry mutation")
            deployment.clear()
            deployment.update(intent)
            intent_persisted = True

    @contextmanager
    def registry_mutation_guard():
        with _app._deploy_lock(run_id):
            status = _app.get_status(run_id)
            active = status.deployment or {}
            if (
                active.get("state") != "deploying"
                or active.get("mutation_id") != mutation_id
            ):
                raise ServingError("deployment ownership changed before registry mutation")
            yield

    try:
        dep = _app.deploy_adapter(
            **deploy_kwargs,
            before_registry_mutation=persist_intent,
            registry_mutation_guard=registry_mutation_guard,
        )
        if not dep.desired_record or dep.target_revision is None or not dep.mutation_id:
            raise ServingError("serving registration did not return deployment identity")
        deployment.update(
            desired_record=dep.desired_record,
            prior_revision=dep.prior_revision,
            target_revision=dep.target_revision,
            mutation_id=dep.mutation_id,
            repo_revision=dep.repo_revision,
            prev_state=prev_state,
        )
        _finalize_registered_deployment(
            run_id,
            spec,
            deployment,
            checkpoint_step=checkpoint_step,
            is_checkpoint=is_checkpoint,
            prev_state=prev_state,
            verify=verify,
        )
    except Exception as exc:
        if intent_persisted:
            with _app._deploy_lock(run_id):
                if not _attempt_owned(run_id, mutation_id):
                    return
                if (
                    deployment.get("target_revision") is not None
                    and deployment.get("mutation_id")
                    and not _disable_failed_attempt(run_id, deployment)
                ):
                    return
        elif not _attempt_owned(run_id, mutation_id):
            return
        error = str(exc)
        if not is_checkpoint and isinstance(exc, AdapterConfigMissing):
            steps = [item["step"] for item in _app.list_checkpoints(spec)]
            if steps:
                error = (
                    f"run {run_id} has no run-level adapter at {deployment.get('adapter_hf_prefix')} "
                    f"(the run likely never finalized); deploy a saved checkpoint instead, e.g. "
                    f"`flash deploy {run_id}/step-{steps[-1]}` "
                    f"(available steps: {', '.join(str(step) for step in steps)})"
                )
        failed = _deployment_state(
            deployment,
            "failed",
            error=error,
            detail="deployment failed; retry `flash deploy` after fixing the error",
        )
        if intent_persisted:
            mark_deployment_failed(run_id, failed)
        else:
            mark_deployment_pre_intent_failed(run_id, deployment_attempt, failed)


def _finish_deployment(**kwargs) -> None:
    run_id = str(kwargs["run_id"])
    mutation_id = str(kwargs["deployment_attempt"]["deployment"]["mutation_id"])
    try:
        _finish_deployment_unlocked(**kwargs)
    finally:
        _clear_deployment_worker_active(run_id, mutation_id)


def _chat_messages_from_payload(payload: dict) -> list[dict]:
    raw = payload.get("messages")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise HTTPException(status_code=400, detail="messages must be a list")
    for index, message in enumerate(raw):
        if not isinstance(message, dict):
            raise HTTPException(
                status_code=400,
                detail=f"messages[{index}] must be a chat message object",
            )
    return raw


def _validate_hf_repo_id(repository: str) -> None:
    """Validate HF repo id grammar early — malformed ids only 502 AFTER downloading the private source adapter."""
    try:
        from huggingface_hub.utils import HFValidationError, validate_repo_id
    except ModuleNotFoundError:
        return
    try:
        validate_repo_id(repository)
    except HFValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"repository is not a valid HuggingFace repo id: {exc}",
        ) from exc


def _resolve_deploy_step(run_id: str, spec, raw_step) -> int | None:
    """Validate optional checkpoint step; returns int or None (final adapter). 400 on bad step, 404 on missing."""
    if raw_step is None:
        return None

    # Reject bool (True -> step 1) and non-integer floats; str path uses fullmatch not isdigit
    # (isdigit accepts unicode digits + "-5" which crash int() -> 500). The length bound also keeps a
    # 4301+ digit string from tripping Python's int-string-conversion limit (another int() -> 500).
    want: int | None = None
    if isinstance(raw_step, bool):
        want = None
    elif isinstance(raw_step, int):
        want = raw_step
    elif isinstance(raw_step, float):
        want = int(raw_step) if raw_step.is_integer() else None
    elif isinstance(raw_step, str):
        s = raw_step.strip()
        want = int(s) if re.fullmatch(r"-?[0-9]{1,18}", s) else None
    if want is not None and want < 0:
        want = None
    if want is None:
        raise HTTPException(status_code=400, detail=f"invalid checkpoint step: {raw_step!r}")
    checkpoints = _app.list_checkpoints(spec)
    if any(c["step"] == want for c in checkpoints):
        return want
    available = ", ".join(str(c["step"]) for c in checkpoints) or "none"
    raise HTTPException(
        status_code=404,
        detail=f"run {run_id} has no deployable checkpoint at step {want} (available: {available})",
    )


def _resolve_deployable_target(
    run_id: str, spec, status, raw_step, *, action: str, enforce_state: bool
) -> tuple[int | None, bool, str]:
    """Resolve the deploy/export target and gate final-adapter targets on training state."""
    checkpoint_step = _resolve_deploy_step(run_id, spec, raw_step)
    is_checkpoint = checkpoint_step is not None
    # A resolved checkpoint step has already proven a servable adapter exists; only final-adapter
    # deploy/export needs the run-state gate because the final adapter exists only after completion.
    if enforce_state and is_checkpoint and status.state == "dry_run":
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} is 'dry_run'; dry-run runs cannot be {action}ed",
        )
    if enforce_state and not is_checkpoint and status.state not in _app._DEPLOYABLE_STATES:
        detail = (
            f"run {run_id} is {status.state!r}; only finished runs with "
            f"trained adapter artifacts can be {'deployed' if action == 'deploy' else 'exported'}"
        )
        raise HTTPException(status_code=409, detail=detail)
    prefix = (
        checkpoint_adapter_prefix(spec, checkpoint_step) if is_checkpoint else adapter_prefix(spec)
    )
    return checkpoint_step, is_checkpoint, prefix


@router.post("/v1/runs/{run_id}/deploy")
def deploy(run_id: str, key: Annotated[dict, Depends(require_key)], payload: dict | None = None):
    payload = payload or {}
    with _app._deploy_lock(run_id):
        status = owned_run(run_id, key)
        spec = JobSpec.from_dict(status.spec)
        dry_run = _require_bool(payload, "dry_run", False)
        verify = _require_bool(payload, "verify", True)
        mutation_id = str(uuid4())
        current_deployment = status.deployment or {}
        current_attempt = status.deployment_attempt
        pending_deployment = (
            current_attempt.get("deployment") if isinstance(current_attempt, dict) else None
        )
        busy_deployment = pending_deployment or current_deployment
        busy_mutation_id = (
            str(busy_deployment.get("mutation_id") or "")
            if isinstance(busy_deployment, dict)
            else ""
        )
        if (
            not dry_run
            and isinstance(busy_deployment, dict)
            and busy_deployment.get("state") in _DEPLOYMENT_BUSY_STATES
            and (
                _deployment_worker_is_active(run_id, busy_mutation_id)
                or not _deployment_attempt_is_stale(busy_deployment)
            )
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} already has a deployment in "
                    f"{busy_deployment.get('state')} state; run `flash deployments` "
                    "to check progress"
                ),
            )
        checkpoint_step, is_checkpoint, deploy_prefix = _resolve_deployable_target(
            run_id, spec, status, payload.get("step"), action="deploy", enforce_state=not dry_run
        )
        if not dry_run and not spec.train.hf_repo:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} has no [train].hf_repo; its adapter artifacts "
                    "cannot be located, so it cannot be deployed"
                ),
            )
        # CAS guard: /cancel's worker + provider teardown runs outside this lock (only its status
        # write is lock-serialized), so capture state before deploy and re-verify it on the write.
        prev_state = status.state
        deploy_org_id = run_org_id(status) or None
        if not dry_run and not deploy_org_id:
            raise HTTPException(
                status_code=409,
                detail="run has no persisted owning org and cannot be deployed",
            )
        deploy_kwargs = {
            "run_id": run_id,
            "model": spec.model,
            "hf_repo": spec.train.hf_repo,
            "adapter_prefix": deploy_prefix,
            "mutation_id": mutation_id,
            "dry_run": dry_run,
            "lora_rank": spec.train.lora_rank,
            # a run trained with thinking serves with thinking (per-run parity)
            "thinking": spec.thinking,
            # a run trained with structured_outputs serves under the same grammar for both thinking
            # and non-thinking adapters; serving owns when the constraint begins applying.
            "structured_outputs": spec.train.structured_outputs,
            "org_id": deploy_org_id,
        }
        if dry_run:
            try:
                dep = _app.deploy_adapter(**deploy_kwargs)
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                raise
            return dep.to_dict()

        # Validate the cheap configured-rank part synchronously so obvious spec errors return 400
        # instead of becoming background deployment failures.
        try:
            from flash.serve.deploy import validate_serving_lora_rank

            validate_serving_lora_rank(
                spec.model, spec.train.lora_rank, rank_source="configured train.lora_rank"
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            dep_dict = _app.deployment_record(
                run_id=run_id,
                model=spec.model,
                adapter_prefix=deploy_prefix,
                state="deploying",
            ).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        dep_dict = _deployment_state(
            dep_dict,
            "deploying",
            detail="deployment queued",
            verify=verify,
            mutation_id=mutation_id,
            requested_at=time.time(),
        )
        if is_checkpoint:
            dep_dict["checkpoint_step"] = checkpoint_step
        active_deployment = (
            current_deployment
            if current_deployment.get("state") in {"ready", "deployed"}
            else None
        )
        deployment_attempt = {
            "phase": "redeploy" if active_deployment is not None else "initial",
            "deployment": dep_dict,
            "active_deployment": active_deployment,
        }
        marked = mark_deployment_attempt_queued(
            run_id,
            deployment_attempt,
            expect_state=prev_state,
            expect_deployment=status.deployment,
            expect_attempt=current_attempt,
        )
        expected_visible = active_deployment if active_deployment is not None else dep_dict
        if (
            marked.deployment != expected_visible
            or marked.deployment_attempt != deployment_attempt
        ):
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} became {marked.state!r} during deploy; aborted",
            )

        job_kwargs = {
            "run_id": run_id,
            "spec_dict": status.spec,
            "checkpoint_step": checkpoint_step,
            "is_checkpoint": is_checkpoint,
            "deploy_kwargs": deploy_kwargs,
            "deployment_attempt": deployment_attempt,
            "prev_state": prev_state,
            "verify": verify,
        }
        _mark_deployment_worker_active(run_id, mutation_id)
    try:
        ran_sync = _app.start_deployment_job(_finish_deployment, **job_kwargs)
    except BaseException:
        _clear_deployment_worker_active(run_id, mutation_id)
        raise
    if ran_sync:
        return _public_deployment(_app.get_status(run_id).deployment or expected_visible)
    return _public_deployment(expected_visible)


@router.delete("/v1/runs/{run_id}/deploy")
def undeploy(run_id: str, key: Annotated[dict, Depends(require_key)]):
    with _app._deploy_lock(run_id):
        status = owned_run(run_id, key)
        try:
            deleted = _app.undeploy_adapter(run_id)
        except ServingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        # Idempotent: clear local record even if serving side already had no adapter.
        if status.deployment or status.deployment_attempt:
            mark_undeployed(run_id)
        # serving_deregistered=False means serving had nothing to delete (already gone or never
        # actually registered) — the record teardown above still happened either way.
        return {
            "run_id": run_id,
            "deleted_endpoints": deleted,
            "serving_deregistered": bool(deleted),
        }


@router.post("/v1/runs/{run_id}/export")
def export(run_id: str, key: Annotated[dict, Depends(require_key)], payload: dict | None = None):
    """Copy a run's trained adapter into a user-owned HuggingFace repo."""
    payload = payload or {}
    with _app._deploy_lock(run_id):
        repository = str(payload.get("repository") or "").strip()
        if not repository:
            raise HTTPException(
                status_code=400,
                detail="repository is required: the destination HuggingFace repo 'owner/name'",
            )
        if len(parts := repository.strip("/").split("/")) != 2 or not all(parts):
            raise HTTPException(
                status_code=400,
                detail=f"repository must be a HuggingFace repo of the form 'owner/name', got {repository!r}",
            )
        repository = "/".join(parts)
        _validate_hf_repo_id(repository)
        hf_token = str(payload.get("hf_token") or "").strip()
        if not hf_token:
            raise HTTPException(
                status_code=400,
                detail="hf_token is required: a HuggingFace token with write access to the destination repo",
            )
        private = _require_bool(payload, "private", True)

        status = owned_run(run_id, key)
        spec = JobSpec.from_dict(status.spec)
        if not spec.train.hf_repo:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} has no [train].hf_repo; its adapter artifacts "
                    "cannot be located, so it cannot be exported"
                ),
            )
        checkpoint_step, is_checkpoint, prefix = _resolve_deployable_target(
            run_id, spec, status, payload.get("step"), action="export", enforce_state=True
        )
        subfolder = f"{prefix}/adapter"
        try:
            url = _app.export_adapter(
                source_repo=spec.train.hf_repo,
                source_subfolder=subfolder,
                dest_repo=repository,
                dest_token=hf_token,
                private=private,
                base_model=spec.model,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ServingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    result = {
        "run_id": run_id,
        "adapter_id": run_id,
        "repository": repository,
        "url": url,
        "source": f"{spec.train.hf_repo}:{subfolder}",
    }
    if is_checkpoint:
        result["step"] = checkpoint_step
    return result


@router.get("/v1/deployments")
def deployments(key: Annotated[dict, Depends(require_key)]):
    out = []
    for row in db.runs_for_key(key["id"]):
        try:
            status = _app.get_status(row["run_id"])
        except FileNotFoundError:
            continue
        if status.deployment and status.deployment.get("state") not in (
            "undeployed",
            "dry_run",
        ):
            data = status.to_dict()
            data["deployment"] = _public_deployment(data["deployment"])
            out.append(data)
    return {"deployments": out}


@router.post("/v1/runs/{run_id}/chat")
def chat(run_id: str, payload: dict, key: Annotated[dict, Depends(require_key)]):
    messages = _chat_messages_from_payload(payload)
    status = owned_run(run_id, key)
    spec = JobSpec.from_dict(status.spec)
    deployment = status.deployment or {}
    deployment_state = deployment.get("state")
    has_ready_deploy = deployment_state in {"ready", "deployed"}
    # A cancelled run can still serve a per-step checkpoint it deployed: checkpoint deploy records
    # a live adapter that /v1/deployments lists as active without requiring a final adapter.
    # Only block chat when there's no active deployment to serve.
    if not has_ready_deploy:
        if deployment_state in _DEPLOYMENT_BUSY_STATES:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} deployment is {deployment_state}; run "
                    "`flash deployments` to check progress"
                ),
            )
        if deployment_state == "failed":
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} deployment failed: {deployment.get('error') or 'unknown error'}"
                ),
            )
        if status.state == "cancelled":
            raise HTTPException(
                status_code=409,
                detail=f"run {run_id} was cancelled; deploy a checkpoint with "
                f"`flash deploy {run_id}/step-<N>` first",
            )
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} has no active deployment; `flash deploy {run_id}` first",
        )
    if not spec.train.hf_repo:
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} has no [train].hf_repo; its adapter cannot be served",
        )
    # Parse sampling params before the broad try so bad values are 400, not 502.
    try:
        temperature = float(payload.get("temperature") or 0.0)
        # Avoid `or 512`: that silently coerces an explicit 0 to 512.
        raw_max_tokens = payload.get("max_tokens")
        # OverflowError (int(inf), an ArithmeticError) is NOT a TypeError/ValueError — catch it too so a
        # JSON `Infinity`/`1e400` max_tokens is a clean 400, not an uncaught 500.
        max_tokens = 512 if raw_max_tokens is None else int(raw_max_tokens)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid temperature/max_tokens: {exc}"
        ) from exc
    if not math.isfinite(temperature):
        raise HTTPException(
            status_code=400, detail=f"temperature must be a finite number, got {temperature}"
        )
    if max_tokens <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens must be a positive integer, got {max_tokens}",
        )
    try:
        if payload.get("stream") is True:
            return StreamingResponse(
                _app.serve_chat_stream(
                    run_id=run_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking=spec.thinking,
                ),
                media_type="text/plain; charset=utf-8",
            )
        return _app.serve_chat(
            run_id=run_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=spec.thinking,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"inference failure: {exc}") from exc
