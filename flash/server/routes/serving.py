"""Serving endpoints: deploy / undeploy an adapter, list deployments, and chat.

Service functions are resolved through ``flash.server.app`` at call time so test-suite patches on
``app.<name>`` are honored here.
"""

from __future__ import annotations

import contextlib
import json
import math
import multiprocessing
import re
import time
from typing import Annotated

import regex as safe_regex
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from jsonschema import SchemaError, ValidationError
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import NoSuchResource, Unresolvable

from flash.engine.structured_outputs import parse_structured_outputs
from flash.runner import (
    adapter_prefix,
    effective_spec_from_status,
    mark_checkpoint_deployed,
    mark_deployed,
    mark_deployment_failed,
    mark_deployment_pending,
    mark_deployment_revocation_failed,
    mark_undeployed,
    read_verified_adapter_revisions,
    verified_adapter_revision_generation,
)
from flash.runner.checkpoints import checkpoint_adapter_prefix
from flash.schema import parse_adapter_revision
from flash.serve.deploy import (
    ActivationOutcomeUnknown,
    AdapterConfigMissing,
    RetryableServingUnavailable,
    ServingError,
)
from flash.serve.urls import public_deployment
from flash.server import app as _app
from flash.server import db
from flash.server._deps import _require_bool, owned_run, require_key
from flash.server._internal_client import run_org_id
from flash.spec import JobSpec

router = APIRouter()

_DEPLOYMENT_BUSY_STATES = {"queued", "smoke_testing", "reconciling"}
_DEPLOYMENT_READY_STATES = {"ready", "deployed"}
_DEPLOYMENT_STALE_SECONDS = 30 * 60


_SMOKE_PROMPT = "Deployment smoke test: answer in one short sentence. What is 2+2?"
# hard wall-clock budget for the trusted fixed-prompt generation and validation.
# it remains below the deployment stale threshold.
_SMOKE_BUDGET_SECONDS = 600.0


def _smoke_timeout_error(budget_s: float) -> ServingError:
    return ServingError(f"deployment_smoke_timeout: bounded smoke exceeded {budget_s:g}s")


def _bounded_call(fn, *, deadline: float, budget_s: float):
    """run the trusted smoke call within the remaining global deadline."""
    import threading

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _smoke_timeout_error(budget_s)
    result: dict = {}

    def _target() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # re-raised on the caller thread below
            result["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=remaining)
    if thread.is_alive():
        raise _smoke_timeout_error(budget_s)
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _reject_external_schema_reference(uri: str):
    raise NoSuchResource(ref=uri)


_DIRECT_REGEX_TIMEOUT_SECONDS = 0.05
_JSON_SCHEMA_TIMEOUT_SECONDS = 3.0
_JSON_SCHEMA_PROCESS_NAME = "flash-json-schema-validation"


def _bounded_regex_fullmatch(pattern: str, value: str, *, deadline: float, budget_s: float):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _smoke_timeout_error(budget_s)
    global_deadline_limits = remaining <= _DIRECT_REGEX_TIMEOUT_SECONDS
    timeout = min(_DIRECT_REGEX_TIMEOUT_SECONDS, remaining)
    try:
        return safe_regex.fullmatch(pattern, value, timeout=timeout)
    except TimeoutError as exc:
        if global_deadline_limits:
            raise _smoke_timeout_error(budget_s) from exc
        raise ServingError(
            f"structured-output regex evaluation exceeded the {_DIRECT_REGEX_TIMEOUT_SECONDS:.2f}s deadline"
        ) from exc


def _sanitized_schema_error(exc: Exception) -> str:
    message = getattr(exc, "message", str(exc))
    return " ".join(str(message).split())[:500]


def _json_schema_validation_worker(connection, instance, schema) -> None:
    try:
        validator_class = validator_for(schema)
        validator_class.check_schema(schema)
        registry = Registry(retrieve=_reject_external_schema_reference)
        validator_class(schema, registry=registry).validate(instance)
    except Unresolvable as exc:
        outcome = ("reference", _sanitized_schema_error(exc))
    except SchemaError as exc:
        outcome = ("schema", _sanitized_schema_error(exc))
    except ValidationError as exc:
        outcome = ("validation", _sanitized_schema_error(exc))
    except Exception as exc:
        outcome = ("error", f"{type(exc).__name__}: {_sanitized_schema_error(exc)}")
    else:
        outcome = ("ok", "")
    try:
        connection.send(outcome)
    finally:
        connection.close()


def _reap_schema_validation_process(process, *, deadline: float) -> None:
    remaining = max(0.0, deadline - time.monotonic())
    process.join(timeout=min(0.1, remaining))
    if process.is_alive():
        process.terminate()
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=min(0.2, remaining))
    if process.is_alive():
        process.kill()
        process.join()


def _validate_json_schema(instance, schema: dict, *, deadline: float, budget_s: float) -> None:
    now = time.monotonic()
    if deadline <= now:
        raise _smoke_timeout_error(budget_s)
    local_deadline = now + _JSON_SCHEMA_TIMEOUT_SECONDS
    validation_deadline = min(deadline, local_deadline)
    global_deadline_limits = deadline <= local_deadline
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_json_schema_validation_worker,
        args=(send_connection, instance, schema),
        name=_JSON_SCHEMA_PROCESS_NAME,
        daemon=True,
    )
    outcome: tuple[str, str] | None = None
    timed_out = False
    try:
        try:
            process.start()
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise _smoke_timeout_error(budget_s) from exc
            raise ServingError(f"could not start isolated JSON schema validation: {exc}") from exc
        finally:
            send_connection.close()
        remaining = max(0.0, validation_deadline - time.monotonic())
        if receive_connection.poll(remaining):
            try:
                outcome = receive_connection.recv()
            except EOFError:
                outcome = None
        else:
            timed_out = True
    finally:
        receive_connection.close()
        if process.pid is not None:
            _reap_schema_validation_process(process, deadline=deadline)
            process.close()

    if outcome is None:
        if timed_out and global_deadline_limits:
            raise _smoke_timeout_error(budget_s)
        raise ServingError(
            f"JSON schema validation exceeded the {_JSON_SCHEMA_TIMEOUT_SECONDS:.1f}s wall-clock deadline"
        )
    status, detail = outcome
    if status == "ok":
        return
    if status == "reference":
        raise ServingError(f"configured JSON schema reference could not be resolved: {detail}")
    if status == "schema":
        raise ServingError(f"configured JSON schema is invalid: {detail}")
    if status == "validation":
        raise ServingError(f"structured smoke output violates the configured JSON schema: {detail}")
    raise ServingError(f"JSON schema validation failed safely: {detail}")


def _strict_json_loads(value: str):
    def reject_constant(constant: str):
        raise ValueError(f"non-finite JSON constant {constant!r} is not allowed")

    try:
        return json.loads(value, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ServingError(f"structured smoke output is not valid JSON: {exc}") from exc


def _deployment_state(deployment: dict, state: str, **fields) -> dict:
    return {**deployment, **fields, "state": state, "updated_at": time.time()}


def _public_deployment(deployment: dict) -> dict:
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


def _deployment_attempt_is_stale(deployment: dict, *, now: float | None = None) -> bool:
    if deployment.get("state") not in _DEPLOYMENT_BUSY_STATES:
        return False
    raw = deployment.get("updated_at") or deployment.get("requested_at")
    try:
        stamp = float(raw)
    except (TypeError, ValueError):
        return True
    return (time.time() if now is None else now) - stamp >= _DEPLOYMENT_STALE_SECONDS


def _previous_ready_deployment(deployment: dict) -> dict | None:
    state = deployment.get("state")
    if state in _DEPLOYMENT_READY_STATES:
        return dict(deployment)
    if state not in _DEPLOYMENT_BUSY_STATES or state == "reconciling":
        return None
    previous = deployment.get("previous_deployment")
    if isinstance(previous, dict) and previous.get("state") in _DEPLOYMENT_READY_STATES:
        return dict(previous)
    return None


def _deployment_predecessor(deployment: dict) -> dict | None:
    ready = _previous_ready_deployment(deployment)
    if ready is not None:
        return ready
    if deployment.get("activation_outcome_unknown"):
        previous = deployment.get("previous_deployment")
        if isinstance(previous, dict) and previous.get("state") in _DEPLOYMENT_READY_STATES:
            return dict(previous)
    return None


def _activation_predecessor(run_id: str, deployment: dict) -> tuple[str | None, dict | None]:
    if not deployment.get("activation_outcome_unknown"):
        predecessor = _deployment_predecessor(deployment)
        revision = predecessor.get("adapter_revision") if predecessor is not None else None
        return (revision if isinstance(revision, str) else None), predecessor

    target = _app.adapter_alias_target(run_id)
    if target is None:
        return None, None
    parsed_target = parse_adapter_revision(target)
    if parsed_target is None or parsed_target[0] != run_id:
        raise ServingError(f"serving alias {run_id} targets invalid revision {target!r}")

    nested = deployment.get("previous_deployment")
    candidates = [deployment, nested if isinstance(nested, dict) else None]
    predecessor = next(
        (
            dict(candidate)
            for candidate in candidates
            if candidate is not None and candidate.get("adapter_revision") == target
        ),
        {
            "run_id": run_id,
            "adapter_revision": target,
            "checkpoint_step": parsed_target[1],
            "openai_model": run_id,
        },
    )
    predecessor.pop("previous_deployment", None)
    predecessor.pop("activation_outcome_unknown", None)
    predecessor.pop("error", None)
    predecessor["state"] = (
        "ready" if target in read_verified_adapter_revisions(run_id) else "reconciling"
    )
    return target, predecessor


def _verified_adapter_revisions(status) -> set[str]:
    return set(read_verified_adapter_revisions(status.run_id))


def recover_deployments() -> int:
    """Clear deployment lifecycle records left busy by a control-plane restart."""
    recovered = 0
    for row in db.all_runs():
        try:
            status = _app.get_status(row["run_id"])
        except FileNotFoundError:
            continue
        deployment = status.deployment or {}
        if not _deployment_attempt_is_stale(deployment):
            continue
        failed = _deployment_state(
            deployment,
            "failed",
            error="deployment lifecycle interrupted by control-plane restart",
            detail="deployment interrupted; retry `flash deploy`",
            recovered_at=time.time(),
        )
        mark_deployment_failed(status.run_id, failed)
        recovered += 1
    return recovered


def _smoke_provenance(result: dict, adapter_revision: str, checkpoint: str) -> tuple[str, object]:
    choice = (result.get("choices") or [{}])[0]
    content = str((choice.get("message") or {}).get("content") or "")
    finish = choice.get("finish_reason")
    if not content.strip():
        raise ServingError(f"smoke generation returned no content (finish_reason={finish!r})")
    provenance = result.get("freesolo")
    if not isinstance(provenance, dict):
        raise ServingError("smoke response omitted immutable revision provenance")
    if provenance.get("adapter_revision") != adapter_revision:
        raise ServingError("smoke response returned the wrong adapter revision")
    if provenance.get("checkpoint") != checkpoint:
        raise ServingError("smoke response returned the wrong checkpoint")
    hf_revision = adapter_revision.rsplit(".", 1)[-1]
    if provenance.get("hf_revision") != hf_revision:
        raise ServingError("smoke response returned the wrong Hub revision")
    headers = result.get("_freesolo_headers")
    expected_headers = {
        "adapter_revision": adapter_revision,
        "checkpoint": checkpoint,
        "hf_revision": hf_revision,
    }
    if headers != expected_headers:
        raise ServingError("smoke response returned mismatched provenance headers")
    return content, finish


def _thinking_structured_answer(content: str) -> str:
    closed = content.find("</think>")
    if closed < 0:
        raise ServingError(
            "structured smoke generation for a thinking adapter never closed its reasoning with "
            "</think>"
        )
    answer = content[closed + len("</think>") :].strip()
    if not answer:
        raise ServingError("structured smoke generation returned no answer after </think>")
    return answer


def _validate_structured_smoke(
    answer: str, constraint: dict, *, deadline: float, budget_s: float
) -> None:
    if time.monotonic() >= deadline:
        raise _smoke_timeout_error(budget_s)
    try:
        if "json" in constraint:
            _validate_json_schema(
                _strict_json_loads(answer),
                constraint["json"],
                deadline=deadline,
                budget_s=budget_s,
            )
        elif constraint.get("json_object") is True:
            if not isinstance(_strict_json_loads(answer), dict):
                raise ServingError("structured smoke output is valid JSON but not a JSON object")
        elif "choice" in constraint:
            if answer not in constraint["choice"]:
                raise ServingError(
                    f"structured smoke output {answer!r} is not one of {constraint['choice']!r}"
                )
        elif (
            "regex" in constraint
            and _bounded_regex_fullmatch(
                str(constraint["regex"]), answer, deadline=deadline, budget_s=budget_s
            )
            is None
        ):
            raise ServingError("structured smoke output does not match the configured regex")
    except safe_regex.error as exc:
        raise ServingError(f"configured structured-output regex is invalid: {exc}") from exc


def _run_deployment_smoke(
    run_id: str,
    spec: JobSpec,
    *,
    serving_model: str,
    expected_checkpoint: str,
    budget_s: float = _SMOKE_BUDGET_SECONDS,
) -> dict:
    started = time.monotonic()
    deadline = started + budget_s
    train = getattr(spec, "train", None)
    constraint = parse_structured_outputs(getattr(train, "structured_outputs", ""))
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _smoke_timeout_error(budget_s)
        try:

            def _smoke_call(timeout_s: float = remaining):
                return _app.serve_chat(
                    run_id=serving_model,
                    messages=[{"role": "user", "content": _SMOKE_PROMPT}],
                    temperature=0.0,
                    max_tokens=256,
                    thinking=spec.thinking,
                    expected_checkpoint=expected_checkpoint,
                    timeout_s=timeout_s,
                    retry_unavailable=True,
                )

            result = _bounded_call(
                _smoke_call,
                deadline=deadline,
                budget_s=budget_s,
            )
        except RetryableServingUnavailable as exc:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _smoke_timeout_error(budget_s) from exc
            time.sleep(min(exc.retry_after_seconds, remaining))
            continue
        break
    content, finish = _smoke_provenance(result, serving_model, expected_checkpoint)
    if constraint and spec.thinking and finish == "length":
        raise ServingError("structured smoke generation was truncated at the maximum token length")
    answer = (
        _thinking_structured_answer(content) if constraint and spec.thinking else content.strip()
    )
    if constraint:
        _validate_structured_smoke(answer, constraint, deadline=deadline, budget_s=budget_s)
    if time.monotonic() > deadline:
        raise _smoke_timeout_error(budget_s)
    return {
        "verified_at": time.time(),
        "verify_kind": "fixed_prompt",
        "verify_turns": 1,
        "verify_latency_s": time.monotonic() - started,
        "verify_finish_reason": finish,
        "thinking_tag": "<think>" in content or "</think>" in content,
        "verify_sample": answer[:160],
    }


def _finish_deployment_unlocked(
    *,
    run_id: str,
    spec_dict: dict,
    checkpoint_step: int | None,
    is_checkpoint: bool,
    deploy_kwargs: dict,
    deployment: dict,
    prev_state: str,
) -> None:
    spec = JobSpec.from_dict(spec_dict)
    active = _app.get_status(run_id).deployment or {}
    if (
        active.get("requested_at") != deployment.get("requested_at")
        or active.get("state") not in _DEPLOYMENT_BUSY_STATES
    ):
        return
    current = dict(deployment)
    smoke_result: dict = {}
    activated = False

    def _assert_activation_fence() -> None:
        latest = _app.get_status(run_id)
        latest_deployment = latest.deployment or {}
        if (
            latest_deployment.get("requested_at") != deployment.get("requested_at")
            or latest_deployment.get("state") not in _DEPLOYMENT_BUSY_STATES
        ):
            raise ServingError("deployment attempt was superseded before alias activation")
        if is_checkpoint:
            if prev_state in _app._DEPLOYABLE_STATES and latest.state != prev_state:
                raise ServingError(
                    f"run state changed from {prev_state!r} to {latest.state!r} before alias activation"
                )
            if latest.state in {"cancelled", "failed", "dry_run"} and latest.state != prev_state:
                raise ServingError(
                    f"run became {latest.state!r} before checkpoint alias activation"
                )
        elif latest.state != prev_state:
            raise ServingError(
                f"run state changed from {prev_state!r} to {latest.state!r} before alias activation"
            )
        expected_generation = deployment.get("verification_generation")
        if verified_adapter_revision_generation(run_id) != expected_generation:
            raise ServingError("deployment verification generation changed before alias activation")

    def _before_activate(adapter_revision: str, checkpoint: str) -> None:
        nonlocal current
        _assert_activation_fence()
        # smoke is unconditional for real deployments: alias activation only ever follows a
        # verified generation against the immutable revision.
        current = _deployment_state(
            {**current, "adapter_revision": adapter_revision},
            "smoke_testing",
            detail="running bounded fixed-prompt smoke",
        )
        mark_deployment_pending(run_id, current, owner_deployment=deployment)
        smoke_result.update(
            _run_deployment_smoke(
                run_id,
                spec,
                serving_model=adapter_revision,
                expected_checkpoint=checkpoint,
            )
        )
        current = _deployment_state(
            current,
            "reconciling",
            detail="activating alias and reconciling the authoritative target",
            activation_outcome_unknown=True,
        )
        mark_deployment_pending(run_id, current, owner_deployment=deployment)
        # cancellation can revoke the ledger while smoke is blocked, so fence again immediately
        # before deploy_adapter issues the activation request.
        _assert_activation_fence()

    try:
        dep = _app.deploy_adapter(**deploy_kwargs, before_activate=_before_activate)
        activated = True
        current = {**current, **dep.to_dict()}
        current.pop("activation_outcome_unknown", None)
        current["verify"] = True
        current = _deployment_state(
            current,
            "ready",
            detail="immutable revision verified and alias activated",
            **smoke_result,
        )
        verification_generation = current.get("verification_generation")
        current = _public_deployment(current)

        def _commit_ready() -> bool:
            state_guard = prev_state
            if is_checkpoint:
                state_guard = prev_state if prev_state in _app._DEPLOYABLE_STATES else None
                marked = mark_checkpoint_deployed(
                    run_id,
                    current,
                    expect_state=state_guard,
                    verification_generation=verification_generation,
                )
                return marked.deployment == current
            marked = mark_deployed(
                run_id,
                current,
                expect_state=prev_state,
                verification_generation=verification_generation,
            )
            return marked.state == "deployed" and marked.deployment == current

        def _reconcile_commit_miss() -> None:
            # deploy_adapter already flipped the serving alias when this runs, so a lost
            # control-plane cas must never be dropped silently — and the alias is never
            # reverted here (post-promotion recovery reads the authoritative alias; a revert
            # could clobber a newer deployment).
            latest = _app.get_status(run_id)
            latest_deployment = latest.deployment or {}
            owned = latest_deployment.get("requested_at") == deployment.get("requested_at")
            if owned and latest_deployment.get("state") in _DEPLOYMENT_BUSY_STATES:
                # this attempt still owns the record; only the run state moved under the
                # guard. retry the write once against the fresh state.
                if is_checkpoint:
                    marked = mark_checkpoint_deployed(
                        run_id,
                        current,
                        verification_generation=verification_generation,
                    )
                else:
                    marked = mark_deployed(
                        run_id,
                        current,
                        expect_state=latest.state,
                        verification_generation=verification_generation,
                    )
                if marked.deployment == current:
                    return
                latest = marked
                latest_deployment = latest.deployment or {}
            # superseded, undeployed, or uncommittable: a newer actor owns the record now, so
            # log the divergence loudly but never write over what that actor recorded — a
            # clobber here would erase a concurrent final deploy's ready record or resurrect
            # an explicit undeploy.
            divergence = (
                "deployment_record_diverged: serving alias targets "
                f"{current.get('adapter_revision')} but the deployment record moved to "
                f"{latest_deployment.get('state')!r} (run state {latest.state!r}) during "
                "activation; serving alias left as activated"
            )
            print(f"deploy[{run_id}]: {divergence}", flush=True)

        if not _commit_ready():
            _reconcile_commit_miss()
            return
    except Exception as exc:
        if isinstance(exc, ActivationOutcomeUnknown):
            reconciling = _deployment_state(
                current,
                "reconciling",
                error=str(exc),
                detail="alias activation outcome is unknown; authoritative reconciliation required",
                activation_outcome_unknown=True,
            )
            mark_deployment_failed(run_id, reconciling)
            return
        if activated:
            try:
                latest = _app.get_status(run_id)
                latest_deployment = latest.deployment or {}
                if (
                    latest_deployment.get("adapter_revision") == current.get("adapter_revision")
                    and latest_deployment.get("state") in _DEPLOYMENT_READY_STATES
                ):
                    return
                if not _commit_ready():
                    _reconcile_commit_miss()
            except Exception as recovery_exc:
                divergence = (
                    "deployment_record_diverged: serving alias was activated for "
                    f"{current.get('adapter_revision')} but ready-state recovery failed after "
                    f"{exc!r}: {recovery_exc!r}"
                )
                print(f"deploy[{run_id}]: {divergence}", flush=True)
            return
        error = str(exc)
        if not is_checkpoint and isinstance(exc, AdapterConfigMissing):
            steps = [c["step"] for c in _app.list_checkpoints(spec)]
            if steps:
                error = (
                    f"run {run_id} has no run-level adapter at "
                    f"{deployment.get('adapter_hf_prefix')} (the run likely never finalized); "
                    f"deploy a saved checkpoint instead, e.g. `flash deploy "
                    f"{run_id}/step-{steps[-1]}` (available steps: "
                    f"{', '.join(str(step) for step in steps)})"
                )
        failed_source = dict(current)
        if not deployment.get("activation_outcome_unknown"):
            failed_source.pop("activation_outcome_unknown", None)
        failed = _deployment_state(
            failed_source,
            "failed",
            error=error,
            detail="deployment failed; previous working alias was preserved",
        )
        mark_deployment_failed(run_id, failed)


def _finish_deployment(**kwargs) -> None:
    with _app._deploy_lock(kwargs["run_id"]):
        _finish_deployment_unlocked(**kwargs)


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
        if spec.model_revision:
            raise HTTPException(
                status_code=400,
                detail=(
                    "deployment does not support revision-pinned base models; "
                    "train without model_revision to deploy this run"
                ),
            )
        try:
            effective_spec = effective_spec_from_status(status)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        dry_run = _require_bool(payload, "dry_run", False)
        # smoke verification is mandatory for every real deployment: a loadable-but-broken
        # revision must never become the bare-run alias target. reject an explicit opt-out
        # before anything is queued or registered (dry runs never register or activate, so
        # the flag is meaningless there too).
        if _require_bool(payload, "verify", True) is False:
            raise HTTPException(
                status_code=400,
                detail=(
                    "verify=false is not supported: deployment smoke verification is "
                    "mandatory before alias activation"
                ),
            )
        current_deployment = status.deployment or {}
        current_deployment_state = current_deployment.get("state")
        completed_unknown_activation = (
            current_deployment_state == "reconciling"
            and current_deployment.get("activation_outcome_unknown") is True
        )
        if (
            not dry_run
            and current_deployment_state in _DEPLOYMENT_BUSY_STATES
            and not completed_unknown_activation
            and not _deployment_attempt_is_stale(current_deployment)
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} already has a deployment in "
                    f"{current_deployment.get('state')} state; run `flash deployments` "
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
        # Prefer org from the run's own context over the caller's key (operator deploys land on run's owner).
        deploy_org_id = run_org_id(status) or str(key.get("org_id") or "").strip() or None
        previous_deployment = _deployment_predecessor(current_deployment)
        expected_adapter_revision = None
        if not dry_run:
            try:
                expected_adapter_revision, previous_deployment = _activation_predecessor(
                    run_id, current_deployment
                )
            except ServingError as exc:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "code": "alias_reconciliation_failed",
                        "run_id": run_id,
                        "retryable": True,
                        "message": str(exc),
                    },
                ) from exc
        deploy_kwargs = {
            "run_id": run_id,
            "model": spec.model,
            "hf_repo": spec.train.hf_repo,
            "adapter_prefix": deploy_prefix,
            "dry_run": dry_run,
            "lora_rank": effective_spec.train.lora_rank,
            # a run trained with thinking serves with thinking (per-run parity)
            "thinking": spec.thinking,
            # a run trained with structured_outputs serves under the same grammar. thinking runs
            # are registered only after serving advertises deferred post-reasoning constraints.
            "structured_outputs": spec.train.structured_outputs,
            "org_id": deploy_org_id,
            "checkpoint_step": checkpoint_step,
            "expected_adapter_revision": expected_adapter_revision,
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
                spec.model,
                effective_spec.train.lora_rank,
                rank_source="effective prepared LoRA rank",
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            dep_dict = _app.deployment_record(
                run_id=run_id,
                model=spec.model,
                adapter_prefix=deploy_prefix,
                state="queued",
                checkpoint_step=checkpoint_step,
            ).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        dep_dict = _deployment_state(
            dep_dict,
            "queued",
            detail="deployment queued",
            verify=True,
            requested_at=time.time(),
        )
        dep_dict["verification_generation"] = verified_adapter_revision_generation(run_id)
        if current_deployment.get("activation_outcome_unknown"):
            dep_dict["activation_outcome_unknown"] = True
        if is_checkpoint:
            dep_dict["checkpoint_step"] = checkpoint_step
        if previous_deployment:
            dep_dict["previous_deployment"] = previous_deployment
        marked = mark_deployment_pending(run_id, dep_dict, expect_state=prev_state)
        if marked.deployment != dep_dict:
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
            "deployment": dep_dict,
            "prev_state": prev_state,
        }
    ran_sync = _app.start_deployment_job(_finish_deployment, **job_kwargs)
    if ran_sync:
        return _public_deployment(_app.get_status(run_id).deployment or dep_dict)
    return _public_deployment(dep_dict)


@router.delete("/v1/runs/{run_id}/deploy")
def undeploy(run_id: str, key: Annotated[dict, Depends(require_key)]):
    with _app._deploy_lock(run_id):
        owned_run(run_id, key)
        try:
            result = _app.undeploy_adapter(run_id)
        except ServingError as exc:
            mark_deployment_revocation_failed(run_id, str(exc))
            raise HTTPException(
                status_code=502,
                detail={
                    "code": "deployment_revocation_failed",
                    "run_id": run_id,
                    "retryable": True,
                    "message": str(exc),
                },
            ) from exc
        mark_undeployed(run_id)
        return result


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
                base_model_revision=spec.model_revision,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ServingError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
    # best-effort product-analytics report: exports never otherwise touch the
    # platform backend (the copy is hf-to-hf inside flash).
    with contextlib.suppress(Exception):
        from flash.server.run_registry import record_model_exported

        record_model_exported(
            status=status,
            key=key,
            repository=repository,
            url=url,
            step=checkpoint_step if is_checkpoint else None,
        )
    result = {
        "run_id": run_id,
        "adapter_id": run_id,
        "repository": repository,
        "url": url,
        "source": f"{run_id}/step-{checkpoint_step}" if is_checkpoint else run_id,
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
    adapter_revision = payload.get("adapter_revision")
    serving_model = run_id
    verified_revisions = _verified_adapter_revisions(status)
    if adapter_revision is not None:
        parsed_revision = (
            parse_adapter_revision(adapter_revision) if isinstance(adapter_revision, str) else None
        )
        if parsed_revision is None:
            raise HTTPException(
                status_code=400,
                detail="adapter_revision must be a full immutable adapter revision",
            )
        if parsed_revision[0] != run_id:
            raise HTTPException(
                status_code=400,
                detail=f"adapter_revision belongs to run {parsed_revision[0]}, not {run_id}",
            )
        serving_model = adapter_revision.strip()
        if serving_model not in verified_revisions:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"adapter_revision {serving_model} has not passed a successful deployment smoke"
                ),
            )
    spec = JobSpec.from_dict(status.spec)
    deployment = status.deployment or {}
    deployment_state = deployment.get("state")
    ready_deployment = _previous_ready_deployment(deployment)
    has_ready_deploy = adapter_revision is not None or ready_deployment is not None
    if adapter_revision is None and ready_deployment is not None:
        ready_revision = ready_deployment.get("adapter_revision")
        parsed_ready_revision = (
            parse_adapter_revision(ready_revision) if isinstance(ready_revision, str) else None
        )
        has_ready_deploy = bool(
            parsed_ready_revision is not None
            and parsed_ready_revision[0] == run_id
            and ready_revision in verified_revisions
        )
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
                    run_id=serving_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking=spec.thinking,
                ),
                media_type="text/plain; charset=utf-8",
            )
        return _app.serve_chat(
            run_id=serving_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=spec.thinking,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"inference failure: {exc}") from exc
