"""Which adapter revision a serving request actually means.

A run can have several deployable artifacts -- the final adapter, a numbered checkpoint, and the
revision a previous deployment is still serving -- and the routes have to agree on which one a
bare run id, an explicit `run/step-N`, or a rollback target resolves to. The predecessor lookups
here are what let a failed activation fall back to the revision that was live before it.

Split out of `flash.server.routes.serving` to keep that module under the file-size limit.
"""

from __future__ import annotations

import re

from fastapi import HTTPException

from flash.content.multimodal import messages_with_image_data_uris, normalize_prompt_images
from flash.core.spec import JobSpec
from flash.runner.lifecycle.state import adapter_prefix
from flash.runner.results.checkpoints import checkpoint_adapter_prefix
from flash.runner.results.verified_revisions import read_verified_adapter_revisions
from flash.schema import parse_adapter_revision
from flash.serve.contract.errors import ServingError
from flash.serve.request.openai import OpenAIRequestError, parse_chat_request
from flash.server.asgi import app as _app

# defined here rather than in the route module because that one imports this one: the routes read
# them back through their own namespace, so there is still exactly one definition.
_DEPLOYMENT_BUSY_STATES = {"queued", "smoke_testing", "reconciling"}
# the nested `deployment["state"]` vocabulary, not the top-level `status.state` one: `deployed` is
# live in the latter and has never been written in the former. the cli copy of this set keeps both
# spellings, because it reads the field from a remote plane whose version it does not control.
_DEPLOYMENT_READY_STATES = {"ready"}


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


def _confirmed_active_failed_deployment(deployment: object, *, run_id: str) -> dict | None:
    if not isinstance(deployment, dict):
        return None
    revision = deployment.get("adapter_revision")
    parsed = parse_adapter_revision(revision) if isinstance(revision, str) else None
    if (
        deployment.get("state") == "failed"
        and deployment.get("alias_activation_confirmed") is True
        and parsed is not None
        and parsed[0] == run_id
    ):
        return dict(deployment)
    return None


def _deployment_predecessor(deployment: dict, *, run_id: str) -> dict | None:
    ready = _previous_ready_deployment(deployment)
    if ready is not None:
        return ready
    active_failed = _confirmed_active_failed_deployment(deployment, run_id=run_id)
    if active_failed is not None:
        return active_failed
    if deployment.get("activation_outcome_unknown"):
        previous = deployment.get("previous_deployment")
        if isinstance(previous, dict) and previous.get("state") in _DEPLOYMENT_READY_STATES:
            return dict(previous)
        return _confirmed_active_failed_deployment(previous, run_id=run_id)
    return None


def _activation_predecessor(run_id: str, deployment: dict) -> tuple[str | None, dict | None]:
    if not deployment.get("activation_outcome_unknown"):
        predecessor = _deployment_predecessor(deployment, run_id=run_id)
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
    failed_predecessor = next(
        (
            confirmed
            for candidate in candidates
            if candidate is not None
            and candidate.get("adapter_revision") == target
            and (confirmed := _confirmed_active_failed_deployment(candidate, run_id=run_id))
            is not None
        ),
        None,
    )
    predecessor = failed_predecessor or next(
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
    if failed_predecessor is None:
        predecessor.pop("error", None)
        predecessor["state"] = (
            "ready" if target in read_verified_adapter_revisions(run_id) else "reconciling"
        )
    return target, predecessor


def _verified_adapter_revisions(status) -> set[str]:
    return set(read_verified_adapter_revisions(status.run_id))


def _verified_step_index(verified_revisions: set[str], run_id: str) -> dict[int | None, list[str]]:
    index: dict[int | None, list[str]] = {}
    for revision in verified_revisions:
        parsed = parse_adapter_revision(revision)
        if parsed is not None and parsed[0] == run_id:
            index.setdefault(parsed[1], []).append(revision)
    return index


def _format_deployed_steps(index: dict[int | None, list[str]]) -> str:
    labels = [str(step) for step in sorted(step for step in index if step is not None)]
    if None in index:
        labels.append("final")
    return ", ".join(labels) or "none"


def _resolve_explicit_chat_revision(
    run_id: str,
    adapter_revision,
    step,
    verified_revisions: set[str],
    *,
    preferred_revision: str | None = None,
) -> str | None:
    """Resolve an explicit chat target to a verified same-run immutable revision to pin, or None
    for bare-alias chat. 400 on malformed/ambiguous targets, 409 on not-yet-verified targets."""
    if adapter_revision is not None and step is not None:
        raise HTTPException(
            status_code=400, detail="pass either adapter_revision or step, not both"
        )
    if adapter_revision is not None:
        parsed_revision = (
            parse_adapter_revision(adapter_revision) if isinstance(adapter_revision, str) else None
        )
        if parsed_revision is None:
            raise HTTPException(
                status_code=400, detail="adapter_revision must be a full immutable adapter revision"
            )
        if parsed_revision[0] != run_id:
            raise HTTPException(
                status_code=400,
                detail=f"adapter_revision belongs to run {parsed_revision[0]}, not {run_id}",
            )
        revision = adapter_revision.strip()
        if revision not in verified_revisions:
            raise HTTPException(
                status_code=409,
                detail=f"adapter_revision {revision} has not passed a successful deployment smoke",
            )
        return revision
    if step is not None:
        want = _parse_checkpoint_step(step)
        index = _verified_step_index(verified_revisions, run_id)
        matches = index.get(want, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            if preferred_revision in matches:
                return preferred_revision
            raise HTTPException(
                status_code=409,
                detail=(
                    f"run {run_id} has multiple verified revisions at step {want}; chat the full "
                    "immutable adapter revision from `flash models deployments`"
                ),
            )
        raise HTTPException(
            status_code=409,
            detail=(
                f"run {run_id} has no deployed checkpoint at step {want}; deploy it first with "
                f"`flash models deploy {run_id}/step-{want}` "
                f"(currently deployed steps: {_format_deployed_steps(index)})"
            ),
        )
    return None


def _authorized_chat_revision(
    run_id: str,
    deployment: dict,
    adapter_revision,
    step,
    verified_revisions: set[str],
) -> str | None:
    """return the one verified immutable revision this chat request may serve."""

    ready_deployment = _previous_ready_deployment(deployment)
    ready_revision = (
        ready_deployment.get("adapter_revision") if ready_deployment is not None else None
    )
    preferred = ready_revision if isinstance(ready_revision, str) else None
    explicit = _resolve_explicit_chat_revision(
        run_id,
        adapter_revision,
        step,
        verified_revisions,
        preferred_revision=preferred,
    )
    if explicit is not None:
        return explicit
    if not isinstance(ready_revision, str) or ready_revision not in verified_revisions:
        return None
    parsed = parse_adapter_revision(ready_revision)
    if parsed is None or parsed[0] != run_id:
        return None
    return ready_revision


def _spec_is_unservable(status) -> bool:
    """Whether the serving routes' own `JobSpec.from_dict` would reject this run's persisted spec.

    Asked with the same call the chat and deploy routes make, so the answer cannot drift from what
    they will actually do with the record.
    """
    try:
        JobSpec.from_dict(status.spec)
    except Exception:
        return True
    return False


def _chat_messages_from_payload(payload: dict) -> list[dict]:
    """compatibility seam for tests and callers that validate only message payloads."""

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
    try:
        parse_chat_request(
            payload,
            require_model=False,
            allow_managed_selectors=True,
        )
    except OpenAIRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raw = payload["messages"]
    assert isinstance(raw, list)
    return _managed_chat_messages(raw)


def _managed_chat_messages(messages: list[dict]) -> list[dict]:
    """apply the training image contract after the shared message grammar."""

    try:
        normalized = normalize_prompt_images({}, messages, None)
        if not normalized.descriptors:
            return messages
        return messages_with_image_data_uris(normalized.messages, normalized.descriptors, None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid image request: {exc}") from exc


def _parse_checkpoint_step(raw_step) -> int:
    """Validate a raw JSON checkpoint step into a non-negative int; 400 on bad input.

    accepts a bare int, an integral float, or a bounded digit string; rejects bool,
    non-integer floats, negatives, and unicode/oversized digit strings that would crash int().
    """
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
    if want is None or want < 0:
        raise HTTPException(status_code=400, detail=f"invalid checkpoint step: {raw_step!r}")
    return want


def _resolve_deploy_step(run_id: str, spec, raw_step) -> int | None:
    """Validate optional checkpoint step; returns int or None (final adapter). 400 on bad step, 404 on missing."""
    if raw_step is None:
        return None

    want = _parse_checkpoint_step(raw_step)
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
