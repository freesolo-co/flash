"""Which adapter revision a serving request actually means.

A run can have several deployable artifacts -- the final adapter, a numbered checkpoint, and the
revision a previous deployment is still serving -- and every caller has to agree on which one a
bare run id, an explicit `run/step-N`, or a rollback target resolves to. The predecessor lookups
here are what let a failed activation fall back to the revision that was live before it.

Pure resolution: everything here takes the records and repositories it needs as arguments and
raises domain errors. Nothing imports FastAPI.
"""

from __future__ import annotations

import re

from flash.core.spec import JobSpec
from flash.schema import parse_adapter_revision
from flash.serve.deploy import ServingError
from flash.server.domain.deployment_ports import (
    ArtifactRepository,
    DeploymentConflict,
    DeploymentNotFound,
    DeploymentRepository,
    InvalidDeploymentRequest,
    RunRepository,
    ServingGateway,
)

DEPLOYMENT_BUSY_STATES = frozenset({"queued", "smoke_testing", "reconciling"})
DEPLOYMENT_READY_STATES = frozenset({"ready", "deployed"})


def previous_ready_deployment(deployment: dict) -> dict | None:
    state = deployment.get("state")
    if state in DEPLOYMENT_READY_STATES:
        return dict(deployment)
    if state not in DEPLOYMENT_BUSY_STATES or state == "reconciling":
        return None
    previous = deployment.get("previous_deployment")
    if isinstance(previous, dict) and previous.get("state") in DEPLOYMENT_READY_STATES:
        return dict(previous)
    return None


def confirmed_active_failed_deployment(deployment: object, *, run_id: str) -> dict | None:
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


def deployment_predecessor(deployment: dict, *, run_id: str) -> dict | None:
    ready = previous_ready_deployment(deployment)
    if ready is not None:
        return ready
    active_failed = confirmed_active_failed_deployment(deployment, run_id=run_id)
    if active_failed is not None:
        return active_failed
    if deployment.get("activation_outcome_unknown"):
        previous = deployment.get("previous_deployment")
        if isinstance(previous, dict) and previous.get("state") in DEPLOYMENT_READY_STATES:
            return dict(previous)
        return confirmed_active_failed_deployment(previous, run_id=run_id)
    return None


def activation_predecessor(
    run_id: str,
    deployment: dict,
    *,
    serving: ServingGateway,
    deployments: DeploymentRepository,
) -> tuple[str | None, dict | None]:
    if not deployment.get("activation_outcome_unknown"):
        predecessor = deployment_predecessor(deployment, run_id=run_id)
        revision = predecessor.get("adapter_revision") if predecessor is not None else None
        return (revision if isinstance(revision, str) else None), predecessor

    target = serving.adapter_alias_target(run_id)
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
            and (confirmed := confirmed_active_failed_deployment(candidate, run_id=run_id))
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
            "ready" if target in set(deployments.verified_revisions(run_id)) else "reconciling"
        )
    return target, predecessor


def verified_step_index(verified_revisions: set[str], run_id: str) -> dict[int | None, list[str]]:
    index: dict[int | None, list[str]] = {}
    for revision in verified_revisions:
        parsed = parse_adapter_revision(revision)
        if parsed is not None and parsed[0] == run_id:
            index.setdefault(parsed[1], []).append(revision)
    return index


def format_deployed_steps(index: dict[int | None, list[str]]) -> str:
    labels = [str(step) for step in sorted(step for step in index if step is not None)]
    if None in index:
        labels.append("final")
    return ", ".join(labels) or "none"


def _resolve_pinned_revision(
    run_id: str, adapter_revision, verified_revisions: set[str]
) -> str | None:
    parsed_revision = (
        parse_adapter_revision(adapter_revision) if isinstance(adapter_revision, str) else None
    )
    if parsed_revision is None:
        raise InvalidDeploymentRequest("adapter_revision must be a full immutable adapter revision")
    if parsed_revision[0] != run_id:
        raise InvalidDeploymentRequest(
            f"adapter_revision belongs to run {parsed_revision[0]}, not {run_id}"
        )
    revision = adapter_revision.strip()
    if revision not in verified_revisions:
        raise DeploymentConflict(
            f"adapter_revision {revision} has not passed a successful deployment smoke"
        )
    return revision


def _resolve_step_revision(
    run_id: str, step, verified_revisions: set[str], preferred_revision: str | None
) -> str:
    want = parse_checkpoint_step(step)
    index = verified_step_index(verified_revisions, run_id)
    matches = index.get(want, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        if preferred_revision in matches:
            return preferred_revision
        raise DeploymentConflict(
            f"run {run_id} has multiple verified revisions at step {want}; chat the full "
            "immutable adapter revision from `flash models deployments`"
        )
    raise DeploymentConflict(
        f"run {run_id} has no deployed checkpoint at step {want}; deploy it first with "
        f"`flash models deploy {run_id}/step-{want}` "
        f"(currently deployed steps: {format_deployed_steps(index)})"
    )


def resolve_explicit_chat_revision(
    run_id: str,
    adapter_revision,
    step,
    verified_revisions: set[str],
    *,
    preferred_revision: str | None = None,
) -> str | None:
    """Resolve an explicit chat target to a verified same-run immutable revision to pin, or None
    for bare-alias chat. Invalid on malformed/ambiguous targets, conflict on unverified ones."""
    if adapter_revision is not None and step is not None:
        raise InvalidDeploymentRequest("pass either adapter_revision or step, not both")
    if adapter_revision is not None:
        return _resolve_pinned_revision(run_id, adapter_revision, verified_revisions)
    if step is not None:
        return _resolve_step_revision(run_id, step, verified_revisions, preferred_revision)
    return None


def spec_is_unservable(status) -> bool:
    """Whether the serving paths' own `JobSpec.from_dict` would reject this run's persisted spec.

    Asked with the same call deploy and chat make, so the answer cannot drift from what they will
    actually do with the record.
    """
    try:
        JobSpec.from_dict(status.spec)
    except Exception:
        return True
    return False


def parse_checkpoint_step(raw_step) -> int:
    """Validate a raw JSON checkpoint step into a non-negative int; invalid on bad input.

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
        raise InvalidDeploymentRequest(f"invalid checkpoint step: {raw_step!r}")
    return want


def resolve_deploy_step(
    run_id: str, spec, raw_step, *, artifacts: ArtifactRepository
) -> int | None:
    """Validate optional checkpoint step; returns int or None (final adapter).

    Invalid on a bad step, not-found on a step with no artifact.
    """
    if raw_step is None:
        return None

    want = parse_checkpoint_step(raw_step)
    checkpoints = artifacts.list_checkpoints(spec)
    if any(c["step"] == want for c in checkpoints):
        return want
    available = ", ".join(str(c["step"]) for c in checkpoints) or "none"
    raise DeploymentNotFound(
        f"run {run_id} has no deployable checkpoint at step {want} (available: {available})"
    )


def resolve_deployable_target(
    run_id: str,
    spec,
    status,
    raw_step,
    *,
    action: str,
    enforce_state: bool,
    artifacts: ArtifactRepository,
    runs: RunRepository,
) -> tuple[int | None, bool, str]:
    """Resolve the deploy/export target and gate final-adapter targets on training state."""
    checkpoint_step = resolve_deploy_step(run_id, spec, raw_step, artifacts=artifacts)
    is_checkpoint = checkpoint_step is not None
    # A resolved checkpoint step has already proven a servable adapter exists; only final-adapter
    # deploy/export needs the run-state gate because the final adapter exists only after completion.
    if enforce_state and is_checkpoint and status.state == "dry_run":
        raise DeploymentConflict(f"run {run_id} is 'dry_run'; dry-run runs cannot be {action}ed")
    if enforce_state and not is_checkpoint and status.state not in runs.deployable_states:
        raise DeploymentConflict(
            f"run {run_id} is {status.state!r}; only finished runs with "
            f"trained adapter artifacts can be {'deployed' if action == 'deploy' else 'exported'}"
        )
    prefix = (
        artifacts.checkpoint_adapter_prefix(spec, checkpoint_step)
        if is_checkpoint
        else artifacts.adapter_prefix(spec)
    )
    return checkpoint_step, is_checkpoint, prefix
