"""explicit checkpoint resolution for managed serving routes."""

from __future__ import annotations

from fastapi import HTTPException

from flash.content.multimodal import messages_with_image_data_uris, normalize_prompt_images
from flash.core.spec import JobSpec
from flash.runner.lifecycle.state import adapter_prefix
from flash.runner.results.checkpoints import checkpoint_adapter_prefix
from flash.runner.results.verified_revisions import read_verified_checkpoints
from flash.schema import format_checkpoint_ref, parse_checkpoint_ref
from flash.server.asgi import app as _app

_DEPLOYMENT_BUSY_STATES = {"queued", "smoke_testing"}
_DEPLOYMENT_READY_STATES = {"ready"}


def _verified_checkpoints(status) -> set[str]:
    return set(read_verified_checkpoints(status.run_id))


def _verified_step_index(checkpoints: set[str], run_id: str) -> dict[int | None, list[str]]:
    index: dict[int | None, list[str]] = {}
    for checkpoint_id in checkpoints:
        parsed = parse_checkpoint_ref(checkpoint_id)
        if parsed is not None and parsed[0] == run_id:
            index.setdefault(parsed[1], []).append(checkpoint_id)
    return index


def _format_deployed_steps(index: dict[int | None, list[str]]) -> str:
    labels = [str(step) for step in sorted(step for step in index if step is not None)]
    if None in index:
        labels.append("final")
    return ", ".join(labels) or "none"


def _authorized_chat_checkpoint(
    run_id: str,
    deployment: dict,
    checkpoint_id: object,
    verified_checkpoints: set[str],
) -> str:
    if not isinstance(checkpoint_id, str) or parse_checkpoint_ref(checkpoint_id) is None:
        raise HTTPException(
            status_code=400,
            detail="checkpoint_id must be `<run_id>/final` or `<run_id>/step-N`",
        )
    parsed = parse_checkpoint_ref(checkpoint_id)
    assert parsed is not None
    if parsed[0] != run_id:
        raise HTTPException(
            status_code=400,
            detail=f"checkpoint_id belongs to run {parsed[0]}, not {run_id}",
        )
    if checkpoint_id not in verified_checkpoints:
        raise HTTPException(
            status_code=409,
            detail=f"checkpoint {checkpoint_id} has not passed a successful deployment smoke",
        )
    # verification is checkpoint-scoped. a newer sibling may queue or fail without revoking this
    # ready checkpoint, so the mutable run-level deployment summary is not an authorization source.
    return checkpoint_id


def _spec_is_unservable(status) -> bool:
    try:
        JobSpec.from_dict(status.spec)
    except Exception:
        return True
    return False


def _managed_chat_messages(messages: list[dict]) -> list[dict]:
    try:
        normalized = normalize_prompt_images({}, messages, None)
        if not normalized.descriptors:
            return messages
        return messages_with_image_data_uris(normalized.messages, normalized.descriptors, None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid image request: {exc}") from exc


def _resolve_deploy_step(run_id: str, spec, checkpoint_id: object) -> int | None:
    if not isinstance(checkpoint_id, str):
        raise HTTPException(status_code=400, detail="checkpoint_id is required")
    parsed = parse_checkpoint_ref(checkpoint_id)
    if parsed is None:
        raise HTTPException(
            status_code=400,
            detail="checkpoint_id must be `<run_id>/final` or `<run_id>/step-N`",
        )
    if parsed[0] != run_id:
        raise HTTPException(
            status_code=400,
            detail=f"checkpoint_id belongs to run {parsed[0]}, not {run_id}",
        )
    step = parsed[1]
    if step is None:
        return None
    checkpoints = _app.list_checkpoints(spec)
    if any(candidate["step"] == step for candidate in checkpoints):
        return step
    available = ", ".join(str(candidate["step"]) for candidate in checkpoints) or "none"
    raise HTTPException(
        status_code=404,
        detail=f"run {run_id} has no deployable checkpoint at step {step} (available: {available})",
    )


def _resolve_deployable_target(
    run_id: str,
    spec,
    status,
    checkpoint_id: object,
    *,
    action: str,
    enforce_state: bool,
) -> tuple[int | None, bool, str]:
    checkpoint_step = _resolve_deploy_step(run_id, spec, checkpoint_id)
    is_checkpoint = checkpoint_step is not None
    if enforce_state and is_checkpoint and status.state == "dry_run":
        raise HTTPException(
            status_code=409,
            detail=f"run {run_id} is 'dry_run'; dry-run runs cannot be {action}ed",
        )
    if enforce_state and not is_checkpoint and status.state not in _app._DEPLOYABLE_STATES:
        detail = (
            f"run {run_id} is {status.state!r}; only finished runs with trained adapter "
            f"artifacts can be {'deployed' if action == 'deploy' else 'exported'}"
        )
        raise HTTPException(status_code=409, detail=detail)
    prefix = (
        checkpoint_adapter_prefix(spec, checkpoint_step) if is_checkpoint else adapter_prefix(spec)
    )
    if format_checkpoint_ref(run_id, checkpoint_step) != checkpoint_id:
        raise HTTPException(status_code=400, detail="checkpoint_id is not canonical")
    return checkpoint_step, is_checkpoint, prefix
