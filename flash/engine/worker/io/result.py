"""publish the immutable terminal result for one fenced worker attempt."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time

from flash._internal.diagnostics import sanitize_diagnostic
from flash.engine.worker.io import hf as hf_io
from flash.engine.worker.runtime import state
from flash.runner.lifecycle.protocol import (
    ResultManifest,
    attempt_prefix,
    canonical_bytes,
    result_path,
)

_RESULT_COMMIT_ATTEMPTS = 3


def _source_attestation() -> dict | None:
    raw = os.environ.get("FLASH_SOURCE_SNAPSHOT_JSON")
    if not raw:
        return None
    from flash.snapshot.archive import parse_descriptor, source_attestation

    descriptor = parse_descriptor(json.loads(raw))
    return source_attestation(
        descriptor,
        run_id=state.RUN_ID,
        attempt=state.ATTEMPT,
        fence=state.FENCE,
    )


def _write_immutable(payload: bytes) -> str:
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    directory = "/tmp/flash-result"
    os.makedirs(directory, exist_ok=True)
    final = os.path.join(directory, digest + ".json")
    if os.path.exists(final):
        with open(final, "rb") as handle:
            if handle.read() != payload:
                raise RuntimeError("immutable result digest conflict")
        return final
    fd, temporary = tempfile.mkstemp(dir=directory, prefix="result-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, final)
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return final
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def _download_result(path: str, *, revision: str) -> ResultManifest:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=state.HF_REPO,
        repo_type="dataset",
        filename=path,
        revision=revision,
        token=os.environ.get("HF_TOKEN"),
        force_download=True,
    )
    with open(local, "rb") as handle:
        manifest = ResultManifest.from_dict(json.loads(handle.read()))
    if result_path(manifest) != path:
        raise RuntimeError("existing result does not match its immutable path")
    return manifest


def _existing_result(api, *, revision: str) -> ResultManifest | None:
    base = f"{attempt_prefix(state.PHASE, state.RUN_ID, state.ATTEMPT, state.FENCE)}/result/"
    paths = [
        path
        for path in api.list_repo_files(
            repo_id=state.HF_REPO,
            repo_type="dataset",
            revision=revision,
        )
        if isinstance(path, str) and path.startswith(base)
    ]
    if not paths:
        return None
    if len(paths) != 1:
        raise RuntimeError("conflicting result manifests exist for one fenced attempt")
    return _download_result(paths[0], revision=revision)


def _same_terminal_claim(existing: ResultManifest, proposed: ResultManifest) -> bool:
    existing_values = existing.to_dict()
    proposed_values = proposed.to_dict()
    existing_values.pop("finished_at")
    proposed_values.pop("finished_at")
    return existing_values == proposed_values


def _accept_existing(existing: ResultManifest, proposed: ResultManifest) -> ResultManifest:
    if not _same_terminal_claim(existing, proposed):
        raise RuntimeError("a conflicting terminal result already exists for this fenced attempt")
    return existing


def _publish_exactly_once(manifest: ResultManifest, local_path: str) -> ResultManifest:
    if not state.HF_REPO:
        return manifest
    from huggingface_hub import CommitOperationAdd

    api = hf_io.hf_api()
    path = result_path(manifest)
    last_error: Exception | None = None
    for attempt in range(_RESULT_COMMIT_ATTEMPTS):
        try:
            hf_io._require_hf_deadline_allowance()
            revision = str(
                api.repo_info(repo_id=state.HF_REPO, repo_type="dataset").sha or ""
            ).strip()
            if not revision:
                raise RuntimeError("artifact repository revision is unavailable")
            existing = _existing_result(api, revision=revision)
            if existing is not None:
                return _accept_existing(existing, manifest)
            api.create_commit(
                repo_id=state.HF_REPO,
                repo_type="dataset",
                operations=[CommitOperationAdd(path_in_repo=path, path_or_fileobj=local_path)],
                commit_message=(
                    f"record terminal result for {state.RUN_ID} "
                    f"attempt {state.ATTEMPT} fence {state.FENCE}"
                ),
                parent_commit=revision,
            )
            return manifest
        except RuntimeError as exc:
            if "conflicting terminal result" in str(exc) or "conflicting result manifests" in str(
                exc
            ):
                raise
            last_error = exc
        except Exception as exc:
            last_error = exc
        if attempt + 1 < _RESULT_COMMIT_ATTEMPTS and not hf_io._sleep_with_hf_deadline(
            5 * (attempt + 1)
        ):
            break
    detail = sanitize_diagnostic(last_error, limit=500)
    raise hf_io.RetriableInfraError(
        f"required exactly-once result publication failed: {detail}"
    ) from last_error


def _latest_local_progress() -> dict:
    directory = "/tmp/flash-progress"
    candidates: list[dict] = []
    with contextlib.suppress(OSError):
        for name in os.listdir(directory):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(directory, name), "rb") as handle:
                    value = json.loads(handle.read())
            except (OSError, TypeError, ValueError):
                continue
            if (
                isinstance(value, dict)
                and value.get("run_id") == state.RUN_ID
                and value.get("phase_namespace") == state.PHASE
                and value.get("attempt_id") == state.ATTEMPT
                and value.get("fence") == state.FENCE
            ):
                candidates.append(value)
    if not candidates:
        return {}
    return max(candidates, key=lambda value: int(value.get("sequence") or 0))


def _publish_supervisor_result(
    *,
    outcome: str,
    failure_class: str | None,
    started_at: float,
    error: str,
) -> ResultManifest:
    """publish a terminal supervisor decision after exact worker process-group termination."""
    progress = _latest_local_progress()
    return publish_result(
        outcome=outcome,
        failure_class=failure_class,
        started_at=started_at,
        training_entered=progress.get("training_entered") is True,
        completed_steps=int(progress.get("completed_steps") or 0),
        metrics=progress.get("metrics") if isinstance(progress.get("metrics"), dict) else {},
        checkpoint=(
            progress.get("checkpoint") if isinstance(progress.get("checkpoint"), dict) else {}
        ),
        artifacts={"console": f"console_{state.PHASE}.txt"},
        diagnostics={"error": error},
    )


def publish_deadline_result(*, started_at: float) -> ResultManifest:
    """publish a deadline outcome after exact top-level process-group termination."""
    return _publish_supervisor_result(
        outcome="deadline",
        failure_class="deadline",
        started_at=started_at,
        error="fixed work deadline expired",
    )


def publish_cancelled_result(*, started_at: float) -> ResultManifest:
    """publish cancellation after exact top-level process-group termination."""
    return _publish_supervisor_result(
        outcome="cancelled",
        failure_class=None,
        started_at=started_at,
        error="worker attempt cancelled",
    )


def publish_result(
    *,
    outcome: str,
    failure_class: str | None,
    started_at: float,
    training_entered: bool,
    completed_steps: int,
    metrics: dict | None = None,
    checkpoint: dict | None = None,
    artifacts: dict | None = None,
    diagnostics: dict | None = None,
) -> ResultManifest:
    """publish the only worker terminal authority for the current attempt."""
    manifest = ResultManifest(
        run_id=state.RUN_ID,
        phase_namespace=state.PHASE,
        attempt_id=state.ATTEMPT,
        fence=state.FENCE,
        outcome=outcome,
        failure_class=failure_class,
        started_at=started_at,
        finished_at=time.time(),
        training_entered=bool(training_entered),
        completed_steps=max(0, int(completed_steps)),
        metrics=dict(metrics or {}),
        checkpoint=dict(checkpoint or {}),
        artifacts=dict(artifacts or {}),
        source_attestation=_source_attestation(),
        diagnostics={
            str(key)[:120]: sanitize_diagnostic(value, limit=1000)
            for key, value in (diagnostics or {}).items()
        },
    )
    payload = canonical_bytes(manifest.to_dict())
    local_path = _write_immutable(payload)
    return _publish_exactly_once(manifest, local_path)
