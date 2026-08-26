"""read immutable fenced attempt progress and result records from hugging face."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from flash.providers.core.base import PollResult
from flash.runner.lifecycle.protocol import (
    ProgressRecord,
    ResultManifest,
    digest_record,
    progress_path,
    receipt,
    result_path,
)
from flash.snapshot.archive import TERMINAL_ATTESTATION_KEY, validate_attestation

_PROGRESS_NAME = re.compile(r"^(?P<sequence>[0-9]{20})-(?P<digest>[0-9a-f]{64})\.json$")
_RESULT_NAME = re.compile(r"^(?P<digest>[0-9a-f]{64})\.json$")
_PROGRESS_CACHE_LIMIT = 256
_PROGRESS_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _ProgressCacheEntry:
    record: ProgressRecord
    identities: tuple[tuple[int, str, str], ...]


_PROGRESS_CACHE: OrderedDict[tuple[str, str], _ProgressCacheEntry] = OrderedDict()


class AttemptArtifactError(RuntimeError):
    """an immutable attempt artifact could not be verified."""


@dataclass(frozen=True)
class AttemptArtifacts:
    revision: str
    observed_at: float
    progress: dict | None
    result: dict | None


def _download_bytes(hf_repo: str, path: str, *, revision: str) -> bytes:
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=hf_repo,
        repo_type="dataset",
        filename=path,
        revision=revision,
        token=os.environ.get("HF_TOKEN"),
        force_download=True,
    )
    with open(local, "rb") as handle:
        return handle.read()


def _repo_snapshot(hf_repo: str) -> tuple[str, list[str]]:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    revision = str(api.repo_info(repo_id=hf_repo, repo_type="dataset").sha or "").strip()
    if not revision:
        raise AttemptArtifactError("artifact repository revision is unavailable")
    paths = api.list_repo_files(repo_id=hf_repo, repo_type="dataset", revision=revision)
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise AttemptArtifactError("artifact repository listing is malformed")
    return revision, paths


def _cached_progress(hf_repo: str, prefix: str) -> _ProgressCacheEntry | None:
    key = (hf_repo, prefix)
    with _PROGRESS_CACHE_LOCK:
        entry = _PROGRESS_CACHE.get(key)
        if entry is not None:
            _PROGRESS_CACHE.move_to_end(key)
        return entry


def _cache_progress(
    hf_repo: str,
    prefix: str,
    record: ProgressRecord,
    identities: tuple[tuple[int, str, str], ...],
) -> None:
    key = (hf_repo, prefix)
    with _PROGRESS_CACHE_LOCK:
        _PROGRESS_CACHE[key] = _ProgressCacheEntry(record, identities)
        _PROGRESS_CACHE.move_to_end(key)
        while len(_PROGRESS_CACHE) > _PROGRESS_CACHE_LIMIT:
            _PROGRESS_CACHE.popitem(last=False)


def _decode_progress(
    hf_repo: str,
    paths: list[str],
    *,
    prefix: str,
    revision: str,
    observed_at: float,
) -> dict | None:
    candidates: dict[int, list[tuple[str, str]]] = {}
    base = f"{prefix}/progress/"
    for path in paths:
        if not path.startswith(base):
            continue
        match = _PROGRESS_NAME.fullmatch(path[len(base) :])
        if match is None:
            continue
        sequence = int(match.group("sequence"))
        candidates.setdefault(sequence, []).append((path, match.group("digest")))
    cached = _cached_progress(hf_repo, prefix)
    previous = cached.record if cached is not None else None
    start_sequence = 1
    if cached is not None:
        current_identities = tuple(
            (sequence, path, digest)
            for sequence in range(1, cached.record.sequence + 1)
            for path, digest in candidates.get(sequence, ())
        )
        if current_identities == cached.identities:
            start_sequence = cached.record.sequence + 1
        else:
            previous = None
    selected_path = progress_path(previous) if previous is not None else ""
    selected_digest = digest_record(previous.to_dict()) if previous is not None else ""
    for sequence in range(start_sequence, max(candidates, default=0) + 1):
        choices = candidates.get(sequence)
        if choices is None or len(choices) != 1:
            break
        path, expected_digest = choices[0]
        try:
            record = ProgressRecord.from_dict(
                json.loads(_download_bytes(hf_repo, path, revision=revision))
            )
        except Exception:
            break
        if progress_path(record) != path or digest_record(record.to_dict()) != expected_digest:
            break
        if not record.follows(previous):
            break
        previous = record
        selected_path = path
        selected_digest = expected_digest
    if previous is None:
        return None
    identities = tuple(
        (sequence, path, digest)
        for sequence in range(1, previous.sequence + 1)
        for path, digest in candidates.get(sequence, ())
    )
    _cache_progress(hf_repo, prefix, previous, identities)
    return {
        **previous.to_dict(),
        "observed_at": observed_at,
        "receipt": receipt(selected_path, revision, selected_digest),
    }


def _decode_result(
    hf_repo: str,
    paths: list[str],
    *,
    prefix: str,
    revision: str,
    observed_at: float,
    source_snapshot: dict,
) -> dict | None:
    base = f"{prefix}/result/"
    candidates: list[tuple[str, str]] = []
    for path in paths:
        if not path.startswith(base):
            continue
        match = _RESULT_NAME.fullmatch(path[len(base) :])
        if match is None:
            raise AttemptArtifactError("result artifact path is malformed")
        candidates.append((path, match.group("digest")))
    if not candidates:
        return None
    manifests: list[tuple[ResultManifest, str, str]] = []
    for path, expected_digest in candidates:
        payload = _download_bytes(hf_repo, path, revision=revision)
        try:
            manifest = ResultManifest.from_dict(json.loads(payload))
            if (
                result_path(manifest) != path
                or digest_record(manifest.to_dict()) != expected_digest
            ):
                raise ValueError("result digest does not match its immutable path")
            validate_attestation(
                manifest.source_attestation,
                source_snapshot,
                run_id=manifest.run_id,
                attempt=manifest.attempt_id,
                fence=manifest.fence,
            )
        except Exception as exc:
            raise AttemptArtifactError("result manifest is invalid or unverifiable") from exc
        manifests.append((manifest, path, expected_digest))
    if len(manifests) != 1:
        raise AttemptArtifactError("conflicting result manifests exist for one fenced attempt")
    manifest, path, manifest_digest = manifests[0]
    return {
        **manifest.to_dict(),
        "observed_at": observed_at,
        "receipt": receipt(path, revision, manifest_digest),
    }


def poll_result_from_manifest(projection: dict) -> PollResult:
    """translate one verified result manifest into the provider poll contract."""
    manifest = ResultManifest.from_dict(
        {
            key: value
            for key, value in projection.items()
            if key in ResultManifest.__dataclass_fields__
        }
    )
    if manifest.outcome == "succeeded":
        metrics = {**manifest.metrics, TERMINAL_ATTESTATION_KEY: manifest.source_attestation}
        return PollResult(True, metrics=metrics)
    if manifest.outcome == "cancelled":
        return PollResult(False, failure="job_failed", detail="worker reported cancellation")
    if manifest.failure_class == "oom":
        failure = "oom"
    elif manifest.failure_class == "provider_preempted":
        failure = "job_preempted"
    elif manifest.failure_class == "artifact_transport":
        failure = "artifact_transport"
    else:
        failure = "job_failed"
    detail = str(manifest.diagnostics.get("error") or manifest.failure_class or manifest.outcome)
    return PollResult(False, failure=failure, detail=detail[:4096])


def persist_attempt_artifacts(run_id: str, artifacts: AttemptArtifacts) -> None:
    """persist current-fence projections without granting them lifecycle authority."""
    from flash.runner.lifecycle.status import record_progress, record_result

    identity = artifacts.result or artifacts.progress
    if not isinstance(identity, dict):
        return
    attempt_id = identity["attempt_id"]
    fence = identity["fence"]
    if artifacts.progress is not None:
        record_progress(run_id, artifacts.progress, attempt_id=attempt_id, fence=fence)
    if artifacts.result is not None:
        record_result(run_id, artifacts.result, attempt_id=attempt_id, fence=fence)


def read_attempt_artifacts(
    hf_repo: str,
    *,
    phase: str,
    run_id: str,
    attempt_id: int,
    fence: int,
    source_snapshot: dict,
) -> AttemptArtifacts:
    """read one pinned repository snapshot for the exact fenced attempt."""
    if not isinstance(hf_repo, str) or not hf_repo.strip():
        raise AttemptArtifactError("attempt artifact repository is unavailable")
    from flash.runner.lifecycle.protocol import attempt_prefix

    prefix = attempt_prefix(phase, run_id, attempt_id, fence)
    revision, paths = _repo_snapshot(hf_repo)
    observed_at = time.time()
    progress = _decode_progress(
        hf_repo,
        paths,
        prefix=prefix,
        revision=revision,
        observed_at=observed_at,
    )
    result = _decode_result(
        hf_repo,
        paths,
        prefix=prefix,
        revision=revision,
        observed_at=observed_at,
        source_snapshot=source_snapshot,
    )
    return AttemptArtifacts(revision, observed_at, progress, result)
