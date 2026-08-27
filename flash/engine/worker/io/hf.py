"""hf artifact channel for code delivery, adapters, metrics, and checkpoints."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import flash.engine.worker.io.heartbeat as _worker_heartbeat
import flash.engine.worker.runtime.state as _worker_state
from flash._internal.diagnostics import sanitize_diagnostic
from flash.adapters.artifacts import (
    ADAPTER_SHARD_PREFIX,
    ADAPTER_WEIGHT_FILES,
    ADAPTER_WEIGHT_SUFFIXES,
    attempt_scoped_artifact_name,
    has_loadable_adapter_weights,
    loadable_adapter_weight_files,
)
from flash.engine.profiling.tokenizer import (  # noqa: F401
    load_tokenizer,
    model_revision_kwargs,
)
from flash.engine.worker.io.local_snapshot import (
    SnapshotContentMismatchError,
    SnapshotFileIdentity,
    UnsafeLocalSnapshotError,
    local_snapshot_root,
    open_snapshot_file,
    revalidate_snapshot,
    snapshot_file_identity,
    snapshot_upload_paths,
    verify_snapshot_content,
)

# `gpu_diagnostics` has no call site here since prefetch moved to `.prefetch`, but it is kept
# imported on purpose: the prefetch tests patch `hf.gpu_diagnostics` and the prefetcher reads it
# back through this module. removing it as unused would silently break them.
from flash.engine.worker.perf import (  # noqa: F401
    RetriableInfraError,
    gpu_diagnostics,
)
from flash.teacher.retry_contract import (
    canonical_opd_optimizer_start_json,
    opd_optimizer_start_marker_path,
)


class RequiredSaveError(RuntimeError):
    """permanent exact-save contract failure."""


class _AdapterPublicationError(RuntimeError):
    """a stable adapter commit failed in a way that a blind retry must not overwrite."""


def error_artifact_name(mode: str, attempt: int = 0) -> str:
    """Per-mode, per-attempt error filename (e.g. error_sft_attempt0.txt). Attempt-scoped so a prior
    attempt's stale traceback can't be mistaken for the current attempt's crash on a retry host-loss."""
    return attempt_scoped_artifact_name("error", mode, attempt)


def ray_log_artifact_name(mode: str, attempt: int = 0) -> str:
    """Per-mode, per-attempt name for ray's own failure logs (e.g. raylogs_rl_attempt0.txt).

    Attempt-scoped for the same reason the traceback beside it is: ``hf_prefix()`` is per-RUN, not
    per-attempt, so an unscoped name would let a retry's ray logs overwrite the attempt that actually
    reproduced the raylet failure -- the one case this artifact exists to explain."""
    return attempt_scoped_artifact_name("raylogs", mode, attempt)


def _disable_xet_upload_staging() -> None:
    """Route this process's hf uploads through the streaming lfs path instead of xet.

    `hf_xet` is an unconditional dependency of `huggingface-hub` on every arch flash runs on, and
    `is_xet_available()` turns it on merely because it imports (utils/_runtime.py) -- so the xet
    path is the default for `upload_folder`. Xet chunks and dedups through a local cache rooted at
    `HF_XET_CACHE` (default `$HF_HOME/xet`), i.e. the SAME container disk holding the checkpoint
    being uploaded. The legacy lfs path streams from the source file handle instead
    (`CommitOperationAdd.as_file` -> `http_backoff(data=fileobj)`), so it adds no second on-disk
    copy of a ~60 GB fsdp save.

    The verl child, the rl child, and the model-merger subprocess already pin this. The parent is
    the process that actually uploads checkpoints, and it was the one left reading the default --
    which is where a real 35b run hit `No space left on device` under
    `.../huggingface/xet/.../staging/`.

    How much that staging holds at peak is not knowable from source (it lives in the compiled
    `hf_xet` extension), which is the point: the streaming path needs no such assumption. The
    checkpoint is already on this disk, so uploading it should not require room for a second copy
    of unknown size.

    `HF_HUB_DISABLE_XET` is captured into a module constant when `huggingface_hub.constants` is
    imported, so this must run before the first import. Every `huggingface_hub` import in the
    worker is function-local, so calling this during worker startup is early enough.
    """
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def hf_api():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def hf_prefix() -> str:
    return f"{_worker_state.PHASE}/{_worker_state.RUN_ID}"


def publish_opd_optimizer_start_marker() -> None:
    """Synchronously publish the first-update mutation marker before optimizer.step()."""
    if not isinstance(_worker_state.HF_REPO, str) or not _worker_state.HF_REPO.strip():
        raise RuntimeError("opd optimizer-start marker requires a private HF repository")
    marker_path = opd_optimizer_start_marker_path(_worker_state.RUN_ID, _worker_state.ATTEMPT)
    local_path = f"/tmp/opd-optimizer-start-attempt-{_worker_state.ATTEMPT}.json"
    payload = canonical_opd_optimizer_start_json(
        run_id=_worker_state.RUN_ID,
        attempt=_worker_state.ATTEMPT,
        seed=_worker_state.SEED,
    )
    with open(local_path, "wb") as file:
        file.write(payload)
        file.flush()
        os.fsync(file.fileno())
    try:
        _require_hf_deadline_allowance()
        hf_api().upload_file(
            path_or_fileobj=local_path,
            path_in_repo=marker_path,
            repo_id=_worker_state.HF_REPO,
            repo_type="dataset",
        )
    except Exception as error:
        detail = sanitize_diagnostic(error, limit=500)
        raise RetriableInfraError(
            f"required upload of OPD optimizer-start marker failed: {detail}"
        ) from error


def _require_hf_deadline_allowance() -> float | None:
    remaining = _worker_state._remaining_worker_wall_seconds()
    if remaining is not None and remaining <= 0:
        raise TimeoutError("run wall deadline exceeded")
    return remaining


def _sleep_with_hf_deadline(delay: float) -> bool:
    remaining = _require_hf_deadline_allowance()
    sleep_for = delay if remaining is None else min(delay, remaining)
    if sleep_for > 0:
        time.sleep(sleep_for)
    remaining = _worker_state._remaining_worker_wall_seconds()
    return remaining is None or remaining > 0


def _hf_upload(do_upload, repo_subpath: str, required: bool, label: str) -> bool:
    """HF upload loop: retries + raises on required artifacts; warn-only on optional.

    Returns True when a commit landed (or HF_REPO is unset), False on best-effort failure.
    """
    if not _worker_state.HF_REPO:
        return True
    attempts = 3 if required else 1
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            _require_hf_deadline_allowance()
            do_upload()
            return True
        except RequiredSaveError:
            raise
        except _AdapterPublicationError as e:
            last_err = e
            detail = sanitize_diagnostic(e, limit=500)
            if not required:
                print(f"{label} warn: {detail}")
                return False
            break
        except Exception as e:
            last_err = e
            detail = sanitize_diagnostic(e, limit=500)
            if required and attempt + 1 < attempts:
                print(f"{label} retry {attempt + 1}/{attempts}: {detail}")
                try:
                    if not _sleep_with_hf_deadline(5 * (attempt + 1)):
                        break
                except Exception:
                    break
                continue
            if not required:
                print(f"{label} warn: {detail}")
                return False
            break
    if required:
        detail = sanitize_diagnostic(last_err, limit=500)
        raise RetriableInfraError(
            f"required upload of {repo_subpath!r} failed: {detail}"
        ) from last_err
    return False


def hf_upload_file(local_path: str, repo_subpath: str, required: bool = False) -> bool:
    """Upload one file to the run's HF prefix. Returns True on success (see ``_hf_upload``)."""
    return _hf_upload(
        lambda: hf_api().upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"{hf_prefix()}/{repo_subpath}",
            repo_id=_worker_state.HF_REPO,
            repo_type="dataset",
        ),
        repo_subpath,
        required,
        "hf_upload_file",
    )


_RESUME_CHECKPOINT_UPLOAD_LOCK = threading.Lock()


@contextlib.contextmanager
def _resume_checkpoint_upload_slot(timeout_s: float | None = None):
    if timeout_s is None:
        acquired = _RESUME_CHECKPOINT_UPLOAD_LOCK.acquire()
    else:
        acquired = _RESUME_CHECKPOINT_UPLOAD_LOCK.acquire(timeout=max(0.0, timeout_s))
    try:
        yield acquired
    finally:
        if acquired:
            _RESUME_CHECKPOINT_UPLOAD_LOCK.release()


_ADAPTER_INDEX_MAX_BYTES = 64 * 1024 * 1024


def _adapter_representation_files(filenames: set[str]) -> set[str]:
    """return the sole complete supported weight representation, or raise."""
    singles = set(ADAPTER_WEIGHT_FILES) & filenames
    shards = {
        name
        for name in filenames
        if name.startswith(ADAPTER_SHARD_PREFIX) and name.endswith(ADAPTER_WEIGHT_SUFFIXES)
    }
    indexes = {
        f"adapter_model{suffix}.index.json"
        for suffix in ADAPTER_WEIGHT_SUFFIXES
        if f"adapter_model{suffix}.index.json" in filenames
    }
    if singles:
        if len(singles) != 1 or shards or indexes:
            raise RequiredSaveError("adapter snapshot contains coexisting weight representations")
        return singles
    selected = set(loadable_adapter_weight_files(filenames))
    if not selected or selected != shards or len(indexes) != 1:
        raise RequiredSaveError("adapter snapshot has no single complete weight representation")
    return selected | indexes


def _read_bounded_file(file, *, label: str) -> bytes:
    try:
        file.seek(0)
        size = os.fstat(file.fileno()).st_size
        if size > _ADAPTER_INDEX_MAX_BYTES:
            raise RequiredSaveError(f"{label} exceeds the adapter index size limit")
        content = file.read(_ADAPTER_INDEX_MAX_BYTES + 1)
        file.seek(0)
    except RequiredSaveError:
        raise
    except (OSError, ValueError) as error:
        raise RequiredSaveError(f"{label} could not be read") from error
    if len(content) > _ADAPTER_INDEX_MAX_BYTES:
        raise RequiredSaveError(f"{label} exceeds the adapter index size limit")
    return content


def _read_bounded_path(path: Path, *, label: str) -> bytes:
    try:
        with path.open("rb") as file:
            return _read_bounded_file(file, label=label)
    except RequiredSaveError:
        raise
    except OSError as error:
        raise RequiredSaveError(f"{label} could not be read") from error


def _strict_json_object(content: bytes, *, label: str) -> dict:
    def reject_duplicate_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"non-json constant {value!r}")

    try:
        parsed = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, TypeError, RecursionError) as error:
        raise RequiredSaveError(f"{label} is not strict valid json") from error
    if not isinstance(parsed, dict):
        raise RequiredSaveError(f"{label} must be a json object")
    return parsed


def _validate_adapter_index(content: bytes, shards: set[str], *, label: str) -> None:
    parsed = _strict_json_object(content, label=label)
    weight_map = parsed.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RequiredSaveError(f"{label} must contain a nonempty object weight_map")

    referenced: set[str] = set()
    normalized_aliases: dict[str, str] = {}
    for tensor_name, shard_path in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise RequiredSaveError(f"{label} contains an invalid weight_map key")
        if not isinstance(shard_path, str) or not shard_path:
            raise RequiredSaveError(f"{label} contains an invalid shard path")
        path = PurePosixPath(shard_path)
        normalized = path.as_posix()
        if (
            path.is_absolute()
            or len(path.parts) != 1
            or shard_path in {".", ".."}
            or "\\" in shard_path
        ):
            raise RequiredSaveError(f"{label} contains an unsafe shard path")
        prior = normalized_aliases.setdefault(normalized, shard_path)
        if prior != shard_path:
            raise RequiredSaveError(f"{label} contains duplicate shard aliases")
        if normalized != shard_path:
            raise RequiredSaveError(f"{label} contains an unsafe shard path")
        if not (
            shard_path.startswith(ADAPTER_SHARD_PREFIX)
            and shard_path.endswith(ADAPTER_WEIGHT_SUFFIXES)
        ):
            raise RequiredSaveError(f"{label} references a non-adapter shard")
        referenced.add(shard_path)
    if referenced != shards:
        raise RequiredSaveError(
            f"{label} shard references do not exactly match the published adapter shards"
        )


def _validate_adapter_snapshot(
    paths: set[str],
    *,
    read_index: Callable[[str], bytes],
    label: str,
) -> set[str]:
    direct_files = {path for path in paths if "/" not in path}
    if "adapter_config.json" not in direct_files:
        raise RequiredSaveError(f"{label} is missing adapter_config.json")
    representation = _adapter_representation_files(direct_files)
    indexes = {name for name in representation if name.endswith(".index.json")}
    if indexes:
        index_name = next(iter(indexes))
        shards = representation - indexes
        _validate_adapter_index(
            read_index(index_name),
            shards,
            label=f"{label} {index_name}",
        )
    return {"adapter_config.json", *representation}


def _immutable_adapter_metadata(
    api,
    *,
    repo_id: str,
    revision: str,
    prefix: str,
    paths: set[str],
) -> dict[str, object]:
    remote_paths = [f"{prefix}{path}" for path in sorted(paths)]
    try:
        infos = api.get_paths_info(
            repo_id=repo_id,
            paths=remote_paths,
            repo_type="dataset",
            revision=revision,
        )
    except Exception as error:
        raise _AdapterPublicationError(
            "immutable adapter commit content metadata could not be read"
        ) from error

    by_path = {}
    for info in infos:
        path = getattr(info, "path", None)
        if path not in remote_paths or path in by_path:
            raise _AdapterPublicationError(
                "immutable adapter commit returned ambiguous content metadata"
            )
        by_path[path] = info
    if set(by_path) != set(remote_paths):
        raise _AdapterPublicationError("immutable adapter commit omitted content metadata")
    return by_path


def _download_immutable_index(
    api,
    *,
    repo_id: str,
    revision: str,
    path: str,
) -> bytes:
    try:
        with tempfile.TemporaryDirectory(prefix="flash-adapter-verify-") as directory:
            local_path = api.hf_hub_download(
                repo_id=repo_id,
                filename=path,
                repo_type="dataset",
                revision=revision,
                local_dir=directory,
                force_download=True,
            )
            return _read_bounded_path(Path(local_path), label="immutable adapter index")
    except RequiredSaveError:
        raise
    except Exception as error:
        raise RequiredSaveError("immutable adapter index could not be downloaded") from error


def _verify_adapter_commit(
    api,
    *,
    repo_id: str,
    revision: str,
    prefix: str,
    expected_remote: set[str],
    adapter_files: set[str],
    identities: dict[str, SnapshotFileIdentity],
) -> None:
    published_files = set(
        api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    )
    published_adapter_files = {path for path in published_files if path.startswith(prefix)}
    if published_adapter_files != expected_remote:
        raise _AdapterPublicationError(
            "immutable adapter commit does not exactly match the local snapshot"
        )
    published_relative = {path[len(prefix) :] for path in published_adapter_files}
    metadata = _immutable_adapter_metadata(
        api,
        repo_id=repo_id,
        revision=revision,
        prefix=prefix,
        paths=set(identities),
    )
    indexes = {path for path in adapter_files if path.endswith(".index.json")}
    if indexes:
        index_name = next(iter(indexes))
        index_size = getattr(metadata[f"{prefix}{index_name}"], "size", None)
        if (
            isinstance(index_size, bool)
            or not isinstance(index_size, int)
            or index_size > _ADAPTER_INDEX_MAX_BYTES
        ):
            raise RequiredSaveError("immutable adapter index exceeds the adapter index size limit")
    published_adapter_files = _validate_adapter_snapshot(
        published_relative,
        read_index=lambda name: _download_immutable_index(
            api,
            repo_id=repo_id,
            revision=revision,
            path=f"{prefix}{name}",
        ),
        label="immutable adapter commit",
    )
    if published_adapter_files != adapter_files:
        raise RequiredSaveError(
            "immutable adapter commit representation differs from the local snapshot"
        )
    try:
        verify_snapshot_content(metadata, prefix=prefix, identities=identities)
    except SnapshotContentMismatchError as error:
        raise _AdapterPublicationError(str(error)) from error


def _replace_adapter_folder(
    local_dir: str,
    repo_subpath: str,
    *,
    ignore_patterns: tuple[str, ...] = (),
) -> str:
    """atomically replace one adapter folder and verify its immutable result."""
    from huggingface_hub import CommitOperation, CommitOperationAdd, CommitOperationDelete

    api = hf_api()
    repo_id = _worker_state.HF_REPO
    target = repo_subpath.strip("/")
    committed = False
    try:
        with contextlib.ExitStack() as stack:
            snapshot = stack.enter_context(local_snapshot_root(local_dir))
            local_paths = snapshot_upload_paths(snapshot.paths, ignore_patterns)
            expected = set(local_paths)
            files = {
                path: stack.enter_context(open_snapshot_file(snapshot, path))
                for path in local_paths
            }
            identities = {path: snapshot_file_identity(files[path], path) for path in local_paths}
            adapter_files = _validate_adapter_snapshot(
                expected,
                read_index=lambda name: _read_bounded_file(
                    files[name], label=f"local adapter snapshot {name}"
                ),
                label="local adapter snapshot",
            )
            revalidate_snapshot(snapshot, files, identities)
            prefix = f"{target}/" if target else ""
            expected_remote = {f"{prefix}{path}" for path in expected}
            additions = [
                CommitOperationAdd(path_in_repo=f"{prefix}{path}", path_or_fileobj=files[path])
                for path in local_paths
            ]
            api.preupload_lfs_files(
                repo_id=repo_id,
                repo_type="dataset",
                additions=additions,
                free_memory=False,
            )
            revalidate_snapshot(snapshot, files, identities)
            parent = getattr(api.repo_info(repo_id=repo_id, repo_type="dataset"), "sha", None)
            if not isinstance(parent, str) or not is_commit_sha(parent):
                raise RequiredSaveError(
                    "adapter publication could not resolve an immutable parent commit"
                )
            current_files = set(
                api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=parent)
            )
            current_adapter_files = {path for path in current_files if path.startswith(prefix)}
            operations: list[CommitOperation] = [
                CommitOperationDelete(path_in_repo=path, is_folder=False)
                for path in sorted(current_adapter_files - expected_remote)
            ]
            operations.extend(additions)
            revalidate_snapshot(snapshot, files, identities)
            try:
                commit = api.create_commit(
                    repo_id=repo_id,
                    repo_type="dataset",
                    operations=operations,
                    commit_message=f"Replace adapter at {target}",
                    parent_commit=parent,
                )
            except Exception as error:
                try:
                    current_parent = getattr(
                        api.repo_info(repo_id=repo_id, repo_type="dataset"), "sha", None
                    )
                except Exception as parent_error:
                    raise _AdapterPublicationError(
                        "adapter publication failed and its parent could not be re-read"
                    ) from parent_error
                if current_parent != parent:
                    raise _AdapterPublicationError(
                        f"adapter publication parent changed from {parent} to {current_parent}"
                    ) from error
                raise
            committed = True
            revision = getattr(commit, "oid", None)
            if not isinstance(revision, str) or not is_commit_sha(revision):
                raise _AdapterPublicationError(
                    "post-commit adapter publication returned no immutable commit"
                )
            revalidate_snapshot(snapshot, files, identities)
            _verify_adapter_commit(
                api,
                repo_id=repo_id,
                revision=revision,
                prefix=prefix,
                expected_remote=expected_remote,
                adapter_files=adapter_files,
                identities=identities,
            )
            revalidate_snapshot(snapshot, files, identities)
            return revision
    except _AdapterPublicationError as error:
        if committed and not str(error).startswith("post-commit"):
            raise _AdapterPublicationError(f"post-commit verification failed: {error}") from error
        raise
    except RequiredSaveError as error:
        if committed:
            raise _AdapterPublicationError(f"post-commit verification failed: {error}") from error
        raise
    except UnsafeLocalSnapshotError as error:
        if committed:
            raise _AdapterPublicationError(f"post-commit verification failed: {error}") from error
        raise RequiredSaveError(str(error)) from error
    except Exception as error:
        if committed:
            detail = sanitize_diagnostic(error, limit=300)
            raise _AdapterPublicationError(f"post-commit verification failed: {detail}") from error
        raise


def hf_upload_folder(local_dir: str, repo_subpath: str, required: bool = False) -> bool:
    """atomically replace an adapter folder under the run's HF prefix."""
    target = f"{hf_prefix()}/{repo_subpath}"
    return _hf_upload(
        lambda: _replace_adapter_folder(local_dir, target),
        repo_subpath,
        required,
        "hf_upload_folder",
    )


def _highest_resume_candidate(
    base: str, candidates: list[tuple[int, str]], prefer: Callable[[str], bool] | None
) -> str:
    """the candidate name hf_resume_checkpoint stages, given the caller's optional ``prefer``.

    every candidate is already staged on local disk by the snapshot_download above, so evaluating
    ``prefer`` per candidate costs no extra fetch. picks the highest step ``prefer`` accepts and
    falls back to the highest step overall when none do -- the same answer this returned before
    ``prefer`` existed, so a caller with nothing to prefer sees no behaviour change, and the caller's
    own discard log (not this function) is left to explain a restart from zero.
    """
    if prefer is not None:
        for _step, name in sorted(candidates, reverse=True):
            if prefer(os.path.join(base, name)):
                return name
    return max(candidates)[1]


def hf_resume_checkpoint(
    fail_closed: bool = False,
    revision: str | None = None,
    *,
    prefer: Callable[[str], bool] | None = None,
) -> str | None:
    """Download the latest streamed verl checkpoint for this run, or return none.

    ``prefer`` selects the highest downloaded candidate it accepts instead of the highest overall;
    see ``_highest_resume_candidate``. Left unset, behaviour is unchanged from before it existed.
    """
    required = bool(revision)
    strict = bool(fail_closed or required)
    if not _worker_state.HF_REPO:
        if required:
            raise RetriableInfraError("required resume checkpoint has no artifact repository")
        return None
    base = os.path.join("/tmp/resume", hf_prefix(), "checkpoint")
    try:
        from huggingface_hub import snapshot_download

        from flash.engine.worker.io.heartbeat import liveness_heartbeat

        # remove prior local materialization so pinned absence cannot reuse a stale checkpoint.
        shutil.rmtree(base, ignore_errors=True)
        # resume checkpoints carry the full optimizer state (multi-gb); keep the heartbeat fresh.
        with liveness_heartbeat("checkpoint_prefetching"):
            _require_hf_deadline_allowance()
            snapshot_download(
                repo_id=_worker_state.HF_REPO,
                repo_type="dataset",
                allow_patterns=[f"{hf_prefix()}/checkpoint/**"],
                local_dir="/tmp/resume",
                token=os.environ.get("HF_TOKEN"),
                revision=revision,
            )
        if not os.path.isdir(base):
            if required:
                detail = f" at revision {revision}" if revision else ""
                raise RetriableInfraError(f"required resume checkpoint is missing{detail}")
            return None
        candidates: list[tuple[int, str]] = []
        for name in os.listdir(base):
            if not name.startswith("checkpoint-"):
                continue
            suffix = name[len("checkpoint-") :]
            if not suffix.isdigit():
                continue
            step = int(suffix)
            if step > 0 and suffix == str(step):
                candidates.append((step, name))
        if not candidates:
            if required:
                detail = f" at revision {revision}" if revision else ""
                raise RetriableInfraError(f"required resume checkpoint is missing{detail}")
            return None
        latest = _highest_resume_candidate(base, candidates, prefer)
        path = os.path.join(base, latest)
        print(f"[resume] found streamed checkpoint: {path}")
        return path
    except Exception as e:
        print("hf_resume_checkpoint warn:", e)
        if strict:
            if isinstance(e, RetriableInfraError):
                raise
            raise RetriableInfraError(f"required resume checkpoint fetch failed: {e}") from e
        return None


# Trainer-state files the serving engine never needs; excluded to keep per-step adapter snapshots small.
_CHECKPOINT_TRAINER_STATE = (
    "optimizer.pt",
    "optimizer.bin",
    "scheduler.pt",
    "scaler.pt",
    "rng_state*.pth",
    "trainer_state.json",
    "training_args.bin",
    "*.distcp",
    "global_step*/**",
    "latest",
    "zero_to_fp32.py",
)


def _has_deployable_adapter(ckpt_dir: str) -> bool:
    """Return True if ckpt_dir has a loadable LoRA adapter (config + weights).

    Weights via the shared rule rather than the two single-file names, because a save past peft's
    shard size writes ``adapter_model-0000N-of-0000M.<ext>`` plus an index instead. Spelling only the
    single-file names here made this the strict side of a disagreement: serving and export accept the
    sharded save, so a sharded per-step adapter was silently skipped -- or failed a required save --
    for an artifact the rest of the pipeline would have deployed.
    """
    if not os.path.isfile(os.path.join(ckpt_dir, "adapter_config.json")):
        return False
    try:
        return has_loadable_adapter_weights(os.listdir(ckpt_dir))
    except OSError:
        return False


def _write_deployable_provenance(ckpt_dir: str) -> None:
    spec = _worker_state.JOB_SPEC
    if spec is not None and spec.model:
        write_base_model_provenance(ckpt_dir, spec.model, getattr(spec, "model_revision", "") or "")


def publish_deployable_checkpoint(
    ckpt_dir: str,
    step: int,
    *,
    retries: int = 1,
    backoff_s: float = 0.0,
    required: bool = False,
    _provenance_ready: bool = False,
    _emit_heartbeat: bool = True,
) -> str | None:
    """Mirror a verl checkpoint's LoRA adapter to a stable per-step path.

    Periodic saves remain best-effort. ``required=True`` fails loudly when an exact required save
    cannot be published.
    """
    if not _worker_state.HF_REPO:
        if required:
            raise RequiredSaveError(f"required save step {step} has no artifact repository")
        return None
    if not _has_deployable_adapter(ckpt_dir):
        if required:
            raise RequiredSaveError(
                f"required save step {step} has no deployable adapter in {ckpt_dir}"
            )
        return None
    # staged optional checkpoints already carry provenance and must remain immutable during upload.
    if not _provenance_ready:
        _write_deployable_provenance(ckpt_dir)
    subfolder = f"{hf_prefix()}/checkpoints/step-{step}/adapter"
    attempts = max(1, int(retries))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            _require_hf_deadline_allowance()
            _replace_adapter_folder(
                ckpt_dir,
                subfolder,
                ignore_patterns=_CHECKPOINT_TRAINER_STATE,
            )
            if _emit_heartbeat:
                _worker_heartbeat.heartbeat("checkpoint_deployable", step=step, subfolder=subfolder)
            return subfolder
        except _AdapterPublicationError as e:
            last_error = e
            print(f"[ckpt] deployable publish warn (step {step}):", e)
            break
        except Exception as e:
            last_error = e
            print(f"[ckpt] deployable publish warn (step {step}):", e)
            if attempt + 1 < attempts:
                try:
                    if not _sleep_with_hf_deadline(backoff_s * (attempt + 1)):
                        break
                except Exception:
                    break
    if required:
        raise RetriableInfraError(
            f"required save step {step} deployable upload failed"
        ) from last_error
    return None


# Retry/backoff for each synchronous checkpoint upload. the watcher BLOCKS on the upload, so a
# transient HF error is retried until the step lands rather than costing the step.
_CKPT_UPLOAD_RETRIES = 3
_CKPT_UPLOAD_BACKOFF_S = 5.0


def _deployable_adapter_on_hf(step: int) -> bool:
    """True when a required step's deployable adapter is durably present on hf.

    Resume credits a required save only after confirming its published adapter exists, so the
    final completeness check verifies the durability guarantee against hf instead of assuming it from the
    restored step counter (a pre-resume worker could have advanced past the step without ever
    landing its deployable). publish_deployable_checkpoint uploads the adapter folder in a single
    atomic upload_folder commit, so the config marker's presence implies the whole folder landed.

    Raises RetriableInfraError when hf cannot be reached: a transient lookup outage must retry the
    resume, not be misread as a permanently-missing required save. file_exists returns False cleanly
    for a genuinely absent file (that stays uncredited and fails the final completeness check).
    """
    if not _worker_state.HF_REPO:
        return False
    marker = f"{hf_prefix()}/checkpoints/step-{step}/adapter/adapter_config.json"
    try:
        return bool(
            hf_api().file_exists(
                repo_id=_worker_state.HF_REPO, filename=marker, repo_type="dataset"
            )
        )
    except Exception as e:
        raise RetriableInfraError(f"could not verify required save step {step} on hf") from e


def _prune_stale_resume_checkpoints(keep_step: int) -> None:
    """Delete older ``{prefix}/checkpoint/checkpoint-N`` directories.

    The streamed resume checkpoint is meant to be latest-only, but ``upload_folder``'s delete_patterns
    are matched relative to path_in_repo (the per-step dir), so they can never reach sibling step dirs.
    Only lower steps are stale: an older upload can finish after a newer one, and must never delete the
    newer checkpoint. A later upload removes any lower directory left by that race. The deployable tree
    (``{prefix}/checkpoints/...``, plural) has a different prefix and is untouched.
    """
    if not _worker_state.HF_REPO:
        return
    api = hf_api()
    base = f"{hf_prefix()}/checkpoint/"
    try:
        _require_hf_deadline_allowance()
        files = api.list_repo_files(repo_id=_worker_state.HF_REPO, repo_type="dataset")
    except Exception as e:
        print("ckpt prune warn (list):", e)
        return
    stale: set[str] = set()
    for f in files:
        if not f.startswith(base):
            continue
        seg = f[len(base) :].split("/", 1)[0]
        n = seg[len("checkpoint-") :]
        if seg.startswith("checkpoint-") and n.isdigit() and int(n) < keep_step:
            stale.add(f"{base}{seg}")
    for folder in sorted(stale):
        try:
            _require_hf_deadline_allowance()
            api.delete_folder(
                path_in_repo=folder, repo_id=_worker_state.HF_REPO, repo_type="dataset"
            )
        except Exception as e:
            print(f"ckpt prune warn ({folder}):", e)
            break


def upload_resume_checkpoint(
    step: int,
    ckpt_dir: str,
    *,
    before_upload=None,
    after_upload=None,
    skip_upload=None,
    emit_heartbeat: bool = True,
    lock_timeout_s: float | None = None,
) -> bool:
    """synchronously stream one full-state resume checkpoint and ordered companion artifacts."""
    if not _worker_state.HF_REPO:
        return True

    from flash.engine.worker.io.heartbeat import liveness_heartbeat

    before_completed = before_upload is None
    resume_completed = False
    after_completed = after_upload is None
    with _resume_checkpoint_upload_slot(lock_timeout_s) as acquired:
        if not acquired:
            print(f"[ckpt] step {step} upload skipped; another checkpoint upload is still active")
            return False
        if skip_upload is not None and skip_upload():
            return True
        heartbeat_context = (
            liveness_heartbeat(
                "checkpoint_uploading", progress=lambda: step, progress_step=True, keepalive=True
            )
            if emit_heartbeat
            else contextlib.nullcontext()
        )
        with heartbeat_context:
            for attempt in range(_CKPT_UPLOAD_RETRIES):
                failure_stage = "resume"
                try:
                    if not before_completed:
                        failure_stage = "before"
                        before_upload()
                        before_completed = True
                    if not resume_completed:
                        failure_stage = "resume"
                        _require_hf_deadline_allowance()
                        hf_api().upload_folder(
                            folder_path=ckpt_dir,
                            path_in_repo=f"{hf_prefix()}/checkpoint/checkpoint-{step}",
                            repo_id=_worker_state.HF_REPO,
                            repo_type="dataset",
                        )
                        # prune only after the atomic folder commit lands, so the prior complete save
                        # survives every failed replacement upload.
                        _prune_stale_resume_checkpoints(step)
                        resume_completed = True
                    if not after_completed:
                        failure_stage = "after"
                        after_upload()
                        after_completed = True
                    if emit_heartbeat:
                        failure_stage = "heartbeat"
                        _worker_heartbeat.heartbeat("checkpoint_uploaded", step=step)
                    return True
                except RequiredSaveError:
                    raise
                except Exception as e:
                    if attempt + 1 < _CKPT_UPLOAD_RETRIES:
                        print(
                            f"[ckpt] step {step} upload retry "
                            f"{attempt + 1}/{_CKPT_UPLOAD_RETRIES}: {e}"
                        )
                        try:
                            if not _sleep_with_hf_deadline(_CKPT_UPLOAD_BACKOFF_S * (attempt + 1)):
                                break
                        except Exception:
                            break
                    else:
                        print(
                            f"[ckpt] step {step} upload FAILED after "
                            f"{_CKPT_UPLOAD_RETRIES} attempts: {e}"
                        )
                        if emit_heartbeat:
                            with contextlib.suppress(Exception):
                                # worker stdout is not part of control-plane run logs, so the stage
                                # name alone would otherwise be the entire failure report.
                                _worker_heartbeat.heartbeat(
                                    "checkpoint_upload_failed",
                                    step=step,
                                    checkpoint_failure={
                                        "step": step,
                                        "operation": failure_stage,
                                        "error": sanitize_diagnostic(e, limit=300),
                                    },
                                )
                        if failure_stage in {"before", "after"}:
                            raise
    return False


# re-exported so the prefetch surface stays reachable as `hf.<name>`: the tests patch
# `_shared_weight_cache_dir`, `resolve_cached_model_commit` and `_hf_cache_bytes` here, and the
# worker entry points import `prefetch_model` from here.
from flash.engine.worker.io.prefetch import (  # noqa: E402,F401
    _hf_cache_bytes,
    _prefetch_error_is_retriable,
    _shared_weight_cache_dir,
    _snapshot_has_weights,
    _SnapshotWeightsMissing,
    prefetch_model,
    resolve_cached_model_commit,
    write_base_model_provenance,
)

# the sha predicate lives in the client-safe `envs.loading.loader`; re-exported here because the
# provenance tests patch and assert it as `hf.is_commit_sha`.
from flash.envs.loading.loader import is_commit_sha  # noqa: E402
