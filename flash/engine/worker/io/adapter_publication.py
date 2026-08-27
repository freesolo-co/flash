"""Atomic adapter folder publication with verified immutable commits.

An adapter folder must land as one commit whose published content exactly matches the local
snapshot. A partial or racing publication would otherwise leave a checkpoint that resolves but
loads the wrong weights, so every commit here is verified against the snapshot it was built from
and a failed commit is never blindly retried over.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from flash._internal.diagnostics import sanitize_diagnostic
from flash.adapters.artifacts import (
    ADAPTER_SHARD_PREFIX,
    ADAPTER_WEIGHT_FILES,
    ADAPTER_WEIGHT_SUFFIXES,
    loadable_adapter_weight_files,
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
from flash.envs.loading.loader import is_commit_sha


class RequiredSaveError(RuntimeError):
    """permanent exact-save contract failure."""


class AdapterPublicationError(RuntimeError):
    """a stable adapter commit failed in a way that a blind retry must not overwrite."""


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
        raise AdapterPublicationError(
            "immutable adapter commit content metadata could not be read"
        ) from error

    by_path = {}
    for info in infos:
        path = getattr(info, "path", None)
        if path not in remote_paths or path in by_path:
            raise AdapterPublicationError(
                "immutable adapter commit returned ambiguous content metadata"
            )
        by_path[path] = info
    if set(by_path) != set(remote_paths):
        raise AdapterPublicationError("immutable adapter commit omitted content metadata")
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
        raise AdapterPublicationError(
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
        raise AdapterPublicationError(str(error)) from error


def _subtree_content_identity(api, *, repo_id: str, revision: str, prefix: str) -> dict[str, tuple]:
    """Content identity of every file under ``prefix`` at one immutable revision.

    Raises when the hub cannot describe a listed file, so an unreadable subtree can never compare
    equal to another unreadable one.
    """
    paths = sorted(
        path
        for path in api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
        if path.startswith(prefix)
    )
    if not paths:
        return {}
    identity: dict[str, tuple] = {}
    for info in api.get_paths_info(
        repo_id=repo_id, paths=paths, repo_type="dataset", revision=revision
    ):
        path = getattr(info, "path", None)
        size = getattr(info, "size", None)
        lfs = getattr(info, "lfs", None)
        digest = getattr(lfs, "sha256", None) if lfs is not None else getattr(info, "blob_id", None)
        if (
            path not in paths
            or path in identity
            or isinstance(size, bool)
            or not isinstance(size, int)
            or not isinstance(digest, str)
        ):
            raise AdapterPublicationError("adapter subtree content identity could not be resolved")
        identity[path] = (size, digest.lower())
    if set(identity) != set(paths):
        raise AdapterPublicationError("adapter subtree content identity is incomplete")
    return identity


def _tip_still_carries_our_adapter(
    api, *, repo_id: str, revision: str, head: str, prefix: str
) -> bool:
    """True when the branch tip's adapter subtree is still exactly the one we published.

    Ownership of a path is a content question, not a head question. Sibling writers share this
    repository -- the heartbeat daemon, the streamed resume checkpoint, and every other step's
    adapter all commit to it -- so the head moves past ours constantly without anyone touching this
    path. Comparing content instead of commits means an unrelated commit no longer strands an
    unverified folder on the tip, while a genuine republication of this exact path is still left
    alone.
    """
    if head == revision:
        return True
    ours = _subtree_content_identity(api, repo_id=repo_id, revision=revision, prefix=prefix)
    return bool(ours) and (
        _subtree_content_identity(api, repo_id=repo_id, revision=head, prefix=prefix) == ours
    )


def _retract_unverified_adapter(api, *, repo_id: str, revision: str | None, target: str) -> None:
    """Remove an adapter folder whose commit landed but could not be verified.

    Resume credits a required save from the presence of ``adapter_config.json`` alone, on the
    stated grounds that the folder lands as one atomic commit. A commit that published and then
    failed verification breaks exactly that implication: the marker exists, so the step is credited
    durable, and a checkpoint nothing ever validated becomes loadable. The marker has to stop
    existing for the inference drawn from it to stay true.

    Another writer may already own this path, and deleting then would destroy a publication that was
    never in doubt. So retraction runs only while the tip still carries our own unverified content,
    is pinned to the exact commit that check read, and only ever names a real subfolder.
    """
    if not target or not isinstance(revision, str) or not is_commit_sha(revision):
        return
    try:
        head = getattr(api.repo_info(repo_id=repo_id, repo_type="dataset"), "sha", None)
        if not isinstance(head, str) or not is_commit_sha(head):
            return
        if not _tip_still_carries_our_adapter(
            api, repo_id=repo_id, revision=revision, head=head, prefix=f"{target}/"
        ):
            return
        # pinned to the head the check above read: a writer that republishes this path in the gap
        # between the two calls loses the delete instead of losing its publication.
        api.delete_folder(
            path_in_repo=target, repo_id=repo_id, repo_type="dataset", parent_commit=head
        )
    except Exception:
        # best effort: the publication error below already fails this save loudly. a retraction that
        # cannot reach the hub leaves the folder exactly as an unverified commit leaves it today.
        return


def replace_adapter_folder(
    api,
    repo_id: str,
    local_dir: str,
    repo_subpath: str,
    *,
    ignore_patterns: tuple[str, ...] = (),
) -> str:
    """atomically replace one adapter folder and verify its immutable result."""
    from huggingface_hub import CommitOperation, CommitOperationAdd, CommitOperationDelete

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
                    raise AdapterPublicationError(
                        "adapter publication failed and its parent could not be re-read"
                    ) from parent_error
                if current_parent != parent:
                    raise AdapterPublicationError(
                        f"adapter publication parent changed from {parent} to {current_parent}"
                    ) from error
                raise
            committed = True
            revision = getattr(commit, "oid", None)
            if not isinstance(revision, str) or not is_commit_sha(revision):
                raise AdapterPublicationError(
                    "post-commit adapter publication returned no immutable commit"
                )
            try:
                _verify_adapter_commit(
                    api,
                    repo_id=repo_id,
                    revision=revision,
                    prefix=prefix,
                    expected_remote=expected_remote,
                    adapter_files=adapter_files,
                    identities=identities,
                )
            except Exception:
                # the commit is already published. leaving it in place would leave a checkpoint that
                # resolves and loads while nothing has confirmed its contents match the snapshot.
                _retract_unverified_adapter(api, repo_id=repo_id, revision=revision, target=target)
                raise
            return revision
    except AdapterPublicationError as error:
        if committed and not str(error).startswith("post-commit"):
            raise AdapterPublicationError(f"post-commit verification failed: {error}") from error
        raise
    except RequiredSaveError as error:
        if committed:
            raise AdapterPublicationError(f"post-commit verification failed: {error}") from error
        raise
    except UnsafeLocalSnapshotError as error:
        if committed:
            raise AdapterPublicationError(f"post-commit verification failed: {error}") from error
        raise RequiredSaveError(str(error)) from error
    except Exception as error:
        if committed:
            detail = sanitize_diagnostic(error, limit=300)
            raise AdapterPublicationError(f"post-commit verification failed: {detail}") from error
        raise
