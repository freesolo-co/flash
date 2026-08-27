"""adapter publication integrity tests with a deterministic fake hub."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import flash.engine.worker.io.adapter_publication as publication
import flash.engine.worker.io.hf as worker_hf
from flash.adapters.artifacts import loadable_adapter_weight_files


# --------------------------------------------------------------------------------------------
# worker publication
# --------------------------------------------------------------------------------------------
class _RecordingHfApi:
    def __init__(self, files: list[str] | None = None):
        self.uploads: list[dict] = []
        self.preuploads: list[object] = []
        self.commits: list[dict] = []
        self.deleted: list[str] = []
        self.events: list[str] = []
        self._head = "a" * 40
        self._snapshots = {self._head: dict.fromkeys(files or [], b"existing")}
        self.omit_after_commit: str | None = None
        self.omit_metadata: str | None = None
        self.stale_after_commit: dict[str, bytes] = {}
        self.race_before_commit = False
        self.post_commit_listing_failures = 0
        self.after_preupload = None
        self.after_repo_info = None

    @property
    def files(self) -> set[str]:
        return set(self._snapshots[self._head])

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)

    def preupload_lfs_files(self, *, repo_id, repo_type, additions, free_memory):
        self.events.append("preupload")
        self.preuploads = list(additions)
        for addition in self.preuploads:
            addition._upload_mode = (
                "lfs" if addition.path_in_repo.endswith(".safetensors") else "regular"
            )
        if self.after_preupload is not None:
            self.after_preupload()

    def repo_info(self, repo_id, repo_type):
        if self.after_repo_info is not None:
            self.after_repo_info()
        return SimpleNamespace(sha=self._head)

    def list_repo_files(self, repo_id, repo_type, revision=None):
        committed_revisions = {commit["revision"] for commit in self.commits}
        if revision in committed_revisions and self.post_commit_listing_failures:
            self.post_commit_listing_failures -= 1
            raise RuntimeError("transient post-commit listing failure")
        return sorted(self._snapshots[revision or self._head])

    def create_commit(self, *, repo_id, repo_type, operations, commit_message, parent_commit):
        self.events.append("commit")
        if self.race_before_commit:
            raced = "b" * 40
            self._snapshots[raced] = {**self._snapshots[self._head], "other-writer.txt": b"x"}
            self._head = raced
            raise RuntimeError("parent commit changed")
        if parent_commit != self._head:
            raise RuntimeError("parent commit changed")
        result = dict(self._snapshots[self._head])
        operation_list = list(operations)
        for operation in operation_list:
            if hasattr(operation, "path_or_fileobj"):
                with operation.as_file() as source:
                    source.seek(0)
                    result[operation.path_in_repo] = source.read()
            else:
                result.pop(operation.path_in_repo, None)
        if self.omit_after_commit is not None:
            result.pop(self.omit_after_commit, None)
        result.update(self.stale_after_commit)
        revision = f"{len(self._snapshots) + 1:040x}"
        self._snapshots[revision] = result
        self._head = revision
        self.commits.append(
            {
                "operations": operation_list,
                "message": commit_message,
                "parent_commit": parent_commit,
                "revision": revision,
            }
        )
        return SimpleNamespace(oid=revision)

    def get_paths_info(self, *, repo_id, paths, repo_type, revision):
        snapshot = self._snapshots[revision]
        infos = []
        for path in paths:
            if path not in snapshot or path == self.omit_metadata:
                continue
            content = snapshot[path]
            if path.endswith(".safetensors"):
                lfs = SimpleNamespace(size=len(content), sha256=hashlib.sha256(content).hexdigest())
                blob_id = "f" * 40
            else:
                lfs = None
                blob = hashlib.sha1(usedforsecurity=False)
                blob.update(f"blob {len(content)}\0".encode())
                blob.update(content)
                blob_id = blob.hexdigest()
            infos.append(SimpleNamespace(path=path, size=len(content), blob_id=blob_id, lfs=lfs))
        return infos

    def hf_hub_download(
        self,
        *,
        repo_id,
        filename,
        repo_type,
        revision,
        local_dir,
        force_download,
    ):
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self._snapshots[revision][filename])
        return str(destination)

    def delete_folder(self, path_in_repo, repo_id, repo_type):
        self.deleted.append(path_in_repo)


def _prime_worker(monkeypatch, recorder, *, repo="org/test-runs", phase="rl", run="flash-ckpt-1"):
    import flash.engine.worker.io.hf as worker

    monkeypatch.setattr(worker._worker_state, "HF_REPO", repo)
    monkeypatch.setattr(worker._worker_state, "PHASE", phase)
    monkeypatch.setattr(worker._worker_state, "RUN_ID", run)
    monkeypatch.setattr(worker._worker_state, "SEED", 0)
    monkeypatch.setattr(worker_hf, "hf_api", lambda: recorder)
    # heartbeat would otherwise commit to hf; silence it for the unit test.
    monkeypatch.setattr(worker._worker_heartbeat, "heartbeat", lambda *a, **k: None)
    return worker


def _write_single_adapter(path):
    path.mkdir()
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"single")


def _write_sharded_adapter(path, *, index=None, extra_shards=()):
    path.mkdir()
    shards = [
        "adapter_model-00001-of-00002.safetensors",
        "adapter_model-00002-of-00002.safetensors",
    ]
    (path / "adapter_config.json").write_text("{}")
    for shard in [*shards, *extra_shards]:
        (path / shard).write_bytes(shard.encode())
    payload = {"weight_map": {"a": shards[0], "b": shards[1]}} if index is None else index
    index_path = path / "adapter_model.safetensors.index.json"
    if isinstance(payload, bytes):
        index_path.write_bytes(payload)
    else:
        index_path.write_text(json.dumps(payload))
    return shards


def _adapter_files(recorder, subfolder):
    prefix = f"{subfolder}/"
    return {path[len(prefix) :] for path in recorder.files if path.startswith(prefix)}


def _deeply_nested_json(depth=20000):
    # The depth has to exhaust the C parser's stack on every supported interpreter, not just the
    # shallowest one: CPython 3.12 raised its own json recursion headroom, so a payload that is
    # pathological on 3.11 parses cleanly there and reaches the weight_map check instead, and the
    # test then asserts a message the reader never emits. 20000 recurses on 3.11 and 3.12 alike and
    # is 220KB, far under the index size cap that would otherwise reject it first.
    return b'{"nested":' * depth + b"0" + b"}" * depth


def test_publish_deployable_checkpoint_uploads_adapter_only(tmp_path, monkeypatch):
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-80"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
    (ckpt / "optimizer.pt").write_bytes(b"opt")

    subfolder = worker.publish_deployable_checkpoint(str(ckpt), 80)

    assert subfolder == "rl/flash-ckpt-1/checkpoints/step-80/adapter"
    assert len(rec.commits) == 1
    prefix = f"{subfolder}/"
    assert {path for path in rec.files if path.startswith(prefix)} == {
        f"{prefix}adapter_config.json",
        f"{prefix}adapter_model.safetensors",
    }
    # trainer-state files are excluded from the stable adapter snapshot.
    assert f"{prefix}optimizer.pt" not in rec.files
    # the deployable path must not prune older steps.
    assert all(
        "checkpoints/step-80/adapter" in op.path_in_repo for op in rec.commits[0]["operations"]
    )


def test_final_adapter_replaces_single_with_sharded_representation(tmp_path, monkeypatch):
    target = "rl/flash-ckpt-1/adapter"
    rec = _RecordingHfApi([f"{target}/adapter_config.json", f"{target}/adapter_model.safetensors"])
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    shards = _write_sharded_adapter(adapter)

    assert worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert _adapter_files(rec, target) == {
        "adapter_config.json",
        "adapter_model.safetensors.index.json",
        *shards,
    }
    published = _adapter_files(rec, target)
    assert "adapter_model.safetensors" not in published
    assert loadable_adapter_weight_files(published) == shards


def test_checkpoint_adapter_replaces_sharded_with_single_representation(tmp_path, monkeypatch):
    target = "rl/flash-ckpt-1/checkpoints/step-80/adapter"
    rec = _RecordingHfApi(
        [
            f"{target}/adapter_config.json",
            f"{target}/adapter_model.safetensors.index.json",
            f"{target}/adapter_model-00001-of-00002.safetensors",
            f"{target}/adapter_model-00002-of-00002.safetensors",
        ]
    )
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)

    assert worker.publish_deployable_checkpoint(str(adapter), 80, required=True) == target

    published = _adapter_files(rec, target)
    assert published == {
        "adapter_config.json",
        "adapter_model.safetensors",
    }
    assert loadable_adapter_weight_files(published) == ["adapter_model.safetensors"]


def test_final_adapter_deletes_a_stale_extra_shard(tmp_path, monkeypatch):
    target = "rl/flash-ckpt-1/adapter"
    stale = "adapter_model-00001-of-00003.safetensors"
    rec = _RecordingHfApi(
        [
            f"{target}/adapter_config.json",
            f"{target}/adapter_model.safetensors.index.json",
            f"{target}/adapter_model-00001-of-00002.safetensors",
            f"{target}/adapter_model-00002-of-00002.safetensors",
            f"{target}/{stale}",
        ]
    )
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    shards = _write_sharded_adapter(adapter)

    assert worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert _adapter_files(rec, target) == {
        "adapter_config.json",
        "adapter_model.safetensors.index.json",
        *shards,
    }
    assert stale not in _adapter_files(rec, target)


def test_final_adapter_rejects_an_incomplete_immutable_result(tmp_path, monkeypatch):
    target = "rl/flash-ckpt-1/adapter"
    rec = _RecordingHfApi()
    rec.omit_after_commit = f"{target}/adapter_model-00002-of-00002.safetensors"
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_sharded_adapter(adapter)

    with pytest.raises(worker_hf.RetriableInfraError, match="does not exactly match"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert len(rec.commits) == 1


def test_final_adapter_rejects_parent_commit_race_without_retry(tmp_path, monkeypatch):
    rec = _RecordingHfApi()
    rec.race_before_commit = True
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)

    with pytest.raises(worker_hf.RetriableInfraError, match="parent changed"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.commits == []
    assert "other-writer.txt" in rec.files


def test_post_commit_listing_failure_never_retries_commit(tmp_path, monkeypatch):
    rec = _RecordingHfApi()
    rec.post_commit_listing_failures = 1
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)

    with pytest.raises(
        worker_hf.RetriableInfraError,
        match="post-commit verification failed: RuntimeError: transient post-commit listing failure",
    ):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert len(rec.commits) == 1
    assert rec.events.count("commit") == 1


def test_an_unverified_commit_is_retracted_so_resume_cannot_credit_it(tmp_path, monkeypatch):
    """A published-but-unverified adapter folder must not stay readable.

    Resume credits a required save from `adapter_config.json` alone, on the stated grounds that the
    folder lands atomically. A commit that published and then failed verification breaks exactly
    that implication, so the marker has to stop existing rather than certify a folder nothing
    confirmed.
    """
    target = "rl/flash-ckpt-1/adapter"
    rec = _RecordingHfApi()
    rec.omit_after_commit = f"{target}/adapter_model.safetensors"
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)

    with pytest.raises(worker_hf.RetriableInfraError):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.deleted == [target], "the unverified folder must be withdrawn"


def test_a_verified_commit_is_never_retracted(tmp_path, monkeypatch):
    """The retraction must fire only on a verification failure.

    Without this, a retraction that ran unconditionally would delete every adapter it had just
    successfully published.
    """
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)

    worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.deleted == []
    assert "rl/flash-ckpt-1/adapter/adapter_config.json" in rec.files


def test_retraction_defers_to_a_writer_that_already_moved_the_head(tmp_path, monkeypatch):
    """A newer commit from another writer owns the path, and must not be deleted.

    Our unverified commit is only safe to withdraw while it is still the head. Once someone else has
    published over it, deleting the folder would destroy a publication that was never in doubt.
    """
    rec = _RecordingHfApi()
    rec.omit_after_commit = "rl/flash-ckpt-1/adapter/adapter_model.safetensors"
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)

    real_verify = publication._verify_adapter_commit

    def _verify_then_race(*args, **kwargs):
        try:
            real_verify(*args, **kwargs)
        finally:
            # another writer lands a commit between our failed verify and the retraction.
            rec._snapshots["c" * 40] = dict(rec._snapshots[rec._head])
            rec._head = "c" * 40

    monkeypatch.setattr(publication, "_verify_adapter_commit", _verify_then_race)

    with pytest.raises(worker_hf.RetriableInfraError):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.deleted == [], "a newer writer's commit must survive our retraction"


def test_a_failed_retraction_does_not_mask_the_publication_error(tmp_path, monkeypatch):
    """Retraction is best effort; the loud failure is the publication error itself.

    If a delete that cannot reach the hub raised, it would replace the diagnostic naming the real
    problem with one naming the cleanup.
    """
    rec = _RecordingHfApi()
    rec.omit_after_commit = "rl/flash-ckpt-1/adapter/adapter_model.safetensors"
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)

    def _unreachable(**_kwargs):
        raise OSError("hub unreachable")

    monkeypatch.setattr(rec, "delete_folder", _unreachable)

    with pytest.raises(worker_hf.RetriableInfraError, match="post-commit verification failed"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)


def test_sidecar_mutation_after_preupload_is_rejected_before_commit(tmp_path, monkeypatch):
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)
    sidecar = adapter / "base_model_provenance.json"
    sidecar.write_bytes(b'{"resolved_commit":"aaaaaaaa"}')
    rec.after_preupload = lambda: sidecar.write_bytes(b'{"resolved_commit":"bbbbbbbb"}')

    with pytest.raises(worker_hf.RequiredSaveError, match="changed after hashing"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.commits == []


def test_sidecar_mutation_while_resolving_the_parent_is_rejected_before_commit(
    tmp_path, monkeypatch
):
    # the parent lookup and the current-file listing both round-trip to the hub after the
    # preupload revalidation, so the snapshot must be revalidated again before the commit is
    # built from those descriptors.
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)
    sidecar = adapter / "base_model_provenance.json"
    sidecar.write_bytes(b'{"resolved_commit":"aaaaaaaa"}')
    rec.after_repo_info = lambda: sidecar.write_bytes(b'{"resolved_commit":"bbbbbbbb"}')

    with pytest.raises(worker_hf.RequiredSaveError, match="changed after hashing"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.commits == []


def test_nested_directory_rename_and_replacement_fails_before_preupload(tmp_path, monkeypatch):
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)
    nested = adapter / "metadata"
    nested.mkdir()
    (nested / "sidecar.json").write_bytes(b"original")
    original_context = publication.local_snapshot_root

    @contextlib.contextmanager
    def swap_after_scan(local_dir):
        with original_context(local_dir) as snapshot:
            nested.rename(adapter / "metadata-original")
            nested.mkdir()
            (nested / "sidecar.json").write_bytes(b"replacement")
            yield snapshot

    monkeypatch.setattr(publication, "local_snapshot_root", swap_after_scan)

    with pytest.raises(
        worker.RequiredSaveError, match="directory changed after scanning: metadata"
    ):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.preuploads == []
    assert rec.commits == []


def test_nested_directory_rename_and_replacement_after_preupload_fails_before_commit(
    tmp_path, monkeypatch
):
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)
    nested = adapter / "metadata"
    nested.mkdir()
    (nested / "sidecar.json").write_bytes(b"original")

    def swap_after_preupload():
        nested.rename(adapter / "metadata-original")
        nested.mkdir()
        (nested / "sidecar.json").write_bytes(b"replacement")

    rec.after_preupload = swap_after_preupload

    with pytest.raises(
        worker.RequiredSaveError, match="directory changed after scanning: metadata"
    ):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.events == ["preupload"]
    assert rec.commits == []


def test_fifo_replacement_is_opened_nonblocking_before_validation(tmp_path, monkeypatch):
    import flash.engine.worker.io.local_snapshot as local_snapshot

    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)
    config = adapter / "adapter_config.json"
    original_open = local_snapshot.os.open
    original_stat = local_snapshot.os.stat
    checked = False
    replaced = False

    def replacing_stat(path, *args, **kwargs):
        nonlocal replaced
        result = original_stat(path, *args, **kwargs)
        if path == "adapter_config.json" and not replaced:
            replaced = True
            config.unlink()
            os.mkfifo(config)
        return result

    def checked_open(path, flags, *args, **kwargs):
        nonlocal checked
        if path == "adapter_config.json":
            checked = True
            assert flags & os.O_NONBLOCK
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(local_snapshot.os, "stat", replacing_stat)
    monkeypatch.setattr(local_snapshot.os, "open", checked_open)
    with (
        pytest.raises(local_snapshot.UnsafeLocalSnapshotError, match="non-regular file"),
        local_snapshot.local_snapshot_root(str(adapter)),
    ):
        pass

    assert replaced
    assert checked


def test_commit_operations_keep_the_hub_union_annotation():
    source = inspect.getsource(publication.replace_adapter_folder)
    tree = ast.parse(source)
    annotations = {
        ast.unparse(node.target): ast.unparse(node.annotation)
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
    }
    assert annotations["operations"] == "list[CommitOperation]"


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (b"{", "not strict valid json"),
        ([], "must be a json object"),
        ({}, "nonempty object weight_map"),
        ({"weight_map": {}}, "nonempty object weight_map"),
        (
            {"weight_map": {"a": "adapter_model-00001-of-00002.safetensors"}},
            "do not exactly match",
        ),
        (
            {
                "weight_map": {
                    "a": "adapter_model-00001-of-00002.safetensors",
                    "b": "adapter_model-00002-of-00002.safetensors",
                    "c": "adapter_model-00003-of-00003.safetensors",
                }
            },
            "do not exactly match",
        ),
        (
            {
                "weight_map": {
                    "a": "../adapter_model-00001-of-00002.safetensors",
                    "b": "adapter_model-00002-of-00002.safetensors",
                }
            },
            "unsafe shard path",
        ),
        (
            {
                "weight_map": {
                    "a": "nested/adapter_model-00001-of-00002.safetensors",
                    "b": "adapter_model-00002-of-00002.safetensors",
                }
            },
            "unsafe shard path",
        ),
        (
            {
                "weight_map": {
                    "a": "/adapter_model-00001-of-00002.safetensors",
                    "b": "adapter_model-00002-of-00002.safetensors",
                }
            },
            "unsafe shard path",
        ),
        (
            {
                "weight_map": {
                    "a": "adapter_model-00001-of-00002.safetensors",
                    "b": "adapter_model-00002-of-00002.safetensors",
                    "c": "./adapter_model-00001-of-00002.safetensors",
                }
            },
            "duplicate shard aliases",
        ),
    ],
)
def test_final_adapter_rejects_invalid_local_index_before_preupload(
    tmp_path, monkeypatch, index, message
):
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_sharded_adapter(adapter, index=index)

    with pytest.raises(worker_hf.RequiredSaveError, match=message):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.preuploads == []
    assert rec.commits == []


def test_final_adapter_rejects_deep_local_json_before_preupload(tmp_path, monkeypatch):
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_sharded_adapter(adapter, index=_deeply_nested_json())

    with pytest.raises(worker_hf.RequiredSaveError, match="not strict valid json"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.preuploads == []
    assert rec.commits == []


def test_final_adapter_rejects_duplicate_index_keys_before_preupload(tmp_path, monkeypatch):
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    shards = _write_sharded_adapter(adapter)
    (adapter / "adapter_model.safetensors.index.json").write_text(
        f'{{"weight_map":{{"a":"{shards[0]}","a":"{shards[1]}"}}}}'
    )

    with pytest.raises(worker_hf.RequiredSaveError, match="not strict valid json"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.preuploads == []
    assert rec.commits == []


@pytest.mark.parametrize(
    "linked_name",
    [
        "adapter_config.json",
        "adapter_model.safetensors.index.json",
        "adapter_model-00001-of-00002.safetensors",
    ],
)
def test_final_adapter_rejects_symlinked_adapter_files_before_preupload(
    tmp_path, monkeypatch, linked_name
):
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_sharded_adapter(adapter)
    outside = tmp_path / "outside"
    outside.write_bytes((adapter / linked_name).read_bytes())
    (adapter / linked_name).unlink()
    (adapter / linked_name).symlink_to(outside)

    with pytest.raises(worker_hf.RequiredSaveError, match="contains symlink"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.preuploads == []
    assert rec.commits == []


def test_final_adapter_rejects_symlinked_path_component_before_preupload(tmp_path, monkeypatch):
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    real = tmp_path / "real"
    adapter = real / "adapter"
    real.mkdir()
    _write_single_adapter(adapter)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(worker_hf.RequiredSaveError, match="symlink or unsafe path component"):
        worker.hf_upload_folder(str(linked / "adapter"), "adapter", required=True)

    assert rec.preuploads == []
    assert rec.commits == []


def test_final_adapter_rejects_extra_local_shard_before_preupload(tmp_path, monkeypatch):
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_sharded_adapter(
        adapter,
        extra_shards=("adapter_model-00003-of-00003.safetensors",),
    )

    with pytest.raises(
        worker_hf.RequiredSaveError, match="no single complete weight representation"
    ):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.preuploads == []
    assert rec.commits == []


@pytest.mark.parametrize(
    ("filename", "stale"),
    [("adapter_config.json", b"[]"), ("adapter_model.safetensors", b"staler")],
)
def test_final_adapter_rejects_stale_remote_file_bytes(tmp_path, monkeypatch, filename, stale):
    target = "rl/flash-ckpt-1/adapter"
    rec = _RecordingHfApi()
    rec.stale_after_commit[f"{target}/{filename}"] = stale
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)

    with pytest.raises(worker_hf.RetriableInfraError, match="content differs"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert len(rec.commits) == 1


def test_final_adapter_rejects_stale_remote_index_bytes(tmp_path, monkeypatch):
    target = "rl/flash-ckpt-1/adapter"
    index_path = f"{target}/adapter_model.safetensors.index.json"
    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_sharded_adapter(adapter)
    local_index = (adapter / "adapter_model.safetensors.index.json").read_bytes()
    rec.stale_after_commit[index_path] = local_index.replace(b'"a":', b'"c":', 1)

    with pytest.raises(worker_hf.RetriableInfraError, match="content differs"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert len(rec.commits) == 1


def test_final_adapter_rejects_malformed_immutable_index(tmp_path, monkeypatch):
    target = "rl/flash-ckpt-1/adapter"
    index_path = f"{target}/adapter_model.safetensors.index.json"
    rec = _RecordingHfApi()
    rec.stale_after_commit[index_path] = b"{"
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_sharded_adapter(adapter)

    with pytest.raises(worker_hf.RetriableInfraError, match="not strict valid json"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert len(rec.commits) == 1


def test_final_adapter_rejects_deep_immutable_json_without_retry(tmp_path, monkeypatch):
    target = "rl/flash-ckpt-1/adapter"
    index_path = f"{target}/adapter_model.safetensors.index.json"
    rec = _RecordingHfApi()
    rec.stale_after_commit[index_path] = _deeply_nested_json()
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_sharded_adapter(adapter)

    with pytest.raises(worker_hf.RetriableInfraError, match="not strict valid json"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert len(rec.commits) == 1
    assert rec.events.count("commit") == 1


def test_final_adapter_rejects_unsafe_immutable_index(tmp_path, monkeypatch):
    target = "rl/flash-ckpt-1/adapter"
    index_path = f"{target}/adapter_model.safetensors.index.json"
    rec = _RecordingHfApi()
    rec.stale_after_commit[index_path] = json.dumps(
        {
            "weight_map": {
                "a": "../adapter_model-00001-of-00002.safetensors",
                "b": "adapter_model-00002-of-00002.safetensors",
            }
        }
    ).encode()
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_sharded_adapter(adapter)

    with pytest.raises(worker_hf.RetriableInfraError, match="unsafe shard path"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert len(rec.commits) == 1


def test_final_adapter_fails_closed_when_content_identity_is_unavailable(tmp_path, monkeypatch):
    target = "rl/flash-ckpt-1/adapter"
    rec = _RecordingHfApi()
    rec.omit_metadata = f"{target}/adapter_config.json"
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    _write_single_adapter(adapter)

    with pytest.raises(worker_hf.RetriableInfraError, match="omitted content metadata"):
        worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert len(rec.commits) == 1


def test_final_adapter_accepts_exact_immutable_content(tmp_path, monkeypatch):
    target = "rl/flash-ckpt-1/adapter"
    rec = _RecordingHfApi(["sibling/run.txt"])
    worker = _prime_worker(monkeypatch, rec)
    adapter = tmp_path / "adapter"
    shards = _write_sharded_adapter(adapter)

    assert worker.hf_upload_folder(str(adapter), "adapter", required=True)

    assert rec.events == ["preupload", "commit"]
    assert "sibling/run.txt" in rec.files
    assert _adapter_files(rec, target) == {
        "adapter_config.json",
        "adapter_model.safetensors.index.json",
        *shards,
    }


def test_publish_deployable_checkpoint_writes_base_model_provenance(tmp_path, monkeypatch):
    # regression (#538 finding 6): a per-step / opd-reconcile deployable is published straight from a
    # trainer dir that never passed through the final _save_adapter path, so publish itself stamps the
    # base-model provenance sidecar, sourced from the job spec's pinned base model, before uploading.
    import json

    import flash.engine.worker.io.hf as hf
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    commit = "e" * 40
    monkeypatch.setattr(
        worker._worker_state, "JOB_SPEC", SimpleNamespace(model="org/base", model_revision="main")
    )
    monkeypatch.setattr(hf, "resolve_cached_model_commit", lambda model_id, revision: commit)
    ckpt = tmp_path / "checkpoint-80"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")

    worker.publish_deployable_checkpoint(str(ckpt), 80)

    payload = json.loads((ckpt / "base_model_provenance.json").read_text())
    assert payload == {
        "model_id": "org/base",
        "requested_revision": "main",
        "resolved_commit": commit,
    }
    # the sidecar rides inside the same atomic commit; it must not be stripped as trainer state.
    assert len(rec.commits) == 1
    assert "rl/flash-ckpt-1/checkpoints/step-80/adapter/base_model_provenance.json" in rec.files


def test_publish_deployable_checkpoint_without_job_spec_writes_no_provenance(tmp_path, monkeypatch):
    # back-compat: with no job_spec (e.g. local recipe runs) publish writes no base_model_provenance.json
    # rather than a misleading empty record, and still publishes the deployable (provenance is additive).
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    monkeypatch.setattr(worker._worker_state, "JOB_SPEC", None, raising=False)
    ckpt = tmp_path / "checkpoint-80"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")

    worker.publish_deployable_checkpoint(str(ckpt), 80)

    assert not (ckpt / "base_model_provenance.json").exists()
    assert len(rec.commits) == 1


def test_publish_deployable_checkpoint_with_empty_model_writes_no_provenance(tmp_path, monkeypatch):
    # #542 finding: the guard mirrors the final-save path (write only for a non-empty base model), so a
    # job_spec with no model stamps no sidecar rather than a misleading empty-model_id record, and still
    # publishes the deployable.
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    monkeypatch.setattr(
        worker._worker_state,
        "JOB_SPEC",
        SimpleNamespace(model="", model_revision=""),
        raising=False,
    )
    ckpt = tmp_path / "checkpoint-80"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")

    worker.publish_deployable_checkpoint(str(ckpt), 80)

    assert not (ckpt / "base_model_provenance.json").exists()
    assert len(rec.commits) == 1


def test_publish_deployable_checkpoint_rejects_bin_weights(tmp_path, monkeypatch):
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-80"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.bin").write_bytes(b"weights")

    assert worker.publish_deployable_checkpoint(str(ckpt), 80) is None
    assert rec.commits == []


def test_publish_deployable_checkpoint_skips_without_adapter(tmp_path, monkeypatch):
    """A checkpoint that carries no PEFT adapter is never advertised as deployable."""
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-10"
    ckpt.mkdir()
    (ckpt / "optimizer.pt").write_bytes(b"opt")  # no adapter_config.json

    assert worker.publish_deployable_checkpoint(str(ckpt), 10) is None
    assert rec.commits == []


def test_publish_deployable_checkpoint_skips_config_without_weights(tmp_path, monkeypatch):
    """A checkpoint with adapter_config.json but no weights file isn't loadable -> not published."""
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-20"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")  # config only, no adapter_model.*

    assert worker.publish_deployable_checkpoint(str(ckpt), 20) is None
    assert rec.commits == []


def test_publish_deployable_checkpoint_no_repo_is_noop(tmp_path, monkeypatch):
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec, repo="")  # local run, no hf repo
    ckpt = tmp_path / "checkpoint-5"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")

    assert worker.publish_deployable_checkpoint(str(ckpt), 5) is None
    assert rec.commits == []


def test_publish_deployable_checkpoint_starts_no_upload_at_deadline(tmp_path, monkeypatch):
    import flash.engine.worker.io.hf as worker

    rec = _RecordingHfApi()
    _prime_worker(monkeypatch, rec)
    monkeypatch.setattr(worker._worker_state, "_remaining_worker_wall_seconds", lambda: 0.0)
    ckpt = tmp_path / "checkpoint-5"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")

    assert worker.publish_deployable_checkpoint(str(ckpt), 5) is None
    assert rec.commits == []


def _coexisting_weights_checkpoint(path):
    """A checkpoint peft cannot load: a single-file weight AND a sharded set in one folder.

    `_validate_adapter_snapshot` rejects this as a local contract violation. Nothing about it is
    transient, so it is the cleanest way to drive a `RequiredSaveError` out of the publish path.
    """
    path.mkdir()
    (path / "adapter_config.json").write_text("{}")
    (path / "adapter_model.safetensors").write_bytes(b"single")
    (path / "adapter_model-00001-of-00002.safetensors").write_bytes(b"shard-1")
    (path / "adapter_model-00002-of-00002.safetensors").write_bytes(b"shard-2")


def test_required_publish_reports_a_local_contract_violation_as_permanent(tmp_path, monkeypatch):
    """A malformed local adapter must not be reported as retriable infrastructure trouble.

    `RetriableInfraError` hands the run back to a resume, and a resume re-reads the same broken
    folder and fails identically -- burning the retry budget and ending in an error that names
    infrastructure rather than the adapter that is actually wrong.
    """
    from flash.engine.worker.io.adapter_publication import RequiredSaveError
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    ckpt = tmp_path / "checkpoint-30"
    _coexisting_weights_checkpoint(ckpt)

    with pytest.raises(RequiredSaveError) as caught:
        worker.publish_deployable_checkpoint(str(ckpt), 30, required=True)

    assert not isinstance(caught.value, RetriableInfraError)
    assert rec.commits == []


def test_required_publish_does_not_retry_a_local_contract_violation(tmp_path, monkeypatch):
    """The permanent failure aborts on the first attempt rather than consuming the retry budget."""
    from flash.engine.worker.io.adapter_publication import RequiredSaveError

    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    sleeps: list[float] = []
    monkeypatch.setattr(worker, "_sleep_with_hf_deadline", lambda s: sleeps.append(s) or True)
    ckpt = tmp_path / "checkpoint-31"
    _coexisting_weights_checkpoint(ckpt)

    with pytest.raises(RequiredSaveError):
        worker.publish_deployable_checkpoint(str(ckpt), 31, required=True, retries=3)

    assert sleeps == [], "a broken local folder is identical on every attempt"


def test_transient_upload_trouble_stays_retriable(tmp_path, monkeypatch):
    """The permanent classification must not swallow genuine infrastructure failures.

    Without this, narrowing the error taxonomy could quietly turn every upload outage into a
    permanent required-save failure.
    """
    from flash.engine.worker.perf.lifecycle import RetriableInfraError

    rec = _RecordingHfApi()
    worker = _prime_worker(monkeypatch, rec)
    monkeypatch.setattr(worker, "_sleep_with_hf_deadline", lambda _s: True)

    def _unreachable(*_args, **_kwargs):
        raise OSError("hub unreachable")

    monkeypatch.setattr(worker, "_replace_adapter_folder", _unreachable)
    ckpt = tmp_path / "checkpoint-32"
    _write_single_adapter(ckpt)

    with pytest.raises(RetriableInfraError):
        worker.publish_deployable_checkpoint(str(ckpt), 32, required=True)


def test_prune_stale_resume_checkpoints_deletes_only_older_steps(monkeypatch):
    """Pruning removes older steps without deleting a newer concurrently published checkpoint."""
    import flash.engine.worker.io.hf as worker_hf

    prefix = "rl/flash-ckpt-1"
    files = [
        f"{prefix}/checkpoint/checkpoint-20/optimizer.pt",
        f"{prefix}/checkpoint/checkpoint-20/adapter_model.safetensors",
        f"{prefix}/checkpoint/checkpoint-40/optimizer.pt",
        f"{prefix}/checkpoint/checkpoint-60/optimizer.pt",
        f"{prefix}/checkpoint/checkpoint-80/optimizer.pt",
        f"{prefix}/checkpoints/step-60/adapter/adapter_model.safetensors",  # deployable (plural) -> keep
        f"{prefix}/metrics.json",
    ]
    rec = _RecordingHfApi(files)
    _prime_worker(monkeypatch, rec)

    worker_hf._prune_stale_resume_checkpoints(60)

    assert sorted(rec.deleted) == [
        f"{prefix}/checkpoint/checkpoint-20",
        f"{prefix}/checkpoint/checkpoint-40",
    ]
    assert f"{prefix}/checkpoint/checkpoint-60" not in rec.deleted
    assert f"{prefix}/checkpoint/checkpoint-80" not in rec.deleted
    assert all("checkpoints/" not in d for d in rec.deleted)  # deployable tree untouched


def test_prune_stale_resume_checkpoints_no_repo_is_noop(monkeypatch):
    import flash.engine.worker.io.hf as worker_hf

    rec = _RecordingHfApi(["rl/r/checkpoint/checkpoint-1/optimizer.pt"])
    _prime_worker(monkeypatch, rec, repo="")  # local run, no hf repo
    worker_hf._prune_stale_resume_checkpoints(5)
    assert rec.deleted == []
