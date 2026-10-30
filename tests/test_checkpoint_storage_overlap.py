"""Exercise storage overlap with blocked I/O, real files, and watcher threads."""

import errno
import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from flash.engine.worker.io import hf
from flash.engine.worker.train.core.lifecycle.persistence import (
    CheckpointPersistence,
    pin_checkpoint,
)
from flash.engine.worker.train.entry import sft_train
from flash.engine.worker.train.rl.launch import checkpoints as grpo
from flash.engine.worker.train.sft.setup.checkpoints import _VerlCheckpointWatcher


def _resume_result():
    return hf.PublicationResult(resume=hf.ArtifactCommit("native", "revision"))


def _write_checkpoint(local, step):
    checkpoint = local / f"global_step_{step}"
    (checkpoint / "actor").mkdir(parents=True)
    (checkpoint / "actor" / "weights").write_text(f"weights-{step}")
    (local / "latest_checkpointed_iteration.txt").write_text(str(step))
    return checkpoint


def _watcher(algorithm, tmp_path, monkeypatch, worker_context):
    def export(actor, adapter, **kwargs):
        Path(adapter).mkdir(parents=True, exist_ok=True)
        (Path(adapter) / "weights").write_text((Path(actor) / "weights").read_text())

    monkeypatch.setattr(sft_train, "_export_checkpoint_adapter", export)
    options = {
        "local_dir": str(tmp_path / "native"),
        "export_root": str(tmp_path / "exports"),
        "python_bin": "python",
        "model_id": "org/model",
        "model_revision": "revision",
        "required_steps": (1, 2),
        "final_step": 2,
        "context": worker_context,
    }
    if algorithm == "sft":
        return _VerlCheckpointWatcher(**options)

    def stage(self, step, path):
        adapter = str(tmp_path / "exports" / f"step-{step}")
        export(str(Path(path) / "actor"), adapter)
        return adapter

    def make_durable(self, step, adapter):
        self._internal_adapter_attempts[step] = 0

    monkeypatch.setattr(grpo._VerlResumeUploader, "_stage_deployable", stage)
    monkeypatch.setattr(grpo._VerlResumeUploader, "_make_required_adapter_durable", make_durable)
    return grpo._VerlResumeUploader(
        **options,
        resume_step=0,
        metric_evidence=SimpleNamespace(checkpoint_manifest_evidence=lambda step: (True, 1)),
    )


@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
@pytest.mark.parametrize("final_staged", [False, True])
def test_next_adapter_publishes_while_recovery_upload_is_blocked(
    algorithm, final_staged, tmp_path, monkeypatch, worker_context
):
    watcher = _watcher(algorithm, tmp_path, monkeypatch, worker_context)
    local = Path(watcher.local_dir)
    first = _write_checkpoint(local, 1)
    uploading = threading.Event()
    release = threading.Event()
    published_second = threading.Event()
    uploaded = []
    snapshots = []

    def publish(adapter, step, **kwargs):
        assert (Path(adapter) / "weights").read_text() == f"weights-{step}"
        if step == 2:
            published_second.set()
        return hf.PublicationResult(deployable=hf.ArtifactCommit("adapter", "revision"))

    def upload(step, path, **kwargs):
        snapshots.append(Path(path))
        if step == 1:
            uploading.set()
            assert release.wait(10), "test did not release the blocked uploader"
        assert (Path(path) / "actor" / "weights").read_text() == f"weights-{step}"
        if algorithm == "grpo":
            manifest = json.loads((Path(path) / "_flash_resume_manifest.json").read_text())
            assert manifest["checkpoint_step"] == step
        uploaded.append(step)
        assert kwargs["final_step"] == 2
        if step == 2 and final_staged:
            return hf.PublicationResult(resume_staged=True)
        return _resume_result()

    monkeypatch.setattr(hf, "publish_deployable_checkpoint", publish)
    monkeypatch.setattr(hf, "upload_resume_checkpoint", upload)
    if algorithm == "grpo":
        watcher.allow_deployable_publication()
    watcher.start()
    try:
        assert uploading.wait(5)
        # Retention may unlink the native source while its pinned upload is blocked.
        shutil.rmtree(first)
        _write_checkpoint(local, 2)
        assert published_second.wait(5), "a recovery upload blocked the next adapter"
        assert uploaded == []
        assert not watcher.lifecycle.facts(1).resume_uploaded
        if algorithm == "sft":
            assert (Path(watcher.published_dir) / "step-1").is_file()
    finally:
        release.set()
        if algorithm == "grpo":
            watcher.stop()
            watcher.raise_if_incomplete()
        else:
            watcher.stop(require_complete=True)
    assert uploaded == [1, 2]
    assert all(not path.exists() for path in snapshots)
    assert watcher.lifecycle.facts(1).resume_uploaded
    assert watcher.lifecycle.facts(2).resume_uploaded is not final_staged


def test_persistence_bounds_admission_and_drains_in_order(worker_context):
    lane = CheckpointPersistence(context=worker_context)
    active = threading.Event()
    release = threading.Event()
    checking_capacity = threading.Event()
    admitted_third = threading.Event()
    completions = []
    failures = []

    def first():
        active.set()
        assert release.wait(10)
        return _resume_result()

    def owner():
        try:
            lane.start()
            lane.submit(first, lambda result: completions.append(1))
            lane.submit(_resume_result, lambda result: completions.append(2))
            checking_capacity.set()
            lane.wait_for_capacity()
            admitted_third.set()
            lane.submit(_resume_result, lambda result: completions.append(3))
            lane.close()
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=owner)
    thread.start()
    try:
        assert active.wait(5)
        assert checking_capacity.wait(5)
        assert not admitted_third.wait(0.1)
        assert completions == []
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert failures == []
    assert completions == [1, 2, 3]


@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
def test_publication_failure_drains_and_cleans_accepted_snapshots(
    algorithm, tmp_path, monkeypatch, worker_context
):
    watcher = _watcher(algorithm, tmp_path, monkeypatch, worker_context)
    local = Path(watcher.local_dir)
    _write_checkpoint(local, 1)
    uploading = threading.Event()
    release = threading.Event()
    publication_failed = threading.Event()

    def upload(step, path, **kwargs):
        uploading.set()
        assert release.wait(10)
        assert (Path(path) / "actor" / "weights").read_text() == "weights-1"
        return _resume_result()

    def publish(adapter, step, **kwargs):
        if step == 2:
            publication_failed.set()
            raise RuntimeError("required adapter unavailable")
        return hf.PublicationResult(deployable=hf.ArtifactCommit("adapter", "revision"))

    monkeypatch.setattr(hf, "upload_resume_checkpoint", upload)
    monkeypatch.setattr(hf, "publish_deployable_checkpoint", publish)
    if algorithm == "grpo":
        watcher.allow_deployable_publication()
    watcher.start()
    try:
        assert uploading.wait(5)
        _write_checkpoint(local, 2)
        assert publication_failed.wait(5)
        assert not watcher.lifecycle.facts(1).resume_uploaded
    finally:
        release.set()
        if algorithm == "grpo":
            watcher.stop()
            with pytest.raises(RuntimeError, match="resume uploader failed"):
                watcher.raise_if_incomplete()
        else:
            with pytest.raises(RuntimeError, match="checkpoint watcher failed"):
                watcher.stop(require_complete=True)
    assert str(watcher._error) == "required adapter unavailable"
    assert watcher.lifecycle.facts(1).resume_uploaded
    assert not watcher.lifecycle.facts(2).deployable_published
    root = local / ".flash-persistence" if algorithm == "grpo" else Path(watcher.staging_root)
    assert list(root.iterdir()) == []


def test_failed_storage_does_not_abandon_queued_work(worker_context):
    lane = CheckpointPersistence(context=worker_context)
    release = threading.Event()
    completed = []
    owner = threading.get_ident()

    def fail():
        assert release.wait(5)
        raise OSError("storage unavailable")

    def complete(result):
        assert threading.get_ident() == owner
        completed.append(result)

    lane.start()
    lane.submit(fail, complete)
    lane.submit(_resume_result, complete)
    release.set()
    with pytest.raises(OSError, match="storage unavailable"):
        lane.close()
    assert completed == [_resume_result()]


@pytest.mark.parametrize("error_number", [errno.EXDEV, errno.EACCES])
def test_snapshot_link_failures_never_use_an_unprotected_source(
    tmp_path, monkeypatch, error_number
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights").write_text("immutable")
    destination = tmp_path / "pinned"

    def fail_link(*args):
        raise OSError(error_number, "link failed")

    monkeypatch.setattr("os.link", fail_link)
    if error_number == errno.EXDEV:
        pin_checkpoint(str(source), str(destination))
        shutil.rmtree(source)
        assert (destination / "weights").read_text() == "immutable"
    else:
        with pytest.raises(shutil.Error):
            pin_checkpoint(str(source), str(destination))
        assert not destination.exists()


def test_snapshot_collision_preserves_the_existing_upload(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights").write_text("immutable")
    destination = tmp_path / "pinned"
    pin_checkpoint(str(source), str(destination))
    with pytest.raises(FileExistsError):
        pin_checkpoint(str(source), str(destination))
    assert (destination / "weights").read_text() == "immutable"
