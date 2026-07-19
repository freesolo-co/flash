from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest


class _BlockingHfApi:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.uploads: list[tuple[str, bytes]] = []

    def upload_folder(self, **kwargs) -> None:
        self.started.set()
        assert self.release.wait(5)
        folder = Path(kwargs["folder_path"])
        self.uploads.append(
            (kwargs["path_in_repo"], (folder / "adapter_model.safetensors").read_bytes())
        )

    def list_repo_files(self, **_kwargs) -> list[str]:
        return []


@pytest.fixture
def upload_worker(monkeypatch, tmp_path):
    import flash.engine.worker as worker
    from flash.engine.worker import hf as worker_hf

    monkeypatch.setattr(worker_hf, "_OPTIONAL_UPLOADER", worker_hf._SingleSlotUploader())
    monkeypatch.setattr(worker_hf, "_OPTIONAL_UPLOAD_STAGE_ROOT", str(tmp_path / "staged"))
    monkeypatch.setattr(worker, "HF_REPO", "org/runs")
    monkeypatch.setattr(worker, "PHASE", "rl")
    monkeypatch.setattr(worker, "RUN_ID", "async-test")
    monkeypatch.setattr(worker, "JOB_SPEC", None)
    monkeypatch.setattr(worker, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_remaining_worker_wall_seconds", lambda: None)
    return worker, worker_hf


@pytest.fixture
def fake_trainer_callback(monkeypatch):
    import sys
    import types

    module = types.ModuleType("transformers")

    class TrainerCallback:
        pass

    module.TrainerCallback = TrainerCallback
    monkeypatch.setitem(sys.modules, "transformers", module)


def _checkpoint(output_dir: Path, step: int, contents: bytes = b"before") -> Path:
    checkpoint = output_dir / f"checkpoint-{step}"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}")
    (checkpoint / "adapter_model.safetensors").write_bytes(contents)
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    return checkpoint


def test_optional_checkpoint_is_nonblocking_immutable_and_flushed(
    upload_worker, fake_trainer_callback, monkeypatch, tmp_path
):
    worker, _worker_hf = upload_worker
    api = _BlockingHfApi()
    monkeypatch.setattr(worker, "hf_api", lambda: api)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    checkpoint = _checkpoint(output_dir, 4)
    callback = worker.make_checkpoint_upload_callback()

    callback.on_save(
        SimpleNamespace(output_dir=str(output_dir)),
        SimpleNamespace(global_step=4),
        SimpleNamespace(),
    )

    assert api.started.wait(1)
    (checkpoint / "adapter_model.safetensors").write_bytes(b"mutated")
    shutil.rmtree(checkpoint)
    _checkpoint(output_dir, 4, b"after")

    finalized = threading.Event()

    def finalize() -> None:
        callback.on_train_end(
            SimpleNamespace(output_dir=str(output_dir)),
            SimpleNamespace(global_step=4),
            SimpleNamespace(),
        )
        finalized.set()

    finalizer = threading.Thread(target=finalize)
    finalizer.start()
    assert not finalized.wait(0.05)
    api.release.set()
    finalizer.join(5)

    assert finalized.is_set()
    assert [contents for _path, contents in api.uploads] == [b"before", b"before"]
    assert [path for path, _contents in api.uploads] == [
        "rl/async-test/checkpoints/step-4/adapter",
        "rl/async-test/checkpoint/checkpoint-4",
    ]


def test_required_checkpoint_stays_synchronous(
    upload_worker, fake_trainer_callback, monkeypatch, tmp_path
):
    worker, _worker_hf = upload_worker
    api = _BlockingHfApi()
    monkeypatch.setattr(worker, "hf_api", lambda: api)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _checkpoint(output_dir, 4)
    callback = worker.make_checkpoint_upload_callback((4,))
    returned = threading.Event()

    def save() -> None:
        callback.on_save(
            SimpleNamespace(output_dir=str(output_dir)),
            SimpleNamespace(global_step=4),
            SimpleNamespace(),
        )
        returned.set()

    save_thread = threading.Thread(target=save)
    save_thread.start()
    assert api.started.wait(1)
    assert not returned.wait(0.05)
    api.release.set()
    save_thread.join(5)

    assert returned.is_set()
    assert [path for path, _contents in api.uploads] == [
        "rl/async-test/checkpoints/step-4/adapter",
        "rl/async-test/checkpoint/checkpoint-4",
    ]


def test_reward_debug_upload_is_enqueued_without_waiting(upload_worker, monkeypatch):
    worker, worker_hf = upload_worker
    started = threading.Event()
    release = threading.Event()
    uploaded: list[list[dict]] = []

    def upload(path: str, repo_name: str) -> bool:
        assert repo_name == "reward_debug_async_test.jsonl"
        started.set()
        assert release.wait(5)
        uploaded.append([json.loads(line) for line in Path(path).read_text().splitlines()])
        return True

    debug_path = Path("/tmp/reward_debug_async_test.jsonl")
    debug_path.unlink(missing_ok=True)
    monkeypatch.setattr(worker, "hf_upload_file", upload)
    worker.upload_debug_jsonl("reward_debug_async_test.jsonl", [{"reward": 1.0}])

    assert started.wait(1)
    assert uploaded == []
    release.set()
    assert worker_hf.flush_optional_uploads(5)
    assert uploaded == [[{"reward": 1.0}]]


def test_single_slot_coalesces_pending_uploads_in_order(tmp_path):
    from flash.engine.worker.hf import _SingleSlotUploader

    uploader = _SingleSlotUploader()
    release = threading.Event()
    started = threading.Event()
    order: list[int] = []

    def stage(number: int) -> str:
        path = tmp_path / str(number)
        path.mkdir()
        return str(path)

    def first() -> None:
        order.append(1)
        started.set()
        assert release.wait(5)

    uploader.enqueue("one", stage(1), first)
    assert started.wait(1)
    uploader.enqueue("two", stage(2), lambda: order.append(2))
    uploader.enqueue("three", stage(3), lambda: order.append(3))
    release.set()

    assert uploader.flush(5)
    assert order == [1, 3]


def test_train_end_timeout_does_not_start_a_concurrent_fallback_upload(
    upload_worker, fake_trainer_callback, monkeypatch, tmp_path
):
    worker, worker_hf = upload_worker
    uploads: list[str] = []

    class RecordingApi:
        def upload_folder(self, **kwargs) -> None:
            uploads.append(kwargs["path_in_repo"])

        def list_repo_files(self, **_kwargs) -> list[str]:
            return []

    monkeypatch.setattr(worker, "hf_api", lambda: RecordingApi())
    monkeypatch.setattr(worker_hf, "flush_optional_uploads", lambda: False)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _checkpoint(output_dir, 4)

    worker.make_checkpoint_upload_callback().on_train_end(
        SimpleNamespace(output_dir=str(output_dir)),
        SimpleNamespace(global_step=4),
        SimpleNamespace(),
    )

    assert uploads == []


def test_concurrent_debug_snapshots_preserve_append_order(upload_worker, monkeypatch):
    worker, worker_hf = upload_worker
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    uploaded: list[list[dict]] = []

    class CoordinatedUploader:
        def enqueue(self, _label: str, staged_dir: str, run) -> None:
            if not first_entered.is_set():
                first_entered.set()
                assert release_first.wait(5)
            else:
                second_entered.set()
            try:
                run()
            finally:
                shutil.rmtree(staged_dir, ignore_errors=True)

    debug_path = Path("/tmp/reward_debug_order_test.jsonl")
    debug_path.unlink(missing_ok=True)

    def upload(path: str, _repo_name: str) -> bool:
        uploaded.append([json.loads(line) for line in Path(path).read_text().splitlines()])
        return True

    monkeypatch.setattr(worker_hf, "_OPTIONAL_UPLOADER", CoordinatedUploader())
    monkeypatch.setattr(worker, "hf_upload_file", upload)
    first = threading.Thread(
        target=worker.upload_debug_jsonl,
        args=("reward_debug_order_test.jsonl", [{"reward": 1.0}]),
    )
    second = threading.Thread(
        target=worker.upload_debug_jsonl,
        args=("reward_debug_order_test.jsonl", [{"reward": 2.0}]),
    )

    first.start()
    assert first_entered.wait(1)
    second.start()
    second_overtook_first = second_entered.wait(0.05)
    release_first.set()
    first.join(5)
    second.join(5)
    debug_path.unlink(missing_ok=True)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not second_overtook_first
    assert uploaded == [
        [{"reward": 1.0}],
        [{"reward": 1.0}, {"reward": 2.0}],
    ]
