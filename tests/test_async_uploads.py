from __future__ import annotations

import threading

import pytest


@pytest.fixture
def upload_worker(monkeypatch):
    import flash.engine.worker as worker
    from flash.engine.worker.io import hf as worker_hf

    monkeypatch.setattr(worker_hf, "_OPTIONAL_CHECKPOINT_UPLOADER", worker_hf._SingleSlotUploader())
    monkeypatch.setattr(worker_hf, "_OPTIONAL_AUX_UPLOADER", worker_hf._SingleSlotUploader())
    monkeypatch.setattr(worker_hf, "_OPTIONAL_DEPLOYABLE_UPLOADER", worker_hf._FifoUploader())
    monkeypatch.setattr(worker, "_remaining_worker_wall_seconds", lambda: None)
    return worker, worker_hf


def test_optional_flush_preserves_terminal_deadline_reserve(upload_worker, monkeypatch, tmp_path):
    worker, worker_hf = upload_worker
    started = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    result: list[bool] = []
    staged_dir = tmp_path / "blocked"
    staged_dir.mkdir()

    def blocked_upload() -> None:
        started.set()
        assert release.wait(5)

    worker_hf._OPTIONAL_CHECKPOINT_UPLOADER.enqueue(
        "blocked checkpoint", str(staged_dir), blocked_upload
    )
    assert started.wait(1)
    monkeypatch.setattr(worker, "_remaining_worker_wall_seconds", lambda: 30.0)

    def flush() -> None:
        result.append(worker_hf.flush_optional_uploads())
        completed.set()

    flush_thread = threading.Thread(target=flush)
    flush_thread.start()
    returned_before_release = completed.wait(0.5)
    release.set()
    flush_thread.join(5)

    assert returned_before_release
    assert result == [False]


def test_single_slot_coalesces_pending_uploads_in_order(tmp_path):
    from flash.engine.worker.io.hf import _SingleSlotUploader

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
