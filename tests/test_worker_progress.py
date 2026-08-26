from __future__ import annotations

import inspect
import itertools
import json
import threading
import time

from flash.engine.worker.io import progress as progress_io
from flash.runner.lifecycle.protocol import digest_record


def _reset(monkeypatch) -> None:
    monkeypatch.setattr(progress_io.worker_state, "RUN_ID", "run-1")
    monkeypatch.setattr(progress_io.worker_state, "PHASE", "rl")
    monkeypatch.setattr(progress_io.worker_state, "ATTEMPT", 2)
    monkeypatch.setattr(progress_io.worker_state, "FENCE", 9)
    monkeypatch.setattr(progress_io, "_PROGRESS_SEQUENCE", 0)
    monkeypatch.setattr(progress_io, "_PROGRESS_PREVIOUS_DIGEST", None)
    monkeypatch.setattr(progress_io, "_PROGRESS_TRAINING_ENTERED", False)
    monkeypatch.setattr(progress_io, "_PROGRESS_COMPLETED_STEPS", 0)
    monkeypatch.setattr(progress_io, "_PROGRESS_PENDING_CHECKPOINT_FAILURE", None)
    monkeypatch.setattr(progress_io, "_PROGRESS_FATAL_ERROR", None)
    monkeypatch.setattr(progress_io, "_PROGRESS_COALESCED", None)
    monkeypatch.setattr(progress_io, "_PROGRESS_COALESCE_STARTED_AT", None)
    progress_io._PROGRESS_QUEUE.clear()


def test_progress_api_exposes_only_initial_and_observed_fields() -> None:
    signature = inspect.signature(progress_io.publish_progress)
    assert list(signature.parameters) == ["stage", "initial", "fields"]
    assert signature.parameters["initial"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["fields"].kind is inspect.Parameter.VAR_KEYWORD

    phase_signature = inspect.signature(progress_io.observe_phase)
    assert list(phase_signature.parameters) == ["stage", "progress", "fields", "progress_step"]


def test_failed_upload_reuses_sequence_and_chain_head(monkeypatch) -> None:
    _reset(monkeypatch)
    records = []
    outcomes = iter([False, True])

    def upload(record, *, required):
        records.append((record, required))
        return next(outcomes)

    monkeypatch.setattr(progress_io, "_upload_record", upload)

    assert progress_io.publish_progress("boot") is False
    assert progress_io._PROGRESS_SEQUENCE == 0
    assert progress_io._PROGRESS_PREVIOUS_DIGEST is None

    assert progress_io.publish_progress("boot") is True
    assert [record.sequence for record, _required in records] == [1, 1]
    assert all(record.previous_digest is None for record, _required in records)
    assert progress_io._PROGRESS_SEQUENCE == 1


def test_concurrent_publish_serializes_uploads_and_reuses_failed_sequence(monkeypatch) -> None:
    _reset(monkeypatch)
    records = []
    state_lock = threading.Lock()
    active_uploads = 0
    max_active_uploads = 0
    failed_once = False
    barrier = threading.Barrier(5)

    def upload(record, *, required):
        nonlocal active_uploads, failed_once, max_active_uploads
        with state_lock:
            active_uploads += 1
            max_active_uploads = max(max_active_uploads, active_uploads)
        try:
            time.sleep(0.01)
            with state_lock:
                records.append(record)
                if not failed_once:
                    failed_once = True
                    return False
                return True
        finally:
            with state_lock:
                active_uploads -= 1

    monkeypatch.setattr(progress_io, "_upload_record", upload)

    def publish(index):
        barrier.wait()
        progress_io.publish_progress("phase_observed", step=index)

    threads = [threading.Thread(target=publish, args=(index,)) for index in range(4)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert max_active_uploads == 1
    assert len(records) == 4
    committed = records[1:]
    assert [record.sequence for record in committed] == [1, 2, 3]
    assert records[0].sequence == records[1].sequence == 1
    assert records[0].previous_digest is None
    assert records[1].previous_digest is None
    for previous, current in itertools.pairwise(committed):
        assert current.previous_digest == digest_record(previous.to_dict())


def test_progress_network_upload_does_not_hold_bookkeeping_lock(monkeypatch) -> None:
    _reset(monkeypatch)
    upload_started = threading.Event()
    release_upload = threading.Event()

    def upload(_record, *, required):
        del required
        upload_started.set()
        assert release_upload.wait(1.0)
        return True

    monkeypatch.setattr(progress_io, "_upload_record", upload)
    first = threading.Thread(target=progress_io.publish_progress, args=("boot",))
    second = threading.Thread(
        target=progress_io.publish_progress,
        args=("phase_observed",),
        kwargs={"step": 1},
    )
    first.start()
    assert upload_started.wait(1.0)
    second.start()
    deadline = time.time() + 1.0
    while time.time() < deadline:
        with progress_io._PROGRESS_LOCK:
            if len(progress_io._PROGRESS_QUEUE) == 2:
                break
        time.sleep(0.001)
    else:
        raise AssertionError("second progress publication could not enqueue during network upload")
    release_upload.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert progress_io._PROGRESS_SEQUENCE == 2


def test_ambiguous_upload_verifies_landed_record_before_advancing(monkeypatch) -> None:
    _reset(monkeypatch)
    monkeypatch.setattr(progress_io.worker_state, "HF_REPO", "org/repo")
    landed = {}
    calls = {"count": 0}

    def upload(local, path, *, required):
        del required
        calls["count"] += 1
        with open(local, "rb") as handle:
            landed[path] = handle.read()
        if calls["count"] == 1:
            raise OSError("response lost after commit")
        return True

    monkeypatch.setattr(progress_io.hf_io, "hf_upload_absolute", upload)
    monkeypatch.setattr(progress_io, "_remote_record_payload", landed.get)

    assert progress_io.publish_progress("boot") is True
    assert progress_io.publish_progress("rl_step", step=1) is False
    assert progress_io.flush_progress() is True

    paths = sorted(landed)
    records = [progress_io.ProgressRecord.from_dict(json.loads(landed[path])) for path in paths]
    assert [record.sequence for record in records] == [1, 2]
    assert records[1].previous_digest == digest_record(records[0].to_dict())


def test_optional_upload_verification_blip_retries_identical_record(monkeypatch) -> None:
    _reset(monkeypatch)
    monkeypatch.setattr(progress_io.worker_state, "HF_REPO", "org/repo")
    uploads = []
    upload_outcomes = iter((False, True, True))
    verification_calls = {"count": 0}

    def upload(local, path, *, required):
        with open(local, "rb") as handle:
            uploads.append((path, handle.read(), required))
        return next(upload_outcomes)

    def verify(_path):
        verification_calls["count"] += 1
        raise ConnectionError("temporary readback failure")

    monkeypatch.setattr(progress_io.hf_io, "hf_upload_absolute", upload)
    monkeypatch.setattr(progress_io, "_remote_record_payload", verify)

    assert progress_io.publish_progress("model_prefetching") is False
    assert progress_io._PROGRESS_FATAL_ERROR is None
    assert len(progress_io._PROGRESS_QUEUE) == 1

    monkeypatch.setattr(progress_io, "_remote_record_payload", lambda _path: None)
    assert progress_io.publish_progress("phase_observed", step=1) is True

    assert len(uploads) == 3
    assert uploads[0] == uploads[1]
    first = progress_io.ProgressRecord.from_dict(json.loads(uploads[1][1]))
    second = progress_io.ProgressRecord.from_dict(json.loads(uploads[2][1]))
    assert first.sequence == 1
    assert second.sequence == 2
    assert second.previous_digest == digest_record(first.to_dict())
    assert verification_calls["count"] == 1
    assert not progress_io._PROGRESS_QUEUE


def test_step_progress_coalesces_by_window_and_terminal_stays_dedicated(monkeypatch) -> None:
    _reset(monkeypatch)
    now = {"value": 0.0}
    records = []
    monkeypatch.setattr(progress_io.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(
        progress_io,
        "_upload_record",
        lambda record, *, required: records.append((record, required)) or True,
    )

    for step in range(1, 1001):
        assert progress_io.publish_progress("rl_step", step=step) is False
    assert records == []
    assert progress_io.flush_progress() is True
    assert [record.completed_steps for record, _required in records] == [1000]

    now["value"] = progress_io._PROGRESS_STEP_CADENCE_S
    for step in range(1001, 2001):
        assert progress_io.publish_progress("rl_step", step=step) is False
    assert progress_io.flush_progress() is True
    progress_io.publish_progress("result_published", step=2000)

    assert [record.completed_steps for record, _required in records] == [1000, 2000, 2000]
    assert [record.phase for record, _required in records] == [
        "rl_step",
        "rl_step",
        "result_published",
    ]
    assert [record.sequence for record, _required in records] == [1, 2, 3]


def test_checkpoint_failure_is_sticky_until_a_successful_checkpoint(monkeypatch) -> None:
    _reset(monkeypatch)
    records = []
    monkeypatch.setattr(
        progress_io,
        "_upload_record",
        lambda record, *, required: records.append(record) or True,
    )
    failure = {"step": 50, "operation": "resume", "error": "quota denied"}

    progress_io.publish_progress("checkpoint_upload_failed", step=50, checkpoint_failure=failure)
    progress_io.publish_progress("sft_step", step=60)
    assert records[-1].checkpoint == failure

    progress_io.publish_progress("checkpoint_uploaded", step=75)
    assert progress_io.pending_checkpoint_failure() is None
    progress_io.publish_progress("sft_step", step=80)
    assert records[-1].checkpoint == {}


def test_bounded_reward_metrics_sanitizes_and_bounds_names() -> None:
    long_name = "x" * 100_000

    bounded = progress_io._bounded_reward_metrics(
        {
            long_name: 1.0,
            "line\nbreak": 2.0,
            "reward": 3.0,
            "step": 4.0,
        }
    )

    assert "x" * 64 in bounded
    assert all(len(name) <= 64 for name in bounded)
    assert "linebreak" in bounded
    assert all("\n" not in name for name in bounded)
    assert "reward" not in bounded
    assert "step" not in bounded
