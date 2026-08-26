from __future__ import annotations

import inspect
import itertools
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
        progress_io.publish_progress("rl_step", step=index)

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
