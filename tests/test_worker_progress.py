from __future__ import annotations

import contextlib
import inspect
import itertools
import json
import os
import threading
import time

import pytest

from flash.engine.worker.io import progress as progress_io
from flash.runner.lifecycle.protocol import digest_record


def _reset(monkeypatch) -> None:
    with contextlib.suppress(OSError):
        os.unlink(progress_io.supervisor_snapshot_path("run-1", "rl", 2, 9))
    monkeypatch.setattr(progress_io.worker_state, "RUN_ID", "run-1")
    monkeypatch.setattr(progress_io.worker_state, "PHASE", "rl")
    monkeypatch.setattr(progress_io.worker_state, "ATTEMPT", 2)
    monkeypatch.setattr(progress_io.worker_state, "FENCE", 9)
    monkeypatch.setattr(progress_io, "_PROGRESS_SEQUENCE", 0)
    monkeypatch.setattr(progress_io, "_PROGRESS_PREVIOUS_DIGEST", None)
    monkeypatch.setattr(progress_io, "_PROGRESS_LAST_COMMITTED_OCCURRED_AT", 0.0)
    monkeypatch.setattr(progress_io, "_PROGRESS_TRAINING_ENTERED", False)
    monkeypatch.setattr(progress_io, "_PROGRESS_COMPLETED_STEPS", 0)
    monkeypatch.setattr(progress_io, "_PROGRESS_LATEST_METRICS", {})
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


def test_optional_deferred_target_stays_queued_and_nonfatal(monkeypatch) -> None:
    _reset(monkeypatch)
    attempts = []

    def upload(record, *, required):
        attempts.append((record, required))
        raise progress_io._ProgressUploadDeferred("optional readback unavailable")

    monkeypatch.setattr(progress_io, "_upload_record", upload)

    assert progress_io.publish_optional_progress("model_prefetching", initial=True) is False
    assert len(progress_io._PROGRESS_QUEUE) == 1
    assert progress_io._PROGRESS_QUEUE[0].record is attempts[0][0]
    assert attempts[0][1] is False
    assert progress_io._PROGRESS_FATAL_ERROR is None
    assert progress_io._PROGRESS_SEQUENCE == 0
    assert progress_io._PROGRESS_PREVIOUS_DIGEST is None


def test_required_boundary_resolves_landed_deferred_optional_predecessor(monkeypatch) -> None:
    _reset(monkeypatch)
    monkeypatch.setattr(progress_io.worker_state, "HF_REPO", "org/repo")
    remote = {}
    uploads = []
    verification_calls = {"count": 0}

    def upload(local, path, *, required):
        with open(local, "rb") as handle:
            payload = handle.read()
        uploads.append((path, payload, required))
        remote[path] = payload
        return len(uploads) == 3

    def verify(path):
        verification_calls["count"] += 1
        if verification_calls["count"] == 1:
            raise ConnectionError("temporary readback failure")
        return remote.get(path)

    monkeypatch.setattr(progress_io.hf_io, "hf_upload_absolute", upload)
    monkeypatch.setattr(progress_io, "_remote_record_payload", verify)

    assert progress_io.publish_optional_progress("model_prefetching", initial=True) is False
    deferred = progress_io._PROGRESS_QUEUE[0]
    assert progress_io.publish_progress("boot", initial=True) is True

    assert uploads[0][:2] == uploads[1][:2]
    assert [required for _path, _payload, required in uploads] == [False, True, True]
    records = [
        progress_io.ProgressRecord.from_dict(json.loads(remote[path])) for path in sorted(remote)
    ]
    assert [record.sequence for record in records] == [1, 2]
    assert records[0].previous_digest is None
    assert records[1].previous_digest == digest_record(records[0].to_dict())
    assert deferred.committed is True
    assert progress_io._PROGRESS_SEQUENCE == 2
    assert digest_record(records[1].to_dict()) == progress_io._PROGRESS_PREVIOUS_DIGEST
    assert len({path for path in remote if "/progress/00000000000000000001-" in path}) == 1
    assert not progress_io._PROGRESS_QUEUE
    assert progress_io._PROGRESS_FATAL_ERROR is None


def test_required_boundary_fails_closed_on_unresolved_optional_predecessor(monkeypatch) -> None:
    _reset(monkeypatch)
    attempts = []

    def upload(record, *, required):
        attempts.append((record, required))
        raise progress_io._ProgressUploadDeferred("optional readback unavailable")

    monkeypatch.setattr(progress_io, "_upload_record", upload)

    assert progress_io.publish_optional_progress("model_prefetching", initial=True) is False
    with pytest.raises(progress_io._ProgressUploadDeferred, match="optional readback unavailable"):
        progress_io.publish_progress("boot", initial=True)

    assert attempts[0][0].to_dict() == attempts[1][0].to_dict()
    assert [required for _record, required in attempts] == [False, True]
    assert progress_io._PROGRESS_SEQUENCE == 0
    assert progress_io._PROGRESS_PREVIOUS_DIGEST is None
    assert isinstance(progress_io._PROGRESS_FATAL_ERROR, progress_io._ProgressUploadDeferred)
    assert not progress_io._PROGRESS_QUEUE


def test_required_boundary_fails_closed_on_conflicting_promoted_predecessor(
    monkeypatch,
) -> None:
    _reset(monkeypatch)
    monkeypatch.setattr(progress_io.worker_state, "HF_REPO", "org/repo")
    remote = {}
    uploads = []
    verification_calls = {"count": 0}

    def upload(local, path, *, required):
        with open(local, "rb") as handle:
            payload = handle.read()
        uploads.append((path, payload, required))
        if len(uploads) == 1:
            remote[path] = payload
        return False

    def verify(path):
        verification_calls["count"] += 1
        if verification_calls["count"] == 1:
            raise ConnectionError("temporary readback failure")
        remote[path] = b'{"conflicting":"immutable record"}'
        return remote[path]

    monkeypatch.setattr(progress_io.hf_io, "hf_upload_absolute", upload)
    monkeypatch.setattr(progress_io, "_remote_record_payload", verify)

    assert progress_io.publish_optional_progress("model_prefetching", initial=True) is False
    with pytest.raises(RuntimeError, match="immutable progress path contains different bytes"):
        progress_io.publish_progress("boot", initial=True)

    assert uploads[0][:2] == uploads[1][:2]
    assert [required for _path, _payload, required in uploads] == [False, True]
    assert len({path for path, _payload, _required in uploads}) == 1
    assert len(remote) == 1
    assert progress_io._PROGRESS_SEQUENCE == 0
    assert progress_io._PROGRESS_PREVIOUS_DIGEST is None
    assert isinstance(progress_io._PROGRESS_FATAL_ERROR, RuntimeError)
    assert not progress_io._PROGRESS_QUEUE


def test_required_deferred_upload_fails_closed(monkeypatch) -> None:
    _reset(monkeypatch)
    monkeypatch.setattr(
        progress_io,
        "_upload_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            progress_io._ProgressUploadDeferred("required readback unavailable")
        ),
    )

    with pytest.raises(progress_io._ProgressUploadDeferred, match="required readback unavailable"):
        progress_io.publish_progress("boot", initial=True)
    assert isinstance(progress_io._PROGRESS_FATAL_ERROR, progress_io._ProgressUploadDeferred)


def test_required_false_upload_latches_fatal_and_clears_queue(monkeypatch) -> None:
    _reset(monkeypatch)
    blocked = progress_io._PendingProgress("phase_observed", False, False, {})

    def upload(*_args, **_kwargs):
        with progress_io._PROGRESS_LOCK:
            progress_io._PROGRESS_QUEUE.append(blocked)
        return False

    monkeypatch.setattr(progress_io, "_upload_record", upload)

    with pytest.raises(
        RuntimeError, match="required progress record was not verified as committed"
    ):
        progress_io.publish_progress("boot", initial=True)

    fatal = progress_io._PROGRESS_FATAL_ERROR
    assert isinstance(fatal, RuntimeError)
    assert blocked.error is fatal
    assert not progress_io._PROGRESS_QUEUE
    with pytest.raises(
        RuntimeError, match="required progress record was not verified as committed"
    ):
        progress_io.publish_progress("phase_observed")


def test_optional_permanent_upload_failure_is_observational(monkeypatch) -> None:
    _reset(monkeypatch)
    monkeypatch.setattr(
        progress_io,
        "_upload_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    assert progress_io.publish_optional_progress("model_prefetching", initial=True) is False
    assert progress_io._PROGRESS_FATAL_ERROR is None
    assert not progress_io._PROGRESS_QUEUE


def test_required_permanent_upload_failure_remains_fatal(monkeypatch) -> None:
    _reset(monkeypatch)
    monkeypatch.setattr(
        progress_io,
        "_upload_record",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(PermissionError("denied")),
    )

    with pytest.raises(PermissionError, match="denied"):
        progress_io.publish_progress("model_prefetching", initial=True)
    assert isinstance(progress_io._PROGRESS_FATAL_ERROR, PermissionError)


def test_deferred_boundary_retries_when_next_step_coalesces(monkeypatch) -> None:
    _reset(monkeypatch)
    records = []
    outcomes = iter((_ProgressUploadDeferredForTest, True, True))

    def upload(record, *, required):
        records.append((record, required))
        outcome = next(outcomes)
        if outcome is _ProgressUploadDeferredForTest:
            raise progress_io._ProgressUploadDeferred("readback unavailable")
        return outcome

    monkeypatch.setattr(progress_io, "_upload_record", upload)

    assert progress_io.publish_progress("model_prefetching") is False
    assert len(progress_io._PROGRESS_QUEUE) == 1
    boundary = progress_io._PROGRESS_QUEUE[0].record

    assert progress_io.publish_progress("rl_step", step=7) is False

    assert records[0][0].to_dict() == records[1][0].to_dict() == boundary.to_dict()
    assert not progress_io._PROGRESS_QUEUE
    assert progress_io._PROGRESS_COALESCED is not None
    assert progress_io.flush_progress() is True
    assert [record.sequence for record, _required in records] == [1, 1, 2]
    assert records[-1][0].previous_digest == digest_record(boundary.to_dict())


_ProgressUploadDeferredForTest = object()


def test_progress_occurred_at_never_moves_backward(monkeypatch) -> None:
    _reset(monkeypatch)
    times = iter((100.0, 90.0, 95.0, 110.0))
    records = []
    monkeypatch.setattr(progress_io.time, "time", lambda: next(times))
    monkeypatch.setattr(
        progress_io,
        "_upload_record",
        lambda record, *, required: records.append(record) or True,
    )

    progress_io.publish_progress("boot")
    progress_io.publish_progress("phase_observed")
    progress_io.publish_progress("checkpoint_uploaded")
    progress_io.publish_progress("result_published")

    assert [record.occurred_at for record in records] == [100.0, 100.0, 100.0, 110.0]
    assert [record.sequence for record in records] == [1, 2, 3, 4]
    assert records[2].previous_digest == digest_record(records[1].to_dict())


def test_cadence_expiry_publishes_latest_step_without_duplicate_flush(monkeypatch) -> None:
    _reset(monkeypatch)
    now = {"value": 0.0}
    records = []
    monkeypatch.setattr(progress_io.time, "monotonic", lambda: now["value"])
    monkeypatch.setattr(
        progress_io,
        "_upload_record",
        lambda record, *, required: records.append(record) or True,
    )

    assert progress_io.publish_progress("rl_step", step=1, loss=1.0) is False
    now["value"] = progress_io._PROGRESS_STEP_CADENCE_S
    assert progress_io.publish_progress("rl_step", step=2, loss=2.0) is False

    assert len(records) == 1
    assert records[0].completed_steps == 2
    assert records[0].metrics["loss"] == 2.0
    assert progress_io.flush_progress() is True
    assert len(records) == 1


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


@pytest.mark.parametrize(
    ("stage", "fields", "expected_checkpoint"),
    [
        ("checkpoint_uploaded", {"step": 50}, {"step": 50}),
        (
            "checkpoint_deployable",
            {"step": 75, "subfolder": "checkpoint-75"},
            {"step": 75, "subfolder": "checkpoint-75"},
        ),
    ],
)
def test_successful_checkpoint_progress_projects_metadata(
    monkeypatch, stage, fields, expected_checkpoint
) -> None:
    _reset(monkeypatch)
    records = []
    monkeypatch.setattr(
        progress_io,
        "_upload_record",
        lambda record, *, required: records.append(record) or True,
    )

    progress_io.publish_progress(stage, **fields)

    assert records[-1].kind == "checkpoint_saved"
    assert records[-1].checkpoint == expected_checkpoint
    assert records[-1].diagnostics == {}


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
    assert records[-1].kind == "checkpoint_failed"
    assert records[-1].checkpoint == failure
    assert records[-1].diagnostics == {}
    progress_io.publish_progress("sft_step", step=60)
    assert records[-1].checkpoint == failure

    progress_io.publish_progress("checkpoint_uploaded", step=75)
    assert progress_io.pending_checkpoint_failure() is None
    progress_io.publish_progress("sft_step", step=80)
    progress_io.flush_progress()
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
