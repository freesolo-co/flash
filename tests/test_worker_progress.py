from __future__ import annotations

from flash.engine.worker.io import progress as progress_io


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


def test_liveness_only_observation_publishes_no_record(monkeypatch) -> None:
    _reset(monkeypatch)
    uploads = []
    monkeypatch.setattr(
        progress_io,
        "_upload_record",
        lambda record, *, required: uploads.append((record, required)) or True,
    )

    assert progress_io.publish_progress("model_prefetching", liveness=True) is False
    assert uploads == []
    assert progress_io._PROGRESS_SEQUENCE == 0


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
