from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from flash.engine.worker.io import result as result_io
from flash.runner.lifecycle.protocol import ResultManifest, result_path

ATTESTATION = {
    "kind": "flash-source-attestation",
    "format_version": 1,
    "sha256": "a" * 64,
    "revision": "b" * 40,
    "run_id": "run-1",
    "attempt": 2,
    "fence": 9,
}


def _manifest(**updates) -> ResultManifest:
    values = {
        "run_id": "run-1",
        "phase_namespace": "rl",
        "attempt_id": 2,
        "fence": 9,
        "outcome": "succeeded",
        "failure_class": None,
        "started_at": 100.0,
        "finished_at": 120.0,
        "training_entered": True,
        "completed_steps": 1,
        "metrics": {"step": 1},
        "checkpoint": {},
        "artifacts": {"adapter": "published"},
        "source_attestation": ATTESTATION,
        "diagnostics": {},
    }
    values.update(updates)
    return ResultManifest(**values)


def _set_identity(monkeypatch) -> None:
    monkeypatch.setattr(result_io.state, "HF_REPO", "org/repo")
    monkeypatch.setattr(result_io.state, "RUN_ID", "run-1")
    monkeypatch.setattr(result_io.state, "PHASE", "rl")
    monkeypatch.setattr(result_io.state, "ATTEMPT", 2)
    monkeypatch.setattr(result_io.state, "FENCE", 9)
    monkeypatch.setattr(result_io.hf_io, "_require_hf_deadline_allowance", lambda: None)
    monkeypatch.setattr(result_io.hf_io, "_sleep_with_hf_deadline", lambda _delay: True)


def test_current_fence_result_preflight_reuses_strict_artifact_validation(monkeypatch) -> None:
    from flash.providers.artifacts import attempts
    from tests._helpers.source_snapshot import valid_source_snapshot

    _set_identity(monkeypatch)
    snapshot = valid_source_snapshot()
    monkeypatch.setenv("FLASH_SOURCE_SNAPSHOT_JSON", json.dumps(snapshot))
    projection = {**_manifest().to_dict(), "observed_at": 123.0, "receipt": {}}
    monkeypatch.setattr(
        attempts,
        "read_attempt_artifacts",
        lambda *args, **kwargs: attempts.AttemptArtifacts("c" * 40, 123.0, None, projection),
    )

    assert result_io.preflight_existing_terminal_result() == _manifest()


def test_current_fence_result_preflight_maps_transient_observation_to_retriable(
    monkeypatch,
) -> None:
    from flash.providers.artifacts import attempts
    from tests._helpers.source_snapshot import valid_source_snapshot

    _set_identity(monkeypatch)
    monkeypatch.setenv("FLASH_SOURCE_SNAPSHOT_JSON", json.dumps(valid_source_snapshot()))
    monkeypatch.setattr(
        attempts,
        "read_attempt_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")),
    )

    with pytest.raises(result_io.hf_io.RetriableInfraError, match="temporarily unavailable"):
        result_io.preflight_existing_terminal_result()


def test_exactly_once_publish_adopts_matching_concurrent_result(monkeypatch, tmp_path) -> None:
    _set_identity(monkeypatch)
    proposed = _manifest(finished_at=121.0)
    existing = _manifest(finished_at=120.0)
    result_prefix = result_path(existing).rsplit("/", 1)[0] + "/"

    class Api:
        def __init__(self) -> None:
            self.revision = "c" * 40
            self.created = 0

        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha=self.revision)

        def list_repo_files(self, **_kwargs):
            return [] if self.created == 0 else [result_path(existing)]

        def create_commit(self, **kwargs):
            assert kwargs["parent_commit"] == "c" * 40
            assert kwargs["operations"][0].path_in_repo.startswith(result_prefix)
            self.created += 1
            self.revision = "d" * 40
            raise RuntimeError("parent commit changed")

    api = Api()
    monkeypatch.setattr(result_io.hf_io, "hf_api", lambda: api)
    monkeypatch.setattr(result_io, "_download_result", lambda _path, *, revision: existing)
    local = tmp_path / "result.json"
    local.write_text("{}")

    observed = result_io._publish_exactly_once(proposed, str(local))

    assert observed == existing
    assert api.created == 1


def test_cancelled_result_recovers_coalesced_supervisor_snapshot(monkeypatch, tmp_path) -> None:
    from flash.engine.worker.io import progress as progress_io

    _set_identity(monkeypatch)
    monkeypatch.setattr(progress_io, "_SUPERVISOR_SNAPSHOT_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(progress_io.worker_state, "RUN_ID", "run-1")
    monkeypatch.setattr(progress_io.worker_state, "PHASE", "rl")
    monkeypatch.setattr(progress_io.worker_state, "ATTEMPT", 2)
    monkeypatch.setattr(progress_io.worker_state, "FENCE", 9)
    progress_io._PROGRESS_QUEUE.clear()
    monkeypatch.setattr(progress_io, "_PROGRESS_COALESCED", None)
    monkeypatch.setattr(progress_io, "_PROGRESS_COALESCE_STARTED_AT", None)
    monkeypatch.setattr(progress_io, "_PROGRESS_SEQUENCE", 0)
    monkeypatch.setattr(progress_io, "_PROGRESS_PREVIOUS_DIGEST", None)
    monkeypatch.setattr(progress_io, "_PROGRESS_LAST_COMMITTED_OCCURRED_AT", 0.0)
    monkeypatch.setattr(progress_io, "_PROGRESS_TRAINING_ENTERED", False)
    monkeypatch.setattr(progress_io, "_PROGRESS_COMPLETED_STEPS", 0)
    monkeypatch.setattr(progress_io, "_PROGRESS_LATEST_METRICS", {})
    monkeypatch.setattr(progress_io, "_PROGRESS_PENDING_CHECKPOINT_FAILURE", None)
    monkeypatch.setattr(progress_io, "_PROGRESS_FATAL_ERROR", None)
    uploads = []
    monkeypatch.setattr(
        progress_io,
        "_upload_record",
        lambda record, *, required: uploads.append(record) or True,
    )

    assert progress_io.publish_progress("boot") is True
    assert progress_io.publish_progress("rl_step", step=7, loss=0.25) is False
    assert len(uploads) == 1

    progress_io._PROGRESS_QUEUE.clear()
    monkeypatch.setattr(progress_io, "_PROGRESS_COALESCED", None)
    monkeypatch.setattr(progress_io, "_PROGRESS_COALESCE_STARTED_AT", None)
    monkeypatch.setattr(progress_io, "_PROGRESS_SEQUENCE", 0)
    monkeypatch.setattr(progress_io, "_PROGRESS_PREVIOUS_DIGEST", None)
    monkeypatch.setattr(progress_io, "_PROGRESS_LAST_COMMITTED_OCCURRED_AT", 0.0)
    monkeypatch.setattr(progress_io, "_PROGRESS_TRAINING_ENTERED", False)
    monkeypatch.setattr(progress_io, "_PROGRESS_COMPLETED_STEPS", 0)
    monkeypatch.setattr(progress_io, "_PROGRESS_LATEST_METRICS", {})
    monkeypatch.setattr(result_io, "_source_attestation", lambda: ATTESTATION)
    monkeypatch.setattr(result_io.time, "time", lambda: 120.0)
    monkeypatch.setattr(result_io, "_write_immutable", lambda _payload: str(tmp_path / "result"))
    monkeypatch.setattr(result_io, "_publish_exactly_once", lambda manifest, _path: manifest)

    manifest = result_io.publish_cancelled_result(started_at=100.0)

    assert manifest.training_entered is True
    assert manifest.completed_steps == 7
    assert manifest.metrics == {"loss": 0.25}
    assert len(uploads) == 1


def test_supervisor_snapshot_rejects_foreign_fence(monkeypatch, tmp_path) -> None:
    from flash.engine.worker.io import progress as progress_io

    _set_identity(monkeypatch)
    monkeypatch.setattr(progress_io, "_SUPERVISOR_SNAPSHOT_DIRECTORY", str(tmp_path))
    path = progress_io.supervisor_snapshot_path("run-1", "rl", 2, 9)
    with open(path, "w") as handle:
        handle.write(
            '{"schema_version":1,"run_id":"run-1","phase_namespace":"rl",'
            '"attempt_id":2,"fence":8,"training_entered":true,"completed_steps":99,'
            '"metrics":{},"checkpoint":{}}'
        )

    assert result_io._supervisor_snapshot() == {}


def test_bootstrap_failure_result_is_fenced_worker_failure(monkeypatch, tmp_path) -> None:
    from flash.providers.artifacts.attempts import poll_result_from_manifest

    _set_identity(monkeypatch)
    monkeypatch.setenv("FLASH_BOOTSTRAP_ERROR", "invalid requirement")
    monkeypatch.setattr(result_io, "_source_attestation", lambda: ATTESTATION)
    monkeypatch.setattr(result_io.time, "time", lambda: 120.0)
    monkeypatch.setattr(result_io, "_write_immutable", lambda _payload: str(tmp_path / "result"))
    publications = []
    monkeypatch.setattr(
        result_io,
        "_publish_exactly_once",
        lambda manifest, _path: publications.append(manifest) or manifest,
    )

    manifest = result_io.publish_bootstrap_failure_result(started_at=100.0)
    projection = poll_result_from_manifest(manifest.to_dict())

    assert len(publications) == 1
    assert manifest.failure_class == "worker"
    assert manifest.training_entered is False
    assert manifest.completed_steps == 0
    assert manifest.source_attestation == ATTESTATION
    assert projection.failure == "job_failed"
    assert projection.detail == "invalid requirement"


def test_transient_bootstrap_failure_result_remains_retriable(monkeypatch, tmp_path) -> None:
    from flash.providers.artifacts.attempts import poll_result_from_manifest

    _set_identity(monkeypatch)
    monkeypatch.setenv("FLASH_BOOTSTRAP_ERROR", "index unavailable")
    monkeypatch.setenv("FLASH_BOOTSTRAP_FAILURE_CLASS", "artifact_transport")
    monkeypatch.setattr(result_io, "_source_attestation", lambda: ATTESTATION)
    monkeypatch.setattr(result_io.time, "time", lambda: 120.0)
    monkeypatch.setattr(result_io, "_write_immutable", lambda _payload: str(tmp_path / "result"))
    monkeypatch.setattr(result_io, "_publish_exactly_once", lambda manifest, _path: manifest)

    manifest = result_io.publish_bootstrap_failure_result(started_at=100.0)
    projection = poll_result_from_manifest(manifest.to_dict())

    assert manifest.failure_class == "artifact_transport"
    assert projection.failure == "artifact_transport"
    assert projection.detail == "index unavailable"


def test_result_publication_continues_after_optional_progress_flush_failure(
    monkeypatch, tmp_path
) -> None:
    from flash.engine.worker.io import progress as progress_io

    _set_identity(monkeypatch)
    monkeypatch.setattr(
        progress_io,
        "flush_progress",
        lambda: (_ for _ in ()).throw(ConnectionError("optional progress unavailable")),
    )
    monkeypatch.setattr(result_io.time, "time", lambda: 120.0)
    monkeypatch.setattr(result_io, "_source_attestation", lambda: ATTESTATION)
    monkeypatch.setattr(
        result_io, "_write_immutable", lambda _payload: str(tmp_path / "result.json")
    )
    monkeypatch.setattr(result_io, "_publish_exactly_once", lambda manifest, _path: manifest)

    published = result_io.publish_result(
        outcome="succeeded",
        failure_class=None,
        started_at=100.0,
        training_entered=True,
        completed_steps=4,
        metrics={"step": 4},
        artifacts={"adapter": "published"},
    )

    assert published.outcome == "succeeded"
    assert published.completed_steps == 4


def test_backward_clock_correction_still_publishes_a_terminal_result(monkeypatch, tmp_path) -> None:
    """a wall-clock correction must not cost an attempt its only terminal authority.

    ``ResultManifest`` rejects ``finished_at < started_at``. if the host clock steps backward
    mid-attempt, an unclamped ``time.time()`` makes that rejection fire on an otherwise successful
    attempt, so no manifest is published at all and the poller eventually reports the finished run
    as preempted. the progress journal already clamps its occurrence times for this reason.
    """
    _set_identity(monkeypatch)
    monkeypatch.setattr(result_io, "_source_attestation", lambda: ATTESTATION)
    monkeypatch.setattr(
        result_io, "_write_immutable", lambda _payload: str(tmp_path / "result.json")
    )
    monkeypatch.setattr(result_io, "_publish_exactly_once", lambda manifest, _path: manifest)
    # the clock stepped back behind the recorded start.
    monkeypatch.setattr(result_io.time, "time", lambda: 90.0)

    published = result_io.publish_result(
        outcome="succeeded",
        failure_class=None,
        started_at=100.0,
        training_entered=True,
        completed_steps=4,
        metrics={"step": 4},
        artifacts={"adapter": "published"},
    )

    assert published.outcome == "succeeded"
    assert published.finished_at == 100.0, "finished_at must clamp forward to started_at"


def test_result_publication_flushes_the_coalesced_step_before_the_manifest(
    monkeypatch, tmp_path
) -> None:
    """the manifest may not outrun the progress record billing reads.

    step progress coalesces to a 900s cadence, so at any moment a completed step can be pending
    rather than published. cancellation racing a terminal manifest drops that manifest and bills
    from progress alone, so if the manifest could be published while a step was still coalesced,
    billing would undercount by up to one full cadence. ``publish_result`` flushes first, which
    makes the pinned snapshot's progress carry the same cumulative count the manifest claims.
    """
    from flash.engine.worker.io import progress as progress_io

    _set_identity(monkeypatch)
    uploaded = []
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
    monkeypatch.setattr(progress_io, "_PROGRESS_LAST_COMMITTED_OCCURRED_AT", 0.0)
    monkeypatch.setattr(progress_io, "_persist_supervisor_snapshot", lambda fields, *, stage: None)
    monkeypatch.setattr(
        progress_io, "_upload_record", lambda record, *, required: uploaded.append(record) or True
    )
    progress_io._PROGRESS_QUEUE.clear()
    monkeypatch.setattr(result_io.time, "time", lambda: 120.0)
    monkeypatch.setattr(result_io, "_source_attestation", lambda: ATTESTATION)
    monkeypatch.setattr(
        result_io, "_write_immutable", lambda _payload: str(tmp_path / "result.json")
    )
    monkeypatch.setattr(result_io, "_publish_exactly_once", lambda manifest, _path: manifest)

    # step 1 opens the coalescing window; steps 2 and 3 coalesce behind the cadence.
    for step in (1, 2, 3):
        progress_io.publish_progress("rl_step", step=step, reward=0.1 * step)
    assert progress_io._PROGRESS_COALESCED is not None, "step 3 must still be pending"
    assert uploaded == [], "the coalescing window must hold every step back"

    published = result_io.publish_result(
        outcome="succeeded",
        failure_class=None,
        started_at=100.0,
        training_entered=True,
        completed_steps=3,
        metrics={"step": 3},
        artifacts={"adapter": "published"},
    )

    assert uploaded, "the manifest was published while a completed step was still coalesced"
    assert uploaded[-1].completed_steps == published.completed_steps == 3


def test_cancelled_result_uses_latest_current_fence_progress(monkeypatch) -> None:
    _set_identity(monkeypatch)
    captured = []
    monkeypatch.setattr(
        result_io,
        "_latest_local_progress",
        lambda: {
            "training_entered": True,
            "completed_steps": 4,
            "metrics": {"loss": 0.25},
            "checkpoint": {"step": 4},
        },
    )
    monkeypatch.setattr(
        result_io,
        "publish_result",
        lambda **kwargs: captured.append(kwargs) or _manifest(),
    )

    result_io.publish_cancelled_result(started_at=100.0)

    assert captured == [
        {
            "outcome": "cancelled",
            "failure_class": None,
            "started_at": 100.0,
            "training_entered": True,
            "completed_steps": 4,
            "metrics": {"loss": 0.25},
            # progress records a save flat as {"step": N}; the manifest carries one shape, so the
            # save moves under "saved" and "failure" stays explicit rather than absent.
            "checkpoint": {"failure": None, "saved": {"step": 4}},
            "artifacts": {"console": "console_rl.txt"},
            "diagnostics": {"error": "worker attempt cancelled"},
        }
    ]


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        (None, {"failure": None}),
        ({}, {"failure": None}),
        ("not-a-dict", {"failure": None}),
        # the exact save shapes, from checkpoint_uploaded and checkpoint_deployable.
        ({"step": 7}, {"failure": None, "saved": {"step": 7}}),
        (
            {"step": 7, "subfolder": "step-7"},
            {"failure": None, "saved": {"step": 7, "subfolder": "step-7"}},
        ),
        # the exact failure shape hf.py emits. it always carries step, so a discriminator keyed
        # on step alone reports every real latched failure as failure: None.
        (
            {"step": 12, "operation": "after", "error": "upload failed: 500"},
            {"failure": {"step": 12, "operation": "after", "error": "upload failed: 500"}},
        ),
        ({"error": "upload failed"}, {"failure": {"error": "upload failed"}}),
    ],
)
def test_supervisor_checkpoint_carries_one_manifest_shape(observed, expected) -> None:
    """a latched checkpoint failure must stay distinguishable from a successful save.

    workers publish ``{"failure": ...}`` while progress writes flat shapes, so passing progress
    straight through would put two schemas under one key and make an upload failure read as a
    save to any reader that later consumes it.

    every case here is a shape some producer actually emits. an earlier revision of this test
    invented ``{"error": ..., "step": None}`` and asserted it as a save, which encoded the bug
    as the contract and let the suite go green over a real misclassification.
    """
    assert result_io._manifest_checkpoint(observed) == expected


def test_real_checkpoint_failure_shape_is_never_read_as_a_save() -> None:
    """pin the discriminator against the producer rather than against a hand-written shape.

    ``hf.py``'s ``checkpoint_upload_failed`` is the only failure emitter and always includes
    ``step``, so this asserts the exact keys it publishes stay on the failure side.
    """
    emitted = {"step": 3, "operation": "before", "error": "boom"}
    carried = result_io._manifest_checkpoint(emitted)

    assert carried["failure"] == emitted
    assert "saved" not in carried


def test_bootstrap_env_requires_the_source_snapshot(monkeypatch, tmp_path) -> None:
    """a worker launched without the snapshot can finish training and then publish nothing.

    every terminal manifest attests its exact source, so the env build must refuse the payload
    rather than let the attempt run to completion and fail at its only terminal authority.
    """
    from flash.providers._lifecycle.bootstrapping import bootstrap

    payload = {
        "phase": "sft",
        "seed": 1,
        "attempt": 0,
        "fence": 1,
        "run_id": "run-1",
        "env": {},
        "job_spec_json": json.dumps({"run_id": "run-1"}),
        "flash_arm": "sft",
        "source_snapshot": None,
    }
    with pytest.raises(RuntimeError, match="missing source_snapshot"):
        bootstrap.build_worker_env(payload)

    payload["source_snapshot"] = ATTESTATION
    env = bootstrap.build_worker_env(payload)
    assert json.loads(env["FLASH_SOURCE_SNAPSHOT_JSON"]) == ATTESTATION


def test_worker_surfaces_required_result_publication_failure(monkeypatch) -> None:
    from flash.engine.worker.entry import worker

    class ResultPublicationError(RuntimeError):
        pass

    original = ValueError("training failed")
    publication = ResultPublicationError("result transport failed")
    monkeypatch.setattr(worker, "_run_worker_mode", lambda: (_ for _ in ()).throw(original))
    monkeypatch.setattr(worker.hf_io, "hf_upload_file", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(worker.backend_common, "collect_ray_failure_logs", lambda **_kwargs: "")
    monkeypatch.setattr(
        worker.result_io,
        "publish_result",
        lambda **_kwargs: (_ for _ in ()).throw(publication),
    )
    monkeypatch.setattr(worker.progress_io, "publish_progress", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(worker.worker_perf, "gpu_diagnostics", dict)
    monkeypatch.setattr(worker.time, "sleep", lambda _seconds: None)

    with pytest.raises(ResultPublicationError) as exc_info:
        worker.main()

    assert exc_info.value is publication
    assert exc_info.value.__cause__ is original


def test_exactly_once_publish_rejects_conflicting_existing_result(monkeypatch, tmp_path) -> None:
    _set_identity(monkeypatch)
    proposed = _manifest()
    existing = _manifest(outcome="failed", failure_class="worker", metrics={})

    class Api:
        def repo_info(self, **_kwargs):
            return SimpleNamespace(sha="c" * 40)

        def list_repo_files(self, **_kwargs):
            return [result_path(existing)]

        def create_commit(self, **_kwargs):
            raise AssertionError("conflicting result must be rejected before upload")

    monkeypatch.setattr(result_io.hf_io, "hf_api", Api)
    monkeypatch.setattr(result_io, "_download_result", lambda _path, *, revision: existing)
    local = tmp_path / "result.json"
    local.write_text("{}")

    with pytest.raises(RuntimeError, match="conflicting terminal result"):
        result_io._publish_exactly_once(proposed, str(local))
