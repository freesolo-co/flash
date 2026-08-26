from __future__ import annotations

import inspect
import threading
import time

from flash.cli.commands.ops.log_follow import _log_follow_metric_rows
from flash.engine.worker.io import progress as progress_io
from flash.engine.worker.io import result as result_io
from flash.providers.artifacts import attempts as attempt_artifacts
from flash.runner.lifecycle.protocol import canonical_bytes, progress_path


def _reset(monkeypatch) -> None:
    monkeypatch.setattr(progress_io.worker_state, "RUN_ID", "run-1")
    monkeypatch.setattr(progress_io.worker_state, "PHASE", "rl")
    monkeypatch.setattr(progress_io.worker_state, "ATTEMPT", 2)
    monkeypatch.setattr(progress_io.worker_state, "FENCE", 9)
    with progress_io._PROGRESS_CONDITION:
        assert progress_io._PROGRESS_PUBLISHER is None
        progress_io._PROGRESS_SEQUENCE = 0
        progress_io._PROGRESS_PREVIOUS_DIGEST = None
        progress_io._PROGRESS_TRAINING_ENTERED = False
        progress_io._PROGRESS_COMPLETED_STEPS = 0
        progress_io._PROGRESS_PENDING_CHECKPOINT_FAILURE = None
        progress_io._PROGRESS_PENDING_UPLOAD = None
        progress_io._PROGRESS_LATEST_OBSERVATION = None
        progress_io._PROGRESS_ACTIVE = False
        progress_io._PROGRESS_FLUSH_REQUIRED = False
        progress_io._PROGRESS_GENERATION = 0
        progress_io._PROGRESS_ERROR = None


def test_progress_api_exposes_only_initial_and_observed_fields() -> None:
    signature = inspect.signature(progress_io.publish_progress)
    assert list(signature.parameters) == ["stage", "initial", "fields"]
    assert signature.parameters["initial"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["fields"].kind is inspect.Parameter.VAR_KEYWORD

    phase_signature = inspect.signature(progress_io.observe_phase)
    assert list(phase_signature.parameters) == ["stage", "progress", "fields", "progress_step"]


def test_initial_required_progress_waits_for_upload(monkeypatch) -> None:
    _reset(monkeypatch)
    upload_started = threading.Event()
    release_upload = threading.Event()

    def upload(_record, *, required, **_kwargs):
        assert required is True
        upload_started.set()
        assert release_upload.wait(timeout=2)
        return True

    monkeypatch.setattr(progress_io, "_upload_record", upload)
    thread = threading.Thread(
        target=progress_io.publish_progress,
        args=("rl_step",),
        kwargs={"initial": True, "step": 0},
    )
    thread.start()

    assert upload_started.wait(timeout=1)
    assert thread.is_alive()
    release_upload.set()
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_failed_upload_retries_the_identical_record_and_path(monkeypatch) -> None:
    _reset(monkeypatch)
    attempts = []
    first_attempt = threading.Event()
    outcomes = iter([False, True, True])

    def upload(record, *, required, local_path=None, remote_path=None):
        attempts.append(
            (
                canonical_bytes(record.to_dict()),
                progress_path(record),
                local_path,
                remote_path,
                required,
            )
        )
        first_attempt.set()
        return next(outcomes)

    monkeypatch.setattr(progress_io, "_upload_record", upload)

    progress_io.publish_progress("boot", gpu={"used_mb": 1})
    assert first_attempt.wait(timeout=1)
    progress_io.publish_progress("rl_step", step=1, reward=0.5)
    progress_io.flush_progress()

    assert attempts[0][:4] == attempts[1][:4]
    assert attempts[2][1] != attempts[1][1]
    assert progress_io._PROGRESS_SEQUENCE == 2
    assert progress_io._PROGRESS_PENDING_UPLOAD is None


def test_blocked_upload_does_not_block_steps_and_coalesces_latest(monkeypatch) -> None:
    _reset(monkeypatch)
    attempted = []
    upload_started = threading.Event()
    release_upload = threading.Event()

    def upload(record, **_kwargs):
        attempted.append(record)
        upload_started.set()
        assert release_upload.wait(timeout=2)
        return True

    monkeypatch.setattr(progress_io, "_upload_record", upload)
    progress_io.publish_progress("boot", gpu={"used_mb": 1})
    assert upload_started.wait(timeout=1)

    started = time.monotonic()
    for step in range(1, 101):
        progress_io.publish_progress("rl_step", step=step, reward=step / 100)
    assert time.monotonic() - started < 0.5

    release_upload.set()
    progress_io.flush_progress()

    assert [record.sequence for record in attempted] == [1, 2]
    assert attempted[1].completed_steps == 100
    assert attempted[1].metrics["reward"] == 1.0
    assert attempted[1].previous_digest == progress_io.digest_record(attempted[0].to_dict())


def test_observation_during_failed_upload_retries_without_another_signal(monkeypatch) -> None:
    _reset(monkeypatch)
    attempted = []
    first_upload_started = threading.Event()
    release_first_upload = threading.Event()
    all_published = threading.Event()

    def upload(record, **_kwargs):
        attempted.append(record)
        if len(attempted) == 1:
            first_upload_started.set()
            assert release_first_upload.wait(timeout=2)
            return False
        if len(attempted) == 3:
            all_published.set()
        return True

    monkeypatch.setattr(progress_io, "_upload_record", upload)
    progress_io.publish_progress("boot", gpu={"used_mb": 1})
    assert first_upload_started.wait(timeout=1)
    progress_io.publish_progress("rl_step", step=1, reward=0.5)
    release_first_upload.set()

    assert all_published.wait(timeout=2)
    deadline = time.monotonic() + 1
    while progress_io._PROGRESS_PUBLISHER is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert [record.sequence for record in attempted] == [1, 1, 2]
    assert canonical_bytes(attempted[0].to_dict()) == canonical_bytes(attempted[1].to_dict())
    assert progress_path(attempted[0]) == progress_path(attempted[1])
    assert attempted[2].previous_digest == progress_io.digest_record(attempted[1].to_dict())
    assert attempted[2].completed_steps == 1
    assert progress_io._PROGRESS_PUBLISHER is None


def test_commit_landed_response_failed_retries_without_breaking_reader(monkeypatch) -> None:
    _reset(monkeypatch)
    uploaded = {}
    upload_calls = []
    first_attempt = threading.Event()

    def upload_absolute(local_path, remote_path, *, required):
        with open(local_path, "rb") as handle:
            payload = handle.read()
        upload_calls.append((remote_path, payload, required))
        uploaded[remote_path] = payload
        first_attempt.set()
        return len(upload_calls) > 1

    monkeypatch.setattr(progress_io.hf_io, "hf_upload_absolute", upload_absolute)

    progress_io.publish_progress("boot", gpu={"used_mb": 1})
    assert first_attempt.wait(timeout=1)
    progress_io.publish_progress("rl_step", step=1, reward=0.5)
    progress_io.flush_progress()
    assert upload_calls[0][:2] == upload_calls[1][:2]
    assert upload_calls[2][0] != upload_calls[1][0]

    monkeypatch.setattr(
        attempt_artifacts,
        "_download_bytes",
        lambda _repo, path, *, revision: uploaded[path],
    )
    decoded = attempt_artifacts._decode_progress(
        "org/repo",
        list(uploaded),
        prefix="rl/run-1/attempts/2-9",
        revision="c" * 40,
        observed_at=200.0,
    )

    assert decoded is not None
    assert decoded["sequence"] == 2
    assert decoded["phase"] == "rl_step"
    assert decoded["metrics"]["reward"] == 0.5


def test_grpo_producer_metrics_reach_current_progress_and_cli(monkeypatch) -> None:
    from flash.engine.worker.train.entry import rl_train_runner

    _reset(monkeypatch)
    records = []
    monkeypatch.setattr(
        progress_io,
        "_upload_record",
        lambda record, **_kwargs: records.append(record) or True,
    )
    monkeypatch.setattr(rl_train_runner, "gpu_diagnostics", lambda **_kwargs: {})
    state = rl_train_runner._StepMetricState()

    rl_train_runner._ingest_step_metrics(
        "step:2 - critic/rewards/mean:0.75 - actor/grad_norm:1.25",
        {"max_completion": 256},
        state,
        dict,
    )
    progress_io.flush_progress()

    record = records[-1]
    assert record.metrics == {
        "grad_norm": 1.25,
        "max_completion_tokens": 256,
        "reward": 0.75,
    }
    assert record.diagnostics["metrics_last"][-1]["reward"] == 0.75
    status = {
        "attempt": {"attempt_id": 2, "fence": 9},
        "progress": record.to_dict(),
    }
    assert _log_follow_metric_rows(status, set()) == [
        "progress_seq=1 step=2 reward=0.75 grad_norm=1.25 max_comp_tokens=256"
    ]


def test_explicit_current_metrics_override_grpo_history() -> None:
    metrics, *_sections = progress_io._progress_sections(
        {
            "reward": 0.9,
            "metrics_last": [{"step": 2, "reward": 0.75, "grad_norm": 1.25}],
        }
    )

    assert metrics == {"reward": 0.9, "grad_norm": 1.25}


def test_opd_producer_discarded_rollouts_reaches_progress_and_cli(monkeypatch) -> None:
    from flash.engine.worker.train.entry import opd_train_runner
    from flash.engine.worker.train.opd.orchestration.progress import _OpdProgressState

    _reset(monkeypatch)
    records = []
    monkeypatch.setattr(
        progress_io,
        "_upload_record",
        lambda record, **_kwargs: records.append(record) or True,
    )
    monkeypatch.setattr(
        opd_train_runner._backend,
        "verify_applied_shim_markers",
        lambda *_args: None,
    )

    class Watcher:
        @staticmethod
        def raise_if_failed() -> None:
            return None

    class Bridge:
        parent_work = None

        @staticmethod
        def accounting_snapshot() -> dict:
            return {
                "aligned_sequences": 4,
                "coverage_sum": 4.0,
                "truncated_rollouts": 3,
                "samples_seen": 4,
                "no_signal_skipped_steps": 0,
            }

    callbacks = opd_train_runner._build_child_callbacks(
        Watcher(),
        _OpdProgressState(),
        Bridge(),
        0,
        "unused",
        ("lora-rollout-guard",),
    )
    callbacks.on_line("step:1 - actor/distillation/loss:0.4")
    callbacks.on_step(1)
    progress_io.flush_progress()

    record = records[-1]
    assert record.metrics["discarded_rollouts"] == 3
    status = {
        "attempt": {"attempt_id": 2, "fence": 9},
        "progress": record.to_dict(),
    }
    assert _log_follow_metric_rows(status, set()) == [
        "progress_seq=1 step=1 trunc=0.75 discarded=3"
    ]


def test_checkpoint_failure_is_sticky_until_a_successful_checkpoint(monkeypatch) -> None:
    _reset(monkeypatch)
    records = []
    monkeypatch.setattr(
        progress_io,
        "_upload_record",
        lambda record, **_kwargs: records.append(record) or True,
    )
    failure = {"step": 50, "operation": "resume", "error": "quota denied"}

    progress_io.publish_progress("checkpoint_upload_failed", step=50, checkpoint_failure=failure)
    progress_io.publish_progress("sft_step", step=60)
    progress_io.flush_progress()
    assert records[-1].checkpoint == failure

    progress_io.publish_progress("checkpoint_uploaded", step=75)
    progress_io.publish_progress("sft_step", step=80)
    progress_io.flush_progress()
    assert records[-1].checkpoint == {}


def test_terminal_result_waits_for_final_progress(monkeypatch) -> None:
    _reset(monkeypatch)
    upload_started = threading.Event()
    release_upload = threading.Event()
    result_published = threading.Event()

    def upload(_record, **_kwargs):
        upload_started.set()
        assert release_upload.wait(timeout=2)
        return True

    monkeypatch.setattr(progress_io, "_upload_record", upload)
    monkeypatch.setattr(result_io, "_source_attestation", lambda: {"source": "test"})
    monkeypatch.setattr(result_io, "_write_immutable", lambda _payload: "/tmp/result.json")
    monkeypatch.setattr(
        result_io,
        "_publish_exactly_once",
        lambda manifest, _path: result_published.set() or manifest,
    )
    progress_io.publish_progress("sft_step", step=4, loss=0.25)
    assert upload_started.wait(timeout=1)

    thread = threading.Thread(
        target=result_io.publish_result,
        kwargs={
            "outcome": "failed",
            "failure_class": "worker",
            "started_at": 1.0,
            "training_entered": True,
            "completed_steps": 4,
        },
    )
    thread.start()

    time.sleep(0.05)
    assert thread.is_alive()
    assert not result_published.is_set()
    release_upload.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert result_published.is_set()


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
