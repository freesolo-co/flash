"""Tests for run-management helpers (runs/cost/cancel) — no GPU/network."""

from __future__ import annotations

import io
import json
import tempfile
from dataclasses import replace
from types import SimpleNamespace

import pytest

import flash.runner.accounting.artifacts as runner_artifacts
import flash.runner.accounting.costs as runner_costs
import flash.runner.accounting.reconciliation as runner_reconciliation
import flash.runner.lifecycle.attempts as runner_attempts
import flash.runner.lifecycle.deadlines as runner_deadlines
import flash.runner.lifecycle.preparation as runner_preparation
import flash.runner.lifecycle.reporting as runner_reporting
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit
import flash.runner.supervise.attach as runner_attach
import flash.runner.supervise.deploy as runner_deploy
import flash.runner.supervise.errors as runner_errors
import flash.runner.supervise.lifecycle as runner_lifecycle
import flash.runner.supervise.recovery as runner_recovery
from flash.providers._lifecycle.net import worker as provider_worker
from flash.providers.core.base import PollResult
from tests._helpers.runner import provisioned_status as base_provisioned_status
from tests._helpers.source_snapshot import valid_source_snapshot

_RUNPOD_FINGERPRINT = "rpk-" + "0" * 64
_SOURCE_SNAPSHOT = valid_source_snapshot()
_RETIRED_MODELS = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-27B")


def _runpod_remote(
    endpoint_id="endpoint",
    job_id="job",
    attempt=0,
    fence=None,
    started_ts=1.0,
    **extra,
):
    remote = {
        "provider": "runpod",
        "endpoint_id": endpoint_id,
        "endpoint_name": f"{endpoint_id}-name",
        "key_fingerprint": _RUNPOD_FINGERPRINT,
        "attempt": attempt,
        "fence": attempt + 1 if fence is None else fence,
        "started_ts": started_ts,
        **extra,
    }
    if job_id is not None:
        remote["job_id"] = job_id
    return remote


def _lambda_remote(instance_id="instance", attempt=0, fence=None, started_ts=1.0, **extra):
    return {
        "provider": "lambda",
        "instance_id": instance_id,
        "instance_type": "gpu_1x_a100",
        "region": "us-east-1",
        "name": f"flash-{instance_id}",
        "gpu": "A100",
        "hourly_usd": 1.0,
        "attempt": attempt,
        "fence": attempt + 1 if fence is None else fence,
        "started_ts": started_ts,
        **extra,
    }


def _vast_remote(instance_id=7, attempt=0, fence=None, started_ts=1.0, **extra):
    return {
        "provider": "vast",
        "instance_id": instance_id,
        "offer_id": 101,
        "machine_id": 202,
        "label": f"flash-{instance_id}",
        "gpu": "RTX 4090",
        "hourly_usd": 0.5,
        "attempt": attempt,
        "fence": attempt + 1 if fence is None else fence,
        "started_ts": started_ts,
        **extra,
    }


def provisioned_status(spec, *, state="running", remote=None, attempt=None, **kwargs):
    from flash.runner.lifecycle.protocol import AttemptRecord

    if remote is not None and attempt is None:
        run_deadline = float(kwargs.get("created_at", 100.0)) + float(spec.gpu.max_wall_seconds)
        attempt = AttemptRecord(
            attempt_id=remote["attempt"],
            fence=remote["fence"],
            state="active",
            reserved_at=float(kwargs.get("created_at", 100.0)),
            grant_deadline_at=min(run_deadline, float(kwargs.get("created_at", 100.0)) + 30.0),
            work_deadline_at=run_deadline,
            run_deadline_at=run_deadline,
            result_deadline_at=run_deadline + 900.0,
            provider=remote["provider"],
            resource=remote,
        ).to_dict()
    return base_provisioned_status(
        spec,
        state=state,
        remote=remote,
        attempt=attempt,
        **kwargs,
    )


def _handleless_status(spec, *, state="provisioning", created_at=100.0):
    from flash.runner.lifecycle.protocol import AttemptRecord

    run_deadline = created_at + float(spec.gpu.max_wall_seconds)
    attempt = AttemptRecord(
        attempt_id=0,
        fence=1,
        state="active",
        reserved_at=created_at,
        grant_deadline_at=min(run_deadline, created_at + 30.0),
        work_deadline_at=run_deadline,
        run_deadline_at=run_deadline,
        result_deadline_at=run_deadline + 900.0,
    )
    status = base_provisioned_status(
        spec,
        state=state,
        remote=None,
        attempt=attempt.to_dict(),
        created_at=created_at,
    )
    status.source_snapshot = _SOURCE_SNAPSHOT
    return status


def _fenced_success_metrics(spec, *, attempt=0, fence=1, wall_seconds=5.0):
    from flash.snapshot.archive import TERMINAL_ATTESTATION_KEY, source_attestation

    return {
        "wall_seconds": wall_seconds,
        TERMINAL_ATTESTATION_KEY: source_attestation(
            _SOURCE_SNAPSHOT,
            run_id=spec.run_id,
            attempt=attempt,
            fence=fence,
        ),
    }


@pytest.mark.parametrize("retired_model", _RETIRED_MODELS)
def test_historical_removed_model_status_remains_listable(monkeypatch, tmp_path, retired_model):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path))
    spec = JobSpec(run_id="historical", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="done", spec=spec.to_dict())
    )
    path = runner_state.runs_file_path(spec.run_id, ".json")
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    stored["spec"]["model"] = retired_model
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stored, handle)

    assert runner_status.list_run_ids() == [spec.run_id]
    assert runner_status.list_runs()[0].spec["model"] == retired_model
    assert runner_status.get_status(spec.run_id).spec["model"] == retired_model


def test_background_run_redacts_private_exception_content(monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    from flash.core.spec import JobSpec

    spec = JobSpec(run_id="background-private", model="Qwen/Qwen3.5-9B", algorithm="sft")
    updates = []

    def fail_run(_spec):
        raise RuntimeError("private provider response")

    monkeypatch.setattr(runner_lifecycle, "_run_job", fail_run)
    monkeypatch.setattr(
        runner_status, "get_status", lambda _run_id: SimpleNamespace(state="running")
    )
    monkeypatch.setattr(
        runner_status,
        "_update",
        lambda run_id, state, **kwargs: updates.append((run_id, state, kwargs)),
    )

    with caplog.at_level(logging.WARNING):
        runner_lifecycle._run_job_background(spec)

    assert updates == [
        (
            spec.run_id,
            "failed",
            {"error": "RuntimeError: background run failed"},
        )
    ]
    assert "private provider response" not in caplog.text
    assert "RuntimeError: background run failed" in caplog.text


def test_list_and_cancel(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        # fixed constant; redirect to tmp via monkeypatch so it's restored after the test.
        monkeypatch.setattr(runner_state, "RUNS_DIR", tmp)
        from flash.core.spec import JobSpec, TrainSpec

        # two dry-run records
        for rid in ("a", "b"):
            runner_submit.submit_job(
                JobSpec(
                    run_id=rid,
                    model="Qwen/Qwen3.5-9B",
                    algorithm="grpo",
                    train=TrainSpec(max_examples=8),
                ),
                dry_run=True,
            )
        runs = {r.run_id for r in runner_status.list_runs()}
        assert {"a", "b"} <= runs

        # cancel a non-terminal run (force it to a running-ish state first; go through
        # _save_status, not _update, since _update now refuses to resurrect a terminal
        # state like the submitted dry_run).
        running = runner_status.get_status("a")
        running.state = "running"
        runner_state._save_status(running)
        status = runner_deploy.cancel_run("a")
        assert status.state == "cancelled"

        # cancelling a terminal run is a no-op
        same = runner_deploy.cancel_run("b")  # b is dry_run (terminal-ish)
        assert same.state in {"dry_run", "cancelled"}


def test_get_status_tolerates_stale_unknown_keys(monkeypatch):
    # A status JSON written by an OLDER control plane can carry a since-removed field (e.g.
    # `resume_seed_index` from the pre-#317 multi-seed era); `~/.flash/runs/*.json` is never GC'd,
    # so those files persist across an upgrade. get_status/list_runs must drop unknown keys rather
    # than 500 (a strict RunStatus(**d) would TypeError, and callers catch only FileNotFoundError).
    import json
    import os

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", tmp)
        stale = {
            "run_id": "old",
            "state": "done",
            "spec": {},
            "cost_usd": 2.0,
            "resume_seed_index": 3,  # removed field
            "totally_unknown_future_key": "x",  # forward-compat unknown field
        }
        os.makedirs(tmp, exist_ok=True)
        with open(runner_state.runs_file_path("old", ".json"), "w") as f:
            json.dump(stale, f)

        s = runner_status.get_status("old")
        assert s.run_id == "old"
        assert s.state == "done"
        assert s.cost_usd == 2.0
        assert not hasattr(s, "resume_seed_index")
        assert "old" in {r.run_id for r in runner_status.list_runs()}


def test_submit_job_persists_quote_and_completion_charges_it(monkeypatch, tmp_path):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.cost.currency import usd_amount
    from flash.cost.spec import estimate_for_spec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_artifacts, "_assign_resolved_env_sha", lambda spec: spec)
    monkeypatch.setattr(
        provider_worker, "publish_source_snapshot", lambda _repo=None: _SOURCE_SNAPSHOT
    )

    seen: dict[str, float] = {}

    def fake_run(spec, runtime_secrets=None):
        status = runner_status.get_status(spec.run_id)
        priced_spec = JobSpec.from_dict(status.spec)
        seen["estimate"] = float(status.estimated_cost_usd)
        seen["expected"] = usd_amount(estimate_for_spec(priced_spec).total_usd)
        runner_status._update(
            spec.run_id,
            "done",
            cost_usd=runner_costs._status_estimated_charge(status, priced_spec, fallback=0.01),
        )

    monkeypatch.setattr(runner_lifecycle, "_run_job", fake_run)

    status = runner_submit.submit_job(
        JobSpec(
            run_id="quoted",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=2),
            gpu=GpuSpec(type=""),
        )
    )

    assert seen["estimate"] == pytest.approx(seen["expected"])
    assert status.estimated_cost_usd == pytest.approx(seen["expected"])
    assert status.cost_usd == pytest.approx(seen["expected"])
    raw = runner_status._load_status_json(status.run_id)
    assert raw[runner_state._RUN_DEADLINE_AT_KEY] == pytest.approx(
        status.created_at + JobSpec.from_dict(status.spec).gpu.max_wall_seconds
    )
    assert raw[runner_state._NEXT_ATTEMPT_KEY] == 0


def test_missing_persisted_run_deadline_is_rejected(monkeypatch, tmp_path):
    import os

    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="missing-deadline",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=900),
    )
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            created_at=123.0,
        )
    )
    path = runner_state.runs_file_path(spec.run_id, ".json")
    raw = runner_status._load_status_json(spec.run_id)
    raw.pop(runner_state._RUN_DEADLINE_AT_KEY)
    with open(path, "w") as file:
        json.dump(raw, file)
    assert os.path.exists(path)

    with pytest.raises(RuntimeError, match="persisted run wall deadline is missing"):
        runner_deadlines._load_run_deadline_at(spec.run_id)


@pytest.mark.parametrize(
    "unsafe_now",
    [True, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), "1000"],
)
def test_remaining_run_wall_seconds_rejects_unsafe_current_clock(monkeypatch, tmp_path, unsafe_now):
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="unsafe-current-clock",
        model="Qwen/Qwen3.5-9B",
        gpu=GpuSpec(max_wall_seconds=900),
    )
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            created_at=123.0,
        ),
        _run_deadline_at=1023.0,
    )

    with pytest.raises(ValueError, match="current clock is invalid"):
        runner_deadlines._remaining_run_wall_seconds(spec.run_id, now=unsafe_now)


def test_persisted_run_deadline_must_match_canonical_value(monkeypatch, tmp_path):
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="mismatched-deadline",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=900),
    )
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            created_at=123.0,
        ),
        _run_deadline_at=1024.0,
    )

    with pytest.raises(RuntimeError, match="does not match canonical"):
        runner_deadlines._load_run_deadline_at(spec.run_id)


@pytest.mark.parametrize("deadline", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_persisted_run_deadline_rejects_nonpositive_or_nonfinite_values(
    monkeypatch, tmp_path, deadline
):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="invalid-deadline", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()),
        _run_deadline_at=deadline,
    )

    with pytest.raises(RuntimeError, match="run wall deadline is invalid"):
        runner_deadlines._load_run_deadline_at(spec.run_id)


@pytest.mark.parametrize("created_at", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_status_save_rejects_invalid_creation_time(monkeypatch, tmp_path, created_at):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="invalid-legacy-deadline", model="Qwen/Qwen3.5-4B", algorithm="sft")
    with pytest.raises(RuntimeError, match="run wall deadline is invalid"):
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                created_at=created_at,
            )
        )


def test_status_projection_sanitizer_bounds_lists():
    from flash.runner.lifecycle.protocol import bounded_json

    metrics = [{"step": step, "reward": step / 100} for step in range(100)]
    sanitized = bounded_json({"metrics": metrics, "other": list(range(100))})

    assert len(sanitized["metrics"]) == 64
    assert sanitized["metrics"][0]["step"] == 0
    assert sanitized["metrics"][-1]["step"] == 63
    assert sanitized["other"] == list(range(64))


def test_finished_at_frozen_at_terminal_survives_later_updated_at_bumps(monkeypatch):
    """finished_at remains fixed after later resource observations and terminal rewrites."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", tmp)
        from flash.core.spec import JobSpec, TrainSpec

        runner_submit.submit_job(
            JobSpec(
                run_id="fa",
                model="Qwen/Qwen3.5-9B",
                algorithm="grpo",
                train=TrainSpec(max_examples=8),
            ),
            dry_run=True,
        )
        s = runner_status.get_status("fa")
        s.state = "running"
        s.finished_at = None  # dry_run created via direct state set, never stamped finished_at
        runner_state._save_status(s)
        runner_attempts._reserve_attempt_record("fa")

        # first terminal transition stamps finished_at to the teardown time
        assert runner_status._update("fa", "done", cost_usd=1.0) is True
        done = runner_status.get_status("fa")
        assert done.finished_at is not None
        teardown = done.finished_at
        assert teardown == done.updated_at

        # a later resource observation may move updated_at but not finished_at
        attempt = runner_status._current_attempt(done)
        assert runner_status.record_resource(
            "fa",
            {"state": "terminated", "observed_at": 123.0},
            attempt_id=attempt.attempt_id,
            fence=attempt.fence,
        )
        bumped = runner_status.get_status("fa")
        assert bumped.updated_at >= done.updated_at
        assert bumped.finished_at == teardown

        # a same-state terminal rewrite keeps the original too
        runner_status._update("fa", "done", cost_usd=2.0)
        assert runner_status.get_status("fa").finished_at == teardown


def test_resource_projection_rejects_older_observations_in_both_delivery_orders(
    monkeypatch, tmp_path
):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    for run_id, observations in (
        (
            "resource-newer-first",
            (("terminal", 20.0, True), ("running", 10.0, False)),
        ),
        (
            "resource-older-first",
            (("running", 10.0, True), ("terminal", 20.0, True)),
        ),
    ):
        spec = JobSpec(run_id=run_id, model="Qwen/Qwen3.5-9B", algorithm="sft")
        remote = _runpod_remote(f"endpoint-{run_id}", f"job-{run_id}", attempt=0)
        status = provisioned_status(spec, state="running", created_at=100.0, remote=remote)
        runner_state._save_status(status, _run_deadline_at=86500.0, _next_attempt=1)
        attempt = runner_status._current_attempt(status)
        identity = runner_reconciliation._remote_resource_identity(remote)

        for resource_state, observed_at, accepted in observations:
            assert (
                runner_status.record_resource(
                    run_id,
                    {
                        "attempt_id": attempt.attempt_id,
                        "fence": attempt.fence,
                        "provider": "runpod",
                        "state": resource_state,
                        "observed_at": observed_at,
                    },
                    attempt_id=attempt.attempt_id,
                    fence=attempt.fence,
                    resource_identity=identity,
                )
                is accepted
            )

        projected = runner_status.get_status(run_id).resource
        assert projected["state"] == "terminal"
        assert projected["observed_at"] == 20.0


def test_public_status_redacts_nested_attempt_receipt_revisions_without_mutation():
    from flash.core.spec import JobSpec

    spec = JobSpec(run_id="public-receipts", model="Qwen/Qwen3.5-9B", algorithm="sft")
    remote = _runpod_remote("endpoint-receipts", "job-receipts", attempt=2, fence=9)
    status = provisioned_status(spec, remote=remote, created_at=100.0)
    result_receipt = {
        "path": "attempt/result.json",
        "digest": "a" * 64,
        "revision": "result-revision",
    }
    progress_receipt = {
        "path": "attempt/progress.json",
        "digest": "b" * 64,
        "revision": "progress-revision",
    }
    status.attempt["result_receipt"] = result_receipt
    status.attempt["progress_receipt"] = progress_receipt

    public = status.to_dict()

    assert public["attempt"]["result_receipt"] == {
        "path": "attempt/result.json",
        "digest": "a" * 64,
    }
    assert public["attempt"]["progress_receipt"] == {
        "path": "attempt/progress.json",
        "digest": "b" * 64,
    }
    assert status.attempt["result_receipt"] == result_receipt
    assert status.attempt["progress_receipt"] == progress_receipt


def test_result_projection_updates_matching_attempt_receipt_atomically(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="result-receipt", model="Qwen/Qwen3.5-9B", algorithm="sft")
    remote = _runpod_remote("endpoint-result", "job-result", attempt=2, fence=9)
    status = provisioned_status(spec, remote=remote, created_at=100.0)
    runner_state._save_status(status)
    receipt = {"path": "attempt/result.json", "digest": "a" * 64, "revision": "rev"}
    result = {
        "attempt_id": 2,
        "fence": 9,
        "outcome": "failed",
        "receipt": receipt,
    }

    assert runner_status.record_result(spec.run_id, result, attempt_id=2, fence=9)

    persisted = runner_status.get_status(spec.run_id)
    attempt = runner_status._current_attempt(persisted)
    assert persisted.result == result
    assert attempt.state == "result_pending"
    assert attempt.result_receipt == receipt


@pytest.mark.parametrize(("attempt_id", "fence"), [(1, 9), (2, 8)])
def test_result_projection_rejects_stale_attempt_or_fence(monkeypatch, tmp_path, attempt_id, fence):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id=f"stale-result-{attempt_id}-{fence}", model="Qwen/Qwen3.5-9B")
    remote = _runpod_remote("endpoint-result", "job-result", attempt=2, fence=9)
    status = provisioned_status(spec, remote=remote, created_at=100.0)
    runner_state._save_status(status)
    before = runner_status.get_status(spec.run_id)
    result = {
        "attempt_id": attempt_id,
        "fence": fence,
        "receipt": {
            "path": "attempt/result.json",
            "revision": "rev",
            "digest": "b" * 64,
        },
    }

    assert not runner_status.record_result(
        spec.run_id,
        result,
        attempt_id=attempt_id,
        fence=fence,
    )

    persisted = runner_status.get_status(spec.run_id)
    assert persisted.result is None
    assert persisted.attempt == before.attempt


def test_cancelled_result_projection_preserves_terminal_attempt_state(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="cancelled-result", model="Qwen/Qwen3.5-9B", algorithm="sft")
    remote = _runpod_remote("endpoint-cancelled", "job-cancelled", attempt=2, fence=9)
    status = provisioned_status(spec, remote=remote, created_at=100.0)
    status.state = "cancelled"
    runner_state._save_status(status)
    before_attempt = runner_status._current_attempt(status)
    result = {
        "attempt_id": 2,
        "fence": 9,
        "outcome": "cancelled",
        "completed_steps": 7,
        "receipt": {
            "path": "attempt/result.json",
            "revision": "rev",
            "digest": "c" * 64,
        },
    }

    assert runner_status.record_result(spec.run_id, result, attempt_id=2, fence=9)

    persisted = runner_status.get_status(spec.run_id)
    assert persisted.state == "cancelled"
    assert persisted.result == result
    assert runner_status._current_attempt(persisted) == before_attempt


@pytest.mark.parametrize(
    "result",
    [
        {
            "attempt_id": 2,
            "fence": 9,
            "outcome": "failed",
            "receipt": {
                "path": "attempt/result.json",
                "revision": "rev",
                "digest": "d" * 64,
            },
        },
        {
            "attempt_id": 2,
            "fence": 9,
            "outcome": "cancelled",
            "receipt": {"path": "attempt/result.json", "digest": "e" * 64},
        },
    ],
)
def test_cancelled_result_projection_rejects_other_outcomes_and_invalid_receipts(
    monkeypatch, tmp_path, result
):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="cancelled-result-rejected", model="Qwen/Qwen3.5-9B")
    remote = _runpod_remote("endpoint-cancelled", "job-cancelled", attempt=2, fence=9)
    status = provisioned_status(spec, remote=remote, created_at=100.0)
    status.state = "cancelled"
    runner_state._save_status(status)

    assert not runner_status.record_result(spec.run_id, result, attempt_id=2, fence=9)

    persisted = runner_status.get_status(spec.run_id)
    assert persisted.state == "cancelled"
    assert persisted.result is None


@pytest.mark.parametrize(("attempt_id", "fence"), [(1, 9), (2, 8)])
def test_cancelled_result_projection_rejects_stale_identity(
    monkeypatch, tmp_path, attempt_id, fence
):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="cancelled-result-stale", model="Qwen/Qwen3.5-9B")
    remote = _runpod_remote("endpoint-cancelled", "job-cancelled", attempt=2, fence=9)
    status = provisioned_status(spec, remote=remote, created_at=100.0)
    status.state = "cancelled"
    runner_state._save_status(status)
    result = {
        "attempt_id": attempt_id,
        "fence": fence,
        "outcome": "cancelled",
        "receipt": {
            "path": "attempt/result.json",
            "revision": "rev",
            "digest": "f" * 64,
        },
    }

    assert not runner_status.record_result(
        spec.run_id,
        result,
        attempt_id=attempt_id,
        fence=fence,
    )

    persisted = runner_status.get_status(spec.run_id)
    assert persisted.state == "cancelled"
    assert persisted.result is None


def test_persist_metrics_keeps_stamped_zero_vast(monkeypatch):
    import json
    import os

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RESULTS_DIR", tmp)
        monkeypatch.setattr(runner_costs, "_gpu_rate", lambda gpu, provider="": 3600.0)
        from flash.core.spec import JobSpec

        spec = JobSpec(run_id="r0", model="Qwen/Qwen3.5-9B", algorithm="grpo")
        # A zero placeholder is not a settled provider cost; use the wall-pricing fallback.
        metrics = {
            "cost_usd": 0.0,
            "wall_seconds": 1.0,
        }
        out = runner_status._persist_metrics(spec, metrics)
        assert out == 1.0
        with open(os.path.join(runner_state.artifacts_dir(spec), "metrics.json")) as f:
            on_disk = json.load(f)
        assert on_disk["cost_usd"] == 1.0
        # No allocated_provider stamped -> say so, rather than attributing the cost to RunPod.
        assert on_disk["notes"]["provider"] == "unknown"


def test_persist_metrics_attributes_the_provider_that_billed_the_run(monkeypatch):
    """The note records the substrate that ran the job, not a hardcoded default.

    A plane with no RunPod key still prices its runs, and a Lambda/Vast run is not filed under
    RunPod's rate table.
    """
    import json
    import os

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RESULTS_DIR", tmp)
        seen = {}

        def _rate(gpu, provider=""):
            seen["gpu"], seen["provider"] = gpu, provider
            return 3600.0

        monkeypatch.setattr(runner_costs, "_gpu_rate", _rate)
        from flash.core.spec import JobSpec

        spec = JobSpec(run_id="r-vast", model="Qwen/Qwen3.5-9B", algorithm="grpo")
        runner_status._persist_metrics(
            spec,
            {"wall_seconds": 1.0, "allocated_gpu": "RTX 5090", "allocated_provider": "vast"},
        )
        assert seen == {"gpu": "RTX 5090", "provider": "vast"}
        with open(os.path.join(runner_state.artifacts_dir(spec), "metrics.json")) as f:
            on_disk = json.load(f)
        assert on_disk["notes"]["provider"] == "vast"
        assert on_disk["notes"]["gpu"] == "RTX 5090"
        assert on_disk["notes"]["gpu_rate_usd_hr"] == 3600.0


def test_persist_metrics_falls_back_when_cost_absent(monkeypatch):
    import json
    import os

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RESULTS_DIR", tmp)
        monkeypatch.setattr(runner_costs, "_gpu_rate", lambda gpu, provider="": 3600.0)
        from flash.core.spec import JobSpec

        spec = JobSpec(run_id="r1", model="Qwen/Qwen3.5-9B", algorithm="grpo")
        # No cost_usd stamped: fall back to wall * rate.
        out = runner_status._persist_metrics(
            spec, {"wall_seconds": 1.0, "allocated_gpu": "RTX 5090"}
        )
        assert out == 1.0  # 1s / 3600 * 3600/hr
        with open(os.path.join(runner_state.artifacts_dir(spec), "metrics.json")) as f:
            on_disk = json.load(f)
        assert on_disk["notes"]["provider"] == "unknown"


def test_persist_metrics_bills_training_wall_not_setup(monkeypatch):
    import json
    import os

    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RESULTS_DIR", tmp)
        monkeypatch.setattr(runner_costs, "_gpu_rate", lambda gpu, provider="": 3600.0)
        from flash.core.spec import JobSpec

        spec = JobSpec(run_id="r-train-only", model="Qwen/Qwen3.5-9B", algorithm="sft")
        metrics = {
            "wall_seconds": 10.0,  # worker training loop only
            "setup_seconds": 590.0,  # reported for observability, not customer cost
            "train_tokens": 190_679,
            "allocated_gpu": "RTX 5090",
        }
        out = runner_status._persist_metrics(spec, metrics)
        assert out == pytest.approx(10.0)  # 10s / 3600 * $3600/hr
        with open(os.path.join(runner_state.artifacts_dir(spec), "metrics.json")) as f:
            on_disk = json.load(f)
        assert on_disk["cost_usd"] == pytest.approx(10.0)
        assert on_disk["setup_seconds"] == pytest.approx(590.0)


def test_run_training_charges_persisted_submit_estimate(monkeypatch, tmp_path):
    import io

    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.runner.supervise import lifecycle

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(
        run_id="quote",
        model="Qwen/Qwen3.5-9B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=2),
        gpu=GpuSpec(type=""),
    )
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="queued",
            spec=spec.to_dict(),
            estimated_cost_usd=7.77,
        )
    )
    monkeypatch.setattr(
        runner_lifecycle,
        "_submit_seed_supervised",
        lambda *a, **k: {"wall_seconds": 1.0, "cost_usd": 0.01},
    )
    monkeypatch.setattr(
        runner_costs,
        "charge_usd_for_spec",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must use submit quote")),
    )

    monkeypatch.setattr(
        runner_status,
        "validate_terminal_source_metrics",
        lambda _status, metrics, expected_attempt=None: (metrics, expected_attempt),
    )
    lifecycle._run_training(spec, io.StringIO(), prior_cost=0.0, source_snapshot=_SOURCE_SNAPSHOT)

    st = runner_status.get_status(spec.run_id)
    assert st.state == "done"
    assert st.cost_usd == pytest.approx(7.77)


def test_supervised_attempt_identities_start_at_zero_and_increment_without_expanding_budget(
    monkeypatch, tmp_path
):
    import io

    import flash.providers.core.allocator as allocator
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core import registry as providers
    from flash.providers.core.base import Allocation, Candidate, PollResult
    from flash.runner.supervise import lifecycle
    from tests._helpers.profile import attach_sft_profile, stub_revision_geometry

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    # the attached profile pins a model revision, which makes the post-allocation quote refresh
    # resolve revision-specific geometry from the hub. read the catalog's numbers instead.
    stub_revision_geometry(monkeypatch)
    spec = attach_sft_profile(
        JobSpec(
            run_id="attempt-sequence",
            model="Qwen/Qwen3.5-9B",
            algorithm="sft",
            train=TrainSpec(max_examples=1),
            gpu=GpuSpec(type="", max_retries=1),
        )
    )
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            source_snapshot=_SOURCE_SNAPSHOT,
        ),
        _next_attempt=0,
    )
    candidate = Candidate("runpod", "RTX 4090", 0.69, 24)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *args, **kwargs: Allocation(
            provider="runpod",
            gpu="RTX 4090",
            hourly_usd=0.69,
            min_vram_gb=24,
            candidates=(candidate,),
        ),
    )
    monkeypatch.setattr(lifecycle.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(lifecycle, "_attempt_result", lambda *_args, **_kwargs: None)

    class FakeProvider:
        supports_weight_cache = False

        def __init__(self):
            self.attempts = []

        def submit_run(self, _spec, _seed, *, on_handle, attempt, **_kwargs):
            self.attempts.append(attempt)
            on_handle(
                _runpod_remote(
                    endpoint_id=f"endpoint-{attempt}",
                    job_id=f"job-{attempt}",
                    attempt=attempt,
                    fence=_kwargs["fence"],
                    started_ts=float(attempt + 1),
                )
            )
            if attempt == 0:
                return PollResult(False, failure="poll_error", detail="transient")
            return PollResult(True, metrics={"wall_seconds": 1.0})

        def cancel(self, _handle):
            return None

        def destroy(self, _handle):
            return None

    provider = FakeProvider()
    monkeypatch.setattr(providers, "get_provider", lambda _name: provider)

    metrics = lifecycle._submit_seed_supervised(
        spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
    )

    assert metrics["wall_seconds"] == 1.0
    assert provider.attempts == [0, 1]
    status = runner_status.get_status(spec.run_id)
    assert status.remote is None
    assert status.cleanup_confirmed_remote["attempt"] == 1
    assert status.realized_cost_remote["attempt"] == 1


def test_attempt_is_consumed_when_provider_fails_before_handle_persistence(monkeypatch, tmp_path):
    import io

    import flash.providers.core.allocator as allocator
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core import registry as providers
    from flash.providers.core.base import Allocation, Candidate, PollResult
    from flash.runner.supervise import lifecycle
    from tests._helpers.profile import attach_sft_profile, stub_revision_geometry

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    # the attached profile pins a model revision, which makes the post-allocation quote refresh
    # resolve revision-specific geometry from the hub. read the catalog's numbers instead.
    stub_revision_geometry(monkeypatch)
    spec = attach_sft_profile(
        JobSpec(
            run_id="pre-handle-attempt",
            model="Qwen/Qwen3.5-9B",
            algorithm="sft",
            train=TrainSpec(max_examples=1),
            gpu=GpuSpec(type="", max_retries=1),
        )
    )
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            source_snapshot=_SOURCE_SNAPSHOT,
        ),
        _next_attempt=0,
    )
    candidate = Candidate("runpod", "RTX 4090", 0.69, 24)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *_args, **_kwargs: Allocation(
            provider="runpod",
            gpu="RTX 4090",
            hourly_usd=0.69,
            min_vram_gb=24,
            candidates=(candidate,),
        ),
    )
    monkeypatch.setattr(lifecycle.time, "sleep", lambda *_args: None)

    class Provider:
        supports_weight_cache = False

        def __init__(self):
            self.attempts = []

        def submit_run(self, _spec, _seed, *, attempt, on_handle, **_kwargs):
            self.attempts.append(attempt)
            if attempt == 0:
                raise RuntimeError("provider accepted create but response was lost")
            on_handle(
                _runpod_remote(
                    endpoint_id="endpoint-1",
                    job_id="job-1",
                    attempt=attempt,
                    fence=_kwargs["fence"],
                    started_ts=float(attempt + 1),
                )
            )
            return PollResult(True, metrics={"wall_seconds": 1.0})

    provider = Provider()
    monkeypatch.setattr(providers, "get_provider", lambda _name: provider)

    lifecycle._submit_seed_supervised(
        spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
    )

    assert provider.attempts == [0, 1]
    assert runner_status.get_status(spec.run_id).remote["attempt"] == 1
    assert runner_status._load_status_json(spec.run_id)[runner_state._NEXT_ATTEMPT_KEY] == 2


def test_retry_receives_only_remaining_run_global_wall_allowance(monkeypatch, tmp_path):
    import io

    import flash.providers.core.allocator as allocator
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core import registry as providers
    from flash.providers.core.base import Allocation, Candidate, PollResult
    from flash.runner.supervise import lifecycle
    from tests._helpers.profile import attach_sft_profile, stub_revision_geometry

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    # the attached profile pins a model revision, which makes the post-allocation quote refresh
    # resolve revision-specific geometry from the hub. read the catalog's numbers instead.
    stub_revision_geometry(monkeypatch)
    spec = attach_sft_profile(
        JobSpec(
            run_id="wall-budget",
            model="Qwen/Qwen3.5-9B",
            algorithm="sft",
            train=TrainSpec(max_examples=1),
            gpu=GpuSpec(type="", max_wall_seconds=200, max_retries=1),
        )
    )
    status = provisioned_status(spec, state="running", created_at=100.0)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(
        status,
        _run_deadline_at=300.0,
        _next_attempt=0,
    )
    now = {"value": 100.0}
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: now["value"])
    candidate = Candidate("runpod", "RTX 4090", 0.69, 24)
    allocation_walls = []

    def fake_allocate(*_args, **kwargs):
        allocation_walls.append(kwargs["max_wall_seconds"])
        return Allocation(
            provider="runpod",
            gpu="RTX 4090",
            hourly_usd=0.69,
            min_vram_gb=24,
            candidates=(candidate,),
        )

    monkeypatch.setattr(allocator, "allocate", fake_allocate)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda *_args: None)
    monkeypatch.setattr(lifecycle, "_attempt_result", lambda *_args, **_kwargs: None)

    class FakeProvider:
        supports_weight_cache = False

        def __init__(self):
            self.walls = []
            self.attempts = []

        def submit_run(self, run_spec, _seed, *, on_handle, attempt, **_kwargs):
            self.walls.append(run_spec.gpu.max_wall_seconds)
            self.attempts.append(attempt)
            on_handle(
                _runpod_remote(
                    endpoint_id=f"endpoint-{attempt}",
                    job_id=f"job-{attempt}",
                    attempt=attempt,
                    fence=_kwargs["fence"],
                    started_ts=float(attempt + 1),
                )
            )
            if attempt == 0:
                now["value"] = 180.0
                return PollResult(False, failure="poll_error", detail="transient")
            return PollResult(True, metrics={"wall_seconds": 1.0})

        def cancel(self, _handle):
            return None

        def destroy(self, _handle):
            return None

    provider = FakeProvider()
    monkeypatch.setattr(providers, "get_provider", lambda _name: provider)

    lifecycle._submit_seed_supervised(
        spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
    )

    assert provider.attempts == [0, 1]
    assert allocation_walls == [200.0, 120.0]
    assert provider.walls == [200, 120]
    raw = runner_status._load_status_json(spec.run_id)
    assert raw[runner_state._RUN_DEADLINE_AT_KEY] == 300.0
    assert raw[runner_state._NEXT_ATTEMPT_KEY] == 2


def test_retry_backoff_cannot_cross_provider_minimum(monkeypatch, tmp_path):
    import io

    import flash.providers.core.allocator as allocator
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core import registry as providers
    from flash.providers.core.base import Allocation, Candidate
    from flash.runner.supervise import lifecycle
    from tests._helpers.profile import attach_sft_profile, stub_revision_geometry

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    # the attached profile pins a model revision, which makes the post-allocation quote refresh
    # resolve revision-specific geometry from the hub. read the catalog's numbers instead.
    stub_revision_geometry(monkeypatch)
    spec = attach_sft_profile(
        JobSpec(
            run_id="retry-deadline-minimum",
            model="Qwen/Qwen3.5-9B",
            algorithm="sft",
            train=TrainSpec(max_examples=1),
            gpu=GpuSpec(type="", max_wall_seconds=200, max_retries=1),
        )
    )
    runner_state._save_status(
        provisioned_status(spec, state="running", created_at=100.0),
        _run_deadline_at=300.0,
        _next_attempt=0,
    )
    clock = {"now": 230.0}
    sleeps = []
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: clock["now"])

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(lifecycle.time, "sleep", sleep)
    candidate = Candidate("runpod", "RTX 4090", 0.69, 24)
    allocations = []

    def fake_allocate(*_args, **_kwargs):
        allocations.append(True)
        return Allocation(
            provider="runpod",
            gpu="RTX 4090",
            hourly_usd=0.69,
            min_vram_gb=24,
            candidates=(candidate,),
        )

    monkeypatch.setattr(allocator, "allocate", fake_allocate)

    class Provider:
        supports_weight_cache = False

        def __init__(self):
            self.attempts = []

        def submit_run(self, _spec, _seed, *, attempt, **_kwargs):
            self.attempts.append(attempt)
            clock["now"] = 245.0
            raise RuntimeError("provider body secret")

    provider = Provider()
    monkeypatch.setattr(providers, "get_provider", lambda _name: provider)

    log = io.StringIO()
    with pytest.raises(RuntimeError, match="60-second minimum provider allowance") as exc_info:
        lifecycle._submit_seed_supervised(spec, spec.seed, log, source_snapshot=_SOURCE_SNAPSHOT)

    assert provider.attempts == [0]
    assert allocations == [True]
    assert sleeps == [10.0]
    assert "provider body secret" not in str(exc_info.value)
    assert "provider body secret" not in log.getvalue()
    assert runner_status._load_status_json(spec.run_id)[runner_state._NEXT_ATTEMPT_KEY] == 1


def test_save_status_flushes_file_and_directory_before_return(monkeypatch, tmp_path):
    import os

    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="durable-status", model="Qwen/Qwen3.5-4B", algorithm="sft")
    events = []
    directory_fd = 987654
    file_fd = {"value": None}
    original_fdopen = runner_state.os.fdopen
    original_open = runner_state.os.open
    original_close = runner_state.os.close
    original_replace = runner_state.os.replace

    class _RecordingFile:
        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._wrapped.__exit__(*args)

        def write(self, value):
            return self._wrapped.write(value)

        def flush(self):
            events.append("flush")
            return self._wrapped.flush()

        def fileno(self):
            return self._wrapped.fileno()

    def _fdopen(fd, *args, **kwargs):
        file_fd["value"] = fd
        return _RecordingFile(original_fdopen(fd, *args, **kwargs))

    def _fsync(fd):
        events.append("fsync-directory" if fd == directory_fd else "fsync-file")

    def _replace(source, destination):
        events.append("replace")
        return original_replace(source, destination)

    def _open(path, flags, *args, **kwargs):
        if path == runner_state.RUNS_DIR and flags == os.O_RDONLY:
            events.append("open-directory")
            return directory_fd
        return original_open(path, flags, *args, **kwargs)

    def _close(fd):
        if fd == directory_fd:
            events.append("close-directory")
            return None
        return original_close(fd)

    monkeypatch.setattr(runner_state.os, "fdopen", _fdopen)
    monkeypatch.setattr(runner_state.os, "fsync", _fsync)
    monkeypatch.setattr(runner_state.os, "replace", _replace)
    monkeypatch.setattr(runner_state.os, "open", _open)
    monkeypatch.setattr(runner_state.os, "close", _close)

    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict()),
        _run_deadline_at=500.0,
        _next_attempt=0,
    )

    assert file_fd["value"] is not None
    assert events == [
        "flush",
        "fsync-file",
        "replace",
        "open-directory",
        "fsync-directory",
        "close-directory",
    ]


def test_save_status_closes_directory_and_cleans_temp_when_directory_fsync_fails(
    monkeypatch, tmp_path
):
    import os

    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="durable-failure", model="Qwen/Qwen3.5-4B", algorithm="sft")
    directory_fd = 987655
    closed = []
    temp_paths = []
    original_mkstemp = runner_state.tempfile.mkstemp
    original_open = runner_state.os.open
    original_close = runner_state.os.close
    original_fsync = runner_state.os.fsync

    def _mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        temp_paths.append(path)
        return fd, path

    def _open(path, flags, *args, **kwargs):
        if path == runner_state.RUNS_DIR and flags == os.O_RDONLY:
            return directory_fd
        return original_open(path, flags, *args, **kwargs)

    def _fsync(fd):
        if fd == directory_fd:
            raise OSError("directory fsync failed")
        return original_fsync(fd)

    def _close(fd):
        if fd == directory_fd:
            closed.append(fd)
            return None
        return original_close(fd)

    monkeypatch.setattr(runner_state.tempfile, "mkstemp", _mkstemp)
    monkeypatch.setattr(runner_state.os, "open", _open)
    monkeypatch.setattr(runner_state.os, "fsync", _fsync)
    monkeypatch.setattr(runner_state.os, "close", _close)

    with pytest.raises(OSError, match="directory fsync failed"):
        runner_state._save_status(
            runner_state.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
        )

    assert closed == [directory_fd]
    assert temp_paths
    assert all(not os.path.exists(path) for path in temp_paths)


def test_save_status_cleans_temp_when_replace_fails(monkeypatch, tmp_path):
    import os

    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="replace-failure", model="Qwen/Qwen3.5-4B", algorithm="sft")
    temp_paths = []
    original_mkstemp = runner_state.tempfile.mkstemp

    def _mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        temp_paths.append(path)
        return fd, path

    monkeypatch.setattr(runner_state.tempfile, "mkstemp", _mkstemp)
    monkeypatch.setattr(
        runner_state.os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        runner_state._save_status(
            runner_state.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
        )

    assert temp_paths
    assert all(not os.path.exists(path) for path in temp_paths)


def test_concurrent_attempt_reservations_are_unique_and_monotonic(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="threaded-attempts", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict()),
        _next_attempt=0,
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        attempts = list(
            pool.map(lambda _index: runner_attempts._reserve_attempt(spec.run_id), range(16))
        )

    assert sorted(attempts) == list(range(16))
    assert runner_status._load_status_json(spec.run_id)[runner_state._NEXT_ATTEMPT_KEY] == 16


def test_multiprocess_attempt_reservations_preserve_concurrent_status_update(monkeypatch, tmp_path):
    import multiprocessing

    from flash.core.spec import JobSpec

    runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(runner_state, "RUNS_DIR", runs_dir)
    spec = JobSpec(run_id="multiprocess-attempts", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict()),
        _next_attempt=0,
    )
    context = multiprocessing.get_context("fork")
    start = context.Barrier(4)
    results = context.Queue()

    def reserve(worker_index):
        import time

        import flash.runner.lifecycle.attempts as child_runner_attempts
        import flash.runner.lifecycle.state as child_runner_state
        import flash.runner.lifecycle.status as child_runner_status

        child_runner_state.RUNS_DIR = runs_dir
        original_save = child_runner_state._save_status_unlocked

        def slow_save(*args, **kwargs):
            time.sleep(0.005)
            return original_save(*args, **kwargs)

        child_runner_state._save_status_unlocked = slow_save
        start.wait()
        if worker_index == 0:
            child_runner_status._update(spec.run_id, "provisioning", error="concurrent-update")
        attempts = [child_runner_attempts._reserve_attempt(spec.run_id) for _ in range(4)]
        results.put(attempts)

    processes = [context.Process(target=reserve, args=(index,)) for index in range(4)]
    for process in processes:
        process.start()
    attempts = []
    for _ in processes:
        attempts.extend(results.get(timeout=10))
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0

    raw = runner_status._load_status_json(spec.run_id)
    assert sorted(attempts) == list(range(16))
    assert raw[runner_state._NEXT_ATTEMPT_KEY] == 16
    assert raw["error"] == "concurrent-update"


@pytest.mark.parametrize(
    "remote",
    [
        _runpod_remote(attempt=2),
        _runpod_remote(job_id=None, attempt=2),
        _vast_remote(attempt=2),
    ],
)
def test_compare_and_clear_remote_uses_exact_provider_resource_identity(
    monkeypatch, tmp_path, remote
):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="compare-clear", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            remote=remote,
        )
    )

    assert runner_reconciliation._compare_and_clear_remote(spec.run_id, remote) is True
    assert runner_status.get_status(spec.run_id).remote is None


@pytest.mark.parametrize(
    ("original", "newer"),
    [
        (
            _runpod_remote(job_id="job-old", attempt=1),
            _runpod_remote(job_id="job-new", attempt=2, started_ts=2.0),
        ),
        (
            _runpod_remote(job_id=None, attempt=1),
            _runpod_remote(job_id="job-new", attempt=2, started_ts=2.0),
        ),
        (
            _lambda_remote(instance_id="instance-old", attempt=1),
            _lambda_remote(instance_id="instance-new", attempt=2, started_ts=2.0),
        ),
    ],
)
def test_compare_and_clear_remote_preserves_newer_resource(monkeypatch, tmp_path, original, newer):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="compare-preserve", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            remote=newer,
        )
    )

    assert runner_reconciliation._compare_and_clear_remote(spec.run_id, original) is False
    assert runner_status.get_status(spec.run_id).remote == newer


def test_cleanup_collection_deduplicates_and_survives_status_writes_and_reload(
    monkeypatch, tmp_path
):
    from flash.core.spec import JobSpec

    runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(runner_state, "RUNS_DIR", runs_dir)
    spec = JobSpec(run_id="cleanup-dedup", model="Qwen/Qwen3.5-4B", algorithm="sft")
    public_remote = _runpod_remote("endpoint-a", "job-a", attempt=0)
    cleanup_remote = _runpod_remote("endpoint-b", None, attempt=1, started_ts=2.0)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="cancelled",
            spec=spec.to_dict(),
            remote=public_remote,
        )
    )

    assert runner_reconciliation._preserve_cleanup_remote(spec.run_id, cleanup_remote) is True
    assert runner_reconciliation._preserve_cleanup_remote(spec.run_id, cleanup_remote) is True
    assert runner_status._update(spec.run_id, "cancelled", error="unchanged terminal state") is True

    raw = runner_status._load_status_json(spec.run_id)
    assert raw["remote"] == public_remote
    assert raw[runner_state._CLEANUP_REMOTES_KEY] == [cleanup_remote]

    monkeypatch.setattr(runner_state, "RUNS_DIR", runs_dir)
    reloaded = runner_status._load_status_json(spec.run_id)
    assert reloaded["remote"] == public_remote
    assert reloaded[runner_state._CLEANUP_REMOTES_KEY] == [cleanup_remote]


def test_record_cleanup_remote_does_not_revive_cleared_remote(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="cleanup-record", model="Qwen/Qwen3.5-4B", algorithm="sft")
    remote = _runpod_remote("endpoint-cleanup", "job-cleanup", attempt=1)
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict())
    )

    assert runner_reconciliation._record_cleanup_remote(spec.run_id, remote) is True
    assert runner_reconciliation._record_cleanup_remote(spec.run_id, remote) is True

    status = runner_status.get_status(spec.run_id)
    assert status.remote is None
    assert runner_status._load_status_json(spec.run_id)[runner_state._CLEANUP_REMOTES_KEY] == [
        remote
    ]


@pytest.mark.parametrize("settlement", ["completion", "failure"])
def test_handleless_terminal_settlement_rejects_replaced_attempt(monkeypatch, tmp_path, settlement):
    from flash.core.spec import JobSpec
    from flash.runner.lifecycle.protocol import AttemptRecord

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(run_id=f"handleless-{settlement}", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(_handleless_status(spec), _next_attempt=1, _next_fence=2)
    observed = runner_status._current_attempt(runner_status.get_status(spec.run_id))
    replacement = replace(observed, attempt_id=1, fence=2, state="active")

    def install_replacement() -> None:
        with runner_state._status_guard(spec.run_id):
            current = runner_status.get_status(spec.run_id)
            current.attempt = replacement.to_dict()
            runner_state._save_status_unlocked(current)

    if settlement == "completion":
        monkeypatch.setattr(
            runner_status,
            "_persist_metrics",
            lambda *_args, **_kwargs: install_replacement() or 0.0,
        )
        applied = runner_reconciliation._compare_and_complete_remote(
            spec.run_id,
            None,
            spec,
            _fenced_success_metrics(spec),
            expected_attempt_id=observed.attempt_id,
            expected_fence=observed.fence,
        )
    else:
        install_replacement()
        applied = runner_reconciliation._compare_and_fail_remote(
            spec.run_id,
            None,
            "attempt failed",
            expected_attempt_id=observed.attempt_id,
            expected_fence=observed.fence,
        )

    assert applied is False
    persisted = runner_status.get_status(spec.run_id)
    assert persisted.state not in runner_state.TERMINAL_STATES
    assert AttemptRecord.from_dict(persisted.attempt) == replacement
    assert AttemptRecord.from_dict(persisted.attempt).state == "active"


@pytest.mark.parametrize("settlement", ["completion", "failure"])
def test_handleless_terminal_settlement_requires_attempt_identity(
    monkeypatch, tmp_path, settlement
):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(run_id=f"identity-free-{settlement}", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict())
    )
    if settlement == "completion":

        def settle():
            return runner_reconciliation._compare_and_complete_remote(spec.run_id, None, spec, {})

    else:

        def settle():
            return runner_reconciliation._compare_and_fail_remote(
                spec.run_id, None, "attempt failed"
            )

    with pytest.raises(ValueError, match="exact attempt id and fence"):
        settle()

    persisted = runner_status.get_status(spec.run_id)
    assert persisted.state == "provisioning"
    assert persisted.attempt is None


def test_recovered_completion_does_not_overwrite_concurrent_cancel(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="completion-cancel", model="Qwen/Qwen3.5-4B", algorithm="sft")
    remote = _runpod_remote("endpoint-active", "job-active", attempt=0)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            remote=remote,
        )
    )
    real_record = runner_reconciliation._record_cleanup_remote

    def clear_then_record(run_id, cleanup_remote):
        with runner_state._status_guard(run_id):
            status = runner_status.get_status(run_id)
            status.remote = None
            runner_state._save_status_unlocked(status)
        return real_record(run_id, cleanup_remote)

    monkeypatch.setattr(runner_reconciliation, "_record_cleanup_remote", clear_then_record)
    monkeypatch.setattr(runner_status, "_persist_metrics", lambda *_args, **_kwargs: 0.0)

    assert (
        runner_reconciliation._compare_and_complete_remote(spec.run_id, remote, spec, {}) is False
    )
    assert runner_status.get_status(spec.run_id).state == "running"
    assert runner_status._update(spec.run_id, "cancelled")

    status = runner_status.get_status(spec.run_id)
    assert status.state == "cancelled"
    assert status.remote is None
    assert runner_status._load_status_json(spec.run_id)[runner_state._CLEANUP_REMOTES_KEY] == [
        remote
    ]


def test_recovered_completion_adopts_after_confirmed_teardown(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(run_id="completion-confirmed", model="Qwen/Qwen3.5-9B", algorithm="sft")
    remote = _runpod_remote("endpoint-completed", "job-completed", attempt=0)
    status = provisioned_status(spec, state="running", remote=remote)
    status.remote = None
    status.cleanup_confirmed_remote = remote
    status.realized_cost_remote = remote
    runner_state._save_status(status)
    monkeypatch.setattr(
        runner_status,
        "validate_terminal_source_metrics",
        lambda _status, metrics, expected_attempt=None, expected_fence=None: (
            metrics,
            expected_attempt,
        ),
    )
    monkeypatch.setattr(runner_status, "_persist_metrics", lambda *_args, **_kwargs: 0.5)

    assert runner_reconciliation._compare_and_complete_remote(
        spec.run_id,
        remote,
        spec,
        {"optimizer_steps_completed": 1},
    )

    completed = runner_status.get_status(spec.run_id)
    assert completed.state == "done"
    assert completed.remote is None
    assert completed.cleanup_confirmed_remote == remote
    assert completed.realized_cost_remote == remote


@pytest.mark.parametrize("terminal_state", ["done", "failed"])
def test_recovered_terminal_runs_keep_remote_for_cost_reconciliation(
    monkeypatch, tmp_path, terminal_state
):
    from flash.core.spec import JobSpec
    from flash.server.domain.ops import reconcile

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id=f"recovered-{terminal_state}", model="Qwen/Qwen3.5-9B", algorithm="sft")
    remote = _runpod_remote("endpoint-cost", "job-cost", attempt=0, started_ts=100.0)
    runner_state._save_status(
        provisioned_status(
            spec,
            state="running",
            created_at=90.0,
            remote=remote,
        )
    )
    if terminal_state == "done":
        monkeypatch.setattr(runner_status, "_persist_metrics", lambda *_args, **_kwargs: 0.5)
        assert (
            runner_reconciliation._compare_and_complete_remote(spec.run_id, remote, spec, {})
            is True
        )
    else:
        assert (
            runner_reconciliation._compare_and_fail_remote(spec.run_id, remote, "provider failed")
            is True
        )

    status = runner_status.get_status(spec.run_id)
    assert status.state == terminal_state
    assert status.remote == remote
    assert runner_status._current_attempt(status).state == "settled"
    assert reconcile._due(status, status.finished_at + reconcile._SETTLE_SECONDS + 1.0)

    if terminal_state == "failed":
        assert runner_reconciliation._record_cleanup_remote(spec.run_id, remote) is True
    assert runner_reconciliation._compare_and_remove_cleanup_remote(spec.run_id, remote) is True

    status = runner_status.get_status(spec.run_id)
    assert status.remote is None
    assert status.realized_cost_remote == remote
    assert reconcile._due(status, status.finished_at + reconcile._SETTLE_SECONDS + 1.0)


def test_cleanup_collection_removes_only_confirmed_exact_records(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="cleanup-drain", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="cancelled", spec=spec.to_dict())
    )
    confirmed = _runpod_remote("endpoint-confirmed", "job-confirmed", attempt=1)
    endpoint_only = _runpod_remote(
        "endpoint-only",
        None,
        attempt=2,
        started_ts=2.0,
    )
    unconfirmed = _runpod_remote(
        "endpoint-unconfirmed",
        "job-unconfirmed",
        attempt=3,
        started_ts=3.0,
    )
    for remote in (confirmed, endpoint_only, unconfirmed):
        assert runner_reconciliation._preserve_cleanup_remote(spec.run_id, remote) is True

    reports = []
    monkeypatch.setattr(runner_reporting, "_report_status", reports.append)
    events = []

    class Provider:
        def cancel(self, handle):
            data = handle.to_dict()
            events.append(("cancel", data["endpoint_id"]))
            if data["endpoint_id"] == "endpoint-unconfirmed":
                raise RuntimeError("cancellation unconfirmed")

        def destroy(self, handle):
            endpoint_id = handle.to_dict()["endpoint_id"]
            events.append(("destroy", endpoint_id))
            if endpoint_id == "endpoint-unconfirmed":
                raise RuntimeError("endpoint deletion unconfirmed")

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())

    attempted = runner_reconciliation._drain_cleanup_remotes(spec.run_id)

    assert attempted == {
        ("runpod", 1, "endpoint-confirmed", "job-confirmed", _RUNPOD_FINGERPRINT),
        ("runpod", 2, "endpoint-only", None, _RUNPOD_FINGERPRINT),
        ("runpod", 3, "endpoint-unconfirmed", "job-unconfirmed", _RUNPOD_FINGERPRINT),
    }
    assert events == [
        ("cancel", "endpoint-confirmed"),
        ("destroy", "endpoint-confirmed"),
        ("destroy", "endpoint-only"),
        ("cancel", "endpoint-unconfirmed"),
        ("destroy", "endpoint-unconfirmed"),
    ]
    raw = runner_status._load_status_json(spec.run_id)
    assert raw[runner_state._CLEANUP_REMOTES_KEY] == [unconfirmed]
    assert raw["remote"] is None
    assert raw["realized_cost_remote"] == confirmed
    assert [report.remote for report in reports] == [None, None]


def test_cleanup_drain_tears_down_a_record_that_fails_strict_canonicalization(
    monkeypatch, tmp_path
):
    """A deployed-format record still names a billable endpoint, so teardown must reach it.

    `key_fingerprint` is validated at exactly 68 chars, but the deployed release writes the 16-char
    form, so such a record fails the strict `from_dict` behind `_canonical_cleanup_remote` and
    `_remote_resource_identity`. The teardown loop builds a base `JobHandle` (which validates only
    `provider`) and `_delete_runpod_endpoint` resolves that exact fingerprint through
    `resolve_prefix_key_fingerprint`. Filtering the record out before teardown would leave a live
    RunPod endpoint billing forever with nothing left to delete it.
    """
    import json as _json

    from flash.core.spec import JobSpec
    from flash.providers.runpod.client import api as runpod_api

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="cleanup-legacy", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="cancelled", spec=spec.to_dict())
    )
    legacy_fingerprint = "rpk-" + "0" * 12
    resolved_fingerprint = "rpk-" + "a" * 64
    legacy = _runpod_remote("endpoint-legacy", None, attempt=1, key_fingerprint=legacy_fingerprint)
    # the strict writer rejects the legacy record, so seed the status file directly.
    path = runner_state.runs_file_path(spec.run_id, ".json")
    with open(path) as f:
        raw = _json.load(f)
    raw[runner_state._CLEANUP_REMOTES_KEY] = [legacy]
    with open(path, "w") as f:
        _json.dump(raw, f)

    def _key_for_fingerprint(fingerprint):
        raise runpod_api.RunpodApiError("no configured key matches the stored fingerprint")

    resolved = []
    deleted = []

    def resolve_legacy(endpoint_id, fingerprint):
        resolved.append((endpoint_id, fingerprint))
        return resolved_fingerprint

    def delete_endpoint(endpoint_id, fingerprint):
        deleted.append((endpoint_id, fingerprint))
        return True

    monkeypatch.setattr(runpod_api, "_key_for_fingerprint", _key_for_fingerprint)
    monkeypatch.setattr(runpod_api, "resolve_prefix_key_fingerprint", resolve_legacy)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", delete_endpoint)

    attempted = runner_reconciliation._drain_cleanup_remotes(spec.run_id)

    assert resolved == [("endpoint-legacy", legacy_fingerprint)], (
        "the legacy fingerprint resolver was never reached"
    )
    assert deleted == [("endpoint-legacy", resolved_fingerprint)], (
        "the billable endpoint was never deleted"
    )
    assert any("endpoint-legacy" in repr(item) for item in attempted)
    # and the confirmed-deleted record must actually leave the file. removal derives its key from
    # the same record the drain admitted, so a strict-only derivation would return None here and
    # clear nothing -- leaving every later sweep to tear down an endpoint that is already gone.
    with open(path) as f:
        assert not _json.load(f).get(runner_state._CLEANUP_REMOTES_KEY), (
            "the confirmed-deleted record survived, so every later sweep retries a deleted endpoint"
        )


def test_confirmed_deferred_cleanup_retains_pending_charge_attribution(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="cleanup-pending-charge", model="Qwen/Qwen3.5-9B", algorithm="sft")
    remote = _runpod_remote("endpoint-charge", "job-charge", attempt=0)
    status = runner_state.RunStatus(
        run_id=spec.run_id,
        state="done",
        spec=spec.to_dict(),
        remote=remote,
        billing_context={"org_id": "org-1"},
        billing_state="failed",
        reconciled_at=100.0,
    )
    runner_state._save_status(status, _cleanup_remotes=[remote])

    assert runner_reconciliation._compare_and_remove_cleanup_remote(spec.run_id, remote)

    persisted = runner_status.get_status(spec.run_id)
    assert persisted.remote is None
    assert persisted.cleanup_confirmed_remote == remote
    assert persisted.realized_cost_remote == remote


def test_cleanup_collection_removes_only_fully_confirmed_runpod_record(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec
    from flash.providers.runpod.client import api as runpod_api

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="cleanup-absent", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="cancelled", spec=spec.to_dict())
    )
    other_fingerprint = "rpk-" + "f" * 64
    confirmed = _runpod_remote("endpoint-shared", "job-confirmed", attempt=1)
    different_owner = _runpod_remote(
        "endpoint-shared",
        "job-confirmed",
        attempt=1,
        key_fingerprint=other_fingerprint,
        started_ts=2.0,
    )
    different_job = _runpod_remote(
        "endpoint-shared",
        "job-other",
        attempt=1,
        started_ts=3.0,
    )
    different_attempt = _runpod_remote(
        "endpoint-shared",
        "job-confirmed",
        attempt=2,
        started_ts=4.0,
    )
    for remote in (confirmed, different_owner, different_job, different_attempt):
        assert runner_reconciliation._preserve_cleanup_remote(spec.run_id, remote) is True

    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kwargs: {
            "id": job_id,
            "status": "CANCELLED",
        },
    )
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda _endpoint_id, _fingerprint: False,
    )
    exact_lookups = []

    def exact_lookup(endpoint_id, fingerprint):
        exact_lookups.append((endpoint_id, fingerprint))
        if len(exact_lookups) == 1:
            return True
        raise runpod_api.RunpodApiError("exact endpoint lookup unconfirmed")

    monkeypatch.setattr(runpod_api, "endpoint_absent_for_fingerprint", exact_lookup)

    attempted = runner_reconciliation._drain_cleanup_remotes(spec.run_id)

    assert attempted == {
        ("runpod", 1, "endpoint-shared", "job-confirmed", _RUNPOD_FINGERPRINT),
        ("runpod", 1, "endpoint-shared", "job-confirmed", other_fingerprint),
        ("runpod", 1, "endpoint-shared", "job-other", _RUNPOD_FINGERPRINT),
        ("runpod", 2, "endpoint-shared", "job-confirmed", _RUNPOD_FINGERPRINT),
    }
    assert exact_lookups == [
        ("endpoint-shared", _RUNPOD_FINGERPRINT),
        ("endpoint-shared", other_fingerprint),
        ("endpoint-shared", _RUNPOD_FINGERPRINT),
        ("endpoint-shared", _RUNPOD_FINGERPRINT),
    ]
    raw = runner_status._load_status_json(spec.run_id)
    assert raw[runner_state._CLEANUP_REMOTES_KEY] == [
        different_owner,
        different_job,
        different_attempt,
    ]
    assert raw["remote"] is None
    assert raw["realized_cost_remote"] == confirmed


def test_next_attempt_requires_persisted_integer_identity():

    assert runner_attempts._infer_next_attempt({"next_attempt": 0}) == 0
    assert runner_attempts._infer_next_attempt({"next_attempt": 7}) == 7
    for raw in ({}, {"next_attempt": True}, {"next_attempt": -1}, {"next_attempt": "1"}):
        with pytest.raises(RuntimeError, match="next attempt identity"):
            runner_attempts._infer_next_attempt(raw)


def test_handleless_state_without_next_attempt_is_rejected(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="missing-next-attempt", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict())
    )
    raw = runner_status._load_status_json(spec.run_id)
    raw.pop(runner_state._NEXT_ATTEMPT_KEY)
    with open(runner_state.runs_file_path(spec.run_id, ".json"), "w") as file:
        json.dump(raw, file)

    with pytest.raises(RuntimeError, match="next attempt identity is missing"):
        runner_attempts._reserve_attempt(spec.run_id)


def test_new_attempt_requires_full_provider_minimum_before_allocation(monkeypatch, tmp_path):
    import io

    import flash.providers.core.allocator as allocator
    from flash.core.spec import GpuSpec, JobSpec
    from flash.runner.supervise import lifecycle

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="wall-minimum",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=60),
    )
    runner_state._save_status(
        provisioned_status(spec, state="running", created_at=99.0),
        _run_deadline_at=159.0,
        _next_attempt=0,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 100.0)
    allocations = []
    monkeypatch.setattr(allocator, "allocate", lambda *_args, **_kwargs: allocations.append(True))

    with pytest.raises(RuntimeError, match="60-second minimum provider allowance"):
        lifecycle._submit_seed_supervised(
            spec, spec.seed, io.StringIO(), source_snapshot=_SOURCE_SNAPSHOT
        )

    assert allocations == []
    assert runner_status._load_status_json(spec.run_id)[runner_state._NEXT_ATTEMPT_KEY] == 0


def test_attempt_reservation_persists_fixed_candidate_grant_budget(monkeypatch, tmp_path):
    import io
    from types import SimpleNamespace

    from flash.core.spec import GpuSpec, JobSpec
    from flash.runner.supervise import seed_submission

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_attempts.time, "time", lambda: 200.0)
    cases = (
        ("grant-one", 1, 1100.0),
        ("grant-four", 4, 3800.0),
        ("grant-capped", 4, 600.0),
    )
    for run_id, gpu_count, expected_grant in cases:
        max_wall = 500.0 if run_id == "grant-capped" else 5000.0
        spec = JobSpec(
            run_id=run_id,
            model="Qwen/Qwen3.5-9B",
            algorithm="sft",
            gpu=GpuSpec(max_wall_seconds=max_wall),
        )
        runner_state._save_status(
            provisioned_status(spec, state="provisioning", created_at=100.0),
            _run_deadline_at=100.0 + max_wall,
            _next_attempt=0,
        )
        ctx = SimpleNamespace(
            spec=spec,
            attempt_start=0,
            current_attempt=None,
            current_fence=None,
            log=io.StringIO(),
        )
        prepared = seed_submission._PreparedAttempt(
            local_attempt=0,
            attempt=0,
            fence=0,
            attempt_spec=spec,
            runtime_secrets={},
            expected_next_attempt=0,
        )
        plan = SimpleNamespace(
            chosen=SimpleNamespace(provider="runpod", gpu_count=gpu_count),
            on_last_gpu=True,
        )
        reserved = seed_submission._reserve_candidate_attempt(ctx, prepared, plan)
        assert runner_status.get_status(run_id).attempt["grant_deadline_at"] == expected_grant
        assert (reserved.attempt, reserved.fence) == (ctx.current_attempt, ctx.current_fence)

        persisted = runner_status.get_status(run_id).attempt
        with pytest.raises(RuntimeError, match="changed after retry verification"):
            seed_submission._reserve_candidate_attempt(ctx, prepared, plan)
        assert runner_status.get_status(run_id).attempt == persisted


def test_reserved_attempt_survives_handleless_restart_without_reusing_zero(monkeypatch, tmp_path):
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="reserved-restart",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=200),
    )
    runner_state._save_status(
        provisioned_status(spec, state="provisioning", created_at=100.0),
        _run_deadline_at=300.0,
        _next_attempt=0,
    )

    assert (
        runner_deadlines._spec_with_remaining_wall(
            spec, require_provider_minimum=True, now=100.0
        ).gpu.max_wall_seconds
        == 200
    )
    monkeypatch.setattr(runner_attempts.time, "time", lambda: 100.0)
    assert runner_attempts._reserve_attempt(spec.run_id) == 0
    assert runner_status.get_status(spec.run_id).remote is None
    assert (
        runner_deadlines._spec_with_remaining_wall(
            spec, require_provider_minimum=True, now=180.0
        ).gpu.max_wall_seconds
        == 120
    )
    monkeypatch.setattr(runner_attempts.time, "time", lambda: 180.0)
    assert runner_attempts._reserve_attempt(spec.run_id) == 1
    raw = runner_status._load_status_json(spec.run_id)
    assert raw[runner_state._NEXT_ATTEMPT_KEY] == 2
    assert raw[runner_state._RUN_DEADLINE_AT_KEY] == 300.0


def test_confirmed_recovery_keeps_accounting_identity_until_replacement_handle(
    monkeypatch, tmp_path
):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="replacement-accounting", model="Qwen/Qwen3.5-9B", algorithm="sft")
    remote = _runpod_remote("endpoint-old", "job-old", attempt=0)
    status = runner_state.RunStatus(
        run_id=spec.run_id,
        state="running",
        spec=spec.to_dict(),
        remote=None,
        cleanup_confirmed_remote=remote,
        realized_cost_remote=remote,
    )
    runner_state._save_status(status)

    assert runner_reconciliation._compare_and_clear_remote(spec.run_id, remote)
    cleared = runner_status.get_status(spec.run_id)
    assert cleared.cleanup_confirmed_remote is None
    assert cleared.realized_cost_remote == remote


def test_attach_resumes_after_cleanup_drain_clears_classified_remote(monkeypatch, tmp_path):
    import io

    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(
        run_id="attach-cleanup-race",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=1, max_wall_seconds=200),
    )
    remote = _runpod_remote("endpoint-old", "job-old", attempt=0)
    status = provisioned_status(spec, state="running", created_at=100.0, remote=remote)
    status.source_snapshot = _SOURCE_SNAPSHOT
    status.cleanup_confirmed_remote = remote
    status.realized_cost_remote = remote
    status.remote = None
    runner_state._save_status(
        status,
        _run_deadline_at=300.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        runner_lifecycle,
        "_attempt_result",
        lambda *_args, **_kwargs: PollResult(
            False, failure="job_preempted", detail="provider confirmed worker loss"
        ),
    )
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    resumed = []

    def fake_run_training(_spec, _log, **kwargs):
        resumed.append(kwargs["attempt_start"])

    monkeypatch.setattr(runner_lifecycle, "_run_training", fake_run_training)

    attached = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert resumed == [1]
    assert attached.state == "running"
    assert attached.remote is None
    assert attached.cleanup_confirmed_remote is None
    assert attached.realized_cost_remote == remote


def test_attach_reconciler_continues_after_confirmed_runpod_teardown(monkeypatch, tmp_path):
    import io

    import flash.runner.supervise.attach as attach_mod
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-confirmed-cleanup",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=1, max_wall_seconds=200),
    )
    remote = _runpod_remote("endpoint-confirmed", "job-confirmed", attempt=0)
    runner_state._save_status(provisioned_status(spec, state="running", remote=remote))
    monkeypatch.setattr(
        runner_lifecycle,
        "_attempt_result",
        lambda *_args, **_kwargs: PollResult(
            False, failure="job_preempted", detail="provider confirmed worker loss"
        ),
    )
    monkeypatch.setattr(attach_mod, "_wait_for_replacement_window", lambda *_args: False)
    monkeypatch.setattr(runner_lifecycle, "_strict_teardown_handle", lambda *_args: True)
    monkeypatch.setattr(
        runner_reconciliation, "_record_cleanup_remote", lambda *_args, **_kwargs: False
    )
    resumed = []
    monkeypatch.setattr(
        attach_mod,
        "_resume_after_confirmed_teardown",
        lambda *_args, **_kwargs: resumed.append(True),
    )

    attach_mod._reconcile_attached_remote(
        spec.run_id,
        remote,
        spec,
        1,
        _SOURCE_SNAPSHOT,
        io.StringIO(),
        "stalled: host vanished",
    )

    assert resumed == [True]
    persisted = runner_status.get_status(spec.run_id)
    assert persisted.remote is None
    assert persisted.cleanup_confirmed_remote == remote


def test_attach_reconciler_persists_unconfirmed_runpod_teardown(monkeypatch, tmp_path):
    import io

    import flash.runner.supervise.attach as attach_mod
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-unconfirmed-cleanup",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=1, max_wall_seconds=200),
    )
    remote = _runpod_remote("endpoint-unconfirmed", "job-unconfirmed", attempt=0)
    status = provisioned_status(spec, state="running", remote=remote)
    runner_state._save_status(status)
    monkeypatch.setattr(
        runner_lifecycle,
        "_attempt_result",
        lambda *_args, **_kwargs: PollResult(
            False, failure="job_preempted", detail="provider confirmed worker loss"
        ),
    )
    monkeypatch.setattr(attach_mod, "_wait_for_replacement_window", lambda *_args: False)
    monkeypatch.setattr(runner_lifecycle, "_strict_teardown_handle", lambda *_args: False)
    resumed = []
    monkeypatch.setattr(
        attach_mod,
        "_resume_after_confirmed_teardown",
        lambda *_args, **_kwargs: resumed.append(True),
    )

    attach_mod._reconcile_attached_remote(
        spec.run_id,
        remote,
        spec,
        1,
        _SOURCE_SNAPSHOT,
        io.StringIO(),
        "stalled: host vanished",
    )

    assert resumed == [True]
    assert runner_status._load_status_json(spec.run_id)[runner_state._CLEANUP_REMOTES_KEY] == [
        remote
    ]


def test_confirmed_cleanup_retries_transient_staging_without_restart(monkeypatch, tmp_path):
    import io

    import flash.runner.supervise.attach as attach_mod
    from flash.core.spec import GpuSpec, JobSpec
    from flash.envs.loading.staged import StagedEnvironmentTransientError

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-confirmed-staging-retry",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=1, max_wall_seconds=200),
    )
    remote = _runpod_remote("endpoint-confirmed", "job-confirmed", attempt=0)
    status = provisioned_status(spec, state="running", remote=remote)
    status.remote = None
    status.cleanup_confirmed_remote = remote
    runner_state._save_status(status, _next_attempt=1)
    monkeypatch.setattr(
        runner_lifecycle,
        "_attempt_result",
        lambda *_args, **_kwargs: PollResult(
            False, failure="job_preempted", detail="provider confirmed worker loss"
        ),
    )
    monkeypatch.setattr(attach_mod, "_wait_for_replacement_window", lambda *_args: False)
    monkeypatch.setattr(attach_mod, "_ATTACH_RECONCILE_INTERVAL_S", 0.0)
    attempts = []

    def resume(*_args, **_kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise StagedEnvironmentTransientError("artifact store temporarily unavailable")
        runner_status._update(spec.run_id, "failed", error="stopped after retry proof")

    monkeypatch.setattr(attach_mod, "_resume_after_confirmed_teardown", resume)

    attach_mod._reconcile_attached_remote(
        spec.run_id,
        remote,
        spec,
        1,
        _SOURCE_SNAPSHOT,
        io.StringIO(),
        "stalled: host vanished",
    )

    assert attempts == [True, True]
    assert runner_status.get_status(spec.run_id).state == "failed"


def test_confirmed_cleanup_adopts_completed_work_before_resubmitting(monkeypatch, tmp_path):
    import io

    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(
        run_id="attach-confirmed-completed",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=1, max_wall_seconds=200),
    )
    remote = _runpod_remote("endpoint-completed", "job-completed", attempt=0)
    status = provisioned_status(spec, state="running", remote=remote)
    status.remote = None
    status.source_snapshot = _SOURCE_SNAPSHOT
    status.cleanup_confirmed_remote = remote
    status.realized_cost_remote = remote
    runner_state._save_status(status, _next_attempt=1)
    metrics = {"wall_seconds": 60.0, "optimizer_steps_completed": 1}
    monkeypatch.setattr(
        runner_lifecycle,
        "_attempt_result",
        lambda *_args, **_kwargs: PollResult(True, metrics=dict(metrics)),
    )
    adopted = []

    def adopt(run_id, worker_spec, expected_remote, completed_metrics, *, log):
        adopted.append((run_id, expected_remote, completed_metrics))
        runner_status._update(run_id, "done")
        return True

    monkeypatch.setattr(runner_lifecycle, "_adopt_completed_attempt", adopt)
    monkeypatch.setattr(
        runner_lifecycle,
        "_run_training",
        lambda *_args, **_kwargs: pytest.fail("completed work was retrained"),
    )

    attached = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert attached.state == "done"
    assert len(adopted) == 1
    assert adopted[0][0] == spec.run_id
    assert adopted[0][1] == remote
    assert adopted[0][2]["optimizer_steps_completed"] == 1
    assert adopted[0][2]["allocated_provider"] == "runpod"


def test_confirmed_completed_work_defers_when_adoption_is_transient(monkeypatch, tmp_path):
    import io

    import flash.runner.supervise.attach as attach_mod
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-confirmed-adoption-pending",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=1, max_wall_seconds=200),
    )
    remote = _runpod_remote("endpoint-completed", "job-completed", attempt=0)
    status = provisioned_status(spec, state="running", remote=remote)
    status.remote = None
    status.cleanup_confirmed_remote = remote
    runner_state._save_status(status, _next_attempt=1)
    monkeypatch.setattr(
        runner_lifecycle,
        "_attempt_result",
        lambda *_args, **_kwargs: PollResult(True, metrics={"optimizer_steps_completed": 1}),
    )
    monkeypatch.setattr(
        runner_lifecycle, "_adopt_completed_attempt", lambda *_args, **_kwargs: False
    )
    scheduled = []
    monkeypatch.setattr(
        attach_mod,
        "_schedule_attach_reconciliation",
        lambda *args, **kwargs: scheduled.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        attach_mod,
        "_resume_after_confirmed_teardown",
        lambda *_args, **_kwargs: pytest.fail("completed work was resubmitted"),
    )

    attached = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert attached.state == "running"
    assert len(scheduled) == 1
    assert scheduled[0][0][1] == remote


def test_unparseable_attach_fails_after_confirmed_cleanup_without_reteardown(monkeypatch, tmp_path):
    import io

    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-confirmed-unparseable",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
    )
    remote = _runpod_remote("endpoint-confirmed", "job-confirmed", attempt=0)
    status = provisioned_status(spec, state="running", remote=remote)
    status.remote = None
    status.cleanup_confirmed_remote = remote
    status.realized_cost_remote = remote
    runner_state._save_status(status)
    path = runner_state.runs_file_path(spec.run_id, ".json")
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    stored["spec"]["algorithm"] = "retired-algorithm"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stored, handle)
    monkeypatch.setattr(
        runner_lifecycle,
        "_strict_teardown_handle",
        lambda *_args, **_kwargs: pytest.fail("confirmed resource was torn down again"),
    )

    attached = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert attached.state == "failed"
    assert "spec is malformed" in (attached.error or "")
    assert attached.remote is None
    assert attached.cleanup_confirmed_remote == remote
    assert attached.realized_cost_remote == remote


def test_confirmed_cleanup_can_fail_a_nonretryable_run(monkeypatch, tmp_path):
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-cleanup-no-retry",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=0, max_wall_seconds=200),
    )
    remote = _runpod_remote("endpoint-old", "job-old", attempt=0)
    status = provisioned_status(spec, state="running", remote=remote)
    status.remote = None
    status.cleanup_confirmed_remote = remote
    runner_state._save_status(status)

    assert runner_reconciliation._compare_and_fail_remote(
        spec.run_id,
        remote,
        "persisted cleanup confirmed the worker was torn down",
    )
    persisted = runner_status.get_status(spec.run_id)
    assert persisted.state == "failed"
    assert persisted.remote is None
    assert persisted.cleanup_confirmed_remote == remote


def test_attach_failed_worker_resumes_with_next_attempt_identity(monkeypatch, tmp_path):
    persisted_attempt = 1
    expected_next = 2
    import io

    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core import registry as providers
    from flash.providers.core.base import PollResult

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(
        run_id=f"attach-attempt-{expected_next}",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=200),
    )
    remote = _runpod_remote(
        "endpoint-old",
        "job-old",
        attempt=persisted_attempt,
        fence=1,
    )
    attempt = {
        "attempt_id": persisted_attempt,
        "fence": 1,
        "state": "active",
        "reserved_at": 100.0,
        "grant_deadline_at": 120.0,
        "work_deadline_at": 250.0,
        "result_deadline_at": 280.0,
        "run_deadline_at": 300.0,
        "provider": "runpod",
        "provider_contract": None,
        "resource": None,
        "allocation": None,
        "progress_receipt": None,
        "result_receipt": None,
        "cleanup": {},
        "schema_version": 1,
    }
    status = provisioned_status(
        spec,
        state="running",
        created_at=100.0,
        attempt=attempt,
        remote=remote,
    )
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(
        status,
        _run_deadline_at=300.0,
        _next_attempt=2,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 100.0)
    poll_walls = []

    class FailedProvider:
        def poll(self, _handle, poll_spec, *_args, **_kwargs):
            poll_walls.append(poll_spec.gpu.max_wall_seconds)
            return PollResult(False, failure="job_preempted", detail="worker stopped")

        def cancel(self, _handle):
            return None

        def destroy(self, _handle):
            return None

    monkeypatch.setattr(providers, "get_provider", lambda _name: FailedProvider())
    monkeypatch.setattr(runner_lifecycle, "_attempt_result", lambda *a, **k: None)
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    resumed = []

    def fake_run_training(_spec, _log, **kwargs):
        resumed.append(kwargs["attempt_start"])

    monkeypatch.setattr(runner_lifecycle, "_run_training", fake_run_training)

    status = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert resumed == [expected_next]
    assert poll_walls == [200]
    assert status.state == "running"
    assert status.remote is None


def test_attach_expired_run_adopts_current_fenced_result_at_deadline(monkeypatch, tmp_path):
    import io

    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(
        run_id="attach-expired-completed",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(max_wall_seconds=120),
    )
    remote = _vast_remote(instance_id=7, attempt=0, started_ts=101.0)
    status = provisioned_status(spec, state="running", created_at=100.0, remote=remote)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(
        status,
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 221.0)
    result_checks = []

    def current_result(run_id, handle):
        result_checks.append((run_id, handle))
        return PollResult(True, metrics=_fenced_success_metrics(spec))

    monkeypatch.setattr(lifecycle, "_attempt_result", current_result)
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(lifecycle, "_charge_completed_run_by_id", lambda *_args: None)
    monkeypatch.setattr(lifecycle, "_register_checkpoints_best_effort", lambda *_args: None)
    monkeypatch.setattr(
        lifecycle,
        "_strict_teardown_handle",
        lambda *_args, **_kwargs: pytest.fail(
            "current-fence success must not be torn down before adoption"
        ),
    )
    log = io.StringIO()

    status = runner_attach.attach_run(spec.run_id, log_stream=log)

    assert result_checks == [(spec.run_id, remote)]
    assert status.state == "done"
    assert status.remote == remote
    assert status.error is None
    assert runner_status._load_status_json(spec.run_id)[runner_state._CLEANUP_REMOTES_KEY] == [
        remote
    ]
    assert "adopted a completed attempt at the wall deadline" in log.getvalue()


def test_attach_oom_retry_plan_escalates_and_reconstructs_budget(monkeypatch):
    import flash.runner.supervise.attach as attach
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core.base import JobHandle

    monkeypatch.setattr(
        runner_deadlines,
        "_spec_with_remaining_wall",
        lambda spec, **_kwargs: spec,
    )
    spec = JobSpec(
        run_id="attach-oom-plan",
        model="Qwen/Qwen3.5-9B",
        gpu=GpuSpec(max_retries=2),
    )
    remote = _vast_remote(
        instance_id=7,
        attempt=0,
        allocated_gpu="A100 SXM 40GB",
        allocated_gpu_count=1,
    )
    context = attach._AttachContext(
        worker_spec=spec,
        persisted_remote=remote,
        handle=JobHandle.from_dict(remote),
        seed=spec.seed,
        recovered_attempt=0,
        next_attempt=1,
        source_snapshot=_SOURCE_SNAPSHOT,
    )

    retry_budget, oom_vram_floor, drop_cache = attach._attach_retry_plan(context, "oom")

    assert retry_budget.oom_used == 1
    assert retry_budget.infra_used == 0
    assert oom_vram_floor == pytest.approx(40.0)
    assert drop_cache is False
    exhausted = replace(
        context,
        recovered_attempt=2,
        next_attempt=3,
        retry_counters={"infra": 0, "oom": 2, "cache": 0},
    )
    assert attach._attach_retry_plan(exhausted, "oom") is None


def test_attach_mixed_retry_history_reconstructs_exact_categories(monkeypatch):
    import flash.runner.supervise.attach as attach
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core.base import JobHandle

    monkeypatch.setattr(
        runner_deadlines,
        "_spec_with_remaining_wall",
        lambda spec, **_kwargs: spec,
    )
    spec = JobSpec(
        run_id="attach-mixed-retries",
        model="Qwen/Qwen3.5-9B",
        gpu=GpuSpec(max_retries=2),
    )
    remote = _vast_remote(
        instance_id=7,
        attempt=7,
        allocated_gpu="A100 SXM 40GB",
        allocated_gpu_count=1,
    )
    context = attach._AttachContext(
        worker_spec=spec,
        persisted_remote=remote,
        handle=JobHandle.from_dict(remote),
        seed=spec.seed,
        recovered_attempt=7,
        next_attempt=8,
        source_snapshot=_SOURCE_SNAPSHOT,
        retry_counters={"infra": 1, "oom": 1, "cache": 0},
    )

    retry_budget, _oom_vram_floor, drop_cache = attach._attach_retry_plan(context, "oom")

    assert retry_budget.infra_used == 1
    assert retry_budget.oom_used == 2
    assert retry_budget.cache_used == 0
    assert drop_cache is False
    assert (
        attach._attach_retry_plan(
            replace(context, retry_counters=retry_budget.counters()),
            "oom",
        )
        is None
    )


def test_retry_counters_persist_atomically_with_remote_clear_and_reload(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="retry-counter-clear", model="Qwen/Qwen3.5-9B")
    remote = _vast_remote(instance_id=7, attempt=2, fence=9)
    status = provisioned_status(spec, state="running", created_at=100.0, remote=remote)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(status, _run_deadline_at=400.0, _next_attempt=3)

    counters = {"infra": 1, "oom": 1, "cache": 1}
    assert runner_reconciliation._compare_and_clear_remote(
        spec.run_id,
        remote,
        retry_counters=counters,
    )

    reloaded = runner_status.get_status(spec.run_id)
    assert reloaded.remote is None
    assert reloaded.retry_counters == counters
    assert "retry_counters" not in reloaded.to_dict()


def test_attach_oom_floor_uses_executed_width_not_rented_count(monkeypatch):
    import flash.runner.supervise.attach as attach
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core.base import JobHandle

    monkeypatch.setattr(
        runner_deadlines,
        "_spec_with_remaining_wall",
        lambda spec, **_kwargs: spec,
    )
    spec = JobSpec(
        run_id="attach-executed-width",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(batch_size=1, max_examples=1),
        gpu=GpuSpec(max_retries=1),
    )
    remote = _vast_remote(
        instance_id=7,
        attempt=0,
        allocated_gpu="A100 SXM 40GB",
        allocated_gpu_count=4,
    )
    context = attach._AttachContext(
        worker_spec=spec,
        persisted_remote=remote,
        handle=JobHandle.from_dict(remote),
        seed=spec.seed,
        recovered_attempt=0,
        next_attempt=1,
        source_snapshot=_SOURCE_SNAPSHOT,
    )

    _retry_budget, oom_vram_floor, drop_cache = attach._attach_retry_plan(context, "oom")

    assert oom_vram_floor == pytest.approx(40.0)
    assert drop_cache is False


def test_attach_reconstructs_cache_fallback_without_spending_infra(monkeypatch):
    import flash.runner.supervise.attach as attach
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core.base import JobHandle
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    monkeypatch.setattr(
        runner_deadlines,
        "_spec_with_remaining_wall",
        lambda spec, **_kwargs: spec,
    )
    monkeypatch.setattr(
        "flash.providers.core.registry.get_provider",
        lambda _name: SimpleNamespace(supports_weight_cache=True),
    )
    spec = JobSpec(
        run_id="attach-cache-fallback",
        model="Qwen/Qwen3.5-9B",
        gpu=GpuSpec(max_retries=1, network_volume=WEIGHT_CACHE_VOLUME_NAME),
    )
    remote = _vast_remote(instance_id=7, attempt=0)
    context = attach._AttachContext(
        worker_spec=spec,
        persisted_remote=remote,
        handle=JobHandle.from_dict(remote),
        seed=spec.seed,
        recovered_attempt=0,
        next_attempt=1,
        source_snapshot=_SOURCE_SNAPSHOT,
        retry_counters={"infra": 0, "oom": 0, "cache": 0},
    )

    retry_budget, oom_vram_floor, drop_cache = attach._attach_retry_plan(context, "poll_error")

    assert retry_budget.infra_used == 0
    assert retry_budget.cache_used == 1
    assert oom_vram_floor == 0.0
    assert drop_cache is True


def test_attach_artifact_transport_uses_infra_budget_and_single_shot_exhausts(monkeypatch):
    import flash.runner.supervise.attach as attach
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core.base import JobHandle

    monkeypatch.setattr(
        runner_deadlines,
        "_spec_with_remaining_wall",
        lambda spec, **_kwargs: spec,
    )
    remote = _vast_remote(instance_id=7, attempt=0)
    retrying_spec = JobSpec(
        run_id="attach-artifact-plan",
        model="Qwen/Qwen3.5-9B",
        gpu=GpuSpec(max_retries=1),
    )
    context = attach._AttachContext(
        worker_spec=retrying_spec,
        persisted_remote=remote,
        handle=JobHandle.from_dict(remote),
        seed=retrying_spec.seed,
        recovered_attempt=0,
        next_attempt=1,
        source_snapshot=_SOURCE_SNAPSHOT,
    )

    retry_budget, oom_vram_floor, drop_cache = attach._attach_retry_plan(
        context, "artifact_transport"
    )

    assert retry_budget.infra_used == 1
    assert retry_budget.oom_used == 0
    assert oom_vram_floor == 0.0
    assert drop_cache is False
    single_shot = replace(
        context,
        worker_spec=replace(
            retrying_spec,
            gpu=replace(retrying_spec.gpu, max_retries=0),
        ),
    )
    assert attach._attach_retry_plan(single_shot, "artifact_transport") is None


def test_attach_authoritative_retry_orders_teardown_cleanup_clear_then_replacement(monkeypatch):
    import flash.runner.supervise.attach as attach
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core.base import JobHandle

    spec = JobSpec(
        run_id="attach-order",
        model="Qwen/Qwen3.5-9B",
        gpu=GpuSpec(max_retries=1),
    )
    remote = _vast_remote(instance_id=7, attempt=0, allocated_gpu="RTX 4090")
    context = attach._AttachContext(
        worker_spec=spec,
        persisted_remote=remote,
        handle=JobHandle.from_dict(remote),
        seed=spec.seed,
        recovered_attempt=0,
        next_attempt=1,
        source_snapshot=_SOURCE_SNAPSHOT,
    )
    events = []
    budget = lifecycle._new_retry_budget(1)
    lifecycle._failure_disposition(budget, "oom")
    monkeypatch.setattr(attach, "_attach_retry_plan", lambda *_args: (budget, 24.0, False))
    monkeypatch.setattr(
        lifecycle,
        "_strict_teardown_handle",
        lambda *_args: events.append("teardown") or False,
    )
    monkeypatch.setattr(lifecycle, "_worker_provably_gone", lambda *_args: True)
    monkeypatch.setattr(
        runner_reconciliation,
        "_record_cleanup_remote",
        lambda *_args: events.append("cleanup") or True,
    )

    def resume(*_args, **kwargs):
        events.extend(("cas-clear", "replacement"))
        assert kwargs["retry_budget"] is budget
        assert kwargs["oom_vram_floor"] == 24.0
        return SimpleNamespace(state="running")

    monkeypatch.setattr(attach, "_resume_after_confirmed_teardown", resume)

    observed = attach._dispose_authoritative_failure(
        spec.run_id,
        context,
        PollResult(False, failure="oom", detail="cuda out of memory"),
        io.StringIO(),
    )

    assert observed.state == "running"
    assert events == ["teardown", "cleanup", "cas-clear", "replacement"]


def test_attach_failed_current_fence_result_tears_down_and_persists_cleanup_before_terminal(
    monkeypatch, tmp_path
):
    import io

    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-fenced-failure",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(hf_repo="org/repo"),
        gpu=GpuSpec(max_wall_seconds=120),
    )
    remote = _vast_remote(instance_id=7, attempt=0, started_ts=101.0)
    status = provisioned_status(spec, state="running", created_at=100.0, remote=remote)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(status, _run_deadline_at=220.0, _next_attempt=1)
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 221.0)
    monkeypatch.setattr(
        lifecycle,
        "_attempt_result",
        lambda *_args, **_kwargs: PollResult(
            False,
            failure="oom",
            detail="cuda out of memory",
        ),
    )
    events = []
    monkeypatch.setattr(
        lifecycle,
        "_strict_teardown_handle",
        lambda *_args, **_kwargs: events.append("teardown") or False,
    )
    monkeypatch.setattr(lifecycle, "_worker_provably_gone", lambda *_args: True)
    real_cleanup = runner_reconciliation._record_cleanup_remote

    def record_cleanup(*args):
        events.append("cleanup")
        return real_cleanup(*args)

    monkeypatch.setattr(runner_reconciliation, "_record_cleanup_remote", record_cleanup)
    real_fail = runner_reconciliation._compare_and_fail_remote

    def terminalize(*args):
        events.append("terminal")
        return real_fail(*args)

    monkeypatch.setattr(runner_reconciliation, "_compare_and_fail_remote", terminalize)
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)

    observed = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert events == ["teardown", "cleanup", "terminal"]
    assert observed.state == "failed"
    assert observed.error == "oom: cuda out of memory"
    assert observed.remote == remote
    assert runner_status._current_attempt(observed).state == "settled"


def test_attach_authoritative_failure_stays_nonterminal_when_absence_is_uncertain(monkeypatch):
    import flash.runner.supervise.attach as attach
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core.base import JobHandle

    spec = JobSpec(
        run_id="attach-uncertain-failure",
        model="Qwen/Qwen3.5-9B",
        gpu=GpuSpec(max_retries=0),
    )
    remote = _vast_remote(instance_id=7, attempt=0)
    context = attach._AttachContext(
        worker_spec=spec,
        persisted_remote=remote,
        handle=JobHandle.from_dict(remote),
        seed=spec.seed,
        recovered_attempt=0,
        next_attempt=1,
        source_snapshot=_SOURCE_SNAPSHOT,
    )
    events = []
    monkeypatch.setattr(attach, "_attach_retry_plan", lambda *_args: None)
    monkeypatch.setattr(
        lifecycle,
        "_strict_teardown_handle",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("uncertain teardown")),
    )
    monkeypatch.setattr(lifecycle, "_worker_provably_gone", lambda *_args: False)
    monkeypatch.setattr(
        runner_reconciliation,
        "_record_cleanup_remote",
        lambda *_args: events.append("cleanup") or True,
    )
    monkeypatch.setattr(
        runner_reconciliation,
        "_compare_and_fail_remote",
        lambda *_args: pytest.fail("uncertain worker absence must block terminalization"),
    )

    observed = attach._dispose_authoritative_failure(
        spec.run_id,
        context,
        PollResult(False, failure="worker", detail="worker failed"),
        io.StringIO(),
    )

    assert observed is None
    assert events == ["cleanup"]


def test_attach_adoption_prices_a_multi_card_run_for_every_card(monkeypatch, tmp_path):
    # current-fence result metrics do not carry provider allocation metadata, so attach recovery must
    # restore the exact persisted provider, gpu class, and card count before terminal adoption.
    import io

    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(
        run_id="attach-adopt-multicard",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(max_wall_seconds=120),
    )
    remote = _vast_remote(
        instance_id=7,
        attempt=0,
        started_ts=101.0,
        allocated_gpu="RTX 4090",
        allocated_gpu_count=4,
    )
    status = provisioned_status(spec, state="running", created_at=100.0, remote=remote)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(
        status,
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 221.0)
    monkeypatch.setattr(
        lifecycle,
        "_attempt_result",
        lambda *_args, **_kwargs: PollResult(
            True,
            metrics=_fenced_success_metrics(spec, wall_seconds=3600.0),
        ),
    )
    adopted = {}
    real_adopt = lifecycle._adopt_completed_attempt

    def capture_adopt(run_id, adopt_spec, expected_remote, metrics, **kwargs):
        adopted.update(metrics)
        return real_adopt(run_id, adopt_spec, expected_remote, metrics, **kwargs)

    monkeypatch.setattr(lifecycle, "_adopt_completed_attempt", capture_adopt)
    monkeypatch.setattr(lifecycle, "_charge_completed_run_by_id", lambda *_args: None)
    monkeypatch.setattr(lifecycle, "_register_checkpoints_best_effort", lambda *_args: None)
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)

    status = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert status.state == "done"
    assert adopted["allocated_gpu_count"] == 4, (
        "an adopted multi-card run reached persistence with no card count, so its wall prices "
        "as a single card"
    )
    assert adopted["allocated_gpu"] == "RTX 4090"
    # and the substrate that billed it. _gpu_rate falls back to whichever configured provider
    # offers the class, so without this the run is priced on a provider that never ran it.
    assert adopted["allocated_provider"] == "vast", (
        "an adopted vast run reached persistence with no provider, so it is priced on whichever "
        "provider the plane happens to try first"
    )


def test_attach_poll_success_carries_the_whole_allocation_stamp(monkeypatch):
    # the OTHER attach exit: a provider poll that returns ok. the wall-deadline route above carries
    # all three fields through `_carry_allocation_stamp`, but this one restored only gpu and count
    # from the context -- so a vast or lambda run that simply finished its poll was priced by
    # `_gpu_rate`'s fallback (normally RunPod) and its notes named a provider that never ran it.
    import io
    from types import SimpleNamespace

    import flash.runner.supervise.attach as attach
    from flash.providers.core.base import JobHandle

    remote = _vast_remote(allocated_gpu="RTX 4090", allocated_gpu_count=4)
    context = attach._AttachContext(
        worker_spec=None,
        persisted_remote=remote,
        handle=JobHandle.from_dict({"provider": "vast", "instance_id": 7}),
        seed=0,
        recovered_attempt=0,
        next_attempt=1,
        source_snapshot=None,
    )
    result = SimpleNamespace(ok=True, metrics={"wall_seconds": 3600.0})
    adopted = {}
    # `_adopt_attached_poll_result` imports the adopter from its owning module at call time, so the
    # patch has to land there rather than on a name bound into `attach`.
    import flash.runner.supervise.lifecycle as lifecycle_mod

    monkeypatch.setattr(
        lifecycle_mod,
        "_adopt_completed_attempt",
        lambda *_a, **_k: adopted.update(result.metrics) or True,
    )

    attach._adopt_attached_poll_result("attach-poll-multicard", context, result, io.StringIO())

    assert adopted["allocated_gpu"] == "RTX 4090"
    assert adopted["allocated_gpu_count"] == 4
    assert adopted["allocated_provider"] == "vast", (
        "a polled vast run reached persistence with no provider, so its wall is priced at "
        "whichever provider offers the class rather than the one that billed it"
    )


def test_attach_transient_result_read_stops_at_persisted_deadline(monkeypatch):
    import flash.runner.supervise.attach as attach
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.providers.core.base import JobHandle

    remote = _vast_remote(instance_id=7, attempt=0)
    handle = JobHandle.from_dict(remote)
    sleeps = []
    monkeypatch.setattr(attach.time, "time", lambda: 99.0)
    monkeypatch.setattr(attach.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        lifecycle,
        "_strict_teardown_handle",
        lambda *_args: pytest.fail("a pre-deadline transient read must not tear down"),
    )

    assert attach._defer_transient_result("run-deadline", remote, handle, 100.0) is False
    assert sleeps == [1.0]

    events = []
    monkeypatch.setattr(attach.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        lifecycle,
        "_strict_teardown_handle",
        lambda *_args: events.append("teardown") or True,
    )
    monkeypatch.setattr(
        runner_reconciliation,
        "_record_cleanup_remote",
        lambda *_args: events.append("cleanup") or True,
    )
    monkeypatch.setattr(
        runner_reconciliation,
        "_compare_and_fail_remote",
        lambda *_args: events.append("terminal") or True,
    )

    assert attach._defer_transient_result("run-deadline", remote, handle, 100.0) is True
    assert events == ["teardown", "cleanup", "terminal"]


def test_attach_normal_and_expired_reconciliation_bound_transient_result_reads(monkeypatch):
    import flash.runner.supervise.attach as attach
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import JobSpec
    from flash.providers.core.base import JobHandle
    from flash.runner.lifecycle.protocol import AttemptRecord

    spec = JobSpec(run_id="expired-result-read", model="Qwen/Qwen3.5-9B")
    remote = _vast_remote(instance_id=7, attempt=0)
    attempt = AttemptRecord(0, 1, "active", 1.0, 2.0, 3.0, 5.0, 4.0)
    status = SimpleNamespace(state="running", remote=remote, attempt=attempt.to_dict())
    monkeypatch.setattr(runner_status, "get_status", lambda _run_id: status)
    monkeypatch.setattr(runner_deadlines, "_load_run_deadline_at", lambda _run_id: 4.0)
    monkeypatch.setattr(
        lifecycle,
        "_attempt_result",
        lambda *_args: (_ for _ in ()).throw(OSError("temporary result outage")),
    )
    observed = []
    monkeypatch.setattr(
        attach,
        "_defer_transient_result",
        lambda run_id, expected, handle, deadline: (
            observed.append((run_id, expected, handle.to_dict(), deadline)) or True
        ),
    )

    handle = JobHandle.from_dict(remote)
    assert (
        attach._reconcile_expired_remote(
            spec.run_id,
            spec,
            remote,
            handle,
            1,
            _SOURCE_SNAPSHOT,
            4.0,
            io.StringIO(),
            "deadline",
        )
        is True
    )
    attach._reconcile_attached_remote(
        spec.run_id,
        remote,
        spec,
        1,
        _SOURCE_SNAPSHOT,
        io.StringIO(),
        "deadline",
    )
    assert observed == [
        (spec.run_id, remote, handle.to_dict(), 5.0),
        (spec.run_id, remote, handle.to_dict(), 5.0),
    ]


def test_attach_transient_current_fence_result_read_stays_pending(monkeypatch, tmp_path):
    import io

    import flash.runner.supervise.attach as attach
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-result-pending",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(hf_repo="org/repo"),
        gpu=GpuSpec(max_wall_seconds=120),
    )
    remote = _vast_remote(instance_id=7, attempt=0, started_ts=101.0)
    status = provisioned_status(spec, state="running", created_at=100.0, remote=remote)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(
        status,
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 221.0)
    monkeypatch.setattr(
        lifecycle,
        "_attempt_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("temporary artifact outage")),
    )
    scheduled = []
    monkeypatch.setattr(
        attach,
        "_schedule_attach_reconciliation",
        lambda *args, **kwargs: scheduled.append((args, kwargs)) or True,
    )

    status = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert status.state == "running"
    assert status.remote == remote
    assert len(scheduled) == 1


def test_failed_attach_poll_defers_transient_result_transport(monkeypatch):
    import io
    from types import SimpleNamespace

    import flash.runner.supervise.attach as attach
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.providers.core.base import JobHandle, PollResult

    remote = _runpod_remote("endpoint-transient", "job-transient", attempt=0)
    context = attach._AttachContext(
        worker_spec=object(),
        persisted_remote=remote,
        handle=JobHandle.from_dict(remote),
        seed=0,
        recovered_attempt=0,
        next_attempt=1,
        source_snapshot=_SOURCE_SNAPSHOT,
    )
    current = SimpleNamespace(state="running", remote=remote)
    monkeypatch.setattr(
        lifecycle,
        "_attempt_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("temporary result outage")),
    )
    monkeypatch.setattr(runner_status, "get_status", lambda _run_id: current)
    scheduled = []
    monkeypatch.setattr(
        attach,
        "_schedule_attach_reconciliation",
        lambda *args, **kwargs: scheduled.append((args, kwargs)) or True,
    )
    monkeypatch.setattr(
        lifecycle,
        "_strict_teardown_handle",
        lambda *_args, **_kwargs: pytest.fail(
            "transient result transport must not trigger teardown"
        ),
    )

    observed = attach._handle_failed_attach_poll(
        "run-transient",
        context,
        PollResult(False, failure="job_preempted", detail="provider ended"),
        io.StringIO(),
    )

    assert observed is current
    assert observed.state == "running"
    assert observed.remote == remote
    assert len(scheduled) == 1


def test_attached_no_capacity_drops_shared_cache_without_spending_infra(monkeypatch):
    import flash.runner.supervise.attach as attach
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core.base import JobHandle
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    spec = JobSpec(
        run_id="attach-no-capacity-cache",
        model="Qwen/Qwen3.5-9B",
        gpu=GpuSpec(max_retries=1, network_volume=WEIGHT_CACHE_VOLUME_NAME),
    )
    remote = _vast_remote(instance_id=7, attempt=0)
    context = attach._AttachContext(
        worker_spec=spec,
        persisted_remote=remote,
        handle=JobHandle.from_dict(remote),
        seed=spec.seed,
        recovered_attempt=0,
        next_attempt=1,
        source_snapshot=_SOURCE_SNAPSHOT,
        retry_counters={"infra": 0, "oom": 0, "cache": 0},
    )
    current = SimpleNamespace(state="running", remote=remote)
    monkeypatch.setattr(lifecycle, "_attempt_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lifecycle, "_strict_teardown_handle", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(runner_status, "get_status", lambda _run_id: current)
    monkeypatch.setattr(
        runner_deadlines, "_spec_with_remaining_wall", lambda value, **_kwargs: value
    )
    monkeypatch.setattr(
        "flash.providers.core.registry.get_provider",
        lambda _name: SimpleNamespace(supports_weight_cache=True),
    )
    resumed = []
    monkeypatch.setattr(
        attach,
        "_resume_after_confirmed_teardown",
        lambda *args, **kwargs: resumed.append((args, kwargs)) or current,
    )

    observed = attach._handle_failed_attach_poll(
        spec.run_id,
        context,
        PollResult(False, failure="no_capacity", detail="shared cache unavailable"),
        io.StringIO(),
    )

    assert observed is current
    assert len(resumed) == 1
    retry_budget = resumed[0][1]["retry_budget"]
    assert resumed[0][1]["drop_weight_cache"] is True
    assert retry_budget.counters() == {"infra": 0, "oom": 0, "cache": 1}


def test_attach_invalid_current_fence_result_fails_closed(monkeypatch, tmp_path):
    import io

    import flash.runner.supervise.attach as attach
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.artifacts.attempts import AttemptArtifactError

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-result-invalid",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(hf_repo="org/repo"),
        gpu=GpuSpec(max_wall_seconds=120),
    )
    remote = _vast_remote(instance_id=7, attempt=0, started_ts=101.0)
    status = provisioned_status(spec, state="running", created_at=100.0, remote=remote)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(
        status,
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 221.0)
    monkeypatch.setattr(
        lifecycle,
        "_attempt_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AttemptArtifactError("result manifest is invalid or unverifiable")
        ),
    )
    monkeypatch.setattr(
        attach,
        "_schedule_attach_reconciliation",
        lambda *_args, **_kwargs: pytest.fail("invalid artifacts must not be retried as transport"),
    )
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)

    status = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert status.state == "failed"
    assert status.remote == remote
    assert "invalid or unverifiable" in (status.error or "")


def test_attach_expired_run_does_not_poll_or_resubmit(monkeypatch, tmp_path):
    import io

    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-expired",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=120),
    )
    runner_state._save_status(
        provisioned_status(
            spec,
            state="running",
            created_at=100.0,
            remote=_runpod_remote("endpoint-old", "job-old", attempt=0),
        ),
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 221.0)
    polled = []
    resumed = []
    teardown = []
    gc_runs = []
    result_checks = []
    monkeypatch.setattr(
        lifecycle,
        "_attempt_result",
        lambda *args, **kwargs: result_checks.append((args, kwargs)) or None,
    )

    class Provider:
        def poll(self, *_args, **_kwargs):
            polled.append(True)
            raise AssertionError("expired recovery must not poll")

    def teardown_handle(handle, _run_id):
        teardown.extend((("cancel", handle.to_dict()), ("destroy", handle.to_dict())))
        return True

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(lifecycle, "_strict_teardown_handle", teardown_handle)
    monkeypatch.setattr(
        runner_recovery,
        "_gc_run_endpoints",
        lambda cleanup_spec: gc_runs.append(cleanup_spec.run_id),
    )
    monkeypatch.setattr(
        runner_lifecycle, "_run_training", lambda *_args, **_kwargs: resumed.append(True)
    )

    status = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert polled == []
    assert resumed == []
    assert result_checks == [((spec.run_id, status.remote), {})]
    assert [action for action, _handle in teardown] == ["cancel", "destroy"]
    assert gc_runs == [spec.run_id]
    assert status.state == "failed"
    assert status.remote is None
    assert status.cleanup_confirmed_remote["endpoint_id"] == "endpoint-old"
    assert status.realized_cost_remote["endpoint_id"] == "endpoint-old"
    assert "deadline exhausted" in status.error


def test_attach_expired_run_retains_handle_when_teardown_is_unconfirmed(monkeypatch, tmp_path):
    import io

    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-expired-unconfirmed",
        model="Qwen/Qwen3.5-9B",
        gpu=GpuSpec(max_wall_seconds=120),
    )
    remote = _runpod_remote("endpoint-old", "job-old", attempt=0)
    persisted = provisioned_status(spec, state="running", created_at=100.0, remote=remote)
    persisted.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(
        persisted,
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 221.0)
    monkeypatch.setattr(runner_lifecycle, "_attempt_result", lambda *_args: None)
    monkeypatch.setattr(
        runner_lifecycle,
        "_strict_teardown_handle",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("teardown unconfirmed")),
    )

    class Provider:
        def cancel(self, _handle):
            return None

        def destroy(self, _handle):
            raise RuntimeError("teardown unconfirmed")

        def poll(self, *_args, **_kwargs):
            raise AssertionError("expired recovery must not poll")

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    resumed = []
    monkeypatch.setattr(
        runner_lifecycle, "_run_training", lambda *_args, **_kwargs: resumed.append(True)
    )

    status = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert resumed == []
    assert status.state == "failed"
    assert status.remote == remote
    assert "deadline exhausted" in status.error


def test_runpod_submit_propagates_attempt_to_worker_environment_and_handle(monkeypatch, tmp_path):
    import flash.providers._lifecycle.net.worker as train
    import flash.providers.runpod.execution.job_execution as job_execution
    import flash.providers.runpod.execution.polling as polling
    from flash.core.spec import JobSpec
    from flash.providers.core.base import PollResult
    from flash.runner.lifecycle.protocol import AttemptRecord

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="worker-attempt", model="Qwen/Qwen3.5-9B", algorithm="sft")
    deadline = 10_000_000_000.0
    attempt = AttemptRecord(2, 3, "reserved", 1.0, 2.0, deadline, deadline + 120, deadline)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="provisioning",
            spec=spec.to_dict(),
            attempt=attempt.to_dict(),
        )
    )
    payloads = []
    handles = []
    monkeypatch.setattr(train, "build_worker_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        job_execution,
        "deploy_train_endpoint",
        lambda *_args, **_kwargs: ("endpoint", "name", _RUNPOD_FINGERPRINT),
    )
    monkeypatch.setattr(
        job_execution.runpod_api,
        "submit_job",
        lambda _endpoint, payload, **_kwargs: payloads.append(payload) or "job",
    )
    monkeypatch.setattr(
        polling,
        "poll_job",
        lambda *_args, **_kwargs: PollResult(True, metrics={"wall_seconds": 1.0}),
    )

    job_execution.submit_run(
        spec,
        0,
        attempt=2,
        fence=3,
        source_snapshot=_SOURCE_SNAPSHOT,
        on_handle=handles.append,
        deadline_at=deadline,
    )

    assert payloads[0]["env"]["ATTEMPT"] == "2"
    assert payloads[0]["env"]["FENCE"] == "3"
    assert handles[0]["attempt"] == 2
    assert handles[0]["fence"] == 3


def test_fail_blocked_recovery_adopts_completed_handleless_attempt(monkeypatch, tmp_path):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(run_id="blocked-complete", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        _handleless_status(spec, created_at=100.0),
        _run_deadline_at=86500.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(
        runtime,
        "_handleless_attempt_result",
        lambda *_args, **_kwargs: PollResult(True, metrics=_fenced_success_metrics(spec)),
    )

    assert runtime._fail_blocked_recovery(spec, "recovery blocked") is True

    status = runner_status.get_status(spec.run_id)
    assert status.state == "done"
    assert status.remote is None
    assert status.error is None


def test_fail_blocked_recovery_keeps_transient_result_read_pending(monkeypatch, tmp_path):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="blocked-pending", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        _handleless_status(spec),
        _next_attempt=1,
    )
    monkeypatch.setattr(runtime.time, "time", lambda: 200.0)
    monkeypatch.setattr(
        runtime,
        "_handleless_attempt_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("temporary artifact outage")),
    )

    assert runtime._fail_blocked_recovery(spec, "recovery blocked") is False
    status = runner_status.get_status(spec.run_id)
    assert status.state == "provisioning"
    assert status.remote is None
    assert status.error is None


def test_fail_blocked_recovery_settles_post_deadline_transient_result_transport(
    monkeypatch, tmp_path
):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="blocked-expired", model="Qwen/Qwen3.5-9B", algorithm="sft")
    status = _handleless_status(spec, created_at=100.0)
    attempt = runner_status._current_attempt(status)
    runner_state._save_status(status, _next_attempt=1)
    monkeypatch.setattr(runtime.time, "time", lambda: attempt.result_deadline_at)
    monkeypatch.setattr(
        runtime,
        "_handleless_attempt_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("temporary artifact outage")),
    )
    monkeypatch.setattr(
        runtime.time,
        "sleep",
        lambda _seconds: pytest.fail("post-deadline result transport must not sleep"),
    )
    captured = []
    real_fail = runner_reconciliation._compare_and_fail_remote

    def fail_exact(run_id, remote, error, **identity):
        captured.append((remote, identity))
        return real_fail(run_id, remote, error, **identity)

    monkeypatch.setattr(runner_reconciliation, "_compare_and_fail_remote", fail_exact)

    assert runtime._fail_blocked_recovery(spec, "recovery blocked") is True

    settled = runner_status.get_status(spec.run_id)
    assert captured == [
        (
            None,
            {"expected_attempt_id": attempt.attempt_id, "expected_fence": attempt.fence},
        )
    ]
    assert settled.state == "failed"
    assert runner_status._current_attempt(settled).state == "settled"


def test_start_resubmit_deadline_adopts_completed_handleless_attempt(monkeypatch, tmp_path):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(
        run_id="deadline-complete",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=120),
    )
    runner_state._save_status(
        _handleless_status(spec, created_at=100.0),
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 221.0)
    monkeypatch.setattr(
        runtime,
        "_handleless_attempt_result",
        lambda *_args, **_kwargs: PollResult(True, metrics=_fenced_success_metrics(spec)),
    )

    assert runtime._start_resubmit(spec, expected_remote=None) is False

    status = runner_status.get_status(spec.run_id)
    assert status.state == "done"
    assert status.remote is None
    assert status.error is None


def test_start_resubmit_adopts_handleless_result_before_open_deadline(monkeypatch, tmp_path):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(run_id="open-complete", model="Qwen/Qwen3.5-9B", algorithm="sft")
    status = _handleless_status(spec, created_at=100.0)
    runner_state._save_status(status, _run_deadline_at=86500.0, _next_attempt=1)
    monkeypatch.setattr(runtime.time, "time", lambda: 200.0)
    monkeypatch.setattr(
        runtime,
        "_handleless_attempt_result",
        lambda *_args, **_kwargs: PollResult(True, metrics=_fenced_success_metrics(spec)),
    )
    monkeypatch.setattr(
        runner_reconciliation,
        "_compare_and_prepare_resubmit",
        lambda *_args, **_kwargs: pytest.fail("verified original result must prevent resubmit cas"),
    )

    assert (
        runtime._start_resubmit(
            spec,
            expected_remote=None,
            expected_state="provisioning",
        )
        is False
    )

    adopted = runner_status.get_status(spec.run_id)
    assert adopted.state == "done"
    assert adopted.remote is None


def test_start_resubmit_fails_authoritative_handleless_result_without_replacement(
    monkeypatch, tmp_path
):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="handleless-failure", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        _handleless_status(spec, created_at=100.0),
        _run_deadline_at=86500.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(
        runtime,
        "_handleless_attempt_result",
        lambda *_args, **_kwargs: PollResult(
            False,
            failure="job_failed",
            detail="checkpoint write failed",
        ),
    )
    monkeypatch.setattr(
        runner_reconciliation,
        "_compare_and_prepare_resubmit",
        lambda *_args, **_kwargs: pytest.fail(
            "authoritative failed result must prevent replacement"
        ),
    )

    assert not runtime._start_resubmit(
        spec,
        expected_remote=None,
        expected_state="provisioning",
    )

    status = runner_status.get_status(spec.run_id)
    assert status.state == "failed"
    assert status.error == "job_failed: checkpoint write failed"
    assert status.remote is None
    assert runner_status._current_attempt(status).state == "settled"


@pytest.mark.parametrize(
    ("failure", "expected_counters", "expected_floor"),
    [
        ("oom", {"infra": 0, "oom": 1, "cache": 0}, 40.0),
        ("artifact_transport", {"infra": 1, "oom": 0, "cache": 0}, 0.0),
    ],
)
def test_handleless_authoritative_failure_retries_after_confirmed_absence(
    monkeypatch, tmp_path, failure, expected_counters, expected_floor
):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id=f"handleless-retry-{failure}",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(type="A100 SXM 40GB", max_retries=1),
    )
    status = _handleless_status(spec, created_at=100.0)
    runner_state._save_status(status, _next_attempt=1, _next_fence=2)
    monkeypatch.setattr(runtime, "_confirm_run_clear", lambda _spec: True)
    monkeypatch.setattr(runtime, "_recovery_block_reason", lambda _spec: None)
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    monkeypatch.setattr(
        runtime,
        "_handleless_attempt_result",
        lambda *_args, **_kwargs: PollResult(False, failure=failure, detail="worker failed"),
    )
    import flash.runner.supervise.lifecycle as lifecycle

    launched = []
    monkeypatch.setattr(
        lifecycle,
        "_run_job_background",
        lambda *args: launched.append(args),
    )

    class Thread:
        def __init__(self, *, target, daemon):
            self.target = target
            self.daemon = daemon

        def start(self):
            self.target()

    monkeypatch.setattr(runtime.threading, "Thread", Thread)

    runtime._resubmit_recovered_runs([(spec, "provisioning")])

    current = runner_status.get_status(spec.run_id)
    assert current.state == "provisioning"
    assert current.retry_counters == expected_counters
    assert runner_status._current_attempt(current).attempt_id == 0
    assert runner_status._current_attempt(current).state == "settling"
    assert runner_status.get_status(spec.run_id).retry_counters == expected_counters
    assert launched == [(spec, None, 1, pytest.approx(expected_floor))]


def test_handleless_retry_claim_schedules_only_one_active_supervisor(monkeypatch, tmp_path):
    import flash.runner.supervise.lifecycle as lifecycle
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="handleless-single-launch",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=1),
    )
    runner_state._save_status(
        _handleless_status(spec, created_at=100.0),
        _next_attempt=1,
        _next_fence=2,
    )
    monkeypatch.setattr(runtime, "_recovery_block_reason", lambda _spec: None)
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    monkeypatch.setattr(
        runtime,
        "_handleless_attempt_result",
        lambda *_args, **_kwargs: PollResult(
            False,
            failure="artifact_transport",
            detail="worker failed",
        ),
    )
    monkeypatch.setattr(lifecycle, "_run_job_background", lambda *_args: None)
    scheduled = []

    class Thread:
        def __init__(self, *, target, daemon):
            scheduled.append(target)

        def start(self):
            return None

    monkeypatch.setattr(runtime.threading, "Thread", Thread)

    assert runtime._start_resubmit(
        spec,
        expected_remote=None,
        expected_state="provisioning",
        confirmed_absent=True,
    )
    assert not runtime._start_resubmit(
        spec,
        expected_remote=None,
        expected_state="provisioning",
        confirmed_absent=True,
    )

    current = runner_status.get_status(spec.run_id)
    assert current.retry_counters == {"infra": 1, "oom": 0, "cache": 0}
    assert runner_status._current_attempt(current).state == "settling"
    assert len(scheduled) == 1
    scheduled[0]()


@pytest.mark.parametrize(
    ("failure_class", "failure"),
    [
        ("artifact_transport", "artifact_transport"),
        ("provider_preempted", "job_preempted"),
    ],
)
def test_handleless_retry_restart_resumes_claim_without_spending_budget_twice(
    monkeypatch, tmp_path, failure_class, failure
):
    import flash.runner.supervise.lifecycle as lifecycle
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.artifacts.attempts import poll_result_from_manifest
    from flash.runner.lifecycle.protocol import ResultManifest
    from flash.server.platform.handleless_retry import failure_retry
    from flash.snapshot.archive import source_attestation

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id=f"handleless-restart-claim-{failure_class}",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=1),
    )
    runner_state._save_status(
        _handleless_status(spec, created_at=100.0),
        _next_attempt=1,
        _next_fence=2,
    )
    status = runner_status.get_status(spec.run_id)
    result = poll_result_from_manifest(
        ResultManifest(
            run_id=spec.run_id,
            phase_namespace="training",
            attempt_id=0,
            fence=1,
            outcome="failed",
            failure_class=failure_class,
            started_at=100.0,
            finished_at=101.0,
            training_entered=True,
            completed_steps=0,
            metrics={},
            checkpoint={},
            artifacts={},
            source_attestation=source_attestation(
                _SOURCE_SNAPSHOT,
                run_id=spec.run_id,
                attempt=0,
                fence=1,
            ),
            diagnostics={"error": "worker failed"},
        ).to_dict()
    )
    assert result.failure == failure
    resolution = failure_retry(spec, status, result)
    assert resolution is not None
    assert runner_reconciliation._compare_and_prepare_resubmit(
        spec.run_id,
        None,
        expected_state="provisioning",
        expected_attempt_id=resolution.attempt_id,
        expected_fence=resolution.fence,
        expected_attempt_state=resolution.attempt_state,
        retry_counters=resolution.retry_counters,
    )
    claimed = runner_status.get_status(spec.run_id)
    expected_counters = {"infra": 1, "oom": 0, "cache": 0}
    assert claimed.retry_counters == expected_counters
    assert runner_status._current_attempt(claimed).state == "settling"

    monkeypatch.setattr(runtime, "_recovery_block_reason", lambda _spec: None)
    monkeypatch.setattr(runtime, "_handleless_attempt_result", lambda *_args, **_kwargs: result)
    launched = []
    monkeypatch.setattr(
        lifecycle,
        "_run_job_background",
        lambda *args: launched.append(args),
    )
    scheduled = []

    class Thread:
        def __init__(self, *, target, daemon):
            scheduled.append(target)

        def start(self):
            return None

    monkeypatch.setattr(runtime.threading, "Thread", Thread)

    assert runtime._start_resubmit(
        spec,
        expected_remote=None,
        expected_state="provisioning",
        confirmed_absent=True,
    )
    assert not runtime._start_resubmit(
        spec,
        expected_remote=None,
        expected_state="provisioning",
        confirmed_absent=True,
    )
    assert runner_status.get_status(spec.run_id).retry_counters == expected_counters
    assert len(scheduled) == 1
    scheduled[0]()
    assert launched == [(spec, None, 1, 0.0)]
    assert runner_status.get_status(spec.run_id).retry_counters == expected_counters


def test_handleless_failure_with_unconfirmed_absence_does_not_consume_or_launch(
    monkeypatch, tmp_path
):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="handleless-unconfirmed",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=1),
    )
    runner_state._save_status(
        _handleless_status(spec, created_at=100.0),
        _next_attempt=1,
        _next_fence=2,
    )
    monkeypatch.setattr(runtime, "_confirm_run_clear", lambda _spec: False)
    monkeypatch.setattr(runtime, "_recovery_block_reason", lambda _spec: None)
    monkeypatch.setattr(
        runtime,
        "_handleless_attempt_result",
        lambda *_args, **_kwargs: pytest.fail("unconfirmed absence must not consume a result"),
    )
    started = []

    class Thread:
        def __init__(self, *, target, args, daemon):
            started.append((target, args, daemon))

        def start(self):
            return None

    monkeypatch.setattr(runtime.threading, "Thread", Thread)

    runtime._resubmit_recovered_runs([(spec, "provisioning")])

    current = runner_status.get_status(spec.run_id)
    assert current.state == "provisioning"
    assert current.retry_counters is None
    assert started == [(runtime._deferred_resubmit_loop, (spec,), True)]


@pytest.mark.parametrize(
    ("failure", "max_retries", "counters"),
    [
        ("oom", 0, None),
        ("oom", 1, {"infra": 0, "oom": 1, "cache": 0}),
        ("artifact_transport", 0, None),
        ("artifact_transport", 1, {"infra": 5, "oom": 0, "cache": 0}),
    ],
)
def test_handleless_failure_with_no_budget_settles_without_replacement(
    monkeypatch, tmp_path, failure, max_retries, counters
):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id=f"handleless-terminal-{failure}-{max_retries}",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(type="A100 SXM 40GB", max_retries=max_retries),
    )
    status = _handleless_status(spec, created_at=100.0)
    status.retry_counters = counters
    runner_state._save_status(status, _next_attempt=1, _next_fence=2)
    monkeypatch.setattr(runtime, "_confirm_run_clear", lambda _spec: True)
    monkeypatch.setattr(runtime, "_recovery_block_reason", lambda _spec: None)
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    monkeypatch.setattr(
        runtime,
        "_handleless_attempt_result",
        lambda *_args, **_kwargs: PollResult(False, failure=failure, detail="worker failed"),
    )
    started = []
    monkeypatch.setattr(
        runtime.threading,
        "Thread",
        lambda **kwargs: started.append(kwargs) or SimpleNamespace(start=lambda: None),
    )

    runtime._resubmit_recovered_runs([(spec, "provisioning")])

    current = runner_status.get_status(spec.run_id)
    assert current.state == "failed"
    assert current.retry_counters == counters
    assert runner_status._current_attempt(current).state == "settled"
    assert started == []


def test_handleless_retry_rejects_stale_attempt_without_consuming_budget(monkeypatch, tmp_path):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec
    from flash.runner.lifecycle.protocol import AttemptRecord
    from flash.server.platform.handleless_retry import failure_retry

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="handleless-stale-retry",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=1),
    )
    status = _handleless_status(spec, created_at=100.0)
    runner_state._save_status(status, _next_attempt=1, _next_fence=2)
    resolution = failure_retry(
        spec,
        status,
        PollResult(False, failure="artifact_transport", detail="worker failed"),
    )
    attempt = runner_status._current_attempt(status)
    replacement = replace(attempt, attempt_id=1, fence=2)
    with runner_state._status_guard(spec.run_id):
        current = runner_status.get_status(spec.run_id)
        current.attempt = replacement.to_dict()
        runner_state._save_status_unlocked(current)
    monkeypatch.setattr(runtime, "_recovery_block_reason", lambda _spec: None)
    monkeypatch.setattr(
        runtime.threading,
        "Thread",
        lambda **_kwargs: pytest.fail("stale retry must not launch"),
    )

    assert not runtime._start_resubmit(
        spec,
        expected_remote=None,
        expected_state="provisioning",
        resolution=resolution,
    )

    current = runner_status.get_status(spec.run_id)
    assert AttemptRecord.from_dict(current.attempt) == replacement
    assert current.retry_counters is None


def test_handleless_oom_with_invalid_effective_allocation_fails_closed(monkeypatch, tmp_path):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="handleless-invalid-oom",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(type="A100 SXM 40GB", max_retries=1),
    )
    status = _handleless_status(spec, created_at=100.0)
    status.effective_preparation["worker_spec"]["gpu"]["count"] = 0
    runner_state._save_status(status, _next_attempt=1, _next_fence=2)
    monkeypatch.setattr(runtime, "_confirm_run_clear", lambda _spec: True)
    monkeypatch.setattr(runtime, "_recovery_block_reason", lambda _spec: None)
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    monkeypatch.setattr(
        runtime,
        "_handleless_attempt_result",
        lambda *_args, **_kwargs: PollResult(False, failure="oom", detail="worker failed"),
    )

    runtime._resubmit_recovered_runs([(spec, "provisioning")])

    current = runner_status.get_status(spec.run_id)
    assert current.state == "failed"
    assert current.retry_counters is None


def test_handleless_opd_guard_blocks_retry_without_consuming_budget(monkeypatch, tmp_path):
    import flash.runner.lifecycle.attempts as attempts
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="handleless-opd-guard",
        model="Qwen/Qwen3.5-9B",
        algorithm="opd",
        gpu=GpuSpec(max_retries=1),
    )
    status = _handleless_status(spec, created_at=100.0)
    runner_state._save_status(status, _next_attempt=1, _next_fence=2)
    monkeypatch.setattr(runtime, "_confirm_run_clear", lambda _spec: True)
    monkeypatch.setattr(runtime, "_recovery_block_reason", lambda _spec: None)
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    monkeypatch.setattr(
        runtime,
        "_handleless_attempt_result",
        lambda *_args, **_kwargs: PollResult(
            False, failure="artifact_transport", detail="worker failed"
        ),
    )
    monkeypatch.setattr(
        attempts,
        "_verified_opd_next_attempt",
        lambda _run_id: (_ for _ in ()).throw(RuntimeError("opd retry evidence invalid")),
    )

    runtime._resubmit_recovered_runs([(spec, "provisioning")])

    current = runner_status.get_status(spec.run_id)
    assert current.state == "failed"
    assert current.retry_counters is None
    assert current.error == "artifact_transport: worker failed"


@pytest.mark.parametrize("status_read_fails", [False, True])
def test_recover_runs_defers_when_resubmit_waits_for_metrics(
    monkeypatch, tmp_path, status_read_fails
):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="recover-pending", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict())
    )
    monkeypatch.setattr(runtime.db, "all_runs", lambda: [{"run_id": spec.run_id}])
    monkeypatch.setattr(runner_reconciliation, "_drain_cleanup_remotes", lambda _run_id: None)
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        runner_preparation, "_mark_warmstart_source", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner_status, "effective_spec_from_status", lambda _status, **_kwargs: spec
    )
    monkeypatch.setattr(providers, "configured_providers", list)
    monkeypatch.setattr(runtime, "_recovery_block_reason", lambda _spec: None)
    monkeypatch.setattr(runtime, "_confirm_run_clear", lambda _spec: True)
    resubmit_attempted = {"value": False}

    def start_resubmit(*_args, **_kwargs):
        resubmit_attempted["value"] = True
        return False

    real_get_status = runtime.get_status

    def get_status(run_id):
        if status_read_fails and resubmit_attempted["value"]:
            raise OSError("status store unavailable")
        return real_get_status(run_id)

    monkeypatch.setattr(runtime, "_start_resubmit", start_resubmit)
    monkeypatch.setattr(runtime, "get_status", get_status)
    started = []

    class Thread:
        def __init__(self, *, target, args, daemon):
            started.append((target, args, daemon))

        def start(self):
            return None

    monkeypatch.setattr(runtime.threading, "Thread", Thread)

    runtime.recover_runs()

    # recover_runs also backgrounds a _drain_cleanup_remotes_bg thread per known run (see
    # recover_runs) so a slow/outage-hit provider teardown can't block the startup path; check
    # for the resubmit-loop thread specifically rather than asserting the full thread list.
    assert (runtime._deferred_resubmit_loop, (spec,), True) in started
    drain_calls = [
        args for target, args, _daemon in started if target.__name__ == "_drain_cleanup_remotes_bg"
    ]
    assert drain_calls == [(spec.run_id,)]


@pytest.mark.parametrize("retired_model", _RETIRED_MODELS)
def test_recover_runs_tears_down_a_handle_backed_run_whose_model_was_removed(
    monkeypatch, tmp_path, retired_model
):
    # a model dropped from the catalog while one of its runs is still nonterminal: the
    # persisted spec remains structurally parseable, but catalog eligibility fails on the new build.
    # dispatching attach_run would otherwise leave the run active long enough to reach provider work,
    # so startup recovery must fail it and remove the resource before dispatch.
    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    remote = _runpod_remote("ep-stale", "job-stale", attempt=0, fence=1)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="recover-unparseable",
            state="running",
            spec=JobSpec(
                run_id="recover-unparseable", model="Qwen/Qwen3.5-9B", algorithm="sft"
            ).to_dict(),
            remote=dict(remote),
        )
    )
    # write a current model first, then sabotage the stored public spec to match a run persisted by
    # the old catalog. structural parsing still succeeds, so only the new eligibility guard catches it.
    stored_path = runner_state.runs_file_path("recover-unparseable", ".json")
    with open(stored_path, encoding="utf-8") as handle:
        stored = json.load(handle)
    stored["spec"]["model"] = retired_model
    with open(stored_path, "w", encoding="utf-8") as handle:
        json.dump(stored, handle)
    assert JobSpec.from_dict(stored["spec"]).model == retired_model
    with pytest.raises(ValueError, match="unsupported model"):
        runner_status.effective_spec_from_status(runner_status.get_status("recover-unparseable"))
    monkeypatch.setattr(runtime.db, "all_runs", lambda: [{"run_id": "recover-unparseable"}])
    monkeypatch.setattr(runner_reconciliation, "_drain_cleanup_remotes", lambda _run_id: None)
    monkeypatch.setattr(providers, "configured_providers", list)
    torn: list[tuple[dict, str]] = []

    def fake_teardown(handle, run_id):
        torn.append((dict(handle), run_id))
        return True

    monkeypatch.setattr("flash.runner.supervise.lifecycle._strict_teardown_handle", fake_teardown)
    attached: list[str] = []
    # recover_runs imports attach_run from flash.runner inside the function, so patch it there.
    monkeypatch.setattr(runner_attach, "attach_run", lambda rid: attached.append(rid))

    class Thread:
        def __init__(self, *, target, args=(), daemon=False):
            self._target, self._args = target, args

        def start(self):
            # only the cleanup drain is backgrounded here; attach_run must never be dispatched.
            self._target(*self._args)

    monkeypatch.setattr(runtime.threading, "Thread", Thread)

    runtime.recover_runs()

    assert attached == []
    assert torn == [(remote, "recover-unparseable")]
    status = runner_status.get_status("recover-unparseable")
    assert status.state == "failed"
    assert "persisted spec cannot be activated" in (status.error or "")


@pytest.mark.parametrize("retired_model", _RETIRED_MODELS)
def test_recover_runs_rejects_handleless_removed_model_before_resubmit_or_gc(
    monkeypatch, tmp_path, retired_model
):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="retired-handleless", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict())
    )
    path = runner_state.runs_file_path(spec.run_id, ".json")
    with open(path, encoding="utf-8") as handle:
        stored = json.load(handle)
    stored["spec"]["model"] = retired_model
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stored, handle)

    monkeypatch.setattr(runtime.db, "all_runs", lambda: [{"run_id": spec.run_id}])
    monkeypatch.setattr(providers, "configured_providers", list)
    monkeypatch.setattr(runner_reconciliation, "_drain_cleanup_remotes", lambda _run_id: None)
    monkeypatch.setattr(
        runner_recovery,
        "_gc_run_endpoints",
        lambda _spec: pytest.fail("removed model reached active endpoint gc"),
    )
    monkeypatch.setattr(
        runner_attach, "attach_run", lambda _run_id: pytest.fail("attach dispatched")
    )
    monkeypatch.setattr(
        runtime, "_start_resubmit", lambda *_args, **_kwargs: pytest.fail("resubmit")
    )
    terminated = []
    monkeypatch.setattr(
        "flash.providers.runpod.execution.provider.terminate_persisted_endpoints",
        lambda raw_spec, run_id: terminated.append((raw_spec["model"], run_id)),
    )

    class Thread:
        def __init__(self, *, target, args=(), daemon=False):
            self._target, self._args = target, args

        def start(self):
            self._target(*self._args)

    monkeypatch.setattr(runtime.threading, "Thread", Thread)

    runtime.recover_runs()

    assert terminated == [(retired_model, spec.run_id)]
    status = runner_status.get_status(spec.run_id)
    assert status.state == "failed"
    assert "unsupported model" in (status.error or "")


def test_recover_runs_reattaches_confirmed_teardown_marker(monkeypatch, tmp_path):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="recover-confirmed", model="Qwen/Qwen3.5-9B", algorithm="sft")
    remote = _runpod_remote("endpoint-confirmed", "job-confirmed", attempt=0)
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            remote=None,
            cleanup_confirmed_remote=remote,
            realized_cost_remote=remote,
        )
    )
    monkeypatch.setattr(runtime.db, "all_runs", lambda: [{"run_id": spec.run_id}])
    monkeypatch.setattr(providers, "configured_providers", list)
    monkeypatch.setattr(runner_reconciliation, "_drain_cleanup_remotes", lambda _run_id: None)
    swept = []
    monkeypatch.setattr(
        runtime,
        "_sweep_provider_orphans",
        lambda active, known: swept.append((set(active), set(known))),
    )
    attached = []
    monkeypatch.setattr(runner_attach, "attach_run", lambda run_id: attached.append(run_id))
    monkeypatch.setattr(
        runtime,
        "_start_resubmit",
        lambda *_args, **_kwargs: pytest.fail("confirmed teardown was resubmitted handleless"),
    )

    class Thread:
        def __init__(self, *, target, args=(), daemon=False):
            self._target, self._args = target, args

        def start(self):
            self._target(*self._args)

    monkeypatch.setattr(runtime.threading, "Thread", Thread)

    runtime.recover_runs()

    assert attached == [spec.run_id]
    assert swept == [({spec.run_id}, {spec.run_id})]


def test_unparseable_spec_retries_a_teardown_it_could_not_confirm(monkeypatch, tmp_path):
    # when `_strict_teardown_handle` cannot confirm the delete, the handle is recorded for the
    # cleanup drain -- but this run's drain was already dispatched at the top of the loop and had
    # snapshotted an empty list, and it returns early on empty. so the record sat there with nothing
    # scheduled to retry it, and the worker kept billing until the next restart.
    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec
    from flash.providers.core import registry as providers

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    # a COMPLETE handle: `_record_cleanup_remote` drops anything it cannot resolve to an exact
    # provider resource identity, so a partial one would make this test pass for the wrong reason.
    remote = {
        "provider": "runpod",
        "endpoint_id": "ep-unconfirmed",
        "endpoint_name": "flash-recover-unconfirmed",
        "key_fingerprint": "rpk-" + "0" * 64,
        "attempt": 0,
        "fence": 1,
        "started_ts": 1.0,
    }
    assert runner_reconciliation._remote_resource_identity(remote) is not None
    runner_state._save_status(
        runner_state.RunStatus(
            run_id="recover-unconfirmed",
            state="running",
            spec=JobSpec(
                run_id="recover-unconfirmed", model="Qwen/Qwen3.5-9B", algorithm="sft"
            ).to_dict(),
            remote=dict(remote),
        )
    )
    stored_path = runner_state.runs_file_path("recover-unconfirmed", ".json")
    with open(stored_path, encoding="utf-8") as handle:
        stored = json.load(handle)
    stored["spec"]["algorithm"] = "retired-algorithm"
    with open(stored_path, "w", encoding="utf-8") as handle:
        json.dump(stored, handle)
    monkeypatch.setattr(runtime.db, "all_runs", lambda: [{"run_id": "recover-unconfirmed"}])
    monkeypatch.setattr(providers, "configured_providers", list)

    torn: list[str] = []

    def fake_teardown(handle, run_id):
        # the direct teardown is handed the raw dict; the drain rebuilds a JobHandle from the
        # persisted record, so the two call sites are distinguishable here.
        torn.append("drain" if hasattr(handle, "provider") else "direct")
        return False  # unconfirmed, both times: this is the case that records for the drain

    monkeypatch.setattr("flash.runner.supervise.lifecycle._strict_teardown_handle", fake_teardown)
    monkeypatch.setattr(runner_attach, "attach_run", lambda rid: None)
    # persisting a cleanup record reports the new status, which blocks on its reporter thread. that
    # thread is real in production; here the fake below would stand in for it and never run.
    monkeypatch.setattr(runner_reporting, "_report_status", lambda status: None)

    class Thread:
        def __init__(self, *, target, args=(), daemon=False):
            self._target, self._args = target, args

        def start(self):
            self._target(*self._args)

    # patch the ATTRIBUTE recover_runs reads, not threading.Thread globally: the module-wide patch
    # also replaced the status reporter's own worker thread, so nothing serviced its queue.
    monkeypatch.setattr(runtime.threading, "Thread", Thread)

    runtime.recover_runs()

    # the drain runs twice: once before the teardown (nothing recorded yet, so it tears down
    # nothing) and once after, which is the retry. without the second dispatch the sequence is
    # ["direct"] and the recorded resource is never attempted at all.
    assert torn == ["direct", "drain"]
    # and it is still recorded, because the retry did not confirm the delete either -- so the next
    # restart's drain will find it and try again, rather than the record being dropped unconfirmed.
    with open(stored_path, encoding="utf-8") as handle:
        assert json.load(handle)["cleanup_remotes"] == [remote]


def test_deferred_handleless_loop_resubmits_when_clear_before_deadline(monkeypatch, tmp_path):
    import time as time_mod

    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec.from_dict(
        {
            "run_id": "deferred-clear",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "train": {"epochs": 1},
        }
    )
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict())
    )
    checks = iter([False, True])
    monkeypatch.setattr(runtime, "_confirm_run_clear", lambda _spec: next(checks))
    monkeypatch.setattr(runner_deadlines, "_load_run_deadline_at", lambda _run_id: 1000.0)
    monkeypatch.setattr(time_mod, "time", lambda: 100.0)
    monkeypatch.setattr(time_mod, "sleep", lambda _seconds: None)
    started = []
    monkeypatch.setattr(
        runtime,
        "_start_resubmit",
        lambda *args, **kwargs: started.append((args, kwargs)) or True,
    )

    runtime._deferred_resubmit_loop(spec)

    assert len(started) == 1
    assert started[0][1]["expected_remote"] is None
    assert started[0][1]["expected_state"] == "provisioning"
    assert started[0][1]["resolution"].attempt_start == 0
    assert runner_status.get_status(spec.run_id).state == "provisioning"


def test_deferred_handleless_loop_waits_through_provider_minimum_window(monkeypatch, tmp_path):
    import time as time_mod

    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="deferred-minimum-window",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=120),
    )
    status = provisioned_status(spec, state="provisioning", created_at=10.0)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(
        status,
        _run_deadline_at=130.0,
        _next_attempt=0,
    )
    clock = {"now": 100.0}
    monkeypatch.setattr(time_mod, "time", lambda: clock["now"])
    monkeypatch.setattr(runtime, "_confirm_run_clear", lambda _spec: True)
    monkeypatch.setattr(runtime, "_handleless_attempt_result", lambda *a, **k: None)
    failures = []
    real_fail = runner_reconciliation._compare_and_fail_remote

    def record_failure(*args, **kwargs):
        failures.append(clock["now"])
        return real_fail(*args, **kwargs)

    def advance(seconds):
        assert runner_status.get_status(spec.run_id).state == "provisioning"
        clock["now"] += seconds

    monkeypatch.setattr(runner_reconciliation, "_compare_and_fail_remote", record_failure)
    monkeypatch.setattr(time_mod, "sleep", advance)

    runtime._deferred_resubmit_loop(spec)

    status = runner_status.get_status(spec.run_id)
    assert failures == [130.0]
    assert status.state == "failed"


def test_deferred_handleless_loop_reconciles_after_resubmit_cas_loss(monkeypatch, tmp_path):
    import time as time_mod

    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="deferred-cas-loss", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict())
    )
    monkeypatch.setattr(runtime, "_confirm_run_clear", lambda _spec: True)
    monkeypatch.setattr(runner_deadlines, "_load_run_deadline_at", lambda _run_id: 1000.0)
    monkeypatch.setattr(time_mod, "time", lambda: 100.0)
    sleeps = []
    monkeypatch.setattr(time_mod, "sleep", sleeps.append)
    attempts = []

    def start_resubmit(*args, **kwargs):
        attempts.append((args, kwargs))
        return len(attempts) == 2

    monkeypatch.setattr(runtime, "_start_resubmit", start_resubmit)

    runtime._deferred_resubmit_loop(spec)

    assert len(attempts) == 2
    assert sleeps == [runtime._DEFERRED_RECOVERY_RETRY_S]
    assert (
        attempts[0][1]
        == attempts[1][1]
        == {
            "expected_remote": None,
            "expected_state": "provisioning",
            "result_resolved": True,
        }
    )


def test_deferred_handleless_loop_deadline_cas_fails_with_retry(monkeypatch, tmp_path):
    import time as time_mod

    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec.from_dict(
        {
            "run_id": "deferred-deadline",
            "model": "Qwen/Qwen3.5-9B",
            "algorithm": "grpo",
            "train": {"epochs": 1},
        }
    )
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict())
    )
    monkeypatch.setattr(runner_deadlines, "_load_run_deadline_at", lambda _run_id: 100.0)
    monkeypatch.setattr(time_mod, "time", lambda: 101.0)
    monkeypatch.setattr(time_mod, "sleep", lambda _seconds: None)
    monkeypatch.setattr(runtime, "_handleless_attempt_result", lambda *a, **k: None)
    real_fail = runner_reconciliation._compare_and_fail_remote
    attempts = []

    def flaky_fail(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise PermissionError("status store unavailable")
        return real_fail(*args, **kwargs)

    monkeypatch.setattr(runner_reconciliation, "_compare_and_fail_remote", flaky_fail)

    runtime._deferred_resubmit_loop(spec)

    status = runner_status.get_status(spec.run_id)
    assert len(attempts) == 2
    assert status.state == "failed"
    assert status.remote is None
    assert "deadline exhausted" in (status.error or "")


def test_deferred_handleless_permanent_result_artifact_fails_without_retry(monkeypatch, tmp_path):
    import time as time_mod

    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec
    from flash.providers.artifacts.attempts import AttemptArtifactError

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="deferred-invalid-result", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        _handleless_status(spec, created_at=100.0),
        _run_deadline_at=86500.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(time_mod, "time", lambda: 86501.0)
    monkeypatch.setattr(
        runtime,
        "_handleless_attempt_result",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AttemptArtifactError("result manifest is invalid or unverifiable")
        ),
    )
    monkeypatch.setattr(
        time_mod,
        "sleep",
        lambda _seconds: pytest.fail("permanent artifact rejection must not retry"),
    )

    runtime._deferred_resubmit_loop(spec)

    status = runner_status.get_status(spec.run_id)
    assert status.state == "failed"
    assert status.remote is None
    assert "invalid or unverifiable" in (status.error or "")


def test_deferred_handleless_legacy_run_without_attempt_metadata_fails_at_deadline(
    monkeypatch, tmp_path
):
    import time as time_mod

    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="deferred-legacy", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="provisioning",
            spec=spec.to_dict(),
            created_at=100.0,
        ),
        _run_deadline_at=86500.0,
    )
    raw = runner_status._load_status_json(spec.run_id)
    raw.pop(runner_state._NEXT_ATTEMPT_KEY, None)
    with open(runner_state.runs_file_path(spec.run_id, ".json"), "w") as file:
        json.dump(raw, file)

    monkeypatch.setattr(time_mod, "time", lambda: 86501.0)
    monkeypatch.setattr(
        time_mod,
        "sleep",
        lambda _seconds: pytest.fail("legacy recovery must converge without retrying"),
    )

    runtime._deferred_resubmit_loop(spec)

    status = runner_status.get_status(spec.run_id)
    assert status.state == "failed"
    assert status.remote is None
    assert "deadline exhausted" in (status.error or "")


@pytest.mark.parametrize("cleanup_confirmed", [True, False])
def test_terminal_handle_race_tears_down_or_preserves_cleanup_identity(
    monkeypatch, tmp_path, cleanup_confirmed
):
    import io

    import flash.providers.core.allocator as allocator
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core import registry as providers
    from flash.providers.core.base import Allocation, Candidate
    from flash.runner.supervise import lifecycle
    from tests._helpers.profile import attach_sft_profile, stub_revision_geometry

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    # the attached profile pins a model revision, which makes the post-allocation quote refresh
    # resolve revision-specific geometry from the hub. read the catalog's numbers instead.
    stub_revision_geometry(monkeypatch)
    spec = attach_sft_profile(
        JobSpec(
            run_id=f"terminal-handle-race-{cleanup_confirmed}",
            model="Qwen/Qwen3.5-9B",
            algorithm="sft",
            train=TrainSpec(max_examples=1),
            gpu=GpuSpec(type="", max_retries=2),
        )
    )
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()),
        _next_attempt=0,
    )
    candidate = Candidate("runpod", "RTX 4090", 0.69, 24)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *_args, **_kwargs: Allocation(
            provider="runpod",
            gpu="RTX 4090",
            hourly_usd=0.69,
            min_vram_gb=24,
            candidates=(candidate,),
        ),
    )

    class Provider:
        supports_weight_cache = False

        def __init__(self):
            self.submits = []
            self.teardown = []

        def submit_run(self, _spec, _seed, *, attempt, on_handle, **_kwargs):
            self.submits.append(attempt)
            runner_status._update(spec.run_id, "cancelled")
            on_handle(
                _runpod_remote(
                    "endpoint-race",
                    "job-race",
                    attempt=attempt,
                )
            )
            raise AssertionError("terminal handle callback must not return")

        def cancel(self, handle):
            self.teardown.append(("cancel", handle.to_dict()))

        def destroy(self, handle):
            self.teardown.append(("destroy", handle.to_dict()))
            if not cleanup_confirmed:
                raise RuntimeError("endpoint deletion unconfirmed")

    provider = Provider()
    monkeypatch.setattr(providers, "get_provider", lambda _name: provider)

    with pytest.raises(runner_errors._TerminalHandleRace):
        lifecycle._submit_seed_supervised(
            spec,
            spec.seed,
            io.StringIO(),
            source_snapshot=_SOURCE_SNAPSHOT,
        )

    status = runner_status.get_status(spec.run_id)
    assert provider.submits == [0]
    assert [event for event, _handle in provider.teardown] == ["cancel", "destroy"]
    assert status.state == "cancelled"
    if cleanup_confirmed:
        assert status.remote is None
    else:
        assert status.remote["endpoint_id"] == "endpoint-race"
        assert status.remote["job_id"] == "job-race"
        assert status.remote["attempt"] == 0
        raw = runner_status._load_status_json(spec.run_id)
        assert raw[runner_state._CLEANUP_REMOTES_KEY] == [
            _runpod_remote("endpoint-race", "job-race", attempt=0)
        ]


def test_terminal_handle_race_retains_second_unconfirmed_cleanup_remote(monkeypatch, tmp_path):
    import io

    import flash.providers.core.allocator as allocator
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core import registry as providers
    from flash.providers.core.base import Allocation, Candidate
    from flash.runner.supervise import lifecycle
    from tests._helpers.profile import attach_sft_profile, stub_revision_geometry

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    # the attached profile pins a model revision, which makes the post-allocation quote refresh
    # resolve revision-specific geometry from the hub. read the catalog's numbers instead.
    stub_revision_geometry(monkeypatch)
    spec = attach_sft_profile(
        JobSpec(
            run_id="terminal-handle-race-two-remotes",
            model="Qwen/Qwen3.5-9B",
            algorithm="sft",
            train=TrainSpec(max_examples=1),
            gpu=GpuSpec(type="", max_retries=0),
        )
    )
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()),
        _next_attempt=0,
    )
    candidate = Candidate("runpod", "RTX 4090", 0.69, 24)
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *_args, **_kwargs: Allocation(
            provider="runpod",
            gpu="RTX 4090",
            hourly_usd=0.69,
            min_vram_gb=24,
            candidates=(candidate,),
        ),
    )
    remote_a = _runpod_remote("endpoint-a", "job-a", attempt=0)
    remote_b = _runpod_remote("endpoint-b", "job-b", attempt=0)

    class Provider:
        supports_weight_cache = False

        def submit_run(self, _spec, _seed, *, on_handle, **_kwargs):
            runner_status._update(spec.run_id, "cancelled", remote=remote_a)
            on_handle(remote_b)
            raise AssertionError("terminal handle callback must not return")

        def cancel(self, _handle):
            return None

        def destroy(self, _handle):
            raise RuntimeError("endpoint deletion unconfirmed")

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())

    with pytest.raises(runner_errors._TerminalHandleRace):
        lifecycle._submit_seed_supervised(
            spec,
            spec.seed,
            io.StringIO(),
            source_snapshot=_SOURCE_SNAPSHOT,
        )

    raw = runner_status._load_status_json(spec.run_id)
    assert raw["remote"] == remote_a
    assert raw[runner_state._CLEANUP_REMOTES_KEY] == [remote_b]


def test_run_training_bails_when_running_cas_rejects(monkeypatch):
    """If a run flips terminal in the race window between the pre-check and the ``running`` CAS,
    _run_training must raise _RunCancelled and never reach the PAID supervised submit. The gate is
    _update's return value (False == rejected by terminal-stickiness)."""
    import pytest

    from flash.core.spec import JobSpec
    from flash.runner.supervise import lifecycle

    spec = JobSpec(run_id="cas", model="Qwen/Qwen3.5-9B", algorithm="grpo")
    # Pre-check sees a live run...
    monkeypatch.setattr(
        runner_status,
        "get_status",
        lambda rid: runner_state.RunStatus(run_id=rid, state="running", spec={}),
    )
    # ...but the CAS rejects because the run went terminal concurrently.
    monkeypatch.setattr(runner_status, "_update", lambda *a, **k: False)
    submitted: list[bool] = []
    monkeypatch.setattr(
        runner_lifecycle,
        "_submit_seed_supervised",
        lambda *a, **k: submitted.append(True) or {},
    )

    with pytest.raises(runner_errors._RunCancelled):
        lifecycle._run_training(spec, None, prior_cost=0.0, source_snapshot=_SOURCE_SNAPSHOT)
    assert submitted == []  # never charged a GPU for an already-terminal run
