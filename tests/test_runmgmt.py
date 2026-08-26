"""Tests for run-management helpers (runs/cost/cancel) — no GPU/network."""

from __future__ import annotations

import json
import tempfile

import pytest

import flash.engine.worker.entry.worker as worker_entry
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
from tests._helpers.runner import provisioned_status
from tests._helpers.source_snapshot import valid_source_snapshot

_RUNPOD_FINGERPRINT = "rpk-" + "0" * 64
_SOURCE_SNAPSHOT = valid_source_snapshot()
_RETIRED_MODELS = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-27B")


def _runpod_remote(endpoint_id="endpoint", job_id="job", attempt=0, started_ts=1.0, **extra):
    remote = {
        "provider": "runpod",
        "endpoint_id": endpoint_id,
        "endpoint_name": f"{endpoint_id}-name",
        "key_fingerprint": _RUNPOD_FINGERPRINT,
        "attempt": attempt,
        "started_ts": started_ts,
        **extra,
    }
    if job_id is not None:
        remote["job_id"] = job_id
    return remote


def _lambda_remote(instance_id="instance", attempt=0, started_ts=1.0, **extra):
    return {
        "provider": "lambda",
        "instance_id": instance_id,
        "instance_type": "gpu_1x_a100",
        "region": "us-east-1",
        "name": f"flash-{instance_id}",
        "gpu": "A100",
        "hourly_usd": 1.0,
        "attempt": attempt,
        "started_ts": started_ts,
        **extra,
    }


def _vast_remote(instance_id=7, attempt=0, started_ts=1.0, **extra):
    return {
        "provider": "vast",
        "instance_id": instance_id,
        "offer_id": 101,
        "machine_id": 202,
        "label": f"flash-{instance_id}",
        "gpu": "RTX 4090",
        "hourly_usd": 0.5,
        "attempt": attempt,
        "started_ts": started_ts,
        **extra,
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


def test_record_heartbeat_updates_status_without_state_change(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", tmp)
        from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
        from tests._helpers.profile import satisfy_sft_profile

        spec = JobSpec(
            run_id="hb",
            model="Qwen/Qwen3.5-9B",
            algorithm="sft",
            environment=EnvironmentSpec(id="team/example"),
            train=TrainSpec(max_examples=8),
        )
        status = runner_submit.submit_job(satisfy_sft_profile(monkeypatch, spec), dry_run=True)
        status.state = "running"
        runner_state._save_status(status)

        runner_status.record_heartbeat(
            "hb",
            {
                "stage": "sft_step",
                "step": 20,
                "ts": 123.0,
                "gpu": {
                    "device_name": "RTX 5090",
                    "gpu_util_pct": 94,
                    "memory_used_gb": 19.5,
                    "processes": [{"pid": 42, "process_name": "python", "used_memory_gb": 19.0}],
                },
            },
        )

        out = runner_status.get_status("hb")
        assert out.state == "running"
        assert out.last_heartbeat["stage"] == "sft_step"
        assert out.last_heartbeat["step"] == 20
        assert out.gpu_status["device_name"] == "RTX 5090"
        assert out.gpu_status["gpu_util_pct"] == 94


def test_an_error_heartbeat_carries_gpu_diagnostics_through_to_status(monkeypatch):
    """The failure heartbeat must spell its diagnostics `gpu`, the key the consumer reads.

    a wrong spelling loses the FAILURE's own diagnostics -- the memory figure that says whether an
    oom was the cause -- and leaves `gpu_status` showing the last healthy sample instead, which
    reads as if nothing was wrong. every other producer already spells it `gpu`.
    """
    import inspect
    import tempfile as _tempfile

    # the producer half: the error path must not reintroduce a spelling the consumer ignores.
    assert "diag=gpu_diagnostics()" not in inspect.getsource(worker_entry)

    with _tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", tmp)
        from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
        from tests._helpers.profile import satisfy_sft_profile

        spec = JobSpec(
            run_id="hb-oom",
            model="Qwen/Qwen3.5-9B",
            algorithm="sft",
            environment=EnvironmentSpec(id="team/example"),
            train=TrainSpec(max_examples=8),
        )
        status = runner_submit.submit_job(satisfy_sft_profile(monkeypatch, spec), dry_run=True)
        status.state = "running"
        runner_state._save_status(status)

        runner_status.record_heartbeat(
            "hb-oom",
            {"stage": "sft_step", "step": 20, "gpu": {"device_name": "B200", "gpu_util_pct": 99}},
        )
        # the consumer half: the failure heartbeat's own diagnostics must survive, and must not be
        # replaced by None just because the run is now failing.
        runner_status.record_heartbeat(
            "hb-oom",
            {
                "stage": "error_sft",
                "oom": True,
                "gpu": {"device_name": "B200", "memory_used_gb": 179.4},
            },
        )

        out = runner_status.get_status("hb-oom")
        assert out.gpu_status is not None, "the oom heartbeat cleared the gpu snapshot"
        assert out.gpu_status["memory_used_gb"] == 179.4


def test_a_heartbeat_without_gpu_keeps_the_attempts_snapshot(monkeypatch):
    """Only 8 of the ~51 heartbeat producers send `gpu`; the rest must not blank the snapshot.

    the periodic liveness tick samples the card, but a checkpoint upload can run for minutes
    between two of them, and every heartbeat in that window omits `gpu`. assigning it
    unconditionally made `flash runs status` and the `gpuStatus` API field report no GPU for a
    healthy running job -- and the longer the upload, the longer the blank.

    a NEW attempt still starts clean: it is a different card, so carrying the old one forward would
    describe hardware this attempt never touched.
    """
    import tempfile as _tempfile

    with _tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", tmp)
        from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
        from tests._helpers.profile import satisfy_sft_profile

        spec = JobSpec(
            run_id="hb-carry",
            model="Qwen/Qwen3.5-9B",
            algorithm="sft",
            environment=EnvironmentSpec(id="team/example"),
            train=TrainSpec(max_examples=8),
        )
        status = runner_submit.submit_job(satisfy_sft_profile(monkeypatch, spec), dry_run=True)
        status.state = "running"
        runner_state._save_status(status)

        runner_status.record_heartbeat(
            "hb-carry",
            {"stage": "sft_step", "attempt": 1, "gpu": {"device_name": "B200", "gpu_util_pct": 91}},
        )
        # the long silent stretch: a checkpoint upload, which sends no gpu sample at all.
        runner_status.record_heartbeat("hb-carry", {"stage": "checkpoint_uploading", "attempt": 1})

        out = runner_status.get_status("hb-carry")
        assert out.gpu_status is not None, "a checkpoint heartbeat blanked the gpu snapshot"
        assert out.gpu_status["device_name"] == "B200"
        assert out.gpu_status["gpu_util_pct"] == 91

        # a retry is a different card; nothing from attempt 1 may describe it.
        runner_status.record_heartbeat("hb-carry", {"stage": "boot", "attempt": 2})
        assert runner_status.get_status("hb-carry").gpu_status is None, (
            "attempt 2 inherited attempt 1's gpu snapshot"
        )


def test_status_sanitizer_preserves_metric_backlog_and_bounds_other_lists():

    metrics = [{"step": step, "reward": step / 1025} for step in range(1025)]
    sanitized = runner_status._sanitize_status_value(
        {"metrics_last": metrics, "other": list(range(32))}
    )

    assert len(sanitized["metrics_last"]) == 1024
    assert sanitized["metrics_last"][0]["step"] == 1
    assert sanitized["metrics_last"][-1]["step"] == 1024
    assert sanitized["other"] == list(range(16))


def test_record_heartbeat_persists_finalize_liveness_ping_with_step(monkeypatch):
    """The finalize-phase daemon pings (liveness=True, stage sft_finalizing, step stamped) must
    land in status.last_heartbeat intact: cancel billing reads .step from the freshest persisted
    heartbeat, and the CLI reads .stage/.ts/.liveness for the status panel."""
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setattr(runner_state, "RUNS_DIR", tmp)
        from flash.core.spec import EnvironmentSpec, JobSpec, TrainSpec
        from tests._helpers.profile import satisfy_sft_profile

        spec = JobSpec(
            run_id="hbf",
            model="Qwen/Qwen3.5-9B",
            algorithm="sft",
            environment=EnvironmentSpec(id="team/example"),
            train=TrainSpec(max_examples=8),
        )
        status = runner_submit.submit_job(satisfy_sft_profile(monkeypatch, spec), dry_run=True)
        status.state = "running"
        runner_state._save_status(status)

        runner_status.record_heartbeat(
            "hbf",
            {"stage": "sft_finalizing", "step": 126, "ts": 123.0, "liveness": True},
        )
        out = runner_status.get_status("hbf")
        assert out.last_heartbeat["stage"] == "sft_finalizing"
        assert out.last_heartbeat["step"] == 126
        assert out.last_heartbeat["liveness"] is True
        assert runner_costs.actual_steps_run(out) == 126, (
            "a cancel during finalize must bill the actual steps trained"
        )


def test_finished_at_frozen_at_terminal_survives_later_updated_at_bumps(monkeypatch):
    """finished_at freezes the training-teardown time on the FIRST terminal transition and is NOT
    moved by later updated_at bumps (heartbeat/deploy/reconcile) — so reconciliation has an
    immutable instance run_end even for a run deployed (or heartbeat-touched) after completion."""
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

        # first terminal transition stamps finished_at to the teardown time
        assert runner_status._update("fa", "done", cost_usd=1.0) is True
        done = runner_status.get_status("fa")
        assert done.finished_at is not None
        teardown = done.finished_at
        assert teardown == done.updated_at

        # a later updated_at bump (a late heartbeat after terminal) must NOT move finished_at
        runner_status.record_heartbeat("fa", {"stage": "rl", "step": 1, "ts": 123.0})
        bumped = runner_status.get_status("fa")
        assert bumped.updated_at >= done.updated_at
        assert bumped.finished_at == teardown

        # a same-state terminal re-write (e.g. terminal cost fields) keeps the original too
        runner_status._update("fa", "done", cost_usd=2.0)
        assert runner_status.get_status("fa").finished_at == teardown


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
        runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()),
        _next_attempt=0,
    )
    candidates = (
        Candidate("runpod", "RTX 4090", 0.69, 24),
        Candidate("runpod", "H100", 1.99, 80),
    )
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *args, **kwargs: Allocation(
            provider="runpod",
            gpu="RTX 4090",
            hourly_usd=0.69,
            min_vram_gb=24,
            candidates=candidates,
        ),
    )
    monkeypatch.setattr(lifecycle.time, "sleep", lambda *_args: None)

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
    assert runner_status.get_status(spec.run_id).remote["attempt"] == 1


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
        runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()),
        _next_attempt=0,
    )
    candidates = (
        Candidate("runpod", "RTX 4090", 0.69, 24),
        Candidate("runpod", "H100", 1.99, 80),
    )
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *_args, **_kwargs: Allocation(
            provider="runpod",
            gpu="RTX 4090",
            hourly_usd=0.69,
            min_vram_gb=24,
            candidates=candidates,
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
    runner_state._save_status(
        provisioned_status(spec, state="running", created_at=100.0),
        _run_deadline_at=300.0,
        _next_attempt=0,
    )
    now = {"value": 100.0}
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: now["value"])
    candidates = (
        Candidate("runpod", "RTX 4090", 0.69, 24),
        Candidate("runpod", "H100", 1.99, 80),
    )
    allocation_walls = []

    def fake_allocate(*_args, **kwargs):
        allocation_walls.append(kwargs["max_wall_seconds"])
        return Allocation(
            provider="runpod",
            gpu="RTX 4090",
            hourly_usd=0.69,
            min_vram_gb=24,
            candidates=candidates,
        )

    monkeypatch.setattr(allocator, "allocate", fake_allocate)
    monkeypatch.setattr(lifecycle.time, "sleep", lambda *_args: None)

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
    candidates = (
        Candidate("runpod", "RTX 4090", 0.69, 24),
        Candidate("runpod", "H100", 1.99, 80),
    )
    allocations = []

    def fake_allocate(*_args, **_kwargs):
        allocations.append(True)
        return Allocation(
            provider="runpod",
            gpu="RTX 4090",
            hourly_usd=0.69,
            min_vram_gb=24,
            candidates=candidates,
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


def test_concurrent_initial_attempt_reservation_has_one_winner(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="threaded-attempts", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict()),
        _next_attempt=0,
    )

    def reserve(_index):
        try:
            return runner_attempts._reserve_attempt(spec.run_id)
        except RuntimeError as exc:
            return str(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(16)))

    assert results.count(0) == 1
    assert all(
        result == 0 or "lacks exact persisted retry authorization" in result for result in results
    )
    assert runner_status._load_status_json(spec.run_id)[runner_state._NEXT_ATTEMPT_KEY] == 1


def test_replacement_reservation_requires_exact_previous_authorization(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec
    from flash.runner.supervise.retry_decision import decide_failure_atomically

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="authorized-attempts", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict())
    )
    assert runner_attempts._reserve_attempt(spec.run_id) == 0
    with pytest.raises(RuntimeError, match="attempt 1 lacks exact persisted retry authorization"):
        runner_attempts._reserve_attempt(spec.run_id)
    raw = runner_status._load_status_json(spec.run_id)
    decision = decide_failure_atomically(
        spec.run_id,
        spec,
        expected_remote=None,
        expected_retry_snapshot=raw[runner_state._RETRY_STATE_KEY],
        failure="poll_error",
        chosen=None,
        candidates=None,
        attempt=0,
    )
    assert decision is not None
    assert decision.plan.retry
    assert runner_attempts._reserve_attempt(spec.run_id) == 1


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
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
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
    assert raw["remote"] == confirmed


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
    assert raw["remote"] == confirmed


def _retry_snapshot_authorizing(spec, attempt: int, *, infra_used: int = 1, floor: float = 0.0):
    from dataclasses import replace

    from flash.runner.supervise.retry_decision import RetryState

    state = RetryState.initial_for_spec(spec)
    if attempt == 0:
        return state.to_snapshot()
    state = replace(
        state,
        infra_used=infra_used,
        usable_vram_floor=floor,
        last_decision_attempt=attempt - 1,
        last_decision_failure="poll_error",
        last_decision_retry=True,
        last_decision_action="retrying allocation (resume from last checkpoint)",
        last_infra_retry_ordinal=infra_used,
    )
    return state.to_snapshot()


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
    assert runner_attempts._reserve_attempt(spec.run_id) == 0
    assert runner_status.get_status(spec.run_id).remote is None
    assert (
        runner_deadlines._spec_with_remaining_wall(
            spec, require_provider_minimum=True, now=180.0
        ).gpu.max_wall_seconds
        == 120
    )
    from flash.runner.supervise.retry_decision import decide_failure_atomically

    raw = runner_status._load_status_json(spec.run_id)
    decision = decide_failure_atomically(
        spec.run_id,
        spec,
        expected_remote=None,
        expected_retry_snapshot=raw[runner_state._RETRY_STATE_KEY],
        failure="poll_error",
        chosen=None,
        candidates=None,
        attempt=0,
    )
    assert decision is not None
    assert decision.plan.retry
    assert runner_attempts._reserve_attempt(spec.run_id) == 1
    raw = runner_status._load_status_json(spec.run_id)
    assert raw[runner_state._NEXT_ATTEMPT_KEY] == 2
    assert raw[runner_state._RUN_DEADLINE_AT_KEY] == 300.0


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
        allocated_gpu="RTX 4090",
        allocated_gpu_count=1,
        allocated_usable_vram_gb=24.0,
    )
    status = provisioned_status(spec, state="running", created_at=100.0, remote=remote)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(
        status,
        _run_deadline_at=300.0,
        _next_attempt=2,
        _retry_state=_retry_snapshot_authorizing(spec, 1),
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 100.0)
    poll_walls = []

    class FailedProvider:
        def poll(self, _handle, poll_spec, *_args, **_kwargs):
            poll_walls.append(poll_spec.gpu.max_wall_seconds)
            return PollResult(False, failure="stalled", detail="worker stopped")

        def cancel(self, _handle):
            return None

        def destroy(self, _handle):
            return None

    monkeypatch.setattr(providers, "get_provider", lambda _name: FailedProvider())
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    resumed = []

    def fake_run_training(_spec, _log, **kwargs):
        resumed.append(expected_next)

    monkeypatch.setattr(runner_lifecycle, "_run_training", fake_run_training)

    status = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert resumed == [expected_next]
    assert poll_walls == [200]
    assert status.state == "running"
    assert status.remote is None


def test_attach_failure_restores_floor_and_skips_equal_cross_provider_candidates(
    monkeypatch, tmp_path
):
    import io

    import flash.providers.core.allocator as allocator
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core import registry as providers
    from flash.providers.core.base import Allocation, Candidate, PollResult

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(
        run_id="attach-floor",
        model="Qwen/Qwen3.5-9B",
        algorithm="grpo",
        gpu=GpuSpec(max_retries=2, max_wall_seconds=200),
    )
    remote = _runpod_remote(
        "endpoint-old",
        "job-old",
        attempt=0,
        allocated_gpu="RTX 4090",
        allocated_gpu_count=1,
        allocated_usable_vram_gb=24.0,
    )
    status = provisioned_status(spec, state="running", remote=remote)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(status, _next_attempt=1)
    candidates = (
        Candidate("lambda", "A10", 1.0, 24),
        Candidate("vast", "RTX 4090", 1.0, 24),
        Candidate("vast", "H100", 2.0, 80),
    )
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *a, **k: Allocation("lambda", "A10", 1.0, 12, candidates),
    )
    submitted = []

    class Provider:
        supports_weight_cache = False

        def poll(self, *_args, **_kwargs):
            return PollResult(False, failure="stalled", detail="worker stopped")

        def cancel(self, _handle):
            return None

        def destroy(self, _handle):
            return None

        def submit_run(self, run_spec, _seed, *, on_handle, attempt, **_kwargs):
            submitted.append((run_spec.gpu.type, run_spec.gpu.count))
            on_handle(_vast_remote(instance_id=8, attempt=attempt))
            return PollResult(True, metrics={"train_tokens": 1})

    provider = Provider()
    monkeypatch.setattr(providers, "get_provider", lambda _name: provider)
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)

    def resume(run_spec, log, **kwargs):
        runner_lifecycle._submit_seed_supervised(
            run_spec,
            run_spec.seed,
            log,
            source_snapshot=kwargs["source_snapshot"],
        )

    monkeypatch.setattr(runner_lifecycle, "_run_training", resume)

    runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert submitted == [("H100", 1)]
    raw = runner_status._load_status_json(spec.run_id)
    retry = raw[runner_state._RETRY_STATE_KEY]
    assert retry["usable_vram_floor"] == 24.0
    assert retry["infra_used"] == 1


def test_attach_cache_fallback_restores_exact_shape_and_drops_managed_cache(monkeypatch, tmp_path):
    import io

    import flash.providers.core.allocator as allocator
    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core import registry as providers
    from flash.providers.core.base import Allocation, Candidate, PollResult
    from flash.runner.accounting.weight_cache import WEIGHT_CACHE_VOLUME_NAME

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-cache",
        model="Qwen/Qwen3.5-9B",
        algorithm="grpo",
        gpu=GpuSpec(
            type="H100",
            count=2,
            max_retries=1,
            max_wall_seconds=200,
            network_volume=WEIGHT_CACHE_VOLUME_NAME,
            network_volume_gb=100,
        ),
    )
    remote = _runpod_remote(
        "endpoint-old",
        "job-old",
        attempt=0,
        allocated_gpu="H100",
        allocated_gpu_count=2,
        allocated_usable_vram_gb=130.4,
    )
    status = provisioned_status(spec, state="running", remote=remote)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(status, _next_attempt=1)
    candidates = (
        Candidate("runpod", "H100", 1.0, 80, 2),
        Candidate("runpod", "B200", 2.0, 180, 1),
    )
    monkeypatch.setattr(
        allocator,
        "allocate",
        lambda *a, **k: Allocation("runpod", "H100", 1.0, 12, candidates, 2),
    )
    submitted = []

    class Provider:
        supports_weight_cache = True

        def poll(self, *_args, **_kwargs):
            return PollResult(False, failure="no_capacity", detail="cached region unavailable")

        def cancel(self, _handle):
            return None

        def destroy(self, _handle):
            return None

        def submit_run(self, run_spec, _seed, *, on_handle, attempt, **_kwargs):
            submitted.append((run_spec.gpu.type, run_spec.gpu.count, run_spec.gpu.network_volume))
            on_handle(_runpod_remote("endpoint-new", "job-new", attempt=attempt))
            return PollResult(True, metrics={"train_tokens": 1})

    provider = Provider()
    monkeypatch.setattr(providers, "get_provider", lambda _name: provider)
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)

    def resume(run_spec, log, **kwargs):
        assert run_spec.gpu.network_volume is None
        runner_lifecycle._submit_seed_supervised(
            run_spec,
            run_spec.seed,
            log,
            source_snapshot=kwargs["source_snapshot"],
        )

    monkeypatch.setattr(runner_lifecycle, "_run_training", resume)

    runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert submitted == [("H100", 2, None)]
    retry = runner_status._load_status_json(spec.run_id)[runner_state._RETRY_STATE_KEY]
    assert retry["cache_used"] == 1
    assert retry["drop_weight_cache"] is True
    assert retry["cache_retry_shape"] == ["runpod", "H100", 2]


def test_nonretryable_attached_failure_tears_down_without_resubmitting(monkeypatch, tmp_path):
    import io

    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core import registry as providers
    from flash.providers.core.base import PollResult

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-terminal-failure",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=2),
    )
    remote = _runpod_remote(
        attempt=0,
        allocated_gpu="RTX 4090",
        allocated_gpu_count=1,
        allocated_usable_vram_gb=24.0,
    )
    runner_state._save_status(
        provisioned_status(spec, state="running", remote=remote),
        _next_attempt=1,
    )
    teardown = []

    class Provider:
        def poll(self, *_args, **_kwargs):
            return PollResult(False, failure="job_failed", detail="worker assertion")

        def cancel(self, _handle):
            teardown.append("cancel")

        def destroy(self, _handle):
            teardown.append("destroy")

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        runner_lifecycle,
        "_run_training",
        lambda *_a, **_k: pytest.fail("nonretryable attached failure resubmitted"),
    )

    status = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert teardown == ["cancel", "destroy"]
    assert status.state == "failed"
    retry = runner_status._load_status_json(spec.run_id)[runner_state._RETRY_STATE_KEY]
    assert retry["infra_used"] == 0
    assert retry["oom_used"] == 0


def test_attach_does_not_reset_a_consumed_infrastructure_budget(monkeypatch, tmp_path):
    import io
    from dataclasses import replace

    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core import registry as providers
    from flash.providers.core.base import PollResult
    from flash.runner.supervise.retry_decision import RetryState

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-consumed-budget",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=1),
    )
    remote = _runpod_remote(
        attempt=5,
        allocated_gpu="H100",
        allocated_gpu_count=1,
        allocated_usable_vram_gb=80.0,
    )
    initial = RetryState.initial_for_spec(spec)
    state = replace(
        initial,
        infra_used=initial.infra_retries,
        usable_vram_floor=24.0,
        last_decision_attempt=4,
        last_decision_failure="stalled",
        last_decision_retry=True,
        last_decision_action="retrying allocation (resume from last checkpoint)",
        last_infra_retry_ordinal=initial.infra_retries,
    )
    runner_state._save_status(
        provisioned_status(spec, state="running", remote=remote),
        _next_attempt=6,
        _retry_state=state.to_snapshot(),
    )

    class Provider:
        def poll(self, *_args, **_kwargs):
            return PollResult(False, failure="stalled", detail="worker stopped")

        def cancel(self, _handle):
            return None

        def destroy(self, _handle):
            return None

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        runner_lifecycle,
        "_run_training",
        lambda *_a, **_k: pytest.fail("consumed retry budget was reset"),
    )

    status = runner_attach.attach_run(spec.run_id, log_stream=io.StringIO())

    assert status.state == "failed"
    retry = runner_status._load_status_json(spec.run_id)[runner_state._RETRY_STATE_KEY]
    assert retry["infra_used"] == state.infra_retries
    assert retry["usable_vram_floor"] == 80.0


def test_atomic_attach_decision_is_immutable_for_attempt(monkeypatch, tmp_path):
    import io

    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core.base import PollResult
    from flash.runner.supervise.retry_decision import (
        decide_failure_atomically,
        retry_candidate_from_remote,
    )

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-decision-race",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=2),
    )
    remote = _runpod_remote(
        attempt=0,
        allocated_gpu="RTX 4090",
        allocated_gpu_count=1,
        allocated_usable_vram_gb=24.0,
    )
    status = provisioned_status(spec, state="running", remote=remote)
    status.source_snapshot = _SOURCE_SNAPSHOT
    runner_state._save_status(status, _next_attempt=1)
    initial_snapshot = runner_status._load_status_json(spec.run_id)[runner_state._RETRY_STATE_KEY]
    stale_context = runner_attach._build_attach_context(spec, remote)
    chosen = retry_candidate_from_remote(remote)

    winner = decide_failure_atomically(
        spec.run_id,
        spec,
        expected_remote=remote,
        expected_retry_snapshot=initial_snapshot,
        failure="stalled",
        chosen=chosen,
        candidates=None,
        attempt=0,
    )
    stale = decide_failure_atomically(
        spec.run_id,
        spec,
        expected_remote=remote,
        expected_retry_snapshot=initial_snapshot,
        failure="job_failed",
        chosen=chosen,
        candidates=None,
        attempt=0,
    )

    assert winner is not None
    assert winner.plan.retry is True
    assert stale is None
    assert sum(decision is not None and decision.plan.retry for decision in (winner, stale)) == 1

    winning_snapshot = winner.snapshot
    conflicting = decide_failure_atomically(
        spec.run_id,
        spec,
        expected_remote=remote,
        expected_retry_snapshot=winning_snapshot,
        failure="job_failed",
        chosen=chosen,
        candidates=None,
        attempt=0,
    )
    assert conflicting is not None
    assert conflicting.plan.retry is True
    assert conflicting.plan.infra_retry_ordinal == winner.plan.infra_retry_ordinal
    assert conflicting.snapshot == winning_snapshot
    assert conflicting.state.last_decision_failure == "stalled"
    assert conflicting.plan.action == winner.plan.action
    assert runner_status._load_status_json(spec.run_id)[runner_state._RETRY_STATE_KEY] == (
        winning_snapshot
    )

    observer_b = runner_attach._build_attach_context(spec, remote)
    observer_c = runner_attach._build_attach_context(spec, remote)
    monkeypatch.setattr(
        runner_lifecycle,
        "_runpod_completed_metrics",
        lambda *_args, **_kwargs: None,
    )
    teardowns = []
    monkeypatch.setattr(
        runner_lifecycle,
        "_strict_teardown_handle",
        lambda *_args, **_kwargs: teardowns.append("teardown") or True,
    )
    launches = []
    monkeypatch.setattr(
        runner_lifecycle,
        "_run_training",
        lambda *_args, **_kwargs: launches.append(1),
    )
    stale_status = runner_attach._handle_failed_attach_poll(
        spec.run_id,
        stale_context,
        PollResult(False, failure="job_failed", detail="stale conflicting poll"),
        io.StringIO(),
    )
    assert stale_status.remote == remote
    assert teardowns == []
    assert launches == []

    recovered = runner_attach._handle_failed_attach_poll(
        spec.run_id,
        observer_b,
        PollResult(False, failure="job_failed", detail="post-win conflicting poll"),
        io.StringIO(),
    )
    assert recovered.state == "running"
    assert recovered.remote is None
    assert teardowns == ["teardown"]
    assert launches == [1]

    repeated = runner_attach._handle_failed_attach_poll(
        spec.run_id,
        observer_c,
        PollResult(False, failure="oom", detail="repeated conflicting poll"),
        io.StringIO(),
    )
    assert repeated.state == "running"
    assert repeated.remote is None
    assert teardowns == ["teardown"]
    assert launches == [1]
    assert runner_status._load_status_json(spec.run_id)[runner_state._RETRY_STATE_KEY] == (
        winning_snapshot
    )


def test_post_win_observer_finishes_immutable_terminal_attach_decision(monkeypatch, tmp_path):
    import io

    from flash.core.spec import GpuSpec, JobSpec
    from flash.providers.core.base import PollResult
    from flash.runner.supervise.retry_decision import (
        decide_failure_atomically,
        retry_candidate_from_remote,
    )

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-terminal-decision-race",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=2),
    )
    remote = _runpod_remote(
        attempt=0,
        allocated_gpu="RTX 4090",
        allocated_gpu_count=1,
        allocated_usable_vram_gb=24.0,
    )
    runner_state._save_status(
        provisioned_status(spec, state="running", remote=remote),
        _next_attempt=1,
    )
    initial_snapshot = runner_status._load_status_json(spec.run_id)[runner_state._RETRY_STATE_KEY]
    winner = decide_failure_atomically(
        spec.run_id,
        spec,
        expected_remote=remote,
        expected_retry_snapshot=initial_snapshot,
        failure="job_failed",
        chosen=retry_candidate_from_remote(remote),
        candidates=None,
        attempt=0,
    )
    assert winner is not None
    assert winner.plan.retry is False
    winning_snapshot = winner.snapshot
    observer_b = runner_attach._build_attach_context(spec, remote)
    observer_c = runner_attach._build_attach_context(spec, remote)

    monkeypatch.setattr(
        runner_lifecycle,
        "_runpod_completed_metrics",
        lambda *_args, **_kwargs: None,
    )
    teardowns = []
    monkeypatch.setattr(
        runner_lifecycle,
        "_strict_teardown_handle",
        lambda *_args, **_kwargs: teardowns.append("teardown") or True,
    )
    monkeypatch.setattr(
        runner_lifecycle,
        "_run_training",
        lambda *_args, **_kwargs: pytest.fail("terminal decision launched replacement"),
    )

    recovered = runner_attach._handle_failed_attach_poll(
        spec.run_id,
        observer_b,
        PollResult(False, failure="stalled", detail="post-win conflicting poll"),
        io.StringIO(),
    )
    assert recovered.state == "failed"
    assert teardowns == ["teardown"]
    repeated = runner_attach._handle_failed_attach_poll(
        spec.run_id,
        observer_c,
        PollResult(False, failure="oom", detail="repeated conflicting poll"),
        io.StringIO(),
    )
    assert repeated.state == "failed"
    assert teardowns == ["teardown"]
    assert runner_status._load_status_json(spec.run_id)[runner_state._RETRY_STATE_KEY] == (
        winning_snapshot
    )


def test_attach_expired_run_adopts_completed_attempt_at_deadline(monkeypatch, tmp_path):
    import io

    import flash.providers.artifacts.hf as hf_artifacts
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
    runner_state._save_status(
        provisioned_status(spec, state="running", created_at=100.0, remote=remote),
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 221.0)
    completion_checks = []
    real_completed_metrics = lifecycle._completed_attempt_metrics

    def artifact_reader(_repo, path):
        def read(force=False):
            if path.endswith("/vast_attempt0.json"):
                return (
                    '{"attempt":0,"error":"","ok":true,"retriable":false,'
                    '"run_id":"attach-expired-completed","ts":219.0}'
                )
            if path.endswith("/metrics.json"):
                return '{"wall_seconds":5.0}'
            return None

        return read

    def completed_metrics(*args, **kwargs):
        completion_checks.append((args, kwargs))
        return real_completed_metrics(*args, **kwargs)

    monkeypatch.setattr(hf_artifacts, "make_hf_text_reader", artifact_reader)
    monkeypatch.setattr(lifecycle, "_completed_attempt_metrics", completed_metrics)
    monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        lifecycle,
        "_strict_teardown_handle",
        lambda *_args, **_kwargs: pytest.fail(
            "completed attempt must not be torn down before adoption"
        ),
    )
    log = io.StringIO()

    status = runner_attach.attach_run(spec.run_id, log_stream=log)

    assert len(completion_checks) == 1
    _, kwargs = completion_checks[0]
    assert kwargs == {
        "provider": "vast",
        "attempt": 0,
        "launch_floor": 101.0,
        "deadline_at": 220.0,
        "log": log,
    }
    assert status.state == "done"
    assert status.remote == remote
    assert status.error is None
    assert runner_status._load_status_json(spec.run_id)[runner_state._CLEANUP_REMOTES_KEY] == [
        remote
    ]
    assert "adopted a completed attempt at the wall deadline" in log.getvalue()


def test_attach_adoption_prices_a_multi_card_run_for_every_card(monkeypatch, tmp_path):
    # same gap as the reconciler's, on the OTHER adoption path: attach_run pops the allocation
    # stamp off the persisted remote before building the handle, and only restores it on the
    # poll-success return. an adopted run leaves through _completed_attempt_metrics instead, whose
    # payload is the worker's own metrics.json -- and the worker never knew the card count. without
    # the carry, a 4-card vast run recovered after a control-plane restart prices its wall as one.
    import io

    import flash.providers.artifacts.hf as hf_artifacts
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
    runner_state._save_status(
        provisioned_status(spec, state="running", created_at=100.0, remote=remote),
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 221.0)

    def artifact_reader(_repo, path):
        def read(force=False):
            if path.endswith("/vast_attempt0.json"):
                return (
                    '{"attempt":0,"error":"","ok":true,"retriable":false,'
                    '"run_id":"attach-adopt-multicard","ts":219.0}'
                )
            if path.endswith("/metrics.json"):
                # exactly what the worker writes: a wall, and nothing about the allocation.
                return '{"wall_seconds":3600.0}'
            return None

        return read

    adopted = {}
    real_adopt = lifecycle._adopt_completed_attempt

    def capture_adopt(run_id, adopt_spec, expected_remote, metrics, **kwargs):
        adopted.update(metrics)
        return real_adopt(run_id, adopt_spec, expected_remote, metrics, **kwargs)

    monkeypatch.setattr(hf_artifacts, "make_hf_text_reader", artifact_reader)
    monkeypatch.setattr(lifecycle, "_adopt_completed_attempt", capture_adopt)
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


def test_attach_success_marker_with_lagging_metrics_stays_pending(monkeypatch, tmp_path):
    import io

    import flash.runner.supervise.attach as attach
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-metrics-pending",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(hf_repo="org/repo"),
        gpu=GpuSpec(max_wall_seconds=120),
    )
    remote = _vast_remote(instance_id=7, attempt=0, started_ts=101.0)
    runner_state._save_status(
        provisioned_status(spec, state="running", created_at=100.0, remote=remote),
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 221.0)
    monkeypatch.setattr(
        lifecycle,
        "_completed_attempt_metrics",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            lifecycle._CompletedAttemptPending("successful marker; waiting for metrics.json")
        ),
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


def test_completed_attempt_metrics_rereads_a_marker_that_is_not_visible_yet(monkeypatch):
    """Recovery must RE-READ the marker, not sample it once.

    The marker and metrics are two uploads of the same completion racing HF read-after-write lag, so
    a marker that is invisible on the first read routinely surfaces moments later. Reading once would
    report ABSENT and let the caller tear down an attempt that had already finished its paid work.
    """
    import io

    import flash.providers._lifecycle.instances.poll_instance as instance_poll
    import flash.providers._lifecycle.instances.terminal_artifacts as ta
    import flash.providers.artifacts.hf as hf_artifacts
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import JobSpec, TrainSpec

    spec = JobSpec(
        run_id="lagging-marker",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(hf_repo="org/repo"),
    )
    # the reread window must stay NONZERO: the shared cutoff is `now + tries * wait_s`, so a zero
    # wait would expire the budget before the second read and the test would pass for the wrong
    # reason. sleep is faked instead, leaving the retry loop real.
    monkeypatch.setattr(instance_poll, "_TERMINAL_REREAD_WAIT_S", 5.0)
    monkeypatch.setattr(lifecycle.time, "time", lambda: 201.0)
    monkeypatch.setattr(ta.time, "sleep", lambda _s: None)
    marker_reads = {"n": 0}

    def artifact_reader(_repo, path):
        def read(force=False):
            if path.endswith("/vast_attempt0.json"):
                marker_reads["n"] += 1
                # invisible on the first read, exactly as HF lag presents it
                if marker_reads["n"] <= 1:
                    return None
                return (
                    '{"attempt":0,"error":"","ok":true,"retriable":false,'
                    '"run_id":"lagging-marker","ts":199.0}'
                )
            return json.dumps({"train_tokens": 4096})

        return read

    monkeypatch.setattr(hf_artifacts, "make_hf_text_reader", artifact_reader)

    assert lifecycle._completed_attempt_metrics(
        spec,
        provider="vast",
        attempt=0,
        launch_floor=100.0,
        deadline_at=200.0,
        log=io.StringIO(),
    ) == {"train_tokens": 4096}
    assert marker_reads["n"] >= 2  # a single sample would have reported ABSENT


def test_completed_attempt_metrics_never_adopts_an_unverifiable_marker(monkeypatch):
    """Recovery must not adopt an artifact it cannot tie to this attempt, and must SAY so.

    Live polling classifies an unverifiable marker as a terminal failure while recovery used to
    swallow the decode error into a bare ``None``, so identical bytes produced different stories
    depending on which layer observed the attempt. Both now route through one resolver: recovery
    still returns ``None`` (there is no completed work to adopt) but names the artifact instead of
    logging silence, and a corrupt marker can never be mistaken for a success.
    """
    import io

    import flash.providers.artifacts.hf as hf_artifacts
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import JobSpec, TrainSpec

    spec = JobSpec(
        run_id="corrupt-marker",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(hf_repo="org/repo"),
    )

    def artifact_reader(_repo, path):
        # a well-formed marker body that speaks for a DIFFERENT run: it decodes as json but can
        # never be validated against this attempt's identity.
        def read(force=False):
            if path.endswith("/vast_attempt0.json"):
                return (
                    '{"attempt":0,"error":"","ok":true,"retriable":false,'
                    '"run_id":"some-other-run","ts":199.0}'
                )
            return json.dumps({"train_tokens": 4096})

        return read

    monkeypatch.setattr(hf_artifacts, "make_hf_text_reader", artifact_reader)
    monkeypatch.setattr(lifecycle.time, "time", lambda: 201.0)
    log = io.StringIO()

    assert (
        lifecycle._completed_attempt_metrics(
            spec,
            provider="vast",
            attempt=0,
            launch_floor=100.0,
            deadline_at=200.0,
            log=log,
        )
        is None
    )
    assert "invalid or unverifiable" in log.getvalue()


@pytest.mark.parametrize(
    ("marker_ts", "expected"),
    [
        (280.0, {"wall_seconds": 5.0}),
        (7420.0, None),
    ],
)
def test_completed_attempt_metrics_bounds_marker_to_wall_grace(monkeypatch, marker_ts, expected):
    import flash.providers.artifacts.hf as hf_artifacts
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import JobSpec, TrainSpec

    spec = JobSpec(
        run_id="late-marker-complete",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
    )
    monkeypatch.setattr(lifecycle.time, "time", lambda: 8000.0)

    def artifact_reader(_repo, path):
        def read(force=False):
            if path.endswith("/vast_attempt0.json"):
                return (
                    '{"attempt":0,"error":"","ok":true,"retriable":false,'
                    f'"run_id":"late-marker-complete","ts":{marker_ts}}}'
                )
            if path.endswith("/metrics.json"):
                return '{"wall_seconds":5.0}'
            return None

        return read

    monkeypatch.setattr(hf_artifacts, "make_hf_text_reader", artifact_reader)

    metrics = lifecycle._completed_attempt_metrics(
        spec,
        provider="vast",
        attempt=0,
        launch_floor=101.0,
        deadline_at=220.0,
    )

    assert metrics == expected


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
    completion_checks = []
    monkeypatch.setattr(
        lifecycle,
        "_completed_attempt_metrics",
        lambda *args, **kwargs: completion_checks.append((args, kwargs)) or None,
    )

    class Provider:
        def poll(self, *_args, **_kwargs):
            polled.append(True)
            raise AssertionError("expired recovery must not poll")

        def cancel(self, handle):
            teardown.append(("cancel", handle.to_dict()))

        def destroy(self, handle):
            teardown.append(("destroy", handle.to_dict()))

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
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
    assert len(completion_checks) == 1
    assert completion_checks[0][1]["attempt"] == 0
    assert [action for action, _handle in teardown] == ["cancel", "destroy"]
    assert gc_runs == [spec.run_id]
    assert status.state == "failed"
    assert status.remote["endpoint_id"] == "endpoint-old"
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
    runner_state._save_status(
        provisioned_status(spec, state="running", created_at=100.0, remote=remote),
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 221.0)

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


def test_runpod_submit_propagates_attempt_to_worker_environment_and_handle(monkeypatch):
    import flash.providers._lifecycle.net.worker as train
    import flash.providers.runpod.execution.job_execution as job_execution
    import flash.providers.runpod.execution.polling as polling
    from flash.core.spec import JobSpec
    from flash.providers.core.base import PollResult

    spec = JobSpec(run_id="worker-attempt", model="Qwen/Qwen3.5-9B", algorithm="sft")
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
        source_snapshot=_SOURCE_SNAPSHOT,
        on_handle=handles.append,
        deadline_at=10_000_000_000.0,
    )

    assert payloads[0]["env"]["ATTEMPT"] == "2"
    assert handles[0]["attempt"] == 2


def test_fail_blocked_recovery_adopts_completed_handleless_attempt(monkeypatch, tmp_path):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(run_id="blocked-complete", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="provisioning",
            spec=spec.to_dict(),
            created_at=100.0,
        ),
        _run_deadline_at=86500.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(
        runtime,
        "_handleless_completed_metrics",
        lambda *_args, **_kwargs: {"wall_seconds": 5.0},
    )

    assert runtime._fail_blocked_recovery(spec, "recovery blocked") is True

    status = runner_status.get_status(spec.run_id)
    assert status.state == "done"
    assert status.remote is None
    assert status.error is None


def test_fail_blocked_recovery_keeps_success_with_lagging_metrics_pending(monkeypatch, tmp_path):
    import flash.runner.supervise.lifecycle as lifecycle
    import flash.server.platform.runtime as runtime
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="blocked-pending", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="provisioning",
            spec=spec.to_dict(),
        ),
        _next_attempt=1,
    )
    monkeypatch.setattr(
        runtime,
        "_handleless_completed_metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            lifecycle._CompletedAttemptPending("successful marker; waiting for metrics.json")
        ),
    )

    assert runtime._fail_blocked_recovery(spec, "recovery blocked") is False
    status = runner_status.get_status(spec.run_id)
    assert status.state == "provisioning"
    assert status.remote is None
    assert status.error is None


@pytest.mark.parametrize("state_name", ["queued", "provisioning"])
def test_handleless_initial_attempt_uses_no_retry_budget(monkeypatch, tmp_path, state_name):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id=f"initial-{state_name}",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=1),
    )
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state=state_name, spec=spec.to_dict())
    )

    captured = {}
    monkeypatch.setattr(
        runtime,
        "_start_resubmit",
        lambda *_args, **kwargs: captured.update(kwargs) or True,
    )

    assert runtime._start_handleless_resubmit(spec, state_name) is True
    assert captured["next_attempt"] == 0
    snapshot = captured["expected_retry_snapshot"]
    assert snapshot["infra_used"] == 0
    assert snapshot["last_decision_attempt"] is None


def test_handleless_reserved_attempt_persists_and_reuses_one_candidate_less_retry(
    monkeypatch, tmp_path
):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="reserved-undecided",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=1),
    )
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict()),
        _next_attempt=1,
    )

    captured = []
    monkeypatch.setattr(
        runtime,
        "_start_resubmit",
        lambda *_args, **kwargs: captured.append(kwargs) or None,
    )

    assert runtime._start_handleless_resubmit(spec, "provisioning") is None
    assert runtime._start_handleless_resubmit(spec, "provisioning") is None

    assert captured[0] == captured[1]
    assert captured[0]["next_attempt"] == 1
    snapshot = captured[0]["expected_retry_snapshot"]
    assert snapshot["infra_used"] == 1
    assert snapshot["last_decision_attempt"] == 0
    assert snapshot["last_decision_failure"] == "poll_error"


def test_reserved_recovery_rejects_a_newer_retry_snapshot_before_allocation(monkeypatch, tmp_path):
    import io

    from flash.core.spec import GpuSpec, JobSpec
    from flash.runner.supervise import seed_submission as submission
    from flash.runner.supervise.retry_decision import RetryState, decide_failure_atomically

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="reserved-recovery-snapshot",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=2, max_wall_seconds=3600),
    )
    claimed = _retry_snapshot_authorizing(spec, 1)
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict()),
        _next_attempt=2,
        _retry_state=claimed,
    )
    decision = decide_failure_atomically(
        spec.run_id,
        spec,
        expected_remote=None,
        expected_retry_snapshot=claimed,
        failure="poll_error",
        chosen=None,
        candidates=None,
        attempt=1,
    )
    assert decision is not None
    assert decision.plan.retry
    ctx = submission._SubmitContext(
        spec=spec,
        seed=spec.seed,
        log=io.StringIO(),
        runtime_secrets={},
        source_snapshot=_SOURCE_SNAPSHOT,
        retry_state=RetryState.from_snapshot(spec, claimed),
        reserved_retry=(1, None, None, claimed),
    )

    with pytest.raises(RuntimeError, match="retry snapshot changed before attempt preparation"):
        submission._prepare_attempt(ctx)


def test_handleless_stale_older_decision_is_replaced_before_launch(monkeypatch, tmp_path):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec
    from flash.runner.supervise.retry_decision import decide_failure_atomically

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="stale-handleless-decision",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=2),
    )
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict()),
        _next_attempt=1,
    )
    raw = runner_status._load_status_json(spec.run_id)
    first = decide_failure_atomically(
        spec.run_id,
        spec,
        expected_remote=None,
        expected_retry_snapshot=raw[runner_state._RETRY_STATE_KEY],
        failure="poll_error",
        chosen=None,
        candidates=None,
        attempt=0,
    )
    assert first is not None
    assert first.plan.retry
    assert runner_attempts._reserve_attempt(spec.run_id) == 1

    captured = {}
    monkeypatch.setattr(
        runtime,
        "_start_resubmit",
        lambda *_args, **kwargs: captured.update(kwargs) or True,
    )

    assert runtime._start_handleless_resubmit(spec, "provisioning") is True
    assert captured["next_attempt"] == 2
    snapshot = captured["expected_retry_snapshot"]
    assert snapshot["last_decision_attempt"] == 1
    assert snapshot["infra_used"] == 2


def test_handleless_lost_attempt_with_zero_retries_blocks_launch(monkeypatch, tmp_path):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="zero-retry-handleless",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=0),
    )
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict()),
        _next_attempt=1,
    )

    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)
    monkeypatch.setattr(
        runtime,
        "_start_resubmit",
        lambda *_args, **_kwargs: pytest.fail("retry=false claimed a replacement"),
    )

    assert runtime._start_handleless_resubmit(spec, "provisioning") is False
    snapshot = runner_status._load_status_json(spec.run_id)[runner_state._RETRY_STATE_KEY]
    assert snapshot["last_decision_attempt"] == 0
    assert snapshot["infra_used"] == 0


def test_handleless_retry_false_blocks_background_launch(monkeypatch, tmp_path):
    import flash.server.platform.runtime as runtime
    from flash.core.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="blocked-handleless-launch",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        gpu=GpuSpec(max_retries=0),
    )
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict()),
        _next_attempt=1,
    )
    monkeypatch.setattr(runtime, "_recovery_block_reason", lambda _spec: None)
    monkeypatch.setattr(runtime, "_confirm_run_clear", lambda _spec: True)
    monkeypatch.setattr(
        runtime,
        "_start_resubmit",
        lambda *_args, **_kwargs: pytest.fail("retry=false launched a background replacement"),
    )
    monkeypatch.setattr(runner_reporting, "_report_status", lambda _status: None)

    runtime._resubmit_recovered_runs([(spec, "provisioning")])

    assert runner_status.get_status(spec.run_id).state == "failed"


def test_recovery_launch_cas_requires_exact_snapshot_and_attempt(monkeypatch, tmp_path):
    from flash.core.spec import JobSpec

    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="recovery-cas", model="Qwen/Qwen3.5-9B", algorithm="sft")
    runner_state._save_status(
        runner_state.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
    )
    raw = runner_status._load_status_json(spec.run_id)
    snapshot = raw[runner_state._RETRY_STATE_KEY]

    assert not runner_reconciliation._compare_and_prepare_resubmit(
        spec.run_id,
        None,
        expected_state="queued",
        expected_retry_snapshot={**snapshot, "infra_used": 1},
        next_attempt=0,
    )
    assert not runner_reconciliation._compare_and_prepare_resubmit(
        spec.run_id,
        None,
        expected_state="queued",
        expected_retry_snapshot=snapshot,
        next_attempt=1,
    )
    assert runner_reconciliation._compare_and_prepare_resubmit(
        spec.run_id,
        None,
        expected_state="queued",
        expected_retry_snapshot=snapshot,
        next_attempt=0,
    )
    assert runner_status._load_status_json(spec.run_id)[runner_state._NEXT_ATTEMPT_KEY] == 1


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
        runner_state.RunStatus(
            run_id=spec.run_id,
            state="provisioning",
            spec=spec.to_dict(),
            created_at=100.0,
        ),
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner_lifecycle.time, "time", lambda: 221.0)
    monkeypatch.setattr(
        runtime,
        "_handleless_completed_metrics",
        lambda *_args, **_kwargs: {"wall_seconds": 5.0},
    )

    raw = runner_status._load_status_json(spec.run_id)
    assert (
        runtime._start_resubmit(
            spec,
            expected_remote=None,
            expected_retry_snapshot=raw[runner_state._RETRY_STATE_KEY],
            next_attempt=1,
        )
        is False
    )

    status = runner_status.get_status(spec.run_id)
    assert status.state == "done"
    assert status.remote is None
    assert status.error is None


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
    remote = {"provider": "runpod", "endpoint_id": "ep-stale", "attempt": 0}
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
    assert started[0][1]["next_attempt"] == 0
    assert isinstance(started[0][1]["expected_retry_snapshot"], dict)
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
    monkeypatch.setattr(runtime, "_handleless_completed_metrics", lambda *a, **k: None)
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
    assert attempts[0][1] == attempts[1][1]
    assert attempts[0][1]["expected_remote"] is None
    assert attempts[0][1]["expected_state"] == "provisioning"
    assert attempts[0][1]["next_attempt"] == 0


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
    monkeypatch.setattr(runtime, "_handleless_completed_metrics", lambda *a, **k: None)
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


@pytest.mark.parametrize(("now", "pending"), [(201.0, True), (321.0, False)])
def test_completed_attempt_metrics_bounds_success_marker_metrics_grace(monkeypatch, now, pending):
    import flash.providers._lifecycle.instances.poll_instance as instance_poll
    import flash.providers.artifacts.hf as hf_artifacts
    import flash.runner.supervise.lifecycle as lifecycle
    from flash.core.spec import JobSpec, TrainSpec

    spec = JobSpec(
        run_id="metrics-lag",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(hf_repo="org/repo"),
    )
    monkeypatch.setattr(instance_poll, "_TERMINAL_REREAD_RETRIES", 1)
    monkeypatch.setattr(instance_poll, "_TERMINAL_REREAD_WAIT_S", 0.0)
    monkeypatch.setattr(instance_poll, "_METRICS_AFTER_SUCCESS_RETRIES", 1)
    monkeypatch.setattr(instance_poll, "_METRICS_AFTER_SUCCESS_WAIT_S", 0.0)
    monkeypatch.setattr(lifecycle.time, "time", lambda: now)

    def artifact_reader(_repo, path):
        def read(force=False):
            if path.endswith("/vast_attempt0.json"):
                return (
                    '{"attempt":0,"error":"","ok":true,"retriable":false,'
                    '"run_id":"metrics-lag","ts":199.0}'
                )
            return None

        return read

    monkeypatch.setattr(hf_artifacts, "make_hf_text_reader", artifact_reader)

    def call():
        return lifecycle._completed_attempt_metrics(
            spec,
            provider="vast",
            attempt=0,
            launch_floor=100.0,
            deadline_at=200.0,
        )

    if pending:
        with pytest.raises(lifecycle._CompletedAttemptPending, match=r"metrics\.json"):
            call()
    else:
        assert call() is None


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
