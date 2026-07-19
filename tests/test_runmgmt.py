"""Tests for run-management helpers (runs/cost/cancel) — no GPU/network."""

from __future__ import annotations

import importlib
import json
import tempfile

import pytest

_RUNPOD_FINGERPRINT = "rpk-0123456789ab"


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


def test_background_run_redacts_private_exception_content(monkeypatch, caplog):
    import logging
    from types import SimpleNamespace

    import flash.runner as runner
    from flash.spec import JobSpec

    spec = JobSpec(run_id="background-private", model="Qwen/Qwen3.5-4B", algorithm="sft")
    updates = []

    def fail_run(_spec):
        raise RuntimeError("private provider response")

    monkeypatch.setattr(runner, "_run_job", fail_run)
    monkeypatch.setattr(runner, "get_status", lambda _run_id: SimpleNamespace(state="running"))
    monkeypatch.setattr(
        runner,
        "_update",
        lambda run_id, state, **kwargs: updates.append((run_id, state, kwargs)),
    )

    with caplog.at_level(logging.WARNING):
        runner._run_job_background(spec)

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
        import flash.runner as runner

        importlib.reload(runner)
        # fixed constant; redirect to tmp via monkeypatch so it's restored after the test.
        monkeypatch.setattr(runner, "RUNS_DIR", tmp)
        from flash.spec import JobSpec

        # two dry-run records
        for rid in ("a", "b"):
            runner.submit_job(
                JobSpec(run_id=rid, model="Qwen/Qwen3.5-4B", algorithm="grpo"),
                dry_run=True,
            )
        runs = {r.run_id for r in runner.list_runs()}
        assert {"a", "b"} <= runs

        # cancel a non-terminal run (force it to a running-ish state first; go through
        # _save_status, not _update, since _update now refuses to resurrect a terminal
        # state like the submitted dry_run).
        running = runner.get_status("a")
        running.state = "running"
        runner._save_status(running)
        status = runner.cancel_run("a")
        assert status.state == "cancelled"

        # cancelling a terminal run is a no-op
        same = runner.cancel_run("b")  # b is dry_run (terminal-ish)
        assert same.state in {"dry_run", "cancelled"}


def test_get_status_tolerates_stale_unknown_keys(monkeypatch):
    # A status JSON written by an OLDER control plane can carry a since-removed field (e.g.
    # `resume_seed_index` from the pre-#317 multi-seed era); `~/.flash/runs/*.json` is never GC'd,
    # so those files persist across an upgrade. get_status/list_runs must drop unknown keys rather
    # than 500 (a strict RunStatus(**d) would TypeError, and callers catch only FileNotFoundError).
    import json
    import os

    with tempfile.TemporaryDirectory() as tmp:
        import flash.runner as runner

        importlib.reload(runner)
        monkeypatch.setattr(runner, "RUNS_DIR", tmp)
        stale = {
            "run_id": "old",
            "state": "done",
            "spec": {},
            "cost_usd": 2.0,
            "resume_seed_index": 3,  # removed field
            "totally_unknown_future_key": "x",  # forward-compat unknown field
        }
        os.makedirs(tmp, exist_ok=True)
        with open(runner.runs_file_path("old", ".json"), "w") as f:
            json.dump(stale, f)

        s = runner.get_status("old")
        assert s.run_id == "old"
        assert s.state == "done"
        assert s.cost_usd == 2.0
        assert not hasattr(s, "resume_seed_index")
        assert "old" in {r.run_id for r in runner.list_runs()}


def test_submit_job_persists_quote_and_completion_charges_it(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.cost.spec import estimate_for_spec
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "_assign_resolved_env_sha", lambda spec: spec)

    seen: dict[str, float] = {}

    def fake_run(spec, runtime_secrets=None):
        status = runner.get_status(spec.run_id)
        priced_spec = JobSpec.from_dict(status.spec)
        seen["estimate"] = float(status.estimated_cost_usd)
        seen["expected"] = float(estimate_for_spec(priced_spec).total_usd)
        runner._update(
            spec.run_id,
            "done",
            cost_usd=runner._status_estimated_charge(status, priced_spec, fallback=0.01),
        )

    monkeypatch.setattr(runner, "_run_job", fake_run)

    status = runner.submit_job(
        JobSpec(
            run_id="quoted",
            model="Qwen/Qwen3.5-4B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=2),
            gpu=GpuSpec(type="RTX 4090"),
        )
    )

    assert seen["estimate"] == pytest.approx(seen["expected"])
    assert status.estimated_cost_usd == pytest.approx(seen["expected"])
    assert status.cost_usd == pytest.approx(seen["expected"])
    raw = runner._load_status_json(status.run_id)
    assert raw[runner._RUN_DEADLINE_AT_KEY] == pytest.approx(
        status.created_at + JobSpec.from_dict(status.spec).gpu.max_wall_seconds
    )
    assert raw[runner._NEXT_ATTEMPT_KEY] == 0


def test_missing_persisted_run_deadline_is_rejected(monkeypatch, tmp_path):
    import os

    import flash.runner as runner
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="missing-deadline",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=900),
    )
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            created_at=123.0,
        )
    )
    path = runner.runs_file_path(spec.run_id, ".json")
    raw = runner._load_status_json(spec.run_id)
    raw.pop(runner._RUN_DEADLINE_AT_KEY)
    with open(path, "w") as file:
        json.dump(raw, file)
    assert os.path.exists(path)

    with pytest.raises(RuntimeError, match="persisted run wall deadline is missing"):
        runner._load_run_deadline_at(spec.run_id)


@pytest.mark.parametrize(
    "unsafe_now",
    [True, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), "1000"],
)
def test_remaining_run_wall_seconds_rejects_unsafe_current_clock(monkeypatch, tmp_path, unsafe_now):
    import flash.runner as runner
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="unsafe-current-clock",
        model="Qwen/Qwen3.5-4B",
        gpu=GpuSpec(max_wall_seconds=900),
    )
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            created_at=123.0,
        ),
        _run_deadline_at=1023.0,
    )

    with pytest.raises(ValueError, match="current clock is invalid"):
        runner._remaining_run_wall_seconds(spec.run_id, now=unsafe_now)


def test_persisted_run_deadline_must_match_canonical_value(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="mismatched-deadline",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=900),
    )
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            created_at=123.0,
        ),
        _run_deadline_at=1024.0,
    )

    with pytest.raises(RuntimeError, match="does not match canonical"):
        runner._load_run_deadline_at(spec.run_id)


@pytest.mark.parametrize("deadline", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_persisted_run_deadline_rejects_nonpositive_or_nonfinite_values(
    monkeypatch, tmp_path, deadline
):
    import flash.runner as runner
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="invalid-deadline", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner._save_status(
        runner.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()),
        _run_deadline_at=deadline,
    )

    with pytest.raises(RuntimeError, match="run wall deadline is invalid"):
        runner._load_run_deadline_at(spec.run_id)


@pytest.mark.parametrize("created_at", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_status_save_rejects_invalid_creation_time(monkeypatch, tmp_path, created_at):
    import flash.runner as runner
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="invalid-legacy-deadline", model="Qwen/Qwen3.5-4B", algorithm="sft")
    with pytest.raises(RuntimeError, match="run wall deadline is invalid"):
        runner._save_status(
            runner.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                created_at=created_at,
            )
        )


def test_record_heartbeat_updates_status_without_state_change(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import flash.runner as runner

        importlib.reload(runner)
        monkeypatch.setattr(runner, "RUNS_DIR", tmp)
        from flash.spec import JobSpec, TrainSpec

        status = runner.submit_job(
            JobSpec(
                run_id="hb",
                model="Qwen/Qwen3.5-4B",
                algorithm="sft",
                train=TrainSpec(max_examples=8),
            ),
            dry_run=True,
        )
        status.state = "running"
        runner._save_status(status)

        runner.record_heartbeat(
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

        out = runner.get_status("hb")
        assert out.state == "running"
        assert out.last_heartbeat["stage"] == "sft_step"
        assert out.last_heartbeat["step"] == 20
        assert out.gpu_status["device_name"] == "RTX 5090"
        assert out.gpu_status["gpu_util_pct"] == 94


def test_record_heartbeat_persists_finalize_liveness_ping_with_step(monkeypatch):
    """The finalize-phase daemon pings (liveness=True, stage sft_finalizing, step stamped) must
    land in status.last_heartbeat intact: cancel billing reads .step from the freshest persisted
    heartbeat, and the CLI reads .stage/.ts/.liveness for the status panel."""
    with tempfile.TemporaryDirectory() as tmp:
        import flash.runner as runner

        importlib.reload(runner)
        monkeypatch.setattr(runner, "RUNS_DIR", tmp)
        from flash.spec import JobSpec, TrainSpec

        status = runner.submit_job(
            JobSpec(
                run_id="hbf",
                model="Qwen/Qwen3.5-4B",
                algorithm="sft",
                train=TrainSpec(max_examples=8),
            ),
            dry_run=True,
        )
        status.state = "running"
        runner._save_status(status)

        runner.record_heartbeat(
            "hbf",
            {"stage": "sft_finalizing", "step": 126, "ts": 123.0, "liveness": True},
        )
        out = runner.get_status("hbf")
        assert out.last_heartbeat["stage"] == "sft_finalizing"
        assert out.last_heartbeat["step"] == 126
        assert out.last_heartbeat["liveness"] is True
        assert runner.actual_steps_run(out) == 126, (
            "a cancel during finalize must bill the actual steps trained"
        )


def test_finished_at_frozen_at_terminal_survives_later_updated_at_bumps(monkeypatch):
    """finished_at freezes the training-teardown time on the FIRST terminal transition and is NOT
    moved by later updated_at bumps (heartbeat/deploy/reconcile) — so reconciliation has an
    immutable instance run_end even for a run deployed (or heartbeat-touched) after completion."""
    with tempfile.TemporaryDirectory() as tmp:
        import flash.runner as runner

        importlib.reload(runner)
        monkeypatch.setattr(runner, "RUNS_DIR", tmp)
        from flash.spec import JobSpec

        runner.submit_job(
            JobSpec(run_id="fa", model="Qwen/Qwen3.5-4B", algorithm="grpo"), dry_run=True
        )
        s = runner.get_status("fa")
        s.state = "running"
        s.finished_at = None  # dry_run created via direct state set, never stamped finished_at
        runner._save_status(s)

        # first terminal transition stamps finished_at to the teardown time
        assert runner._update("fa", "done", cost_usd=1.0) is True
        done = runner.get_status("fa")
        assert done.finished_at is not None
        teardown = done.finished_at
        assert teardown == done.updated_at

        # a later updated_at bump (a late heartbeat after terminal) must NOT move finished_at
        runner.record_heartbeat("fa", {"stage": "rl", "step": 1, "ts": 123.0})
        bumped = runner.get_status("fa")
        assert bumped.updated_at >= done.updated_at
        assert bumped.finished_at == teardown

        # a same-state terminal re-write (e.g. terminal cost fields) keeps the original too
        runner._update("fa", "done", cost_usd=2.0)
        assert runner.get_status("fa").finished_at == teardown


def test_legacy_finished_at_backfill_uses_prior_updated_at_on_same_state_touch(monkeypatch):
    """A LEGACY run (finished_at never stamped) that is ALREADY terminal and gets a same-state
    field-only touch (e.g. billing_state via _update(run_id, current_state, ...)) must backfill
    finished_at from the PRIOR persisted terminal updated_at, NOT the freshly-set now — otherwise
    a routine post-completion update would move the billed run_end / reconcile window forward."""
    with tempfile.TemporaryDirectory() as tmp:
        import flash.runner as runner

        importlib.reload(runner)
        monkeypatch.setattr(runner, "RUNS_DIR", tmp)
        from flash.spec import JobSpec

        runner.submit_job(
            JobSpec(run_id="leg", model="Qwen/Qwen3.5-4B", algorithm="grpo"), dry_run=True
        )
        # Simulate a legacy record: already `done`, real teardown time in updated_at, no finished_at.
        teardown = 1_000.0
        s = runner.get_status("leg")
        s.state = "done"
        s.updated_at = teardown
        s.finished_at = None
        runner._save_status(s)

        # A same-state field-only touch (the run is ALREADY done) backfills from the PRIOR updated_at,
        # not now -- and updated_at still advances to now as usual.
        assert runner._update("leg", "done", billing_state="charged") is True
        out = runner.get_status("leg")
        assert out.finished_at == teardown  # frozen to the prior terminal time, NOT now
        assert out.updated_at > teardown  # the touch still bumped updated_at

        # Contrast: a genuine non-terminal -> terminal transition stamps finished_at to the NEW
        # updated_at (the real teardown), as before.
        runner.submit_job(
            JobSpec(run_id="fresh", model="Qwen/Qwen3.5-4B", algorithm="grpo"), dry_run=True
        )
        s2 = runner.get_status("fresh")
        s2.state = "running"
        s2.finished_at = None
        runner._save_status(s2)
        assert runner._update("fresh", "done") is True
        done2 = runner.get_status("fresh")
        assert done2.finished_at == done2.updated_at  # transition: stamps to now


def test_persist_metrics_keeps_stamped_zero_vast(monkeypatch):
    import json
    import os

    with tempfile.TemporaryDirectory() as tmp:
        import flash.runner as runner

        importlib.reload(runner)
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp)
        monkeypatch.setattr(runner, "_gpu_rate", lambda gpu: 3600.0)
        from flash.spec import JobSpec

        spec = JobSpec(run_id="r0", model="Qwen/Qwen3.5-4B", algorithm="grpo")
        # A zero placeholder is not a settled provider cost; use the RunPod wall-pricing fallback.
        metrics = {
            "cost_usd": 0.0,
            "wall_seconds": 1.0,
        }
        out = runner._persist_metrics(spec, metrics)
        assert out == 1.0
        with open(os.path.join(runner.artifacts_dir(spec), "metrics.json")) as f:
            on_disk = json.load(f)
        assert on_disk["cost_usd"] == 1.0
        assert on_disk["notes"]["provider"] == "runpod"


def test_persist_metrics_falls_back_when_cost_absent(monkeypatch):
    import json
    import os

    with tempfile.TemporaryDirectory() as tmp:
        import flash.runner as runner

        importlib.reload(runner)
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp)
        monkeypatch.setattr(runner, "_gpu_rate", lambda gpu: 3600.0)
        from flash.spec import JobSpec

        spec = JobSpec(run_id="r1", model="Qwen/Qwen3.5-4B", algorithm="grpo")
        # No cost_usd stamped (RunPod path): fall back to wall * rate and attribute runpod.
        out = runner._persist_metrics(spec, {"wall_seconds": 1.0, "allocated_gpu": "RTX 5090"})
        assert out == 1.0  # 1s / 3600 * 3600/hr
        with open(os.path.join(runner.artifacts_dir(spec), "metrics.json")) as f:
            on_disk = json.load(f)
        assert on_disk["notes"]["provider"] == "runpod"


def test_persist_metrics_bills_training_wall_not_setup(monkeypatch):
    import json
    import os

    with tempfile.TemporaryDirectory() as tmp:
        import flash.runner as runner

        importlib.reload(runner)
        monkeypatch.setattr(runner, "RESULTS_DIR", tmp)
        monkeypatch.setattr(runner, "_gpu_rate", lambda gpu: 3600.0)
        from flash.spec import JobSpec

        spec = JobSpec(run_id="r-train-only", model="Qwen/Qwen3.5-4B", algorithm="sft")
        metrics = {
            "wall_seconds": 10.0,  # worker training loop only
            "setup_seconds": 590.0,  # reported for observability, not customer cost
            "train_tokens": 190_679,
            "allocated_gpu": "RTX 5090",
        }
        out = runner._persist_metrics(spec, metrics)
        assert out == pytest.approx(10.0)  # 10s / 3600 * $3600/hr
        with open(os.path.join(runner.artifacts_dir(spec), "metrics.json")) as f:
            on_disk = json.load(f)
        assert on_disk["cost_usd"] == pytest.approx(10.0)
        assert on_disk["setup_seconds"] == pytest.approx(590.0)


def test_run_training_charges_persisted_submit_estimate(monkeypatch, tmp_path):
    import io

    import flash.runner as runner
    from flash.runner import lifecycle
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(
        run_id="quote",
        model="Qwen/Qwen3.5-4B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=2),
        gpu=GpuSpec(type="RTX 4090"),
    )
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="queued",
            spec=spec.to_dict(),
            estimated_cost_usd=7.77,
        )
    )
    monkeypatch.setattr(
        runner,
        "_submit_seed_supervised",
        lambda *a, **k: {"wall_seconds": 1.0, "cost_usd": 0.01},
    )
    monkeypatch.setattr(
        runner,
        "charge_usd_for_spec",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must use submit quote")),
    )

    lifecycle._run_training(spec, io.StringIO(), prior_cost=0.0)

    st = runner.get_status(spec.run_id)
    assert st.state == "done"
    assert st.cost_usd == pytest.approx(7.77)


def test_supervised_attempt_identities_start_at_zero_and_increment_without_expanding_budget(
    monkeypatch, tmp_path
):
    import io

    import flash.providers as providers
    import flash.providers.allocator as allocator
    import flash.runner as runner
    from flash.providers.base import Allocation, Candidate, PollResult
    from flash.runner import lifecycle
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attempt-sequence",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(type="RTX 4090", max_retries=1),
    )
    runner._save_status(
        runner.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()),
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

    metrics = lifecycle._submit_seed_supervised(spec, spec.seed, io.StringIO())

    assert metrics["wall_seconds"] == 1.0
    assert provider.attempts == [0, 1]
    assert runner.get_status(spec.run_id).remote["attempt"] == 1


def test_attempt_is_consumed_when_provider_fails_before_handle_persistence(monkeypatch, tmp_path):
    import io

    import flash.providers as providers
    import flash.providers.allocator as allocator
    import flash.runner as runner
    from flash.providers.base import Allocation, Candidate, PollResult
    from flash.runner import lifecycle
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="pre-handle-attempt",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(type="RTX 4090", max_retries=1),
    )
    runner._save_status(
        runner.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()),
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
                    started_ts=float(attempt + 1),
                )
            )
            return PollResult(True, metrics={"wall_seconds": 1.0})

    provider = Provider()
    monkeypatch.setattr(providers, "get_provider", lambda _name: provider)

    lifecycle._submit_seed_supervised(spec, spec.seed, io.StringIO())

    assert provider.attempts == [0, 1]
    assert runner.get_status(spec.run_id).remote["attempt"] == 1
    assert runner._load_status_json(spec.run_id)[runner._NEXT_ATTEMPT_KEY] == 2


def test_retry_receives_only_remaining_run_global_wall_allowance(monkeypatch, tmp_path):
    import io

    import flash.providers as providers
    import flash.providers.allocator as allocator
    import flash.runner as runner
    from flash.providers.base import Allocation, Candidate, PollResult
    from flash.runner import lifecycle
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="wall-budget",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(type="RTX 4090", max_wall_seconds=200, max_retries=1),
    )
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            created_at=100.0,
        ),
        _run_deadline_at=300.0,
        _next_attempt=0,
    )
    now = {"value": 100.0}
    monkeypatch.setattr(runner.time, "time", lambda: now["value"])
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

    lifecycle._submit_seed_supervised(spec, spec.seed, io.StringIO())

    assert provider.attempts == [0, 1]
    assert allocation_walls == [200.0, 120.0]
    assert provider.walls == [200, 120]
    raw = runner._load_status_json(spec.run_id)
    assert raw[runner._RUN_DEADLINE_AT_KEY] == 300.0
    assert raw[runner._NEXT_ATTEMPT_KEY] == 2


def test_retry_backoff_cannot_cross_provider_minimum(monkeypatch, tmp_path):
    import io

    import flash.providers as providers
    import flash.providers.allocator as allocator
    import flash.runner as runner
    from flash.providers.base import Allocation, Candidate
    from flash.runner import lifecycle
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="retry-deadline-minimum",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(type="RTX 4090", max_wall_seconds=200, max_retries=1),
    )
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            created_at=100.0,
        ),
        _run_deadline_at=300.0,
        _next_attempt=0,
    )
    clock = {"now": 230.0}
    sleeps = []
    monkeypatch.setattr(runner.time, "time", lambda: clock["now"])

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
        lifecycle._submit_seed_supervised(spec, spec.seed, log)

    assert provider.attempts == [0]
    assert allocations == [True]
    assert sleeps == [10.0]
    assert "provider body secret" not in str(exc_info.value)
    assert "provider body secret" not in log.getvalue()
    assert runner._load_status_json(spec.run_id)[runner._NEXT_ATTEMPT_KEY] == 1


def test_save_status_flushes_file_and_directory_before_return(monkeypatch, tmp_path):
    import os

    import flash.runner as runner
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="durable-status", model="Qwen/Qwen3.5-4B", algorithm="sft")
    events = []
    directory_fd = 987654
    file_fd = {"value": None}
    original_fdopen = runner.os.fdopen
    original_open = runner.os.open
    original_close = runner.os.close
    original_replace = runner.os.replace

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
        if path == runner.RUNS_DIR and flags == os.O_RDONLY:
            events.append("open-directory")
            return directory_fd
        return original_open(path, flags, *args, **kwargs)

    def _close(fd):
        if fd == directory_fd:
            events.append("close-directory")
            return None
        return original_close(fd)

    monkeypatch.setattr(runner.os, "fdopen", _fdopen)
    monkeypatch.setattr(runner.os, "fsync", _fsync)
    monkeypatch.setattr(runner.os, "replace", _replace)
    monkeypatch.setattr(runner.os, "open", _open)
    monkeypatch.setattr(runner.os, "close", _close)

    runner._save_status(
        runner.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict()),
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

    import flash.runner as runner
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="durable-failure", model="Qwen/Qwen3.5-4B", algorithm="sft")
    directory_fd = 987655
    closed = []
    temp_paths = []
    original_mkstemp = runner.tempfile.mkstemp
    original_open = runner.os.open
    original_close = runner.os.close
    original_fsync = runner.os.fsync

    def _mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        temp_paths.append(path)
        return fd, path

    def _open(path, flags, *args, **kwargs):
        if path == runner.RUNS_DIR and flags == os.O_RDONLY:
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

    monkeypatch.setattr(runner.tempfile, "mkstemp", _mkstemp)
    monkeypatch.setattr(runner.os, "open", _open)
    monkeypatch.setattr(runner.os, "fsync", _fsync)
    monkeypatch.setattr(runner.os, "close", _close)

    with pytest.raises(OSError, match="directory fsync failed"):
        runner._save_status(
            runner.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
        )

    assert closed == [directory_fd]
    assert temp_paths
    assert all(not os.path.exists(path) for path in temp_paths)


def test_save_status_cleans_temp_when_replace_fails(monkeypatch, tmp_path):
    import os

    import flash.runner as runner
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="replace-failure", model="Qwen/Qwen3.5-4B", algorithm="sft")
    temp_paths = []
    original_mkstemp = runner.tempfile.mkstemp

    def _mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        temp_paths.append(path)
        return fd, path

    monkeypatch.setattr(runner.tempfile, "mkstemp", _mkstemp)
    monkeypatch.setattr(
        runner.os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        runner._save_status(
            runner.RunStatus(run_id=spec.run_id, state="queued", spec=spec.to_dict())
        )

    assert temp_paths
    assert all(not os.path.exists(path) for path in temp_paths)


def test_concurrent_attempt_reservations_are_unique_and_monotonic(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    import flash.runner as runner
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="threaded-attempts", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner._save_status(
        runner.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict()),
        _next_attempt=0,
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        attempts = list(pool.map(lambda _index: runner._reserve_attempt(spec.run_id), range(16)))

    assert sorted(attempts) == list(range(16))
    assert runner._load_status_json(spec.run_id)[runner._NEXT_ATTEMPT_KEY] == 16


def test_multiprocess_attempt_reservations_preserve_concurrent_status_update(monkeypatch, tmp_path):
    import multiprocessing

    import flash.runner as runner
    from flash.spec import JobSpec

    runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)
    spec = JobSpec(run_id="multiprocess-attempts", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner._save_status(
        runner.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict()),
        _next_attempt=0,
    )
    context = multiprocessing.get_context("fork")
    start = context.Barrier(4)
    results = context.Queue()

    def reserve(worker_index):
        import time

        import flash.runner as child_runner

        child_runner.RUNS_DIR = runs_dir
        original_save = child_runner._save_status_unlocked

        def slow_save(*args, **kwargs):
            time.sleep(0.005)
            return original_save(*args, **kwargs)

        child_runner._save_status_unlocked = slow_save
        start.wait()
        if worker_index == 0:
            child_runner._update(spec.run_id, "provisioning", error="concurrent-update")
        attempts = [child_runner._reserve_attempt(spec.run_id) for _ in range(4)]
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

    raw = runner._load_status_json(spec.run_id)
    assert sorted(attempts) == list(range(16))
    assert raw[runner._NEXT_ATTEMPT_KEY] == 16
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
    import flash.runner as runner
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="compare-clear", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            remote=remote,
        )
    )

    assert runner._compare_and_clear_remote(spec.run_id, remote) is True
    assert runner.get_status(spec.run_id).remote is None


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
    import flash.runner as runner
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="compare-preserve", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            remote=newer,
        )
    )

    assert runner._compare_and_clear_remote(spec.run_id, original) is False
    assert runner.get_status(spec.run_id).remote == newer


def test_cleanup_collection_deduplicates_and_survives_status_writes_and_reload(
    monkeypatch, tmp_path
):
    import flash.runner as runner
    from flash.spec import JobSpec

    runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)
    spec = JobSpec(run_id="cleanup-dedup", model="Qwen/Qwen3.5-4B", algorithm="sft")
    public_remote = _runpod_remote("endpoint-a", "job-a", attempt=0)
    cleanup_remote = _runpod_remote("endpoint-b", None, attempt=1, started_ts=2.0)
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="cancelled",
            spec=spec.to_dict(),
            remote=public_remote,
        )
    )

    assert runner._preserve_cleanup_remote(spec.run_id, cleanup_remote) is True
    assert runner._preserve_cleanup_remote(spec.run_id, cleanup_remote) is True
    assert runner._update(spec.run_id, "cancelled", error="unchanged terminal state") is True

    raw = runner._load_status_json(spec.run_id)
    assert raw["remote"] == public_remote
    assert raw[runner._CLEANUP_REMOTES_KEY] == [cleanup_remote]

    importlib.reload(runner)
    monkeypatch.setattr(runner, "RUNS_DIR", runs_dir)
    reloaded = runner._load_status_json(spec.run_id)
    assert reloaded["remote"] == public_remote
    assert reloaded[runner._CLEANUP_REMOTES_KEY] == [cleanup_remote]


def test_cleanup_collection_removes_only_confirmed_exact_records(monkeypatch, tmp_path):
    import flash.providers as providers
    import flash.runner as runner
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="cleanup-drain", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner._save_status(
        runner.RunStatus(run_id=spec.run_id, state="cancelled", spec=spec.to_dict())
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
        assert runner._preserve_cleanup_remote(spec.run_id, remote) is True

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

    attempted = runner._drain_cleanup_remotes(spec.run_id)

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
    raw = runner._load_status_json(spec.run_id)
    assert raw[runner._CLEANUP_REMOTES_KEY] == [unconfirmed]
    assert raw["remote"] == confirmed


def test_next_attempt_requires_persisted_integer_identity():
    import flash.runner as runner

    assert runner._infer_next_attempt({"next_attempt": 0}) == 0
    assert runner._infer_next_attempt({"next_attempt": 7}) == 7
    for raw in ({}, {"next_attempt": True}, {"next_attempt": -1}, {"next_attempt": "1"}):
        with pytest.raises(RuntimeError, match="next attempt identity"):
            runner._infer_next_attempt(raw)


def test_handleless_state_without_next_attempt_is_rejected(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="missing-next-attempt", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner._save_status(runner.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()))
    raw = runner._load_status_json(spec.run_id)
    raw.pop(runner._NEXT_ATTEMPT_KEY)
    with open(runner.runs_file_path(spec.run_id, ".json"), "w") as file:
        json.dump(raw, file)

    with pytest.raises(RuntimeError, match="next attempt identity is missing"):
        runner._reserve_attempt(spec.run_id)


def test_new_attempt_requires_full_provider_minimum_before_allocation(monkeypatch, tmp_path):
    import io

    import flash.providers.allocator as allocator
    import flash.runner as runner
    from flash.runner import lifecycle
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="wall-minimum",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=60),
    )
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            created_at=99.0,
        ),
        _run_deadline_at=159.0,
        _next_attempt=0,
    )
    monkeypatch.setattr(runner.time, "time", lambda: 100.0)
    allocations = []
    monkeypatch.setattr(allocator, "allocate", lambda *_args, **_kwargs: allocations.append(True))

    with pytest.raises(RuntimeError, match="60-second minimum provider allowance"):
        lifecycle._submit_seed_supervised(spec, spec.seed, io.StringIO())

    assert allocations == []
    assert runner._load_status_json(spec.run_id)[runner._NEXT_ATTEMPT_KEY] == 0


def test_reserved_attempt_survives_handleless_restart_without_reusing_zero(monkeypatch, tmp_path):
    import flash.runner as runner
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="reserved-restart",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=200),
    )
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="provisioning",
            spec=spec.to_dict(),
            created_at=100.0,
        ),
        _run_deadline_at=300.0,
        _next_attempt=0,
    )

    assert (
        runner._spec_with_remaining_wall(
            spec, require_provider_minimum=True, now=100.0
        ).gpu.max_wall_seconds
        == 200
    )
    assert runner._reserve_attempt(spec.run_id) == 0
    assert runner.get_status(spec.run_id).remote is None
    assert (
        runner._spec_with_remaining_wall(
            spec, require_provider_minimum=True, now=180.0
        ).gpu.max_wall_seconds
        == 120
    )
    assert runner._reserve_attempt(spec.run_id) == 1
    raw = runner._load_status_json(spec.run_id)
    assert raw[runner._NEXT_ATTEMPT_KEY] == 2
    assert raw[runner._RUN_DEADLINE_AT_KEY] == 300.0


def test_attach_failed_worker_resumes_with_next_attempt_identity(monkeypatch, tmp_path):
    persisted_attempt = 1
    expected_next = 2
    import io

    import flash.providers as providers
    import flash.runner as runner
    from flash.providers.base import PollResult
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(
        run_id=f"attach-attempt-{expected_next}",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=200),
    )
    remote = _runpod_remote(
        "endpoint-old",
        "job-old",
        attempt=persisted_attempt,
        code_prefix="code/revision",
    )
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            created_at=100.0,
            remote=remote,
        ),
        _run_deadline_at=300.0,
        _next_attempt=2,
    )
    monkeypatch.setattr(runner.time, "time", lambda: 100.0)
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
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        runner, "_resolve_init_from_adapter", lambda public_spec, **_kwargs: public_spec
    )
    resumed = []

    def fake_run_training(_spec, _log, **kwargs):
        resumed.append(kwargs["attempt_start"])

    monkeypatch.setattr(runner, "_run_training", fake_run_training)

    status = runner.attach_run(spec.run_id, log_stream=io.StringIO())

    assert resumed == [expected_next]
    assert poll_walls == [200]
    assert status.state == "running"
    assert status.remote is None


def test_attach_expired_run_adopts_completed_attempt_at_deadline(monkeypatch, tmp_path):
    import io

    import flash.providers._hf_artifacts as hf_artifacts
    import flash.runner as runner
    import flash.runner.lifecycle as lifecycle
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(
        run_id="attach-expired-completed",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(max_wall_seconds=120),
    )
    remote = _vast_remote(instance_id=7, attempt=0, started_ts=101.0)
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            created_at=100.0,
            remote=remote,
        ),
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner.time, "time", lambda: 221.0)
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
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda _spec: None)
    monkeypatch.setattr(
        lifecycle,
        "_strict_teardown_handle",
        lambda *_args, **_kwargs: pytest.fail(
            "completed attempt must not be torn down before adoption"
        ),
    )
    log = io.StringIO()

    status = runner.attach_run(spec.run_id, log_stream=log)

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
    assert status.remote is None
    assert status.error is None
    assert runner._load_status_json(spec.run_id)[runner._CLEANUP_REMOTES_KEY] == [remote]
    assert "adopted a completed attempt at the wall deadline" in log.getvalue()


def test_completed_attempt_metrics_accepts_marker_just_after_wall(monkeypatch):
    import flash.providers._hf_artifacts as hf_artifacts
    import flash.runner.lifecycle as lifecycle
    from flash.spec import JobSpec, TrainSpec

    spec = JobSpec(
        run_id="late-marker-complete",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
    )
    monkeypatch.setattr(lifecycle.time, "time", lambda: 221.0)

    def artifact_reader(_repo, path):
        def read(force=False):
            if path.endswith("/vast_attempt0.json"):
                return (
                    '{"attempt":0,"error":"","ok":true,"retriable":false,'
                    '"run_id":"late-marker-complete","ts":220.5}'
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

    assert metrics == {"wall_seconds": 5.0}


def test_attach_expired_run_does_not_poll_or_resubmit(monkeypatch, tmp_path):
    import io

    import flash.providers as providers
    import flash.runner as runner
    import flash.runner.lifecycle as lifecycle
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-expired",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=120),
    )
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            created_at=100.0,
            remote=_runpod_remote("endpoint-old", "job-old", attempt=0),
        ),
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner.time, "time", lambda: 221.0)
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
        runner, "_gc_run_endpoints", lambda cleanup_spec: gc_runs.append(cleanup_spec.run_id)
    )
    monkeypatch.setattr(runner, "_run_training", lambda *_args, **_kwargs: resumed.append(True))

    status = runner.attach_run(spec.run_id, log_stream=io.StringIO())

    assert polled == []
    assert resumed == []
    assert len(completion_checks) == 1
    assert completion_checks[0][1]["attempt"] == 0
    assert [action for action, _handle in teardown] == ["cancel", "destroy"]
    assert gc_runs == [spec.run_id]
    assert status.state == "failed"
    assert status.remote is None
    assert "deadline exhausted" in status.error


def test_attach_expired_run_retains_handle_when_teardown_is_unconfirmed(monkeypatch, tmp_path):
    import io

    import flash.providers as providers
    import flash.runner as runner
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="attach-expired-unconfirmed",
        model="Qwen/Qwen3.5-4B",
        gpu=GpuSpec(max_wall_seconds=120),
    )
    remote = _runpod_remote("endpoint-old", "job-old", attempt=0)
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="running",
            spec=spec.to_dict(),
            created_at=100.0,
            remote=remote,
        ),
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner.time, "time", lambda: 221.0)

    class Provider:
        def cancel(self, _handle):
            return None

        def destroy(self, _handle):
            raise RuntimeError("teardown unconfirmed")

        def poll(self, *_args, **_kwargs):
            raise AssertionError("expired recovery must not poll")

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(runner, "_gc_run_endpoints", lambda _spec: None)
    resumed = []
    monkeypatch.setattr(runner, "_run_training", lambda *_args, **_kwargs: resumed.append(True))

    status = runner.attach_run(spec.run_id, log_stream=io.StringIO())

    assert resumed == []
    assert status.state == "failed"
    assert status.remote == remote
    assert "deadline exhausted" in status.error


def test_runpod_submit_propagates_attempt_to_worker_environment_and_handle(monkeypatch):
    import flash.providers.runpod.jobs as jobs
    import flash.providers.runpod.train as train
    from flash.providers.base import PollResult
    from flash.spec import JobSpec

    spec = JobSpec(run_id="worker-attempt", model="Qwen/Qwen3.5-4B", algorithm="sft")
    payloads = []
    handles = []
    monkeypatch.setattr(train, "build_worker_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train, "chalk_extra_pip", lambda _spec: [])
    monkeypatch.setattr(
        jobs,
        "deploy_train_endpoint",
        lambda *_args, **_kwargs: ("endpoint", "name", _RUNPOD_FINGERPRINT),
    )
    monkeypatch.setattr(jobs, "build_function_input", lambda payload: payload)
    monkeypatch.setattr(
        jobs.runpod_api,
        "submit_job",
        lambda _endpoint, payload, **_kwargs: payloads.append(payload) or "job",
    )
    monkeypatch.setattr(
        jobs,
        "poll_job",
        lambda *_args, **_kwargs: PollResult(True, metrics={"wall_seconds": 1.0}),
    )

    jobs.submit_run(
        spec,
        0,
        attempt=2,
        code_prefix="code/revision",
        on_handle=handles.append,
        deadline_at=10_000_000_000.0,
    )

    assert payloads[0]["env"]["ATTEMPT"] == "2"
    assert handles[0]["attempt"] == 2


def test_fail_blocked_recovery_adopts_completed_handleless_attempt(monkeypatch, tmp_path):
    import flash.runner as runner
    import flash.server._runtime as runtime
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(run_id="blocked-complete", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner._save_status(
        runner.RunStatus(
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

    status = runner.get_status(spec.run_id)
    assert status.state == "done"
    assert status.remote is None
    assert status.error is None


def test_start_resubmit_deadline_adopts_completed_handleless_attempt(monkeypatch, tmp_path):
    import flash.runner as runner
    import flash.server._runtime as runtime
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(runner, "RESULTS_DIR", str(tmp_path / "results"))
    spec = JobSpec(
        run_id="deadline-complete",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(max_wall_seconds=120),
    )
    runner._save_status(
        runner.RunStatus(
            run_id=spec.run_id,
            state="provisioning",
            spec=spec.to_dict(),
            created_at=100.0,
        ),
        _run_deadline_at=220.0,
        _next_attempt=1,
    )
    monkeypatch.setattr(runner.time, "time", lambda: 221.0)
    monkeypatch.setattr(
        runtime,
        "_handleless_completed_metrics",
        lambda *_args, **_kwargs: {"wall_seconds": 5.0},
    )

    assert runtime._start_resubmit(spec, expected_remote=None) is False

    status = runner.get_status(spec.run_id)
    assert status.state == "done"
    assert status.remote is None
    assert status.error is None


def test_deferred_handleless_loop_resubmits_when_clear_before_deadline(monkeypatch, tmp_path):
    import time as time_mod

    import flash.runner as runner
    import flash.server._runtime as runtime
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec.from_dict(
        {
            "run_id": "deferred-clear",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "train": {"epochs": 1},
        }
    )
    runner._save_status(
        runner.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict())
    )
    checks = iter([False, True])
    monkeypatch.setattr(runtime, "_confirm_run_clear", lambda _spec: next(checks))
    monkeypatch.setattr(runner, "_load_run_deadline_at", lambda _run_id: 1000.0)
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
    assert started[0][1] == {
        "expected_remote": None,
        "expected_state": "provisioning",
    }
    assert runner.get_status(spec.run_id).state == "provisioning"


def test_deferred_handleless_loop_reconciles_after_resubmit_cas_loss(monkeypatch, tmp_path):
    import time as time_mod

    import flash.runner as runner
    import flash.server._runtime as runtime
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(run_id="deferred-cas-loss", model="Qwen/Qwen3.5-4B", algorithm="sft")
    runner._save_status(
        runner.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict())
    )
    monkeypatch.setattr(runtime, "_confirm_run_clear", lambda _spec: True)
    monkeypatch.setattr(runner, "_load_run_deadline_at", lambda _run_id: 1000.0)
    monkeypatch.setattr(time_mod, "time", lambda: 100.0)
    attempts = []

    def start_resubmit(*args, **kwargs):
        attempts.append((args, kwargs))
        return len(attempts) == 2

    monkeypatch.setattr(runtime, "_start_resubmit", start_resubmit)

    runtime._deferred_resubmit_loop(spec)

    assert len(attempts) == 2
    assert (
        attempts[0][1]
        == attempts[1][1]
        == {
            "expected_remote": None,
            "expected_state": "provisioning",
        }
    )


def test_deferred_handleless_loop_deadline_cas_fails_with_retry(monkeypatch, tmp_path):
    import time as time_mod

    import flash.runner as runner
    import flash.server._runtime as runtime
    from flash.spec import JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec.from_dict(
        {
            "run_id": "deferred-deadline",
            "model": "Qwen/Qwen3.5-4B",
            "algorithm": "grpo",
            "train": {"epochs": 1},
        }
    )
    runner._save_status(
        runner.RunStatus(run_id=spec.run_id, state="provisioning", spec=spec.to_dict())
    )
    monkeypatch.setattr(runner, "_load_run_deadline_at", lambda _run_id: 100.0)
    monkeypatch.setattr(time_mod, "time", lambda: 101.0)
    monkeypatch.setattr(time_mod, "sleep", lambda _seconds: None)
    monkeypatch.setattr(runtime, "_handleless_completed_metrics", lambda *a, **k: None)
    real_fail = runner._compare_and_fail_remote
    attempts = []

    def flaky_fail(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise PermissionError("status store unavailable")
        return real_fail(*args, **kwargs)

    monkeypatch.setattr(runner, "_compare_and_fail_remote", flaky_fail)

    runtime._deferred_resubmit_loop(spec)

    status = runner.get_status(spec.run_id)
    assert len(attempts) == 2
    assert status.state == "failed"
    assert status.remote is None
    assert "deadline exhausted" in (status.error or "")


@pytest.mark.parametrize("cleanup_confirmed", [True, False])
def test_terminal_handle_race_tears_down_or_preserves_cleanup_identity(
    monkeypatch, tmp_path, cleanup_confirmed
):
    import io

    import flash.providers as providers
    import flash.providers.allocator as allocator
    import flash.runner as runner
    from flash.providers.base import Allocation, Candidate
    from flash.runner import lifecycle
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id=f"terminal-handle-race-{cleanup_confirmed}",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(type="RTX 4090", max_retries=2),
    )
    runner._save_status(
        runner.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()),
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
            runner._update(spec.run_id, "cancelled")
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

    with pytest.raises(runner._TerminalHandleRace):
        lifecycle._submit_seed_supervised(
            spec,
            spec.seed,
            io.StringIO(),
            code_prefix="code/revision",
        )

    status = runner.get_status(spec.run_id)
    assert provider.submits == [0]
    assert [event for event, _handle in provider.teardown] == ["cancel", "destroy"]
    assert status.state == "cancelled"
    if cleanup_confirmed:
        assert status.remote is None
    else:
        assert status.remote["endpoint_id"] == "endpoint-race"
        assert status.remote["job_id"] == "job-race"
        assert status.remote["attempt"] == 0
        raw = runner._load_status_json(spec.run_id)
        assert raw[runner._CLEANUP_REMOTES_KEY] == [
            _runpod_remote("endpoint-race", "job-race", attempt=0)
        ]


def test_terminal_handle_race_retains_second_unconfirmed_cleanup_remote(monkeypatch, tmp_path):
    import io

    import flash.providers as providers
    import flash.providers.allocator as allocator
    import flash.runner as runner
    from flash.providers.base import Allocation, Candidate
    from flash.runner import lifecycle
    from flash.spec import GpuSpec, JobSpec

    monkeypatch.setattr(runner, "RUNS_DIR", str(tmp_path / "runs"))
    spec = JobSpec(
        run_id="terminal-handle-race-two-remotes",
        model="Qwen/Qwen3.5-4B",
        algorithm="sft",
        gpu=GpuSpec(type="RTX 4090", max_retries=0),
    )
    runner._save_status(
        runner.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()),
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
            runner._update(spec.run_id, "cancelled", remote=remote_a)
            on_handle(remote_b)
            raise AssertionError("terminal handle callback must not return")

        def cancel(self, _handle):
            return None

        def destroy(self, _handle):
            raise RuntimeError("endpoint deletion unconfirmed")

    monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())

    with pytest.raises(runner._TerminalHandleRace):
        lifecycle._submit_seed_supervised(
            spec,
            spec.seed,
            io.StringIO(),
            code_prefix="code/revision",
        )

    raw = runner._load_status_json(spec.run_id)
    assert raw["remote"] == remote_a
    assert raw[runner._CLEANUP_REMOTES_KEY] == [remote_b]


def test_run_training_bails_when_running_cas_rejects(monkeypatch):
    """If a run flips terminal in the race window between the pre-check and the ``running`` CAS,
    _run_training must raise _RunCancelled and never reach the PAID supervised submit. The gate is
    _update's return value (False == rejected by terminal-stickiness)."""
    import pytest

    import flash.runner as runner
    from flash.runner import lifecycle
    from flash.spec import JobSpec

    importlib.reload(runner)
    spec = JobSpec(run_id="cas", model="Qwen/Qwen3.5-4B", algorithm="grpo")
    # Pre-check sees a live run...
    monkeypatch.setattr(
        runner,
        "get_status",
        lambda rid: runner.RunStatus(run_id=rid, state="running", spec={}),
    )
    # ...but the CAS rejects because the run went terminal concurrently.
    monkeypatch.setattr(runner, "_update", lambda *a, **k: False)
    submitted: list[bool] = []
    monkeypatch.setattr(
        runner,
        "_submit_seed_supervised",
        lambda *a, **k: submitted.append(True) or {},
    )

    with pytest.raises(runner._RunCancelled):
        lifecycle._run_training(spec, None, prior_cost=0.0)
    assert submitted == []  # never charged a GPU for an already-terminal run
