"""Tests for run-management helpers (runs/cost/cancel) — no GPU/network."""

from __future__ import annotations

import importlib
import tempfile

import pytest


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


def test_record_heartbeat_updates_status_without_state_change(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        import flash.runner as runner

        importlib.reload(runner)
        monkeypatch.setattr(runner, "RUNS_DIR", tmp)
        from flash.spec import JobSpec

        status = runner.submit_job(
            JobSpec(run_id="hb", model="Qwen/Qwen3.5-4B", algorithm="sft"),
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
