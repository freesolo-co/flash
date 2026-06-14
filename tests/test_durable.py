"""Durable run primitives: handle persistence, polling state machine, supervisor retry,
cross-process cancel, and attach (CPU-only; all network mocked)."""

from __future__ import annotations

import base64
import importlib
import os
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# decode_output / JobHandle
# ---------------------------------------------------------------------------


def test_job_handle_roundtrip():
    from autoslm.flash.durable import JobHandle

    h = JobHandle("ep123", "autoslm-train-5090-abc", "job456")
    assert JobHandle.from_dict(h.to_dict()) == h


def test_decode_output_success():
    import cloudpickle

    from autoslm.flash.durable import decode_output

    metrics = {"trained_eval_acc": 0.9, "cost_usd": 0.5}
    out = {"success": True, "result": base64.b64encode(cloudpickle.dumps(metrics)).decode()}
    assert decode_output(out) == metrics


def test_decode_output_error_includes_stdout_tail():
    from autoslm.flash.durable import decode_output

    with pytest.raises(RuntimeError) as ei:
        decode_output({"success": False, "error": "boom", "stdout": "x" * 5000})
    assert "boom" in str(ei.value)


# ---------------------------------------------------------------------------
# poll_job state machine (mocked runpod_api)
# ---------------------------------------------------------------------------
def _poll(monkeypatch, statuses, heartbeats=None, stall_after_s=10.0):
    import cloudpickle

    from autoslm.flash import durable, runpod_api

    seq = iter(statuses)
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: next(seq))
    monkeypatch.setattr(durable.time, "sleep", lambda s: None)
    hb_iter = iter(heartbeats) if heartbeats is not None else None

    reader = (lambda: next(hb_iter, None)) if hb_iter is not None else None
    h = durable.JobHandle("ep", "name", "job")
    ok_payload = {
        "success": True,
        "result": base64.b64encode(cloudpickle.dumps({"acc": 1.0})).decode(),
    }
    return durable.poll_job(
        h,
        interval_s=0,
        heartbeat_reader=reader,
        stall_after_s=stall_after_s,
    ), ok_payload


def test_poll_job_completes(monkeypatch):
    import cloudpickle

    ok = {
        "success": True,
        "result": base64.b64encode(cloudpickle.dumps({"acc": 1.0})).decode(),
    }
    res, _ = _poll(
        monkeypatch,
        [
            {"status": "IN_QUEUE"},
            {"status": "IN_PROGRESS"},
            {"status": "COMPLETED", "output": ok},
        ],
    )
    assert res.ok
    assert res.metrics == {"acc": 1.0}


def test_poll_job_failure(monkeypatch):
    res, _ = _poll(
        monkeypatch,
        [{"status": "IN_PROGRESS"}, {"status": "FAILED", "error": "worker exploded"}],
    )
    assert not res.ok
    assert res.failure == "job_failed"
    assert "worker exploded" in res.detail


def test_poll_job_stall_detection(monkeypatch):
    # job stays IN_PROGRESS forever, heartbeat never advances -> stall
    import itertools

    from autoslm.flash import durable, runpod_api

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(durable.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(durable.time, "time", lambda: next(clock))
    h = durable.JobHandle("ep", "name", "job")
    res = durable.poll_job(h, interval_s=0, heartbeat_reader=lambda: None, stall_after_s=150.0)
    assert not res.ok
    assert res.failure == "stalled"


def test_poll_job_tolerates_transient_api_errors(monkeypatch):
    import cloudpickle

    from autoslm.flash import durable, runpod_api

    ok = {
        "success": True,
        "result": base64.b64encode(cloudpickle.dumps({"acc": 0.7})).decode(),
    }
    calls = {"n": 0}

    def flaky(eid, jid):
        calls["n"] += 1
        if calls["n"] < 4:
            raise runpod_api.RunpodApiError("blip")
        return {"status": "COMPLETED", "output": ok}

    monkeypatch.setattr(runpod_api, "job_status", flaky)
    monkeypatch.setattr(durable.time, "sleep", lambda s: None)
    res = durable.poll_job(durable.JobHandle("ep", "n", "j"), interval_s=0, stall_after_s=1e9)
    assert res.ok
    assert calls["n"] == 4


# ---------------------------------------------------------------------------
# Supervisor retry logic (orchestrator) with mocked durable submit
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _restore_skip_net():
    """Tests below exercise the durable (network-shaped) path with mocks, so they
    unset AUTOSLM_SKIP_NET locally — restore it afterwards for the rest of the suite."""
    saved = os.environ.get("AUTOSLM_SKIP_NET")
    yield
    if saved is not None:
        os.environ["AUTOSLM_SKIP_NET"] = saved


def _fresh_orchestrator(tmp):
    os.environ["AUTOSLM_RUNS_DIR"] = os.path.join(tmp, "runs")
    os.environ["RESULTS_DIR"] = os.path.join(tmp, "results")
    os.environ.pop("AUTOSLM_SKIP_NET", None)
    os.environ.pop("AUTOSLM_LEGACY_SUBMIT", None)
    import autoslm.orchestrator as orch

    importlib.reload(orch)
    return orch


def _spec(run_id):
    from autoslm.worker_spec import GpuSpec, JobSpec, TrainSpec

    return JobSpec(
        run_id=run_id,
        model="Qwen/Qwen3-0.6B",
        algorithm="grpo",
        train=TrainSpec(seeds=(0,), steps=1),
        gpu=GpuSpec(type="RTX 4090", max_retries=2),
    )


def test_supervisor_retries_on_stall_then_succeeds(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp)
        import autoslm.flash.durable as durable
        import autoslm.flash.train as flash_train

        calls = {"n": 0}

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0):
            calls["n"] += 1
            if on_handle:
                on_handle({"endpoint_id": "ep", "endpoint_name": "n", "job_id": f"j{calls['n']}"})
            if calls["n"] == 1:
                return durable.PollResult(False, failure="stalled", detail="frozen")
            return durable.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(durable, "submit_train_durable", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        orch.submit_job(_spec("retry-ok"), dry_run=False, background=False)
        st = orch.get_status("retry-ok")
        assert st.state == "done"
        assert calls["n"] == 2
        assert st.remote["job_id"] == "j2"  # latest handle persisted


def test_supervisor_does_not_retry_worker_code_errors(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp)
        import autoslm.flash.durable as durable
        import autoslm.flash.train as flash_train

        calls = {"n": 0}

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0):
            calls["n"] += 1
            return durable.PollResult(
                False, failure="job_failed", detail="Remote execution failed: ValueError"
            )

        monkeypatch.setattr(durable, "submit_train_durable", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        with pytest.raises(RuntimeError):
            orch.submit_job(_spec("fail-fast"), dry_run=False, background=False)
        assert calls["n"] == 1
        assert orch.get_status("fail-fast").state == "failed"


# ---------------------------------------------------------------------------
# Cross-process cancel via REST handle + attach
# ---------------------------------------------------------------------------
def test_cancel_uses_rest_handle(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp)
        from autoslm.flash import runpod_api

        status = orch.RunStatus(
            run_id="c1",
            state="running",
            spec=_spec("c1").to_dict(),
            remote={"endpoint_id": "epX", "endpoint_name": "n", "job_id": "jX"},
        )
        orch._save_status(status)
        cancelled, deleted = [], []
        monkeypatch.setattr(runpod_api, "cancel_job", lambda e, j: cancelled.append((e, j)))
        monkeypatch.setattr(runpod_api, "delete_endpoint", lambda e: deleted.append(e))
        import autoslm.flash.train as flash_train

        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        st = orch.cancel_run("c1")
        assert st.state == "cancelled"
        assert cancelled == [("epX", "jX")]
        assert deleted == ["epX"]


def test_attach_completes_run(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp)
        import autoslm.flash.durable as durable
        import autoslm.flash.train as flash_train

        status = orch.RunStatus(
            run_id="a1",
            state="running",
            spec=_spec("a1").to_dict(),
            remote={"endpoint_id": "epA", "endpoint_name": "n", "job_id": "jA"},
        )
        orch._save_status(status)
        monkeypatch.setattr(
            durable,
            "poll_job",
            lambda *a, **k: durable.PollResult(True, metrics={"cost_usd": 0.2}),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        st = orch.attach_run("a1", log_stream=sys.stderr)
        assert st.state == "done"
        assert abs(st.cost_usd - 0.2) < 1e-9


def test_attach_clears_stale_handle_before_resuming_seeds(monkeypatch):
    # After a recovered seed of a multi-seed run completes, the stale completed
    # handle must be cleared before the remaining seeds run: a restart in the
    # provisioning gap must not reattach recovery to the finished job.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp)
        import autoslm.flash.durable as durable
        import autoslm.flash.train as flash_train
        from autoslm.worker_spec import GpuSpec, JobSpec, TrainSpec

        spec = JobSpec(
            run_id="m1",
            model="Qwen/Qwen3-0.6B",
            algorithm="grpo",
            train=TrainSpec(seeds=(0, 1), steps=1),
            gpu=GpuSpec(type="RTX 4090", max_retries=2),
        )
        orch._save_status(
            orch.RunStatus(
                run_id="m1",
                state="running",
                spec=spec.to_dict(),
                remote={"endpoint_id": "epA", "endpoint_name": "n", "job_id": "jA", "seed": 0},
            )
        )
        monkeypatch.setattr(
            durable, "poll_job", lambda *a, **k: durable.PollResult(True, metrics={"cost_usd": 0.2})
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        seen = {}

        def fake_loop(spec, log, *, start_index, prior_cost):
            seen["remote"] = orch.get_status(spec.run_id).remote
            seen["start_index"] = start_index
            orch._update(spec.run_id, "done", cost_usd=prior_cost)

        monkeypatch.setattr(orch, "_run_seed_loop", fake_loop)
        orch.attach_run("m1", log_stream=sys.stderr)
        assert seen["start_index"] == 1
        assert seen["remote"] is None, "stale completed handle must be cleared before resuming"


def test_attach_requires_handle():
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp)
        orch._save_status(orch.RunStatus(run_id="nh", state="running", spec=_spec("nh").to_dict()))
        with pytest.raises(ValueError, match="no persisted job handle"):
            orch.attach_run("nh")


def test_update_will_not_overwrite_terminal_with_lifecycle_state():
    # Terminal states are STICKY: once cancelled, no other state may overwrite it —
    # neither a non-terminal lifecycle write (provisioning/running) NOR a late terminal
    # done/failed from a worker that finished as the cancel arrived. Same-state writes
    # still pass so terminal field updates (cost_usd, error) are preserved.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp)
        orch._save_status(orch.RunStatus(run_id="c", state="cancelled", spec=_spec("c").to_dict()))
        orch._update("c", "provisioning")
        assert orch.get_status("c").state == "cancelled", "cancelled must not become provisioning"
        orch._update("c", "running", cost_usd=1.0)
        assert orch.get_status("c").state == "cancelled"
        # A late terminal completion must NOT resurrect/relabel a user cancellation.
        orch._update("c", "failed", error="x")
        assert orch.get_status("c").state == "cancelled"
        # Same-state writes still apply terminal field updates.
        orch._update("c", "cancelled", cost_usd=2.0)
        assert orch.get_status("c").state == "cancelled"
        assert orch.get_status("c").cost_usd == 2.0
