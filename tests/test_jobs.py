"""Durable run primitives: handle persistence, polling state machine, supervisor retry,
cross-process cancel, and attach (CPU-only; all network mocked)."""

from __future__ import annotations

import base64
import io
import re
import sys
import tempfile

import pytest

_RUNPOD_FINGERPRINT = "rpk-0123456789ab"


def _runpod_handle(
    jobs, endpoint_id="ep", endpoint_name="name", job_id="job", attempt=0, started_ts=1.0
):
    return jobs.JobHandle(
        endpoint_id,
        endpoint_name,
        _RUNPOD_FINGERPRINT,
        job_id,
        attempt,
        started_ts,
    )


def _runpod_handle_dict(
    jobs,
    endpoint_id="ep",
    endpoint_name="name",
    job_id="job",
    attempt=0,
    started_ts=1.0,
):
    return _runpod_handle(
        jobs,
        endpoint_id=endpoint_id,
        endpoint_name=endpoint_name,
        job_id=job_id,
        attempt=attempt,
        started_ts=started_ts,
    ).to_dict()


# ---------------------------------------------------------------------------
# decode_output / JobHandle
# ---------------------------------------------------------------------------


def test_job_handle_roundtrip_and_rejects_legacy_shapes():
    from flash.providers.runpod.jobs import JobHandle

    handle = JobHandle(
        "ep123",
        "flash-5090-abc",
        _RUNPOD_FINGERPRINT,
        "job456",
        2,
        12_345.0,
    )
    assert JobHandle.from_dict(handle.to_dict()) == handle
    endpoint_only = JobHandle(
        "ep-cleanup",
        "flash-cleanup",
        _RUNPOD_FINGERPRINT,
        None,
        3,
        12_346.0,
    )
    assert "job_id" not in endpoint_only.to_dict()
    assert JobHandle.from_dict(endpoint_only.to_dict()) == endpoint_only

    valid = handle.to_dict()
    for missing in (
        "provider",
        "endpoint_id",
        "endpoint_name",
        "key_fingerprint",
        "attempt",
        "started_ts",
    ):
        legacy = dict(valid)
        legacy.pop(missing)
        with pytest.raises(ValueError, match="persisted RunPod"):
            JobHandle.from_dict(legacy)
    with pytest.raises(ValueError, match="attempt identity is invalid"):
        JobHandle.from_dict({**valid, "attempt": "2"})


def test_decode_output_success():
    import cloudpickle

    from flash.providers.runpod.jobs import decode_output

    metrics = {"trained_eval_acc": 0.9, "cost_usd": 0.5}
    out = {"success": True, "result": base64.b64encode(cloudpickle.dumps(metrics)).decode()}
    assert decode_output(out) == metrics


def test_decode_output_error_includes_stdout_tail():
    from flash.providers.runpod.jobs import decode_output

    stdout = "STDOUT-BEGIN\n" + ("x" * 5000) + "\nSTDOUT-END"
    with pytest.raises(RuntimeError) as ei:
        decode_output({"success": False, "error": "boom", "stdout": stdout})
    msg = str(ei.value)
    assert "boom" in msg
    assert "STDOUT-BEGIN" in msg
    assert "STDOUT-END" in msg


def test_decode_output_client_mode_serverless_handler():
    """Baked-image path: the serverless rp_handler returns the metrics dict directly (RunPod
    surfaces it as job["output"]), with no Flash success/result envelope — return it as-is."""
    from flash.providers.runpod.jobs import decode_output

    metrics = {"trained_eval_acc": 0.87, "train_wall": 12.3, "cost_usd": 0.04}
    assert decode_output(metrics) == metrics
    # a client-mode error surfaces as an error key
    with pytest.raises(RuntimeError):
        decode_output({"error": "handler blew up"})


def test_decode_output_client_mode_error_includes_stdout_tail():
    """Client-mode failures must also carry the worker stdout tail (poll_job root-causes
    crashes from it) — same as the Flash envelope path."""
    from flash.providers.runpod.jobs import decode_output

    stdout = "STDOUT-BEGIN\n" + ("z" * 5000) + "\nSTDOUT-END"
    with pytest.raises(RuntimeError) as ei:
        decode_output({"error": "vllm crashed", "stdout": stdout})
    msg = str(ei.value)
    assert "vllm crashed" in msg
    assert "worker stdout tail" in msg
    assert "STDOUT-BEGIN" in msg
    assert "STDOUT-END" in msg


# ---------------------------------------------------------------------------
# poll_job state machine (mocked runpod_api)
# ---------------------------------------------------------------------------
def _poll(monkeypatch, statuses, heartbeats=None, stall_after_s=10.0):
    import cloudpickle

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    seq = iter(statuses)
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: next(seq))
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    hb_iter = iter(heartbeats) if heartbeats is not None else None

    reader = (lambda force=False: next(hb_iter, None)) if hb_iter is not None else None
    h = _runpod_handle(jobs)
    ok_payload = {
        "success": True,
        "result": base64.b64encode(cloudpickle.dumps({"acc": 1.0})).decode(),
    }
    return jobs.poll_job(
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


def test_surface_heartbeat_logs_gpu_status(monkeypatch):
    from flash.providers._poll import surface_heartbeat

    lines = []
    hb = {
        "stage": "sft_step",
        "step": 12,
        "loss": 1.23456,
        "ts": 123.0,
        "attempt": 0,
        "gpu": {
            "device_name": "RTX 5090",
            "driver_version": "575.57",
            "torch_cuda": "12.8",
            "gpu_util_pct": 97,
            "memory_used_gb": 22.25,
            "memory_total_gb": 31.8,
            "temperature_c": 68,
            "power_w": 411.3,
            "power_limit_w": 575.0,
            "processes": [{"pid": 1234, "process_name": "/usr/bin/python", "used_memory_gb": 21.9}],
        },
    }
    monkeypatch.setattr("flash.providers._poll._record_heartbeat", lambda _hb: None)

    key, stage = surface_heartbeat(lambda: hb, None, lines.append)

    assert key == ("sft_step", 12, 123.0, 0)  # attempt is part of the key (shared seed hb path)
    assert stage == "sft_step"
    assert len(lines) == 1
    line = lines[0]
    assert "worker: stage=sft_step step=12 loss=1.2346" in line
    assert "gpu[RTX 5090" in line
    assert "util=97%" in line
    assert "mem=22.2GB/31.8GB" in line
    assert "power=411W/575W" in line
    assert "procs=python:1234:21.9GB" in line


def test_poll_job_failure(monkeypatch):
    res, _ = _poll(
        monkeypatch,
        [{"status": "IN_PROGRESS"}, {"status": "FAILED", "error": "worker exploded"}],
    )
    assert not res.ok
    assert res.failure == "job_failed"
    assert "worker exploded" in res.detail


def test_poll_job_failure_surfaces_forced_heartbeat(monkeypatch):
    import io

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda eid, jid, **_kw: {"status": "FAILED", "error": "worker exploded"},
    )
    log = io.StringIO()
    hb = {
        "run_id": "missing-local-status-is-ok",
        "stage": "boot",
        "ts": 456.0,
        "gpu": {"device_name": "RTX 5090", "gpu_util_pct": 1, "memory_total_gb": 31.8},
    }

    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda force=False: hb,
        log=log,
    )

    assert not res.ok
    assert res.failure == "job_failed"
    assert "worker: stage=boot" in log.getvalue()
    assert "gpu[RTX 5090" in log.getvalue()


def test_poll_job_failure_appends_worker_artifacts(monkeypatch):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    seq = iter(
        [
            {"status": "IN_PROGRESS"},
            {
                "status": "FAILED",
                "error": "train phase 'sft' produced no /tmp/metrics.json (it crashed before finishing)",
            },
        ]
    )
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: next(seq))
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    calls = {"force": None}

    def failure_detail_reader(force=False):
        calls["force"] = force
        return (
            "--- error_sft.txt ---\n"
            "Traceback (most recent call last):\nImportError: no module named flash_attn\n"
            "--- console_sft.txt ---\nworker console tail"
        )

    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda force=False: None,
        failure_detail_reader=failure_detail_reader,
    )
    assert not res.ok
    assert res.failure == "job_failed"
    assert calls["force"] is True
    assert "produced no /tmp/metrics.json" in res.detail
    assert "ImportError: no module named flash_attn" in res.detail
    assert "worker console tail" in res.detail


def test_poll_job_platform_preempt_maps_to_job_preempted(monkeypatch):
    # Platform terminations -> structured "job_preempted" (retried), not "job_failed".
    for status in ("CANCELLED", "TIMED_OUT"):
        res, _ = _poll(
            monkeypatch,
            [{"status": "IN_PROGRESS"}, {"status": status}],
        )
        assert not res.ok
        assert res.failure == "job_preempted", status
        assert f"[{status}]" in res.detail


def test_poll_job_platform_preempt_does_not_read_worker_artifacts(monkeypatch):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda eid, jid, **_kw: {"status": "TIMED_OUT", "error": "timeout"},
    )
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)

    def failure_detail_reader(force=False):
        raise AssertionError("platform terminations should not read worker artifacts")

    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda force=False: None,
        failure_detail_reader=failure_detail_reader,
    )
    assert not res.ok
    assert res.failure == "job_preempted"
    assert "timeout" in res.detail


def _poll_failed_with_heartbeat(monkeypatch, hb):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    hb = {"ts": 2.0, "attempt": 0, **hb}
    seq = iter([{"status": "IN_PROGRESS"}, {"status": "FAILED", "error": "boom"}])
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: next(seq))
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    return jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda force=False: hb,
        stall_after_s=10.0,
    )


def test_poll_job_failed_with_retriable_heartbeat_is_job_preempted(monkeypatch):
    # Worker stamped retriable=True -> infra-shaped, retried.
    res = _poll_failed_with_heartbeat(monkeypatch, {"stage": "error_sft", "retriable": True})
    assert not res.ok
    assert res.failure == "job_preempted"


def test_poll_job_failed_without_retriable_heartbeat_is_job_failed(monkeypatch):
    # No retriable flag -> a real code error, fails fast.
    res = _poll_failed_with_heartbeat(monkeypatch, {"stage": "error_sft", "error": "ValueError"})
    assert not res.ok
    assert res.failure == "job_failed"


def test_worker_flagged_retriable_requires_exact_attempt_and_timestamp():
    from flash.providers.runpod.jobs import worker_flagged_retriable

    def reader(hb):
        return lambda force=False: hb

    current = {"stage": "rl_step", "ts": 10_500.0, "attempt": 1, "retriable": True}
    assert worker_flagged_retriable(reader(current), launch_ts=10_000.0, current_attempt=1)

    invalid = (
        {"stage": "rl_step", "ts": 10_500.0, "attempt": 0, "retriable": True},
        {"stage": "rl_step", "ts": 10_500.0, "retriable": True},
        {"stage": "rl_step", "attempt": 1, "retriable": True},
        {"stage": "rl_step", "ts": 9_000.0, "attempt": 1, "retriable": True},
        {"stage": "rl_step", "ts": 10_500.0, "attempt": "1", "retriable": True},
    )
    for heartbeat in invalid:
        assert not worker_flagged_retriable(
            reader(heartbeat),
            launch_ts=10_000.0,
            current_attempt=1,
        )


def test_poll_job_completed_decode_error_consults_worker_flags(monkeypatch):
    # COMPLETED but the output decodes as an error (a handler exception). An infra failure can
    # surface here too, so poll_job must consult the worker heartbeat -> job_preempted when the
    # worker stamped retriable, not silently drop it as a plain job_failed.
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    # An output envelope that decode_output raises RuntimeError on (success False).
    bad = {"success": False, "error": "boom", "stdout": "x"}
    monkeypatch.setattr(
        runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "COMPLETED", "output": bad}
    )
    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda force=False: {"retriable": True, "attempt": 0, "ts": 2.0},
        failure_detail_reader=lambda force=False: "--- error_sft.txt ---\nCUDA out of memory",
    )
    assert res.failure == "job_preempted"
    assert "CUDA out of memory" in res.detail


def test_poll_job_stall_detection(monkeypatch):
    # job stays IN_PROGRESS forever, heartbeat never advances -> stall
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    h = _runpod_handle(jobs)
    res = jobs.poll_job(h, interval_s=0, heartbeat_reader=lambda: None, stall_after_s=150.0)
    assert not res.ok
    assert res.failure == "stalled"


def test_poll_job_in_queue_capacity_stall(monkeypatch):
    # Job sits IN_QUEUE forever (no worker ever accepts it: no RunPod capacity for the pinned
    # GPU class). RunPod surfaces no THROTTLED/UNHEALTHY worker, so the health-probe fast-fails
    # never arm -> the queue_grace_s backstop must trip a retryable stall well before the ~50 min
    # setup_grace_s, so the runner's gpu-walk re-provisions on the next-best class.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    # endpoint_health raises (the common real case for a brand-new endpoint with no workers): its
    # block is swallowed by `except: pass`, so the throttled/unhealthy fast-fails can't arm — the
    # queue backstop must still trip off the authoritative job status alone.
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: (_ for _ in ()).throw(RuntimeError("no workers yet")),
    )
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    h = _runpod_handle(jobs)
    res = jobs.poll_job(
        h,
        interval_s=0,
        heartbeat_reader=lambda: None,
        setup_grace_s=5000.0,  # large cold-start budget must NOT govern a never-scheduled queue job
        queue_grace_s=900.0,
    )
    assert not res.ok
    # Never scheduled (no capacity) is reported distinctly from a scheduled-then-stalled worker.
    assert res.failure == "no_capacity"
    assert "IN_QUEUE" in res.detail
    assert "next-best GPU" in res.detail


def test_capacity_grace_scales_with_gpu_walk_position():
    # The two no-capacity backstops — IN_QUEUE with no worker (queue_grace_s) and a worker stuck
    # THROTTLED (throttled_grace_s) — are tuned to the gpu-walk position. While a next-best class
    # still exists they wait ~5 min, long enough to ride out a brief blip but short enough to hand
    # off promptly to the next-best class. On the LAST candidate there is nowhere to walk, so they
    # wait ~15 min before giving up. A *placed* worker that is still cold-starting is governed by
    # the much larger setup_grace_s and must NOT be shortened — assert it stays large so we never
    # abandon a legitimately-initializing worker.
    import inspect

    from flash.providers.runpod import jobs

    not_last = jobs.stall_kwargs()  # default on_last_gpu=False
    assert not_last["queue_grace_s"] == 300.0
    assert not_last["throttled_grace_s"] == 300.0
    assert not_last["setup_grace_s"] >= 1800.0  # cold-start budget unchanged

    last = jobs.stall_kwargs(on_last_gpu=True)
    assert last["queue_grace_s"] == 900.0
    assert last["throttled_grace_s"] == 900.0
    assert last["setup_grace_s"] == not_last["setup_grace_s"]  # only the capacity backstops move

    sig = inspect.signature(jobs.poll_job)
    assert sig.parameters["queue_grace_s"].default == 300.0
    assert sig.parameters["throttled_grace_s"].default == 300.0
    assert sig.parameters["setup_grace_s"].default >= 1800.0


def test_reattach_poll_reproduces_persisted_on_last_gpu(monkeypatch):
    from flash.providers.base import JobHandle
    from flash.providers.runpod import PROVIDER
    from flash.providers.runpod import jobs as jobs
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    captured: dict = {}

    def fake_poll_job(handle, **kw):
        captured.update(kw)
        return jobs.PollResult(True, metrics={})

    monkeypatch.setattr(jobs, "poll_job", fake_poll_job)

    spec = JobSpec(
        run_id="reattach",
        model="Qwen/Qwen3.5-0.8B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=1, hf_repo=""),
        gpu=GpuSpec(type="A100 PCIe"),
    )
    base = {
        "provider": "runpod",
        "endpoint_id": "ep",
        "endpoint_name": "n",
        "key_fingerprint": _RUNPOD_FINGERPRINT,
        "job_id": "j",
        "started_ts": 1.0,
    }

    PROVIDER.poll(JobHandle.from_dict({**base, "on_last_gpu": True, "attempt": 2}), spec, spec.seed)
    assert captured["queue_grace_s"] == 900.0
    assert captured["throttled_grace_s"] == 900.0
    assert captured["current_attempt"] == 2

    captured.clear()
    PROVIDER.poll(
        JobHandle.from_dict({**base, "on_last_gpu": False, "attempt": 0}), spec, spec.seed
    )
    assert captured["queue_grace_s"] == 300.0
    assert captured["current_attempt"] == 0

    with pytest.raises(ValueError, match="attempt identity is invalid"):
        PROVIDER.poll(JobHandle.from_dict(base), spec, spec.seed)


def test_submit_run_payload_carries_code_prefix(monkeypatch):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    spec = JobSpec(
        run_id="flash-code-prefix",
        model="Qwen/Qwen3.5-0.8B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(type="RTX 4090"),
    )
    submitted: dict = {}
    monkeypatch.setattr(
        jobs, "deploy_train_endpoint", lambda *a, **k: ("ep", "endpoint-name", _RUNPOD_FINGERPRINT)
    )
    monkeypatch.setattr(jobs, "build_function_input", lambda payload: payload)
    monkeypatch.setattr(
        runpod_api,
        "submit_job",
        lambda endpoint_id, payload, **_kw: (
            submitted.update({"endpoint_id": endpoint_id, "payload": payload}) or "job-1"
        ),
    )
    monkeypatch.setattr(jobs, "poll_job", lambda *a, **k: jobs.PollResult(True, metrics={}))

    pinned_prefix = "code/0123456789abcdef0123456789abcdef/flash"
    assert jobs.submit_run(
        spec,
        seed=spec.seed,
        code_prefix=pinned_prefix,
        deadline_at=10_000_000_000.0,
    ).ok
    assert submitted["endpoint_id"] == "ep"
    assert submitted["payload"]["code_prefix"] == pinned_prefix


def test_runpod_submit_failure_is_retryable_only_after_confirmed_endpoint_deletion(monkeypatch):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs, train
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    spec = JobSpec(
        run_id="runpod-submit-retryable",
        model="Qwen/Qwen3.5-0.8B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(type="RTX 4090"),
    )
    original = RuntimeError("ambiguous queue post")
    handles = []
    deleted = []
    monkeypatch.setattr(train, "build_worker_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train, "chalk_extra_pip", lambda _spec: [])
    monkeypatch.setattr(
        jobs,
        "deploy_train_endpoint",
        lambda *_args, **_kwargs: ("endpoint", "name", _RUNPOD_FINGERPRINT),
    )
    monkeypatch.setattr(
        runpod_api,
        "submit_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(original),
    )
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda endpoint_id, _fingerprint: deleted.append(endpoint_id) or True,
    )

    with pytest.raises(RuntimeError) as caught:
        jobs.submit_run(
            spec,
            spec.seed,
            attempt=4,
            on_handle=handles.append,
            deadline_at=10_000_000_000.0,
        )

    assert caught.value is original
    assert deleted == ["endpoint"]
    assert handles == []


@pytest.mark.parametrize("deletion_mode", ["false", "exception"])
def test_runpod_submit_failure_persists_endpoint_only_cleanup_handle(monkeypatch, deletion_mode):
    from flash.providers.base import UnreconciledCreateError
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs, train
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    spec = JobSpec(
        run_id=f"runpod-submit-unreconciled-{deletion_mode}",
        model="Qwen/Qwen3.5-0.8B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(type="RTX 4090"),
    )
    handles = []
    monkeypatch.setattr(train, "build_worker_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(train, "chalk_extra_pip", lambda _spec: [])
    monkeypatch.setattr(
        jobs,
        "deploy_train_endpoint",
        lambda *_args, **_kwargs: ("endpoint", "name", _RUNPOD_FINGERPRINT),
    )
    monkeypatch.setattr(
        runpod_api,
        "submit_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ambiguous queue post")),
    )

    def delete_endpoint(_endpoint_id, _fingerprint):
        if deletion_mode == "exception":
            raise RuntimeError("deletion response unavailable")
        return False

    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", delete_endpoint)

    with pytest.raises(UnreconciledCreateError, match="could not be reconciled"):
        jobs.submit_run(
            spec,
            spec.seed,
            attempt=4,
            on_handle=handles.append,
            deadline_at=10_000_000_000.0,
        )

    assert len(handles) == 1
    assert handles[0]["provider"] == "runpod"
    assert handles[0]["endpoint_id"] == "endpoint"
    assert handles[0]["attempt"] == 4
    assert "job_id" not in handles[0]


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"id": "different-job", "status": "CANCELLED"},
        {"id": "job-1", "status": "IN_PROGRESS"},
    ],
)
def test_runpod_cancel_rejects_unconfirmed_acknowledgement(monkeypatch, response):
    from flash.providers.base import JobHandle
    from flash.providers.runpod import RunpodProvider, jobs
    from flash.providers.runpod import api as runpod_api

    monkeypatch.setattr(runpod_api, "cancel_job", lambda _endpoint_id, _job_id, **_kw: response)
    handle = JobHandle.from_dict(
        _runpod_handle_dict(
            jobs,
            endpoint_id="endpoint-1",
            endpoint_name="endpoint-1",
            job_id="job-1",
        )
    )

    with pytest.raises(runpod_api.RunpodApiError, match="could not be confirmed"):
        RunpodProvider().cancel(handle)


def test_runpod_cancel_accepts_exact_cancelled_acknowledgement(monkeypatch):
    from flash.providers.base import JobHandle
    from flash.providers.runpod import RunpodProvider, jobs
    from flash.providers.runpod import api as runpod_api

    calls = []

    def cancel_job(endpoint_id, job_id, **_kw):
        calls.append((endpoint_id, job_id))
        return {"id": job_id, "status": "CANCELLED"}

    monkeypatch.setattr(runpod_api, "cancel_job", cancel_job)
    handle = JobHandle.from_dict(
        _runpod_handle_dict(
            jobs,
            endpoint_id="endpoint-1",
            endpoint_name="endpoint-1",
            job_id="job-1",
        )
    )

    RunpodProvider().cancel(handle)

    assert calls == [("endpoint-1", "job-1")]


def test_runpod_initial_and_reattached_poll_use_same_absolute_deadline(monkeypatch):
    from flash.providers.base import JobHandle, PollResult
    from flash.providers.runpod import RunpodProvider, jobs
    from flash.providers.runpod import api as runpod_api
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    spec = JobSpec(
        run_id="runpod-shared-deadline",
        model="Qwen/Qwen3.5-0.8B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(type="RTX 4090"),
    )
    deadline_at = 12_345.0
    captured = []

    monkeypatch.setattr(jobs.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        jobs, "deploy_train_endpoint", lambda *a, **k: ("ep", "endpoint-name", _RUNPOD_FINGERPRINT)
    )
    monkeypatch.setattr(runpod_api, "submit_job", lambda *_a, **_k: "job-1")

    def fake_poll(handle, **kwargs):
        captured.append(kwargs["deadline_at"])
        return PollResult(True, metrics={})

    monkeypatch.setattr(jobs, "poll_job", fake_poll)
    provider = RunpodProvider()
    assert provider.submit_run(spec, seed=spec.seed, _deadline_at=deadline_at).ok
    handle = JobHandle.from_dict(
        _runpod_handle_dict(
            jobs,
            endpoint_id="ep",
            endpoint_name="endpoint-name",
            job_id="job-1",
        )
    )
    assert provider.poll(handle, spec, seed=spec.seed, _deadline_at=deadline_at).ok

    assert captured == [deadline_at, deadline_at]


def test_runpod_endpoint_time_consumption_blocks_queue_job_creation(monkeypatch):
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    spec = JobSpec(
        run_id="runpod-deadline-boundary",
        model="Qwen/Qwen3.5-0.8B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(type="RTX 4090"),
    )
    now = {"value": 100.0}
    monkeypatch.setattr(jobs.time, "time", lambda: now["value"])

    def _deploy(*_args, **_kwargs):
        now["value"] = 141.0
        return "ep", "endpoint-name", _RUNPOD_FINGERPRINT

    monkeypatch.setattr(jobs, "deploy_train_endpoint", _deploy)
    submitted = []
    deleted = []
    monkeypatch.setattr(
        runpod_api,
        "submit_job",
        lambda *_args, **_kwargs: submitted.append(True) or "job-1",
    )
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda endpoint_id, _fingerprint: deleted.append(endpoint_id) or True,
    )

    with pytest.raises(RuntimeError, match="60-second minimum provider allowance"):
        jobs.submit_run(spec, seed=spec.seed, deadline_at=200.0)

    assert submitted == []
    assert deleted == ["ep"]


def test_poll_job_in_queue_then_progress_does_not_false_stall(monkeypatch):
    # A job that leaves IN_QUEUE (a worker picks it up) must clear the queue timer: the later
    # IN_PROGRESS/COMPLETED path is governed by the heartbeat/setup windows, never by queue_grace_s.
    import cloudpickle

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    ok = {"success": True, "result": base64.b64encode(cloudpickle.dumps({"acc": 1.0})).decode()}
    # A few IN_QUEUE polls, THEN IN_PROGRESS (a worker picked it up), then COMPLETED — exercising the
    # actual leave-the-queue transition the queue timer must clear on. Real wall-clock (no fake clock)
    # so elapsed stays far under queue_grace_s; the timer clears on leaving IN_QUEUE and never
    # false-stalls (the IN_PROGRESS path is governed by heartbeat/setup windows, not queue_grace_s).
    seq = iter(
        [{"status": "IN_QUEUE"}] * 5
        + [{"status": "IN_PROGRESS"}] * 3
        + [{"status": "COMPLETED", "output": ok}]
    )
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: next(seq))
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    # The IN_QUEUE phase triggers poll_job's endpoint_health probe on the first loop; stub it so the
    # test is hermetic (never hits the network even if RUNPOD_API_KEY is set in the environment).
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: {"workers": {"ready": 1, "running": 1}},
    )
    h = _runpod_handle(jobs)
    res = jobs.poll_job(h, interval_s=0, heartbeat_reader=lambda: None, queue_grace_s=900.0)
    assert res.ok


def test_poll_job_throttled_timer_resets_on_leaving_queue(monkeypatch):
    # A worker throttled in its FIRST queue window must not carry a stale arm-time across an
    # IN_PROGRESS spell: if RunPod re-queues the job (still throttled), the throttled grace must be
    # measured from the re-queue, not the original arm. Otherwise the first re-queue probe fires
    # no_capacity instantly, defeating throttled_grace_s. Clock advances one tick per job_status poll.
    import base64

    import cloudpickle

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    ok = {"success": True, "result": base64.b64encode(cloudpickle.dumps({"acc": 1.0})).decode()}
    statuses = iter(
        [
            {"status": "IN_QUEUE"},  # arm throttled_timer (~t=100)
            {
                "status": "IN_QUEUE"
            },  # accumulate (~t=200, still < grace 150 from t=100... fires at 250)
            {"status": "IN_PROGRESS"},  # leaves the queue -> timer must reset
            {
                "status": "IN_QUEUE"
            },  # re-queued, throttled: with a stale arm this would fire instantly
            {"status": "COMPLETED", "output": ok},
        ]
    )
    clock = {"t": 0.0}

    def fake_job_status(eid, jid, **_kw):
        clock["t"] += 100.0
        return next(statuses)

    monkeypatch.setattr(runpod_api, "job_status", fake_job_status)
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: {"workers": {"throttled": 1}},
    )
    monkeypatch.setattr(jobs.time, "time", lambda: clock["t"])
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)

    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: None,
        throttled_grace_s=150.0,
        queue_grace_s=100_000.0,
        setup_grace_s=100_000.0,
        stall_after_s=100_000.0,
    )
    assert res.ok, (
        res.detail
    )  # completed; the re-queue throttle timer re-armed fresh, no false no_capacity


def test_poll_job_setup_grace_before_first_heartbeat(monkeypatch):
    # No heartbeat ever (cold start that never finishes): must NOT trip the tight
    # stall_after_s window — it waits for the larger setup_grace_s instead.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    h = _runpod_handle(jobs)
    res = jobs.poll_job(
        h, interval_s=0, heartbeat_reader=lambda: None, stall_after_s=150.0, setup_grace_s=5000.0
    )
    assert res.failure == "stalled"
    # The larger setup budget governed, not the 150s training window.
    assert "during setup" in res.detail
    assert "limit 5000s" in res.detail


def test_poll_job_tight_stall_after_first_heartbeat(monkeypatch):
    # One heartbeat arrives (training started), then progress freezes: now the tight
    # stall_after_s window applies, not the big setup grace.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))

    hbs = iter([{"stage": "train", "step": 1, "ts": 1, "attempt": 0}])  # then StopIteration -> None

    def reader():
        return next(hbs, None)

    h = _runpod_handle(jobs)
    res = jobs.poll_job(
        h, interval_s=0, heartbeat_reader=reader, stall_after_s=150.0, setup_grace_s=5000.0
    )
    assert res.failure == "stalled"
    assert "during training" in res.detail


def test_poll_job_ignores_prior_attempt_heartbeat_keeps_setup_grace(monkeypatch):
    # On a retry (current_attempt=1) the shared seed heartbeat path first returns the PRIOR attempt's
    # leftover TRAINING heartbeat (attempt=0). It must be IGNORED so this cold start keeps the larger
    # setup_grace_s instead of latching the dead attempt and dropping to the tight stall_after_s —
    # otherwise a healthy-but-slow retry cold start (image pull + big snapshot_download) false-stalls.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))

    # Every read returns attempt 0's leftover training heartbeat; this poll is attempt 1.
    leftover = {"stage": "train", "step": 5, "ts": 1, "attempt": 0}
    h = _runpod_handle(jobs, attempt=1)
    res = jobs.poll_job(
        h,
        interval_s=0,
        heartbeat_reader=lambda force=False: leftover,
        stall_after_s=150.0,
        setup_grace_s=5000.0,
        current_attempt=1,
    )
    assert res.failure == "stalled"
    # The foreign-attempt heartbeat did NOT tighten the window: setup grace governed.
    assert "during setup" in res.detail
    assert "limit 5000s" in res.detail


def test_reattach_normalizes_persisted_attempt_once_for_failure_reader_and_poll(monkeypatch):
    import types

    from flash.providers import base
    from flash.providers.runpod import RunpodProvider, jobs

    captured = {}

    def fake_poll_job(rh, **kw):
        captured["handle"] = rh
        captured["current_attempt"] = kw["current_attempt"]
        return base.PollResult(ok=True)

    def fake_failure_reader(*_args, attempt, **_kwargs):
        captured["failure_attempt"] = attempt
        return lambda: None

    monkeypatch.setattr(jobs, "poll_job", fake_poll_job)
    monkeypatch.setattr(jobs, "make_hf_failure_detail_reader", fake_failure_reader)
    spec = types.SimpleNamespace(
        phase="sft", run_id="r1", seed=0, train=types.SimpleNamespace(hf_repo="org/repo")
    )

    handle = base.JobHandle(
        provider="runpod",
        data={
            "endpoint_id": "ep",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "j",
            "attempt": 2,
            "started_ts": 12_345.0,
        },
    )
    RunpodProvider().poll(handle, spec, spec.seed)
    assert captured["current_attempt"] == 2
    assert captured["failure_attempt"] == 2
    assert captured["handle"].attempt == 2
    assert captured["handle"].started_ts == 12_345.0

    handle = base.JobHandle(
        provider="runpod",
        data={
            "endpoint_id": "ep",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "j",
            "started_ts": 12_345.0,
        },
    )
    with pytest.raises(ValueError, match="attempt identity is invalid"):
        RunpodProvider().poll(handle, spec, spec.seed)


@pytest.mark.parametrize("raw_attempt", ["", "x", None, True, -1, 1.5])
def test_reattach_rejects_explicit_malformed_persisted_attempt(monkeypatch, raw_attempt):
    import types

    from flash.providers import base
    from flash.providers.runpod import RunpodProvider, jobs

    monkeypatch.setattr(
        jobs,
        "poll_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not poll")),
    )
    spec = types.SimpleNamespace(
        phase="sft", run_id="r1", seed=0, train=types.SimpleNamespace(hf_repo=None)
    )
    handle = base.JobHandle(
        provider="runpod",
        data={
            "endpoint_id": "ep",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "j",
            "attempt": raw_attempt,
            "started_ts": 1.0,
        },
    )

    with pytest.raises(ValueError, match="attempt identity is invalid"):
        RunpodProvider().poll(handle, spec, spec.seed)


def test_reattach_rejects_endpoint_only_handle(monkeypatch):
    import types

    from flash.providers import base
    from flash.providers.runpod import RunpodProvider, jobs

    monkeypatch.setattr(
        jobs,
        "poll_job",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not poll")),
    )
    spec = types.SimpleNamespace(
        phase="sft", run_id="r1", seed=0, train=types.SimpleNamespace(hf_repo=None)
    )
    handle = base.JobHandle(
        provider="runpod",
        data={
            "endpoint_id": "ep",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "attempt": 0,
            "started_ts": 1.0,
        },
    )

    with pytest.raises(ValueError, match="endpoint-only"):
        RunpodProvider().poll(handle, spec, spec.seed)


def test_poll_job_setup_heartbeat_does_not_tighten(monkeypatch):
    # A cold-start (setup) heartbeat like "boot" proves liveness but must NOT switch to the
    # tight training window — the slow model-load/vLLM-init still has to fit setup_grace_s.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))

    hbs = iter([{"stage": "boot", "step": None, "ts": 1, "attempt": 0}])  # then None

    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: next(hbs, None),
        stall_after_s=150.0,
        setup_grace_s=5000.0,
    )
    assert res.failure == "stalled"
    assert "during setup" in res.detail
    assert "limit 5000s" in res.detail


def test_poll_job_liveness_heartbeat_does_not_reset_progress(monkeypatch):
    # Liveness pings refresh visible status/logs, but they must not extend the provider's progress
    # clock. Otherwise a wedged setup thread could ping "alive" forever and mask the stall.
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    state = {"t": 0.0}

    def _time():
        state["t"] += 100.0
        return state["t"]

    monkeypatch.setattr(jobs.time, "time", _time)
    hbs = iter(
        [
            {"stage": "boot", "step": None, "ts": 1000.0, "attempt": 0},
            {
                "stage": "sft_initializing",
                "step": None,
                "ts": 2000.0,
                "attempt": 0,
                "liveness": True,
            },
        ]
    )

    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: next(hbs, None),
        stall_after_s=150.0,
        setup_grace_s=250.0,
    )
    assert res.failure == "stalled"
    assert "during setup" in res.detail
    assert state["t"] < 1200.0, "liveness must not buy a fresh setup-grace window"


def test_poll_job_stale_late_heartbeat_does_not_reset_progress(monkeypatch):
    # A newer heartbeat can be skipped (bounded _HB_UPLOAD_LOCK) while an OLDER one lands late,
    # changing heartbeat.json content but carrying an OLDER ts. That stale heartbeat must NOT buy a
    # fresh stall window for a genuinely stuck worker — progress is gated on the heartbeat ts
    # ADVANCING, not on the content merely changing. Proven by ABSOLUTE simulated stall time: a run
    # whose 2nd heartbeat is STALE stalls at the SAME time as a run with no 2nd heartbeat (stale ==
    # no-op), while a run whose 2nd heartbeat is FRESH stalls strictly LATER (it reset progress).
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)

    def _stall_abs_time(second_hb):
        state = {"t": 0.0}

        def _time():
            state["t"] += 100.0
            return state["t"]

        monkeypatch.setattr(jobs.time, "time", _time)
        seq = [{"stage": "train", "step": 5, "ts": 1000, "attempt": 0}]
        if second_hb is not None:
            seq.append(second_hb)
        hbs = iter(seq)
        res = jobs.poll_job(
            _runpod_handle(jobs),
            interval_s=0,
            heartbeat_reader=lambda: next(hbs, None),
            stall_after_s=150.0,
            setup_grace_s=5000.0,
        )
        assert res.failure == "stalled"
        assert "during training" in res.detail  # the fresh ts=1000 hb tightened the window
        return state["t"]  # absolute simulated time when it stalled

    none_run = _stall_abs_time(None)
    stale_run = _stall_abs_time({"stage": "train", "step": 4, "ts": 500, "attempt": 0})
    fresh_run = _stall_abs_time({"stage": "train", "step": 6, "ts": 2000, "attempt": 0})

    assert stale_run == none_run, "a stale late heartbeat must be a no-op for progress"
    assert fresh_run > none_run, "a genuinely newer heartbeat does reset progress (stalls later)"


def test_poll_job_gapfill_step0_does_not_tighten(monkeypatch):
    # The train-liveness gap-filler emits rl_step/sft_step at step=0 throughout the silent FIRST step
    # (a cold vLLM rollout can run many minutes before global_step ticks to 1). That step=0 ping is a
    # NON-setup stage but reports NO completed step, so it proves liveness without meaning training has
    # started — it must keep the larger setup grace, not switch to the tight training window.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))

    hbs = iter(
        [{"stage": "rl_step", "step": 0, "ts": 1, "attempt": 0}]
    )  # gap-filler before the first real step
    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: next(hbs, None),
        stall_after_s=150.0,
        setup_grace_s=5000.0,
    )
    assert res.failure == "stalled"
    assert "during setup" in res.detail  # step=0 did NOT tighten
    assert "limit 5000s" in res.detail
    # Sanity: the SAME stage at step>=1 (a real completed step) DOES tighten to the training window.
    clock2 = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock2))
    hbs2 = iter([{"stage": "rl_step", "step": 1, "ts": 1, "attempt": 0}])
    res2 = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: next(hbs2, None),
        stall_after_s=150.0,
        setup_grace_s=5000.0,
    )
    assert res2.failure == "stalled"
    assert "during training" in res2.detail


def test_poll_job_malformed_step_does_not_crash(monkeypatch):
    # a heartbeat whose `step` is missing or non-numeric must NOT raise inside the poll loop (there is
    # no local handler — a ValueError would abort poll_job). the step is validated strictly through the
    # same bounded integer helper; an invalid step stays setup-classified and must not raise.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))

    hbs = iter(
        [{"stage": "rl_step", "step": "not-a-number", "ts": 1, "attempt": 0}]
    )  # malformed step
    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: next(hbs, None),
        stall_after_s=150.0,
        setup_grace_s=5000.0,
    )
    assert res.failure == "stalled"  # did NOT raise
    assert "during setup" in res.detail  # unparseable step -> treated as 0 -> setup grace kept
    assert "limit 5000s" in res.detail


def test_poll_job_older_attempt_heartbeat_does_not_reset_progress(monkeypatch):
    # Attempts (retries / preemptions) SHARE this run's HF heartbeat path. A prior attempt's worker,
    # still shutting down, can upload a heartbeat with an ADVANCING ts but a LOWER attempt number. By
    # ts alone that looks like progress, but it belongs to a dead attempt and must NOT buy a fresh
    # stall window for the new attempt's stuck worker. Proven by ABSOLUTE simulated stall time: the
    # older-attempt second heartbeat stalls at the SAME time as no second heartbeat (it's a no-op),
    # while a same-attempt heartbeat with a newer ts resets progress (stalls strictly later).
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)

    def _stall_abs_time(second_hb):
        state = {"t": 0.0}

        def _time():
            state["t"] += 100.0
            return state["t"]

        monkeypatch.setattr(jobs.time, "time", _time)
        # First heartbeat is attempt 1 at ts=1000 (training started under the current attempt).
        seq = [{"stage": "train", "step": 5, "ts": 1000, "attempt": 1}]
        if second_hb is not None:
            seq.append(second_hb)
        hbs = iter(seq)
        res = jobs.poll_job(
            _runpod_handle(jobs, attempt=1),
            interval_s=0,
            heartbeat_reader=lambda: next(hbs, None),
            stall_after_s=150.0,
            setup_grace_s=5000.0,
        )
        assert res.failure == "stalled"
        return state["t"]

    none_run = _stall_abs_time(None)
    # Attempt 0 (older than the current attempt 1) with an even NEWER ts -> a dead attempt's late
    # heartbeat. Must be ignored -> same stall time as no second heartbeat.
    older_run = _stall_abs_time({"stage": "train", "step": 9, "ts": 9999, "attempt": 0})
    # Same attempt (1) with a newer ts -> real progress -> resets, stalls later.
    same_run = _stall_abs_time({"stage": "train", "step": 6, "ts": 2000, "attempt": 1})

    assert older_run == none_run, "an older attempt's late heartbeat must be a no-op for progress"
    assert same_run > none_run, "a same-attempt newer heartbeat does reset progress (stalls later)"


def test_poll_job_fast_fails_on_stuck_unhealthy_worker(monkeypatch):
    # A worker stuck UNHEALTHY while IN_QUEUE (e.g. a mutable image tag republished mid-pull) won't
    # self-recover, so poll_job must fail fast on unhealthy_grace_s and NOT burn the full
    # setup_grace_s (~50 min) — returning a retryable stall so the runner re-provisions a fresh
    # endpoint. Regression guard for the multi-hour "waited on a dead worker" failure.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: {
            "workers": {"unhealthy": 1, "running": 0, "ready": 0, "idle": 0, "initializing": 0}
        },
    )
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: None,
        stall_after_s=150.0,
        setup_grace_s=100000.0,  # huge: only the unhealthy fast-fail can trip here
        queue_grace_s=100000.0,  # huge: isolate the unhealthy path from the (tight) queue backstop
        unhealthy_grace_s=240.0,
    )
    assert res.failure == "stalled"  # infra-shaped -> runner retries on a fresh endpoint
    assert "unhealthy" in res.detail


def test_poll_job_transient_unhealthy_then_recovers_does_not_fail(monkeypatch):
    # A brief unhealthy blip during cold start that then yields a usable worker must NOT trip the
    # fast-fail (it resets once a usable/initializing worker appears) — only a STUCK unhealthy does.
    import base64
    import itertools

    import cloudpickle

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    statuses = iter(
        [
            {"status": "IN_QUEUE"},  # probe 1: unhealthy -> arm unhealthy_since
            {"status": "IN_QUEUE"},  # probe 2: usable worker -> reset (no fail)
            {"status": "IN_PROGRESS"},
            {
                "status": "COMPLETED",
                "output": {
                    "success": True,
                    "result": base64.b64encode(cloudpickle.dumps({"acc": 1.0})).decode(),
                },
            },
        ]
    )
    healths = iter(
        [
            {"workers": {"unhealthy": 1, "running": 0, "ready": 0, "idle": 0, "initializing": 0}},
            {"workers": {"unhealthy": 0, "running": 1, "ready": 0, "idle": 0, "initializing": 0}},
        ]
    )
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: next(statuses))
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: next(healths, {"workers": {}}),
    )
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: None,
        stall_after_s=150.0,
        setup_grace_s=100000.0,
        queue_grace_s=100000.0,  # huge: isolate the transient-unhealthy recovery from the queue backstop
        unhealthy_grace_s=240.0,
    )
    assert res.ok
    assert res.metrics == {"acc": 1.0}


def test_poll_job_fast_fails_on_stuck_throttled_worker(monkeypatch):
    # A worker stuck THROTTLED while IN_QUEUE (RunPod has no capacity for the pinned GPU class)
    # won't self-recover, so poll_job must fail fast on throttled_grace_s and NOT burn the full
    # setup_grace_s (~50 min) — returning a retryable stall so the runner walks to the next-best
    # GPU. Regression guard for the observed "stuck queued for the whole wall-clock" failure.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    # poll_job is a pure function of its args (throttled_grace_s is passed below), so no env setup.
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: {
            "workers": {"throttled": 1, "running": 0, "ready": 0, "idle": 0, "initializing": 0}
        },
    )
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: None,
        stall_after_s=150.0,
        setup_grace_s=100000.0,  # huge: only the throttled fast-fail can trip here
        unhealthy_grace_s=100000.0,  # huge: isolate the throttled path
        queue_grace_s=100000.0,  # huge: isolate the throttled path from the (tight) queue backstop
        throttled_grace_s=300.0,
    )
    # A throttled-only worker was never usable -> reported as no_capacity (never scheduled),
    # infra-shaped so the runner retries on the next-best GPU.
    assert res.failure == "no_capacity"
    assert "THROTTLED" in res.detail


def test_poll_job_transient_throttled_then_recovers_does_not_fail(monkeypatch):
    # A brief throttle during cold start that then yields a usable worker must NOT trip the
    # fast-fail (it resets once a usable worker appears) — only a STUCK throttle does.
    import base64
    import itertools

    import cloudpickle

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    statuses = iter(
        [
            {"status": "IN_QUEUE"},  # probe 1: throttled -> arm throttled_since
            {"status": "IN_QUEUE"},  # probe 2: usable worker -> reset (no fail)
            {"status": "IN_PROGRESS"},
            {
                "status": "COMPLETED",
                "output": {
                    "success": True,
                    "result": base64.b64encode(cloudpickle.dumps({"acc": 1.0})).decode(),
                },
            },
        ]
    )
    healths = iter(
        [
            {"workers": {"throttled": 1, "running": 0, "ready": 0, "idle": 0, "initializing": 0}},
            {"workers": {"throttled": 0, "running": 1, "ready": 0, "idle": 0, "initializing": 0}},
        ]
    )
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: next(statuses))
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: next(healths, {"workers": {}}),
    )
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: None,
        stall_after_s=150.0,
        setup_grace_s=100000.0,
        queue_grace_s=100000.0,  # huge: isolate the transient-throttle recovery from the queue backstop
        throttled_grace_s=300.0,
    )
    assert res.ok
    assert res.metrics == {"acc": 1.0}


def test_failure_detail_reader_is_attempt_scoped(monkeypatch, tmp_path):
    """The failure-detail reader fetches error_<phase>_attempt<N>.txt (matching the worker's
    error_artifact_name), so a retry can't surface a prior attempt's stale traceback as the crash."""
    import huggingface_hub

    from flash.providers.runpod.jobs import make_hf_failure_detail_reader

    requested: list[str] = []
    err_file = tmp_path / "err.txt"
    err_file.write_text("BOOM traceback")

    def fake_dl(repo, path_in_repo, **kw):
        requested.append(path_in_repo)
        if path_in_repo.endswith("error_sft_attempt2.txt"):
            return str(err_file)
        raise FileNotFoundError(path_in_repo)  # console + other attempts absent

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_dl)
    reader = make_hf_failure_detail_reader("org/repo", "sft/run-1/seed0", "sft", attempt=2)
    detail = reader(force=True)
    assert "sft/run-1/seed0/error_sft_attempt2.txt" in requested
    assert "sft/run-1/seed0/error_sft_attempt0.txt" not in requested
    assert "--- error_sft_attempt2.txt ---" in detail
    assert "BOOM traceback" in detail


def test_failure_detail_reader_preserves_full_worker_artifacts(monkeypatch):
    from flash.providers import _hf_artifacts

    def fake_reader(hf_repo, path_in_repo, min_interval_s):
        def read(force=False):
            if path_in_repo.endswith("error_sft_attempt2.txt"):
                return "ERROR-BEGIN\n" + ("e" * 5000) + "\nERROR-END"
            if path_in_repo.endswith("console_sft.txt"):
                return "CONSOLE-BEGIN\n" + ("c" * 5000) + "\nCONSOLE-END"
            return None

        return read

    monkeypatch.setattr(_hf_artifacts, "make_hf_text_reader", fake_reader)
    reader = _hf_artifacts.make_hf_failure_detail_reader(
        "org/repo", "sft/run-1/seed0", "sft", attempt=2
    )

    detail = reader(force=True)

    assert "ERROR-BEGIN" in detail
    assert "ERROR-END" in detail
    assert "CONSOLE-BEGIN" in detail
    assert "CONSOLE-END" in detail


def test_poll_job_no_reader_keeps_tight_window(monkeypatch):
    # Without a heartbeat_reader we can't tell setup from training, so the larger
    # setup_grace must NOT silently slow stall detection — stay on stall_after_s.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=None,
        stall_after_s=150.0,
        setup_grace_s=5000.0,
    )
    assert res.failure == "stalled"
    assert "limit 150s" in res.detail


def test_poll_job_tolerates_transient_api_errors(monkeypatch):
    import cloudpickle

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    ok = {
        "success": True,
        "result": base64.b64encode(cloudpickle.dumps({"acc": 0.7})).decode(),
    }
    calls = {"n": 0}

    def flaky(eid, jid, **_kw):
        calls["n"] += 1
        if calls["n"] < 4:
            raise runpod_api.RunpodApiError("blip")
        return {"status": "COMPLETED", "output": ok}

    monkeypatch.setattr(runpod_api, "job_status", flaky)
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    res = jobs.poll_job(
        _runpod_handle(jobs, endpoint_name="n", job_id="j"), interval_s=0, stall_after_s=1e9
    )
    assert res.ok
    assert calls["n"] == 4


# ---------------------------------------------------------------------------
# Supervisor retry logic (runner) with mocked job submit
# ---------------------------------------------------------------------------
def _fresh_orchestrator(tmp, monkeypatch):
    from tests._helpers.runner import fresh_runner

    runner = fresh_runner(tmp, monkeypatch)
    import flash.lora_rank as rank_mod
    from flash.providers.runpod import api as runpod_api

    monkeypatch.setattr(
        runpod_api, "delete_endpoint_for_fingerprint", lambda endpoint_id, _fingerprint: True
    )
    monkeypatch.setattr(
        rank_mod,
        "adapter_artifact_identity",
        lambda *a, **k: rank_mod.AdapterArtifactIdentity(
            digest="identity-v1",
            config_sha256="config-v1",
            weight_filename="adapter_model.safetensors",
            weight_identity="weight-v1:123",
        ),
    )
    return runner


def _confirm_runpod_retry_teardown(monkeypatch):
    from flash.providers.runpod import api as runpod_api

    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(
        runpod_api, "delete_endpoint_for_fingerprint", lambda _endpoint_id, _fingerprint: True
    )


def _spec(run_id):
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    return JobSpec(
        run_id=run_id,
        model="Qwen/Qwen3.5-0.8B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=1),
        gpu=GpuSpec(type="RTX 4090", max_retries=2),
    )


def _adapter_config(*, rank=32, alpha=64):
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "Qwen/Qwen3.5-0.8B",
        "r": rank,
        "lora_alpha": alpha,
    }


def test_supervisor_retries_on_stall_then_succeeds(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        calls = {"n": 0}
        code_prefixes: list[str | None] = []
        uploaded: dict[str, str | None] = {}

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, code_prefix=None, **_):
            calls["n"] += 1
            code_prefixes.append(code_prefix)
            if on_handle:
                on_handle(
                    {
                        "provider": "runpod",
                        "endpoint_id": "ep",
                        "endpoint_name": "n",
                        "key_fingerprint": _RUNPOD_FINGERPRINT,
                        "job_id": f"j{calls['n']}",
                        "attempt": attempt,
                        "started_ts": 1.0,
                    }
                )
            if calls["n"] == 1:
                return jobs.PollResult(False, failure="stalled", detail="frozen")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr(
            "flash.providers._worker.upload_code",
            lambda repo=None, code_prefix=None: (
                uploaded.setdefault("code_prefix", code_prefix) or "repo"
            ),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        orch.submit_job(_spec("retry-ok"), dry_run=False, background=False)
        st = orch.get_status("retry-ok")
        assert st.state == "done"
        assert calls["n"] == 2
        assert code_prefixes == [uploaded["code_prefix"], uploaded["code_prefix"]]
        assert re.fullmatch(r"code/[0-9a-f]{32}/flash", uploaded["code_prefix"] or "")
        assert st.remote["job_id"] == "j2"  # latest handle persisted


def test_submit_keeps_public_short_init_ref_but_launches_storage_ref(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.lora_rank as rank_mod
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train
        import flash.runner.checkpoints as checkpoints
        from flash.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "grpo",
                "train": {
                    "epochs": 1,
                    "max_examples": 1,
                    "hf_repo": "Freesolo-Co/flashrun-source-env",
                },
            }
        )
        orch._save_status(orch.RunStatus(run_id="source-run", state="done", spec=source.to_dict()))
        monkeypatch.setattr(rank_mod, "resolve_hf_dataset_revision", lambda repo, token: "a" * 40)
        monkeypatch.setattr(
            checkpoints,
            "adapter_artifact_exists",
            lambda spec, *, step, revision=None: step == 40,
        )
        monkeypatch.setattr(
            rank_mod,
            "adapter_artifact_identity",
            lambda *a, **k: rank_mod.AdapterArtifactIdentity(
                "digest", "config", "adapter_model.safetensors", "weights:1"
            ),
        )
        base = _spec("warm-run").to_dict()
        spec = JobSpec.from_dict(
            {
                **base,
                "train": {
                    **base["train"],
                    "init_from_adapter": "source-run/step-40",
                    "lora_rank": 8,
                    "lora_alpha": 16,
                },
            }
        )
        launched: dict[str, object] = {}

        def fake_submit(spec, *_, **__):
            launched["init_from_adapter"] = spec.train.init_from_adapter
            launched["lora_rank"] = spec.train.lora_rank
            launched["lora_alpha"] = spec.train.lora_alpha
            return jobs.PollResult(True, metrics={"cost_usd": 0.1})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr(rank_mod, "load_hf_adapter_config", lambda *a, **k: _adapter_config())
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        orch.submit_job(spec, dry_run=False, background=False)

        st = orch.get_status("warm-run")
        assert st.spec["train"]["init_from_adapter"] == "source-run/step-40"
        assert "lora_rank" not in st.spec["train"]
        # the public spec reports the source adapter's authoritative alpha so status matches
        # what the worker actually trains with
        assert st.spec["train"]["lora_alpha"] == 64
        assert (
            launched["init_from_adapter"]
            == "Freesolo-Co/flashrun-source-env:rl/source-run/checkpoints/step-40"
        )
        assert (launched["lora_rank"], launched["lora_alpha"]) == (32, 64)


def test_submit_rejects_cross_org_init_ref(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        from flash.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        orch._save_status(
            orch.RunStatus(
                run_id="source-run",
                state="done",
                spec=source.to_dict(),
                billing_context={"org_id": "org-a"},
            )
        )
        base = _spec("warm-run").to_dict()
        spec = JobSpec.from_dict(
            {
                **base,
                "train": {**base["train"], "init_from_adapter": "source-run"},
            }
        )

        with pytest.raises(ValueError, match="same Freesolo org"):
            orch.submit_job(
                spec,
                dry_run=True,
                background=False,
                billing_context={"org_id": "org-b"},
            )


def test_submit_allows_missing_source_org_when_same_owner_key(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        from flash.server import db
        from flash.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        orch._save_status(orch.RunStatus(run_id="source-run", state="done", spec=source.to_dict()))
        import flash.lora_rank as rank_mod
        import flash.runner.checkpoints as checkpoints

        monkeypatch.setattr(db, "run_owner", lambda run_id: 7 if run_id == "source-run" else None)
        monkeypatch.setattr(rank_mod, "resolve_hf_dataset_revision", lambda repo, token: "a" * 40)
        monkeypatch.setattr(
            checkpoints,
            "adapter_artifact_exists",
            lambda spec, *, step, revision=None: True,
        )
        monkeypatch.setattr(rank_mod, "load_hf_adapter_config", lambda *a, **k: _adapter_config())
        monkeypatch.setattr(
            rank_mod,
            "adapter_artifact_identity",
            lambda *a, **k: rank_mod.AdapterArtifactIdentity(
                "digest", "config", "adapter_model.safetensors", "weights:1"
            ),
        )
        base = _spec("warm-run").to_dict()
        spec = JobSpec.from_dict(
            {
                **base,
                "train": {**base["train"], "init_from_adapter": "source-run"},
            }
        )

        status = orch.submit_job(
            spec,
            dry_run=True,
            background=False,
            billing_context={"org_id": "org-a"},
            owner_key_id=7,
        )

        assert status.state == "dry_run"


def test_submit_dry_run_omits_public_warmstart_rank_and_resolves_alpha(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        from flash.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        orch._save_status(orch.RunStatus(run_id="source-run", state="done", spec=source.to_dict()))
        import flash.lora_rank as rank_mod
        import flash.runner.checkpoints as checkpoints

        monkeypatch.setattr(
            checkpoints, "adapter_artifact_exists", lambda spec, step, revision=None: True
        )
        monkeypatch.setattr(rank_mod, "resolve_hf_dataset_revision", lambda repo, token: "a" * 40)
        monkeypatch.setattr(
            rank_mod,
            "load_hf_adapter_config",
            lambda *a, **k: _adapter_config(rank=64, alpha=128),
        )
        monkeypatch.setattr(
            rank_mod,
            "adapter_artifact_identity",
            lambda *a, **k: rank_mod.AdapterArtifactIdentity(
                "digest", "config", "adapter_model.safetensors", "weights:1"
            ),
        )
        base = _spec("warm-run").to_dict()
        spec = JobSpec.from_dict(
            {
                **base,
                "train": {
                    **base["train"],
                    "init_from_adapter": "source-run",
                    "lora_rank": 8,
                    "lora_alpha": 16,
                },
            }
        )

        status = orch.submit_job(spec, dry_run=True, background=False)

        assert status.state == "dry_run"
        assert "lora_rank" not in status.spec["train"]
        assert "init_from_adapter_revision" not in status.spec["train"]
        # dry-run displays the source adapter's authoritative alpha, not the submitted value,
        # so preflight and execution report the same effective spec
        assert status.spec["train"]["lora_alpha"] == 128
        assert status.to_dict()["spec"] == status.spec


def test_submit_rejects_bare_init_ref_to_unfinished_source_run(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        from flash.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        orch._save_status(
            orch.RunStatus(run_id="source-run", state="running", spec=source.to_dict())
        )
        base = _spec("warm-run").to_dict()
        spec = JobSpec.from_dict(
            {
                **base,
                "train": {**base["train"], "init_from_adapter": "source-run"},
            }
        )

        with pytest.raises(ValueError, match="concrete source-run/step-N checkpoint"):
            orch.submit_job(spec, dry_run=True, background=False)


def test_submit_rejects_bare_init_ref_without_final_adapter(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.lora_rank as rank_mod
        import flash.runner.checkpoints as checkpoints
        from flash.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        orch._save_status(orch.RunStatus(run_id="source-run", state="done", spec=source.to_dict()))
        monkeypatch.setattr(rank_mod, "resolve_hf_dataset_revision", lambda repo, token: "a" * 40)
        monkeypatch.setattr(
            checkpoints,
            "adapter_artifact_exists",
            lambda spec, *, step, revision=None: False,
        )
        base = _spec("warm-run").to_dict()
        spec = JobSpec.from_dict(
            {
                **base,
                "train": {**base["train"], "init_from_adapter": "source-run"},
            }
        )

        with pytest.raises(ValueError, match="complete adapter artifact was not found"):
            orch.submit_job(spec, dry_run=True, background=False)


def test_submit_rejects_missing_source_org_without_same_owner_key(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        from flash.server import db
        from flash.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        orch._save_status(orch.RunStatus(run_id="source-run", state="done", spec=source.to_dict()))
        monkeypatch.setattr(db, "run_owner", lambda run_id: 8 if run_id == "source-run" else None)
        base = _spec("warm-run").to_dict()
        spec = JobSpec.from_dict(
            {
                **base,
                "train": {**base["train"], "init_from_adapter": "source-run"},
            }
        )

        with pytest.raises(ValueError, match="same Freesolo org"):
            orch.submit_job(
                spec,
                dry_run=True,
                background=False,
                billing_context={"org_id": "org-a"},
                owner_key_id=7,
            )


def test_submit_rejects_missing_init_checkpoint_step(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.lora_rank as rank_mod
        import flash.runner.checkpoints as checkpoints
        from flash.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        orch._save_status(
            orch.RunStatus(
                run_id="source-run",
                state="done",
                spec=source.to_dict(),
                billing_context={"org_id": "org-a"},
            )
        )
        monkeypatch.setattr(rank_mod, "resolve_hf_dataset_revision", lambda repo, token: "a" * 40)
        monkeypatch.setattr(
            checkpoints,
            "adapter_artifact_exists",
            lambda spec, *, step, revision=None: False,
        )
        base = _spec("warm-run").to_dict()
        spec = JobSpec.from_dict(
            {
                **base,
                "train": {**base["train"], "init_from_adapter": "source-run/step-40"},
            }
        )

        with pytest.raises(ValueError, match="complete adapter artifact was not found"):
            orch.submit_job(
                spec,
                dry_run=True,
                background=False,
                billing_context={"org_id": "org-a"},
            )


def test_submit_surfaces_checkpoint_listing_error_before_launch(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.lora_rank as rank_mod
        import flash.runner.checkpoints as checkpoints
        from flash.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-0.8B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        orch._save_status(
            orch.RunStatus(
                run_id="source-run",
                state="done",
                spec=source.to_dict(),
                billing_context={"org_id": "org-a"},
            )
        )

        monkeypatch.setattr(rank_mod, "resolve_hf_dataset_revision", lambda repo, token: "a" * 40)

        def fail_listing(spec, *, step, revision=None):
            raise checkpoints.CheckpointListingError(
                "could not verify adapter artifacts for source-run: 503"
            )

        monkeypatch.setattr(checkpoints, "adapter_artifact_exists", fail_listing)
        base = _spec("warm-run").to_dict()
        spec = JobSpec.from_dict(
            {
                **base,
                "train": {**base["train"], "init_from_adapter": "source-run/step-40"},
            }
        )

        with pytest.raises(ValueError, match="could not verify adapter artifacts"):
            orch.submit_job(
                spec,
                dry_run=True,
                background=False,
                billing_context={"org_id": "org-a"},
            )


def test_attach_polls_live_warmstart_handle_without_source_revalidation(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.lora_rank as rank_mod
        import flash.providers as providers
        from flash.providers.base import PollResult
        from flash.spec import JobSpec

        base = _spec("warm-recover").to_dict()
        public_spec = JobSpec.from_dict(
            {
                **base,
                "train": {**base["train"], "init_from_adapter": "source-run/step-40"},
            }
        )
        worker_dict = public_spec.to_internal_dict()
        worker_dict["train"].update(
            {
                "init_from_adapter": "Freesolo-Co/source:rl/source-run/checkpoints/step-40",
                "init_from_adapter_revision": "a" * 40,
                "lora_rank": 64,
            }
        )
        worker_spec = JobSpec.from_dict(worker_dict)
        identity = rank_mod.AdapterArtifactIdentity(
            "digest-v1", "config-v1", "adapter_model.safetensors", "weights-v1:123"
        )
        orch._save_status(
            orch.RunStatus(
                run_id="warm-recover",
                state="running",
                spec=public_spec.to_dict(),
                remote={
                    "provider": "runpod",
                    "endpoint_id": "ep",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "job",
                    "attempt": 0,
                    "started_ts": 1.0,
                },
                effective_preparation={
                    "worker_spec": worker_spec.to_internal_dict(),
                    "adapter_identity": identity.to_dict(),
                    "preparation_digest": orch._preparation_digest(
                        public_spec, worker_spec, identity.to_dict()
                    ),
                },
            )
        )
        monkeypatch.setattr(
            rank_mod,
            "load_hf_adapter_config",
            lambda *a, **k: pytest.fail("live handle recovery must not reread the source"),
        )
        polled = {}

        class Provider:
            def poll(self, handle, spec, seed, **kwargs):
                polled.update(
                    init_from_adapter=spec.train.init_from_adapter,
                    revision=spec.train.init_from_adapter_revision,
                    lora_rank=spec.train.lora_rank,
                )
                return PollResult(True, metrics={"wall_seconds": 1.0})

        monkeypatch.setattr(providers, "get_provider", lambda name: Provider())
        monkeypatch.setattr(orch, "_gc_run_endpoints", lambda *a, **k: None)

        status = orch.attach_run("warm-recover", log_stream=sys.stderr)

        assert status.state == "done"
        assert polled == {
            "init_from_adapter": "Freesolo-Co/source:rl/source-run/checkpoints/step-40",
            "revision": "a" * 40,
            "lora_rank": 64,
        }


def test_attach_reuses_verified_effective_snapshot_before_recovery_launch(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.lora_rank as rank_mod
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train
        from flash.spec import JobSpec

        base = _spec("warm-recover").to_dict()
        public_spec = JobSpec.from_dict(
            {
                **base,
                "train": {
                    **base["train"],
                    "init_from_adapter": "source-run/step-40",
                    "lora_rank": 8,
                },
            }
        )
        worker_dict = public_spec.to_dict()
        worker_dict["train"] = {
            **worker_dict["train"],
            "init_from_adapter": "Freesolo-Co/source:rl/source-run/checkpoints/step-40",
            "init_from_adapter_revision": "a" * 40,
            "lora_rank": 64,
        }
        worker_spec = JobSpec.from_dict(worker_dict)
        identity = rank_mod.AdapterArtifactIdentity(
            "digest-v1", "config-v1", "adapter_model.safetensors", "weights-v1:123"
        )
        orch._save_status(
            orch.RunStatus(
                run_id="warm-recover",
                state="running",
                spec=public_spec.to_dict(),
                billing_context={"org_id": "org-a"},
                remote={
                    "provider": "runpod",
                    "endpoint_id": "ep",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "job",
                    "attempt": 0,
                    "started_ts": 1.0,
                },
                effective_preparation={
                    "worker_spec": worker_spec.to_internal_dict(),
                    "adapter_identity": identity.to_dict(),
                    "preparation_digest": orch._preparation_digest(
                        public_spec, worker_spec, identity.to_dict()
                    ),
                },
            )
        )
        monkeypatch.setattr(
            rank_mod, "load_hf_adapter_config", lambda *a, **k: _adapter_config(rank=64)
        )
        monkeypatch.setattr(rank_mod, "adapter_artifact_identity", lambda *a, **k: identity)
        launched: dict[str, object] = {}
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="stalled", detail="stalled"),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda *a, **k: "repo")
        monkeypatch.setattr(orch, "_gc_run_endpoints", lambda *a, **k: None)
        monkeypatch.setattr(
            orch,
            "_run_training",
            lambda spec, *a, **k: launched.update(
                init_from_adapter=spec.train.init_from_adapter,
                init_from_adapter_revision=spec.train.init_from_adapter_revision,
                lora_rank=spec.train.lora_rank,
            ),
        )

        orch.attach_run("warm-recover", log_stream=sys.stderr)

        assert launched == {
            "init_from_adapter": "Freesolo-Co/source:rl/source-run/checkpoints/step-40",
            "init_from_adapter_revision": "a" * 40,
            "lora_rank": 64,
        }


def test_attach_revalidates_source_before_handleless_resubmission(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.lora_rank as rank_mod
        import flash.providers.runpod.jobs as jobs
        from flash.spec import JobSpec

        base = _spec("warm-recover").to_dict()
        public_spec = JobSpec.from_dict(
            {
                **base,
                "train": {**base["train"], "init_from_adapter": "source-run/step-40"},
            }
        )
        worker_dict = public_spec.to_dict()
        worker_dict["train"] = {
            **worker_dict["train"],
            "init_from_adapter": "private-owner/private-repo:rl/source-run/checkpoints/step-40",
            "init_from_adapter_revision": "a" * 40,
        }
        worker_spec = JobSpec.from_dict(worker_dict)
        original_identity = {
            "digest": "original",
            "config_sha256": "config-v1",
            "weight_filename": "adapter_model.safetensors",
            "weight_identity": "weights-v1:123",
        }
        orch._save_status(
            orch.RunStatus(
                run_id="warm-recover",
                state="running",
                spec=public_spec.to_dict(),
                remote={
                    "provider": "runpod",
                    "endpoint_id": "ep",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "job",
                    "attempt": 0,
                    "started_ts": 1.0,
                },
                effective_preparation={
                    "worker_spec": worker_dict,
                    "adapter_identity": original_identity,
                    "preparation_digest": orch._preparation_digest(
                        public_spec, worker_spec, original_identity
                    ),
                },
            )
        )
        monkeypatch.setattr(rank_mod, "load_hf_adapter_config", lambda *a, **k: _adapter_config())
        monkeypatch.setattr(
            rank_mod,
            "adapter_artifact_identity",
            lambda *a, **k: rank_mod.AdapterArtifactIdentity(
                "changed", "config-v2", "adapter_model.safetensors", "weights-v2:123"
            ),
        )
        polls = []
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: (
                polls.append("polled")
                or jobs.PollResult(False, failure="stalled", detail="stalled")
            ),
        )
        monkeypatch.setattr(orch, "_gc_run_endpoints", lambda *a, **k: None)
        monkeypatch.setattr(
            orch,
            "_run_training",
            lambda *a, **k: pytest.fail("drifted source must block resubmission"),
        )

        status = orch.attach_run("warm-recover", log_stream=sys.stderr)

        assert polls == ["polled"]
        assert status.state == "failed"
        assert "source-run/step-40" in (status.error or "")
        assert "changed after submission" in (status.error or "")
        assert "private-owner" not in (status.error or "")


def test_attach_legacy_warmstart_without_snapshot_fails_closed(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.jobs as jobs
        from flash.spec import JobSpec

        base = _spec("warm-recover").to_dict()
        spec = JobSpec.from_dict(
            {
                **base,
                "train": {**base["train"], "init_from_adapter": "source-run/step-40"},
            }
        )
        remote = {
            "provider": "runpod",
            "endpoint_id": "ep",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "job",
            "attempt": 0,
            "started_ts": 1.0,
        }
        orch._save_status(
            orch.RunStatus(
                run_id="warm-recover",
                state="running",
                spec=spec.to_dict(),
                remote=remote,
            )
        )
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: pytest.fail("legacy warm start must fail before provider polling"),
        )
        monkeypatch.setattr(orch, "_gc_run_endpoints", lambda *a, **k: None)

        status = orch.attach_run("warm-recover", log_stream=sys.stderr)

        assert status.state == "failed"
        assert status.remote is None
        assert "original preparation snapshot is unavailable" in (status.error or "")
        assert "source-run/step-40" in (status.error or "")


def test_attach_setup_failure_does_not_overwrite_concurrent_cancel(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        remote = {
            "provider": "runpod",
            "endpoint_id": "ep",
            "endpoint_name": "name",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "job",
            "attempt": 0,
            "started_ts": 1.0,
        }
        orch._save_status(
            orch.RunStatus(
                run_id="cancelled-attach",
                state="running",
                spec=_spec("cancelled-attach").to_dict(),
                remote=remote,
            )
        )
        monkeypatch.setattr(
            orch,
            "effective_spec_from_status",
            lambda _status: (_ for _ in ()).throw(RuntimeError("setup failed")),
        )
        real_record = orch._record_cleanup_remote

        def cancel_then_record(run_id, cleanup_remote):
            assert orch._update(run_id, "cancelled", remote=None)
            return real_record(run_id, cleanup_remote)

        monkeypatch.setattr(orch, "_record_cleanup_remote", cancel_then_record)
        monkeypatch.setattr(orch, "_gc_run_endpoints", lambda *_args, **_kwargs: None)

        status = orch.attach_run("cancelled-attach", log_stream=sys.stderr)

        assert status.state == "cancelled"
        assert status.remote is None
        assert orch._load_status_json("cancelled-attach")[orch._CLEANUP_REMOTES_KEY] == [remote]


def test_attach_setup_failure_does_not_steal_precommit_cancel(monkeypatch):
    # a cancel that has cleared status.remote but not yet flipped state to terminal must win: the
    # attach-setup-failure cas no-ops (the identifiable remote no longer matches) and must NOT
    # force-fail the run, else a user cancel gets recorded as failed.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        remote = {
            "provider": "runpod",
            "endpoint_id": "ep",
            "endpoint_name": "name",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "job",
            "attempt": 0,
            "started_ts": 1.0,
        }
        orch._save_status(
            orch.RunStatus(
                run_id="precancel-attach",
                state="running",
                spec=_spec("precancel-attach").to_dict(),
                remote=remote,
            )
        )
        monkeypatch.setattr(
            orch,
            "effective_spec_from_status",
            lambda _status: (_ for _ in ()).throw(RuntimeError("setup failed")),
        )
        real_record = orch._record_cleanup_remote

        def clear_remote_then_record(run_id, cleanup_remote):
            # a concurrent cancel cleared the remote but has NOT yet flipped state to cancelled
            assert orch._compare_and_clear_remote(run_id, remote)
            return real_record(run_id, cleanup_remote)

        monkeypatch.setattr(orch, "_record_cleanup_remote", clear_remote_then_record)
        monkeypatch.setattr(orch, "_gc_run_endpoints", lambda *_args, **_kwargs: None)

        status = orch.attach_run("precancel-attach", log_stream=sys.stderr)

        assert status.state != "failed", "attach-setup-failure stole a pre-commit cancel"


def test_cancel_during_attempt_reaps_walked_endpoint(monkeypatch):
    """A cancel landing mid-attempt raised _RunCancelled straight out of the retry loop, skipping
    _gc_seen_endpoints — leaking a walk-provisioned endpoint (one _gc_run_endpoints can't name, whose
    `running` write lost the terminal-stickiness race so it's absent from status.remote). The cancel
    path must now reap seen_endpoints."""
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train
        from flash.providers.runpod import api as runpod_api

        deleted: list[str] = []
        monkeypatch.setattr(
            runpod_api,
            "cancel_job",
            lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
        )
        monkeypatch.setattr(
            runpod_api,
            "delete_endpoint_for_fingerprint",
            lambda eid, _fingerprint, **_kw: deleted.append(eid) or True,
        )

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, **_):
            orch._update(spec.run_id, "cancelled")  # cancel lands during provisioning
            if on_handle:  # endpoint comes up anyway; its "running" write is rejected (terminal)
                on_handle(
                    {
                        "provider": "runpod",
                        "endpoint_id": "epWALK",
                        "endpoint_name": "n",
                        "key_fingerprint": _RUNPOD_FINGERPRINT,
                        "job_id": "jW",
                        "attempt": attempt,
                        "started_ts": 1.0,
                    }
                )
            return jobs.PollResult(True, metrics={"cost_usd": 0.1})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        orch.submit_job(_spec("cancel-reap"), dry_run=False, background=False)

        assert orch.get_status("cancel-reap").state == "cancelled"
        assert (
            orch.get_status("cancel-reap").remote is None
        )  # handle write lost the stickiness race
        assert "epWALK" in deleted  # the walked endpoint was reaped on the cancel path


def test_supervisor_retries_runpod_cancelled_then_succeeds(monkeypatch):
    # A "job_preempted" first attempt retries on a fresh endpoint and completes.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        calls = {"n": 0}

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, **_):
            calls["n"] += 1
            if on_handle:
                on_handle(
                    {
                        "provider": "runpod",
                        "endpoint_id": "ep",
                        "endpoint_name": "n",
                        "key_fingerprint": _RUNPOD_FINGERPRINT,
                        "job_id": f"j{calls['n']}",
                        "attempt": attempt,
                        "started_ts": 1.0,
                    }
                )
            if calls["n"] == 1:
                return jobs.PollResult(False, failure="job_preempted", detail="[CANCELLED] None")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        orch.submit_job(_spec("cancel-retry"), dry_run=False, background=False)
        assert orch.get_status("cancel-retry").state == "done"
        assert calls["n"] == 2


def test_supervisor_does_not_retry_worker_code_errors(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        calls = {"n": 0}

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, **_):
            calls["n"] += 1
            return jobs.PollResult(
                False, failure="job_failed", detail="Remote execution failed: ValueError"
            )

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        with pytest.raises(RuntimeError):
            orch.submit_job(_spec("fail-fast"), dry_run=False, background=False)
        assert calls["n"] == 1
        assert orch.get_status("fail-fast").state == "failed"


def test_supervisor_infra_failure_retries_up_to_floor(monkeypatch):
    """A STREAK of infra-shaped failures (broken/busy GPU -> stalled/job_preempted) walks past up to
    INFRA_RETRY_FLOOR hosts even though the spec's max_retries is only 2 — so a run of bad GPUs finds a
    healthy host instead of dying on the small default budget. (Genuine worker errors still fail fast:
    test_supervisor_does_not_retry_worker_code_errors.)"""
    from flash.runner.lifecycle import INFRA_RETRY_FLOOR

    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        calls = {"n": 0}

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, **_):
            calls["n"] += 1
            return jobs.PollResult(False, failure="stalled", detail="GPU never became ready")

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        with pytest.raises(RuntimeError):
            orch.submit_job(_spec("infra-floor"), dry_run=False, background=False)  # max_retries=2
        # floor=5 -> 6 attempts (walk 0..5), NOT the 3 the raw max_retries=2 would give.
        assert calls["n"] == INFRA_RETRY_FLOOR + 1
        assert orch.get_status("infra-floor").state == "failed"


def test_supervisor_infra_floor_respects_explicit_zero_retries(monkeypatch):
    """An explicit max_retries=0 (deliberate single-shot) is NOT forced to retry by the infra floor."""
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        calls = {"n": 0}

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, **_):
            calls["n"] += 1
            return jobs.PollResult(False, failure="stalled", detail="frozen")

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        spec = JobSpec(
            run_id="no-retry",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="RTX 4090", max_retries=0),
        )
        with pytest.raises(RuntimeError):
            orch.submit_job(spec, dry_run=False, background=False)
        assert calls["n"] == 1  # single shot — floor does not apply at max_retries=0
        assert orch.get_status("no-retry").state == "failed"


@pytest.mark.parametrize("failure", ["no_capacity", "poll_error"])
def test_shared_cache_zero_retry_budget_submits_exactly_once(monkeypatch, failure):
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        submissions = []

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, on_last_gpu=False, **_):
            submissions.append((attempt, spec.gpu.network_volume, on_last_gpu))
            return jobs.PollResult(False, failure=failure, detail="cache-constrained failure")

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        spec = JobSpec(
            run_id=f"cache-zero-{failure}",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="RTX 4090", max_retries=0),
        )

        with pytest.raises(RuntimeError):
            orch.submit_job(spec, dry_run=False, background=False)

        assert submissions == [(0, orch.WEIGHT_CACHE_VOLUME_NAME, True)]


def test_supervisor_walks_to_next_gpu_class_on_infra_retry(monkeypatch):
    # a managed gpu request that keeps hitting infra-shaped failures must walk
    # down the ranked candidate list, not burn every retry on the same capacity-starved
    # class. with static rates the validated >=24 gb pool for a 0.8b grpo run ranks
    # rtx 4090 < rtx 5090 < a100 pcie < ... by $/hr, so successive attempts step through them.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        gpus_seen: list[str] = []

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, **_):
            gpus_seen.append(spec.gpu.type)
            if on_handle:
                on_handle(
                    {
                        "provider": "runpod",
                        "endpoint_id": "ep",
                        "endpoint_name": "n",
                        "key_fingerprint": _RUNPOD_FINGERPRINT,
                        "job_id": f"j{attempt}",
                        "attempt": attempt,
                        "started_ts": 1.0,
                    }
                )
            if attempt < 2:
                return jobs.PollResult(False, failure="stalled", detail="frozen")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="walk",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="RTX 4090", max_retries=2),
        )
        orch.submit_job(spec, dry_run=False, background=False)

        assert orch.get_status("walk").state == "done"
        # Three attempts, three distinct classes, each at least as expensive as the last.
        assert len(gpus_seen) == 3
        assert len(set(gpus_seen)) == 3
        from flash.providers.runpod.pricing import hourly_rate

        rates = [hourly_rate(g) for g in gpus_seen]
        assert rates == sorted(rates)
        # cheapest validated class with >= 24 GB
        assert gpus_seen[0] == "RTX 4090"


def test_supervisor_oom_walks_only_to_strictly_larger_gpu(monkeypatch):
    """An OOM retry must not re-roll the same VRAM class; it walks to a strictly larger card."""
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.allocator as allocator
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train
        from flash.providers.base import Allocation, Candidate
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        candidates = (
            Candidate("runpod", "A100 SXM 40GB", 1.00, 40),
            Candidate("runpod", "A100 SXM 40GB", 1.01, 40),
            Candidate("runpod", "A100 PCIe", 1.39, 80),
        )

        monkeypatch.setattr(
            allocator,
            "allocate",
            lambda *a, **k: Allocation(
                provider="runpod",
                gpu="A100 SXM 40GB",
                hourly_usd=1.00,
                min_vram_gb=40,
                candidates=candidates,
            ),
        )
        gpus_seen: list[str] = []

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, **_):
            gpus_seen.append(spec.gpu.type)
            if on_handle:
                on_handle(
                    {
                        "provider": "runpod",
                        "endpoint_id": "ep",
                        "endpoint_name": "n",
                        "key_fingerprint": _RUNPOD_FINGERPRINT,
                        "job_id": f"j{attempt}",
                        "attempt": attempt,
                        "started_ts": 1.0,
                    }
                )
            if attempt == 0:
                return jobs.PollResult(False, failure="oom", detail="vLLM free-memory preflight")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        import types

        import huggingface_hub

        class FakePrivateHf:
            def repo_info(self, **_kwargs):
                return types.SimpleNamespace(sha="private-pinned-sha")

            def get_paths_info(self, **_kwargs):
                return []

        fake_private_hf = FakePrivateHf()
        monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: fake_private_hf)
        monkeypatch.setattr(
            huggingface_hub,
            "hf_hub_download",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError("no marker is present")),
        )

        spec = JobSpec(
            run_id="oom-walk",
            model="Qwen/Qwen3.5-4B",
            algorithm="opd",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="RTX 4090", max_retries=2),
        )
        orch.submit_job(spec, dry_run=False, background=False)

        assert orch.get_status("oom-walk").state == "done"
        assert gpus_seen == ["A100 SXM 40GB", "A100 PCIe"]


def test_supervisor_job_failed_without_marker_does_not_retry(monkeypatch):
    # A plain job_failed (no retriable flag — a genuine code crash) is NOT retried: the retry
    # budget exists only for infra-shaped failures, not code bugs.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        calls = {"n": 0}

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, **_):
            calls["n"] += 1
            return jobs.PollResult(
                False, failure="job_failed", detail="ValueError: bad reward fn (no infra marker)"
            )

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="code-crash",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="RTX 4090", max_retries=2),
        )
        with pytest.raises(RuntimeError, match="bad reward fn"):
            orch.submit_job(spec, dry_run=False, background=False)
        assert calls["n"] == 1  # genuine code error burns no retry budget
        assert orch.get_status("code-crash").state == "failed"


def test_supervisor_gpu_walk_exhausts_classes_then_retries_cheapest(monkeypatch):
    # The candidate walk steps to the next class on each infra retry, then — once every distinct
    # class on the (single) provider has been tried — falls back to the CHEAPEST one rather than
    # clamping on the priciest (no point re-rolling the most expensive card when re-trying an
    # already-tried option). It must never index past the ranked list. Force a VRAM need that only
    # the 80 GB+ VALIDATED tier satisfies, then trim the ranked candidate list to exactly the two
    # cheapest 80 GB classes (A100 PCIe, A100 SXM): attempt 0 takes the cheaper (A100 PCIe @ $1.39),
    # attempt 1 walks to the pricier (A100 SXM @ $1.49), attempt 2 (both now tried) re-rolls the
    # cheapest (A100 PCIe).
    import dataclasses

    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.allocator as allocator
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        # Need 80 GB; the validated 80 GB+ RunPod pool is A100 PCIe ($1.39), A100 SXM ($1.49), RTX
        # Pro 6000 Server ($2.09), H100 ($3.29). Trim the ranked candidates to the two cheapest so
        # exactly TWO candidates remain for a clean walk+clamp assertion.
        monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 80)
        real_allocate = allocator.allocate

        def two_candidate_allocate(*a, **k):
            alloc = real_allocate(*a, **k)
            keep = tuple(c for c in alloc.candidates if c.gpu in ("A100 PCIe", "A100 SXM"))
            best = keep[0]
            return dataclasses.replace(
                alloc, gpu=best.gpu, hourly_usd=best.hourly_usd, candidates=keep
            )

        monkeypatch.setattr(allocator, "allocate", two_candidate_allocate)
        gpus_seen: list[str] = []

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, **_):
            gpus_seen.append(spec.gpu.type)
            if on_handle:
                on_handle(
                    {
                        "provider": "runpod",
                        "endpoint_id": "ep",
                        "endpoint_name": "n",
                        "key_fingerprint": _RUNPOD_FINGERPRINT,
                        "job_id": f"j{attempt}",
                        "attempt": attempt,
                        "started_ts": 1.0,
                    }
                )
            if attempt < 2:
                return jobs.PollResult(False, failure="stalled", detail="frozen")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="clamp",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="RTX 4090", max_retries=2),
        )
        orch.submit_job(spec, dry_run=False, background=False)

        assert orch.get_status("clamp").state == "done"
        # Walk advances through both classes, then re-rolls the cheapest (never out of range).
        assert gpus_seen == ["A100 PCIe", "A100 SXM", "A100 PCIe"]


def test_supervisor_marks_on_last_gpu_only_at_end_of_walk(monkeypatch):
    # on_last_gpu must reach the provider so the no-capacity backstops know whether there is a
    # next-best class to fall to: False while the walk still has somewhere to go (attempt 0 on the
    # cheaper of two classes), True once it lands on (and clamps to) the last candidate.
    import dataclasses

    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.allocator as allocator
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        # Same trim as the clamp test: exactly two 80 GB candidates so the walk has one step.
        monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 80)
        real_allocate = allocator.allocate

        def two_candidate_allocate(*a, **k):
            alloc = real_allocate(*a, **k)
            keep = tuple(c for c in alloc.candidates if c.gpu in ("A100 PCIe", "A100 SXM"))
            best = keep[0]
            return dataclasses.replace(
                alloc, gpu=best.gpu, hourly_usd=best.hourly_usd, candidates=keep
            )

        monkeypatch.setattr(allocator, "allocate", two_candidate_allocate)
        last_flags: list[bool] = []

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, on_last_gpu=False, **_):
            last_flags.append(on_last_gpu)
            if on_handle:
                on_handle(
                    {
                        "provider": "runpod",
                        "endpoint_id": "ep",
                        "endpoint_name": "n",
                        "key_fingerprint": _RUNPOD_FINGERPRINT,
                        "job_id": f"j{attempt}",
                        "attempt": attempt,
                        "started_ts": 1.0,
                    }
                )
            if attempt < 2:
                return jobs.PollResult(False, failure="stalled", detail="frozen")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="lastgpu",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="RTX 4090", max_retries=2),
        )
        orch.submit_job(spec, dry_run=False, background=False)

        assert orch.get_status("lastgpu").state == "done"
        # attempt 0: cheaper class, a next-best still exists -> False; attempts 1 & 2: on the last
        # candidate (and clamped onto it) -> True.
        assert last_flags == [False, True, True]
        # The winning (last) attempt persisted on_last_gpu into the handle so a reattach reproduces
        # its stall tuning (see test_reattach_poll_reproduces_persisted_on_last_gpu).
        assert orch.get_status("lastgpu").remote.get("on_last_gpu") is True


def test_supervisor_allocation_failure_does_not_skip_cheapest(monkeypatch):
    # An allocation/pricing failure must NOT advance the candidate walk: that attempt never
    # provisioned a class, so the retry has to start over from the cheapest, not a pricier
    # one. (Regression guard for the walk-offset-vs-attempt-counter bug.)
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.allocator as allocator
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        real_allocate = allocator.allocate
        alloc_calls = {"n": 0}

        def flaky_allocate(*a, **k):
            alloc_calls["n"] += 1
            if alloc_calls["n"] == 1:
                raise RuntimeError("pricing API blip")  # not Unsupported -> infra-shaped retry
            return real_allocate(*a, **k)

        monkeypatch.setattr(allocator, "allocate", flaky_allocate)

        gpus_seen: list[str] = []

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, **_):
            gpus_seen.append(spec.gpu.type)
            if on_handle:
                on_handle(
                    {
                        "provider": "runpod",
                        "endpoint_id": "ep",
                        "endpoint_name": "n",
                        "key_fingerprint": _RUNPOD_FINGERPRINT,
                        "job_id": f"j{attempt}",
                        "attempt": attempt,
                        "started_ts": 1.0,
                    }
                )
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr("flash.providers._worker.upload_code", lambda repo=None, **_: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="alloc-blip",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="RTX 4090", max_retries=2),
        )
        orch.submit_job(spec, dry_run=False, background=False)

        assert orch.get_status("alloc-blip").state == "done"
        # First allocation failed (no provision); the retry provisioned the cheapest class
        # (RTX 4090, the cheapest validated RunPod class that fits 24 GB).
        assert gpus_seen == ["RTX 4090"]


def test_attach_costs_recovered_run_with_walked_gpu(monkeypatch):
    # A policy run that walked to a pricier class persists that class in the handle, so a
    # recovery via attach_run costs the card it actually ran on, not the provisional one.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        status = orch.RunStatus(
            run_id="walked",
            state="running",
            spec=_spec("walked").to_dict(),  # provisional spec.gpu.type == "RTX 4090"
            remote={
                "provider": "runpod",
                "endpoint_id": "epW",
                "endpoint_name": "n",
                "key_fingerprint": _RUNPOD_FINGERPRINT,
                "job_id": "jW",
                "allocated_gpu": "RTX 5090",
                "on_last_gpu": True,
                "attempt": 2,
                "started_ts": 1.0,
            },
        )
        orch._save_status(status)
        # Worker output carries wall time but neither cost nor allocated_gpu (the in-process
        # success path that stamps allocated_gpu is bypassed on recovery).
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(True, metrics={"wall_seconds": 3600.0}),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        st = orch.attach_run("walked", log_stream=sys.stderr)

        assert st.state == "done"
        import json
        import os

        from flash.providers.runpod.pricing import hourly_rate

        # cost_usd is now the QUOTE (flash.cost estimate we charge); the MEASURED cost in metrics.json
        # is what proves recovery costed the walked 5090, not the provisional 4090.
        with open(os.path.join(orch.artifacts_dir(_spec("walked")), "metrics.json")) as f:
            measured = json.load(f)["cost_usd"]
        assert abs(measured - hourly_rate("RTX 5090")) < 1e-6  # ~1 GPU-hour on the 5090
        assert measured > hourly_rate("RTX 4090")
        assert st.cost_usd == orch.charge_usd_for_spec(_spec("walked"))


# ---------------------------------------------------------------------------
# Cross-process cancel via REST handle + attach
# ---------------------------------------------------------------------------
def test_cancel_prices_and_cleans_up_with_effective_warmstart_spec(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        from flash.spec import JobSpec

        base = _spec("warm-cancel").to_dict()
        public_spec = JobSpec.from_dict(
            {
                **base,
                "train": {
                    **base["train"],
                    "init_from_adapter": "source-run",
                    "lora_rank": 8,
                },
            }
        )
        worker_dict = public_spec.to_dict()
        worker_dict["train"] = {
            **worker_dict["train"],
            "init_from_adapter": "private-owner/private-repo:rl/source-run",
            "init_from_adapter_revision": "a" * 40,
            "lora_rank": 64,
        }
        worker_spec = JobSpec.from_dict(worker_dict)
        identity = {"digest": "immutable-v1"}
        orch._save_status(
            orch.RunStatus(
                run_id="warm-cancel",
                state="running",
                spec=public_spec.to_dict(),
                billing_context={"org_id": "org-a"},
                last_heartbeat={"stage": "rl_step", "step": 2},
                effective_preparation={
                    "worker_spec": worker_dict,
                    "adapter_identity": identity,
                    "preparation_digest": orch._preparation_digest(
                        public_spec, worker_spec, identity
                    ),
                },
            )
        )
        priced = []
        cleaned = []

        def fake_charge(spec, *, steps=None, fallback=0.0):
            priced.append((spec.train.lora_rank, spec.train.init_from_adapter, steps))
            return 3.25

        monkeypatch.setattr(orch, "charge_usd_for_spec", fake_charge)
        monkeypatch.setattr(orch, "_gc_run_endpoints", lambda spec: cleaned.append(spec))

        status = orch.cancel_run("warm-cancel")

        assert status.state == "cancelled"
        assert status.cost_usd == 3.25
        assert priced == [(64, "private-owner/private-repo:rl/source-run", 2)]
        assert cleaned[0].train.lora_rank == 64


@pytest.mark.parametrize("snapshot", [None, {"worker_spec": "malformed"}])
def test_cancel_with_invalid_preparation_uses_zero_failed_billing(monkeypatch, snapshot):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        from flash.spec import JobSpec

        base = _spec("warm-cancel-bad-snapshot").to_dict()
        public_spec = JobSpec.from_dict(
            {
                **base,
                "train": {
                    **base["train"],
                    "init_from_adapter": "source-run",
                    "lora_rank": 8,
                },
            }
        )
        orch._save_status(
            orch.RunStatus(
                run_id=public_spec.run_id,
                state="running",
                spec=public_spec.to_dict(),
                billing_context={"org_id": "org-a"},
                remote={
                    "provider": "runpod",
                    "endpoint_id": "ep-billing",
                    "endpoint_name": "flash-billing",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "job-1",
                    "attempt": 0,
                    "started_ts": 1.0,
                },
                deployment={"state": "ready"},
                effective_preparation=snapshot,
            )
        )
        calls = []

        class Provider:
            def cancel(self, handle):
                calls.append("cancel")

            def destroy(self, handle):
                calls.append("destroy")

        import flash.providers
        import flash.serve.deploy

        monkeypatch.setattr(flash.providers, "get_provider", lambda name: Provider())
        monkeypatch.setattr(
            flash.serve.deploy, "undeploy_adapter", lambda run_id: calls.append("undeploy")
        )
        monkeypatch.setattr(
            orch, "_gc_run_endpoints", lambda spec: calls.append(("gc", spec.train.lora_rank))
        )
        monkeypatch.setattr(
            orch,
            "charge_usd_for_spec",
            lambda *a, **k: pytest.fail("public child rank must not be priced"),
        )

        status = orch.cancel_run(public_spec.run_id)

        assert status.state == "cancelled"
        assert status.cost_usd == 0.0
        assert status.billing_state == "failed"
        assert "private preparation snapshot" in (status.billing_error or "")
        assert "cancel" in calls
        assert "destroy" in calls
        assert "undeploy" in calls
        assert ("gc", 32) in calls


def test_cancel_uses_rest_handle(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        from flash.providers.runpod import api as runpod_api

        status = orch.RunStatus(
            run_id="c1",
            state="running",
            spec=_spec("c1").to_dict(),
            remote={
                "provider": "runpod",
                "endpoint_id": "epX",
                "endpoint_name": "n",
                "key_fingerprint": _RUNPOD_FINGERPRINT,
                "job_id": "jX",
                "attempt": 0,
                "started_ts": 1.0,
            },
        )
        orch._save_status(status)
        cancelled, deleted = [], []
        monkeypatch.setattr(
            runpod_api,
            "cancel_job",
            lambda e, j, **_kw: cancelled.append((e, j)) or {"id": j, "status": "CANCELLED"},
        )
        monkeypatch.setattr(
            runpod_api,
            "delete_endpoint_for_fingerprint",
            lambda e, _fingerprint: deleted.append(e) or True,
        )
        import flash.providers.runpod.train as flash_train

        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        st = orch.cancel_run("c1")
        assert st.state == "cancelled"
        assert cancelled
        assert all(item == ("epX", "jX") for item in cancelled)
        # cancel_run now also destroys the handle's endpoint (idempotent); the GC backstop may
        # delete it again — endpoint id was torn down, which is what matters.
        assert deleted
        assert all(e == "epX" for e in deleted)


def test_attach_completes_run(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        status = orch.RunStatus(
            run_id="a1",
            state="running",
            spec=_spec("a1").to_dict(),
            remote={
                "provider": "runpod",
                "endpoint_id": "epA",
                "endpoint_name": "n",
                "key_fingerprint": _RUNPOD_FINGERPRINT,
                "job_id": "jA",
                "on_last_gpu": False,
                "attempt": 0,
                "started_ts": 1.0,
            },
        )
        orch._save_status(status)
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(True, metrics={"cost_usd": 0.2}),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        st = orch.attach_run("a1", log_stream=sys.stderr)
        assert st.state == "done"
        # We charge the QUOTE (flash.cost estimate); the measured 0.2 is recorded in metrics.json.
        import json
        import os

        assert st.cost_usd == orch.charge_usd_for_spec(_spec("a1"))
        with open(os.path.join(orch.artifacts_dir(_spec("a1")), "metrics.json")) as f:
            assert json.load(f)["cost_usd"] == 0.2


def test_attach_cleanup_survives_unreadable_final_status(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs

        run_id = "attach-finally-status"
        orch._save_status(
            orch.RunStatus(
                run_id=run_id,
                state="running",
                spec=_spec(run_id).to_dict(),
                remote={
                    "provider": "runpod",
                    "endpoint_id": "ep-finally",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "j-finally",
                    "attempt": 0,
                    "started_ts": 1.0,
                },
            )
        )
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(True, metrics={"cost_usd": 0.2}),
        )

        real_get_status = orch.get_status
        real_update = orch._update
        fail_next_status_read = {"armed": False}

        def update_then_arm_status_failure(run_id, state, **updates):
            applied = real_update(run_id, state, **updates)
            if state == "done":
                fail_next_status_read["armed"] = True
            return applied

        def transiently_unreadable_status(run_id):
            if fail_next_status_read["armed"]:
                fail_next_status_read["armed"] = False
                raise PermissionError("status file is transiently unreadable")
            return real_get_status(run_id)

        gc_calls = []
        monkeypatch.setattr(orch, "_update", update_then_arm_status_failure)
        monkeypatch.setattr(orch, "get_status", transiently_unreadable_status)
        monkeypatch.setattr(orch, "_gc_run_endpoints", lambda spec: gc_calls.append(spec.run_id))

        status = orch.attach_run(run_id, log_stream=sys.stderr)

        assert status.state == "done", "the final status read must not mask the completed outcome"
        assert gc_calls == [run_id], "terminal endpoint cleanup must still run"


def test_attach_confirmed_cancel_survives_unreadable_cleanup_status(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs

        run_id = "attach-confirmed-cancel"
        orch._save_status(
            orch.RunStatus(
                run_id=run_id,
                state="running",
                spec=_spec(run_id).to_dict(),
                remote={
                    "provider": "runpod",
                    "endpoint_id": "ep-cancel",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "j-cancel",
                    "attempt": 0,
                    "started_ts": 1.0,
                },
            )
        )
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(True, metrics={"cost_usd": 0.2}),
        )

        def cancel_during_metrics(spec, metrics):
            orch._update(run_id, "cancelled")
            return 0.2

        real_get_status = orch.get_status
        status_failures = {"remaining": 0, "cancelled_reads": 0}

        def unreadable_after_confirmed_cancel(run_id):
            if status_failures["remaining"]:
                status_failures["remaining"] -= 1
                raise PermissionError("status file is transiently unreadable")
            current = real_get_status(run_id)
            if current.state == "cancelled":
                status_failures["cancelled_reads"] += 1
                if status_failures["cancelled_reads"] == 2:
                    status_failures["remaining"] = 2
            return current

        gc_calls = []
        monkeypatch.setattr(orch, "_persist_metrics", cancel_during_metrics)
        monkeypatch.setattr(orch, "get_status", unreadable_after_confirmed_cancel)
        monkeypatch.setattr(orch, "_gc_run_endpoints", lambda spec: gc_calls.append(spec.run_id))

        status = orch.attach_run(run_id, log_stream=sys.stderr)

        assert status.state == "cancelled"
        assert gc_calls == [run_id], "the positively observed terminal cancel must still be reaped"


def test_attach_duplicate_supervisor_unreadable_status_preserves_live_owner(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.jobs as jobs

        run_id = "attach-live-owner"
        stale_remote = {
            "provider": "runpod",
            "endpoint_id": "ep-stale",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "j-stale",
            "attempt": 0,
            "started_ts": 1.0,
        }
        live_remote = {
            "provider": "runpod",
            "endpoint_id": "ep-live",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "j-live",
            "attempt": 1,
            "started_ts": 2.0,
        }
        orch._save_status(
            orch.RunStatus(
                run_id=run_id,
                state="running",
                spec=_spec(run_id).to_dict(),
                remote=stale_remote,
            )
        )
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="stalled", detail="redeploy"),
        )
        monkeypatch.setattr(
            "flash.providers._worker.upload_code", lambda repo, *, code_prefix: repo
        )

        real_get_status = orch.get_status
        status_failures = {"remaining": 0}

        def transiently_unreadable_status(run_id):
            if status_failures["remaining"]:
                status_failures["remaining"] -= 1
                raise PermissionError("status file is transiently unreadable")
            return real_get_status(run_id)

        def duplicate_supervisor_refusal(*args, **kwargs):
            orch._update(run_id, "running", remote=live_remote)
            status_failures["remaining"] = 2
            raise orch._RunCancelled("another supervisor owns the durable provider handle")

        gc_calls = []
        monkeypatch.setattr(orch, "get_status", transiently_unreadable_status)
        monkeypatch.setattr(orch, "_run_training", duplicate_supervisor_refusal)
        monkeypatch.setattr(orch, "_gc_run_endpoints", lambda spec: gc_calls.append(spec.run_id))

        status = orch.attach_run(run_id, log_stream=sys.stderr)

        assert status.state == "running"
        assert status.remote == live_remote
        assert gc_calls == [], (
            "captured-handle recovery must not run run-wide gc after newer ownership appears"
        )


def test_attach_requires_handle(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        orch._save_status(orch.RunStatus(run_id="nh", state="running", spec=_spec("nh").to_dict()))
        with pytest.raises(ValueError, match="no persisted job handle"):
            orch.attach_run("nh")


def test_attach_resumes_from_checkpoint_on_poll_failure(monkeypatch):
    # A recovered run whose remote job ended not-ok (it died while the control plane was down for
    # the redeploy) must NOT be failed — reattach resumes training on a fresh host (worker resumes
    # from the latest HF checkpoint), exactly like the fresh-submit retry loop. It also clears the
    # stale handle so a second restart during the fresh allocation re-resumes cleanly.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        orch._save_status(
            orch.RunStatus(
                run_id="i1",
                state="running",
                spec=_spec("i1").to_dict(),
                cost_usd=0.0,
                remote={
                    "provider": "runpod",
                    "endpoint_id": "epA",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "jA",
                    "on_last_gpu": False,
                    "attempt": 0,
                    "started_ts": 1.0,
                },
            )
        )
        # Poll reports a dead/abandoned job (the common redeploy-window outcome).
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="stalled", detail="host vanished"),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        uploads = []
        monkeypatch.setattr(
            "flash.providers._worker.upload_code",
            lambda repo, *, code_prefix: uploads.append((repo, code_prefix)) or repo,
        )
        seen = {}

        def fake_training(
            spec,
            log,
            *,
            prior_cost,
            runtime_secrets=None,
            code_prefix=None,
            attempt_start,
        ):
            seen["remote"] = orch.get_status(spec.run_id).remote
            seen["code_prefix"] = code_prefix
            seen["hf_repo"] = spec.train.hf_repo
            orch._update(spec.run_id, "done", cost_usd=prior_cost)

        monkeypatch.setattr(orch, "_run_training", fake_training)

        st = orch.attach_run("i1", log_stream=sys.stderr)

        assert seen["remote"] is None, "stale dead handle must be cleared before resuming"
        assert re.fullmatch(r"code/[0-9a-f]{32}/flash", seen["code_prefix"] or "")
        assert uploads == [(seen["hf_repo"], seen["code_prefix"])]
        assert st.state != "failed", "a job lost to the redeploy must be resumed, not failed"
        assert st.state == "done"


def test_attach_one_shot_failure_does_not_submit_attempt_one(monkeypatch):
    from dataclasses import replace

    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        spec = _spec("one-shot-recovery")
        spec = replace(spec, gpu=replace(spec.gpu, max_retries=0))
        orch._save_status(
            orch.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                remote={
                    "provider": "runpod",
                    "endpoint_id": "epA",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "jA",
                    "on_last_gpu": True,
                    "attempt": 0,
                    "started_ts": 1.0,
                },
            )
        )
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="stalled", detail="host vanished"),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        training_calls = []
        monkeypatch.setattr(orch, "_run_training", lambda *a, **k: training_calls.append((a, k)))

        status = orch.attach_run(spec.run_id, log_stream=sys.stderr)

        assert status.state == "failed"
        assert status.error == "stalled: host vanished"
        assert training_calls == []
        assert status.remote is None


def test_attach_resume_reuses_persisted_code_prefix(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        orch._save_status(
            orch.RunStatus(
                run_id="pinned-code",
                state="running",
                spec=_spec("pinned-code").to_dict(),
                cost_usd=0.25,
                remote={
                    "provider": "runpod",
                    "endpoint_id": "epA",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "jA",
                    "on_last_gpu": False,
                    "attempt": 0,
                    "started_ts": 1.0,
                    "code_prefix": "code/0123456789abcdef0123456789abcdef/flash",
                },
            )
        )
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="stalled", detail="host vanished"),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        seen = {}

        def fake_training(
            spec,
            log,
            *,
            prior_cost,
            runtime_secrets=None,
            code_prefix=None,
            attempt_start,
        ):
            seen["prior_cost"] = prior_cost
            seen["code_prefix"] = code_prefix
            seen["attempt_start"] = attempt_start
            orch._update(spec.run_id, "done", cost_usd=prior_cost)

        monkeypatch.setattr(orch, "_run_training", fake_training)

        st = orch.attach_run("pinned-code", log_stream=sys.stderr)

        assert st.state == "done"
        assert seen == {
            "prior_cost": 0.25,
            "code_prefix": "code/0123456789abcdef0123456789abcdef/flash",
            "attempt_start": 1,
        }


def test_attach_resume_that_fails_again_marks_run_failed(monkeypatch):
    # The resume delegates the genuine-vs-infra decision to the training submit (unchanged): a run
    # that is truly broken reproduces the failure on the resumed attempt, _run_training fails it, and
    # attach surfaces that terminal `failed` — so a broken run still terminates (nothing hangs).
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        orch._save_status(
            orch.RunStatus(
                run_id="g1",
                state="running",
                spec=_spec("g1").to_dict(),
                remote={
                    "provider": "runpod",
                    "endpoint_id": "epA",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "jA",
                    "on_last_gpu": False,
                    "attempt": 0,
                    "started_ts": 1.0,
                },
            )
        )
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(
                False, failure="worker_error", detail="Traceback ...\nRuntimeError: bad reward fn"
            ),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        monkeypatch.setattr(
            "flash.providers._worker.upload_code", lambda repo, *, code_prefix: repo
        )
        monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
        resumed = {"called": False}
        replacement_remote = {
            "provider": "runpod",
            "endpoint_id": "epB",
            "endpoint_name": "replacement",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "jB",
            "on_last_gpu": False,
            "attempt": 1,
            "started_ts": 2.0,
        }

        def fake_training(
            spec,
            log,
            *,
            prior_cost,
            runtime_secrets=None,
            code_prefix=None,
            attempt_start,
        ):
            # the training submit re-runs the run; a genuinely broken run fails there (matches
            # _submit_seed_supervised raising after a non-infra failure with no retries left).
            resumed["called"] = True
            resumed["attempt_start"] = attempt_start
            orch._update(spec.run_id, "running", remote=replacement_remote)
            raise RuntimeError("run failed after retries: worker_error: bad reward fn")

        monkeypatch.setattr(orch, "_run_training", fake_training)

        st = orch.attach_run("g1", log_stream=sys.stderr)

        assert resumed["called"] is True, (
            "attach must attempt a checkpoint resume on any non-ok poll"
        )
        assert st.state == "failed", "a resume that fails again must terminate the run"
        assert st.remote is None
        assert "bad reward fn" in (st.error or "")
        assert orch._load_status_json("g1")[orch._CLEANUP_REMOTES_KEY] == [
            {key: value for key, value in replacement_remote.items() if key != "on_last_gpu"},
            {
                "provider": "runpod",
                "endpoint_id": "epA",
                "endpoint_name": "n",
                "key_fingerprint": _RUNPOD_FINGERPRINT,
                "job_id": "jA",
                "attempt": 0,
                "started_ts": 1.0,
            },
        ]


@pytest.mark.parametrize(
    ("teardown_failure", "status_mode"),
    [
        ("cancel", "in_progress"),
        ("delete_false", "in_progress"),
        ("delete_exception", "raises"),
    ],
)
def test_attach_does_not_resume_over_unconfirmed_runpod_teardown(
    monkeypatch, teardown_failure, status_mode
):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        from flash.providers.runpod import api as runpod_api

        remote = {
            "provider": "runpod",
            "endpoint_id": "ep-old",
            "endpoint_name": "old",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "job-old",
            "attempt": 0,
            "started_ts": 1.0,
        }
        orch._save_status(
            orch.RunStatus(
                run_id="runpod-unconfirmed",
                state="running",
                spec=_spec("runpod-unconfirmed").to_dict(),
                remote=remote,
            )
        )
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="stalled", detail="stalled"),
        )
        teardown_events = []

        def cancel_job(endpoint_id, job_id, **_kw):
            teardown_events.append(("cancel", endpoint_id, job_id))
            if teardown_failure == "cancel":
                raise runpod_api.RunpodApiError("cancel unconfirmed")
            return {"id": job_id, "status": "CANCELLED"}

        def delete_endpoint(endpoint_id, _fingerprint):
            teardown_events.append(("delete", endpoint_id))
            if teardown_failure == "delete_false":
                return False
            if teardown_failure == "delete_exception":
                raise runpod_api.RunpodApiError("delete unconfirmed")
            return teardown_failure != "cancel"

        def job_status(*_args, **_kwargs):
            if status_mode == "raises":
                raise runpod_api.RunpodApiError("status unconfirmed")
            return {"status": "IN_PROGRESS"}

        monkeypatch.setattr(runpod_api, "cancel_job", cancel_job)
        monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", delete_endpoint)
        monkeypatch.setattr(runpod_api, "job_status", job_status)
        monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: None)
        import flash.runner.deploy as deploy_mod

        scheduled = []
        monkeypatch.setattr(
            deploy_mod,
            "_schedule_attach_reconciliation",
            lambda *args, **kwargs: scheduled.append((args, kwargs)) or True,
        )
        resumed = []
        monkeypatch.setattr(orch, "_run_training", lambda *a, **k: resumed.append(True))

        status = orch.attach_run("runpod-unconfirmed", log_stream=sys.stderr)

        assert teardown_events[:2] == [
            ("cancel", "ep-old", "job-old"),
            ("delete", "ep-old"),
        ]
        assert resumed == []
        assert len(scheduled) == 1
        assert status.state == "running"
        assert status.remote == remote


def test_attach_preserves_newer_remote_before_compare_and_clear(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.jobs as jobs

        old_remote = {
            "provider": "runpod",
            "endpoint_id": "ep-old",
            "endpoint_name": "old",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "job-old",
            "attempt": 0,
            "started_ts": 1.0,
        }
        newer_remote = {
            "provider": "runpod",
            "endpoint_id": "ep-new",
            "endpoint_name": "new",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "job-new",
            "attempt": 1,
            "started_ts": 2.0,
        }
        orch._save_status(
            orch.RunStatus(
                run_id="attach-newer-remote",
                state="running",
                spec=_spec("attach-newer-remote").to_dict(),
                remote=old_remote,
            )
        )
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="stalled", detail="stalled"),
        )
        import flash.runner.lifecycle as lifecycle_mod

        real_teardown = lifecycle_mod._strict_teardown_handle

        def teardown_then_replace(handle, run_id=None):
            result = real_teardown(handle, run_id)
            assert orch._update("attach-newer-remote", "running", remote=newer_remote)
            return result

        monkeypatch.setattr(lifecycle_mod, "_strict_teardown_handle", teardown_then_replace)
        gc_calls = []
        monkeypatch.setattr(orch, "_gc_run_endpoints", lambda _spec: gc_calls.append(True))
        resumed = []
        monkeypatch.setattr(orch, "_run_training", lambda *a, **k: resumed.append(True))

        status = orch.attach_run("attach-newer-remote", log_stream=sys.stderr)

        assert resumed == []
        assert gc_calls == []
        assert status.state == "running"
        assert status.remote == newer_remote


@pytest.mark.parametrize("remaining_mode", ["present", "raises"])
def test_attach_does_not_resume_over_unconfirmed_vast_teardown(monkeypatch, remaining_mode):
    # Codex: a recovered Vast run whose poll ended not-ok must CONFIRM the in-flight instance is gone
    # before resuming. If destroy() raises (unconfirmed DELETE — the old worker may still be running and
    # writing this run's HF artifacts), attach must NOT launch a second worker (double-bill + corrupt
    # the shared DONE/metrics); it keeps the handle and leaves the run non-terminal so a later
    # recovery/sweep reconciles. Mirrors the retry-loop MtzrH guard.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers as providers
        import flash.providers.runpod.train as flash_train
        from flash.providers.base import PollResult
        from flash.providers.vast import api as vast_api

        orch._save_status(
            orch.RunStatus(
                run_id="v1",
                state="running",
                spec=_spec("v1").to_dict(),
                cost_usd=0.0,
                remote={
                    "provider": "vast",
                    "instance_id": 101,
                    "offer_id": 202,
                    "machine_id": 303,
                    "label": "flash-v1",
                    "gpu": "RTX 4090",
                    "hourly_usd": 0.5,
                    "attempt": 0,
                    "started_ts": 1.0,
                },
            )
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        class _RaisingVast:
            def poll(self, handle, spec, seed, *, log=None, _deadline_at=None):
                assert _deadline_at == pytest.approx(
                    orch.get_status("v1").created_at + _spec("v1").gpu.max_wall_seconds
                )
                return PollResult(False, failure="stalled", detail="host vanished")

            def destroy(self, handle):  # unconfirmed teardown -> attach must not resume over it
                raise vast_api.VastApiError("destroy unconfirmed (success:false)")

            def gc(self, spec):  # best-effort label reap
                pass

            def run_instances_remaining(self, run_id):
                if remaining_mode == "raises":
                    raise vast_api.VastApiError("instance listing unavailable")
                return [101]

            def is_configured(self):  # available_providers() probes this in the terminal-GC finally
                return False

        real_get = providers.get_provider
        monkeypatch.setattr(
            providers,
            "get_provider",
            lambda name: _RaisingVast() if name == "vast" else real_get(name),
        )

        resumed = {"called": False}

        def fake_loop(spec, log, *, prior_cost, runtime_secrets=None):
            resumed["called"] = True

        monkeypatch.setattr(orch, "_run_training", fake_loop)
        import flash.runner.deploy as deploy_mod

        scheduled = []
        monkeypatch.setattr(
            deploy_mod,
            "_schedule_attach_reconciliation",
            lambda *args, **kwargs: scheduled.append((args, kwargs)) or True,
        )

        st = orch.attach_run("v1", log_stream=sys.stderr)

        assert resumed["called"] is False, (
            "must NOT resume a second worker over a possibly-live box"
        )
        assert len(scheduled) == 1
        assert st.state == "running", "run left non-terminal for a later recovery/sweep"
        remote = orch.get_status("v1").remote
        assert remote is not None, "handle preserved"
        assert remote.get("instance_id") == 101, "handle preserved"


def _vast_recovery_remote(instance_id=101, attempt=0):
    return {
        "provider": "vast",
        "instance_id": instance_id,
        "offer_id": 202,
        "machine_id": 303,
        "label": "flash-reconcile",
        "gpu": "RTX 4090",
        "hourly_usd": 0.5,
        "attempt": attempt,
        "started_ts": 100.0,
        "code_prefix": "code/revision",
    }


def test_attach_reconciliation_thread_start_failure_releases_guard(monkeypatch):
    import flash.runner.deploy as deploy_mod

    run_id = "attach-reconcile-thread-start"
    remote = _vast_recovery_remote()
    spec = _spec(run_id)
    with deploy_mod._ATTACH_RECONCILING_LOCK:
        deploy_mod._ATTACH_RECONCILING.discard(run_id)

    class RaisingThread:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(deploy_mod.threading, "Thread", RaisingThread)
    with pytest.raises(RuntimeError, match="thread start failed"):
        deploy_mod._schedule_attach_reconciliation(
            run_id,
            remote,
            spec,
            1,
            "code/revision",
            io.StringIO(),
            "stalled: host vanished",
        )
    with deploy_mod._ATTACH_RECONCILING_LOCK:
        assert run_id not in deploy_mod._ATTACH_RECONCILING

    monkeypatch.setattr(deploy_mod, "_reconcile_attached_remote", lambda *a, **k: None)

    class ImmediateThread:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(deploy_mod.threading, "Thread", ImmediateThread)
    assert deploy_mod._schedule_attach_reconciliation(
        run_id,
        remote,
        spec,
        1,
        "code/revision",
        io.StringIO(),
        "stalled: host vanished",
    )
    with deploy_mod._ATTACH_RECONCILING_LOCK:
        assert run_id not in deploy_mod._ATTACH_RECONCILING


def test_attach_reconciler_resumes_after_vast_strict_absence(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers as providers
        import flash.runner.deploy as deploy_mod

        remote = _vast_recovery_remote()
        spec = _spec("vast-reconcile-clear")
        orch._save_status(
            orch.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                remote=remote,
            )
        )

        class Provider:
            def destroy(self, _handle):
                raise RuntimeError("delete acknowledgement unavailable")

            def run_instances_remaining(self, run_id):
                return []

        monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
        monkeypatch.setattr(deploy_mod, "_ATTACH_RECONCILE_INTERVAL_S", 0.0)
        resumed = []
        monkeypatch.setattr(
            orch,
            "_run_training",
            lambda *args, **kwargs: resumed.append(kwargs["attempt_start"]),
        )

        deploy_mod._reconcile_attached_remote(
            spec.run_id,
            remote,
            spec,
            1,
            "code/revision",
            io.StringIO(),
            "stalled: host vanished",
        )

        assert resumed == [1]
        assert orch.get_status(spec.run_id).state == "running"
        assert orch.get_status(spec.run_id).remote is None


def test_attach_reconciler_deadline_retries_terminal_persistence(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.runner.deploy as deploy_mod
        import flash.runner.lifecycle as lifecycle_mod

        remote = _vast_recovery_remote()
        spec = _spec("vast-reconcile-deadline")
        orch._save_status(
            orch.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                created_at=100.0,
                remote=remote,
            )
        )
        deadline = orch._load_run_deadline_at(spec.run_id)
        monkeypatch.setattr(deploy_mod.time, "time", lambda: deadline + 1.0)
        monkeypatch.setattr(deploy_mod, "_ATTACH_RECONCILE_INTERVAL_S", 0.0)
        monkeypatch.setattr(lifecycle_mod, "_completed_attempt_metrics", lambda *a, **k: None)
        real_fail = orch._compare_and_fail_remote
        calls = []

        def flaky_fail(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                raise PermissionError("status store unavailable")
            return real_fail(*args, **kwargs)

        monkeypatch.setattr(orch, "_compare_and_fail_remote", flaky_fail)

        deploy_mod._reconcile_attached_remote(
            spec.run_id,
            remote,
            spec,
            1,
            "code/revision",
            io.StringIO(),
            "stalled: host vanished",
        )

        status = orch.get_status(spec.run_id)
        assert len(calls) == 2
        assert status.state == "failed"
        assert status.remote is None
        assert orch._load_status_json(spec.run_id)[orch._CLEANUP_REMOTES_KEY] == [
            {key: value for key, value in remote.items() if key != "code_prefix"}
        ]


def test_attach_reconciler_adopts_completed_phantom_at_deadline(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.runner.deploy as deploy_mod
        import flash.runner.lifecycle as lifecycle_mod

        remote = _vast_recovery_remote()
        spec = _spec("vast-reconcile-complete")
        orch._save_status(
            orch.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                created_at=100.0,
                remote=remote,
            )
        )
        deadline = orch._load_run_deadline_at(spec.run_id)
        monkeypatch.setattr(deploy_mod.time, "time", lambda: deadline + 1.0)
        monkeypatch.setattr(deploy_mod, "_ATTACH_RECONCILE_INTERVAL_S", 0.0)
        monkeypatch.setattr(
            lifecycle_mod,
            "_completed_attempt_metrics",
            lambda *a, **k: {"wall_seconds": 5.0},
        )

        deploy_mod._reconcile_attached_remote(
            spec.run_id,
            remote,
            spec,
            1,
            "code/revision",
            io.StringIO(),
            "stalled: host vanished",
        )

        status = orch.get_status(spec.run_id)
        assert status.state == "done"
        assert status.remote is None
        assert status.error is None
        assert orch._load_status_json(spec.run_id)[orch._CLEANUP_REMOTES_KEY] == [
            {key: value for key, value in remote.items() if key != "code_prefix"}
        ]


def test_attach_reconciler_does_not_clobber_newer_remote(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.runner.deploy as deploy_mod

        old_remote = _vast_recovery_remote(instance_id=101, attempt=0)
        newer_remote = _vast_recovery_remote(instance_id=202, attempt=1)
        spec = _spec("vast-reconcile-newer")
        orch._save_status(
            orch.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                remote=newer_remote,
            )
        )
        monkeypatch.setattr(
            deploy_mod,
            "_resume_after_confirmed_teardown",
            lambda *a, **k: pytest.fail("stale reconciler must not resume"),
        )
        monkeypatch.setattr(
            orch,
            "_compare_and_fail_remote",
            lambda *a, **k: pytest.fail("stale reconciler must not fail the run"),
        )

        deploy_mod._reconcile_attached_remote(
            spec.run_id,
            old_remote,
            spec,
            1,
            "code/revision",
            io.StringIO(),
            "stalled: host vanished",
        )

        status = orch.get_status(spec.run_id)
        assert status.state == "running"
        assert status.remote == newer_remote


def test_update_will_not_overwrite_terminal_with_lifecycle_state(monkeypatch):
    # Terminal states are STICKY: once cancelled, no other state may overwrite it —
    # neither a non-terminal lifecycle write (provisioning/running) NOR a late terminal
    # done/failed from a worker that finished as the cancel arrived. Same-state writes
    # still pass so terminal field updates (cost_usd, error) are preserved.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
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


# ---------------------------------------------------------------------------
# deploy_train_endpoint: quota-error sweep-and-retry
# ---------------------------------------------------------------------------


def _make_runpod_flash_mocks(monkeypatch, FakeRM):
    """Inject fake runpod_flash modules so deploy_train_endpoint can be called without the SDK."""
    import sys
    import types

    class FakeEndpoint:
        def __init__(self, **kwargs):
            pass

        def _build_resource_config(self):
            return {}

    # Mark every stub as a package (via __path__) so Python allows dotted imports from them,
    # e.g. `from runpod_flash.core.resources.resource_manager import ResourceManager`.
    rf_mod = types.ModuleType("runpod_flash")
    rf_mod.__path__ = []
    rf_mod.Endpoint = FakeEndpoint
    monkeypatch.setitem(sys.modules, "runpod_flash", rf_mod)

    core_mod = types.ModuleType("runpod_flash.core")
    core_mod.__path__ = []
    monkeypatch.setitem(sys.modules, "runpod_flash.core", core_mod)

    res_mod = types.ModuleType("runpod_flash.core.resources")
    res_mod.__path__ = []
    monkeypatch.setitem(sys.modules, "runpod_flash.core.resources", res_mod)

    rm_mod = types.ModuleType("runpod_flash.core.resources.resource_manager")
    rm_mod.ResourceManager = FakeRM
    monkeypatch.setitem(sys.modules, "runpod_flash.core.resources.resource_manager", rm_mod)


def _patch_deploy_deps(monkeypatch, jobs):
    """Patch all module-level symbols in jobs that deploy_train_endpoint uses."""
    import os

    import flash.providers.runpod.auth as auth_mod
    import flash.providers.runpod.keys as keys

    if not os.environ.get("RUNPOD_API_KEY"):
        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
        keys.reset()
    monkeypatch.setattr(jobs, "FLASH_SDK_LOCK", __import__("threading").Lock())
    monkeypatch.setattr(jobs, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(auth_mod, "ensure_auth", lambda: keys.active_key())
    monkeypatch.setattr(jobs, "_patch_runpod_backoff", lambda: None)
    monkeypatch.setattr(jobs, "flash_gpu", lambda g: g)
    monkeypatch.setattr(jobs, "canonical_gpu", lambda g: g)
    monkeypatch.setattr(jobs, "endpoint_name", lambda g, s: f"flash-{g}-test")
    monkeypatch.setattr(jobs, "min_cuda_for", lambda g: "12.8")
    monkeypatch.setattr(jobs, "WORKER_IMAGE", "fake-image")
    monkeypatch.setattr(jobs, "worker_image_for_gpu", lambda g, allow_default=True: "fake-image")
    monkeypatch.setattr(jobs, "DEFAULT_EXECUTION_TIMEOUT_MS", 3600000)
    monkeypatch.setattr(jobs, "apply_disk_gb", lambda c, d: None)
    monkeypatch.setattr(jobs, "time", type("T", (), {"sleep": staticmethod(lambda s: None)})())


def test_deploy_train_endpoint_retries_on_quota_error(monkeypatch):
    """On a workers-quota error, deploy_train_endpoint sweeps idle endpoints and retries."""
    import flash.providers.runpod.jobs as jobs

    attempts = {"count": 0}
    swept = {"count": 0}

    class FakeResource:
        id = "ep-new"

    class FakeRM:
        async def get_or_deploy_resource(self, config):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError(
                    "GraphQL errors: Max workers across all endpoints must not exceed "
                    "your workers quota (30)"
                )
            return FakeResource()

    _patch_deploy_deps(monkeypatch, jobs)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)

    def fake_sweep(protected, min_idle_s=0.0, reap_warm=True):
        swept["count"] += 1
        return 5

    monkeypatch.setattr(jobs, "_sweep_idle_flash_endpoints", fake_sweep)

    ep_id, _ep_name, fingerprint = jobs.deploy_train_endpoint("A100", name_suffix="testrun")

    assert ep_id == "ep-new"
    assert fingerprint == jobs.runpod_api.key_fingerprint("test-key")
    assert attempts["count"] == 3, "should take 3 attempts (2 quota failures + 1 success)"
    assert swept["count"] == 2, "should sweep once per quota-error retry"


def test_deploy_train_endpoint_raises_after_max_quota_retries(monkeypatch):
    """deploy_train_endpoint re-raises the quota error after all retries are exhausted."""
    import flash.providers.runpod.jobs as jobs

    class FakeRM:
        async def get_or_deploy_resource(self, config):
            raise RuntimeError(
                "GraphQL errors: Max workers across all endpoints must not exceed "
                "your workers quota (30)"
            )

    _patch_deploy_deps(monkeypatch, jobs)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)
    monkeypatch.setattr(
        jobs, "_sweep_idle_flash_endpoints", lambda protected, min_idle_s=0.0, reap_warm=True: 0
    )

    with pytest.raises(RuntimeError, match="workers quota"):
        jobs.deploy_train_endpoint("A100", name_suffix="testrun")


def test_deploy_fails_over_to_next_account_on_quota(monkeypatch):
    """A multi-account RUNPOD_API_KEY fails the deploy over to the next account on quota."""
    import flash.providers.runpod.jobs as jobs
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "kA,kB")
    keys.reset()

    class FakeResource:
        id = "ep-on-kB"

    class FakeRM:
        async def get_or_deploy_resource(self, config):
            # Account kA is out of worker quota; kB has room. advance_key() (the deploy's
            # failover) moves keys.active_key() from kA to kB.
            if keys.active_key() == "kA":
                raise RuntimeError(
                    "GraphQL errors: Max workers across all endpoints must not exceed "
                    "your workers quota (30)"
                )
            return FakeResource()

    _patch_deploy_deps(monkeypatch, jobs)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)
    monkeypatch.setattr(
        jobs, "_sweep_idle_flash_endpoints", lambda protected, min_idle_s=0.0, reap_warm=True: 0
    )

    ep_id, _name, _fingerprint = jobs.deploy_train_endpoint("A100", name_suffix="testrun")
    assert ep_id == "ep-on-kB"
    assert keys.active_key() == "kB"  # provisioning pointer advanced to the working account


def test_deploy_captures_owner_before_concurrent_failover_after_sdk_create(monkeypatch):
    """A second deploy may fail over immediately after account A creates its endpoint. The returned
    fingerprint must remain account A's, never the newly active account B's."""
    import threading

    import flash.providers.runpod.jobs as jobs
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "account-a,account-b")
    keys.reset()

    class FakeResource:
        id = "ep-on-account-a"

    class FakeRM:
        async def get_or_deploy_resource(self, config):
            return FakeResource()

    class FailoverAfterCreateLock:
        def __init__(self):
            self._lock = threading.Lock()

        def __enter__(self):
            self._lock.acquire()
            return self

        def __exit__(self, exc_type, exc, tb):
            self._lock.release()
            failover = threading.Thread(target=keys.advance_key)
            failover.start()
            failover.join()
            return False

    _patch_deploy_deps(monkeypatch, jobs)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)
    monkeypatch.setattr(jobs, "FLASH_SDK_LOCK", FailoverAfterCreateLock())

    endpoint_id, _name, fingerprint = jobs.deploy_train_endpoint("A100", name_suffix="testrun")

    assert endpoint_id == "ep-on-account-a"
    assert keys.active_key() == "account-b"
    assert fingerprint == jobs.runpod_api.key_fingerprint("account-a")


def test_deploy_raises_when_all_accounts_exhausted_without_looping(monkeypatch):
    """When EVERY account is quota-exhausted, deploy fails over once per account and then RAISES —
    it must NOT loop forever. The deploy bounds its failovers by a key_count()-based COUNT (NOT by
    advance_key()'s return value, which always advances/wraps for a multi-key pool); a regression
    would spin here indefinitely."""
    import flash.providers.runpod.jobs as jobs
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "kA,kB")
    keys.reset()

    calls = {"count": 0}

    class FakeRM:
        async def get_or_deploy_resource(self, config):
            calls["count"] += 1
            if calls["count"] > 20:  # safety net: fail loudly instead of hanging on a regression
                raise AssertionError("deploy failover did not terminate (looped past the pool)")
            raise RuntimeError(
                "GraphQL errors: Max workers across all endpoints must not exceed "
                "your workers quota (30)"
            )

    _patch_deploy_deps(monkeypatch, jobs)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)
    monkeypatch.setattr(
        jobs, "_sweep_idle_flash_endpoints", lambda protected, min_idle_s=0.0, reap_warm=True: 0
    )

    with pytest.raises(RuntimeError, match="workers quota"):
        jobs.deploy_train_endpoint("A100", name_suffix="testrun")
    # 2 accounts x _QUOTA_MAX_RETRIES (3) = 6 deploy attempts, then it stops — never the unbounded spin.
    assert calls["count"] == 6


def test_deploy_failover_from_midpool_tries_every_remaining_account(monkeypatch):
    """REGRESSION: a deploy whose failover STARTS on a non-first account (a prior run already
    advanced the active key) must still try EVERY remaining account before giving up. The earlier
    'advance_key() returns False on wrap-to-index-0' exhaustion heuristic broke this — starting on
    kB, the wrap kB→kC→kA hit index 0 at kA and stopped one account early, skipping kA. The fix
    bounds failovers by key_count()-1, so each remaining account is tried exactly once from any
    start. Here only kA has room; a mid-pool start MUST still reach it."""
    import flash.providers.runpod.jobs as jobs
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "kA,kB,kC")
    keys.reset()
    # Simulate a prior run that left the pointer on kB (mid-pool start for THIS deploy).
    assert keys.advance_key() is True
    assert keys.active_key() == "kB"

    tried: list[str] = []

    class FakeResource:
        id = "ep-on-kA"

    class FakeRM:
        async def get_or_deploy_resource(self, config):
            tried.append(keys.active_key())
            # Only kA has worker quota; kB and kC are exhausted. The deploy must fail over off the
            # mid-pool start, wrap past kC, and reach kA — not stop at the index-0 wrap.
            if keys.active_key() != "kA":
                raise RuntimeError(
                    "GraphQL errors: Max workers across all endpoints must not exceed "
                    "your workers quota (30)"
                )
            return FakeResource()

    _patch_deploy_deps(monkeypatch, jobs)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)
    monkeypatch.setattr(
        jobs, "_sweep_idle_flash_endpoints", lambda protected, min_idle_s=0.0, reap_warm=True: 0
    )

    ep_id, _name, _fingerprint = jobs.deploy_train_endpoint("A100", name_suffix="testrun")
    assert ep_id == "ep-on-kA"
    assert keys.active_key() == "kA"  # wrapped past kC to the working account
    # Every account was tried (each exhausted one _QUOTA_MAX_RETRIES times before failover); the
    # mid-pool start reached kA via the wrap kB→kC→kA instead of stopping at the index-0 wrap.
    assert set(tried) == {"kA", "kB", "kC"}
    # Order: the failover walks kB then kC then wraps to kA — kA is the LAST distinct account tried.
    first_seen = list(dict.fromkeys(tried))
    assert first_seen == ["kB", "kC", "kA"]


def test_sweep_idle_flash_endpoints(monkeypatch):
    """_sweep_idle_flash_endpoints deletes only idle flash-* / live-flash-* endpoints."""
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.jobs as jobs

    # RunPod Flash registers endpoints as "live-<endpoint_name>", so real names are
    # "live-flash-<gpu>-<suffix>". Both the bare "flash-*" and "live-flash-*" forms
    # must be swept; the current run's endpoint (and its "live-" form) must be skipped.
    endpoints = [
        {"id": "ep-live-idle", "name": "live-flash-a100-abc"},  # scaled to zero, idle → delete
        {"id": "ep-warm-idle", "name": "live-flash-a100-warm"},  # WARM idle/ready worker → delete
        {"id": "ep-live-busy", "name": "live-flash-a100-xyz"},  # running a job → keep
        {"id": "ep-initing", "name": "flash-a100-init"},  # worker spinning up → keep
        {"id": "ep-live-skip", "name": "live-flash-a100-cur"},  # live- form of current → skip
        {"id": "ep-bare-idle", "name": "flash-a100-old"},  # bare prefix, idle → delete
        {"id": "ep-skip-bare", "name": "flash-a100-cur"},  # current run (bare) → skip
        {"id": "ep-other", "name": "other-ep"},  # not flash-* → skip
    ]

    def fake_list_by_key():
        return {"k": endpoints}, []

    def fake_health(eid, key):
        if eid in ("ep-live-idle", "ep-bare-idle"):
            return {
                "workers": {"running": 0, "ready": 0, "idle": 0, "initializing": 0},
                "jobs": {"inQueue": 0, "inProgress": 0},
            }
        if eid == "ep-warm-idle":  # warm worker left over after a job, nothing pending → reapable
            return {
                "workers": {"running": 0, "ready": 1, "idle": 1, "initializing": 0},
                "jobs": {"inQueue": 0, "inProgress": 0},
            }
        if eid == "ep-live-busy":
            return {
                "workers": {"running": 1, "ready": 0, "idle": 0, "initializing": 0},
                "jobs": {"inQueue": 0, "inProgress": 1},
            }
        if eid == "ep-initing":  # initializing worker is busy (spinning up) → not reapable
            return {
                "workers": {"running": 0, "ready": 0, "idle": 0, "initializing": 1},
                "jobs": {"inQueue": 0, "inProgress": 0},
            }
        return {}

    deleted = []

    def fake_delete(eid, key):
        deleted.append(eid)
        return True

    monkeypatch.setattr(runpod_api, "list_endpoints_by_key", fake_list_by_key)
    monkeypatch.setattr(runpod_api, "endpoint_health_for_fingerprint", fake_health)
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", fake_delete)
    jobs._idle_since.clear()

    count = jobs._sweep_idle_flash_endpoints(protected={"flash-a100-cur", "live-flash-a100-cur"})

    # warm idle/ready (ep-warm-idle) is reaped too — the dominant leak the old scaled-to-zero rule
    # never caught; running/initializing stay, current-run endpoints are protected.
    assert count == 3
    assert sorted(deleted) == sorted(["ep-live-idle", "ep-warm-idle", "ep-bare-idle"])


def test_sweep_reap_warm_false_keeps_warm_endpoints(monkeypatch):
    """reap_warm=False (the deploy-time reactive sweep, which protects only the current run) reaps
    ONLY fully scaled-to-zero endpoints — never another run's warm idle/ready leftover one."""
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.jobs as jobs

    endpoints = [
        {"id": "ep-warm", "name": "live-flash-a100-warm"},  # warm idle/ready worker
        {"id": "ep-zero", "name": "flash-a100-zero"},  # fully scaled to zero
    ]

    def health(eid, key):
        if eid == "ep-warm":
            return {
                "workers": {"running": 0, "ready": 1, "idle": 1, "initializing": 0},
                "jobs": {"inQueue": 0, "inProgress": 0},
            }
        return {
            "workers": {"running": 0, "ready": 0, "idle": 0, "initializing": 0},
            "jobs": {"inQueue": 0, "inProgress": 0},
        }

    deleted = []
    monkeypatch.setattr(runpod_api, "list_endpoints_by_key", lambda: ({"k": endpoints}, []))
    monkeypatch.setattr(runpod_api, "endpoint_health_for_fingerprint", health)
    monkeypatch.setattr(
        runpod_api, "delete_endpoint_for_fingerprint", lambda eid, key: deleted.append(eid) or True
    )

    # Deploy-path mode: warm endpoint is treated as busy and kept; only scaled-to-zero is reaped.
    jobs._idle_since.clear()
    assert jobs._sweep_idle_flash_endpoints(protected=set(), reap_warm=False) == 1
    assert deleted == ["ep-zero"]

    # Periodic-reaper mode (default): the warm endpoint is reaped too.
    deleted.clear()
    jobs._idle_since.clear()
    assert jobs._sweep_idle_flash_endpoints(protected=set()) == 2
    assert sorted(deleted) == sorted(["ep-warm", "ep-zero"])


def test_sweep_idle_grace_requires_sustained_idleness(monkeypatch):
    """With min_idle_s > 0, an endpoint that reports a single transient zero (cold start / between
    jobs) is NOT deleted; only one idle across sweeps for >= min_idle_s is reaped."""
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.jobs as jobs

    monkeypatch.setattr(
        runpod_api,
        "list_endpoints_by_key",
        lambda: ({"k": [{"id": "ep-x", "name": "flash-a100-x"}]}, []),
    )
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, key: {
            "workers": {"running": 0, "ready": 0, "idle": 0, "initializing": 0},
            "jobs": {"inQueue": 0, "inProgress": 0},
        },
    )
    deleted = []
    monkeypatch.setattr(
        runpod_api, "delete_endpoint_for_fingerprint", lambda eid, key: deleted.append(eid) or True
    )
    jobs._idle_since.clear()

    clock = {"t": 1000.0}
    monkeypatch.setattr(jobs.time, "time", lambda: clock["t"])

    # First sweep: idle observed, but grace (300s) not elapsed -> not deleted, timer recorded.
    assert jobs._sweep_idle_flash_endpoints(protected=set(), min_idle_s=300.0) == 0
    assert deleted == []
    assert "ep-x" in jobs._idle_since

    # Still within grace -> still not deleted.
    clock["t"] = 1200.0
    assert jobs._sweep_idle_flash_endpoints(protected=set(), min_idle_s=300.0) == 0
    assert deleted == []

    # Past grace -> reaped.
    clock["t"] = 1400.0
    assert jobs._sweep_idle_flash_endpoints(protected=set(), min_idle_s=300.0) == 1
    assert deleted == ["ep-x"]


def test_sweep_grace_resets_when_endpoint_becomes_busy(monkeypatch):
    """A busy reading clears the grace timer, so the idle clock restarts if it goes idle again —
    a long-running endpoint that dips idle briefly is never reaped."""
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.jobs as jobs

    state = {"busy": False}
    monkeypatch.setattr(
        runpod_api,
        "list_endpoints_by_key",
        lambda: ({"k": [{"id": "ep-x", "name": "flash-a100-x"}]}, []),
    )

    def health(eid, key):
        w = {"running": 1 if state["busy"] else 0, "ready": 0, "idle": 0, "initializing": 0}
        j = {"inQueue": 0, "inProgress": 1 if state["busy"] else 0}
        return {"workers": w, "jobs": j}

    monkeypatch.setattr(runpod_api, "endpoint_health_for_fingerprint", health)
    deleted = []
    monkeypatch.setattr(
        runpod_api, "delete_endpoint_for_fingerprint", lambda eid, key: deleted.append(eid) or True
    )
    jobs._idle_since.clear()
    clock = {"t": 1000.0}
    monkeypatch.setattr(jobs.time, "time", lambda: clock["t"])

    # idle at t=1000 -> timer set
    assert jobs._sweep_idle_flash_endpoints(protected=set(), min_idle_s=300.0) == 0
    assert "ep-x" in jobs._idle_since
    # busy at t=1200 -> timer cleared
    state["busy"] = True
    clock["t"] = 1200.0
    assert jobs._sweep_idle_flash_endpoints(protected=set(), min_idle_s=300.0) == 0
    assert "ep-x" not in jobs._idle_since
    # idle again at t=1400 -> fresh timer (not deleted: only 0s of new idleness)
    state["busy"] = False
    clock["t"] = 1400.0
    assert jobs._sweep_idle_flash_endpoints(protected=set(), min_idle_s=300.0) == 0
    assert deleted == []


def test_sweep_serializes_on_idle_since_lock(monkeypatch):
    """_idle_since access is guarded: a sweep blocks while another holds the lock (the periodic
    reaper and a deploy-time sweep run on different threads, so the prune can't race mid-iteration)."""
    import threading

    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.jobs as jobs

    monkeypatch.setattr(
        runpod_api,
        "list_endpoints_by_key",
        lambda: ({"k": [{"id": "e", "name": "flash-a100-x"}]}, []),
    )
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, key: {
            "workers": {"running": 0, "ready": 0, "idle": 0, "initializing": 0},
            "jobs": {"inQueue": 0, "inProgress": 0},
        },
    )
    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", lambda eid, key: True)
    jobs._idle_since.clear()

    done = threading.Event()

    def run_sweep():
        jobs._sweep_idle_flash_endpoints(protected=set())
        done.set()

    with jobs._idle_since_lock:
        t = threading.Thread(target=run_sweep)
        t.start()
        # The sweep must block on the lock we hold -> it cannot finish.
        assert not done.wait(0.2)
    t.join(timeout=2)
    assert done.is_set()  # completes as soon as the lock is released
