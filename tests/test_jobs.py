"""Durable run primitives: handle persistence, polling state machine, supervisor retry,
cross-process cancel, and attach (CPU-only; all network mocked)."""

from __future__ import annotations

import base64
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# decode_output / JobHandle
# ---------------------------------------------------------------------------


def test_job_handle_roundtrip():
    from flash.providers.runpod.jobs import JobHandle

    h = JobHandle("ep123", "flash-5090-abc", "job456")
    assert JobHandle.from_dict(h.to_dict()) == h


def test_decode_output_success():
    import cloudpickle

    from flash.providers.runpod.jobs import decode_output

    metrics = {"trained_eval_acc": 0.9, "cost_usd": 0.5}
    out = {"success": True, "result": base64.b64encode(cloudpickle.dumps(metrics)).decode()}
    assert decode_output(out) == metrics


def test_decode_output_error_includes_stdout_tail():
    from flash.providers.runpod.jobs import decode_output

    with pytest.raises(RuntimeError) as ei:
        decode_output({"success": False, "error": "boom", "stdout": "x" * 5000})
    assert "boom" in str(ei.value)


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

    with pytest.raises(RuntimeError) as ei:
        decode_output({"error": "vllm crashed", "stdout": "trace-line\n" + "z" * 5000})
    msg = str(ei.value)
    assert "vllm crashed" in msg
    assert "worker stdout tail" in msg


# ---------------------------------------------------------------------------
# poll_job state machine (mocked runpod_api)
# ---------------------------------------------------------------------------
def _poll(monkeypatch, statuses, heartbeats=None, stall_after_s=10.0):
    import cloudpickle

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    seq = iter(statuses)
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: next(seq))
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    hb_iter = iter(heartbeats) if heartbeats is not None else None

    reader = (lambda force=False: next(hb_iter, None)) if hb_iter is not None else None
    h = jobs.JobHandle("ep", "name", "job")
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
            "processes": [
                {"pid": 1234, "process_name": "/usr/bin/python", "used_memory_gb": 21.9}
            ],
        },
    }
    monkeypatch.setattr("flash.providers._poll._record_heartbeat", lambda _hb: None)

    key, stage = surface_heartbeat(lambda: hb, None, lines.append)

    assert key == ("sft_step", 12, 123.0)
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
        lambda eid, jid: {"status": "FAILED", "error": "worker exploded"},
    )
    log = io.StringIO()
    hb = {
        "run_id": "missing-local-status-is-ok",
        "stage": "boot",
        "ts": 456.0,
        "gpu": {"device_name": "RTX 5090", "gpu_util_pct": 1, "memory_total_gb": 31.8},
    }

    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
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
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: next(seq))
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
        jobs.JobHandle("ep", "name", "job"),
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
        runpod_api, "job_status", lambda eid, jid: {"status": "TIMED_OUT", "error": "timeout"}
    )
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)

    def failure_detail_reader(force=False):
        raise AssertionError("platform terminations should not read worker artifacts")

    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
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

    seq = iter([{"status": "IN_PROGRESS"}, {"status": "FAILED", "error": "boom"}])
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: next(seq))
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    return jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
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
        runpod_api, "job_status", lambda eid, jid: {"status": "COMPLETED", "output": bad}
    )
    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
        interval_s=0,
        heartbeat_reader=lambda force=False: {"retriable": True},
        failure_detail_reader=lambda force=False: "--- error_sft.txt ---\nCUDA out of memory",
    )
    assert res.failure == "job_preempted"
    assert "CUDA out of memory" in res.detail


def test_poll_job_stall_detection(monkeypatch):
    # job stays IN_PROGRESS forever, heartbeat never advances -> stall
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    h = jobs.JobHandle("ep", "name", "job")
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

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_QUEUE"})
    # endpoint_health raises (the common real case for a brand-new endpoint with no workers): its
    # block is swallowed by `except: pass`, so the throttled/unhealthy fast-fails can't arm — the
    # queue backstop must still trip off the authoritative job status alone.
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health",
        lambda eid: (_ for _ in ()).throw(RuntimeError("no workers yet")),
    )
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    h = jobs.JobHandle("ep", "name", "job")
    res = jobs.poll_job(
        h,
        interval_s=0,
        heartbeat_reader=lambda: None,
        setup_grace_s=5000.0,  # large cold-start budget must NOT govern a never-scheduled queue job
        queue_grace_s=900.0,
    )
    assert not res.ok
    assert res.failure == "stalled"
    assert "IN_QUEUE" in res.detail
    assert "next-best GPU" in res.detail


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
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: next(seq))
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    # The IN_QUEUE phase triggers poll_job's endpoint_health probe on the first loop; stub it so the
    # test is hermetic (never hits the network even if RUNPOD_API_KEY is set in the environment).
    monkeypatch.setattr(runpod_api, "endpoint_health", lambda eid: {"workers": {"ready": 1, "running": 1}})
    h = jobs.JobHandle("ep", "name", "job")
    res = jobs.poll_job(h, interval_s=0, heartbeat_reader=lambda: None, queue_grace_s=900.0)
    assert res.ok


def test_poll_job_setup_grace_before_first_heartbeat(monkeypatch):
    # No heartbeat ever (cold start that never finishes): must NOT trip the tight
    # stall_after_s window — it waits for the larger setup_grace_s instead.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    h = jobs.JobHandle("ep", "name", "job")
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

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))

    hbs = iter([{"stage": "train", "step": 1, "ts": 1}])  # then StopIteration -> None

    def reader():
        return next(hbs, None)

    h = jobs.JobHandle("ep", "name", "job")
    res = jobs.poll_job(
        h, interval_s=0, heartbeat_reader=reader, stall_after_s=150.0, setup_grace_s=5000.0
    )
    assert res.failure == "stalled"
    assert "during training" in res.detail


def test_poll_job_setup_heartbeat_does_not_tighten(monkeypatch):
    # A cold-start (setup) heartbeat like "boot" proves liveness but must NOT switch to the
    # tight training window — the slow model-load/vLLM-init still has to fit setup_grace_s.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))

    hbs = iter([{"stage": "boot", "step": None, "ts": 1}])  # then None

    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
        interval_s=0,
        heartbeat_reader=lambda: next(hbs, None),
        stall_after_s=150.0,
        setup_grace_s=5000.0,
    )
    assert res.failure == "stalled"
    assert "during setup" in res.detail
    assert "limit 5000s" in res.detail


def test_poll_job_fast_fails_on_stuck_unhealthy_worker(monkeypatch):
    # A worker stuck UNHEALTHY while IN_QUEUE (e.g. a mutable image tag republished mid-pull) won't
    # self-recover, so poll_job must fail fast on unhealthy_grace_s and NOT burn the full
    # setup_grace_s (~50 min) — returning a retryable stall so the runner re-provisions a fresh
    # endpoint. Regression guard for the multi-hour "waited on a dead worker" failure.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_QUEUE"})
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health",
        lambda eid: {"workers": {"unhealthy": 1, "running": 0, "ready": 0, "idle": 0, "initializing": 0}},
    )
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
        interval_s=0,
        heartbeat_reader=lambda: None,
        stall_after_s=150.0,
        setup_grace_s=100000.0,  # huge: only the unhealthy fast-fail can trip here
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
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: next(statuses))
    monkeypatch.setattr(runpod_api, "endpoint_health", lambda eid: next(healths, {"workers": {}}))
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
        interval_s=0,
        heartbeat_reader=lambda: None,
        stall_after_s=150.0,
        setup_grace_s=100000.0,
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
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_QUEUE"})
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health",
        lambda eid: {"workers": {"throttled": 1, "running": 0, "ready": 0, "idle": 0, "initializing": 0}},
    )
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
        interval_s=0,
        heartbeat_reader=lambda: None,
        stall_after_s=150.0,
        setup_grace_s=100000.0,  # huge: only the throttled fast-fail can trip here
        unhealthy_grace_s=100000.0,  # huge: isolate the throttled path
        throttled_grace_s=300.0,
    )
    assert res.failure == "stalled"  # infra-shaped -> runner retries on the next-best GPU
    assert "throttled" in res.detail


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
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: next(statuses))
    monkeypatch.setattr(runpod_api, "endpoint_health", lambda eid: next(healths, {"workers": {}}))
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
        interval_s=0,
        heartbeat_reader=lambda: None,
        stall_after_s=150.0,
        setup_grace_s=100000.0,
        throttled_grace_s=300.0,
    )
    assert res.ok
    assert res.metrics == {"acc": 1.0}


def test_poll_job_no_reader_keeps_tight_window(monkeypatch):
    # Without a heartbeat_reader we can't tell setup from training, so the larger
    # setup_grace must NOT silently slow stall detection — stay on stall_after_s.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
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

    def flaky(eid, jid):
        calls["n"] += 1
        if calls["n"] < 4:
            raise runpod_api.RunpodApiError("blip")
        return {"status": "COMPLETED", "output": ok}

    monkeypatch.setattr(runpod_api, "job_status", flaky)
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    res = jobs.poll_job(jobs.JobHandle("ep", "n", "j"), interval_s=0, stall_after_s=1e9)
    assert res.ok
    assert calls["n"] == 4


# ---------------------------------------------------------------------------
# Supervisor retry logic (runner) with mocked job submit
# ---------------------------------------------------------------------------
def _fresh_orchestrator(tmp, monkeypatch):
    from tests._helpers.runner import fresh_runner

    return fresh_runner(tmp, monkeypatch)


def _spec(run_id):
    from flash.spec import GpuSpec, JobSpec, TrainSpec

    return JobSpec(
        run_id=run_id,
        model="Qwen/Qwen3.5-0.8B",
        algorithm="grpo",
        train=TrainSpec(seeds=(0,), steps=1),
        gpu=GpuSpec(type="RTX 4090", max_retries=2),
    )


def test_supervisor_retries_on_stall_then_succeeds(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        calls = {"n": 0}

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0):
            calls["n"] += 1
            if on_handle:
                on_handle({"endpoint_id": "ep", "endpoint_name": "n", "job_id": f"j{calls['n']}"})
            if calls["n"] == 1:
                return jobs.PollResult(False, failure="stalled", detail="frozen")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        orch.submit_job(_spec("retry-ok"), dry_run=False, background=False)
        st = orch.get_status("retry-ok")
        assert st.state == "done"
        assert calls["n"] == 2
        assert st.remote["job_id"] == "j2"  # latest handle persisted


def test_supervisor_retries_runpod_cancelled_then_succeeds(monkeypatch):
    # A "job_preempted" first attempt retries on a fresh endpoint and completes.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        calls = {"n": 0}

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0):
            calls["n"] += 1
            if on_handle:
                on_handle({"endpoint_id": "ep", "endpoint_name": "n", "job_id": f"j{calls['n']}"})
            if calls["n"] == 1:
                return jobs.PollResult(False, failure="job_preempted", detail="[CANCELLED] None")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "repo")
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

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0):
            calls["n"] += 1
            return jobs.PollResult(
                False, failure="job_failed", detail="Remote execution failed: ValueError"
            )

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        with pytest.raises(RuntimeError):
            orch.submit_job(_spec("fail-fast"), dry_run=False, background=False)
        assert calls["n"] == 1
        assert orch.get_status("fail-fast").state == "failed"


def test_supervisor_walks_to_next_gpu_class_on_infra_retry(monkeypatch):
    # A policy ("cheapest") request that keeps hitting infra-shaped failures must walk
    # down the ranked candidate list, not burn every retry on the same (capacity-starved)
    # class. With static rates the validated >=24 GB pool for a 0.8B GRPO run ranks
    # RTX 3090 < RTX A6000 < ... by $/hr, so successive attempts step through them.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        gpus_seen: list[str] = []

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0):
            gpus_seen.append(spec.gpu.type)
            if on_handle:
                on_handle({"endpoint_id": "ep", "endpoint_name": "n", "job_id": f"j{attempt}"})
            if attempt < 2:
                return jobs.PollResult(False, failure="stalled", detail="frozen")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="walk",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(seeds=(0,), steps=1),
            gpu=GpuSpec(type="cheapest", max_retries=2),
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
        assert gpus_seen[0] == "RTX 3090"


def test_supervisor_job_failed_without_marker_does_not_retry(monkeypatch):
    # A plain job_failed (no retriable flag — a genuine code crash) is NOT retried: the retry
    # budget exists only for infra-shaped failures, not code bugs.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        calls = {"n": 0}

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0):
            calls["n"] += 1
            return jobs.PollResult(
                False, failure="job_failed", detail="ValueError: bad reward fn (no infra marker)"
            )

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="code-crash",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(seeds=(0,), steps=1),
            gpu=GpuSpec(type="cheapest", max_retries=2),
        )
        with pytest.raises(RuntimeError, match="bad reward fn"):
            orch.submit_job(spec, dry_run=False, background=False)
        assert calls["n"] == 1  # genuine code error burns no retry budget
        assert orch.get_status("code-crash").state == "failed"


def test_supervisor_gpu_walk_clamps_at_last_candidate(monkeypatch):
    # The candidate walk steps to the next-cheapest class on each infra retry but CLAMPS at
    # the last fitting candidate — it must never index past the ranked list. Force a VRAM need that
    # only the 80 GB+ VALIDATED tier satisfies, then trim the ranked candidate list to exactly the
    # two cheapest 80 GB classes (A100 PCIe, A100 SXM) for a clean walk+clamp assertion: attempt 0
    # takes the cheaper (A100 PCIe @ $1.39), attempt 1 walks to the pricier (A100 SXM @ $1.49),
    # attempt 2's walk offset clamps back onto that same last class.
    import dataclasses

    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
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
            return dataclasses.replace(alloc, gpu=best.gpu, hourly_usd=best.hourly_usd, candidates=keep)

        monkeypatch.setattr(allocator, "allocate", two_candidate_allocate)
        gpus_seen: list[str] = []

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0):
            gpus_seen.append(spec.gpu.type)
            if on_handle:
                on_handle({"endpoint_id": "ep", "endpoint_name": "n", "job_id": f"j{attempt}"})
            if attempt < 2:
                return jobs.PollResult(False, failure="stalled", detail="frozen")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="clamp",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(seeds=(0,), steps=1),
            gpu=GpuSpec(type="cheapest", max_retries=2),
        )
        orch.submit_job(spec, dry_run=False, background=False)

        assert orch.get_status("clamp").state == "done"
        # Walk advances to the last candidate then clamps on it (never out of range).
        assert gpus_seen == ["A100 PCIe", "A100 SXM", "A100 SXM"]


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

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0):
            gpus_seen.append(spec.gpu.type)
            if on_handle:
                on_handle({"endpoint_id": "ep", "endpoint_name": "n", "job_id": f"j{attempt}"})
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="alloc-blip",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(seeds=(0,), steps=1),
            gpu=GpuSpec(type="cheapest", max_retries=2),
        )
        orch.submit_job(spec, dry_run=False, background=False)

        assert orch.get_status("alloc-blip").state == "done"
        # First allocation failed (no provision); the retry provisioned the cheapest class
        # (RTX 3090, the cheapest validated 24 GB RunPod class).
        assert gpus_seen == ["RTX 3090"]


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
                "endpoint_id": "epW",
                "endpoint_name": "n",
                "job_id": "jW",
                "allocated_gpu": "RTX 5090",
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
        from flash.providers.runpod.pricing import hourly_rate

        # ~1 GPU-hour on the walked 5090, not the cheaper provisional 4090.
        assert abs(st.cost_usd - hourly_rate("RTX 5090")) < 1e-6
        assert st.cost_usd > hourly_rate("RTX 4090")


# ---------------------------------------------------------------------------
# Cross-process cancel via REST handle + attach
# ---------------------------------------------------------------------------
def test_cancel_uses_rest_handle(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        from flash.providers.runpod import api as runpod_api

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
        import flash.providers.runpod.train as flash_train

        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        st = orch.cancel_run("c1")
        assert st.state == "cancelled"
        assert cancelled == [("epX", "jX")]
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
            remote={"endpoint_id": "epA", "endpoint_name": "n", "job_id": "jA"},
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
        assert abs(st.cost_usd - 0.2) < 1e-9


def test_attach_clears_stale_handle_before_resuming_seeds(monkeypatch):
    # After a recovered seed of a multi-seed run completes, the stale completed
    # handle must be cleared before the remaining seeds run: a restart in the
    # provisioning gap must not reattach recovery to the finished job.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        spec = JobSpec(
            run_id="m1",
            model="Qwen/Qwen3.5-0.8B",
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
            jobs, "poll_job", lambda *a, **k: jobs.PollResult(True, metrics={"cost_usd": 0.2})
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


def test_attach_requires_handle(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        orch._save_status(orch.RunStatus(run_id="nh", state="running", spec=_spec("nh").to_dict()))
        with pytest.raises(ValueError, match="no persisted job handle"):
            orch.attach_run("nh")


def test_attach_resumes_from_checkpoint_on_poll_failure(monkeypatch):
    # A recovered run whose remote job ended not-ok (it died while the control plane was down for
    # the redeploy) must NOT be failed — reattach resumes the in-flight seed on a fresh host
    # (worker resumes from the latest HF checkpoint), exactly like the fresh-submit retry loop. It
    # also records resume_seed_index + clears the stale handle so a second restart during the
    # fresh allocation resumes the right seed.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        orch._save_status(
            orch.RunStatus(
                run_id="i1",
                state="running",
                spec=_spec("i1").to_dict(),
                cost_usd=0.0,
                remote={"endpoint_id": "epA", "endpoint_name": "n", "job_id": "jA", "seed": 0},
            )
        )
        # Poll reports a dead/abandoned job (the common redeploy-window outcome).
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="stalled", detail="host vanished"),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        seen = {}

        def fake_loop(spec, log, *, start_index, prior_cost):
            seen["start_index"] = start_index
            seen["remote"] = orch.get_status(spec.run_id).remote
            seen["resume_seed_index"] = orch.get_status(spec.run_id).resume_seed_index
            orch._update(spec.run_id, "done", cost_usd=prior_cost)

        monkeypatch.setattr(orch, "_run_seed_loop", fake_loop)

        st = orch.attach_run("i1", log_stream=sys.stderr)

        assert seen["start_index"] == 0, "must resume the in-flight seed (index 0), not skip it"
        assert seen["remote"] is None, "stale dead handle must be cleared before resuming"
        assert seen["resume_seed_index"] == 0, "resume marker must be set for a second restart"
        assert st.state != "failed", "a job lost to the redeploy must be resumed, not failed"
        assert st.state == "done"


def test_attach_resume_that_fails_again_marks_run_failed(monkeypatch):
    # The resume delegates the genuine-vs-infra decision to the seed loop (unchanged): a run that
    # is truly broken reproduces the failure on the resumed attempt, the seed loop fails it, and
    # attach surfaces that terminal `failed` — so a broken run still terminates (nothing hangs).
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.train as flash_train

        orch._save_status(
            orch.RunStatus(
                run_id="g1",
                state="running",
                spec=_spec("g1").to_dict(),
                remote={"endpoint_id": "epA", "endpoint_name": "n", "job_id": "jA", "seed": 0},
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
        resumed = {"called": False}

        def fake_loop(spec, log, *, start_index, prior_cost):
            # The seed loop re-runs the seed; a genuinely broken run fails there (matches
            # _submit_seed_supervised raising after a non-infra failure with no retries left).
            resumed["called"] = True
            raise RuntimeError("seed 0 failed after retries: worker_error: bad reward fn")

        monkeypatch.setattr(orch, "_run_seed_loop", fake_loop)

        st = orch.attach_run("g1", log_stream=sys.stderr)

        assert resumed["called"] is True, "attach must attempt a checkpoint resume on any non-ok poll"
        assert st.state == "failed", "a resume that fails again must terminate the run"
        assert "bad reward fn" in (st.error or "")


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


def _make_runpod_flash_mocks(monkeypatch, FakeRM, quota_error_msg=None):
    """Inject fake runpod_flash modules so deploy_train_endpoint can be called without the SDK."""
    import sys
    import types

    class FakeEndpoint:
        def __init__(self, **kwargs): pass
        def _build_resource_config(self): return {}

    rf_mod = types.ModuleType("runpod_flash")
    rf_mod.Endpoint = FakeEndpoint
    monkeypatch.setitem(sys.modules, "runpod_flash", rf_mod)

    rm_mod = types.ModuleType("runpod_flash.core.resources.resource_manager")
    rm_mod.ResourceManager = FakeRM
    monkeypatch.setitem(sys.modules, "runpod_flash.core.resources.resource_manager", rm_mod)

    core_mod = types.ModuleType("runpod_flash.core")
    monkeypatch.setitem(sys.modules, "runpod_flash.core", core_mod)
    res_mod = types.ModuleType("runpod_flash.core.resources")
    monkeypatch.setitem(sys.modules, "runpod_flash.core.resources", res_mod)


def _patch_deploy_deps(monkeypatch, jobs):
    """Patch all module-level symbols in jobs that deploy_train_endpoint uses."""
    import flash.providers.runpod.auth as auth_mod

    monkeypatch.setattr(jobs, "FLASH_SDK_LOCK", __import__("threading").Lock())
    monkeypatch.setattr(jobs, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(auth_mod, "ensure_auth", lambda: None)
    monkeypatch.setattr(jobs, "_patch_runpod_backoff", lambda: None)
    monkeypatch.setattr(jobs, "flash_gpu", lambda g: g)
    monkeypatch.setattr(jobs, "canonical_gpu", lambda g: g)
    monkeypatch.setattr(jobs, "endpoint_name", lambda g, s: f"flash-{g}-test")
    monkeypatch.setattr(jobs, "min_cuda_for", lambda g: "12.8")
    monkeypatch.setattr(jobs, "volume_endpoint_kwargs", lambda s: {})
    monkeypatch.setattr(jobs, "WORKER_IMAGE", "fake-image")
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

    def fake_sweep(skip_name):
        swept["count"] += 1
        return 5

    monkeypatch.setattr(jobs, "_sweep_idle_flash_endpoints", fake_sweep)

    ep_id, ep_name = jobs.deploy_train_endpoint("A100", name_suffix="testrun")

    assert ep_id == "ep-new"
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
    monkeypatch.setattr(jobs, "_sweep_idle_flash_endpoints", lambda skip_name: 0)

    with pytest.raises(RuntimeError, match="workers quota"):
        jobs.deploy_train_endpoint("A100", name_suffix="testrun")


def test_sweep_idle_flash_endpoints(monkeypatch):
    """_sweep_idle_flash_endpoints deletes only idle flash-* endpoints, skips current."""
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.jobs as jobs

    endpoints = [
        {"id": "ep-idle", "name": "flash-a100-abc"},   # idle, should be deleted
        {"id": "ep-busy", "name": "flash-a100-xyz"},   # has running worker, keep
        {"id": "ep-skip", "name": "flash-a100-cur"},   # current run, skip
        {"id": "ep-other", "name": "other-ep"},         # not flash-*, skip
    ]

    def fake_list_endpoints():
        return endpoints

    def fake_health(eid):
        if eid == "ep-idle":
            return {"workers": {"running": 0, "ready": 0, "idle": 0, "initializing": 0},
                    "jobs": {"inQueue": 0, "inProgress": 0}}
        if eid == "ep-busy":
            return {"workers": {"running": 1, "ready": 0, "idle": 0, "initializing": 0},
                    "jobs": {"inQueue": 0, "inProgress": 1}}
        return {}

    deleted = []

    def fake_delete(eid):
        deleted.append(eid)
        return True

    monkeypatch.setattr(runpod_api, "list_endpoints", fake_list_endpoints)
    monkeypatch.setattr(runpod_api, "endpoint_health", fake_health)
    monkeypatch.setattr(runpod_api, "delete_endpoint", fake_delete)

    count = jobs._sweep_idle_flash_endpoints(skip_name="flash-a100-cur")

    assert count == 1
    assert deleted == ["ep-idle"]
