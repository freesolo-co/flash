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

    reader = (lambda: next(hb_iter, None)) if hb_iter is not None else None
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
    # class. With static rates the validated pool for a 0.6B GRPO run ranks
    # A5000 < RTX 4090 < RTX 5090 by $/hr, so successive attempts step through them.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.pricing as pricing
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        # Force the deterministic static ranking (no live pricing fetch) so the validated-only
        # pool ranks A5000 < RTX 4090 < RTX 5090 and successive attempts step through them.
        monkeypatch.setattr(pricing, "live_rates", lambda *a, **k: {})

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
        assert gpus_seen[0] == "RTX A5000"  # cheapest validated class with >= 12 GB


def test_supervisor_gpu_walk_clamps_at_last_candidate(monkeypatch):
    # The candidate walk steps to the next-cheapest class on each infra retry but CLAMPS at
    # the last fitting candidate — it must never index past the ranked list. Force a VRAM need
    # only the 80 GB+ VALIDATED tier satisfies: attempt 0 takes the cheaper (A100 PCIe @ $1.39),
    # attempt 1 walks to the pricier (RTX Pro 6000 WK @ $1.79), attempt 2's walk offset clamps
    # back onto that same last class. (Allocation is restricted to the validated pool, so the
    # unvalidated RTX Pro 6000 Server Edition and A100 SXM are not candidates.)
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.allocator as allocator
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.pricing as pricing
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        monkeypatch.setattr(pricing, "live_rates", lambda *a, **k: {})
        # Need 80 GB -> the two VALIDATED big-VRAM RunPod classes fit: A100 PCIe (80 GB @ $1.39)
        # and RTX Pro 6000 WK (96 GB @ $1.79) -> two candidates.
        monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 80)
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
        assert gpus_seen == ["A100 PCIe", "RTX Pro 6000 WK", "RTX Pro 6000 WK"]


def test_supervisor_allocation_failure_does_not_skip_cheapest(monkeypatch):
    # An allocation/pricing failure must NOT advance the candidate walk: that attempt never
    # provisioned a class, so the retry has to start over from the cheapest, not a pricier
    # one. (Regression guard for the walk-offset-vs-attempt-counter bug.)
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.allocator as allocator
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.pricing as pricing
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        monkeypatch.setattr(pricing, "live_rates", lambda *a, **k: {})

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
        # First allocation failed (no provision); the retry provisioned the cheapest class.
        assert gpus_seen == ["RTX A5000"]


def test_attach_costs_recovered_run_with_walked_gpu(monkeypatch):
    # A policy run that walked to a pricier class persists that class in the handle, so a
    # recovery via attach_run costs the card it actually ran on, not the provisional one.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.pricing as pricing
        import flash.providers.runpod.train as flash_train

        monkeypatch.setattr(pricing, "live_rates", lambda *a, **k: {})
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
        # cancel_run now also destroys the handle's endpoint for cost-safety symmetry
        # with vast (idempotent); the GC backstop may delete it again — endpoint id
        # was torn down, which is what matters.
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


def test_attach_resumes_from_checkpoint_on_infra_failure(monkeypatch):
    # The remote job died for INFRASTRUCTURE reasons (host vanished / TIMED_OUT) while the
    # control plane was down for a redeploy. Reattach must NOT fail the run — it must resume
    # the seed on a fresh host (worker resumes from the latest HF checkpoint), exactly like the
    # fresh-submit retry loop. It also records resume_seed_index + clears the stale handle so a
    # second restart during the fresh allocation resumes the right seed.
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
        # Poll reports a stalled/dead host — the canonical infra-shaped failure.
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
        assert st.state != "failed", "an infra-shaped death must be resumed, not failed"
        assert st.state == "done"


def test_attach_fails_on_genuine_worker_crash(monkeypatch):
    # A genuine worker CODE crash (a real captured traceback, not an infra flake) must fail the
    # run fast — re-running would only reproduce the crash and burn money. So attach must NOT
    # resume in this case.
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
        monkeypatch.setattr(
            orch, "_run_seed_loop", lambda *a, **k: resumed.__setitem__("called", True)
        )

        st = orch.attach_run("g1", log_stream=sys.stderr)

        assert resumed["called"] is False, "a genuine worker crash must NOT be resumed"
        assert st.state == "failed"
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
