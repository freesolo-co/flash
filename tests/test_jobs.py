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


def test_poll_job_no_capacity_when_stuck_in_queue(monkeypatch):
    # A job that can't LEAVE the queue (a CONFIRMED-empty pool: no usable worker and none
    # initializing) is on a capacity-starved class -> fail FAST with no_capacity (well before
    # setup_grace_s) so the orchestrator walks to the next-cheapest available class instead of
    # burning the full grace.
    # NOTE (merge with dev's THROTTLED fast-fail, #40): a worker reported `throttled` is now owned
    # by the throttled_grace_s path, which fires FIRST and returns a (retryable) `stalled` so the
    # runner walks to the next-best GPU. The no_capacity fast-path is for an empty pool that is NOT
    # throttled (all-zero counters), so this test uses that signal and keeps throttled_grace_s huge
    # to isolate the no_capacity path.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_QUEUE"})
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health",
        lambda eid: {"workers": {"running": 0, "ready": 0, "idle": 0, "initializing": 0}},
    )
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    h = jobs.JobHandle("ep", "name", "job")
    res = jobs.poll_job(
        h,
        interval_s=0,
        heartbeat_reader=lambda: None,
        queue_grace_s=600.0,
        setup_grace_s=5000.0,  # MUCH larger: the queue grace must fire first
        throttled_grace_s=100000.0,  # huge: isolate the no_capacity path from the throttled fast-fail
    )
    assert not res.ok
    assert res.failure == "no_capacity"
    assert "IN_QUEUE" in res.detail


def test_poll_job_initializing_worker_suppresses_no_capacity(monkeypatch):
    # A worker that is `initializing` (large baked image / slow pull) keeps the job IN_QUEUE past
    # queue_grace_s while ACTIVELY coming up — that's a cold start, not capacity starvation. The
    # no_capacity fast-path must NOT fire (it would delete the endpoint mid cold-start); the worker
    # then runs and the job completes. setup_grace_s + the unhealthy fast-path still bound the wait.
    import base64
    import itertools

    import cloudpickle

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    statuses = iter(
        [{"status": "IN_QUEUE"}] * 4
        + [
            {
                "status": "COMPLETED",
                "output": {
                    "success": True,
                    "result": base64.b64encode(cloudpickle.dumps({"acc": 1.0})).decode(),
                },
            }
        ]
    )
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: next(statuses))
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health",
        lambda eid: {"workers": {"initializing": 1, "running": 0, "ready": 0, "idle": 0}},
    )
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
        interval_s=0,
        heartbeat_reader=lambda: None,
        queue_grace_s=200.0,  # small: would fire fast WERE it capacity-starved
        setup_grace_s=100000.0,
    )
    assert res.ok
    assert res.metrics == {"acc": 1.0}


def test_poll_job_usable_workers_suppress_no_capacity(monkeypatch):
    # Re-review guard: a job stuck IN_QUEUE past queue_grace_s while the pool reports USABLE workers
    # (running/ready/idle) is a backlog / scheduling delay, NOT capacity starvation. The no_capacity
    # fast-path must NOT fire here (it gates on a CONFIRMED-empty pool, i.e. no usable worker AND
    # none initializing) — deleting an endpoint that has live workers would walk GPU classes blindly.
    # It is bounded by setup_grace_s instead -> a stall, never no_capacity.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_QUEUE"})
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health",
        # A usable worker exists (idle), nothing initializing — but the job hasn't left the queue.
        lambda eid: {"workers": {"idle": 1, "running": 0, "ready": 0, "initializing": 0}},
    )
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
        interval_s=0,
        heartbeat_reader=lambda: None,
        queue_grace_s=200.0,  # small: WOULD trip no_capacity if the fast-path ignored usable workers
        setup_grace_s=600.0,  # the real bound while a usable worker is present
    )
    assert not res.ok
    assert res.failure == "stalled"  # NOT no_capacity
    assert "setup" in res.detail


def test_poll_job_probe_error_does_not_premature_no_capacity(monkeypatch):
    # If every endpoint_health probe ERRORS while IN_QUEUE we have no trustworthy view of the
    # worker pool, so the no_capacity fast-path must NOT fire on a stale/unverified
    # `worker_initializing=False` (deleting an endpoint whose worker may be initializing is the
    # expensive failure direction). Instead the job is bounded by setup_grace_s -> a stall, never a
    # no_capacity that would walk GPU classes blind.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_QUEUE"})

    def _boom(eid):
        raise RuntimeError("health endpoint flaking")

    monkeypatch.setattr(runpod_api, "endpoint_health", _boom)
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
        interval_s=0,
        heartbeat_reader=lambda: None,
        queue_grace_s=200.0,  # small: WOULD trip no_capacity if the fast-path fired on stale False
        setup_grace_s=600.0,  # the real bound when probes are blind
    )
    assert not res.ok
    assert res.failure == "stalled"  # NOT no_capacity
    assert "setup" in res.detail


def test_poll_job_empty_worker_summary_does_not_confirm_capacity(monkeypatch):
    # A /health response whose `workers` summary carries NONE of the expected counter keys (an empty
    # `{}` or a partial/malformed payload) is UNKNOWN, not a confirmed-empty pool: every `.get(...)`
    # would return None and naively read as "no usable, none initializing" -> capacity_confirmed.
    # The fast-path must treat it like a probe flake (no confirmation) and stay bounded by
    # setup_grace_s, never deleting a healthy endpoint on data we never actually saw.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_QUEUE"})
    monkeypatch.setattr(runpod_api, "endpoint_health", lambda eid: {"workers": {}})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
        interval_s=0,
        heartbeat_reader=lambda: None,
        queue_grace_s=200.0,  # small: WOULD trip no_capacity if an empty summary confirmed capacity
        setup_grace_s=600.0,  # the real bound when the pool view is unknown
    )
    assert not res.ok
    assert res.failure == "stalled"  # NOT no_capacity
    assert "setup" in res.detail


def test_poll_job_confirmed_no_capacity_survives_later_probe_flake(monkeypatch):
    # Re-review guard: once a GOOD probe CONFIRMS an empty, non-initializing queue (a real
    # capacity-starved read), a LATER transient probe exception must NOT revive the "maybe a
    # worker is initializing" doubt and defer the no_capacity fast-path to the full setup grace.
    # The confirmed verdict is latched and survives the flake, so no_capacity still fires once
    # queue_grace_s elapses (the orchestrator walks GPU classes promptly instead of stalling ~50m).
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_QUEUE"})

    # First probe: CONFIRMED capacity-starved (empty pool, nothing initializing). Every later
    # probe flakes. The fast-path must still fire off the latched confirmation, not stall.
    calls = {"n": 0}

    def _health(eid):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"workers": {"throttled": 1, "running": 0, "ready": 0, "idle": 0, "initializing": 0}}
        raise RuntimeError("health endpoint flaking")

    monkeypatch.setattr(runpod_api, "endpoint_health", _health)
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
        interval_s=0,
        heartbeat_reader=lambda: None,
        queue_grace_s=300.0,  # fires soon after the confirming probe; setup_grace would be ~50m
        setup_grace_s=100000.0,  # MUCH larger: a stall here would prove the flake erased the verdict
    )
    assert not res.ok
    assert res.failure == "no_capacity"  # latched confirmation survived the flake; walked promptly
    assert "IN_QUEUE" in res.detail


def test_poll_job_reprobes_at_no_capacity_boundary(monkeypatch):
    # PR #4 review (thread 1): the 90s probe cadence means capacity_confirmed can be stale at the
    # queue_grace_s boundary. A worker that starts `initializing` AFTER the confirming probe but
    # before the boundary must NOT be walked away from as if capacity-starved. poll_job must REPROBE
    # at the boundary: seeing a fresh `initializing` worker, it defers (cold start), keeps polling,
    # and the job completes — it does NOT return no_capacity off the stale latch.
    import base64
    import itertools

    import cloudpickle

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    # Stay IN_QUEUE long enough to reach the boundary, then complete.
    statuses = iter(
        [{"status": "IN_QUEUE"}] * 6
        + [
            {
                "status": "COMPLETED",
                "output": {
                    "success": True,
                    "result": base64.b64encode(cloudpickle.dumps({"acc": 1.0})).decode(),
                },
            }
        ]
    )
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: next(statuses))

    # First 90s probe: CONFIRMED empty (capacity-starved) -> latches capacity_confirmed=True.
    # At the no_capacity boundary the REPROBE sees a worker now `initializing` -> defer, not walk.
    calls = {"n": 0}

    def _health(eid):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"workers": {"throttled": 1, "running": 0, "ready": 0, "idle": 0, "initializing": 0}}
        return {"workers": {"initializing": 1, "running": 0, "ready": 0, "idle": 0}}

    monkeypatch.setattr(runpod_api, "endpoint_health", _health)
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
        interval_s=0,
        heartbeat_reader=lambda: None,
        queue_grace_s=200.0,  # small: fires soon after the confirming probe
        setup_grace_s=100000.0,  # large: the deferred cold start must NOT trip a stall here
    )
    # The boundary reprobe found a worker coming up, deferred, and the job ran to completion.
    assert res.ok
    assert res.metrics == {"acc": 1.0}
    assert calls["n"] >= 2  # the boundary reprobe actually happened


def test_poll_job_clears_init_flags_when_not_in_queue(monkeypatch):
    # Unit-level guard for the cross-episode reset (the "stale init flag after queue exit" thread):
    # `worker_initializing` / `init_probe_fresh` are meaningful only WHILE IN_QUEUE and are set
    # solely by the IN_QUEUE health probe. poll_job must zero them on any non-IN_QUEUE status so a
    # later re-queue (IN_QUEUE -> RUNNING -> worker dies -> IN_QUEUE) can't inherit a prior episode's
    # initializing/fresh view. We assert by spying on the locals the loop carries forward: after a
    # RUNNING tick following an initializing IN_QUEUE tick, no_capacity in the *next* IN_QUEUE
    # episode must depend on a NEW probe, never the stale one.
    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    # Episode 1 sets initializing=True + fresh=True; RUNNING clears; episode 2 is genuinely throttled
    # and, once its OWN probe confirms an empty pool, must fire no_capacity (not stay suppressed by
    # the stale initializing=True). Generous spacing so a fresh episode-2 probe always runs first.
    statuses = iter(
        [{"status": "IN_QUEUE"}]
        + [{"status": "RUNNING"}]
        + [{"status": "IN_QUEUE"}] * 10
    )
    healths = iter(
        [{"workers": {"initializing": 1, "running": 0, "ready": 0, "idle": 0}}]
        + [{"workers": {"throttled": 1, "running": 0, "ready": 0, "idle": 0, "initializing": 0}}] * 6
    )
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: next(statuses))
    monkeypatch.setattr(runpod_api, "endpoint_health", lambda eid: next(healths, {"workers": {}}))
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    # 100s/iter: every IN_QUEUE iteration re-probes (>90s gate), so episode 2 always has a fresh,
    # episode-local throttled probe -> no_capacity fires on THAT, proving the stale initializing=True
    # from episode 1 did not permanently suppress the walk.
    import itertools

    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))
    res = jobs.poll_job(
        jobs.JobHandle("ep", "name", "job"),
        interval_s=0,
        heartbeat_reader=lambda: None,
        queue_grace_s=200.0,
        setup_grace_s=100000.0,
    )
    assert not res.ok
    assert res.failure == "no_capacity"  # episode 2 walked; stale init flag did not strand the run
    assert "IN_QUEUE" in res.detail


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


def test_rl_server_boot_is_a_setup_stage():
    # The disaggregated rollout server's boot heartbeat must be classified as SETUP, otherwise the
    # first rl_server_boot upload flips to the tight training window while the server is still
    # booting (vLLM model load can take >20 min on a big model).
    from flash.providers.runpod.jobs import _SETUP_HEARTBEAT_STAGES

    assert "rl_server_boot" in _SETUP_HEARTBEAT_STAGES


def test_poll_job_rl_server_boot_does_not_tighten(monkeypatch):
    # An rl_server_boot heartbeat (emitted during the disaggregated vLLM server boot) proves
    # liveness but must keep the larger setup_grace_s — the boot is still pre-training.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))

    hbs = iter([{"stage": "rl_server_boot", "step": None, "ts": 1}])  # then None

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


def test_model_prefetched_is_a_setup_stage():
    # PR #4 review (thread 3): prefetch_model() emits `model_prefetched` the instant the weight
    # download finishes — BEFORE the standalone vLLM server is launched in a disaggregated run. It
    # must be classified SETUP, else the prefetch ping flips to the tight training window and the
    # subsequent ~20-min 35B server boot is killed despite its rl_server_boot heartbeats.
    from flash.providers.runpod.jobs import _SETUP_HEARTBEAT_STAGES

    assert "model_prefetched" in _SETUP_HEARTBEAT_STAGES


def test_poll_job_model_prefetched_does_not_tighten(monkeypatch):
    # A model_prefetched heartbeat (download done, training/server-boot not started) proves
    # liveness but must keep the larger setup_grace_s — it is still pre-training.
    import itertools

    from flash.providers.runpod import api as runpod_api
    from flash.providers.runpod import jobs

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(jobs.time, "time", lambda: next(clock))

    hbs = iter([{"stage": "model_prefetched", "step": None, "ts": 1}])  # then None

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
        # Large queue grace too: this test exercises the unhealthy-recovery path, not the
        # capacity fast-path (queue_grace_s), and the synthetic 100s/step clock would otherwise
        # trip no_capacity before the worker leaves IN_QUEUE.
        queue_grace_s=100000.0,
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


def test_supervisor_no_capacity_excludes_starved_class(monkeypatch):
    # PR #4 review (thread 2): a no_capacity failure must EXCLUDE the starved class from the next
    # allocation, not just bump an index into a freshly re-ranked candidate list. Otherwise a live
    # re-ranking could re-select the same capacity-starved class at the new offset. We assert the
    # exact class that failed no_capacity is passed in exclude_gpu_classes to the re-allocation, and
    # that the walk lands on a DIFFERENT class.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.allocator as allocator
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.pricing as pricing
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        monkeypatch.setattr(pricing, "live_rates", lambda *a, **k: {})

        real_allocate = allocator.allocate
        excluded_seen: list[frozenset] = []

        def spy_allocate(*a, **k):
            excluded_seen.append(frozenset(k.get("exclude_gpu_classes", frozenset())))
            return real_allocate(*a, **k)

        monkeypatch.setattr(allocator, "allocate", spy_allocate)

        gpus_seen: list[str] = []

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0):
            gpus_seen.append(spec.gpu.type)
            if on_handle:
                on_handle({"endpoint_id": "ep", "endpoint_name": "n", "job_id": f"j{attempt}"})
            if attempt < 1:
                # First class is capacity-starved (stuck IN_QUEUE, no free workers).
                return jobs.PollResult(
                    False, failure="no_capacity", detail="no worker assigned; throttled"
                )
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="starved",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(seeds=(0,), steps=1),
            gpu=GpuSpec(type="cheapest", max_retries=2),
        )
        orch.submit_job(spec, dry_run=False, background=False)

        assert orch.get_status("starved").state == "done"
        # Two attempts on two DIFFERENT classes (the first starved, the second served the run).
        assert len(gpus_seen) == 2
        assert gpus_seen[0] != gpus_seen[1]
        # The first allocation excluded nothing; the retry allocation excluded the starved class
        # PROVIDER-SCOPED — (provider, class) — so a RunPod queue-starvation can't drop the same
        # class on Vast (PR #4 thread 2). The suite is RunPod-only, so the pair is tagged runpod.
        assert excluded_seen[0] == frozenset()
        assert ("runpod", gpus_seen[0]) in excluded_seen[1]


def test_supervisor_capacity_walk_resets_offset_after_infra_crash(monkeypatch):
    # PR #4 review (thread): an infra crash advances gpu_walk_offset, then a no_capacity
    # excludes the starved class and re-allocates a SHORTER, re-ranked candidate list. The
    # capacity hop is driven by exclusion (not the index), so the stale offset from the earlier
    # crash must be RESET to 0 — otherwise candidates[offset] on the shorter list skips past the
    # cheapest remaining class and a viable GPU is wrongly walked over. We assert the post-capacity
    # attempt lands on the CHEAPEST remaining class, not an offset-skipped pricier one.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.allocator as allocator
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.pricing as pricing
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        monkeypatch.setattr(pricing, "live_rates", lambda *a, **k: {})

        real_allocate = allocator.allocate
        # Capture the cheapest class of the FULL pool and the cheapest of the pool once the
        # crashed-then-starved class is excluded, so the assertion is independent of the catalog.
        cheapest_full = {"gpu": None}
        cheapest_after_exclude = {"gpu": None}

        def spy_allocate(*a, **k):
            res = real_allocate(*a, **k)
            excluded = frozenset(k.get("exclude_gpu_classes", frozenset()))
            if not excluded and cheapest_full["gpu"] is None:
                cheapest_full["gpu"] = res.candidates[0].gpu
            if excluded:
                cheapest_after_exclude["gpu"] = res.candidates[0].gpu
            return res

        monkeypatch.setattr(allocator, "allocate", spy_allocate)

        gpus_seen: list[str] = []

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0):
            gpus_seen.append(spec.gpu.type)
            if on_handle:
                on_handle({"endpoint_id": "ep", "endpoint_name": "n", "job_id": f"j{attempt}"})
            if attempt == 0:
                # Infra crash on the cheapest class -> keeps the class in the pool, bumps the offset.
                return jobs.PollResult(False, failure="stalled", detail="frozen host")
            if attempt == 1:
                # The offset now points at the 2nd-cheapest class; it is capacity-starved.
                # Exclusion drives the hop and the offset must RESET to the cheapest remaining.
                return jobs.PollResult(
                    False, failure="no_capacity", detail="no worker assigned; throttled"
                )
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="crash-then-starve",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(seeds=(0,), steps=1),
            gpu=GpuSpec(type="cheapest", max_retries=3),
        )
        orch.submit_job(spec, dry_run=False, background=False)

        assert orch.get_status("crash-then-starve").state == "done"
        assert len(gpus_seen) == 3
        # Attempt 0 takes the cheapest; attempt 1 (offset=1 after the crash) takes the 2nd-cheapest.
        assert gpus_seen[0] == cheapest_full["gpu"]
        assert gpus_seen[1] != gpus_seen[0]
        # Attempt 2: the starved 2nd class is excluded and the offset reset, so we land on the
        # cheapest of the REMAINING pool — NOT skipped past it by the stale offset.
        assert cheapest_after_exclude["gpu"] is not None
        assert gpus_seen[2] == cheapest_after_exclude["gpu"]
        # The cheapest remaining (which the run took) is the original cheapest class, re-tried:
        # the crash kept it in the pool, so the reset returns to it rather than walking over it.
        assert gpus_seen[2] == cheapest_full["gpu"]


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


def test_attach_no_capacity_excludes_recovered_class_on_reprovision(monkeypatch):
    # PR #4 review (thread): a run recovered from `no_capacity` re-provisions via the
    # supervised submit. Without carrying the recovered (provider, class) into the capacity
    # walk, the first re-allocation could immediately re-pick the just-starved class and burn
    # another queue grace before walking. Assert the recovered (runpod, allocated_gpu) pair is
    # excluded from the FIRST re-provision allocation, and the run lands on a different class.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.allocator as allocator
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.pricing as pricing
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        monkeypatch.delenv("FLASH_GPU_ALLOW_UNVALIDATED", raising=False)
        monkeypatch.setattr(pricing, "live_rates", lambda *a, **k: {})

        spec = JobSpec(
            run_id="recap",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(seeds=(0,), steps=1),
            gpu=GpuSpec(type="cheapest", max_retries=2),
        )
        # Persist a handle whose allocated class is the one that starved IN_QUEUE.
        status = orch.RunStatus(
            run_id="recap",
            state="running",
            spec=spec.to_dict(),
            remote={
                "endpoint_id": "epR",
                "endpoint_name": "n",
                "job_id": "jR",
                "seed": 0,
                "allocated_gpu": "RTX 5090",
            },
        )
        orch._save_status(status)

        # The recovered poll reports the class is capacity-starved -> recovery re-provisions.
        monkeypatch.setattr(
            jobs,
            "poll_job",
            lambda *a, **k: jobs.PollResult(
                False, failure="no_capacity", detail="no worker assigned; throttled"
            ),
        )

        real_allocate = allocator.allocate
        excluded_seen: list[frozenset] = []

        def spy_allocate(*a, **k):
            excluded_seen.append(frozenset(k.get("exclude_gpu_classes", frozenset())))
            return real_allocate(*a, **k)

        monkeypatch.setattr(allocator, "allocate", spy_allocate)

        gpus_seen: list[str] = []

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0):
            gpus_seen.append(spec.gpu.type)
            if on_handle:
                on_handle({"endpoint_id": "ep2", "endpoint_name": "n", "job_id": f"j{attempt}"})
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        st = orch.attach_run("recap", log_stream=sys.stderr)

        assert st.state == "done"
        # The very first re-provision allocation excluded the recovered, just-starved pair,
        # provider-scoped (runpod) — so the walk can't re-pick it on a live re-ranking.
        assert excluded_seen, "recovery did not re-allocate"
        assert ("runpod", "RTX 5090") in excluded_seen[0]
        # And it landed on a different class than the one that starved.
        assert gpus_seen
        assert gpus_seen[0] != "RTX 5090"


def test_resume_run_carries_persisted_starved_exclusion(monkeypatch):
    # PR #4 review (thread 2): the no_capacity recovery branch clears `remote` and records
    # `resume_seed_index` for a control-plane restart, but it ALSO persists the just-starved
    # (provider, class) into `resume_starved`. If the control plane restarts in the handle-less
    # re-provision gap, recover_runs dispatches to resume_run (no live job to attach to), which
    # starts a FRESH seed loop. Without rehydrating `resume_starved` that loop has an empty
    # exclusion and can immediately re-pick the throttled class. Assert resume_run feeds the
    # persisted pair into the FIRST allocation and lands on a different class.
    with tempfile.TemporaryDirectory() as tmp:
        orch = _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.allocator as allocator
        import flash.providers.runpod.jobs as jobs
        import flash.providers.runpod.pricing as pricing
        import flash.providers.runpod.train as flash_train
        from flash.spec import GpuSpec, JobSpec, TrainSpec

        monkeypatch.delenv("FLASH_GPU_ALLOW_UNVALIDATED", raising=False)
        monkeypatch.setattr(pricing, "live_rates", lambda *a, **k: {})

        spec = JobSpec(
            run_id="resstarve",
            model="Qwen/Qwen3.5-0.8B",
            algorithm="grpo",
            train=TrainSpec(seeds=(0,), steps=1),
            gpu=GpuSpec(type="cheapest", max_retries=2),
        )
        # Persist the post-recovery, handle-less resume state the no_capacity branch leaves:
        # remote cleared, resume_seed_index recorded, and the starved pair stashed.
        status = orch.RunStatus(
            run_id="resstarve",
            state="running",
            spec=spec.to_dict(),
            remote=None,
            resume_seed_index=0,
            resume_starved=[["runpod", "RTX 5090"]],
        )
        orch._save_status(status)

        # Round-trip through disk to prove the field survives (de)serialization, as it would
        # across a real control-plane restart.
        reloaded = orch.get_status("resstarve")
        assert reloaded.resume_starved == [["runpod", "RTX 5090"]]

        real_allocate = allocator.allocate
        excluded_seen: list[frozenset] = []

        def spy_allocate(*a, **k):
            excluded_seen.append(frozenset(k.get("exclude_gpu_classes", frozenset())))
            return real_allocate(*a, **k)

        monkeypatch.setattr(allocator, "allocate", spy_allocate)

        gpus_seen: list[str] = []

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0):
            gpus_seen.append(spec.gpu.type)
            if on_handle:
                on_handle({"endpoint_id": "ep2", "endpoint_name": "n", "job_id": f"j{attempt}"})
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(jobs, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "upload_code", lambda repo=None: "repo")
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        st = orch.resume_run("resstarve", log_stream=sys.stderr)

        assert st.state == "done"
        # The FIRST resumed allocation excluded the persisted, provider-scoped starved pair.
        assert excluded_seen, "resume did not re-allocate"
        assert ("runpod", "RTX 5090") in excluded_seen[0]
        # And it landed on a different class than the one that starved.
        assert gpus_seen
        assert gpus_seen[0] != "RTX 5090"
        # Terminal done clears the persisted exclusion so it can't leak into a later resume.
        assert st.resume_starved is None


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
# worker_image() gate + build_function_input / deploy_train_endpoint branching
# (regression for PR #4: `... or WORKER_IMAGE` made the boot-install fallback
#  unreachable; the baked-vs-fallback choice is now a single shared gate with an
#  explicit FLASH_WORKER_IMAGE disable sentinel.)
# ---------------------------------------------------------------------------


def test_worker_image_default_is_baked(monkeypatch):
    """Unset FLASH_WORKER_IMAGE -> the baked WORKER_IMAGE constant (baked path)."""
    from flash.providers.runpod.train import deps

    monkeypatch.delenv("FLASH_WORKER_IMAGE", raising=False)
    assert deps.worker_image() == deps.WORKER_IMAGE
    assert deps.worker_image()  # non-empty -> baked path selected


def test_worker_image_override(monkeypatch):
    """A non-sentinel FLASH_WORKER_IMAGE overrides the baked tag (operator hotfix)."""
    from flash.providers.runpod.train import deps

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "ghcr.io/x/flash-worker:hotfix")
    assert deps.worker_image() == "ghcr.io/x/flash-worker:hotfix"


def test_worker_image_whitespace_falls_back_to_default(monkeypatch):
    from flash.providers.runpod.train import deps

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "   ")
    assert deps.worker_image() == deps.WORKER_IMAGE


@pytest.mark.parametrize("sentinel", ["none", "NONE", "0", "off", "false", "no"])
def test_worker_image_disable_sentinel_selects_fallback(monkeypatch, sentinel):
    """A disable sentinel makes the boot-install fallback ACTUALLY reachable (returns "")."""
    from flash.providers.runpod.train import deps

    monkeypatch.setenv("FLASH_WORKER_IMAGE", sentinel)
    assert deps.worker_image() == ""  # empty -> boot-install / live-function fallback


def test_build_function_input_baked_returns_payload(monkeypatch):
    """Baked image set (default) -> the payload is the job input verbatim (client mode)."""
    from flash.providers.runpod.jobs import build_function_input

    monkeypatch.delenv("FLASH_WORKER_IMAGE", raising=False)
    payload = {"task": "train", "model": "m"}
    assert build_function_input(payload) is payload


def test_build_function_input_fallback_ships_live_function(monkeypatch):
    """Disable sentinel -> the boot-install fallback: a live-function envelope, NOT the bare
    payload. Proves the previously-dead branch is now reachable and well-formed."""
    from flash.providers.runpod.jobs import build_function_input

    monkeypatch.setenv("FLASH_WORKER_IMAGE", "none")
    out = build_function_input({"task": "train"})
    assert out != {"task": "train"}
    assert out["function_name"] == "_train_body"
    assert out["function_code"]
    assert out["accelerate_downloads"] is True
    # GPU-scoped deps are attached for boot-install on first use.
    assert isinstance(out["dependencies"], list)
