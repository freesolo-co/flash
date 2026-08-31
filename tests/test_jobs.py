"""Durable run primitives: handle persistence, polling state machine, supervisor retry,
cross-process cancel, and attach (CPU-only; all network mocked)."""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import time
from types import SimpleNamespace

import pytest

import flash.providers.runpod.execution.job_execution as job_execution
import flash.providers.runpod.execution.polling as polling
import flash.providers.runpod.execution.resources as runpod_resources
import flash.providers.runpod.serverless.endpoints as runpod_endpoints
import flash.runner.accounting.artifacts as runner_artifacts
import flash.runner.accounting.costs as runner_costs
import flash.runner.accounting.reconciliation as runner_reconciliation
import flash.runner.accounting.weight_cache as runner_weight_cache
import flash.runner.lifecycle.deadlines as runner_deadlines
import flash.runner.lifecycle.preparation as runner_preparation
import flash.runner.lifecycle.state as runner_state
import flash.runner.lifecycle.status as runner_status
import flash.runner.lifecycle.submit as runner_submit
import flash.runner.supervise.attach as runner_attach
import flash.runner.supervise.deploy as runner_deploy
import flash.runner.supervise.errors as runner_errors
import flash.runner.supervise.lifecycle as runner_lifecycle
import flash.runner.supervise.recovery as runner_recovery
from flash.core import catalog
from flash.providers._lifecycle.net import worker as provider_worker
from tests._helpers.runner import provisioned_status
from tests._helpers.source_snapshot import valid_source_snapshot

_RUNPOD_FINGERPRINT = "rpk-" + "0" * 64
_SOURCE_SNAPSHOT = valid_source_snapshot()


def _live_clock_handle(jobs):
    """A handle for a test that runs on the REAL clock rather than a faked one.

    The poll loop measures its stall window from the persisted launch, so the fixed 1.0 default
    below describes a job launched in 1970: fine for a test that also fakes `job_execution.time.time` onto a
    synthetic timeline, instantly "wedged" for one that does not. Tests asserting on the status
    machine rather than on timing use this so their launch sits on the same clock they run against.
    """
    return _runpod_handle(jobs, started_ts=time.time())


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
    return {
        **_runpod_handle(
            jobs,
            endpoint_id=endpoint_id,
            endpoint_name=endpoint_name,
            job_id=job_id,
            attempt=attempt,
            started_ts=started_ts,
        ).to_dict(),
        "launch_claim_token": f"claim-{attempt}",
    }


# ---------------------------------------------------------------------------
# decode_output / JobHandle
# ---------------------------------------------------------------------------


def test_job_handle_roundtrip_and_rejects_legacy_shapes():
    from flash.providers.runpod.execution.jobs import JobHandle

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


def test_strict_teardown_uses_valid_runpod_owner_without_inventory(monkeypatch):
    from flash.providers.core.base import JobHandle
    from flash.providers.runpod.client import api as runpod_api
    from flash.runner.supervise import lifecycle

    fingerprint = "rpk-" + "a" * 64
    handle = JobHandle.from_dict(
        {
            "provider": "runpod",
            "endpoint_id": "ep-direct",
            "endpoint_name": "flash-direct",
            "key_fingerprint": fingerprint,
            "job_id": "job-direct",
            "attempt": 0,
            "started_ts": 1.0,
        }
    )
    monkeypatch.setattr(runpod_api, "_key_for_fingerprint", lambda value: "owner-key")
    cancelled = []
    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda endpoint_id, job_id, **_kwargs: (
            cancelled.append((endpoint_id, job_id)) or {"id": job_id, "status": "CANCELLED"}
        ),
    )
    deleted = []

    def delete_endpoint(endpoint_id, owner):
        deleted.append((endpoint_id, runpod_api._key_for_fingerprint(owner)))
        return True

    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", delete_endpoint)
    monkeypatch.setattr(
        runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: pytest.fail("valid owner must not inventory other accounts"),
    )

    assert lifecycle._strict_teardown_handle(handle, "run-direct") is True
    assert cancelled == [("ep-direct", "job-direct")]
    assert deleted == [("ep-direct", "owner-key")]


@pytest.mark.parametrize(
    "fingerprint",
    [
        pytest.param("rpk-" + "a" * 12, id="legacy-16-character"),
        pytest.param(None, id="missing"),
        pytest.param("", id="empty"),
        pytest.param("corrupt", id="corrupt"),
    ],
)
def test_strict_teardown_discovers_runpod_owner_for_invalid_fingerprint(monkeypatch, fingerprint):
    from flash.providers.core.base import JobHandle
    from flash.providers.runpod.client import api as runpod_api
    from flash.runner.supervise import lifecycle

    owner_fingerprint = "rpk-" + "b" * 64
    data = {
        "provider": "runpod",
        "endpoint_id": "ep-discovered",
        "endpoint_name": "flash-discovered",
        "job_id": "job-discovered",
        "attempt": 0,
        "started_ts": 1.0,
    }
    if fingerprint is not None:
        data["key_fingerprint"] = fingerprint
    handle = JobHandle.from_dict(data)

    def resolve(value):
        if value == owner_fingerprint:
            return "discovered-owner-key"
        raise runpod_api.RunpodApiError("unresolvable fingerprint")

    monkeypatch.setattr(runpod_api, "_key_for_fingerprint", resolve)
    monkeypatch.setattr(
        runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: (
            {owner_fingerprint: [{"id": "ep-discovered", "name": "flash-discovered"}]},
            [],
        ),
    )
    deleted = []

    def delete_endpoint(endpoint_id, owner):
        deleted.append((endpoint_id, runpod_api._key_for_fingerprint(owner)))
        return True

    monkeypatch.setattr(runpod_api, "delete_endpoint_for_fingerprint", delete_endpoint)

    assert lifecycle._strict_teardown_handle(handle, "run-discovered") is True
    assert deleted == [("ep-discovered", "discovered-owner-key")]


def test_strict_teardown_keeps_runpod_record_when_no_configured_account_owns_it(monkeypatch):
    """An inventory over the CONFIGURED keys cannot prove an endpoint is gone.

    this path runs only when the persisted fingerprint did not resolve, so the owning credential
    may simply have been removed from `RUNPOD_API_KEY`. "none of my accounts list it" is then
    indistinguishable from "it was deleted", and reporting deletion would let the caller drop the
    cleanup record while an unreachable endpoint keeps billing. refuse instead, so the record
    survives for a drain that may have the owning key configured again.
    """
    from flash.providers.core.base import JobHandle
    from flash.providers.runpod.client import api as runpod_api
    from flash.runner.supervise import lifecycle

    handle = JobHandle.from_dict(
        {
            "provider": "runpod",
            "endpoint_id": "ep-gone",
            "endpoint_name": "flash-gone",
            "key_fingerprint": "legacy-owner",
            "job_id": "job-gone",
            "attempt": 0,
            "started_ts": 1.0,
        }
    )
    monkeypatch.setattr(
        runpod_api,
        "_key_for_fingerprint",
        lambda _value: (_ for _ in ()).throw(runpod_api.RunpodApiError("unresolvable")),
    )
    monkeypatch.setattr(
        runpod_api,
        "list_endpoints_by_key",
        lambda **_kwargs: ({"rpk-" + "a" * 64: []}, []),
    )
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda *_args: pytest.fail("an absent endpoint must not be deleted blindly"),
    )

    with pytest.raises(RuntimeError, match="endpoint deletion could not be confirmed") as exc_info:
        lifecycle._strict_teardown_handle(handle, "run-gone")
    assert "no reachable owner account" in str(exc_info.value.__cause__)


@pytest.mark.parametrize("mode", ["incomplete", "multiple-owners"])
def test_strict_teardown_rejects_unconfirmed_runpod_owner_discovery(monkeypatch, mode):
    from flash.providers.core.base import JobHandle
    from flash.providers.runpod.client import api as runpod_api
    from flash.runner.supervise import lifecycle

    handle = JobHandle.from_dict(
        {
            "provider": "runpod",
            "endpoint_id": "ep-ambiguous",
            "endpoint_name": "flash-ambiguous",
            "key_fingerprint": "legacy-owner",
            "job_id": "job-ambiguous",
            "attempt": 0,
            "started_ts": 1.0,
        }
    )
    monkeypatch.setattr(
        runpod_api,
        "_key_for_fingerprint",
        lambda _value: (_ for _ in ()).throw(runpod_api.RunpodApiError("unresolvable")),
    )
    owner_a = "rpk-" + "a" * 64
    owner_b = "rpk-" + "b" * 64
    inventory = (
        ({owner_a: []}, [owner_b])
        if mode == "incomplete"
        else (
            {
                owner_a: [{"id": "ep-ambiguous"}],
                owner_b: [{"id": "ep-ambiguous"}],
            },
            [],
        )
    )
    monkeypatch.setattr(runpod_api, "list_endpoints_by_key", lambda **_kwargs: inventory)
    monkeypatch.setattr(
        runpod_api,
        "delete_endpoint_for_fingerprint",
        lambda *_args: pytest.fail("ambiguous ownership must not delete"),
    )

    with pytest.raises(RuntimeError, match="endpoint deletion could not be confirmed") as exc_info:
        lifecycle._strict_teardown_handle(handle, "run-ambiguous")
    assert "cleanup unconfirmed" in str(exc_info.value.__cause__)


def test_decode_output_success():
    from flash.providers.runpod.execution.jobs import decode_output

    metrics = {"trained_eval_acc": 0.9, "cost_usd": 0.5}
    assert decode_output(metrics) == metrics


def test_decode_output_error_includes_stdout_tail():
    from flash.providers.runpod.execution.jobs import decode_output

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
    from flash.providers.runpod.execution.jobs import decode_output

    metrics = {"trained_eval_acc": 0.87, "train_wall": 12.3, "cost_usd": 0.04}
    assert decode_output(metrics) == metrics
    # a client-mode error surfaces as an error key
    with pytest.raises(RuntimeError):
        decode_output({"error": "handler blew up"})


def test_decode_output_client_mode_error_includes_stdout_tail():
    """Client-mode failures must also carry the worker stdout tail (poll_job root-causes
    crashes from it) — same as the Flash envelope path."""
    from flash.providers.runpod.execution.jobs import decode_output

    stdout = "STDOUT-BEGIN\n" + ("z" * 5000) + "\nSTDOUT-END"
    with pytest.raises(RuntimeError) as ei:
        decode_output({"error": "vllm crashed", "stdout": stdout})
    msg = str(ei.value)
    assert "vllm crashed" in msg
    assert "worker stdout tail" in msg
    assert "STDOUT-BEGIN" in msg
    assert "STDOUT-END" in msg


def test_decode_output_error_redacts_credentials(monkeypatch):
    """The decoded error reaches the user-readable run log, so it is sanitized like the
    instance providers' failure details are."""
    from flash.providers.runpod.execution.jobs import decode_output

    secret = "hf_ZZZdecodeoutputsecret0123456789"
    monkeypatch.setenv("HF_TOKEN", secret)
    with pytest.raises(RuntimeError) as ei:
        decode_output({"error": f"vllm crashed using {secret}", "stdout": f"401 for {secret}"})
    msg = str(ei.value)
    assert secret not in msg
    assert "vllm crashed using <redacted>" in msg
    assert "--- worker stdout tail ---\n401 for <redacted>" in msg


def test_decode_output_tail_is_sanitized_before_the_bound(monkeypatch):
    """slicing the raw text first can cut a credential at the boundary, leaving a suffix that no
    longer value-matches; the complete text is sanitized before the tail is selected."""
    from flash.providers.runpod.execution.jobs import decode_output

    secret = "hf_ZZZboundarystraddler0123456789abcdef"
    monkeypatch.setenv("HF_TOKEN", secret)
    # non-json output whose 200-char boundary lands inside the secret.
    raw = "x" * 500 + f"auth {secret}" + "y" * 180
    with pytest.raises(RuntimeError) as ei:
        decode_output(raw)
    msg = str(ei.value)
    assert secret not in msg
    for fragment_length in range(6, len(secret)):
        assert secret[-fragment_length:] not in msg
    assert "<redacted>" in msg


# ---------------------------------------------------------------------------
# poll_job state machine (mocked runpod_api)
# ---------------------------------------------------------------------------
def _poll(monkeypatch, statuses, heartbeats=None, stall_after_s=10.0):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    seq = iter(statuses)
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: next(seq))
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    hb_iter = iter(heartbeats) if heartbeats is not None else None

    reader = (lambda force=False: next(hb_iter, None)) if hb_iter is not None else None
    h = _live_clock_handle(jobs)
    ok_payload = {"acc": 1.0}
    return polling.poll_job(
        h,
        interval_s=0,
        heartbeat_reader=reader,
        stall_after_s=stall_after_s,
    ), ok_payload


def test_poll_job_completes(monkeypatch):
    ok = {"acc": 1.0}
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


def test_poll_job_surfaces_heartbeat_before_terminal_return(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    recorded = []
    heartbeat = {
        "run_id": "terminal-heartbeat",
        "stage": "done",
        "ts": 123.0,
        "attempt": 0,
        "metrics_last": [{"step": 4, "reward": 0.75}],
    }
    forces = []
    monkeypatch.setattr(
        "flash.providers._lifecycle.instances.poll._record_heartbeat", recorded.append
    )
    monkeypatch.setattr(jobs, "decode_output", lambda _output: {"acc": 1.0})

    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda _endpoint_id, _job_id, **_kwargs: {"status": "COMPLETED", "output": {}},
    )

    def read_heartbeat(force=False, **_kwargs):
        forces.append(force)
        return heartbeat

    res = polling.poll_job(
        _live_clock_handle(jobs),
        interval_s=0,
        heartbeat_reader=read_heartbeat,
    )

    assert res.ok
    assert forces == [True]
    assert recorded == [heartbeat]


def test_surface_heartbeat_logs_gpu_status(monkeypatch):
    from flash.providers._lifecycle.instances.poll import surface_heartbeat

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
    monkeypatch.setattr(
        "flash.providers._lifecycle.instances.poll._record_heartbeat", lambda _hb: None
    )

    key, stage = surface_heartbeat(lambda: hb, None, lines.append)

    assert key == ("sft_step", 12, 123.0, 0)  # attempt is part of the key (shared seed hb path)
    assert stage == "sft_step"
    assert len(lines) == 1
    line = lines[0]
    assert "worker: stage=sft_step attempt=0 step=12 loss=1.2346" in line
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


def test_poll_job_failure_detail_redacts_secrets_in_provider_error_and_stdout(monkeypatch):
    """A control-plane secret echoed by the worker must not reach the run log (Vast/Lambda
    sanitize every part of their failure detail; RunPod has to match)."""
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    secret = "hf_ZZZterminaldetailsecret0123456789"
    monkeypatch.setenv("HF_TOKEN", secret)
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda eid, jid, **_kw: {
            "status": "FAILED",
            "error": f"worker exploded while authenticating with {secret}",
            "output": {"stdout": f"HTTPError: 401 Unauthorized (used {secret})"},
        },
    )
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)

    res = polling.poll_job(
        _live_clock_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda force=False: None,
    )

    assert not res.ok
    assert secret not in res.detail
    assert "worker exploded while authenticating with <redacted>" in res.detail
    assert "--- worker stdout tail ---\nHTTPError: 401 Unauthorized (used <redacted>)" in res.detail


def test_poll_job_failure_surfaces_forced_heartbeat(monkeypatch):
    import io

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

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

    res = polling.poll_job(
        _live_clock_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda force=False: hb,
        log=log,
    )

    assert not res.ok
    assert res.failure == "job_failed"
    assert "worker: stage=boot" in log.getvalue()
    assert "gpu[RTX 5090" in log.getvalue()


def test_poll_job_failure_appends_worker_artifacts(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

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
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    calls = {"force": None}

    def failure_detail_reader(force=False):
        calls["force"] = force
        return (
            "--- error_sft.txt ---\n"
            "Traceback (most recent call last):\nImportError: no module named flash_attn\n"
            "--- console_sft.txt ---\nworker console tail"
        )

    res = polling.poll_job(
        _live_clock_handle(jobs),
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
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda eid, jid, **_kw: {"status": "TIMED_OUT", "error": "timeout"},
    )
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)

    def failure_detail_reader(force=False):
        raise AssertionError("platform terminations should not read worker artifacts")

    res = polling.poll_job(
        _live_clock_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda force=False: None,
        failure_detail_reader=failure_detail_reader,
    )
    assert not res.ok
    assert res.failure == "job_preempted"
    assert "timeout" in res.detail


def _poll_failed_with_heartbeat(monkeypatch, hb):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    # The heartbeat has to sit on the same timeline as the launch: `worker_flagged_retriable` only
    # honours a heartbeat stamped at or after the launch it claims to describe, so a fixed ts=2.0
    # against a real-clock launch is a 1970 artifact that is correctly ignored, and the retriable
    # flag under test would never be read at all.
    launch_ts = time.time()
    hb = {"ts": launch_ts + 1.0, "attempt": 0, **hb}
    seq = iter([{"status": "IN_PROGRESS"}, {"status": "FAILED", "error": "boom"}])
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: next(seq))
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    return polling.poll_job(
        _runpod_handle(jobs, started_ts=launch_ts),
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
    from flash.providers.runpod.execution.jobs import worker_flagged_retriable

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
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    # An output envelope that decode_output raises RuntimeError on (success False).
    bad = {"success": False, "error": "boom", "stdout": "x"}
    monkeypatch.setattr(
        runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "COMPLETED", "output": bad}
    )
    # the heartbeat shares the launch's timeline: a retriable flag stamped before launch is stale
    # evidence and is correctly ignored, which would make this assert on the wrong reason.
    launch_ts = time.time()
    res = polling.poll_job(
        _runpod_handle(jobs, started_ts=launch_ts),
        interval_s=0,
        heartbeat_reader=lambda force=False: {
            "retriable": True,
            "attempt": 0,
            "ts": launch_ts + 1.0,
        },
        failure_detail_reader=lambda force=False: "--- error_sft.txt ---\nCUDA out of memory",
    )
    assert res.failure == "job_preempted"
    assert "CUDA out of memory" in res.detail


def test_poll_job_stall_detection(monkeypatch):
    # job stays IN_PROGRESS forever, heartbeat never advances -> stall
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    h = _runpod_handle(jobs)
    res = polling.poll_job(h, interval_s=0, heartbeat_reader=lambda: None, stall_after_s=150.0)
    assert not res.ok
    assert res.failure == "stalled"


def test_poll_job_in_queue_capacity_stall(monkeypatch):
    # Job sits IN_QUEUE forever (no worker ever accepts it: no RunPod capacity for the pinned
    # GPU class). RunPod surfaces no THROTTLED/UNHEALTHY worker, so the health-probe fast-fails
    # never arm -> the queue_grace_s backstop must trip a retryable stall well before the ~50 min
    # setup_grace_s, so the runner's gpu-walk re-provisions on the next-best class.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    # endpoint_health raises (the common real case for a brand-new endpoint with no workers): its
    # block is swallowed by `except: pass`, so the throttled/unhealthy fast-fails can't arm — the
    # queue backstop must still trip off the authoritative job status alone.
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: (_ for _ in ()).throw(RuntimeError("no workers yet")),
    )
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    h = _runpod_handle(jobs)
    res = polling.poll_job(
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
    # the detail states the OBSERVED condition and stops there. poll_job cannot know what happens
    # next: the retry disposition lives in _run_training, which owns the candidate list and the
    # retry budget and already prints "retrying ..." / "not retrying" alongside this detail.
    # a provider-side guess contradicts that line whenever the budget is exhausted.
    assert "no RunPod capacity" in res.detail
    assert "retrying" not in res.detail
    assert "GPU-class escalation" not in res.detail


def test_no_capacity_detail_never_predicts_the_retry_disposition(monkeypatch):
    # grace length does not determine retry disposition. lifecycle.py:781 combines last-class and
    # exhausted-budget cases, so poll detail must claim neither a retry nor a class choice.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: (_ for _ in ()).throw(RuntimeError("no workers yet")),
    )
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    h = _runpod_handle(jobs)
    res = polling.poll_job(
        h,
        interval_s=0,
        heartbeat_reader=lambda: None,
        setup_grace_s=5000.0,
        queue_grace_s=300.0,  # not the last GPU
    )
    assert not res.ok
    assert res.failure == "no_capacity"
    assert "no RunPod capacity" in res.detail
    # no next-step claim in EITHER direction -- that is _run_training's line to print.
    assert "retrying" not in res.detail
    assert "next-best" not in res.detail


def _queued_forever(monkeypatch, health, *, step=20.0, queue_grace_s=900.0, log=None):
    """Poll a permanently queued job with controlled endpoint health.

    The clock step preserves the 90-second probe cadence and 300-second worker-coming-up TTL.
    """
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    monkeypatch.setattr(runpod_api, "endpoint_health_for_fingerprint", health)
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=step)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    return polling.poll_job(
        _runpod_handle(jobs),
        log=log,
        interval_s=0,
        heartbeat_reader=lambda: None,
        setup_grace_s=3000.0,
        queue_grace_s=queue_grace_s,
    )


def test_empty_workers_log_elapsed_capacity_grace_before_no_capacity(monkeypatch):
    from itertools import pairwise

    logs = io.StringIO()
    res = _queued_forever(
        monkeypatch,
        lambda eid, _fp, **_kw: {"workers": {}},
        queue_grace_s=700.0,
        log=logs,
    )

    assert not res.ok
    assert res.failure == "no_capacity"
    assert "no RunPod capacity" in res.detail
    capacity_lines = [line for line in logs.getvalue().splitlines() if "capacity grace" in line]
    elapsed_values = []
    for line in capacity_lines:
        match = re.search(r"; waited (\d+)s of 700s capacity grace$", line)
        assert match
        elapsed_values.append(int(match.group(1)))
    assert len(elapsed_values) >= 2
    assert all(later > earlier for earlier, later in pairwise(elapsed_values))


def test_unbounded_capacity_grace_keeps_throttled_worker_failure(monkeypatch):
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    statuses = iter(
        [
            *itertools.repeat({"status": "IN_QUEUE"}, 8),
            {"status": "FAILED", "error": "throttled timer was skipped"},
        ]
    )
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: next(statuses))
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: {"workers": {"throttled": 1}},
    )
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    logs = io.StringIO()

    res = polling.poll_job(
        _runpod_handle(jobs),
        log=logs,
        interval_s=0,
        heartbeat_reader=lambda **_kw: None,
        setup_grace_s=100000.0,
        unhealthy_grace_s=100000.0,
        throttled_grace_s=300.0,
        queue_grace_s=float("inf"),
    )

    assert res.failure == "no_capacity"
    assert "worker stuck THROTTLED" in res.detail
    capacity_lines = [line for line in logs.getvalue().splitlines() if "capacity grace" in line]
    assert capacity_lines
    assert all("of unbounded capacity grace" in line for line in capacity_lines)


def test_four_card_last_gpu_log_names_only_scaled_capacity_grace(monkeypatch):
    from flash.providers.runpod.execution import jobs

    logs = io.StringIO()
    grace = jobs.stall_kwargs(on_last_gpu=True, gpu_count=4)["queue_grace_s"]
    res = _queued_forever(
        monkeypatch,
        lambda eid, _fp, **_kw: {"workers": {}},
        queue_grace_s=grace,
        log=logs,
    )

    assert res.failure == "no_capacity"
    capacity_output = "\n".join(
        line for line in logs.getvalue().splitlines() if "capacity grace" in line
    )
    assert "of 3600s capacity grace" in capacity_output
    assert "pinned" not in capacity_output.lower()
    assert "escalation" not in capacity_output.lower()
    assert "retry" not in capacity_output.lower()


@pytest.mark.parametrize(
    "workers",
    [
        {"initializing": 1},
        {"ready": 1, "unhealthy": 1},
    ],
)
def test_allocated_worker_log_suppresses_capacity_grace(monkeypatch, workers):
    logs = io.StringIO()
    res = _queued_forever(
        monkeypatch,
        lambda eid, _fp, **_kw: {"workers": workers},
        log=logs,
    )

    assert res.failure == "stalled"
    queued_lines = [line for line in logs.getvalue().splitlines() if "queued; workers:" in line]
    assert queued_lines
    assert all("capacity grace" not in line for line in queued_lines)


def test_unhealthy_worker_log_suppresses_capacity_grace(monkeypatch):
    logs = io.StringIO()
    res = _queued_forever(
        monkeypatch,
        lambda eid, _fp, **_kw: {"workers": {"unhealthy": 1}},
        log=logs,
    )

    assert not res.ok
    assert res.failure == "stalled"
    assert "worker stuck unhealthy" in res.detail
    queued_lines = [line for line in logs.getvalue().splitlines() if "queued; workers:" in line]
    assert queued_lines
    assert all("capacity grace" not in line for line in queued_lines)


@pytest.mark.parametrize("granted_workers", [{"initializing": 1}, {"ready": 1}])
def test_empty_health_after_worker_grant_suppresses_capacity_grace(monkeypatch, granted_workers):
    logs = io.StringIO()
    probes = {"count": 0}

    def health(eid, _fp, **_kw):
        probes["count"] += 1
        return {"workers": granted_workers if probes["count"] == 1 else {}}

    res = _queued_forever(monkeypatch, health, queue_grace_s=5000.0, log=logs)

    assert probes["count"] >= 2
    assert not res.ok
    assert res.failure == "stalled"
    assert "setup (pre-training)" in res.detail
    queued_lines = [line for line in logs.getvalue().splitlines() if "queued; workers:" in line]
    assert queued_lines
    assert any("workers: {}" in line for line in queued_lines)
    assert all("capacity grace" not in line for line in queued_lines)


def test_slow_image_pull_is_not_reported_as_missing_capacity(monkeypatch):
    # IN_QUEUE includes worker cold start. once health shows a worker, the setup grace must govern;
    # treating a long image pull as no_capacity would tear it down and restart the same cold pull.
    res = _queued_forever(
        monkeypatch, lambda eid, _fp, **_kw: {"workers": {"initializing": 1, "ready": 0}}
    )
    assert not res.ok
    assert res.failure == "stalled"
    assert "setup (pre-training)" in res.detail


def test_capacity_timer_stays_suppressed_once_the_worker_is_ready(monkeypatch):
    # Same shape one step later in the cold start: the worker has finished pulling and reports
    # ready/idle, but the job has not been dequeued yet. This is the exact health series the real
    # false-positive run showed (idle: 1, ready: 1 fifteen minutes before it was failed
    # no_capacity), so it must not be read as RunPod withholding the GPU either.
    res = _queued_forever(monkeypatch, lambda eid, _fp, **_kw: {"workers": {"idle": 1, "ready": 1}})
    assert not res.ok
    assert res.failure == "stalled"


def test_capacity_timer_rearms_when_worker_health_stops_being_readable(monkeypatch):
    # The health probe lives inside `except Exception: pass`, so a bare "a worker came up" flag
    # would survive a probe outage and suppress the capacity timer for the rest of the run. The
    # observation is stamped and expires after WORKER_COMING_UP_TTL_S: one good reading buys a
    # bounded suppression, not a permanent one. Here the probe answers once and then fails
    # forever, so the capacity verdict must come back.
    from flash.providers.runpod.execution import jobs

    calls = {"n": 0}

    def health(eid, _fp, **_kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"workers": {"initializing": 1}}
        raise RuntimeError("health unavailable")

    res = _queued_forever(monkeypatch, health)
    assert not res.ok
    assert res.failure == "no_capacity"
    assert calls["n"] > 1  # the probe really did keep being attempted and kept failing
    # the suppression window is bounded by a named constant, not by however long the run lasts.
    assert jobs.WORKER_COMING_UP_TTL_S == 300.0


def test_no_worker_at_all_is_still_reported_as_missing_capacity(monkeypatch):
    # Non-regression for the case the backstop actually exists for: health is readable and reports
    # NO worker in any state. Nothing is coming up, so nothing suppresses the timer and the run
    # hands off to the next-best GPU class on schedule.
    res = _queued_forever(monkeypatch, lambda eid, _fp, **_kw: {"workers": {}})
    assert not res.ok
    assert res.failure == "no_capacity"
    assert "no RunPod capacity" in res.detail


def test_a_worker_arriving_inside_the_probe_gap_is_not_abandoned(monkeypatch):
    """See a worker that appears between the last probe and the grace boundary.

    The boundary must re-probe health or it may abandon an already allocated GPU on a stale reading.
    """

    probes = {"n": 0}
    # every SCHEDULED probe answers "no worker". The 8th read is the extra one taken at the grace
    # boundary itself, and only that one can see the worker -- which is exactly the gap being closed:
    # with the boundary probe removed there is no 8th read at all, and the run is failed on the 7th.
    scheduled_probes_before_boundary = 7

    def health(_eid, _fp, **_kw):
        probes["n"] += 1
        if probes["n"] <= scheduled_probes_before_boundary:
            return {"workers": {}}
        return {"workers": {"initializing": 1}}

    res = _queued_forever(monkeypatch, health)

    assert probes["n"] > scheduled_probes_before_boundary, (
        "no health read happened at the grace boundary, so the worker was never observable; the "
        "attempt was abandoned on a reading taken up to 90s earlier"
    )
    assert not res.ok
    # the worker was found, so this is a cold start, not starvation: the verdict must come from the
    # much larger setup grace rather than the capacity timer.
    assert res.failure == "stalled", (
        f"reported {res.failure!r} for a worker RunPod had already allocated; the attempt was "
        "abandoned on health read up to 90s before the boundary"
    )
    assert "setup (pre-training)" in res.detail


def test_capacity_grace_scales_with_gpu_walk_position():
    # capacity backstops wait 5 minutes while another class exists and 15 on the last candidate.
    # placed workers remain governed by the larger setup grace.
    import inspect

    from flash.providers.runpod.execution import jobs, polling

    not_last = jobs.stall_kwargs()  # default on_last_gpu=False
    assert not_last["queue_grace_s"] == 300.0
    assert not_last["throttled_grace_s"] == 300.0
    assert not_last["setup_grace_s"] >= 1800.0  # cold-start budget unchanged

    last = jobs.stall_kwargs(on_last_gpu=True)
    assert last["queue_grace_s"] == 900.0
    assert last["throttled_grace_s"] == 900.0
    assert last["setup_grace_s"] == not_last["setup_grace_s"]  # only the capacity backstops move

    sig = inspect.signature(polling.poll_job)
    assert sig.parameters["queue_grace_s"].default == 300.0
    assert sig.parameters["throttled_grace_s"].default == 300.0
    assert sig.parameters["setup_grace_s"].default >= 1800.0


def test_capacity_grace_scales_with_the_card_count():
    # Multi-card shapes are scarcer than single cards, so a grace sized for 1x expires on a 4x wait
    # that was merely slow rather than starved -- and expiring it does not find capacity faster:
    # the supervisor tears the endpoint down and re-requests the SAME class, paying a fresh cold
    # start to rejoin the queue it just left. Observed as 3-5 attempts and ~55 min of queueing per
    # arm before a single optimizer step, worst on the multi-GPU arms.
    from flash.providers.runpod.execution import jobs

    single = jobs.stall_kwargs(on_last_gpu=True, gpu_count=1)
    assert single["queue_grace_s"] == 900.0

    for count in (2, 4):
        scaled = jobs.stall_kwargs(on_last_gpu=True, gpu_count=count)
        assert scaled["queue_grace_s"] == 900.0 * count, count
        assert scaled["throttled_grace_s"] == 900.0 * count, count
        # only the capacity backstops move: a placed worker's cold start is governed by the setup
        # grace, which has nothing to do with how scarce the shape was to obtain.
        assert scaled["setup_grace_s"] == single["setup_grace_s"], count
        assert scaled["stall_after_s"] == single["stall_after_s"], count

    # the smaller mid-walk budget scales too -- scarcity is a property of the shape, not of where
    # the walk has reached.
    assert jobs.stall_kwargs(gpu_count=4)["queue_grace_s"] == 300.0 * 4

    # bounded, so a hypothetical very wide shape cannot wait without limit. The run's absolute wall
    # deadline is checked every poll iteration regardless, so the cap is a second bound, not the only one.
    assert jobs.stall_kwargs(on_last_gpu=True, gpu_count=8)["queue_grace_s"] == (
        900.0 * jobs.CAPACITY_GRACE_PER_GPU_CAP
    )


def _queued_forever_scaled(monkeypatch, *, gpu_count, heartbeat_reader, workers=None):
    """Drive the REAL poll loop against a job that never leaves IN_QUEUE, on a scaled grace."""
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fp, **_kw: {"workers": workers or {}},
    )
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=20.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    return polling.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=heartbeat_reader,
        **jobs.stall_kwargs(on_last_gpu=True, gpu_count=gpu_count),
    )


@pytest.mark.parametrize("heartbeat_reader", [lambda: None, None])
def test_the_scaled_capacity_grace_is_actually_reached_not_preempted_by_the_stall_timer(
    monkeypatch, heartbeat_reader
):
    # The scaled grace is only real if the poll loop actually waits it out. `_classify_stall` runs
    # on the SAME iteration as the capacity check with its own independent limits (setup_grace_s
    # 3000, stall_after_s 1500), so a 4-card grace of 3600s was cut short at ~3000s -- and, worse,
    # reported as "stalled" rather than "no_capacity".
    #
    # The label is not cosmetic. The supervisor's weight-cache fallback fires only on
    # `no_capacity`/`poll_error`, so a mislabelled capacity failure ALSO stops a cached run from
    # dropping its datacenter-restricting volume and retrying on the unrestricted all-DC pool. It
    # is the whole failure the scaling exists to avoid, with the wait merely renamed.
    #
    # Parametrized over both heartbeat-reader states because they select different stall limits
    # (3000 vs 1500) and both preempted 3600s.
    res = _queued_forever_scaled(monkeypatch, gpu_count=4, heartbeat_reader=heartbeat_reader)
    assert res.failure == "no_capacity", res.detail
    waited = int(res.detail.split("IN_QUEUE for ")[1].split("s ")[0])
    assert waited > 3600, f"gave up at {waited}s, before the 3600s the 4-card shape was granted"


def _queued_until_worker_granted(monkeypatch, *, gpu_count, grant_at, workers=None):
    """Poll a job that sits IN_QUEUE with no worker until ``grant_at``, then reports one placed."""
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    clock_now = {"t": 0.0}

    def health(_eid, _fp, **_kw):
        granted = workers if workers is not None else {"initializing": 1}
        return {"workers": granted if clock_now["t"] >= grant_at else {}}

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    monkeypatch.setattr(runpod_api, "endpoint_health_for_fingerprint", health)
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    ticks = itertools.count(start=0, step=20.0)

    def tick():
        clock_now["t"] = next(ticks)
        return clock_now["t"]

    monkeypatch.setattr(polling.time, "time", tick)
    res = polling.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: None,
        deadline_at=10_000_000.0,
        **jobs.stall_kwargs(on_last_gpu=True, gpu_count=gpu_count),
    )
    return res, clock_now["t"]


def test_a_late_granted_worker_gets_a_full_setup_window_not_the_queues_leftovers(monkeypatch):
    # Deferring the stall timer while IN_QUEUE leaves `last_progress` anchored at queue entry for
    # the whole capacity wait. So a worker granted LATE -- at 3200s of a 4-card shape's 3600s
    # grace -- was measured against a 3000s setup limit its QUEUE wait had already spent, and was
    # torn down as "stalled" on the very poll that found it. That discards a GPU RunPod had just
    # granted and re-requests the same class: exactly the churn this PR exists to remove, arriving
    # by a different route.
    grant_at = 3200.0
    res, ended_at = _queued_until_worker_granted(monkeypatch, gpu_count=4, grant_at=grant_at)
    assert res.failure == "stalled"
    # Measure from the GRANT on the absolute clock, not from the failure's reported "no worker
    # progress for Ns" figure. That figure is computed from whatever `last_progress` holds, so it
    # reads ~3000s in BOTH the fixed and unfixed cases and cannot tell them apart -- a test
    # asserting on it passes against the broken code. What distinguishes them is WHEN the run died:
    # on the grant (~3200s) or a full cold start after it (~6200s).
    granted_for = ended_at - grant_at
    assert granted_for >= 3000.0, (
        f"the placed worker was torn down {granted_for:.0f}s after RunPod granted it (needs the "
        "full 3000s setup window); the setup baseline was not restarted on placement"
    )


def test_a_worker_that_stays_placed_does_not_renew_its_setup_budget_forever(monkeypatch):
    # Guardrail on the SHAPE of the re-anchoring, not a regression of past behavior (it passes
    # against the pre-fix code too, which never rolled the baseline forward at all). It pins the
    # roll-forward to the queued exemption: rolling `last_progress` on every poll that still
    # reports a worker -- the obvious wrong way to write this -- would mean a wedged image pull
    # never reaches the setup grace at all, holding a paid box until the run's wall deadline.
    res, ended_at = _queued_until_worker_granted(monkeypatch, gpu_count=4, grant_at=0.0)
    assert res.failure == "stalled"
    assert "limit 3000s" in res.detail, res.detail
    assert ended_at < 2 * 3000.0, (
        f"a continuously-placed worker ran to {ended_at}s; the setup baseline is being renewed on "
        "every sighting instead of only on placement"
    )


def test_an_unhealthy_worker_counts_as_a_grant(monkeypatch):
    # An `unhealthy` worker is an ALLOCATED box whose image failed to start, so it proves the grant
    # exactly as a healthy one does -- `preload_runpod._has_worker` counts it for the same reason.
    # The latch checked only `usable or recovering`, so health flickering between `{"unhealthy": 1}`
    # and empty never recorded it: each empty snapshot reset the unhealthy timer, the job stayed
    # exempt as never-granted, and the broken box was finally misreported as `no_capacity` (which
    # can trip the weight-cache drop on a run whose capacity was fine all along).
    #
    # Pristine `dev` reports `stalled` here, so this is a regression to avoid, not a change.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    flip = {"n": 0}

    def health(_eid, _fp, **_kw):
        flip["n"] += 1
        return {"workers": {"unhealthy": 1} if flip["n"] % 2 == 0 else {}}

    monkeypatch.setattr(runpod_api, "job_status", lambda *a, **k: {"status": "IN_QUEUE"})
    monkeypatch.setattr(runpod_api, "endpoint_health_for_fingerprint", health)
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    ticks = itertools.count(start=0, step=120.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(ticks))

    res = polling.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: None,
        deadline_at=500_000.0,
        **jobs.stall_kwargs(on_last_gpu=True, gpu_count=4),
    )
    assert res.failure == "stalled", (
        f"a flickering unhealthy worker reported {res.failure!r}; an allocated-but-broken box is "
        f"not absent capacity ({res.detail})"
    )


def test_one_flaky_job_status_does_not_prove_a_worker_grant(monkeypatch):
    # The grant latch must be an allowlist of statuses that PROVE the job left the queue, never
    # `!= "IN_QUEUE"` -- that shape also matches None and any unrecognized string, so a single flaky
    # job_status response would permanently "prove" a grant on a job RunPod never scheduled. The
    # queued-wait exemption would then drop, and a genuine capacity wait would die as `stalled`
    # instead of `no_capacity`, losing the weight-cache fallback that only `no_capacity` triggers.
    #
    # `flash/providers/artifacts/preload_runpod.py` hit this first; its comment warns against
    # exactly this pattern.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    reads = {"n": 0}

    def job_status(_eid, _jid, **_kw):
        reads["n"] += 1
        # never granted -- except one garbled reading partway through.
        if reads["n"] == 3:
            return {"status": "SOMETHING_WEIRD"}
        return {"status": "IN_QUEUE"}

    monkeypatch.setattr(runpod_api, "job_status", job_status)
    monkeypatch.setattr(
        runpod_api, "endpoint_health_for_fingerprint", lambda *a, **k: {"workers": {}}
    )
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    ticks = itertools.count(start=0, step=120.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(ticks))

    res = polling.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: None,
        deadline_at=500_000.0,
        **jobs.stall_kwargs(on_last_gpu=True, gpu_count=4),
    )
    assert res.failure == "no_capacity", (
        f"a never-granted job reported {res.failure!r} after one unrecognized status; only an "
        f"allowlisted status may prove a grant ({res.detail})"
    )


def test_a_current_attempt_heartbeat_proves_a_grant_on_the_reattach_path(monkeypatch):
    # Reattach starts with `ever_saw_worker` false and, if RunPod already requeued the job before
    # recovery attached, never gets to observe the earlier IN_PROGRESS. With health also unreadable
    # the job looked never-granted: exempt from the stall check forever, then reported `no_capacity`
    # despite a heartbeat for THIS attempt proving a worker ran and wrote it.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda *a, **k: {"status": "IN_QUEUE"})

    def dead_health(_eid, _fp, **_kw):
        raise RuntimeError("health endpoint unavailable")

    monkeypatch.setattr(runpod_api, "endpoint_health_for_fingerprint", dead_health)
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    ticks = itertools.count(start=0, step=120.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(ticks))

    # one setup heartbeat for the current attempt (attempt 0), then silence: the worker ran, wrote
    # it, and was taken away. `stage` non-None with a None step keeps this pre-training, so the
    # setup grace (not the training limit) is the one that must bound the wait.
    beats = iter([{"stage": "setup", "step": None, "ts": 100.0, "attempt": 0}])

    def heartbeat_reader():
        return next(beats, None)

    res = polling.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=heartbeat_reader,
        deadline_at=500_000.0,
        current_attempt=0,
        **jobs.stall_kwargs(on_last_gpu=True, gpu_count=4),
    )
    assert res.failure == "stalled", (
        f"a job with a current-attempt heartbeat reported {res.failure!r}; the heartbeat proves a "
        f"worker ran, so this is a lost worker, not absent capacity ({res.detail})"
    )


def test_a_liveness_ping_for_this_attempt_proves_a_grant(monkeypatch):
    # `ever_saw_worker` was latched below `surface_heartbeat`'s `stage is None` return, which drops
    # liveness pings -- and liveness pings are most of what setup publishes (`sft_model_load`,
    # `*_data_loading`, `*_configuring`). So a reattached job whose worker was pinging through model
    # load, with health unreadable, still looked never-granted: exempt from the stall check, then
    # reported `no_capacity` (which can trip the supervisor's weight-cache drop) for a GPU it plainly
    # had. A ping must not advance the stall CLOCK -- that is what the `stage is None` return is for,
    # and it still happens -- but it is proof a worker ran.
    #
    # This PR's scaled grace makes it worse than `dev`, not merely equal: `dev` mislabels at 1200s,
    # a 4-card shape here waited 4200s for the same wrong answer.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda *a, **k: {"status": "IN_QUEUE"})

    def dead_health(_eid, _fp, **_kw):
        raise RuntimeError("health endpoint unavailable")

    monkeypatch.setattr(runpod_api, "endpoint_health_for_fingerprint", dead_health)
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    ticks = itertools.count(start=0, step=120.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(ticks))

    # one liveness ping for the current attempt, then silence: a real setup stage, `liveness` set,
    # so `surface_heartbeat` returns stage None exactly as it does in production.
    beats = iter([{"stage": "sft_model_load", "ts": 100.0, "attempt": 0, "liveness": True}])

    res = polling.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: next(beats, None),
        deadline_at=500_000.0,
        current_attempt=0,
        **jobs.stall_kwargs(on_last_gpu=True, gpu_count=4),
    )
    assert res.failure == "stalled", (
        f"a job whose worker sent a liveness ping for THIS attempt reported {res.failure!r}; the "
        f"ping proves a worker ran, so this is a lost worker, not absent capacity ({res.detail})"
    )
    assert "limit 3000s" in res.detail, res.detail


def test_a_stale_attempt_liveness_ping_does_not_prove_a_grant(monkeypatch):
    # The grant latch moved above the `stage is None` return, so it now sees liveness pings. It must
    # stay BELOW the attempt guard: a previous attempt's worker ran on an allocation this attempt no
    # longer holds, so its ping says nothing about whether this one was ever granted a GPU.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda *a, **k: {"status": "IN_QUEUE"})

    def dead_health(_eid, _fp, **_kw):
        raise RuntimeError("health endpoint unavailable")

    monkeypatch.setattr(runpod_api, "endpoint_health_for_fingerprint", dead_health)
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    ticks = itertools.count(start=0, step=120.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(ticks))

    beats = iter([{"stage": "sft_model_load", "ts": 100.0, "attempt": 3, "liveness": True}])

    res = polling.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: next(beats, None),
        deadline_at=500_000.0,
        current_attempt=0,
        **jobs.stall_kwargs(on_last_gpu=True, gpu_count=4),
    )
    assert res.failure == "no_capacity", (
        f"a ping from attempt 3 was treated as proof that attempt 0 held a GPU (got "
        f"{res.failure!r}); only a heartbeat for the CURRENT attempt proves this grant ({res.detail})"
    )


def test_a_requeued_job_that_already_ran_is_not_reported_as_no_capacity(monkeypatch):
    # `ever_saw_worker` was latched only from endpoint-health snapshots, which go quiet whenever the
    # health endpoint is unreachable (`_probe_worker_coming_up_at` swallows the error, and the
    # periodic health block ends in a bare except). RunPod can requeue a job that already ran, so
    # with health dead the requeued job looked never-granted: it skipped the setup-stall check
    # forever and was finally reported `no_capacity` for a worker it demonstrably had -- and
    # `no_capacity` can trip the supervisor's weight-cache drop.
    #
    # An IN_PROGRESS observation is itself proof of a grant, independent of health. Pristine `dev`
    # reports `stalled` here, so this is a regression to avoid, not a behavior change.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    now = {"t": 0.0}

    # count STATUS READS rather than keying on the clock: one loop iteration consumes several
    # time.time() calls, so a wall-clock window can be stepped straight over without ever being
    # observed (an earlier version of this test did exactly that and passed against broken code).
    reads = {"n": 0}

    def job_status(_eid, _jid, **_kw):
        # queued, then RUNNING (the grant), then requeued by RunPod and never scheduled again.
        reads["n"] += 1
        if reads["n"] <= 2:
            return {"status": "IN_QUEUE"}
        if reads["n"] <= 4:
            return {"status": "IN_PROGRESS"}
        return {"status": "IN_QUEUE"}

    def dead_health(_eid, _fp, **_kw):
        raise RuntimeError("health endpoint unavailable")

    monkeypatch.setattr(runpod_api, "job_status", job_status)
    monkeypatch.setattr(runpod_api, "endpoint_health_for_fingerprint", dead_health)
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    ticks = itertools.count(start=0, step=120.0)

    def tick():
        now["t"] = next(ticks)
        return now["t"]

    monkeypatch.setattr(polling.time, "time", tick)

    res = polling.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: None,
        deadline_at=500_000.0,
        **jobs.stall_kwargs(on_last_gpu=True, gpu_count=4),
    )
    assert res.failure == "stalled", (
        f"a job that reached IN_PROGRESS and was requeued reported {res.failure!r}; leaving the "
        f"queue proves a grant even when health reads fail ({res.detail})"
    )
    assert "limit 3000s" in res.detail, res.detail


def test_a_worker_granted_then_lost_is_stalled_not_reported_as_no_capacity(monkeypatch):
    # A worker granted once and then gone from health (permanent gap, job still IN_QUEUE) must stay
    # with the SETUP timer. Exempting the stall check on `worker_coming_up_at` alone did not: that
    # is a TTL'd sighting, so the gap re-entered the exemption forever, the stall check never ran,
    # and the queue timer -- rearmed by the same gap -- ran to the scaled capacity grace and
    # reported `no_capacity` for a GPU RunPod HAD granted.
    #
    # Both halves are wrong and the label is the worse one: `no_capacity` can trip the supervisor's
    # weight-cache drop on a run that never had a capacity problem. Pristine `dev` reports `stalled`
    # here, so this is a regression this PR must not introduce, not a behavior change.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    now = {"t": 0.0}

    def health(_eid, _fp, **_kw):
        # placed for the first stretch, then health never reports it again.
        return {"workers": {"initializing": 1} if now["t"] <= 600.0 else {}}

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    monkeypatch.setattr(runpod_api, "endpoint_health_for_fingerprint", health)
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    # 120s steps so every poll clears the 90s health-probe throttle.
    ticks = itertools.count(start=0, step=120.0)

    def tick():
        now["t"] = next(ticks)
        return now["t"]

    monkeypatch.setattr(polling.time, "time", tick)

    res = polling.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: None,
        deadline_at=500_000.0,
        **jobs.stall_kwargs(on_last_gpu=True, gpu_count=4),
    )
    assert res.failure == "stalled", (
        f"a granted-then-lost worker was reported as {res.failure!r}; past the first grant the "
        f"setup timer owns the wait, not the capacity timer ({res.detail})"
    )
    assert "limit 3000s" in res.detail, res.detail


def test_flapping_health_cannot_rearm_the_cold_start_budget_forever(monkeypatch):
    # The queued-wait stall exemption keys on `worker_coming_up_at`, which is a TTL'd SIGHTING and
    # therefore goes false again on any health gap. So endpoint health that alternates between a
    # placed worker and an empty snapshot re-entered the exemption on every gap and rolled
    # `last_progress` forward each time, while the placed polls in between disarmed the queue timer
    # -- neither timer could ever fire, and a wedged PAID worker ran to the run's outer wall
    # deadline. Latching the first grant bounds it at the setup grace instead.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    flip = {"n": 0}

    def health(_eid, _fp, **_kw):
        flip["n"] += 1
        return {"workers": {"initializing": 1} if flip["n"] % 2 == 0 else {}}

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    monkeypatch.setattr(runpod_api, "endpoint_health_for_fingerprint", health)
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    # 120s steps so every poll clears the 90s health-probe throttle and actually re-reads health.
    ticks = itertools.count(start=0, step=120.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(ticks))

    wall = 200_000.0
    res = polling.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: None,
        deadline_at=wall,
        **jobs.stall_kwargs(on_last_gpu=True, gpu_count=4),
    )
    assert res.failure == "stalled", res.detail
    # the wall deadline is the backstop, never the intended bound -- reaching it means both timers
    # were held off for the run's entire lifetime while a granted worker sat wedged and billing.
    assert "wall deadline" not in res.detail, (
        "flapping health held off both timers to the wall deadline; the first-grant latch is not "
        f"bounding the cold start ({res.detail})"
    )
    assert "limit 3000s" in res.detail, res.detail


def test_a_worker_that_is_coming_up_still_gets_the_unscaled_setup_grace(monkeypatch):
    # The capacity timer is suppressed once RunPod grants a worker, so from there the wait is a
    # cold start, not starvation -- and a cold start's budget has nothing to do with how scarce the
    # shape was to obtain. Deferring the stall timer for a PLACED worker would let a wide shape sit
    # through a wedged image pull for the full scaled grace instead of the setup grace.
    res = _queued_forever_scaled(
        monkeypatch,
        gpu_count=4,
        heartbeat_reader=lambda: None,
        workers={"initializing": 1},
    )
    assert res.failure == "stalled"
    assert "limit 3000s" in res.detail, res.detail


def test_a_job_outside_the_queue_still_stalls_on_its_own_limit(monkeypatch):
    # The queue deferral must be scoped to IN_QUEUE-with-no-worker. A running job that stops making
    # progress is genuinely stalled and must still be caught on the unscaled limit; disabling the
    # stall timer generally would let a wedged trainer burn the whole run deadline.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(
        runpod_api, "endpoint_health_for_fingerprint", lambda eid, _fp, **_kw: {"workers": {}}
    )
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=20.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    res = polling.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: None,
        **jobs.stall_kwargs(on_last_gpu=True, gpu_count=4),
    )
    assert res.failure == "stalled"
    assert "limit 3000s" in res.detail, res.detail


def test_single_card_capacity_grace_is_unchanged_by_the_scaling():
    # The whole change must be invisible to 1x runs, which are the overwhelming majority: an
    # absent, unusable, or explicitly-single count all multiply by exactly 1. A default that
    # silently scaled would lengthen every single-card run's failover.
    from flash.providers.runpod.execution import jobs

    baseline = jobs.stall_kwargs(on_last_gpu=True)
    assert baseline["queue_grace_s"] == 900.0
    for count in (1, 0, -3, None, True):
        # bool is not a card count: `True` would otherwise multiply by 1 by accident of int
        # subclassing rather than by rejection.
        assert jobs.stall_kwargs(on_last_gpu=True, gpu_count=count) == baseline, count


def test_reattach_poll_reproduces_the_multi_card_capacity_grace(monkeypatch):
    # A run adopted after a control-plane restart must wait on the same budget its submission used.
    # The count is read from the persisted EFFECTIVE worker spec (which submission stamped with the
    # count allocation resolved), not from the handle: _build_attach_context pops
    # `allocated_gpu_count` off the remote before the handle ever reaches the provider, so sourcing
    # it there would silently read 1 and halve a 2x run's grace on every recovery.
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core.base import JobHandle
    from flash.providers.runpod.execution import jobs as jobs
    from flash.providers.runpod.execution.provider import PROVIDER

    captured: dict = {}

    def fake_poll_job(handle, **kw):
        captured.update(kw)
        return jobs.PollResult(True, metrics={})

    monkeypatch.setattr(polling, "poll_job", fake_poll_job)

    spec = JobSpec(
        run_id="reattach-multi",
        model="Qwen/Qwen3.5-9B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=1, hf_repo=""),
        gpu=GpuSpec(type="B200", count=2),
    )
    handle = JobHandle.from_dict(
        {
            "provider": "runpod",
            "endpoint_id": "ep",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "j",
            "started_ts": 1.0,
            "on_last_gpu": True,
            "attempt": 1,
        }
    )
    PROVIDER.poll_attempt(handle, spec)
    assert captured["queue_grace_s"] == 1800.0
    assert captured["throttled_grace_s"] == 1800.0


def test_reattach_poll_reproduces_persisted_on_last_gpu(monkeypatch):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core.base import JobHandle
    from flash.providers.runpod.execution import jobs as jobs
    from flash.providers.runpod.execution.provider import PROVIDER

    captured: dict = {}

    def fake_poll_job(handle, **kw):
        captured.update(kw)
        return jobs.PollResult(True, metrics={})

    monkeypatch.setattr(polling, "poll_job", fake_poll_job)

    spec = JobSpec(
        run_id="reattach",
        model="Qwen/Qwen3.5-9B",
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

    PROVIDER.poll_attempt(JobHandle.from_dict({**base, "on_last_gpu": True, "attempt": 2}), spec)
    assert captured["queue_grace_s"] == 900.0
    assert captured["throttled_grace_s"] == 900.0
    assert captured["current_attempt"] == 2

    captured.clear()
    PROVIDER.poll_attempt(JobHandle.from_dict({**base, "on_last_gpu": False, "attempt": 0}), spec)
    assert captured["queue_grace_s"] == 300.0
    assert captured["current_attempt"] == 0

    with pytest.raises(ValueError, match="attempt identity is invalid"):
        PROVIDER.poll_attempt(JobHandle.from_dict(base), spec)


def test_submit_attempt_payload_carries_structured_source_snapshot(monkeypatch):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution, jobs

    spec = JobSpec(
        run_id="flash-source-snapshot",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(type=""),
    )
    submitted: dict = {}
    monkeypatch.setattr(
        job_execution,
        "deploy_train_endpoint",
        lambda *a, **k: ("ep", "endpoint-name", _RUNPOD_FINGERPRINT),
    )
    monkeypatch.setattr(
        runpod_api,
        "submit_job",
        lambda endpoint_id, payload, **_kw: (
            submitted.update({"endpoint_id": endpoint_id, "payload": payload}) or "job-1"
        ),
    )
    monkeypatch.setattr(polling, "poll_job", lambda *a, **k: jobs.PollResult(True, metrics={}))

    source_snapshot = valid_source_snapshot()
    assert job_execution.submit_attempt(
        spec,
        source_snapshot=source_snapshot,
        deadline_at=10_000_000_000.0,
    ).ok
    assert submitted["endpoint_id"] == "ep"
    assert submitted["payload"]["source_snapshot"] == source_snapshot
    assert "code_prefix" not in submitted["payload"]


def test_submit_attempt_rejects_malformed_source_before_deploy(monkeypatch):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.runpod.execution import job_execution
    from flash.snapshot.archive import SourceSnapshotError

    spec = JobSpec(
        run_id="flash-source-snapshot-invalid",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(type=""),
    )
    deploy_calls = []
    monkeypatch.setattr(
        job_execution,
        "deploy_train_endpoint",
        lambda *args, **kwargs: deploy_calls.append((args, kwargs)),
    )
    malformed = valid_source_snapshot()
    malformed["archive_path"] = "source/wrong/flash-source.zip"

    with pytest.raises(SourceSnapshotError, match="archive path"):
        job_execution.submit_attempt(
            spec,
            source_snapshot=malformed,
            deadline_at=10_000_000_000.0,
        )
    assert deploy_calls == []


def test_submit_attempt_polls_a_multi_card_shape_on_the_scaled_capacity_grace(monkeypatch):
    # The submitting process must apply the scaled budget too, not just a recovering one -- this is
    # the path that burned the queue time in the first place. The count is read off the effective
    # spec this attempt is launching, which allocation may have resolved to FEWER cards than the
    # run's ceiling named, so the wait matches what was actually rented.
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution, jobs

    captured: dict = {}
    monkeypatch.setattr(
        job_execution,
        "deploy_train_endpoint",
        lambda *a, **k: ("ep", "endpoint-name", _RUNPOD_FINGERPRINT),
    )
    monkeypatch.setattr(runpod_api, "submit_job", lambda *_a, **_kw: "job-1")
    monkeypatch.setattr(
        polling,
        "poll_job",
        lambda *a, **kw: captured.update(kw) or jobs.PollResult(True, metrics={}),
    )

    def _submit(count):
        captured.clear()
        spec = JobSpec(
            run_id="scaled-grace",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, hf_repo="org/repo"),
            gpu=GpuSpec(type="H200", count=count),
        )
        assert job_execution.submit_attempt(
            spec,
            on_last_gpu=True,
            source_snapshot=_SOURCE_SNAPSHOT,
            deadline_at=10_000_000_000.0,
        ).ok
        return captured["queue_grace_s"]

    assert _submit(4) == 3600.0
    # and a single-card run on the identical path is untouched.
    assert _submit(1) == 900.0


def test_runpod_submit_failure_is_retryable_only_after_confirmed_endpoint_deletion(monkeypatch):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution

    spec = JobSpec(
        run_id="runpod-submit-retryable",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(type=""),
    )
    original = RuntimeError("ambiguous queue post")
    handles = []
    deleted = []
    monkeypatch.setattr(provider_worker, "build_worker_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        job_execution,
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
        job_execution.submit_attempt(
            spec,
            attempt=4,
            on_handle=handles.append,
            source_snapshot=_SOURCE_SNAPSHOT,
            deadline_at=10_000_000_000.0,
        )

    assert caught.value is original
    assert deleted == ["endpoint"]
    assert handles == []


@pytest.mark.parametrize("deletion_mode", ["false", "exception"])
def test_runpod_submit_failure_persists_endpoint_only_cleanup_handle(monkeypatch, deletion_mode):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core.base import UnreconciledCreateError
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution

    spec = JobSpec(
        run_id=f"runpod-submit-unreconciled-{deletion_mode}",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(type=""),
    )
    handles = []
    monkeypatch.setattr(provider_worker, "build_worker_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        job_execution,
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
        job_execution.submit_attempt(
            spec,
            attempt=4,
            on_handle=handles.append,
            source_snapshot=_SOURCE_SNAPSHOT,
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
    from flash.providers.core.base import JobHandle
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs
    from flash.providers.runpod.execution.provider import RunpodProvider

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
    from flash.providers.core.base import JobHandle
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs
    from flash.providers.runpod.execution.provider import RunpodProvider

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
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core.base import JobHandle, PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution, jobs, polling
    from flash.providers.runpod.execution.provider import RunpodProvider

    spec = JobSpec(
        run_id="runpod-shared-deadline",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(type=""),
    )
    deadline_at = 12_345.0
    captured = []

    monkeypatch.setattr(job_execution.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        job_execution,
        "deploy_train_endpoint",
        lambda *a, **k: ("ep", "endpoint-name", _RUNPOD_FINGERPRINT),
    )
    monkeypatch.setattr(runpod_api, "submit_job", lambda *_a, **_k: "job-1")

    def fake_poll(handle, **kwargs):
        captured.append(kwargs["deadline_at"])
        return PollResult(True, metrics={})

    monkeypatch.setattr(polling, "poll_job", fake_poll)
    provider = RunpodProvider()
    assert provider.submit_attempt(
        spec,
        source_snapshot=_SOURCE_SNAPSHOT,
        _deadline_at=deadline_at,
    ).ok
    handle = JobHandle.from_dict(
        _runpod_handle_dict(
            jobs,
            endpoint_id="ep",
            endpoint_name="endpoint-name",
            job_id="job-1",
        )
    )
    assert provider.poll_attempt(handle, spec, _deadline_at=deadline_at).ok

    assert captured == [deadline_at, deadline_at]


def test_runpod_endpoint_time_consumption_blocks_queue_job_creation(monkeypatch):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution, polling

    spec = JobSpec(
        run_id="runpod-deadline-boundary",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(type=""),
    )
    now = {"value": 100.0}
    monkeypatch.setattr(polling.time, "time", lambda: now["value"])

    def _deploy(*_args, **_kwargs):
        now["value"] = 141.0
        return "ep", "endpoint-name", _RUNPOD_FINGERPRINT

    monkeypatch.setattr(job_execution, "deploy_train_endpoint", _deploy)
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
        job_execution.submit_attempt(spec, source_snapshot=_SOURCE_SNAPSHOT, deadline_at=200.0)

    assert submitted == []
    assert deleted == ["ep"]


def test_poll_job_in_queue_then_progress_does_not_false_stall(monkeypatch):
    # A job that leaves IN_QUEUE (a worker picks it up) must clear the queue timer: the later
    # IN_PROGRESS/COMPLETED path is governed by the heartbeat/setup windows, never by queue_grace_s.
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    ok = {"acc": 1.0}
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
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    # The IN_QUEUE phase triggers poll_job's endpoint_health probe on the first loop; stub it so the
    # test is hermetic (never hits the network even if RUNPOD_API_KEY is set in the environment).
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: {"workers": {"ready": 1, "running": 1}},
    )
    h = _live_clock_handle(jobs)
    res = polling.poll_job(h, interval_s=0, heartbeat_reader=lambda: None, queue_grace_s=900.0)
    assert res.ok


def test_poll_job_throttled_timer_resets_on_leaving_queue(monkeypatch):
    # A worker throttled in its FIRST queue window must not carry a stale arm-time across an
    # IN_PROGRESS spell: if RunPod re-queues the job (still throttled), the throttled grace must be
    # measured from the re-queue, not the original arm. Otherwise the first re-queue probe fires
    # no_capacity instantly, defeating throttled_grace_s. Clock advances one tick per job_status poll.

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    ok = {"acc": 1.0}
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
    monkeypatch.setattr(polling.time, "time", lambda: clock["t"])
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)

    res = polling.poll_job(
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

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    h = _runpod_handle(jobs)
    res = polling.poll_job(
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

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))

    hbs = iter([{"stage": "train", "step": 1, "ts": 1, "attempt": 0}])  # then StopIteration -> None

    def reader():
        return next(hbs, None)

    h = _runpod_handle(jobs)
    res = polling.poll_job(
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

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))

    # Every read returns attempt 0's leftover training heartbeat; this poll is attempt 1.
    leftover = {"stage": "train", "step": 5, "ts": 1, "attempt": 0}
    h = _runpod_handle(jobs, attempt=1)
    res = polling.poll_job(
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

    from flash.providers.artifacts import hf as hf_artifacts
    from flash.providers.core import base
    from flash.providers.runpod.execution.provider import RunpodProvider

    captured = {}

    def fake_poll_job(rh, **kw):
        captured["handle"] = rh
        captured["current_attempt"] = kw["current_attempt"]
        return base.PollResult(ok=True)

    def fake_failure_reader(*_args, attempt, **_kwargs):
        captured["failure_attempt"] = attempt
        return lambda: None

    monkeypatch.setattr(polling, "poll_job", fake_poll_job)
    monkeypatch.setattr(hf_artifacts, "make_hf_failure_detail_reader", fake_failure_reader)
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
    RunpodProvider().poll_attempt(handle, spec)
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
        RunpodProvider().poll_attempt(handle, spec)


@pytest.mark.parametrize("raw_attempt", ["", "x", None, True, -1, 1.5])
def test_reattach_rejects_explicit_malformed_persisted_attempt(monkeypatch, raw_attempt):
    import types

    from flash.providers.core import base
    from flash.providers.runpod.execution.provider import RunpodProvider

    monkeypatch.setattr(
        polling,
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
        RunpodProvider().poll_attempt(handle, spec)


def test_reattach_rejects_endpoint_only_handle(monkeypatch):
    import types

    from flash.providers.core import base
    from flash.providers.runpod.execution.provider import RunpodProvider

    monkeypatch.setattr(
        polling,
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
        RunpodProvider().poll_attempt(handle, spec)


def test_poll_job_setup_heartbeat_does_not_tighten(monkeypatch):
    # A cold-start (setup) heartbeat like "boot" proves liveness but must NOT switch to the
    # tight training window — the slow model-load/vLLM-init still has to fit setup_grace_s.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))

    hbs = iter([{"stage": "boot", "step": None, "ts": 1, "attempt": 0}])  # then None

    res = polling.poll_job(
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
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    state = {"t": 0.0}

    def _time():
        state["t"] += 100.0
        return state["t"]

    monkeypatch.setattr(polling.time, "time", _time)
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

    res = polling.poll_job(
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
    # an older heartbeat may land after a newer upload was skipped. only an advancing timestamp may
    # reset the stall window; stale content changes are no-ops.
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)

    def _stall_abs_time(second_hb):
        state = {"t": 0.0}

        def _time():
            state["t"] += 100.0
            return state["t"]

        monkeypatch.setattr(polling.time, "time", _time)
        seq = [{"stage": "train", "step": 5, "ts": 1000, "attempt": 0}]
        if second_hb is not None:
            seq.append(second_hb)
        hbs = iter(seq)
        res = polling.poll_job(
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


@pytest.mark.parametrize(
    ("last_hb_attempt", "last_hb_ts"),
    [(-1, 0.0), (0, 50.0)],
    ids=["first-current-attempt-heartbeat", "later-current-attempt-heartbeat"],
)
def test_older_heartbeat_cannot_regress_status_progress_anchor(
    monkeypatch, last_hb_attempt, last_hb_ts
):
    # a queue exemption or status transition can advance the shared progress anchor beyond the
    # worker timestamp. a heartbeat that becomes visible later may still advance heartbeat-specific
    # bookkeeping, but neither credit branch may move the stall anchor backward.
    from flash.providers.runpod.execution import polling

    heartbeat_key = ("boot", None, 100.0, 0)
    monkeypatch.setattr(
        polling,
        "surface_heartbeat",
        lambda _reader, _last_key, _say: (heartbeat_key, "boot"),
    )
    monkeypatch.setattr(polling.time, "time", lambda: 600.0)
    context = SimpleNamespace(
        heartbeat_reader=lambda: None,
        say=lambda _message: None,
        current_attempt=0,
        launch_ts=1.0,
    )
    state = SimpleNamespace(
        last_hb_key=None,
        last_hb_ts=last_hb_ts,
        last_hb_attempt=last_hb_attempt,
        last_progress=500.0,
        seen_training_hb=False,
        ever_saw_worker=True,
    )

    polling._update_heartbeat(context, state)

    assert state.last_progress == 500.0


def _stall_clock_at_giveup(
    monkeypatch, *, started_ts, heartbeats=(), step_s=100.0, clock_at=5000.0
):
    """Poll an ALREADY-PLACED job and return the absolute fake-clock time it gave up at.

    IN_PROGRESS from the first read, so the queued-capacity exemption never applies and the value
    seeded into ``last_progress`` is the only thing anchoring the stall window. ``clock_at`` is where
    this poll's wall clock starts; a ``started_ts`` well below it is a launch in the past, i.e. a
    delayed reattach. The handle rejects a non-positive launch, so the clock starts high rather than
    the launch going negative.
    """
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    state = {"t": clock_at}

    def _time():
        state["t"] += step_s
        return state["t"]

    monkeypatch.setattr(polling.time, "time", _time)
    hbs = iter(list(heartbeats))
    res = polling.poll_job(
        _runpod_handle(jobs, started_ts=started_ts),
        interval_s=0,
        heartbeat_reader=lambda: next(hbs, None),
        stall_after_s=150.0,
        setup_grace_s=600.0,
    )
    assert res.failure == "stalled", res
    return state["t"]


def test_a_reattached_wedged_worker_is_measured_from_launch_not_from_this_polls_start(monkeypatch):
    # A reattach has been BILLING since launch, so the stall window has to be measured from the
    # persisted launch timestamp. Seeding it from `time.time()` at poll start meant every reattach
    # handed a worker that was already wedged a complete fresh setup window, and a supervisor that
    # reattaches repeatedly could keep a dead worker on a paid GPU until the run's wall deadline.
    #
    # The job is IN_PROGRESS from the first read, so the queued-capacity exemption -- which re-anchors
    # `last_progress` on every pre-grant poll, and is what made the wrong seed survivable for a job
    # queued from the start -- never fires here. This is the placed-at-attach case where the seed is
    # the only anchor.
    #
    # Assert on the absolute clock at give-up, not on the "no worker progress for Ns" figure in the
    # detail: that figure is derived from `last_progress` and reads ~the same either way, so it
    # cannot tell the fixed and broken code apart.
    # attached at launch: the setup grace has not been spent, so the full window remains.
    at_launch = _stall_clock_at_giveup(monkeypatch, started_ts=5000.0)
    # same wedge, but launched 5000s before this poll attached: that grace is already gone.
    long_after_launch = _stall_clock_at_giveup(monkeypatch, started_ts=1.0)
    assert long_after_launch < at_launch, (
        f"a worker wedged since a launch 5000s ago was torn down at {long_after_launch}s, no sooner "
        f"than one attached at launch ({at_launch}s); the stall clock anchors on poll start, not launch"
    )


def test_a_heartbeat_written_before_the_reattach_buys_no_fresh_stall_window(monkeypatch):
    # Crediting progress at READ time means a heartbeat written long ago and surfaced now pays out a
    # full fresh window, so a worker that stopped producing heartbeats before we attached looks alive
    # for one more grace period. Credit the heartbeat's OWN ts (clamped to [launch, now]) instead:
    # it describes when the worker made progress, not when we got around to looking.
    #
    # Both runs see exactly ONE staged heartbeat, so the only difference is its timestamp.
    stale = _stall_clock_at_giveup(
        monkeypatch,
        started_ts=5000.0,
        heartbeats=[{"stage": "train", "step": 1, "ts": 5001.0, "attempt": 0}],
    )
    current = _stall_clock_at_giveup(
        monkeypatch,
        started_ts=5000.0,
        heartbeats=[{"stage": "train", "step": 1, "ts": 5400.0, "attempt": 0}],
    )
    assert stale < current, (
        f"a heartbeat stamped at launch (ts=5001.0) held the run to {stale}s, as long as one stamped "
        f"at ts=5400.0 ({current}s); progress is credited at read time, not at the heartbeat's own ts"
    )


def test_a_heartbeat_from_a_prior_attempt_buys_this_attempt_no_progress(monkeypatch):
    # A retry reuses the same heartbeat path, so a leftover heartbeat from attempt 0 is still
    # readable while attempt 1 is running. Crediting it would let the PREVIOUS attempt's work stand
    # as proof that THIS one is progressing. The attempt identity has to gate the credit, not just
    # the OOM flags.
    #
    # Both runs are attempt 1 and see exactly one heartbeat; only its `attempt` field differs. The
    # own-attempt run gives up SOONER, which is the tell: its heartbeat is a training heartbeat, so
    # it both moves the clock and tightens the window from setup grace to the stall limit. The
    # prior-attempt heartbeat must do neither, leaving the wider setup window running from launch.
    # Asserting "stale lasts longer" therefore proves the heartbeat was ignored end to end; the
    # reverse ordering would mean prior-attempt evidence had been credited.
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    def _giveup_with(hb_attempt):
        monkeypatch.setattr(
            runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"}
        )
        monkeypatch.setattr(polling.time, "sleep", lambda s: None)
        state = {"t": 5000.0}

        def _time():
            state["t"] += 100.0
            return state["t"]

        monkeypatch.setattr(polling.time, "time", _time)
        hbs = iter([{"stage": "train", "step": 1, "ts": 5400.0, "attempt": hb_attempt}])
        res = polling.poll_job(
            _runpod_handle(jobs, attempt=1, started_ts=5000.0),
            interval_s=0,
            heartbeat_reader=lambda: next(hbs, None),
            stall_after_s=150.0,
            setup_grace_s=600.0,
            current_attempt=1,
        )
        assert res.failure == "stalled", res
        return state["t"]

    stale_attempt = _giveup_with(0)
    this_attempt = _giveup_with(1)
    assert stale_attempt > this_attempt, (
        f"attempt 1 given only a stale attempt=0 heartbeat behaved identically to one given its "
        f"own ({stale_attempt}s vs {this_attempt}s); prior-attempt evidence is being credited"
    )


def test_poll_job_gapfill_step0_does_not_tighten(monkeypatch):
    # The train-liveness gap-filler emits rl_step/sft_step at step=0 throughout the silent FIRST step
    # (a cold vLLM rollout can run many minutes before global_step ticks to 1). That step=0 ping is a
    # NON-setup stage but reports NO completed step, so it proves liveness without meaning training has
    # started — it must keep the larger setup grace, not switch to the tight training window.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))

    hbs = iter(
        [{"stage": "rl_step", "step": 0, "ts": 1, "attempt": 0}]
    )  # gap-filler before the first real step
    res = polling.poll_job(
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
    monkeypatch.setattr(polling.time, "time", lambda: next(clock2))
    hbs2 = iter([{"stage": "rl_step", "step": 1, "ts": 1, "attempt": 0}])
    res2 = polling.poll_job(
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

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))

    hbs = iter(
        [{"stage": "rl_step", "step": "not-a-number", "ts": 1, "attempt": 0}]
    )  # malformed step
    res = polling.poll_job(
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
    # attempts share one heartbeat path. a newer timestamp from an older attempt is still stale and
    # must not reset the current attempt's stall window.
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)

    def _stall_abs_time(second_hb):
        state = {"t": 0.0}

        def _time():
            state["t"] += 100.0
            return state["t"]

        monkeypatch.setattr(polling.time, "time", _time)
        # First heartbeat is attempt 1 at ts=1000 (training started under the current attempt).
        seq = [{"stage": "train", "step": 5, "ts": 1000, "attempt": 1}]
        if second_hb is not None:
            seq.append(second_hb)
        hbs = iter(seq)
        res = polling.poll_job(
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

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: {
            "workers": {"unhealthy": 1, "running": 0, "ready": 0, "idle": 0, "initializing": 0}
        },
    )
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    res = polling.poll_job(
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
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    statuses = iter(
        [
            {"status": "IN_QUEUE"},  # probe 1: unhealthy -> arm unhealthy_since
            {"status": "IN_QUEUE"},  # probe 2: usable worker -> reset (no fail)
            {"status": "IN_PROGRESS"},
            {
                "status": "COMPLETED",
                "output": {"acc": 1.0},
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
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    res = polling.poll_job(
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

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    # poll_job is a pure function of its args (throttled_grace_s is passed below), so no env setup.
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: {
            "workers": {"throttled": 1, "running": 0, "ready": 0, "idle": 0, "initializing": 0}
        },
    )
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    res = polling.poll_job(
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
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    statuses = iter(
        [
            {"status": "IN_QUEUE"},  # probe 1: throttled -> arm throttled_since
            {"status": "IN_QUEUE"},  # probe 2: usable worker -> reset (no fail)
            {"status": "IN_PROGRESS"},
            {
                "status": "COMPLETED",
                "output": {"acc": 1.0},
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
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    res = polling.poll_job(
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

    from flash.providers.artifacts.hf import make_hf_failure_detail_reader

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
    from flash.providers.artifacts import hf as _hf_artifacts

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


def test_failure_detail_reader_reads_only_the_current_attempt_console(monkeypatch):
    from flash.providers.artifacts import hf as _hf_artifacts

    requested: list[str] = []

    def fake_reader(_hf_repo, path_in_repo, _min_interval_s):
        requested.append(path_in_repo)
        return lambda force=False: (
            "last live bytes" if path_in_repo.endswith("console_sft_attempt2.txt") else None
        )

    monkeypatch.setattr(_hf_artifacts, "make_hf_text_reader", fake_reader)
    reader = _hf_artifacts.make_hf_failure_detail_reader(
        "org/repo", "sft/run-1/seed0", "sft", attempt=2
    )

    detail = reader(force=True)

    assert requested[-2:] == [
        "sft/run-1/seed0/console_sft.txt",
        "sft/run-1/seed0/console_sft_attempt2.txt",
    ]
    assert "--- console_sft_attempt2.txt ---" in detail
    assert "last live bytes" in detail


def test_poll_job_no_reader_keeps_tight_window(monkeypatch):
    # Without a heartbeat_reader we can't tell setup from training, so the larger
    # setup_grace must NOT silently slow stall detection — stay on stall_after_s.
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_PROGRESS"})
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    res = polling.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=None,
        stall_after_s=150.0,
        setup_grace_s=5000.0,
    )
    assert res.failure == "stalled"
    assert "limit 150s" in res.detail


def test_poll_job_tolerates_transient_api_errors(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    ok = {"acc": 0.7}
    calls = {"n": 0}

    def flaky(eid, jid, **_kw):
        calls["n"] += 1
        if calls["n"] < 4:
            raise runpod_api.RunpodApiError("blip")
        return {"status": "COMPLETED", "output": ok}

    monkeypatch.setattr(runpod_api, "job_status", flaky)
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    res = polling.poll_job(
        _runpod_handle(jobs, endpoint_name="n", job_id="j"), interval_s=0, stall_after_s=1e9
    )
    assert res.ok
    assert calls["n"] == 4


# ---------------------------------------------------------------------------
# Supervisor retry logic (runner) with mocked job submit
# ---------------------------------------------------------------------------
def _fresh_orchestrator(tmp, monkeypatch):
    monkeypatch.setattr(runner_state, "RUNS_DIR", os.path.join(str(tmp), "runs"))
    monkeypatch.setattr(runner_state, "RESULTS_DIR", os.path.join(str(tmp), "results"))
    import flash.adapters.lora_rank as rank_mod
    from flash.providers.runpod.client import api as runpod_api

    monkeypatch.setattr(
        runpod_api, "delete_endpoint_for_fingerprint", lambda endpoint_id, _fingerprint: True
    )
    monkeypatch.setattr(
        provider_worker, "publish_source_snapshot", lambda _repo=None: _SOURCE_SNAPSHOT
    )
    monkeypatch.setattr(
        runner_status,
        "validate_terminal_source_metrics",
        lambda _status, metrics, expected_attempt=None: (dict(metrics), expected_attempt),
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
    monkeypatch.setattr(runner_artifacts, "stage_environment_package", lambda spec, **_kwargs: spec)


def _confirm_runpod_retry_teardown(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api

    monkeypatch.setattr(
        runpod_api,
        "cancel_job",
        lambda _endpoint_id, job_id, **_kw: {"id": job_id, "status": "CANCELLED"},
    )
    monkeypatch.setattr(
        runpod_api, "delete_endpoint_for_fingerprint", lambda _endpoint_id, _fingerprint: True
    )


def _spec(run_id):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec

    return JobSpec(
        run_id=run_id,
        model="Qwen/Qwen3.5-9B",
        algorithm="grpo",
        train=TrainSpec(epochs=1, max_examples=1),
        gpu=GpuSpec(type="", max_retries=2),
    )


def test_unstructured_prepare_does_not_import_serving_preflight(monkeypatch):
    import builtins

    class ReachedModelResolution(Exception):
        pass

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "flash.serve.deployment.preflight":
            pytest.fail("unstructured preparation imported optional serving dependencies")
        return original_import(name, *args, **kwargs)

    def stop_at_model_resolution(*_args, **_kwargs):
        raise ReachedModelResolution

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    monkeypatch.setattr(catalog, "resolve_model", stop_at_model_resolution)

    with pytest.raises(ReachedModelResolution):
        runner_submit.prepare_job(_spec("base-install-dry-run"))


def _adapter_config(*, rank=32, alpha=64):
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "Qwen/Qwen3.5-9B",
        "r": rank,
        "lora_alpha": alpha,
        # peft>=0.19 writes this on every save, so a config without it is not a shape any supported
        # writer produces -- and submit rejects an unmarked adapter outright. these cases are about
        # ref/rank/pin handling, so they need the marker a real source adapter carries.
        "exclude_modules": None,
    }


@pytest.mark.parametrize("cancel_during_status", [False, True])
def test_supervisor_adopts_runpod_completion_before_retry(monkeypatch, cancel_during_status):
    from dataclasses import replace

    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.core.allocator as allocator
        import flash.runner.supervise.lifecycle as lifecycle
        from flash.providers.core import registry as providers
        from flash.providers.core.base import Allocation, Candidate, PollResult
        from flash.providers.runpod.client import api as runpod_api

        spec = _spec("completed-before-retry")
        spec = replace(spec, gpu=replace(spec.gpu, max_retries=0))
        runner_state._save_status(
            runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()),
            _next_attempt=0,
        )
        candidate = Candidate("runpod", "RTX 4090", 0.5, 24)
        monkeypatch.setattr(
            allocator,
            "allocate",
            lambda *args, **kwargs: Allocation(
                provider="runpod",
                gpu="RTX 4090",
                hourly_usd=0.5,
                min_vram_gb=24,
                candidates=(candidate,),
            ),
        )
        calls = {"n": 0}

        class Provider:
            supports_weight_cache = False

            def submit_attempt(self, spec, log=None, on_handle=None, attempt=0, **_):
                calls["n"] += 1
                if on_handle:
                    on_handle(
                        {
                            "provider": "runpod",
                            "endpoint_id": "ep-completed",
                            "endpoint_name": "completed",
                            "key_fingerprint": _RUNPOD_FINGERPRINT,
                            "job_id": "job-completed",
                            "attempt": attempt,
                            "started_ts": 1.0,
                        }
                    )
                return PollResult(False, failure="poll_error", detail="provider api outage")

            def cancel(self, _handle):
                return None

            def destroy(self, _handle):
                return None

            def gc(self, _spec):
                return None

        monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())

        def job_status(*_args, **_kwargs):
            if cancel_during_status:
                runner_status._update(spec.run_id, "cancelled")
            return {
                "status": "COMPLETED",
                "output": {"wall_seconds": 5.0, "trained_eval_acc": 0.9},
            }

        monkeypatch.setattr(runpod_api, "job_status", job_status)

        if cancel_during_status:
            with pytest.raises(runner_errors._RunCancelled):
                lifecycle._run_attempts_supervised(
                    spec,
                    io.StringIO(),
                    source_snapshot=_SOURCE_SNAPSHOT,
                )
        else:
            metrics = lifecycle._run_attempts_supervised(
                spec,
                io.StringIO(),
                source_snapshot=_SOURCE_SNAPSHOT,
            )
            assert metrics["trained_eval_acc"] == 0.9
        assert calls["n"] == 1


def test_supervisor_retries_on_stall_then_succeeds(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        calls = {"n": 0}
        source_snapshots: list[dict | None] = []

        def fake_submit(spec, log=None, on_handle=None, attempt=0, source_snapshot=None, **_):
            calls["n"] += 1
            source_snapshots.append(source_snapshot)
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

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        runner_submit.submit_job(_spec("retry-ok"), dry_run=False, background=False)
        st = runner_status.get_status("retry-ok")
        assert st.state == "done"
        assert calls["n"] == 2
        assert source_snapshots == [_SOURCE_SNAPSHOT, _SOURCE_SNAPSHOT]
        assert st.remote is None
        assert st.cleanup_confirmed_remote["job_id"] == "j2"


def test_submit_keeps_public_short_init_ref_but_launches_storage_ref(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.adapters.lora_rank as rank_mod
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train
        import flash.runner.results.checkpoints as checkpoints
        from flash.core.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "train": {
                    "epochs": 1,
                    "max_examples": 1,
                    "hf_repo": "Freesolo-Co/flashrun-source-env",
                },
            }
        )
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="source-run",
                state="done",
                spec=source.to_dict(),
                source_snapshot=_SOURCE_SNAPSHOT,
                effective_preparation={"worker_spec": source.to_internal_dict()},
            )
        )
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
                # run_id is platform-managed and stripped from to_dict(); restore it so the child
                # submits under "warm-run" instead of the from_dict "local" default.
                "run_id": "warm-run",
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

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(rank_mod, "load_hf_adapter_config", lambda *a, **k: _adapter_config())
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        runner_submit.submit_job(spec, dry_run=False, background=False)

        st = runner_status.get_status("warm-run")
        assert st.spec["train"]["init_from_adapter"] == "source-run/step-40"
        # lora_rank and lora_alpha are not authorable for warm-start: the child inherits the
        # source adapter's rank and alpha, so both are stripped from the public spec. the
        # authoritative resolved values live in the worker spec — what the worker actually trains
        # with, and what the launch captures below.
        assert "lora_rank" not in st.spec["train"]
        assert "lora_alpha" not in st.spec["train"]
        worker_train = st.effective_preparation["worker_spec"]["train"]
        assert (worker_train["lora_rank"], worker_train["lora_alpha"]) == (32, 64)
        assert (
            launched["init_from_adapter"]
            == "Freesolo-Co/flashrun-source-env:rl/source-run/checkpoints/step-40"
        )
        assert (launched["lora_rank"], launched["lora_alpha"]) == (32, 64)


def test_submit_rejects_cross_org_init_ref(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        from flash.core.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="source-run",
                state="done",
                spec=source.to_dict(),
                source_snapshot=_SOURCE_SNAPSHOT,
                effective_preparation={"worker_spec": source.to_internal_dict()},
                billing_context={"org_id": "org-a"},
            )
        )
        base = _spec("warm-run").to_dict()
        spec = JobSpec.from_dict(
            {
                **base,
                "train": {**base["train"], "init_from_adapter": "source-run/final"},
            }
        )

        with pytest.raises(ValueError, match="same Freesolo org"):
            runner_submit.submit_job(
                spec,
                dry_run=True,
                background=False,
                billing_context={"org_id": "org-b"},
            )


def test_submit_allows_missing_source_org_when_same_owner_key(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        from flash.core.spec import JobSpec
        from flash.server.platform import db

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="source-run",
                state="done",
                spec=source.to_dict(),
                source_snapshot=_SOURCE_SNAPSHOT,
                effective_preparation={"worker_spec": source.to_internal_dict()},
            )
        )
        import flash.adapters.lora_rank as rank_mod
        import flash.runner.results.checkpoints as checkpoints

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
                "train": {**base["train"], "init_from_adapter": "source-run/final"},
            }
        )

        status = runner_submit.submit_job(
            spec,
            dry_run=True,
            background=False,
            billing_context={"org_id": "org-a"},
            owner_key_id=7,
        )

        assert status.state == "dry_run"


def test_submit_dry_run_omits_public_warmstart_rank_and_resolves_alpha(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        from flash.core.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="source-run",
                state="done",
                spec=source.to_dict(),
                source_snapshot=_SOURCE_SNAPSHOT,
                effective_preparation={"worker_spec": source.to_internal_dict()},
            )
        )
        import flash.adapters.lora_rank as rank_mod
        import flash.runner.results.checkpoints as checkpoints

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
                    "init_from_adapter": "source-run/final",
                    "lora_rank": 8,
                    "lora_alpha": 16,
                },
            }
        )

        status = runner_submit.submit_job(spec, dry_run=True, background=False)

        assert status.state == "dry_run"
        assert "lora_rank" not in status.spec["train"]
        assert "lora_alpha" not in status.spec["train"]
        assert "init_from_adapter_revision" not in status.spec["train"]
        # lora_rank/lora_alpha are not authorable for a warm start and are stripped from the public
        # spec; the dry-run still resolves the source adapter's authoritative rank and alpha into
        # the worker spec, so preflight and execution report the same effective spec.
        worker_train = status.effective_preparation["worker_spec"]["train"]
        assert (worker_train["lora_rank"], worker_train["lora_alpha"]) == (64, 128)
        assert status.to_dict()["spec"] == status.spec


def test_submit_rejects_final_checkpoint_from_unfinished_source_run(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        from flash.core.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="source-run",
                state="running",
                spec=source.to_dict(),
                source_snapshot=_SOURCE_SNAPSHOT,
                effective_preparation={"worker_spec": source.to_internal_dict()},
            )
        )
        base = _spec("warm-run").to_dict()
        spec = JobSpec.from_dict(
            {
                **base,
                "train": {**base["train"], "init_from_adapter": "source-run/final"},
            }
        )

        with pytest.raises(ValueError, match="concrete source-run/step-N checkpoint"):
            runner_submit.submit_job(spec, dry_run=True, background=False)


def test_submit_rejects_final_checkpoint_without_adapter_artifact(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.adapters.lora_rank as rank_mod
        import flash.runner.results.checkpoints as checkpoints
        from flash.core.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="source-run",
                state="done",
                spec=source.to_dict(),
                source_snapshot=_SOURCE_SNAPSHOT,
                effective_preparation={"worker_spec": source.to_internal_dict()},
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
                "train": {**base["train"], "init_from_adapter": "source-run/final"},
            }
        )

        with pytest.raises(ValueError, match="complete adapter artifact was not found"):
            runner_submit.submit_job(spec, dry_run=True, background=False)


def test_submit_rejects_missing_source_org_without_same_owner_key(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        from flash.core.spec import JobSpec
        from flash.server.platform import db

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="source-run",
                state="done",
                spec=source.to_dict(),
                source_snapshot=_SOURCE_SNAPSHOT,
                effective_preparation={"worker_spec": source.to_internal_dict()},
            )
        )
        monkeypatch.setattr(db, "run_owner", lambda run_id: 8 if run_id == "source-run" else None)
        base = _spec("warm-run").to_dict()
        spec = JobSpec.from_dict(
            {
                **base,
                "train": {**base["train"], "init_from_adapter": "source-run/final"},
            }
        )

        with pytest.raises(ValueError, match="same Freesolo org"):
            runner_submit.submit_job(
                spec,
                dry_run=True,
                background=False,
                billing_context={"org_id": "org-a"},
                owner_key_id=7,
            )


def test_submit_rejects_missing_init_checkpoint_step(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.adapters.lora_rank as rank_mod
        import flash.runner.results.checkpoints as checkpoints
        from flash.core.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="source-run",
                state="done",
                spec=source.to_dict(),
                source_snapshot=_SOURCE_SNAPSHOT,
                effective_preparation={"worker_spec": source.to_internal_dict()},
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
            runner_submit.submit_job(
                spec,
                dry_run=True,
                background=False,
                billing_context={"org_id": "org-a"},
            )


def test_submit_surfaces_checkpoint_listing_error_before_launch(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.adapters.lora_rank as rank_mod
        import flash.runner.results.checkpoints as checkpoints
        from flash.core.spec import JobSpec

        source = JobSpec.from_dict(
            {
                "run_id": "source-run",
                "model": "Qwen/Qwen3.5-9B",
                "algorithm": "grpo",
                "train": {"epochs": 1, "max_examples": 1, "hf_repo": "Freesolo-Co/source"},
            }
        )
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="source-run",
                state="done",
                spec=source.to_dict(),
                source_snapshot=_SOURCE_SNAPSHOT,
                effective_preparation={"worker_spec": source.to_internal_dict()},
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
            runner_submit.submit_job(
                spec,
                dry_run=True,
                background=False,
                billing_context={"org_id": "org-a"},
            )


def test_attach_polls_live_warmstart_handle_without_source_revalidation(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.adapters.lora_rank as rank_mod
        from flash.core.spec import JobSpec
        from flash.providers.core import registry as providers
        from flash.providers.core.base import PollResult

        base = _spec("warm-recover").to_dict()
        public_spec = JobSpec.from_dict(
            {
                **base,
                # run_id is platform-managed and stripped from to_dict(); restore it so the public
                # and worker specs stay keyed to the persisted status ("warm-recover"), not the
                # from_dict "local" default.
                "run_id": "warm-recover",
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
        runner_state._save_status(
            runner_state.RunStatus(
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
                    "launch_claim_token": "claim-0",
                    "started_ts": 1.0,
                },
                source_snapshot=_SOURCE_SNAPSHOT,
                effective_preparation={
                    "worker_spec": worker_spec.to_internal_dict(),
                    "adapter_identity": identity.to_dict(),
                    "version": 1,
                    "preparation_digest": runner_preparation._preparation_digest(
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
            def poll_attempt(self, handle, spec, **kwargs):
                polled.update(
                    init_from_adapter=spec.train.init_from_adapter,
                    revision=spec.train.init_from_adapter_revision,
                    lora_rank=spec.train.lora_rank,
                )
                return PollResult(True, metrics={"wall_seconds": 1.0})

        monkeypatch.setattr(providers, "get_provider", lambda name: Provider())
        monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda *a, **k: None)

        status = runner_attach.attach_run("warm-recover", log_stream=sys.stderr)

        assert status.state == "done"
        assert polled == {
            "init_from_adapter": "Freesolo-Co/source:rl/source-run/checkpoints/step-40",
            "revision": "a" * 40,
            "lora_rank": 64,
        }


def test_attach_reuses_verified_effective_snapshot_before_recovery_launch(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.adapters.lora_rank as rank_mod
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train
        from flash.core.spec import JobSpec

        base = _spec("warm-recover").to_dict()
        public_spec = JobSpec.from_dict(
            {
                **base,
                # run_id is platform-managed and stripped from to_dict(); restore it so the specs
                # stay keyed to the persisted status rather than the from_dict "local" default.
                "run_id": "warm-recover",
                "train": {
                    **base["train"],
                    "init_from_adapter": "source-run/step-40",
                    "lora_rank": 8,
                },
            }
        )
        # the worker spec is the complete internal carrier (run_id, hf_repo, ... retained), so build
        # it from to_internal_dict() -- to_dict() would strip the managed fields recovery reads back.
        worker_dict = public_spec.to_internal_dict()
        worker_dict["train"] = {
            **worker_dict["train"],
            "init_from_adapter": "Freesolo-Co/source:rl/source-run/checkpoints/step-40",
            "init_from_adapter_revision": "a" * 40,
            # the resolved warm-start topology carries the source adapter's rank AND its derived
            # alpha together (runner sets both from the source metadata); source revalidation checks
            # both, so the worker carrier must pin lora_alpha=64 to match _adapter_config(rank=64).
            "lora_rank": 64,
            "lora_alpha": 64,
        }
        worker_spec = JobSpec.from_dict(worker_dict)
        identity = rank_mod.AdapterArtifactIdentity(
            "digest-v1", "config-v1", "adapter_model.safetensors", "weights-v1:123"
        )
        runner_state._save_status(
            runner_state.RunStatus(
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
                    "launch_claim_token": "claim-0",
                    "started_ts": 1.0,
                    "allocated_gpu": "RTX 4090",
                    "allocated_gpu_count": 1,
                    "allocated_usable_vram_gb": 24.0,
                },
                source_snapshot=_SOURCE_SNAPSHOT,
                effective_preparation={
                    "worker_spec": worker_spec.to_internal_dict(),
                    "adapter_identity": identity.to_dict(),
                    "version": 1,
                    "preparation_digest": runner_preparation._preparation_digest(
                        public_spec, worker_spec, identity.to_dict()
                    ),
                },
            ),
            _next_attempt=1,
        )
        monkeypatch.setattr(
            rank_mod, "load_hf_adapter_config", lambda *a, **k: _adapter_config(rank=64)
        )
        monkeypatch.setattr(rank_mod, "adapter_artifact_identity", lambda *a, **k: identity)
        launched: dict[str, object] = {}
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="stalled", detail="stalled"),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda *a, **k: None)
        monkeypatch.setattr(
            runner_lifecycle,
            "_run_training",
            lambda spec, *a, **k: launched.update(
                init_from_adapter=spec.train.init_from_adapter,
                init_from_adapter_revision=spec.train.init_from_adapter_revision,
                lora_rank=spec.train.lora_rank,
            ),
        )

        runner_attach.attach_run("warm-recover", log_stream=sys.stderr)

        assert launched == {
            "init_from_adapter": "Freesolo-Co/source:rl/source-run/checkpoints/step-40",
            "init_from_adapter_revision": "a" * 40,
            "lora_rank": 64,
        }


def test_attach_revalidates_source_before_handleless_resubmission(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.adapters.lora_rank as rank_mod
        import flash.providers.runpod.execution.jobs as jobs
        from flash.core.spec import JobSpec

        base = _spec("warm-recover").to_dict()
        public_spec = JobSpec.from_dict(
            {
                **base,
                # run_id is platform-managed and stripped from to_dict(); restore it so the specs
                # stay keyed to the persisted status rather than the from_dict "local" default.
                "run_id": "warm-recover",
                "train": {**base["train"], "init_from_adapter": "source-run/step-40"},
            }
        )
        # the worker spec is the complete internal carrier (run_id, hf_repo, ... retained), so build
        # it from to_internal_dict() -- to_dict() would strip the managed fields recovery reads back.
        worker_dict = public_spec.to_internal_dict()
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
        runner_state._save_status(
            runner_state.RunStatus(
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
                    "launch_claim_token": "claim-0",
                    "started_ts": 1.0,
                    "allocated_gpu": "RTX 4090",
                    "allocated_gpu_count": 1,
                    "allocated_usable_vram_gb": 24.0,
                },
                source_snapshot=_SOURCE_SNAPSHOT,
                effective_preparation={
                    "worker_spec": worker_dict,
                    "adapter_identity": original_identity,
                    "version": 1,
                    "preparation_digest": runner_preparation._preparation_digest(
                        public_spec, worker_spec, original_identity
                    ),
                },
            ),
            _next_attempt=1,
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
            polling,
            "poll_job",
            lambda *a, **k: (
                polls.append("polled")
                or jobs.PollResult(False, failure="stalled", detail="stalled")
            ),
        )
        monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda *a, **k: None)
        monkeypatch.setattr(
            runner_lifecycle,
            "_run_training",
            lambda *a, **k: pytest.fail("drifted source must block resubmission"),
        )

        status = runner_attach.attach_run("warm-recover", log_stream=sys.stderr)

        assert polls == ["polled"]
        assert status.state == "failed"
        assert "source-run/step-40" in (status.error or "")
        assert "changed after submission" in (status.error or "")
        assert "private-owner" not in (status.error or "")


def test_attach_legacy_warmstart_without_snapshot_fails_closed(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        from flash.core.spec import JobSpec

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
            "launch_claim_token": "claim-0",
            "started_ts": 1.0,
        }
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="warm-recover",
                state="running",
                spec=spec.to_dict(),
                remote=remote,
            )
        )
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: pytest.fail("legacy warm start must fail before provider polling"),
        )
        monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda *a, **k: None)

        status = runner_attach.attach_run("warm-recover", log_stream=sys.stderr)

        assert status.state == "failed"
        assert status.remote == remote
        assert "original preparation snapshot is unavailable" in (status.error or "")
        assert "source-run/step-40" in (status.error or "")


def test_attach_setup_failure_does_not_overwrite_concurrent_cancel(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        remote = {
            "provider": "runpod",
            "endpoint_id": "ep",
            "endpoint_name": "name",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "job",
            "attempt": 0,
            "launch_claim_token": "claim-0",
            "started_ts": 1.0,
        }
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="cancelled-attach",
                state="running",
                spec=_spec("cancelled-attach").to_dict(),
                remote=remote,
            )
        )
        monkeypatch.setattr(
            runner_status,
            "effective_spec_from_status",
            lambda _status: (_ for _ in ()).throw(RuntimeError("setup failed")),
        )
        real_record = runner_reconciliation._record_cleanup_remote

        def cancel_then_record(run_id, cleanup_remote):
            assert runner_status._update(run_id, "cancelled", remote=None)
            return real_record(run_id, cleanup_remote)

        monkeypatch.setattr(runner_reconciliation, "_record_cleanup_remote", cancel_then_record)
        monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda *_args, **_kwargs: None)

        status = runner_attach.attach_run("cancelled-attach", log_stream=sys.stderr)

        assert status.state == "cancelled"
        assert status.remote is None
        assert runner_status._load_status_json("cancelled-attach")[
            runner_state._CLEANUP_REMOTES_KEY
        ] == [{key: value for key, value in remote.items() if key != "launch_claim_token"}]


def test_attach_setup_failure_does_not_steal_precommit_cancel(monkeypatch):
    # a cancel that has cleared status.remote but not yet flipped state to terminal must win: the
    # attach-setup-failure cas no-ops (the identifiable remote no longer matches) and must NOT
    # force-fail the run, else a user cancel gets recorded as failed.
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        remote = {
            "provider": "runpod",
            "endpoint_id": "ep",
            "endpoint_name": "name",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "job",
            "attempt": 0,
            "launch_claim_token": "claim-0",
            "started_ts": 1.0,
        }
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="precancel-attach",
                state="running",
                spec=_spec("precancel-attach").to_dict(),
                remote=remote,
            )
        )
        monkeypatch.setattr(
            runner_status,
            "effective_spec_from_status",
            lambda _status: (_ for _ in ()).throw(RuntimeError("setup failed")),
        )
        real_record = runner_reconciliation._record_cleanup_remote

        def clear_remote_then_record(run_id, cleanup_remote):
            # a concurrent cancel cleared the remote but has NOT yet flipped state to cancelled
            assert runner_reconciliation._compare_and_clear_remote(run_id, remote)
            return real_record(run_id, cleanup_remote)

        monkeypatch.setattr(
            runner_reconciliation, "_record_cleanup_remote", clear_remote_then_record
        )
        monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda *_args, **_kwargs: None)

        status = runner_attach.attach_run("precancel-attach", log_stream=sys.stderr)

        assert status.state != "failed", "attach-setup-failure stole a pre-commit cancel"


def test_cancel_during_attempt_reaps_walked_endpoint(monkeypatch):
    """A cancel landing mid-attempt raised _RunCancelled straight out of the retry loop, skipping
    _gc_seen_endpoints — leaking a walk-provisioned endpoint (one _gc_run_endpoints can't name, whose
    `running` write lost the terminal-stickiness race so it's absent from status.remote). The cancel
    path must now reap seen_endpoints."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train
        from flash.providers.runpod.client import api as runpod_api

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

        def fake_submit(spec, log=None, on_handle=None, attempt=0, **_):
            runner_status._update(spec.run_id, "cancelled")  # cancel lands during provisioning
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

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        runner_submit.submit_job(_spec("cancel-reap"), dry_run=False, background=False)

        assert runner_status.get_status("cancel-reap").state == "cancelled"
        assert (
            runner_status.get_status("cancel-reap").remote is None
        )  # handle write lost the stickiness race
        assert "epWALK" in deleted  # the walked endpoint was reaped on the cancel path


def test_supervisor_retries_runpod_cancelled_then_succeeds(monkeypatch):
    # A "job_preempted" first attempt retries on a fresh endpoint and completes.
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        calls = {"n": 0}

        def fake_submit(spec, log=None, on_handle=None, attempt=0, **_):
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

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        runner_submit.submit_job(_spec("cancel-retry"), dry_run=False, background=False)
        assert runner_status.get_status("cancel-retry").state == "done"
        assert calls["n"] == 2


def test_supervisor_does_not_retry_worker_code_errors(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        calls = {"n": 0}

        def fake_submit(spec, log=None, on_handle=None, attempt=0, **_):
            calls["n"] += 1
            return jobs.PollResult(
                False, failure="job_failed", detail="Remote execution failed: ValueError"
            )

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        with pytest.raises(RuntimeError):
            runner_submit.submit_job(_spec("fail-fast"), dry_run=False, background=False)
        assert calls["n"] == 1
        assert runner_status.get_status("fail-fast").state == "failed"


def test_supervisor_infra_floor_respects_explicit_zero_retries(monkeypatch):
    """An explicit max_retries=0 (deliberate single-shot) is NOT forced to retry by the infra floor."""
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec

    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        calls = {"n": 0}

        def fake_submit(spec, log=None, on_handle=None, attempt=0, **_):
            calls["n"] += 1
            return jobs.PollResult(False, failure="stalled", detail="frozen")

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        spec = JobSpec(
            run_id="no-retry",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=0),
        )
        with pytest.raises(RuntimeError):
            runner_submit.submit_job(spec, dry_run=False, background=False)
        assert calls["n"] == 1  # single shot — floor does not apply at max_retries=0
        assert runner_status.get_status("no-retry").state == "failed"


@pytest.mark.parametrize("failure", ["no_capacity", "poll_error"])
def test_shared_cache_zero_retry_budget_submits_exactly_once(monkeypatch, failure):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec

    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        submissions = []

        def fake_submit(spec, log=None, on_handle=None, attempt=0, on_last_gpu=False, **_):
            submissions.append((attempt, spec.gpu.network_volume, on_last_gpu))
            return jobs.PollResult(False, failure=failure, detail="cache-constrained failure")

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        spec = JobSpec(
            run_id=f"cache-zero-{failure}",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=0),
        )

        with pytest.raises(RuntimeError):
            runner_submit.submit_job(spec, dry_run=False, background=False)

        assert submissions == [(0, runner_weight_cache.WEIGHT_CACHE_VOLUME_NAME, True)]


def test_supervisor_walks_to_next_gpu_class_on_infra_retry(monkeypatch):
    # a managed gpu request that keeps hitting infra-shaped failures must walk
    # down the ranked candidate list, not burn every retry on the same capacity-starved
    # class. with static rates the validated >=24 gb pool for a 0.8b grpo run ranks
    # rtx 4090 < rtx 5090 < a100 pcie < ... by $/hr, so successive attempts step through them.
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train
        from flash.core.spec import GpuSpec, JobSpec, TrainSpec

        gpus_seen: list[str] = []

        def fake_submit(spec, log=None, on_handle=None, attempt=0, **_):
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

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="walk",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=2),
        )
        runner_submit.submit_job(spec, dry_run=False, background=False)

        assert runner_status.get_status("walk").state == "done"
        # Three attempts, three distinct classes, each at least as expensive as the last.
        assert len(gpus_seen) == 3
        assert len(set(gpus_seen)) == 3
        # Escalation is ordered by what the allocator ranks on -- dollars per optimizer step, not
        # the hourly rate. The two agree only when the step is compute-bound. This spec retains one
        # prompt, so the step is latency-bound and a faster card can finish it for less despite a
        # higher hourly rate; asserting sorted hourly rates would pin the wrong invariant.
        from flash.providers.core.base import GPU_INFO, _run_cost_key

        cost_key = _run_cost_key("Qwen/Qwen3.5-9B", "grpo", train={"epochs": 1, "max_examples": 1})
        step_costs = [cost_key(gpu, GPU_INFO[gpu].hourly_usd) for gpu in gpus_seen]
        assert step_costs == sorted(step_costs)
        # and the first attempt is the cheapest per step among the classes that fit.
        assert step_costs[0] == min(step_costs)


def test_supervisor_oom_walks_only_to_strictly_larger_gpu(monkeypatch):
    """An OOM retry must not re-roll the same VRAM class; it walks to a strictly larger card."""
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import contextlib

        import flash.providers.core.allocator as allocator
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train
        import flash.server.domain.teacher.broker as teacher_broker
        from flash.core.spec import GpuSpec, JobSpec, TrainSpec
        from flash.providers.core.base import Allocation, Candidate

        monkeypatch.setattr(
            teacher_broker,
            "require_teacher_broker_configuration",
            lambda _spec, **_kwargs: "https://broker.example",
        )

        @contextlib.contextmanager
        def teacher_transport(_spec, **_kwargs):
            yield {
                "FLASH_PUBLIC_URL": "https://broker.example",
                "FLASH_TEACHER_CAPABILITY": "capability-test-value",
            }

        monkeypatch.setattr(teacher_broker, "teacher_attempt_transport", teacher_transport)

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

        def fake_submit(spec, log=None, on_handle=None, attempt=0, **_):
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

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        import types

        import huggingface_hub

        class FakePrivateHf:
            def repo_info(self, **_kwargs):
                return types.SimpleNamespace(sha="private-pinned-sha")

            def get_paths_info(self, **_kwargs):
                return []

            def list_repo_tree(self, **_kwargs):
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
            model="Qwen/Qwen3.5-9B",
            algorithm="opd",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=2),
        )
        runner_submit.submit_job(spec, dry_run=False, background=False)

        assert runner_status.get_status("oom-walk").state == "done"
        assert gpus_seen == ["A100 SXM 40GB", "A100 PCIe"]


def test_supervisor_job_failed_without_marker_does_not_retry(monkeypatch):
    # A plain job_failed (no retriable flag — a genuine code crash) is NOT retried: the retry
    # budget exists only for infra-shaped failures, not code bugs.
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train
        from flash.core.spec import GpuSpec, JobSpec, TrainSpec

        calls = {"n": 0}

        def fake_submit(spec, log=None, on_handle=None, attempt=0, **_):
            calls["n"] += 1
            return jobs.PollResult(
                False, failure="job_failed", detail="ValueError: bad reward fn (no infra marker)"
            )

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="code-crash",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=2),
        )
        with pytest.raises(RuntimeError, match="bad reward fn"):
            runner_submit.submit_job(spec, dry_run=False, background=False)
        assert calls["n"] == 1  # genuine code error burns no retry budget
        assert runner_status.get_status("code-crash").state == "failed"


def test_supervisor_marks_on_last_gpu_on_the_largest_survivor(monkeypatch):
    # on_last_gpu stays false for the cached attempt while its exact cacheless fallback remains.
    import dataclasses

    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.core.allocator as allocator
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train
        from flash.core.spec import GpuSpec, JobSpec, TrainSpec

        # keep two candidates with different usable vram so the floor permits exactly one retry.
        monkeypatch.setattr(allocator, "required_vram_gb", lambda *a, **k: 80)
        real_allocate = allocator.allocate

        def two_candidate_allocate(*a, **k):
            alloc = real_allocate(*a, **k)
            keep = tuple(c for c in alloc.candidates if c.gpu in ("A100 PCIe", "RTX Pro 6000"))
            best = keep[0]
            return dataclasses.replace(
                alloc, gpu=best.gpu, hourly_usd=best.hourly_usd, candidates=keep
            )

        monkeypatch.setattr(allocator, "allocate", two_candidate_allocate)
        last_flags: list[bool] = []

        def fake_submit(spec, log=None, on_handle=None, attempt=0, on_last_gpu=False, **_):
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
            if attempt == 0:
                return jobs.PollResult(False, failure="stalled", detail="frozen")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="lastgpu",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=2),
        )
        runner_submit.submit_job(spec, dry_run=False, background=False)

        assert runner_status.get_status("lastgpu").state == "done"
        assert last_flags == [False, False]
        # the cached attempt keeps ordinary queue grace because its exact cacheless fallback remains.
        status = runner_status.get_status("lastgpu")
        assert status.remote is None
        assert status.cleanup_confirmed_remote.get("on_last_gpu") is False


def test_supervisor_allocation_failure_does_not_skip_cheapest(monkeypatch):
    # An allocation/pricing failure must NOT advance the candidate walk: that attempt never
    # provisioned a class, so the retry has to start over from the cheapest, not a pricier
    # one. (Regression guard for the walk-offset-vs-attempt-counter bug.)
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.core.allocator as allocator
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train
        from flash.core.spec import GpuSpec, JobSpec, TrainSpec

        real_allocate = allocator.allocate
        alloc_calls = {"n": 0}
        successful_allocations = []

        def flaky_allocate(*a, **k):
            alloc_calls["n"] += 1
            if alloc_calls["n"] == 1:
                raise RuntimeError("pricing API blip")  # not unsupported -> infra-shaped retry
            allocation = real_allocate(*a, **k)
            successful_allocations.append(allocation)
            return allocation

        monkeypatch.setattr(allocator, "allocate", flaky_allocate)

        gpus_seen: list[str] = []

        def fake_submit(spec, log=None, on_handle=None, attempt=0, **_):
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

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="alloc-blip",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=2),
        )
        runner_submit.submit_job(spec, dry_run=False, background=False)

        assert runner_status.get_status("alloc-blip").state == "done"
        # first allocation failed without provisioning; the retry used the selected cheapest class
        # rather than skipping it. derive the class from the successful current-catalog allocation.
        assert len(successful_allocations) == 1
        assert successful_allocations[0].gpu == successful_allocations[0].candidates[0].gpu
        assert gpus_seen == [successful_allocations[0].gpu]


def test_prepare_job_freezes_the_displayed_whole_cent_quote(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        monkeypatch.setattr(
            "flash.cost.spec.estimate_for_spec",
            lambda _spec: SimpleNamespace(total_usd=1.005),
        )

        prepared = runner_submit.prepare_job(_spec("whole-cent-quote"))

        assert prepared.estimated_cost_usd == 1.01


def test_selected_provider_never_refreshes_the_accepted_cost_quote(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train
        from flash.core.spec import GpuSpec, JobSpec, TrainSpec

        estimate_allocations = []

        def fixed_estimate(_spec, *, allocation=None):
            estimate_allocations.append(allocation)
            return SimpleNamespace(total_usd=1.005)

        monkeypatch.setattr("flash.cost.spec.estimate_for_spec", fixed_estimate)
        submitted = []

        def fake_submit(spec, log=None, on_handle=None, attempt=0, **_kwargs):
            submitted.append((spec.gpu.type, attempt))
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(job_execution, "submit_attempt", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="immutable-cost-quote",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=2),
        )
        runner_submit.submit_job(spec, dry_run=False, background=False)

        status = runner_status.get_status(spec.run_id)
        assert status.state == "done"
        assert status.estimated_cost_usd == 1.01
        assert submitted == [("A100 PCIe", 0)]
        # the only estimate is the same offline calculation shown by `flash train --cost`.
        assert estimate_allocations == [None]


def test_attach_costs_recovered_run_with_walked_gpu(monkeypatch):
    # A policy run that walked to a pricier class persists that class in the handle, so a
    # recovery via attach_run costs the card it actually ran on, not the provisional one.
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train
        from flash.providers.runpod.client import api as runpod_api

        status = provisioned_status(
            _spec("walked"),
            state="running",
            remote={
                "provider": "runpod",
                "endpoint_id": "epW",
                "endpoint_name": "n",
                "key_fingerprint": _RUNPOD_FINGERPRINT,
                "job_id": "jW",
                "allocated_gpu": "RTX 5090",
                "on_last_gpu": True,
                "attempt": 2,
                "launch_claim_token": "claim-2",
                "started_ts": 1.0,
            },
        )
        runner_state._save_status(status)
        # worker output carries wall time but neither cost nor allocated_gpu, and the failed
        # poll forces the job-status recovery shortcut to preserve the persisted gpu class.
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="poll_error", detail="api outage"),
        )
        status_deadlines = []

        def completed_status(*_args, **kwargs):
            status_deadlines.append(kwargs.get("deadline_at"))
            return {"status": "COMPLETED", "output": {"wall_seconds": 3600.0}}

        monkeypatch.setattr(runpod_api, "job_status", completed_status)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        st = runner_attach.attach_run("walked", log_stream=sys.stderr)

        assert st.state == "done"
        # the completion probe uses a short fresh timeout cap, never the far-future run wall
        # deadline (which would let a status-api outage burn the full retry budget and stall
        # recovery). the exact cap is pinned by
        # test_runpod_completed_metrics_caps_probe_deadline_when_wall_deadline_far_future.
        assert len(status_deadlines) == 1
        assert status_deadlines[0] < runner_deadlines._load_run_deadline_at("walked")
        import json
        import os

        from flash.providers.runpod.client.pricing import hourly_rate

        # cost_usd is now the QUOTE (flash.cost estimate we charge); the MEASURED cost in metrics.json
        # is what proves recovery costed the walked 5090, not the provisional 4090.
        with open(os.path.join(runner_state.artifacts_dir(_spec("walked")), "metrics.json")) as f:
            measured = json.load(f)["cost_usd"]
        assert abs(measured - hourly_rate("RTX 5090")) < 1e-6  # ~1 GPU-hour on the 5090
        assert measured > hourly_rate("RTX 4090")
        assert st.cost_usd == runner_costs.charge_usd_for_spec(_spec("walked"))


# ---------------------------------------------------------------------------
# Cross-process cancel via REST handle + attach
# ---------------------------------------------------------------------------
def test_cancel_prices_and_cleans_up_with_effective_warmstart_spec(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        from flash.core.spec import JobSpec

        base = _spec("warm-cancel").to_dict()
        public_spec = JobSpec.from_dict(
            {
                **base,
                "train": {
                    **base["train"],
                    "init_from_adapter": "source-run/final",
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
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="warm-cancel",
                state="running",
                spec=public_spec.to_dict(),
                billing_context={"org_id": "org-a"},
                last_heartbeat={"stage": "rl_step", "step": 2},
                source_snapshot=_SOURCE_SNAPSHOT,
                effective_preparation={
                    "worker_spec": worker_dict,
                    "adapter_identity": identity,
                    "version": 1,
                    "preparation_digest": runner_preparation._preparation_digest(
                        public_spec, worker_spec, identity
                    ),
                },
            )
        )
        priced = []
        cleaned = []

        def fake_charge(spec, *, steps=None, fallback=0.0, provider=None, gpu_type="", gpu_count=0):
            priced.append((spec.train.lora_rank, spec.train.init_from_adapter, steps))
            return 3.25

        monkeypatch.setattr(runner_costs, "charge_usd_for_spec", fake_charge)
        monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda spec: cleaned.append(spec))

        status = runner_deploy.cancel_run("warm-cancel")

        assert status.state == "cancelled"
        assert status.cost_usd == 3.25
        assert priced == [(64, "private-owner/private-repo:rl/source-run", 2)]
        assert cleaned[0].train.lora_rank == 64


@pytest.mark.parametrize("snapshot", [None, {"worker_spec": "malformed"}])
def test_cancel_with_invalid_preparation_uses_zero_failed_billing(monkeypatch, snapshot):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        from flash.core.spec import JobSpec

        base = _spec("warm-cancel-bad-snapshot").to_dict()
        public_spec = JobSpec.from_dict(
            {
                **base,
                "train": {
                    **base["train"],
                    "init_from_adapter": "source-run/final",
                    "lora_rank": 8,
                },
            }
        )
        runner_state._save_status(
            runner_state.RunStatus(
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
                    "launch_claim_token": "claim-0",
                    "started_ts": 1.0,
                },
                deployment={
                    "state": "ready",
                    "checkpoint_id": f"{public_spec.run_id}/final",
                },
                effective_preparation=snapshot,
            )
        )
        calls = []

        class Provider:
            def cancel(self, handle):
                calls.append("cancel")

            def destroy(self, handle):
                calls.append("destroy")

        import flash.serve.deployment.deploy
        from flash.providers.core import registry as provider_registry

        monkeypatch.setattr(provider_registry, "get_provider", lambda name: Provider())
        monkeypatch.setattr(
            flash.serve.deployment.deploy,
            "undeploy_adapter",
            lambda checkpoint_id, *, org_id: calls.append("undeploy"),
        )
        monkeypatch.setattr(
            runner_recovery,
            "_gc_run_endpoints",
            lambda spec: calls.append(("gc", spec.train.lora_rank)),
        )
        monkeypatch.setattr(
            runner_costs,
            "charge_usd_for_spec",
            lambda *a, **k: pytest.fail("public child rank must not be priced"),
        )

        status = runner_deploy.cancel_run(public_spec.run_id)

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
        _fresh_orchestrator(tmp, monkeypatch)
        from flash.providers.runpod.client import api as runpod_api

        status = runner_state.RunStatus(
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
                "launch_claim_token": "claim-0",
                "started_ts": 1.0,
            },
        )
        runner_state._save_status(status)
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
        import flash.providers.runpod.serverless.endpoints as flash_train

        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        st = runner_deploy.cancel_run("c1")
        assert st.state == "cancelled"
        assert cancelled
        assert all(item == ("epX", "jX") for item in cancelled)
        # cancel_run now also destroys the handle's endpoint (idempotent); the GC backstop may
        # delete it again — endpoint id was torn down, which is what matters.
        assert deleted
        assert all(e == "epX" for e in deleted)


def test_attach_completes_run(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        status = provisioned_status(
            _spec("a1"),
            state="running",
            remote={
                "provider": "runpod",
                "endpoint_id": "epA",
                "endpoint_name": "n",
                "key_fingerprint": _RUNPOD_FINGERPRINT,
                "job_id": "jA",
                "on_last_gpu": False,
                "attempt": 0,
                "launch_claim_token": "claim-0",
                "started_ts": 1.0,
                "allocated_gpu": "RTX 4090",
                "allocated_gpu_count": 1,
                "allocated_usable_vram_gb": 24.0,
            },
        )
        runner_state._save_status(status)
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(True, metrics={"cost_usd": 0.2}),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        st = runner_attach.attach_run("a1", log_stream=sys.stderr)
        assert st.state == "done"
        # We charge the QUOTE (flash.cost estimate); the measured 0.2 is recorded in metrics.json.
        import json
        import os

        assert st.cost_usd == runner_costs.charge_usd_for_spec(_spec("a1"))
        with open(os.path.join(runner_state.artifacts_dir(_spec("a1")), "metrics.json")) as f:
            assert json.load(f)["cost_usd"] == 0.2


def test_attach_cleanup_survives_unreadable_final_status(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs

        run_id = "attach-finally-status"
        runner_state._save_status(
            provisioned_status(
                _spec(run_id),
                state="running",
                remote={
                    "provider": "runpod",
                    "endpoint_id": "ep-finally",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "j-finally",
                    "attempt": 0,
                    "launch_claim_token": "claim-0",
                    "started_ts": 1.0,
                },
            )
        )
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(True, metrics={"cost_usd": 0.2}),
        )

        real_get_status = runner_status.get_status
        real_update = runner_status._update
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
        monkeypatch.setattr(runner_status, "_update", update_then_arm_status_failure)
        monkeypatch.setattr(runner_status, "get_status", transiently_unreadable_status)
        monkeypatch.setattr(
            runner_recovery, "_gc_run_endpoints", lambda spec: gc_calls.append(spec.run_id)
        )

        status = runner_attach.attach_run(run_id, log_stream=sys.stderr)

        assert status.state == "done", "the final status read must not mask the completed outcome"
        assert gc_calls == [run_id], "terminal endpoint cleanup must still run"


def test_attach_confirmed_cancel_survives_unreadable_cleanup_status(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs

        run_id = "attach-confirmed-cancel"
        runner_state._save_status(
            provisioned_status(
                _spec(run_id),
                state="running",
                remote={
                    "provider": "runpod",
                    "endpoint_id": "ep-cancel",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "j-cancel",
                    "attempt": 0,
                    "launch_claim_token": "claim-0",
                    "started_ts": 1.0,
                },
            )
        )
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(True, metrics={"cost_usd": 0.2}),
        )

        def cancel_during_metrics(spec, metrics):
            runner_status._update(run_id, "cancelled")
            return 0.2

        real_get_status = runner_status.get_status
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
        monkeypatch.setattr(runner_status, "_persist_metrics", cancel_during_metrics)
        monkeypatch.setattr(runner_status, "get_status", unreadable_after_confirmed_cancel)
        monkeypatch.setattr(
            runner_recovery, "_gc_run_endpoints", lambda spec: gc_calls.append(spec.run_id)
        )

        status = runner_attach.attach_run(run_id, log_stream=sys.stderr)

        assert status.state == "cancelled"
        assert gc_calls == [run_id], "the positively observed terminal cancel must still be reaped"


def test_attach_duplicate_supervisor_unreadable_status_preserves_live_owner(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs

        run_id = "attach-live-owner"
        stale_remote = {
            "provider": "runpod",
            "endpoint_id": "ep-stale",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "j-stale",
            "attempt": 0,
            "launch_claim_token": "claim-0",
            "started_ts": 1.0,
            "allocated_gpu": "RTX 4090",
            "allocated_gpu_count": 1,
            "allocated_usable_vram_gb": 24.0,
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
        status = provisioned_status(
            _spec(run_id),
            state="running",
            remote=stale_remote,
        )
        status.source_snapshot = _SOURCE_SNAPSHOT
        runner_state._save_status(status, _next_attempt=1)
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="stalled", detail="redeploy"),
        )

        real_get_status = runner_status.get_status
        status_failures = {"remaining": 0}

        def transiently_unreadable_status(run_id):
            if status_failures["remaining"]:
                status_failures["remaining"] -= 1
                raise PermissionError("status file is transiently unreadable")
            return real_get_status(run_id)

        def duplicate_supervisor_refusal(*args, **kwargs):
            runner_status._update(run_id, "running", remote=live_remote)
            status_failures["remaining"] = 2
            raise runner_errors._RunCancelled("another supervisor owns the durable provider handle")

        gc_calls = []
        monkeypatch.setattr(runner_status, "get_status", transiently_unreadable_status)
        monkeypatch.setattr(runner_lifecycle, "_run_training", duplicate_supervisor_refusal)
        monkeypatch.setattr(
            runner_recovery, "_gc_run_endpoints", lambda spec: gc_calls.append(spec.run_id)
        )

        status = runner_attach.attach_run(run_id, log_stream=sys.stderr)

        assert status.state == "running"
        assert status.remote == live_remote
        assert gc_calls == [], (
            "captured-handle recovery must not run run-wide gc after newer ownership appears"
        )


def test_attach_requires_handle(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        runner_state._save_status(
            runner_state.RunStatus(run_id="nh", state="running", spec=_spec("nh").to_dict())
        )
        with pytest.raises(ValueError, match="no persisted job handle"):
            runner_attach.attach_run("nh")


def test_attach_unparseable_spec_fails_closed_and_tears_down(monkeypatch):
    """Terminate a run whose persisted spec no longer parses.

    Recovery runs on a daemon thread; an uncaught parse error would leave the live handle billing
    under a nonterminal status.
    """
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import json

        import flash.providers.runpod.serverless.endpoints as flash_train
        import flash.runner.supervise.lifecycle as lifecycle

        remote = {
            "provider": "runpod",
            "endpoint_id": "epBad",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "jBad",
            "attempt": 0,
            "launch_claim_token": "claim-0",
            "started_ts": 1.0,
        }
        from dataclasses import replace

        spec = _spec("bad")
        spec = replace(spec, gpu=replace(spec.gpu, type="RTX 5090"))
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="bad", state="running", spec=spec.to_dict(), remote=remote
            )
        )
        # Rewrite the spec on disk: the plane can no longer WRITE this record, so go around the
        # writer the same way an older plane's leftover file would have arrived.
        raw = runner_status._load_status_json("bad")
        raw["spec"] = {**raw["spec"], "environment": {"path": "/legacy/local/env"}}
        with open(runner_state.runs_file_path("bad", ".json"), "w") as file:
            json.dump(raw, file)

        torn_down = []
        terminated = []
        monkeypatch.setattr(
            lifecycle,
            "_strict_teardown_handle",
            lambda handle, rid: torn_down.append((handle.data.get("endpoint_id"), rid)) or True,
        )
        monkeypatch.setattr(
            flash_train, "terminate_endpoint", lambda gpu, rid: terminated.append((gpu, rid))
        )

        status = runner_attach.attach_run("bad", log_stream=sys.stderr)

        assert status.state == "failed"
        assert "spec is malformed" in (status.error or "")
        # the exact endpoint the handle names, plus the rN retry endpoints it cannot name.
        assert torn_down == [("epBad", "bad")]
        assert terminated == [("RTX 5090", "bad")]


def test_attach_resumes_from_checkpoint_on_poll_failure(monkeypatch):
    # A recovered run whose remote job ended not-ok (it died while the control plane was down for
    # the redeploy) must NOT be failed — reattach resumes training on a fresh host (worker resumes
    # from the latest HF checkpoint), exactly like the fresh-submit retry loop. It also clears the
    # stale handle so a second restart during the fresh allocation re-resumes cleanly.
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        status = provisioned_status(
            _spec("i1"),
            state="running",
            cost_usd=0.0,
            remote={
                "provider": "runpod",
                "endpoint_id": "epA",
                "endpoint_name": "n",
                "key_fingerprint": _RUNPOD_FINGERPRINT,
                "job_id": "jA",
                "on_last_gpu": False,
                "attempt": 0,
                "launch_claim_token": "claim-0",
                "started_ts": 1.0,
                "allocated_gpu": "RTX 4090",
                "allocated_gpu_count": 1,
                "allocated_usable_vram_gb": 24.0,
            },
        )
        status.source_snapshot = _SOURCE_SNAPSHOT
        runner_state._save_status(status, _next_attempt=1)
        # Poll reports a dead/abandoned job (the common redeploy-window outcome).
        monkeypatch.setattr(
            polling,
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
            source_snapshot=None,
            reserved_claim=None,
        ):
            assert reserved_claim is not None
            seen["remote"] = runner_status.get_status(spec.run_id).remote
            seen["source_snapshot"] = source_snapshot
            runner_status._update(spec.run_id, "done", cost_usd=prior_cost)

        monkeypatch.setattr(runner_lifecycle, "_run_training", fake_training)

        st = runner_attach.attach_run("i1", log_stream=sys.stderr)

        assert seen["remote"] is None, "stale dead handle must be cleared before resuming"
        assert seen["source_snapshot"] == _SOURCE_SNAPSHOT
        assert st.state != "failed", "a job lost to the redeploy must be resumed, not failed"
        assert st.state == "done"


def test_attach_one_shot_failure_does_not_submit_attempt_one(monkeypatch):
    from dataclasses import replace

    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        spec = _spec("one-shot-recovery")
        spec = replace(spec, gpu=replace(spec.gpu, max_retries=0))
        runner_state._save_status(
            provisioned_status(
                spec,
                state="running",
                remote={
                    "provider": "runpod",
                    "endpoint_id": "epA",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "jA",
                    "on_last_gpu": True,
                    "attempt": 0,
                    "launch_claim_token": "claim-0",
                    "started_ts": 1.0,
                    "allocated_gpu": "RTX 4090",
                    "allocated_gpu_count": 1,
                    "allocated_usable_vram_gb": 24.0,
                },
            ),
            _next_attempt=1,
        )
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="stalled", detail="host vanished"),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        training_calls = []
        monkeypatch.setattr(
            runner_lifecycle, "_run_training", lambda *a, **k: training_calls.append((a, k))
        )

        status = runner_attach.attach_run(spec.run_id, log_stream=sys.stderr)

        assert status.state == "failed"
        assert status.error == "stalled: host vanished"
        assert training_calls == []
        assert status.remote is None
        assert status.cleanup_confirmed_remote["endpoint_id"] == "epA"


def test_attach_resume_reuses_persisted_source_snapshot(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        status = provisioned_status(
            _spec("pinned-code"),
            state="running",
            cost_usd=0.25,
            remote={
                "provider": "runpod",
                "endpoint_id": "epA",
                "endpoint_name": "n",
                "key_fingerprint": _RUNPOD_FINGERPRINT,
                "job_id": "jA",
                "on_last_gpu": False,
                "attempt": 0,
                "launch_claim_token": "claim-0",
                "started_ts": 1.0,
                "allocated_gpu": "RTX 4090",
                "allocated_gpu_count": 1,
                "allocated_usable_vram_gb": 24.0,
            },
        )
        status.source_snapshot = _SOURCE_SNAPSHOT
        runner_state._save_status(status, _next_attempt=1)
        monkeypatch.setattr(
            polling,
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
            source_snapshot=None,
            reserved_claim=None,
        ):
            assert reserved_claim is not None
            seen["prior_cost"] = prior_cost
            seen["source_snapshot"] = source_snapshot
            runner_status._update(spec.run_id, "done", cost_usd=prior_cost)

        monkeypatch.setattr(runner_lifecycle, "_run_training", fake_training)

        st = runner_attach.attach_run("pinned-code", log_stream=sys.stderr)

        assert st.state == "done"
        assert seen == {
            "prior_cost": 0.25,
            "source_snapshot": _SOURCE_SNAPSHOT,
        }


def test_attach_worker_error_fails_without_replacement(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs

        remote = {
            "provider": "runpod",
            "endpoint_id": "epA",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "jA",
            "on_last_gpu": False,
            "attempt": 0,
            "launch_claim_token": "claim-0",
            "started_ts": 1.0,
            "allocated_gpu": "RTX 4090",
            "allocated_gpu_count": 1,
            "allocated_usable_vram_gb": 24.0,
        }
        status = provisioned_status(_spec("g1"), state="running", remote=remote)
        status.source_snapshot = _SOURCE_SNAPSHOT
        runner_state._save_status(status, _next_attempt=1)
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(
                False, failure="worker_error", detail="RuntimeError: bad reward fn"
            ),
        )
        monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
        monkeypatch.setattr(
            runner_lifecycle,
            "_run_training",
            lambda *_a, **_k: pytest.fail("worker error must not allocate a replacement"),
        )

        status = runner_attach.attach_run("g1", log_stream=sys.stderr)

        assert status.state == "failed"
        assert status.remote is None
        assert status.cleanup_confirmed_remote == remote
        assert "bad reward fn" in (status.error or "")


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
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        from flash.providers.runpod.client import api as runpod_api

        remote = {
            "provider": "runpod",
            "endpoint_id": "ep-old",
            "endpoint_name": "old",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "job-old",
            "attempt": 0,
            "launch_claim_token": "claim-0",
            "started_ts": 1.0,
            "allocated_gpu": "RTX 4090",
            "allocated_gpu_count": 1,
            "allocated_usable_vram_gb": 24.0,
        }
        runner_state._save_status(
            provisioned_status(
                _spec("runpod-unconfirmed"),
                state="running",
                remote=remote,
            ),
            _next_attempt=1,
        )
        monkeypatch.setattr(
            polling,
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
        monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
        import flash.runner.supervise.attach as attach_mod

        scheduled = []
        monkeypatch.setattr(
            attach_mod,
            "_schedule_attach_reconciliation",
            lambda *args, **kwargs: scheduled.append((args, kwargs)) or True,
        )
        resumed = []
        monkeypatch.setattr(runner_lifecycle, "_run_training", lambda *a, **k: resumed.append(True))

        status = runner_attach.attach_run("runpod-unconfirmed", log_stream=sys.stderr)

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
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs

        old_remote = {
            "provider": "runpod",
            "endpoint_id": "ep-old",
            "endpoint_name": "old",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "job-old",
            "attempt": 0,
            "launch_claim_token": "claim-0",
            "started_ts": 1.0,
            "allocated_gpu": "RTX 4090",
            "allocated_gpu_count": 1,
            "allocated_usable_vram_gb": 24.0,
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
        runner_state._save_status(
            provisioned_status(
                _spec("attach-newer-remote"),
                state="running",
                remote=old_remote,
            ),
            _next_attempt=1,
        )
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="stalled", detail="stalled"),
        )
        import flash.runner.supervise.lifecycle as lifecycle_mod

        real_teardown = lifecycle_mod._strict_teardown_handle

        def teardown_then_replace(handle, run_id=None):
            result = real_teardown(handle, run_id)
            assert runner_status._update("attach-newer-remote", "running", remote=newer_remote)
            return result

        monkeypatch.setattr(lifecycle_mod, "_strict_teardown_handle", teardown_then_replace)
        gc_calls = []
        monkeypatch.setattr(
            runner_recovery, "_gc_run_endpoints", lambda _spec: gc_calls.append(True)
        )
        resumed = []
        monkeypatch.setattr(runner_lifecycle, "_run_training", lambda *a, **k: resumed.append(True))

        status = runner_attach.attach_run("attach-newer-remote", log_stream=sys.stderr)

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
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.serverless.endpoints as flash_train
        from flash.providers.core import registry as providers
        from flash.providers.core.base import PollResult
        from flash.providers.vast.client import api as vast_api

        runner_state._save_status(
            provisioned_status(
                _spec("v1"),
                state="running",
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
                    "launch_claim_token": "claim-0",
                    "started_ts": 1.0,
                    "allocated_gpu": "RTX 4090",
                    "allocated_gpu_count": 1,
                    "allocated_usable_vram_gb": 24.0,
                },
            ),
            _next_attempt=1,
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        class _RaisingVast:
            def poll_attempt(self, handle, spec, *, log=None, _deadline_at=None):
                assert _deadline_at == pytest.approx(
                    runner_status.get_status("v1").created_at + _spec("v1").gpu.max_wall_seconds
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

        monkeypatch.setattr(runner_lifecycle, "_run_training", fake_loop)
        import flash.runner.supervise.attach as attach_mod

        scheduled = []
        monkeypatch.setattr(
            attach_mod,
            "_schedule_attach_reconciliation",
            lambda *args, **kwargs: scheduled.append((args, kwargs)) or True,
        )

        st = runner_attach.attach_run("v1", log_stream=sys.stderr)

        assert resumed["called"] is False, (
            "must NOT resume a second worker over a possibly-live box"
        )
        assert len(scheduled) == 1
        assert st.state == "running", "run left non-terminal for a later recovery/sweep"
        remote = runner_status.get_status("v1").remote
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
        "launch_claim_token": f"claim-{attempt}",
        "started_ts": 100.0,
    }


def test_attach_reconciliation_thread_start_failure_releases_guard(monkeypatch):
    import flash.runner.supervise.attach as attach_mod

    run_id = "attach-reconcile-thread-start"
    remote = _vast_recovery_remote()
    spec = _spec(run_id)
    with attach_mod._ATTACH_RECONCILING_LOCK:
        attach_mod._ATTACH_RECONCILING.discard(run_id)

    class RaisingThread:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            raise RuntimeError("thread start failed")

    monkeypatch.setattr(attach_mod, "threading", SimpleNamespace(Thread=RaisingThread))
    with pytest.raises(RuntimeError, match="thread start failed"):
        attach_mod._schedule_attach_reconciliation(
            run_id,
            remote,
            spec,
            1,
            _SOURCE_SNAPSHOT,
            io.StringIO(),
            "stalled: host vanished",
        )
    with attach_mod._ATTACH_RECONCILING_LOCK:
        assert run_id not in attach_mod._ATTACH_RECONCILING

    monkeypatch.setattr(attach_mod, "_reconcile_attached_remote", lambda *a, **k: None)

    class ImmediateThread:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(attach_mod, "threading", SimpleNamespace(Thread=ImmediateThread))
    assert attach_mod._schedule_attach_reconciliation(
        run_id,
        remote,
        spec,
        1,
        "code/revision",
        io.StringIO(),
        "stalled: host vanished",
    )
    with attach_mod._ATTACH_RECONCILING_LOCK:
        assert run_id not in attach_mod._ATTACH_RECONCILING


def test_attach_reconciliation_cleans_endpoint_after_background_completion(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.runner.supervise.attach as attach_mod

        run_id = "attach-reconcile-cleanup"
        remote = _vast_recovery_remote()
        spec = _spec(run_id)
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=run_id,
                state="running",
                spec=spec.to_dict(),
                remote=remote,
            )
        )

        def complete_recovery(*_args, **_kwargs):
            runner_status._update(run_id, "done")

        cleaned = []
        monkeypatch.setattr(attach_mod, "_reconcile_attached_remote", complete_recovery)
        monkeypatch.setattr(
            runner_recovery, "_gc_run_endpoints", lambda current: cleaned.append(current.run_id)
        )

        class ImmediateThread:
            def __init__(self, *, target, daemon):
                self.target = target

            def start(self):
                self.target()

        monkeypatch.setattr(attach_mod, "threading", SimpleNamespace(Thread=ImmediateThread))

        assert attach_mod._schedule_attach_reconciliation(
            run_id,
            remote,
            spec,
            1,
            _SOURCE_SNAPSHOT,
            io.StringIO(),
            "stalled: host vanished",
        )
        assert cleaned == [run_id]


def test_attach_reconciler_resumes_after_vast_strict_absence(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        from dataclasses import replace

        import flash.runner.supervise.attach as attach_mod
        from flash.providers.core import registry as providers
        from flash.runner.supervise.retry_decision import (
            PersistedRetryDecision,
            RetryPlan,
            RetryState,
        )

        remote = _vast_recovery_remote()
        spec = _spec("vast-reconcile-clear")
        retry_state = replace(
            RetryState.initial_for_spec(spec),
            infra_used=1,
            last_decision=PersistedRetryDecision(
                0,
                "stalled",
                RetryPlan(True, "retrying", infra_retry_ordinal=1),
            ),
        )
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                remote=remote,
                source_snapshot=_SOURCE_SNAPSHOT,
            ),
            _next_attempt=1,
            _retry_state=retry_state.to_snapshot(),
        )

        class Provider:
            def destroy(self, _handle):
                raise RuntimeError("delete acknowledgement unavailable")

            def run_instances_remaining(self, run_id):
                return []

        monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
        monkeypatch.setattr(attach_mod, "_ATTACH_RECONCILE_INTERVAL_S", 0.0)
        resumed = []

        def resume_training(*_args, **kwargs):
            assert kwargs["reserved_claim"].attempt == 1
            resumed.append(1)
            runner_status._update(spec.run_id, "running")

        monkeypatch.setattr(runner_lifecycle, "_run_training", resume_training)

        attach_mod._reconcile_attached_remote(
            spec.run_id,
            remote,
            spec,
            1,
            _SOURCE_SNAPSHOT,
            io.StringIO(),
            "stalled: host vanished",
        )

        assert resumed == [1]
        assert runner_status.get_status(spec.run_id).state == "running"
        assert runner_status.get_status(spec.run_id).remote is None


def test_attach_reconciler_deadline_retries_terminal_persistence(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.runner.supervise.attach as attach_mod
        import flash.runner.supervise.lifecycle as lifecycle_mod

        remote = _vast_recovery_remote()
        spec = _spec("vast-reconcile-deadline")
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                created_at=100.0,
                remote=remote,
            )
        )
        deadline = runner_deadlines._load_run_deadline_at(spec.run_id)
        monkeypatch.setattr(attach_mod.time, "time", lambda: deadline + 1.0)
        monkeypatch.setattr(attach_mod, "_ATTACH_RECONCILE_INTERVAL_S", 0.0)
        monkeypatch.setattr(lifecycle_mod, "_completed_attempt_metrics", lambda *a, **k: None)
        real_fail = runner_reconciliation._compare_and_fail_remote
        calls = []

        def flaky_fail(*args, **kwargs):
            calls.append(True)
            if len(calls) == 1:
                raise PermissionError("status store unavailable")
            return real_fail(*args, **kwargs)

        monkeypatch.setattr(runner_reconciliation, "_compare_and_fail_remote", flaky_fail)

        attach_mod._reconcile_attached_remote(
            spec.run_id,
            remote,
            spec,
            1,
            _SOURCE_SNAPSHOT,
            io.StringIO(),
            "stalled: host vanished",
        )

        status = runner_status.get_status(spec.run_id)
        assert len(calls) == 2
        assert status.state == "failed"
        assert status.remote == remote
        assert runner_status._load_status_json(spec.run_id)[runner_state._CLEANUP_REMOTES_KEY] == [
            {key: value for key, value in remote.items() if key != "launch_claim_token"}
        ]


def test_attach_reconciler_adopts_completed_phantom_at_deadline(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.runner.supervise.attach as attach_mod
        import flash.runner.supervise.lifecycle as lifecycle_mod

        remote = _vast_recovery_remote()
        spec = _spec("vast-reconcile-complete")
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                created_at=100.0,
                remote=remote,
            )
        )
        deadline = runner_deadlines._load_run_deadline_at(spec.run_id)
        monkeypatch.setattr(attach_mod.time, "time", lambda: deadline + 1.0)
        monkeypatch.setattr(attach_mod, "_ATTACH_RECONCILE_INTERVAL_S", 0.0)
        monkeypatch.setattr(
            lifecycle_mod,
            "_completed_attempt_metrics",
            lambda *a, **k: {"wall_seconds": 5.0},
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

        status = runner_status.get_status(spec.run_id)
        assert status.state == "done"
        assert status.remote == remote
        assert status.error is None
        assert runner_status._load_status_json(spec.run_id)[runner_state._CLEANUP_REMOTES_KEY] == [
            {key: value for key, value in remote.items() if key != "launch_claim_token"}
        ]


def test_attach_reconciler_caps_completed_adoption_retry_to_grace(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.runner.supervise.attach as attach_mod
        import flash.runner.supervise.lifecycle as lifecycle_mod

        remote = _runpod_handle_dict(jobs, started_ts=1.0)
        spec = _spec("runpod-reconcile-adoption-grace")
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                remote=remote,
            )
        )
        deadline = 1_000.0
        clock = {"now": deadline + lifecycle_mod._RECOVERY_MARKER_GRACE_S - 1.0}
        sleeps = []
        monkeypatch.setattr(runner_deadlines, "_load_run_deadline_at", lambda _run_id: deadline)
        monkeypatch.setattr(attach_mod.time, "time", lambda: clock["now"])

        def advance(seconds):
            sleeps.append(seconds)
            clock["now"] += seconds

        monkeypatch.setattr(attach_mod.time, "sleep", advance)
        monkeypatch.setattr(
            lifecycle_mod,
            "_runpod_completed_metrics",
            lambda *_args, **_kwargs: {"wall_seconds": 60.0},
        )
        monkeypatch.setattr(lifecycle_mod, "_adopt_completed_attempt", lambda *_a, **_k: False)

        attach_mod._reconcile_attached_remote(
            spec.run_id,
            remote,
            spec,
            1,
            _SOURCE_SNAPSHOT,
            io.StringIO(),
            "stalled: host vanished",
        )

        assert sleeps == [1.0]
        assert runner_status.get_status(spec.run_id).state == "failed"
        assert "could not be adopted" in runner_status.get_status(spec.run_id).error


def test_attach_reconciler_reprobes_completion_after_deadline_capped_sleep(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.runner.supervise.attach as attach_mod
        import flash.runner.supervise.lifecycle as lifecycle_mod

        remote = _runpod_handle_dict(jobs, started_ts=1.0)
        spec = _spec("runpod-reconcile-deadline-reprobe")
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                remote=remote,
            )
        )
        deadline = 1_000.0
        clock = {"now": deadline - 1.0}
        probes = []
        monkeypatch.setattr(runner_deadlines, "_load_run_deadline_at", lambda _run_id: deadline)
        monkeypatch.setattr(attach_mod.time, "time", lambda: clock["now"])
        monkeypatch.setattr(
            attach_mod.time,
            "sleep",
            lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
        )

        def completed_metrics(*_args, **_kwargs):
            probes.append(clock["now"])
            if clock["now"] >= deadline:
                return {"wall_seconds": 60.0}
            return None

        monkeypatch.setattr(lifecycle_mod, "_runpod_completed_metrics", completed_metrics)
        monkeypatch.setattr(lifecycle_mod, "_adopt_completed_attempt", lambda *_a, **_k: True)
        monkeypatch.setattr(
            lifecycle_mod,
            "_strict_teardown_handle",
            lambda *_args, **_kwargs: pytest.fail("completed attempt must not be torn down"),
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

        assert probes == [deadline - 1.0, deadline]


def test_attach_reconciler_rate_limits_failed_terminal_cas_past_grace(monkeypatch):
    # past the recovery grace window the terminal compare-and-fail CAS is the only exit
    # from the completed-but-unadoptable branch. if that CAS transiently raises, the
    # reconciler must rate-limit each retry at the full reconcile interval instead of
    # sleeping 0 (remaining grace is <= 0 past the window) and busy-spinning the loop.
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.runner.supervise.attach as attach_mod
        import flash.runner.supervise.lifecycle as lifecycle_mod

        remote = _runpod_handle_dict(jobs, started_ts=1.0)
        spec = _spec("runpod-reconcile-cas-ratelimit")
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                remote=remote,
            )
        )
        deadline = 1_000.0
        # start already past the recovery grace window so remaining grace is <= 0.
        clock = {"now": deadline + lifecycle_mod._RECOVERY_MARKER_GRACE_S}
        sleeps = []
        monkeypatch.setattr(runner_deadlines, "_load_run_deadline_at", lambda _run_id: deadline)
        monkeypatch.setattr(attach_mod.time, "time", lambda: clock["now"])

        def advance(seconds):
            sleeps.append(seconds)
            clock["now"] += seconds

        monkeypatch.setattr(attach_mod.time, "sleep", advance)
        monkeypatch.setattr(
            lifecycle_mod,
            "_runpod_completed_metrics",
            lambda *_args, **_kwargs: {"wall_seconds": 60.0},
        )
        monkeypatch.setattr(lifecycle_mod, "_adopt_completed_attempt", lambda *_a, **_k: False)
        # the reconciler imports _compare_and_fail_remote / _record_cleanup_remote from
        # flash.runner (== orch), so patch them there rather than on lifecycle.
        monkeypatch.setattr(runner_reconciliation, "_record_cleanup_remote", lambda *_a, **_k: True)

        real_fail = runner_reconciliation._compare_and_fail_remote
        cas_calls = {"n": 0}

        def flaky_fail(run_id, expected_remote, reason):
            cas_calls["n"] += 1
            if cas_calls["n"] <= 2:
                raise RuntimeError("transient status store failure")
            return real_fail(run_id, expected_remote, reason)

        monkeypatch.setattr(runner_reconciliation, "_compare_and_fail_remote", flaky_fail)

        attach_mod._reconcile_attached_remote(
            spec.run_id,
            remote,
            spec,
            1,
            _SOURCE_SNAPSHOT,
            io.StringIO(),
            "stalled: host vanished",
        )

        # two transient CAS failures -> two full-interval rate-limited retries (never sleep 0),
        # then the third CAS sticks and fails the run.
        assert sleeps == [
            attach_mod._ATTACH_RECONCILE_INTERVAL_S,
            attach_mod._ATTACH_RECONCILE_INTERVAL_S,
        ]
        assert cas_calls["n"] == 3
        assert runner_status.get_status(spec.run_id).state == "failed"
        assert "could not be adopted" in runner_status.get_status(spec.run_id).error


def test_an_adopted_instance_run_is_still_priced_for_every_card_it_occupied(monkeypatch):
    # _completed_attempt_metrics returns the WORKER's metrics.json verbatim, and the worker never
    # knew how many cards the allocator gave it -- the plane stamped that onto the persisted remote
    # at launch. so an adopted multi-card vast/lambda run reaches _persist_metrics with no count
    # and prices its whole wall as ONE card, silently understating a 4-card run 4x.
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.runner.supervise.attach as attach_mod
        import flash.runner.supervise.lifecycle as lifecycle_mod

        remote = _vast_recovery_remote()
        remote["allocated_gpu"] = "RTX 4090"
        remote["allocated_gpu_count"] = 4
        spec = _spec("vast-adopt-multicard")
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                remote=remote,
            )
        )
        deadline = 1_000.0
        monkeypatch.setattr(runner_deadlines, "_load_run_deadline_at", lambda _run_id: deadline)
        monkeypatch.setattr(attach_mod.time, "time", lambda: deadline)
        monkeypatch.setattr(attach_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(lifecycle_mod, "_runpod_completed_metrics", lambda *_a, **_k: None)
        # what the worker actually wrote: a wall, and nothing about the allocation.
        monkeypatch.setattr(
            lifecycle_mod,
            "_completed_attempt_metrics",
            lambda *_a, **_k: {"wall_seconds": 3600.0},
        )
        monkeypatch.setattr(runner_reconciliation, "_record_cleanup_remote", lambda *_a, **_k: True)

        adopted = {}

        def capture(_run_id, _spec, _expected, metrics, **_kwargs):
            adopted.update(metrics)
            return True

        monkeypatch.setattr(lifecycle_mod, "_adopt_completed_attempt", capture)

        attach_mod._reconcile_attached_remote(
            spec.run_id,
            remote,
            spec,
            1,
            _SOURCE_SNAPSHOT,
            io.StringIO(),
            "stalled: host vanished",
        )

        assert adopted["allocated_gpu_count"] == 4, (
            "an adopted multi-card run reached persistence with no card count, so its wall "
            "prices as a single card"
        )
        assert adopted["allocated_gpu"] == "RTX 4090"
        # same argument for the substrate: _gpu_rate falls back to whichever configured provider
        # offers the class, normally RunPod, so an adopted vast run is otherwise priced at RunPod's
        # rate and its notes name a provider that never ran it.
        assert adopted["allocated_provider"] == "vast", (
            "an adopted vast run reached persistence with no provider, so it is priced on "
            "whichever provider the plane happens to try first"
        )


def test_attach_reconciler_does_not_clobber_newer_remote(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.runner.supervise.attach as attach_mod

        old_remote = _vast_recovery_remote(instance_id=101, attempt=0)
        newer_remote = _vast_recovery_remote(instance_id=202, attempt=1)
        spec = _spec("vast-reconcile-newer")
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                remote=newer_remote,
            )
        )
        monkeypatch.setattr(
            attach_mod,
            "_resume_after_confirmed_teardown",
            lambda *a, **k: pytest.fail("stale reconciler must not resume"),
        )
        monkeypatch.setattr(
            runner_reconciliation,
            "_compare_and_fail_remote",
            lambda *a, **k: pytest.fail("stale reconciler must not fail the run"),
        )

        attach_mod._reconcile_attached_remote(
            spec.run_id,
            old_remote,
            spec,
            1,
            _SOURCE_SNAPSHOT,
            io.StringIO(),
            "stalled: host vanished",
        )

        status = runner_status.get_status(spec.run_id)
        assert status.state == "running"
        assert status.remote == newer_remote


def test_update_will_not_overwrite_terminal_with_lifecycle_state(monkeypatch):
    # Terminal states are STICKY: once cancelled, no other state may overwrite it —
    # neither a non-terminal lifecycle write (provisioning/running) NOR a late terminal
    # done/failed from a worker that finished as the cancel arrived. Same-state writes
    # still pass so terminal field updates (cost_usd, error) are preserved.
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        runner_state._save_status(
            runner_state.RunStatus(run_id="c", state="cancelled", spec=_spec("c").to_dict())
        )
        runner_status._update("c", "provisioning")
        assert runner_status.get_status("c").state == "cancelled", (
            "cancelled must not become provisioning"
        )
        runner_status._update("c", "running", cost_usd=1.0)
        assert runner_status.get_status("c").state == "cancelled"
        # A late terminal completion must NOT resurrect/relabel a user cancellation.
        runner_status._update("c", "failed", error="x")
        assert runner_status.get_status("c").state == "cancelled"
        # Same-state writes still apply terminal field updates.
        runner_status._update("c", "cancelled", cost_usd=2.0)
        assert runner_status.get_status("c").state == "cancelled"
        assert runner_status.get_status("c").cost_usd == 2.0


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


def _patch_deploy_deps(monkeypatch, jobs, *, set_key: bool = True):
    """Patch ``deploy_train_endpoint`` dependencies.

    ``set_key=False`` preserves a caller-installed multi-account pool for failover tests.
    """
    import flash.providers.runpod.client.auth as auth_mod
    import flash.providers.runpod.client.auth as keys

    # Pin the pool for the default case rather than reading the ambient env: callers assert on the
    # fingerprint of THIS key, and a real operator key would both leak into the assertion and
    # change it.
    if set_key:
        monkeypatch.setenv("RUNPOD_API_KEY", "test-key")
        keys.reset()
    monkeypatch.setattr(runpod_endpoints, "FLASH_SDK_LOCK", __import__("threading").Lock())
    monkeypatch.setattr(runpod_endpoints, "isolate_flash_state", lambda *a, **k: None)
    monkeypatch.setattr(auth_mod, "ensure_auth", lambda: keys.active_key())
    monkeypatch.setattr(runpod_endpoints, "_patch_runpod_backoff", lambda: None)
    monkeypatch.setattr(job_execution, "flash_gpu", lambda g: g)
    monkeypatch.setattr(job_execution, "canonical_gpu", lambda g: g)
    monkeypatch.setattr(runpod_endpoints, "endpoint_name", lambda g, s: f"flash-{g}-test")
    monkeypatch.setattr(runpod_endpoints, "min_cuda_for", lambda g: "12.8")
    monkeypatch.setattr(job_execution.runpod_worker, "WORKER_IMAGE", "fake-image")
    monkeypatch.setattr(job_execution.runpod_worker, "worker_image_for_gpu", lambda g: "fake-image")
    monkeypatch.setattr(job_execution.runpod_worker, "DEFAULT_EXECUTION_TIMEOUT_MS", 3600000)
    monkeypatch.setattr(job_execution, "apply_disk_gb", lambda c, d: None)
    monkeypatch.setattr(
        job_execution, "time", type("T", (), {"sleep": staticmethod(lambda s: None)})()
    )


def test_deploy_train_endpoint_retries_on_quota_error(monkeypatch):
    """On a workers-quota error, deploy_train_endpoint sweeps idle endpoints and retries."""
    import flash.providers.runpod.execution.job_execution as job_execution
    import flash.providers.runpod.execution.jobs as jobs

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

    monkeypatch.setattr(runpod_resources, "_sweep_idle_flash_endpoints", fake_sweep)

    ep_id, _ep_name, fingerprint = job_execution.deploy_train_endpoint(
        "A100", name_suffix="testrun"
    )

    assert ep_id == "ep-new"
    assert fingerprint == job_execution.runpod_api.key_fingerprint("test-key")
    assert attempts["count"] == 3, "should take 3 attempts (2 quota failures + 1 success)"
    assert swept["count"] == 2, "should sweep once per quota-error retry"


def test_deploy_train_endpoint_raises_after_max_quota_retries(monkeypatch):
    """deploy_train_endpoint re-raises the quota error after all retries are exhausted."""
    import flash.providers.runpod.execution.job_execution as job_execution
    import flash.providers.runpod.execution.jobs as jobs

    class FakeRM:
        async def get_or_deploy_resource(self, config):
            raise RuntimeError(
                "GraphQL errors: Max workers across all endpoints must not exceed "
                "your workers quota (30)"
            )

    _patch_deploy_deps(monkeypatch, jobs)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)
    monkeypatch.setattr(
        runpod_resources,
        "_sweep_idle_flash_endpoints",
        lambda protected, min_idle_s=0.0, reap_warm=True: 0,
    )

    with pytest.raises(RuntimeError, match="workers quota"):
        job_execution.deploy_train_endpoint("A100", name_suffix="testrun")


def test_deploy_fails_over_to_next_account_on_quota(monkeypatch):
    """A multi-account RUNPOD_API_KEY fails the deploy over to the next account on quota."""
    import flash.providers.runpod.client.auth as keys
    import flash.providers.runpod.execution.job_execution as job_execution
    import flash.providers.runpod.execution.jobs as jobs

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

    _patch_deploy_deps(monkeypatch, jobs, set_key=False)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)
    monkeypatch.setattr(
        runpod_resources,
        "_sweep_idle_flash_endpoints",
        lambda protected, min_idle_s=0.0, reap_warm=True: 0,
    )

    ep_id, _name, _fingerprint = job_execution.deploy_train_endpoint("A100", name_suffix="testrun")
    assert ep_id == "ep-on-kB"
    assert keys.active_key() == "kB"  # provisioning pointer advanced to the working account


def test_deploy_fails_over_to_next_account_on_balance(monkeypatch):
    """An out-of-balance account fails the deploy over to the next account WITHOUT sweeping idle
    endpoints (sweeping can't add balance)."""
    import flash.providers.runpod.client.auth as keys
    import flash.providers.runpod.execution.job_execution as job_execution
    import flash.providers.runpod.execution.jobs as jobs

    monkeypatch.setenv("RUNPOD_API_KEY", "kA,kB")
    keys.reset()
    swept = {"count": 0}

    class FakeResource:
        id = "ep-on-kB"

    class FakeRM:
        async def get_or_deploy_resource(self, config):
            if keys.active_key() == "kA":
                raise RuntimeError(
                    "GraphQL errors: You must have at least $0.01 in your account "
                    "balance to create an endpoint."
                )
            return FakeResource()

    _patch_deploy_deps(monkeypatch, jobs, set_key=False)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)

    def fake_sweep(protected, min_idle_s=0.0, reap_warm=True):
        swept["count"] += 1
        return 0

    monkeypatch.setattr(runpod_resources, "_sweep_idle_flash_endpoints", fake_sweep)

    ep_id, _name, _fingerprint = job_execution.deploy_train_endpoint("A100", name_suffix="testrun")
    assert ep_id == "ep-on-kB"
    assert keys.active_key() == "kB"
    assert swept["count"] == 0, "balance failover must not sweep idle endpoints (can't add balance)"


def test_deploy_balance_error_single_account_raises_fast(monkeypatch):
    """A balance error on the only account re-raises (not swallowed) and fails fast without
    sweep-and-retry on the same broke account."""
    import flash.providers.runpod.client.auth as keys
    import flash.providers.runpod.execution.job_execution as job_execution
    import flash.providers.runpod.execution.jobs as jobs

    monkeypatch.setenv("RUNPOD_API_KEY", "solo-key")
    keys.reset()
    attempts = {"count": 0}

    class FakeRM:
        async def get_or_deploy_resource(self, config):
            attempts["count"] += 1
            raise RuntimeError(
                "GraphQL errors: You must have at least $0.01 in your account "
                "balance to create an endpoint."
            )

    _patch_deploy_deps(monkeypatch, jobs, set_key=False)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)
    monkeypatch.setattr(
        runpod_resources,
        "_sweep_idle_flash_endpoints",
        lambda protected, min_idle_s=0.0, reap_warm=True: 0,
    )

    with pytest.raises(RuntimeError, match="account balance"):
        job_execution.deploy_train_endpoint("A100", name_suffix="testrun")
    assert attempts["count"] == 1, "balance error must fail fast, not sweep-retry the broke account"


def test_deploy_captures_owner_before_concurrent_failover_after_sdk_create(monkeypatch):
    """A second deploy may fail over immediately after account A creates its endpoint. The returned
    fingerprint must remain account A's, never the newly active account B's."""
    import threading

    import flash.providers.runpod.client.auth as keys
    import flash.providers.runpod.execution.job_execution as job_execution
    import flash.providers.runpod.execution.jobs as jobs

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

    _patch_deploy_deps(monkeypatch, jobs, set_key=False)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)
    monkeypatch.setattr(runpod_endpoints, "FLASH_SDK_LOCK", FailoverAfterCreateLock())

    endpoint_id, _name, fingerprint = job_execution.deploy_train_endpoint(
        "A100", name_suffix="testrun"
    )

    assert endpoint_id == "ep-on-account-a"
    assert keys.active_key() == "account-b"
    assert fingerprint == job_execution.runpod_api.key_fingerprint("account-a")


def test_deploy_raises_when_all_accounts_exhausted_without_looping(monkeypatch):
    """When EVERY account is quota-exhausted, deploy fails over once per account and then RAISES —
    it must NOT loop forever. The deploy bounds its failovers by a key_count()-based COUNT (NOT by
    advance_key()'s return value, which always advances/wraps for a multi-key pool); a regression
    would spin here indefinitely."""
    import flash.providers.runpod.client.auth as keys
    import flash.providers.runpod.execution.job_execution as job_execution
    import flash.providers.runpod.execution.jobs as jobs

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

    _patch_deploy_deps(monkeypatch, jobs, set_key=False)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)
    monkeypatch.setattr(
        runpod_resources,
        "_sweep_idle_flash_endpoints",
        lambda protected, min_idle_s=0.0, reap_warm=True: 0,
    )

    with pytest.raises(RuntimeError, match="workers quota"):
        job_execution.deploy_train_endpoint("A100", name_suffix="testrun")
    # 2 accounts x _QUOTA_MAX_RETRIES (3) = 6 deploy attempts, then it stops — never the unbounded spin.
    assert calls["count"] == 6


def test_deploy_failover_from_midpool_tries_every_remaining_account(monkeypatch):
    """Try every remaining account when failover starts mid-pool.

    Bound failovers by ``key_count() - 1``; treating wrap to index zero as exhaustion skips kA when
    starting from kB in the kA, kB, kC pool.
    """
    import flash.providers.runpod.client.auth as keys
    import flash.providers.runpod.execution.job_execution as job_execution
    import flash.providers.runpod.execution.jobs as jobs

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

    _patch_deploy_deps(monkeypatch, jobs, set_key=False)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)
    monkeypatch.setattr(
        runpod_resources,
        "_sweep_idle_flash_endpoints",
        lambda protected, min_idle_s=0.0, reap_warm=True: 0,
    )

    ep_id, _name, _fingerprint = job_execution.deploy_train_endpoint("A100", name_suffix="testrun")
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
    import flash.providers.runpod.client.api as runpod_api
    import flash.providers.runpod.execution.resources as runpod_resources

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
    runpod_resources._idle_since.clear()

    count = runpod_resources._sweep_idle_flash_endpoints(
        protected={"flash-a100-cur", "live-flash-a100-cur"}
    )

    # warm idle/ready (ep-warm-idle) is reaped too — the dominant leak the old scaled-to-zero rule
    # never caught; running/initializing stay, current-run endpoints are protected.
    assert count == 3
    assert sorted(deleted) == sorted(["ep-live-idle", "ep-warm-idle", "ep-bare-idle"])


def test_sweep_reap_warm_false_keeps_warm_endpoints(monkeypatch):
    """reap_warm=False (the deploy-time reactive sweep, which protects only the current run) reaps
    ONLY fully scaled-to-zero endpoints — never another run's warm idle/ready leftover one."""
    import flash.providers.runpod.client.api as runpod_api
    import flash.providers.runpod.execution.resources as runpod_resources

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
    runpod_resources._idle_since.clear()
    assert runpod_resources._sweep_idle_flash_endpoints(protected=set(), reap_warm=False) == 1
    assert deleted == ["ep-zero"]

    # Periodic-reaper mode (default): the warm endpoint is reaped too.
    deleted.clear()
    runpod_resources._idle_since.clear()
    assert runpod_resources._sweep_idle_flash_endpoints(protected=set()) == 2
    assert sorted(deleted) == sorted(["ep-warm", "ep-zero"])


def test_sweep_idle_grace_requires_sustained_idleness(monkeypatch):
    """With min_idle_s > 0, an endpoint that reports a single transient zero (cold start / between
    jobs) is NOT deleted; only one idle across sweeps for >= min_idle_s is reaped."""
    import flash.providers.runpod.client.api as runpod_api
    import flash.providers.runpod.execution.resources as runpod_resources

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
    runpod_resources._idle_since.clear()

    clock = {"t": 1000.0}
    monkeypatch.setattr(runpod_resources.time, "time", lambda: clock["t"])

    # First sweep: idle observed, but grace (300s) not elapsed -> not deleted, timer recorded.
    assert runpod_resources._sweep_idle_flash_endpoints(protected=set(), min_idle_s=300.0) == 0
    assert deleted == []
    assert "ep-x" in runpod_resources._idle_since

    # Still within grace -> still not deleted.
    clock["t"] = 1200.0
    assert runpod_resources._sweep_idle_flash_endpoints(protected=set(), min_idle_s=300.0) == 0
    assert deleted == []

    # Past grace -> reaped.
    clock["t"] = 1400.0
    assert runpod_resources._sweep_idle_flash_endpoints(protected=set(), min_idle_s=300.0) == 1
    assert deleted == ["ep-x"]


def test_sweep_grace_resets_when_endpoint_becomes_busy(monkeypatch):
    """A busy reading clears the grace timer, so the idle clock restarts if it goes idle again —
    a long-running endpoint that dips idle briefly is never reaped."""
    import flash.providers.runpod.client.api as runpod_api
    import flash.providers.runpod.execution.resources as runpod_resources

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
    runpod_resources._idle_since.clear()
    clock = {"t": 1000.0}
    monkeypatch.setattr(runpod_resources.time, "time", lambda: clock["t"])

    # idle at t=1000 -> timer set
    assert runpod_resources._sweep_idle_flash_endpoints(protected=set(), min_idle_s=300.0) == 0
    assert "ep-x" in runpod_resources._idle_since
    # busy at t=1200 -> timer cleared
    state["busy"] = True
    clock["t"] = 1200.0
    assert runpod_resources._sweep_idle_flash_endpoints(protected=set(), min_idle_s=300.0) == 0
    assert "ep-x" not in runpod_resources._idle_since
    # idle again at t=1400 -> fresh timer (not deleted: only 0s of new idleness)
    state["busy"] = False
    clock["t"] = 1400.0
    assert runpod_resources._sweep_idle_flash_endpoints(protected=set(), min_idle_s=300.0) == 0
    assert deleted == []


def test_sweep_serializes_on_idle_since_lock(monkeypatch):
    """_idle_since access is guarded: a sweep blocks while another holds the lock (the periodic
    reaper and a deploy-time sweep run on different threads, so the prune can't race mid-iteration)."""
    import threading

    import flash.providers.runpod.client.api as runpod_api
    import flash.providers.runpod.execution.resources as runpod_resources

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
    runpod_resources._idle_since.clear()

    done = threading.Event()

    def run_sweep():
        runpod_resources._sweep_idle_flash_endpoints(protected=set())
        done.set()

    with runpod_resources._idle_since_lock:
        t = threading.Thread(target=run_sweep)
        t.start()
        # The sweep must block on the lock we hold -> it cannot finish.
        assert not done.wait(0.2)
    t.join(timeout=2)
    assert done.is_set()  # completes as soon as the lock is released


def test_runpod_completed_metrics_undecodable_output_pending_within_grace(monkeypatch):
    # regression (#613): a terminal-ok RunPod job whose output metrics are present but not yet
    # DECODABLE (decode_output raises, not merely returns a non-dict) must be treated as pending
    # within the recovery grace. previously the raise fell through to the broad handler and
    # returned None, letting callers tear down / resubmit a job that had already completed.
    import time as _time

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs
    from flash.runner.supervise import lifecycle

    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda eid, jid, **_kw: {"status": "COMPLETED", "output": {"present": "but-bad"}},
    )

    def _raise(_output):
        raise RuntimeError("output envelope not decodable yet")

    monkeypatch.setattr(jobs, "decode_output", _raise)
    handle = _runpod_handle(jobs)
    now = _time.time()
    # within the grace window -> keep reconciling (raise pending), never return None
    with pytest.raises(lifecycle._CompletedAttemptPending):
        lifecycle._runpod_completed_metrics(handle, deadline_at=now + 10_000.0)
    # once the grace has expired -> give up (return None)
    assert lifecycle._runpod_completed_metrics(handle, deadline_at=now - 10_000.0) is None


def test_runpod_completed_metrics_probes_after_expired_recovery_grace(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs
    from flash.runner.supervise import lifecycle

    now = 1_000.0
    monkeypatch.setattr(lifecycle.time, "time", lambda: now)
    probe_deadlines = []

    def completed_status(_endpoint_id, _job_id, **kwargs):
        probe_deadlines.append(kwargs["deadline_at"])
        return {"status": "COMPLETED", "output": {"wall_seconds": 60.0}}

    monkeypatch.setattr(runpod_api, "job_status", completed_status)
    metrics = lifecycle._runpod_completed_metrics(
        _live_clock_handle(jobs),
        deadline_at=now - lifecycle._RECOVERY_MARKER_GRACE_S - 1.0,
    )

    assert metrics == {"wall_seconds": 60.0}
    assert probe_deadlines == [now + lifecycle._RUNPOD_STATUS_PROBE_TIMEOUT_S]


def test_runpod_completed_metrics_caps_probe_deadline_when_wall_deadline_far_future(monkeypatch):
    # regression (#613): the status probe timeout must stay a short fresh cap even when the run's
    # wall deadline is hours away. otherwise job_status is handed a multi-hour deadline and, during
    # a runpod api outage, burns its full per-request retry budget (~minutes) before returning,
    # stalling the attach/recovery reconciler each pass. the wall+grace value governs only the
    # pending-output decision, never the per-probe timeout.
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs
    from flash.runner.supervise import lifecycle

    now = 1_000.0
    monkeypatch.setattr(lifecycle.time, "time", lambda: now)
    probe_deadlines = []

    def completed_status(_endpoint_id, _job_id, **kwargs):
        probe_deadlines.append(kwargs["deadline_at"])
        return {"status": "COMPLETED", "output": {"wall_seconds": 60.0}}

    monkeypatch.setattr(runpod_api, "job_status", completed_status)
    metrics = lifecycle._runpod_completed_metrics(
        _live_clock_handle(jobs),
        deadline_at=now + 7_200.0,  # wall deadline hours in the future
    )

    assert metrics == {"wall_seconds": 60.0}
    # short fresh cap, NOT now + 7200 + grace (which would let an outage stall the reconciler)
    assert probe_deadlines == [now + lifecycle._RUNPOD_STATUS_PROBE_TIMEOUT_S]


def test_runpod_completed_metrics_readable_failure_not_pending(monkeypatch):
    # regression (#613): a terminal-ok RunPod job whose output is a READABLE worker-failure
    # envelope (success=False) is a definitive completion-with-failure, not lagging metrics.
    # it must return None (not raise _CompletedAttemptPending), so callers take the failed
    # path instead of reconciling a job that already failed.
    import time as _time

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs
    from flash.runner.supervise import lifecycle

    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda eid, jid, **_kw: {
            "status": "COMPLETED",
            "output": {"success": False, "error": "boom"},
        },
    )
    handle = _runpod_handle(jobs)
    now = _time.time()
    # even well within the grace window, a readable failure envelope is never pending
    assert lifecycle._runpod_completed_metrics(handle, deadline_at=now + 10_000.0) is None


def test_deploy_train_endpoint_threads_gpu_count(monkeypatch):
    """gpu.count from the job spec becomes the runpod Endpoint gpu_count (multi-gpu pod)."""
    import sys

    import flash.providers.runpod.execution.job_execution as job_execution
    import flash.providers.runpod.execution.jobs as jobs
    from flash.core.spec import GpuSpec, JobSpec

    captured: dict = {}

    class FakeResource:
        id = "ep-multi"

    class FakeRM:
        async def get_or_deploy_resource(self, config):
            return FakeResource()

    _patch_deploy_deps(monkeypatch, jobs)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)

    class CapturingEndpoint:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def _build_resource_config(self):
            return {}

    sys.modules["runpod_flash"].Endpoint = CapturingEndpoint

    spec = JobSpec(gpu=GpuSpec(count=2))
    # endpoint_kwargs={} bypasses weight_cache_endpoint_kwargs so the base gpu_count is asserted directly.
    job_execution.deploy_train_endpoint(
        "A100", name_suffix="testrun", spec=spec, endpoint_kwargs={}
    )
    assert captured["gpu_count"] == 2


def test_deploy_train_endpoint_gpu_count_defaults_to_one(monkeypatch):
    """No spec keeps the historical single-gpu Endpoint payload (count == 1)."""
    import sys

    import flash.providers.runpod.execution.job_execution as job_execution
    import flash.providers.runpod.execution.jobs as jobs

    captured: dict = {}

    class FakeResource:
        id = "ep-single"

    class FakeRM:
        async def get_or_deploy_resource(self, config):
            return FakeResource()

    _patch_deploy_deps(monkeypatch, jobs)
    _make_runpod_flash_mocks(monkeypatch, FakeRM)

    class CapturingEndpoint:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def _build_resource_config(self):
            return {}

    sys.modules["runpod_flash"].Endpoint = CapturingEndpoint

    job_execution.deploy_train_endpoint("A100", name_suffix="testrun", endpoint_kwargs={})
    assert captured["gpu_count"] == 1


def _poll_in_queue_forever(monkeypatch, **poll_kwargs):
    """Drive poll_job against a job that never leaves IN_QUEUE (no capacity for the pinned class)."""
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs, polling

    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: (_ for _ in ()).throw(RuntimeError("no workers yet")),
    )
    monkeypatch.setattr(polling.time, "sleep", lambda s: None)
    clock = itertools.count(start=0, step=100.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    return polling.poll_job(
        _runpod_handle(jobs),
        interval_s=0,
        heartbeat_reader=lambda: None,
        setup_grace_s=5000.0,
        queue_grace_s=900.0,
        **poll_kwargs,
    )


def test_capacity_detail_contains_no_retry_policy(monkeypatch):
    """The provider reports capacity evidence while the supervisor owns retry policy."""
    result = _poll_in_queue_forever(monkeypatch)

    assert result.failure == "no_capacity"
    assert "next-best" not in result.detail
    assert "retrying" not in result.detail
    assert "GPU-class escalation" not in result.detail


def test_reattach_keeps_the_stall_grace_but_not_the_capacity_wording(monkeypatch):
    """Use the persisted last-GPU flag only for the scarcity grace.

    Recovery rebuilds an unpinned candidate walk, so the stale flag must not constrain capacity
    wording even though it still selects the longer wait.
    """
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core.base import JobHandle
    from flash.providers.runpod.execution import jobs as jobs
    from flash.providers.runpod.execution.provider import PROVIDER

    captured: dict = {}

    def fake_poll_job(handle, **kw):
        captured.update(kw)
        return jobs.PollResult(True, metrics={})

    monkeypatch.setattr(polling, "poll_job", fake_poll_job)
    spec = JobSpec(
        run_id="reattach-lastgpu",
        model="Qwen/Qwen3.5-9B",
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
        "attempt": 2,
    }

    PROVIDER.poll_attempt(JobHandle.from_dict({**base, "on_last_gpu": True}), spec)
    # the capacity wording is left at poll_job's neutral default: escalation may follow, because
    # after recovery it genuinely can.
    assert "on_last_gpu" not in captured, captured
    # the scarcity grace still honours the snapshot: 900s, not the 300s of a normal attempt.
    assert captured["queue_grace_s"] == 900.0, captured
    assert captured["throttled_grace_s"] == 900.0, captured

    captured.clear()
    PROVIDER.poll_attempt(JobHandle.from_dict({**base, "on_last_gpu": False}), spec)
    assert "on_last_gpu" not in captured, captured
    assert captured["queue_grace_s"] == 300.0, captured
