"""Durable run primitives: handle persistence, polling state machine, supervisor retry,
cross-process cancel, and attach (CPU-only; all network mocked)."""

from __future__ import annotations

import io
import os
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
    """Return a fenced handle whose launch timestamp uses the real clock."""
    return _runpod_handle(jobs, started_ts=time.time())


def _runpod_handle(
    jobs,
    endpoint_id="ep",
    endpoint_name="name",
    job_id="job",
    attempt=0,
    fence=1,
    started_ts=1.0,
):
    return jobs.JobHandle(
        endpoint_id,
        endpoint_name,
        _RUNPOD_FINGERPRINT,
        job_id,
        attempt,
        fence,
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
    from flash.providers.runpod.execution.jobs import JobHandle

    handle = JobHandle(
        "ep123",
        "flash-5090-abc",
        _RUNPOD_FINGERPRINT,
        "job456",
        2,
        7,
        12_345.0,
    )
    assert JobHandle.from_dict(handle.to_dict()) == handle
    endpoint_only = JobHandle(
        "ep-cleanup",
        "flash-cleanup",
        _RUNPOD_FINGERPRINT,
        None,
        3,
        8,
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
        "fence",
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
            "fence": 1,
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
        "fence": 1,
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
            "fence": 1,
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
            "fence": 1,
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
# runpod resource and fenced-result polling
# ---------------------------------------------------------------------------
def _poll_spec():
    return SimpleNamespace(
        run_id="run-poll",
        phase="sft",
        train=SimpleNamespace(hf_repo="org/repo"),
    )


def _attempt_record(
    *,
    attempt_id=0,
    fence=1,
    grant_deadline_at=5.0,
    work_deadline_at=200.0,
    result_deadline_at=220.0,
):
    from flash.runner.lifecycle.protocol import AttemptRecord

    return AttemptRecord.from_dict(
        {
            "attempt_id": attempt_id,
            "fence": fence,
            "state": "active",
            "reserved_at": 1.0,
            "grant_deadline_at": grant_deadline_at,
            "work_deadline_at": work_deadline_at,
            "result_deadline_at": result_deadline_at,
            "run_deadline_at": work_deadline_at,
            "provider": "runpod",
            "provider_contract": None,
            "resource": None,
            "allocation": None,
            "progress_receipt": None,
            "result_receipt": None,
            "cleanup": {},
            "schema_version": 1,
        }
    )


def _stepped_clock(start=0.0, step=10.0):
    value = start - step

    def now():
        nonlocal value
        value += step
        return value

    return now


def _wire_runpod_poll(monkeypatch, *, attempt=None, results=()):
    from flash.providers.runpod.execution import polling

    result_iter = iter(results)
    monkeypatch.setattr(
        polling,
        "_current_attempt",
        lambda _run_id, _handle: (attempt or _attempt_record(), dict(_SOURCE_SNAPSHOT)),
    )
    monkeypatch.setattr(polling, "_record_resource", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(polling, "_observe_artifacts", lambda _context: next(result_iter, None))
    monkeypatch.setattr(polling.time, "sleep", lambda _seconds: None)
    return polling


def test_poll_job_returns_current_fenced_result_before_provider_status(monkeypatch):
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(
        monkeypatch,
        results=[PollResult(True, metrics={"optimizer_steps": 2})],
    )
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda *_args, **_kwargs: pytest.fail("result authority must precede provider status"),
    )

    result = polling.poll_job(_runpod_handle(jobs), _poll_spec(), interval_s=0)

    assert result.ok
    assert result.metrics == {"optimizer_steps": 2}


@pytest.mark.parametrize(
    ("failure", "detail"),
    [("oom", "cuda out of memory"), ("job_failed", "worker error")],
)
def test_poll_job_returns_manifest_failure_without_provider_reclassification(
    monkeypatch, failure, detail
):
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(
        monkeypatch,
        results=[PollResult(False, failure=failure, detail=detail)],
    )
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda *_args, **_kwargs: pytest.fail("manifest failure is terminal authority"),
    )

    result = polling.poll_job(_runpod_handle(jobs), _poll_spec(), interval_s=0)

    assert result.failure == failure
    assert result.detail == detail


def test_poll_job_terminal_resource_waits_for_result_deadline(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(
        monkeypatch,
        attempt=_attempt_record(work_deadline_at=20.0, result_deadline_at=30.0),
        results=[None, None],
    )
    monkeypatch.setattr(polling.time, "time", _stepped_clock())
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda *_args, **_kwargs: {"status": "FAILED"},
    )

    result = polling.poll_job(_runpod_handle(jobs), _poll_spec(), interval_s=0)

    assert result.failure == "job_preempted"
    assert "FAILED" in result.detail
    assert "without a result manifest" in result.detail


@pytest.mark.parametrize("provider", ["lambda", "vast"])
def test_instance_terminal_resource_uses_bounded_result_visibility_window(monkeypatch, provider):
    from flash.providers._lifecycle.instances import poll_instance

    attempt = _attempt_record(work_deadline_at=10_000.0, result_deadline_at=10_120.0)
    monkeypatch.setattr(poll_instance, "_current_attempt", lambda _adapter: attempt)
    monkeypatch.setattr(poll_instance, "_record_resource", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(poll_instance, "_observe_result", lambda _adapter: None)
    clock = {"now": 100.0}
    monkeypatch.setattr(poll_instance.time, "time", lambda: clock["now"])
    monkeypatch.setattr(
        poll_instance.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    adapter = poll_instance.InstancePollAdapter(
        provider=provider,
        instance_id="instance-1",
        run_id="run-poll",
        current_attempt=0,
        fence=1,
        launch_ts=1.0,
        hf_repo="org/repo",
        phase="sft",
        source_snapshot=dict(_SOURCE_SNAPSHOT),
        fetch_instance=lambda: {"status": "terminated"},
        poll_error_exceptions=(RuntimeError,),
        status_field="status",
        running_status="running",
        dead_states=frozenset({"terminated"}),
        missing_dead_threshold=2,
        stamp_cost_and_notes=lambda *_args, **_kwargs: None,
    )

    result = poll_instance.poll_instance_job(adapter, interval_s=20.0)

    assert result.failure == "job_preempted"
    assert clock["now"] == 220.0


@pytest.mark.parametrize(
    ("provider", "running_status", "dead_status", "intermediate_status"),
    [
        ("lambda", "active", "terminated", "booting"),
        ("vast", "running", "exited", "loading"),
    ],
)
def test_instance_terminal_latch_survives_intermediate_recovery_state(
    monkeypatch,
    provider,
    running_status,
    dead_status,
    intermediate_status,
):
    from flash.providers._lifecycle.instances import poll_instance
    from flash.providers.core.base import PollResult

    attempt = _attempt_record(
        grant_deadline_at=90.0,
        work_deadline_at=10_000.0,
        result_deadline_at=10_120.0,
    )
    observations = iter([None] * 4 + [PollResult(True, metrics={"optimizer_steps": 11})])
    statuses = iter([running_status, dead_status, intermediate_status, running_status])
    monkeypatch.setattr(poll_instance, "_current_attempt", lambda _adapter: attempt)
    monkeypatch.setattr(poll_instance, "_record_resource", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(poll_instance, "_observe_result", lambda _adapter: next(observations))
    clock = {"now": 100.0}
    monkeypatch.setattr(poll_instance.time, "time", lambda: clock["now"])
    monkeypatch.setattr(
        poll_instance.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    adapter = poll_instance.InstancePollAdapter(
        provider=provider,
        instance_id="instance-1",
        run_id="run-poll",
        current_attempt=0,
        fence=1,
        launch_ts=1.0,
        hf_repo="org/repo",
        phase="sft",
        source_snapshot=dict(_SOURCE_SNAPSHOT),
        fetch_instance=lambda: {"status": next(statuses)},
        poll_error_exceptions=(RuntimeError,),
        status_field="status",
        running_status=running_status,
        dead_states=frozenset({dead_status}),
        missing_dead_threshold=2,
        stamp_cost_and_notes=lambda *_args, **_kwargs: None,
    )

    result = poll_instance.poll_instance_job(adapter, interval_s=20.0)

    assert result.ok
    assert result.metrics == {"optimizer_steps": 11}
    assert clock["now"] == 180.0


def test_instance_terminal_manifest_inside_visibility_window_wins(monkeypatch):
    from flash.providers._lifecycle.instances import poll_instance
    from flash.providers.core.base import PollResult

    attempt = _attempt_record(work_deadline_at=10_000.0, result_deadline_at=10_120.0)
    observations = iter((None, PollResult(True, metrics={"optimizer_steps": 7})))
    monkeypatch.setattr(poll_instance, "_current_attempt", lambda _adapter: attempt)
    monkeypatch.setattr(poll_instance, "_record_resource", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(poll_instance, "_observe_result", lambda _adapter: next(observations))
    clock = {"now": 100.0}
    monkeypatch.setattr(poll_instance.time, "time", lambda: clock["now"])
    monkeypatch.setattr(
        poll_instance.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    adapter = poll_instance.InstancePollAdapter(
        provider="lambda",
        instance_id="instance-1",
        run_id="run-poll",
        current_attempt=0,
        fence=1,
        launch_ts=1.0,
        hf_repo="org/repo",
        phase="sft",
        source_snapshot=dict(_SOURCE_SNAPSHOT),
        fetch_instance=lambda: {"status": "terminated"},
        poll_error_exceptions=(RuntimeError,),
        status_field="status",
        running_status="running",
        dead_states=frozenset({"terminated"}),
        missing_dead_threshold=2,
        stamp_cost_and_notes=lambda *_args, **_kwargs: None,
    )

    result = poll_instance.poll_instance_job(adapter, interval_s=20.0)

    assert result.ok
    assert result.metrics == {"optimizer_steps": 7}
    assert clock["now"] == 120.0


def test_runpod_terminal_resource_uses_bounded_result_visibility_window(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(
        monkeypatch,
        attempt=_attempt_record(work_deadline_at=10_000.0, result_deadline_at=10_120.0),
        results=[None] * 20,
    )
    clock = {"now": 100.0}
    monkeypatch.setattr(polling.time, "time", lambda: clock["now"])
    monkeypatch.setattr(
        polling.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda *_args, **_kwargs: {"status": "FAILED"},
    )

    result = polling.poll_job(_runpod_handle(jobs), _poll_spec(), interval_s=20.0)

    assert result.failure == "job_preempted"
    assert clock["now"] == 220.0


def test_runpod_terminal_manifest_inside_visibility_window_wins(monkeypatch):
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(
        monkeypatch,
        attempt=_attempt_record(work_deadline_at=10_000.0, result_deadline_at=10_120.0),
        results=[None, PollResult(True, metrics={"optimizer_steps": 7})],
    )
    clock = {"now": 100.0}
    monkeypatch.setattr(polling.time, "time", lambda: clock["now"])
    monkeypatch.setattr(
        polling.time,
        "sleep",
        lambda seconds: clock.__setitem__("now", clock["now"] + seconds),
    )
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda *_args, **_kwargs: {"status": "FAILED"},
    )

    result = polling.poll_job(_runpod_handle(jobs), _poll_spec(), interval_s=20.0)

    assert result.ok
    assert result.metrics == {"optimizer_steps": 7}
    assert clock["now"] == 120.0


def test_poll_job_running_without_progress_uses_fixed_attempt_deadline(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(
        monkeypatch,
        attempt=_attempt_record(work_deadline_at=25.0, result_deadline_at=45.0),
        results=[None, None, None],
    )
    monkeypatch.setattr(polling.time, "time", _stepped_clock())
    statuses = []

    def status(*_args, **_kwargs):
        statuses.append("IN_PROGRESS")
        return {"status": "IN_PROGRESS"}

    monkeypatch.setattr(runpod_api, "job_status", status)

    result = polling.poll_job(
        _runpod_handle(jobs),
        _poll_spec(),
        interval_s=0,
        deadline_at=10_000.0,
    )

    assert statuses
    assert result.failure == "job_preempted"
    assert "work deadline expired" in result.detail


def test_poll_job_queue_deadline_remains_no_capacity(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(monkeypatch, results=[None, None, None])
    monkeypatch.setattr(polling.time, "time", _stepped_clock())
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda *_args, **_kwargs: {"status": "IN_QUEUE"},
    )
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("not visible yet")),
    )

    result = polling.poll_job(
        _runpod_handle(jobs),
        _poll_spec(),
        interval_s=0,
        queue_grace_s=5.0,
    )

    assert result.failure == "no_capacity"
    assert "IN_QUEUE" in result.detail


@pytest.mark.parametrize(
    ("workers", "grace_name", "failure"),
    [
        ({"unhealthy": 1}, "unhealthy_grace_s", "job_preempted"),
        ({"throttled": 1}, "throttled_grace_s", "no_capacity"),
    ],
)
def test_poll_job_preserves_provider_health_observations(monkeypatch, workers, grace_name, failure):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(
        monkeypatch,
        attempt=_attempt_record(grant_deadline_at=100.0),
        results=[None, None, None],
    )
    monkeypatch.setattr(polling.time, "time", _stepped_clock())
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda *_args, **_kwargs: {"status": "IN_QUEUE"},
    )
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: {"workers": workers},
    )
    kwargs = {grace_name: 5.0, "queue_grace_s": 1_000.0}

    result = polling.poll_job(
        _runpod_handle(jobs),
        _poll_spec(),
        interval_s=0,
        **kwargs,
    )

    assert result.failure == failure


def test_poll_job_unhealthy_queue_does_not_latch_grant_or_expire_queue_first(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(
        monkeypatch,
        attempt=_attempt_record(
            grant_deadline_at=100.0,
            work_deadline_at=200.0,
            result_deadline_at=220.0,
        ),
        results=[None] * 10,
    )
    clock = iter((0.0, 0.0, 1.0, 1.0, 6.0, 6.0, 6.0, 6.0))
    monkeypatch.setattr(polling.time, "time", lambda: next(clock, 6.0))
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda *_args, **_kwargs: {"status": "IN_QUEUE"},
    )
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: {"workers": {"unhealthy": 1}},
    )

    result = polling.poll_job(
        _runpod_handle(jobs),
        _poll_spec(),
        interval_s=0,
        queue_grace_s=1.0,
        unhealthy_grace_s=5.0,
    )

    assert result.failure == "job_preempted"
    assert result.detail == "RunPod worker remained unhealthy"


def test_poll_job_queue_health_reset_cannot_extend_grant_deadline(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(
        monkeypatch,
        attempt=_attempt_record(
            grant_deadline_at=5.0,
            work_deadline_at=200.0,
            result_deadline_at=220.0,
        ),
        results=[None] * 10,
    )
    observed_times = []
    clock = _stepped_clock(step=1.0)

    def now():
        value = clock()
        observed_times.append(value)
        return value

    health = iter(({"workers": {"unhealthy": 1}}, {"workers": {}}))
    health_calls = []
    monkeypatch.setattr(polling.time, "time", now)
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda *_args, **_kwargs: {"status": "IN_QUEUE"},
    )
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda *_args, **_kwargs: health_calls.append(True) or next(health, {"workers": {}}),
    )

    result = polling.poll_job(
        _runpod_handle(jobs),
        _poll_spec(),
        interval_s=0,
        queue_grace_s=100.0,
        unhealthy_grace_s=100.0,
    )

    assert result.failure == "no_capacity"
    assert max(observed_times) == 5.0
    assert len(health_calls) == 2


def test_poll_job_recovers_transient_result_download_to_current_fenced_success(monkeypatch):
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(
        monkeypatch,
        attempt=_attempt_record(work_deadline_at=100.0, result_deadline_at=120.0),
        results=[],
    )
    observations = iter(
        [
            OSError("temporary result download failure"),
            PollResult(True, metrics={"optimizer_steps": 2}),
        ]
    )

    def observe(_context):
        observed = next(observations)
        if isinstance(observed, Exception):
            raise observed
        return observed

    monkeypatch.setattr(polling, "_observe_artifacts", observe)
    monkeypatch.setattr(polling.time, "time", _stepped_clock(step=1.0))
    monkeypatch.setattr(
        runpod_api, "job_status", lambda *_args, **_kwargs: {"status": "IN_PROGRESS"}
    )

    result = polling.poll_job(_runpod_handle(jobs), _poll_spec(), interval_s=0)

    assert result.ok
    assert result.metrics == {"optimizer_steps": 2}


def test_poll_job_recovers_transient_status_errors_to_current_fenced_result(monkeypatch):
    from flash.providers.core.base import PollResult
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(
        monkeypatch,
        attempt=_attempt_record(work_deadline_at=100.0, result_deadline_at=120.0),
        results=[None, None, None, PollResult(True, metrics={"optimizer_steps": 2})],
    )
    monkeypatch.setattr(polling.time, "time", _stepped_clock(step=1.0))
    calls = {"count": 0}

    def transient_status(*_args, **_kwargs):
        calls["count"] += 1
        raise runpod_api.RunpodApiError("temporary status failure")

    monkeypatch.setattr(runpod_api, "job_status", transient_status)

    result = polling.poll_job(_runpod_handle(jobs), _poll_spec(), interval_s=0)

    assert result.ok
    assert result.metrics == {"optimizer_steps": 2}
    assert calls["count"] == 3


def test_poll_job_bounds_status_transport_failures(monkeypatch):
    from flash.providers._lifecycle.instances import poll as poll_helpers
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(
        monkeypatch,
        attempt=_attempt_record(work_deadline_at=1_000.0, result_deadline_at=1_020.0),
        results=[None] * 10,
    )
    monkeypatch.setattr(polling.time, "time", _stepped_clock(step=1.0))
    monkeypatch.setattr(poll_helpers.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        runpod_api,
        "job_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(runpod_api.RunpodApiError("offline")),
    )

    result = polling.poll_job(_runpod_handle(jobs), _poll_spec(), interval_s=0)

    assert result.failure == "poll_error"


def test_runpod_provider_poll_rejects_endpoint_only_handle(monkeypatch):
    from flash.providers.core.base import JobHandle
    from flash.providers.runpod.execution import jobs
    from flash.providers.runpod.execution.provider import RunpodProvider

    monkeypatch.setattr(
        polling,
        "poll_job",
        lambda *_args, **_kwargs: pytest.fail("endpoint-only handle must not reach polling"),
    )
    handle = JobHandle.from_dict(
        _runpod_handle(jobs, job_id=None, started_ts=time.time()).to_dict()
    )

    spec = _spec("endpoint-only")
    with pytest.raises(ValueError, match="endpoint-only"):
        RunpodProvider().poll(handle, spec, seed=spec.seed)


def test_current_attempt_rejects_a_stale_runpod_fence(monkeypatch):
    from flash.providers.runpod.execution import jobs, polling
    from flash.runner.lifecycle import status as status_ops

    monkeypatch.setattr(
        status_ops,
        "get_status",
        lambda _run_id: SimpleNamespace(attempt=_attempt_record(fence=2).to_dict()),
    )

    with pytest.raises(RuntimeError, match="current fenced attempt"):
        polling._current_attempt("run-poll", _runpod_handle(jobs, fence=1))


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
def test_supervisor_adopts_provider_completion_before_retry(monkeypatch, cancel_during_status):
    from dataclasses import replace

    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.core.allocator as allocator
        import flash.runner.supervise.lifecycle as lifecycle
        from flash.providers.core import registry as providers
        from flash.providers.core.base import Allocation, Candidate, PollResult

        spec = _spec("completed-before-retry")
        spec = replace(spec, gpu=replace(spec.gpu, max_retries=0, count=4))
        runner_state._save_status(
            runner_state.RunStatus(run_id=spec.run_id, state="running", spec=spec.to_dict()),
            _next_attempt=0,
        )
        candidate = Candidate("vast", "H100 SXM", 2.5, 80, gpu_count=4)
        monkeypatch.setattr(
            allocator,
            "allocate",
            lambda *args, **kwargs: Allocation(
                provider="vast",
                gpu="H100 SXM",
                hourly_usd=2.5,
                min_vram_gb=80,
                candidates=(candidate,),
                gpu_count=4,
            ),
        )
        calls = {"n": 0}

        class Provider:
            supports_weight_cache = False

            def submit_run(self, spec, seed, log=None, on_handle=None, attempt=0, **_):
                calls["n"] += 1
                if on_handle:
                    on_handle(
                        {
                            "provider": "vast",
                            "instance_id": 42,
                            "offer_id": 7,
                            "machine_id": 9,
                            "label": "flash-completed-s0-a0",
                            "gpu": "H100 SXM",
                            "hourly_usd": 2.5,
                            "attempt": attempt,
                            "fence": 1,
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

        def completed_metrics(*_args, **_kwargs):
            if cancel_during_status:
                runner_status._update(spec.run_id, "cancelled")
            return {"wall_seconds": 5.0, "trained_eval_acc": 0.9}

        monkeypatch.setattr(runner_lifecycle, "_attempt_result_metrics", completed_metrics)

        if cancel_during_status:
            with pytest.raises(runner_errors._RunCancelled):
                lifecycle._submit_seed_supervised(
                    spec,
                    spec.seed,
                    io.StringIO(),
                    source_snapshot=_SOURCE_SNAPSHOT,
                )
        else:
            metrics = lifecycle._submit_seed_supervised(
                spec,
                spec.seed,
                io.StringIO(),
                source_snapshot=_SOURCE_SNAPSHOT,
            )
            assert metrics["trained_eval_acc"] == 0.9
            assert metrics["allocated_provider"] == "vast"
            assert metrics["allocated_gpu"] == "H100 SXM"
            assert metrics["allocated_gpu_count"] == 4
        assert calls["n"] == 1


def test_supervisor_retries_on_provider_loss_then_succeeds(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        calls = {"n": 0}
        source_snapshots: list[dict | None] = []

        def fake_submit(
            spec,
            seed,
            log=None,
            on_handle=None,
            attempt=0,
            fence=1,
            source_snapshot=None,
            **_,
        ):
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
                        "fence": fence,
                        "started_ts": 1.0,
                    }
                )
            if calls["n"] == 1:
                return jobs.PollResult(False, failure="job_preempted", detail="frozen")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        runner_submit.submit_job(_spec("retry-ok"), dry_run=False, background=False)
        st = runner_status.get_status("retry-ok")
        assert st.state == "done"
        assert calls["n"] == 2
        assert source_snapshots == [_SOURCE_SNAPSHOT, _SOURCE_SNAPSHOT]
        assert st.remote["job_id"] == "j2"  # latest handle persisted


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

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
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
                "train": {**base["train"], "init_from_adapter": "source-run"},
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
                "train": {**base["train"], "init_from_adapter": "source-run"},
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
                    "init_from_adapter": "source-run",
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


def test_submit_rejects_bare_init_ref_to_unfinished_source_run(monkeypatch):
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
                "train": {**base["train"], "init_from_adapter": "source-run"},
            }
        )

        with pytest.raises(ValueError, match="concrete source-run/step-N checkpoint"):
            runner_submit.submit_job(spec, dry_run=True, background=False)


def test_submit_rejects_bare_init_ref_without_final_adapter(monkeypatch):
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
                "train": {**base["train"], "init_from_adapter": "source-run"},
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
                "train": {**base["train"], "init_from_adapter": "source-run"},
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
                attempt=_attempt_record(
                    work_deadline_at=time.time() + 600.0,
                    result_deadline_at=time.time() + 660.0,
                ).to_dict(),
                remote={
                    "provider": "runpod",
                    "endpoint_id": "ep",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "job",
                    "attempt": 0,
                    "fence": 1,
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
            def poll(self, handle, spec, seed, **kwargs):
                polled.update(
                    init_from_adapter=spec.train.init_from_adapter,
                    revision=spec.train.init_from_adapter_revision,
                    lora_rank=spec.train.lora_rank,
                )
                return PollResult(True, metrics={"wall_seconds": 1.0})

        monkeypatch.setattr(providers, "get_provider", lambda name: Provider())
        monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda *a, **k: None)
        monkeypatch.setattr(
            runner_lifecycle,
            "_adopt_completed_attempt",
            lambda *_args, **_kwargs: runner_status._update("warm-recover", "done"),
        )

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
                attempt=_attempt_record(
                    work_deadline_at=time.time() + 600.0,
                    result_deadline_at=time.time() + 660.0,
                ).to_dict(),
                billing_context={"org_id": "org-a"},
                remote={
                    "provider": "runpod",
                    "endpoint_id": "ep",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "job",
                    "attempt": 0,
                    "fence": 1,
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
            rank_mod, "load_hf_adapter_config", lambda *a, **k: _adapter_config(rank=64)
        )
        monkeypatch.setattr(rank_mod, "adapter_artifact_identity", lambda *a, **k: identity)
        launched: dict[str, object] = {}
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(
                False,
                failure="job_preempted",
                detail="provider resource was lost",
            ),
        )
        monkeypatch.setattr(runner_lifecycle, "_attempt_result_metrics", lambda *a, **k: None)
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
                attempt=_attempt_record(
                    work_deadline_at=time.time() + 600.0,
                    result_deadline_at=time.time() + 660.0,
                ).to_dict(),
                remote={
                    "provider": "runpod",
                    "endpoint_id": "ep",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "job",
                    "attempt": 0,
                    "fence": 1,
                    "started_ts": 1.0,
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
            polling,
            "poll_job",
            lambda *a, **k: (
                polls.append("polled")
                or jobs.PollResult(
                    False,
                    failure="job_preempted",
                    detail="provider resource was lost",
                )
            ),
        )
        monkeypatch.setattr(runner_lifecycle, "_attempt_result_metrics", lambda *a, **k: None)
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
            "fence": 1,
            "started_ts": 1.0,
        }
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="warm-recover",
                state="running",
                spec=spec.to_dict(),
                attempt=_attempt_record().to_dict(),
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
            "fence": 1,
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
        ] == [remote]


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
            "fence": 1,
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

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, fence=1, **_):
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
                        "fence": fence,
                        "started_ts": 1.0,
                    }
                )
            return jobs.PollResult(True, metrics={"cost_usd": 0.1})

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
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

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, fence=1, **_):
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
                        "fence": fence,
                        "started_ts": 1.0,
                    }
                )
            if calls["n"] == 1:
                return jobs.PollResult(False, failure="job_preempted", detail="[CANCELLED] None")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
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

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, fence=1, **_):
            calls["n"] += 1
            return jobs.PollResult(
                False, failure="job_failed", detail="Remote execution failed: ValueError"
            )

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        with pytest.raises(RuntimeError):
            runner_submit.submit_job(_spec("fail-fast"), dry_run=False, background=False)
        assert calls["n"] == 1
        assert runner_status.get_status("fail-fast").state == "failed"


def test_supervisor_infra_failure_retries_up_to_floor(monkeypatch):
    """A streak of infra-shaped failures (broken or lost GPU -> job_preempted) walks past up to
    INFRA_RETRY_FLOOR hosts even though the spec's max_retries is only 2 — so a run of bad GPUs finds a
    healthy host instead of dying on the small default budget. (Genuine worker errors still fail fast:
    test_supervisor_does_not_retry_worker_code_errors.)"""
    from flash.runner.supervise.lifecycle import INFRA_RETRY_FLOOR

    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        calls = {"n": 0}

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, fence=1, **_):
            calls["n"] += 1
            return jobs.PollResult(False, failure="job_preempted", detail="GPU never became ready")

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        with pytest.raises(RuntimeError):
            runner_submit.submit_job(
                _spec("infra-floor"), dry_run=False, background=False
            )  # max_retries=2
        # floor=5 -> 6 attempts (walk 0..5), NOT the 3 the raw max_retries=2 would give.
        assert calls["n"] == INFRA_RETRY_FLOOR + 1
        assert runner_status.get_status("infra-floor").state == "failed"


def test_supervisor_infra_floor_respects_explicit_zero_retries(monkeypatch):
    """An explicit max_retries=0 (deliberate single-shot) is NOT forced to retry by the infra floor."""
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec

    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        calls = {"n": 0}

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, fence=1, **_):
            calls["n"] += 1
            return jobs.PollResult(False, failure="job_preempted", detail="frozen")

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
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

        def fake_submit(
            spec, seed, log=None, on_handle=None, attempt=0, fence=1, on_last_gpu=False, **_
        ):
            submissions.append((attempt, spec.gpu.network_volume, on_last_gpu))
            return jobs.PollResult(False, failure=failure, detail="cache-constrained failure")

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
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

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, fence=1, **_):
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
                        "fence": fence,
                        "started_ts": 1.0,
                    }
                )
            if attempt < 2:
                return jobs.PollResult(False, failure="job_preempted", detail="frozen")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
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

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, fence=1, **_):
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
                        "fence": fence,
                        "started_ts": 1.0,
                    }
                )
            if attempt == 0:
                return jobs.PollResult(False, failure="oom", detail="vLLM free-memory preflight")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
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

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, fence=1, **_):
            calls["n"] += 1
            return jobs.PollResult(
                False, failure="job_failed", detail="ValueError: bad reward fn (no infra marker)"
            )

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
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


def test_supervisor_gpu_walk_exhausts_classes_then_retries_cheapest(monkeypatch):
    # after every distinct candidate is tried, retries wrap to the cheapest candidate rather than
    # clamping on the priciest or indexing past the list.
    import dataclasses

    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.core.allocator as allocator
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train
        from flash.core.spec import GpuSpec, JobSpec, TrainSpec

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

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, fence=1, **_):
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
                        "fence": fence,
                        "started_ts": 1.0,
                    }
                )
            if attempt < 2:
                return jobs.PollResult(False, failure="job_preempted", detail="frozen")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])

        spec = JobSpec(
            run_id="clamp",
            model="Qwen/Qwen3.5-9B",
            algorithm="grpo",
            train=TrainSpec(epochs=1, max_examples=1),
            gpu=GpuSpec(type="", max_retries=2),
        )
        runner_submit.submit_job(spec, dry_run=False, background=False)

        assert runner_status.get_status("clamp").state == "done"
        # Walk advances through both classes, then re-rolls the cheapest (never out of range).
        assert gpus_seen == ["A100 PCIe", "A100 SXM", "A100 PCIe"]


def test_supervisor_marks_on_last_gpu_only_at_end_of_walk(monkeypatch):
    # on_last_gpu must reach the provider so the no-capacity backstops know whether there is a
    # next-best class to fall to: False while the walk still has somewhere to go (attempt 0 on the
    # cheaper of two classes), True once it lands on (and clamps to) the last candidate.
    import dataclasses

    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.core.allocator as allocator
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train
        from flash.core.spec import GpuSpec, JobSpec, TrainSpec

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

        def fake_submit(
            spec, seed, log=None, on_handle=None, attempt=0, fence=1, on_last_gpu=False, **_
        ):
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
                        "fence": fence,
                        "started_ts": 1.0,
                    }
                )
            if attempt < 2:
                return jobs.PollResult(False, failure="job_preempted", detail="frozen")
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
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
        # attempt 0: cheaper class, a next-best still exists -> False; attempts 1 & 2: on the last
        # candidate (and clamped onto it) -> True.
        assert last_flags == [False, True, True]
        # the winning attempt persists on_last_gpu so reattachment keeps its queue-capacity window.
        assert runner_status.get_status("lastgpu").remote.get("on_last_gpu") is True


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

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, fence=1, **_):
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
                        "fence": fence,
                        "started_ts": 1.0,
                    }
                )
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
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

        def fake_submit(spec, seed, log=None, on_handle=None, attempt=0, fence=1, **_kwargs):
            submitted.append((spec.gpu.type, attempt))
            return jobs.PollResult(True, metrics={"cost_usd": 0.1, "trained_eval_acc": 0.9})

        monkeypatch.setattr(job_execution, "submit_run", fake_submit)
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
                "fence": 1,
                "started_ts": 1.0,
            },
        )
        runner_state._save_status(status)
        # the provider poll can fail after the current fenced result has become immutable. recovery
        # adopts only that result while preserving the allocated gpu stamp from the remote handle.
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="poll_error", detail="api outage"),
        )
        monkeypatch.setattr(
            runner_lifecycle,
            "_attempt_result_metrics",
            lambda *a, **k: {"wall_seconds": 3600.0},
        )
        monkeypatch.setattr(
            runner_status,
            "validate_terminal_source_metrics",
            lambda _status, metrics, **_kwargs: (metrics, None),
        )
        from flash.providers.runpod.client.pricing import hourly_rate

        monkeypatch.setattr(
            runner_costs,
            "_gpu_rate",
            lambda gpu, provider="": (
                hourly_rate(gpu)
                if provider == "runpod"
                else pytest.fail("recovery lost the allocated provider stamp")
            ),
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        st = runner_attach.attach_run("walked", log_stream=sys.stderr)

        assert st.state == "done"
        import json
        import os

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
def test_cancellation_billing_prefers_newer_verified_current_fence_result(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        from flash.providers.artifacts import attempts as artifact_attempts

        spec = _spec("cancel-result")
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                billing_context={"org_id": "org-a"},
                attempt=_attempt_record().to_dict(),
                progress={
                    "attempt_id": 0,
                    "fence": 1,
                    "completed_steps": 1,
                    "training_entered": True,
                },
                source_snapshot=_SOURCE_SNAPSHOT,
            )
        )
        result = {
            "attempt_id": 0,
            "fence": 1,
            "completed_steps": 7,
            "training_entered": True,
        }
        observations = iter((result,))

        def read(*_args, **_kwargs):
            current = next(observations)
            return artifact_attempts.AttemptArtifacts("revision", 100.0, None, current)

        monkeypatch.setattr(artifact_attempts, "read_attempt_artifacts", read)
        monkeypatch.setattr(runner_deploy.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(
            runner_costs, "charge_usd_for_spec", lambda _spec, **kwargs: kwargs["steps"]
        )

        runner_deploy._refresh_cancellation_result(spec.run_id, spec)
        charge, diagnostic = runner_deploy._cancellation_billing(
            spec.run_id,
            spec,
            bill_cancel=True,
            rented_remote={"provider": "runpod", "gpu_type": "B200", "gpu_count": 1},
        )

        assert charge == 7
        assert diagnostic == {}
        assert runner_status.get_status(spec.run_id).result["completed_steps"] == 7


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
        runner_state._save_status(
            runner_state.RunStatus(
                run_id="warm-cancel",
                state="running",
                spec=public_spec.to_dict(),
                billing_context={"org_id": "org-a"},
                attempt=_attempt_record().to_dict(),
                progress={
                    "attempt_id": 0,
                    "fence": 1,
                    "completed_steps": 2,
                    "training_entered": True,
                },
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
                    "init_from_adapter": "source-run",
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
                    "fence": 1,
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

        import flash.serve.deployment.deploy
        from flash.providers.core import registry as provider_registry

        monkeypatch.setattr(provider_registry, "get_provider", lambda name: Provider())
        monkeypatch.setattr(
            flash.serve.deployment.deploy,
            "undeploy_adapter",
            lambda run_id: calls.append("undeploy"),
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
                "fence": 1,
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
                "fence": 1,
                "started_ts": 1.0,
            },
        )
        runner_state._save_status(status)
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(True, metrics={"cost_usd": 0.2}),
        )
        monkeypatch.setattr(
            runner_status,
            "validate_terminal_source_metrics",
            lambda _status, metrics, **_kwargs: (metrics, None),
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
                    "fence": 1,
                    "started_ts": 1.0,
                },
            )
        )
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(True, metrics={"cost_usd": 0.2}),
        )
        monkeypatch.setattr(
            runner_status,
            "validate_terminal_source_metrics",
            lambda _status, metrics, **_kwargs: (metrics, None),
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
                    "fence": 1,
                    "started_ts": 1.0,
                },
            )
        )
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(True, metrics={"cost_usd": 0.2}),
        )
        monkeypatch.setattr(
            runner_status,
            "validate_terminal_source_metrics",
            lambda _status, metrics, **_kwargs: (metrics, None),
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
            "fence": 1,
            "started_ts": 1.0,
        }
        live_remote = {
            "provider": "runpod",
            "endpoint_id": "ep-live",
            "endpoint_name": "n",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "j-live",
            "attempt": 1,
            "fence": 1,
            "started_ts": 2.0,
        }
        status = provisioned_status(
            _spec(run_id),
            state="running",
            remote=stale_remote,
        )
        status.source_snapshot = _SOURCE_SNAPSHOT
        runner_state._save_status(status)
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="job_preempted", detail="redeploy"),
        )
        monkeypatch.setattr(runner_lifecycle, "_attempt_result_metrics", lambda *a, **k: None)

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
            "fence": 1,
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
                "fence": 1,
                "started_ts": 1.0,
            },
        )
        status.source_snapshot = _SOURCE_SNAPSHOT
        runner_state._save_status(status)
        # Poll reports a dead/abandoned job (the common redeploy-window outcome).
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="job_preempted", detail="host vanished"),
        )
        monkeypatch.setattr(runner_lifecycle, "_attempt_result_metrics", lambda *a, **k: None)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        seen = {}

        def fake_training(
            spec,
            log,
            *,
            prior_cost,
            runtime_secrets=None,
            source_snapshot=None,
            attempt_start,
        ):
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
                attempt=_attempt_record().to_dict(),
                remote={
                    "provider": "runpod",
                    "endpoint_id": "epA",
                    "endpoint_name": "n",
                    "key_fingerprint": _RUNPOD_FINGERPRINT,
                    "job_id": "jA",
                    "on_last_gpu": True,
                    "attempt": 0,
                    "fence": 1,
                    "started_ts": 1.0,
                },
            )
        )
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="job_preempted", detail="host vanished"),
        )
        monkeypatch.setattr(runner_lifecycle, "_attempt_result_metrics", lambda *a, **k: None)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        training_calls = []
        monkeypatch.setattr(
            runner_lifecycle, "_run_training", lambda *a, **k: training_calls.append((a, k))
        )

        status = runner_attach.attach_run(spec.run_id, log_stream=sys.stderr)

        assert status.state == "failed"
        assert status.error == "job_preempted: host vanished"
        assert training_calls == []
        assert status.remote["endpoint_id"] == "epA"


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
            attempt=_attempt_record().to_dict(),
            remote={
                "provider": "runpod",
                "endpoint_id": "epA",
                "endpoint_name": "n",
                "key_fingerprint": _RUNPOD_FINGERPRINT,
                "job_id": "jA",
                "on_last_gpu": False,
                "attempt": 0,
                "fence": 1,
                "started_ts": 1.0,
            },
        )
        status.source_snapshot = _SOURCE_SNAPSHOT
        runner_state._save_status(status)
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(False, failure="job_preempted", detail="host vanished"),
        )
        monkeypatch.setattr(runner_lifecycle, "_attempt_result_metrics", lambda *a, **k: None)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        seen = {}

        def fake_training(
            spec,
            log,
            *,
            prior_cost,
            runtime_secrets=None,
            source_snapshot=None,
            attempt_start,
        ):
            seen["prior_cost"] = prior_cost
            seen["source_snapshot"] = source_snapshot
            seen["attempt_start"] = attempt_start
            runner_status._update(spec.run_id, "done", cost_usd=prior_cost)

        monkeypatch.setattr(runner_lifecycle, "_run_training", fake_training)

        st = runner_attach.attach_run("pinned-code", log_stream=sys.stderr)

        assert st.state == "done"
        assert seen == {
            "prior_cost": 0.25,
            "source_snapshot": _SOURCE_SNAPSHOT,
            "attempt_start": 1,
        }


def test_attach_resume_that_fails_again_marks_run_failed(monkeypatch):
    # The resume delegates the genuine-vs-infra decision to the training submit (unchanged): a run
    # that is truly broken reproduces the failure on the resumed attempt, _run_training fails it, and
    # attach surfaces that terminal `failed` — so a broken run still terminates (nothing hangs).
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        _confirm_runpod_retry_teardown(monkeypatch)
        import flash.providers.runpod.execution.jobs as jobs
        import flash.providers.runpod.serverless.endpoints as flash_train

        status = provisioned_status(
            _spec("g1"),
            state="running",
            attempt=_attempt_record().to_dict(),
            remote={
                "provider": "runpod",
                "endpoint_id": "epA",
                "endpoint_name": "n",
                "key_fingerprint": _RUNPOD_FINGERPRINT,
                "job_id": "jA",
                "on_last_gpu": False,
                "attempt": 0,
                "fence": 1,
                "started_ts": 1.0,
            },
        )
        status.source_snapshot = _SOURCE_SNAPSHOT
        runner_state._save_status(status)
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(
                False, failure="job_failed", detail="Traceback ...\nRuntimeError: bad reward fn"
            ),
        )
        monkeypatch.setattr(runner_lifecycle, "_attempt_result_metrics", lambda *a, **k: None)
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        monkeypatch.setattr(runner_recovery, "_gc_run_endpoints", lambda _spec: None)
        resumed = {"called": False}
        replacement_remote = {
            "provider": "runpod",
            "endpoint_id": "epB",
            "endpoint_name": "replacement",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "jB",
            "on_last_gpu": False,
            "attempt": 1,
            "fence": 1,
            "started_ts": 2.0,
        }

        def fake_training(
            spec,
            log,
            *,
            prior_cost,
            runtime_secrets=None,
            source_snapshot=None,
            attempt_start,
        ):
            assert source_snapshot == _SOURCE_SNAPSHOT
            # the training submit re-runs the run; a genuinely broken run fails there (matches
            # _submit_seed_supervised raising after a non-infra failure with no retries left).
            resumed["called"] = True
            resumed["attempt_start"] = attempt_start
            runner_status._update(spec.run_id, "running", remote=replacement_remote)
            raise RuntimeError("run failed after retries: worker_error: bad reward fn")

        monkeypatch.setattr(runner_lifecycle, "_run_training", fake_training)

        st = runner_attach.attach_run("g1", log_stream=sys.stderr)

        assert resumed["called"] is True, (
            "attach must attempt a checkpoint resume on any non-ok poll"
        )
        assert st.state == "failed", "a resume that fails again must terminate the run"
        assert st.remote == replacement_remote
        assert "bad reward fn" in (st.error or "")
        assert runner_status._load_status_json("g1")[runner_state._CLEANUP_REMOTES_KEY] == [
            {key: value for key, value in replacement_remote.items() if key != "on_last_gpu"},
            {
                "provider": "runpod",
                "endpoint_id": "epA",
                "endpoint_name": "n",
                "key_fingerprint": _RUNPOD_FINGERPRINT,
                "job_id": "jA",
                "attempt": 0,
                "fence": 1,
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
            "fence": 1,
            "started_ts": 1.0,
        }
        runner_state._save_status(
            provisioned_status(
                _spec("runpod-unconfirmed"),
                state="running",
                remote=remote,
            )
        )
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(
                False,
                failure="job_preempted",
                detail="provider resource was lost",
            ),
        )
        monkeypatch.setattr(runner_lifecycle, "_attempt_result_metrics", lambda *a, **k: None)
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
            "fence": 1,
            "started_ts": 1.0,
        }
        newer_remote = {
            "provider": "runpod",
            "endpoint_id": "ep-new",
            "endpoint_name": "new",
            "key_fingerprint": _RUNPOD_FINGERPRINT,
            "job_id": "job-new",
            "attempt": 1,
            "fence": 1,
            "started_ts": 2.0,
        }
        runner_state._save_status(
            provisioned_status(
                _spec("attach-newer-remote"),
                state="running",
                remote=old_remote,
            )
        )
        monkeypatch.setattr(
            polling,
            "poll_job",
            lambda *a, **k: jobs.PollResult(
                False,
                failure="job_preempted",
                detail="provider resource was lost",
            ),
        )
        monkeypatch.setattr(runner_lifecycle, "_attempt_result_metrics", lambda *a, **k: None)
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
                    "fence": 1,
                    "started_ts": 1.0,
                },
            )
        )
        monkeypatch.setattr(flash_train, "terminate_endpoint", lambda *a, **k: [])
        monkeypatch.setattr(runner_lifecycle, "_attempt_result_metrics", lambda *a, **k: None)

        class _RaisingVast:
            def poll(self, handle, spec, seed, *, log=None, _deadline_at=None):
                assert _deadline_at == pytest.approx(
                    runner_status.get_status("v1").created_at + _spec("v1").gpu.max_wall_seconds
                )
                return PollResult(False, failure="job_preempted", detail="host vanished")

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
        "fence": 1,
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
            "job_preempted: host vanished",
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
        "job_preempted: host vanished",
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
                attempt=_attempt_record().to_dict(),
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
            "job_preempted: host vanished",
        )
        assert cleaned == [run_id]


def test_attach_reconciler_resumes_after_vast_strict_absence(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.runner.supervise.attach as attach_mod
        from flash.providers.core import registry as providers

        remote = _vast_recovery_remote()
        spec = _spec("vast-reconcile-clear")
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                attempt=_attempt_record(
                    grant_deadline_at=time.time() + 60.0,
                    work_deadline_at=time.time() + 600.0,
                    result_deadline_at=time.time() + 660.0,
                ).to_dict(),
                remote=remote,
                source_snapshot=_SOURCE_SNAPSHOT,
            )
        )

        class Provider:
            def destroy(self, _handle):
                raise RuntimeError("delete acknowledgement unavailable")

            def run_instances_remaining(self, run_id):
                return []

        monkeypatch.setattr(providers, "get_provider", lambda _name: Provider())
        monkeypatch.setattr(attach_mod, "_ATTACH_RECONCILE_INTERVAL_S", 0.0)
        monkeypatch.setattr(runner_lifecycle, "_attempt_result_metrics", lambda *_a, **_k: None)
        resumed = []
        monkeypatch.setattr(
            runner_lifecycle,
            "_run_training",
            lambda *args, **kwargs: resumed.append(kwargs["attempt_start"]),
        )

        attach_mod._reconcile_attached_remote(
            spec.run_id,
            remote,
            spec,
            1,
            _SOURCE_SNAPSHOT,
            io.StringIO(),
            "job_preempted: host vanished",
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
                attempt=_attempt_record().to_dict(),
                remote=remote,
            )
        )
        deadline = runner_deadlines._load_run_deadline_at(spec.run_id)
        monkeypatch.setattr(attach_mod.time, "time", lambda: deadline + 1.0)
        monkeypatch.setattr(attach_mod, "_ATTACH_RECONCILE_INTERVAL_S", 0.0)
        monkeypatch.setattr(lifecycle_mod, "_attempt_result_metrics", lambda *a, **k: None)
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
            "job_preempted: host vanished",
        )

        status = runner_status.get_status(spec.run_id)
        assert len(calls) == 2
        assert status.state == "failed"
        assert status.remote == remote
        assert runner_status._load_status_json(spec.run_id)[runner_state._CLEANUP_REMOTES_KEY] == [
            remote
        ]


def test_attach_reconciler_fails_permanent_result_artifact_without_retry(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        _fresh_orchestrator(tmp, monkeypatch)
        import flash.runner.supervise.attach as attach_mod
        import flash.runner.supervise.lifecycle as lifecycle_mod
        from flash.providers.artifacts.attempts import AttemptArtifactError

        remote = _vast_recovery_remote()
        spec = _spec("vast-reconcile-invalid-result")
        attempt = _attempt_record(
            grant_deadline_at=900.0,
            work_deadline_at=1_000.0,
            result_deadline_at=1_020.0,
        )
        runner_state._save_status(
            runner_state.RunStatus(
                run_id=spec.run_id,
                state="running",
                spec=spec.to_dict(),
                attempt=attempt.to_dict(),
                remote=remote,
            )
        )
        monkeypatch.setattr(attach_mod.time, "time", lambda: 1_021.0)
        monkeypatch.setattr(
            attach_mod.time,
            "sleep",
            lambda _seconds: pytest.fail("permanent artifact errors must not retry"),
        )
        monkeypatch.setattr(
            lifecycle_mod,
            "_attempt_result_metrics",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AttemptArtifactError("result manifest is invalid or unverifiable")
            ),
        )

        attach_mod._reconcile_attached_remote(
            spec.run_id,
            remote,
            spec,
            1,
            _SOURCE_SNAPSHOT,
            io.StringIO(),
            "job_preempted: host vanished",
        )

        status = runner_status.get_status(spec.run_id)
        assert status.state == "failed"
        assert status.remote == remote
        assert "invalid or unverifiable" in (status.error or "")
        assert runner_status._load_status_json(spec.run_id)[runner_state._CLEANUP_REMOTES_KEY] == [
            remote
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
                attempt=_attempt_record().to_dict(),
                remote=remote,
            )
        )
        deadline = runner_deadlines._load_run_deadline_at(spec.run_id)
        monkeypatch.setattr(attach_mod.time, "time", lambda: deadline + 1.0)
        monkeypatch.setattr(attach_mod, "_ATTACH_RECONCILE_INTERVAL_S", 0.0)
        monkeypatch.setattr(
            lifecycle_mod,
            "_attempt_result_metrics",
            lambda *a, **k: {"wall_seconds": 5.0},
        )

        def adopt(run_id, _spec, expected_remote, _metrics, **_kwargs):
            assert runner_reconciliation._record_cleanup_remote(run_id, expected_remote)
            return runner_status._update(run_id, "done")

        monkeypatch.setattr(lifecycle_mod, "_adopt_completed_attempt", adopt)

        attach_mod._reconcile_attached_remote(
            spec.run_id,
            remote,
            spec,
            1,
            _SOURCE_SNAPSHOT,
            io.StringIO(),
            "job_preempted: host vanished",
        )

        status = runner_status.get_status(spec.run_id)
        assert status.state == "done", status.error
        assert status.remote == remote
        assert status.error is None
        assert runner_status._load_status_json(spec.run_id)[runner_state._CLEANUP_REMOTES_KEY] == [
            remote
        ]


def test_attach_reconciler_caps_completed_adoption_retry_to_result_deadline(monkeypatch):
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
                attempt=_attempt_record().to_dict(),
                remote=remote,
            )
        )
        deadline = 1_000.0
        result_deadline = deadline + 30.0
        status = runner_status.get_status(spec.run_id)
        status.attempt = _attempt_record(
            grant_deadline_at=900.0,
            work_deadline_at=deadline,
            result_deadline_at=result_deadline,
        ).to_dict()
        runner_state._save_status(status)
        clock = {"now": result_deadline - 1.0}
        sleeps = []
        monkeypatch.setattr(runner_deadlines, "_load_run_deadline_at", lambda _run_id: deadline)
        monkeypatch.setattr(attach_mod.time, "time", lambda: clock["now"])

        def advance(seconds):
            sleeps.append(seconds)
            clock["now"] += seconds

        monkeypatch.setattr(attach_mod.time, "sleep", advance)
        monkeypatch.setattr(
            lifecycle_mod,
            "_attempt_result_metrics",
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
            "job_preempted: host vanished",
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
                attempt=_attempt_record().to_dict(),
                remote=remote,
            )
        )
        deadline = 1_000.0
        status = runner_status.get_status(spec.run_id)
        status.attempt = _attempt_record(
            grant_deadline_at=900.0,
            work_deadline_at=deadline,
            result_deadline_at=deadline + 60.0,
        ).to_dict()
        runner_state._save_status(status)
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

        monkeypatch.setattr(lifecycle_mod, "_attempt_result_metrics", completed_metrics)
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
            "job_preempted: host vanished",
        )

        assert probes == [deadline - 1.0, deadline]


def test_attach_reconciler_rate_limits_failed_terminal_cas_past_result_deadline(monkeypatch):
    # past the result deadline the terminal compare-and-fail CAS is the only exit
    # from the completed-but-unadoptable branch. if that CAS transiently raises, the
    # reconciler must rate-limit each retry at the full reconcile interval instead of
    # sleeping 0 once the result window has closed and busy-spinning the loop.
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
                attempt=_attempt_record().to_dict(),
                remote=remote,
            )
        )
        deadline = 1_000.0
        status = runner_status.get_status(spec.run_id)
        status.attempt = _attempt_record(
            grant_deadline_at=900.0,
            work_deadline_at=deadline,
            result_deadline_at=deadline,
        ).to_dict()
        runner_state._save_status(status)
        # start at the result deadline so the terminal cas is the only exit.
        clock = {"now": deadline}
        sleeps = []
        monkeypatch.setattr(runner_deadlines, "_load_run_deadline_at", lambda _run_id: deadline)
        monkeypatch.setattr(attach_mod.time, "time", lambda: clock["now"])

        def advance(seconds):
            sleeps.append(seconds)
            clock["now"] += seconds

        monkeypatch.setattr(attach_mod.time, "sleep", advance)
        monkeypatch.setattr(
            lifecycle_mod,
            "_attempt_result_metrics",
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
            "job_preempted: host vanished",
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
    # the fenced result carries worker metrics, but the worker never knew how many cards the allocator
    # gave it -- the plane stamped that onto the persisted remote
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
                attempt=_attempt_record().to_dict(),
                remote=remote,
            )
        )
        deadline = 1_000.0
        monkeypatch.setattr(runner_deadlines, "_load_run_deadline_at", lambda _run_id: deadline)
        monkeypatch.setattr(attach_mod.time, "time", lambda: deadline)
        monkeypatch.setattr(attach_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(lifecycle_mod, "_attempt_result_metrics", lambda *_a, **_k: None)
        # what the worker actually wrote: a wall, and nothing about the allocation.
        monkeypatch.setattr(
            lifecycle_mod,
            "_attempt_result_metrics",
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
            "job_preempted: host vanished",
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
            "job_preempted: host vanished",
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
    """Drive poll_job against a job that never leaves IN_QUEUE."""
    import itertools

    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import jobs

    polling = _wire_runpod_poll(
        monkeypatch,
        attempt=_attempt_record(
            grant_deadline_at=900.0,
            work_deadline_at=1_000.0,
            result_deadline_at=1_100.0,
        ),
    )
    monkeypatch.setattr(runpod_api, "job_status", lambda eid, jid, **_kw: {"status": "IN_QUEUE"})
    monkeypatch.setattr(
        runpod_api,
        "endpoint_health_for_fingerprint",
        lambda eid, _fingerprint, **_kw: (_ for _ in ()).throw(RuntimeError("no workers yet")),
    )
    clock = itertools.count(start=0, step=25.0)
    monkeypatch.setattr(polling.time, "time", lambda: next(clock))
    return polling.poll_job(
        _runpod_handle(jobs),
        _poll_spec(),
        interval_s=0,
        queue_grace_s=900.0,
        **poll_kwargs,
    )


def test_capacity_detail_does_not_promise_a_next_best_gpu_on_the_last_class(monkeypatch):
    """LS-008/AT-013: do not promise a next-best GPU on the last class.

    ``on_last_gpu`` does not identify a reused class; only the supervisor owns that candidate list.
    """
    res = _poll_in_queue_forever(monkeypatch, on_last_gpu=True)
    assert res.failure == "no_capacity"
    assert "next-best" not in res.detail, res.detail
    assert "no further GPU-class escalation follows" in res.detail, res.detail
    assert "same class" not in res.detail, res.detail


def test_capacity_detail_claims_neither_a_retry_nor_class_exhaustion(monkeypatch):
    """Claim neither retry nor class exhaustion from ``on_last_gpu``.

    It can mean no untried class remains or the retry budget is spent; only the supervisor knows.
    """
    res = _poll_in_queue_forever(monkeypatch, on_last_gpu=True)
    assert res.failure == "no_capacity"
    # not a retry promise: this detail is also emitted on the attempt that ends the run.
    assert "retrying" not in res.detail, res.detail
    # not a class-exhaustion claim: untried classes can remain when the budget is what ran out.
    assert "untried" not in res.detail, res.detail


def test_reattach_keeps_the_scarcity_grace_but_not_the_capacity_wording(monkeypatch):
    """Use the persisted last-GPU flag only for the scarcity grace.

    Recovery rebuilds an unpinned candidate walk, so the stale flag must not constrain capacity
    wording even though it still selects the longer wait.
    """
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers.core.base import JobHandle
    from flash.providers.runpod.execution import jobs as jobs
    from flash.providers.runpod.execution.provider import PROVIDER

    captured: dict = {}

    def fake_poll_job(handle, spec, **kw):
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
        "fence": 1,
    }

    PROVIDER.poll(JobHandle.from_dict({**base, "on_last_gpu": True}), spec, spec.seed)
    # the capacity wording is left at poll_job's neutral default: escalation may follow, because
    # after recovery it genuinely can.
    assert "on_last_gpu" not in captured, captured
    # the scarcity grace still honours the snapshot: 900s, not the 300s of a normal attempt.
    assert captured["queue_grace_s"] == 900.0, captured
    assert captured["throttled_grace_s"] == 900.0, captured

    captured.clear()
    PROVIDER.poll(JobHandle.from_dict({**base, "on_last_gpu": False}), spec, spec.seed)
    assert "on_last_gpu" not in captured, captured
    assert captured["queue_grace_s"] == 300.0, captured


def test_capacity_detail_promises_no_retry_when_a_next_class_exists(monkeypatch):
    """Keep the default capacity detail neutral about retries and class walks.

    A cache-drop retry may reselect the same class; see
    ``test_cache_drop_retry_names_the_same_class_it_reselects``.
    """
    res = _poll_in_queue_forever(monkeypatch)
    assert res.failure == "no_capacity"
    assert "GPU-class escalation may follow" in res.detail, res.detail
    assert "retrying" not in res.detail, res.detail
    assert "next-best" not in res.detail, res.detail


# preserved provider submission and cleanup coverage


def _persist_runpod_attempt(spec, *, attempt_id=0, fence=1, deadline_at=10_000_000_000.0):
    attempt = _attempt_record(
        attempt_id=attempt_id,
        fence=fence,
        grant_deadline_at=deadline_at - 120.0,
        work_deadline_at=deadline_at - 60.0,
        result_deadline_at=deadline_at,
    )
    runner_state._save_status(
        provisioned_status(
            spec,
            state="running",
            attempt=attempt.to_dict(),
            source_snapshot=_SOURCE_SNAPSHOT,
        )
    )


def test_submit_run_payload_carries_structured_source_snapshot(monkeypatch):
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
    _persist_runpod_attempt(spec)
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
    assert job_execution.submit_run(
        spec,
        seed=spec.seed,
        source_snapshot=source_snapshot,
        deadline_at=10_000_000_000.0,
    ).ok
    assert submitted["endpoint_id"] == "ep"
    assert submitted["payload"]["source_snapshot"] == source_snapshot
    assert "code_prefix" not in submitted["payload"]


def test_submit_run_endpoint_timeout_covers_result_visibility_window(monkeypatch):
    from flash.core.spec import GpuSpec, JobSpec, TrainSpec
    from flash.providers._lifecycle.net import deadline as deadline_ops
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.execution import job_execution, jobs

    spec = JobSpec(
        run_id="flash-result-window-timeout",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(epochs=1, hf_repo="org/repo"),
        gpu=GpuSpec(type=""),
    )
    attempt = _attempt_record(
        grant_deadline_at=150.0,
        work_deadline_at=200.0,
        result_deadline_at=320.0,
    )
    runner_state._save_status(
        provisioned_status(
            spec,
            state="running",
            attempt=attempt.to_dict(),
            source_snapshot=_SOURCE_SNAPSHOT,
        )
    )
    deployed = {}
    submitted = {}
    monkeypatch.setattr(deadline_ops.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        job_execution,
        "deploy_train_endpoint",
        lambda *args, **kwargs: (
            deployed.update({"args": args, "kwargs": kwargs})
            or ("ep", "endpoint-name", _RUNPOD_FINGERPRINT)
        ),
    )
    monkeypatch.setattr(
        runpod_api,
        "submit_job",
        lambda _endpoint_id, payload, **_kwargs: submitted.update(payload) or "job-1",
    )
    monkeypatch.setattr(polling, "poll_job", lambda *_args, **_kwargs: jobs.PollResult(True))

    result = job_execution.submit_run(
        spec,
        seed=spec.seed,
        source_snapshot=_SOURCE_SNAPSHOT,
        deadline_at=200.0,
    )

    assert result.ok
    assert deployed["kwargs"]["execution_timeout_ms"] == 220_000
    assert deployed["kwargs"]["deadline_at"] == 200.0
    assert submitted["work_deadline_at"] == 200.0
    assert submitted["result_deadline_at"] == 320.0


def test_submit_run_rejects_malformed_source_before_deploy(monkeypatch):
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
        job_execution.submit_run(
            spec,
            seed=spec.seed,
            source_snapshot=malformed,
            deadline_at=10_000_000_000.0,
        )
    assert deploy_calls == []


def test_submit_run_polls_a_multi_card_shape_on_the_scaled_capacity_grace(monkeypatch):
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
        _persist_runpod_attempt(spec)
        assert job_execution.submit_run(
            spec,
            seed=spec.seed,
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
    _persist_runpod_attempt(spec, attempt_id=4)
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
        job_execution.submit_run(
            spec,
            spec.seed,
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
    _persist_runpod_attempt(spec, attempt_id=4)
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
        job_execution.submit_run(
            spec,
            spec.seed,
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
    _persist_runpod_attempt(spec, deadline_at=deadline_at)
    captured = []

    monkeypatch.setattr(job_execution.time, "time", lambda: 100.0)
    monkeypatch.setattr(
        job_execution,
        "deploy_train_endpoint",
        lambda *a, **k: ("ep", "endpoint-name", _RUNPOD_FINGERPRINT),
    )
    monkeypatch.setattr(runpod_api, "submit_job", lambda *_a, **_k: "job-1")

    def fake_poll(handle, spec, **kwargs):
        captured.append(kwargs["deadline_at"])
        return PollResult(True, metrics={})

    monkeypatch.setattr(polling, "poll_job", fake_poll)
    provider = RunpodProvider()
    assert provider.submit_run(
        spec,
        seed=spec.seed,
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
    assert provider.poll(handle, spec, seed=spec.seed, _deadline_at=deadline_at).ok

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
    _persist_runpod_attempt(spec, deadline_at=200.0)
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
        job_execution.submit_run(
            spec, seed=spec.seed, source_snapshot=_SOURCE_SNAPSHOT, deadline_at=200.0
        )

    assert submitted == []
    assert deleted == ["ep"]
