"""Modal launch supervision preserves worker payloads and exact cleanup ownership."""

import time

import pytest

from flash.core.spec import GpuSpec, JobSpec, TrainSpec
from flash.providers.core.base import JobHandle, PollResult
from flash.providers.modal import api, jobs
from flash.providers.modal.handles import ModalJobHandle
from flash.providers.modal.pricing import hourly_rate
from tests._helpers.source_snapshot import valid_source_snapshot


@pytest.fixture
def launch_state(monkeypatch):
    events = []
    monkeypatch.setattr("flash.runtime_capsule.build_capsule", lambda _: (b"capsule", {}))

    def create(**kwargs):
        events.append(("create", kwargs))
        return "sb-job"

    monkeypatch.setattr(
        "flash.providers.modal.checkpoint_upload.cleanup",
        lambda handle: events.append(("cleanup", handle.instance_id)),
    )
    monkeypatch.setattr(api, "create", create)
    monkeypatch.setattr(api, "terminate", lambda identity: events.append(("terminate", identity)))
    return events


def spec(**gpu_changes):
    return JobSpec(
        run_id="modal-lifecycle",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        seed=42,
        train=TrainSpec(hf_repo="org/artifacts"),
        gpu=GpuSpec(type="L4", count=2, network_volume="flash-weights", **gpu_changes),
    )


def submit(job, **kwargs):
    return jobs.submit_attempt(
        job,
        source_snapshot=valid_source_snapshot(),
        deadline_at=time.time() + 600,
        **kwargs,
    )


@pytest.mark.parametrize("old_switch", [None, "0", "1"])
def test_launch_persists_identity_before_poll_and_stops_after_success(
    monkeypatch, launch_state, old_switch
):
    if old_switch is None:
        monkeypatch.delenv("FLASH_MODAL_CPU_UPLOADS", raising=False)
    else:
        monkeypatch.setenv("FLASH_MODAL_CPU_UPLOADS", old_switch)
    persisted = []

    def poll(handle, job, **kwargs):
        assert persisted == [handle.to_dict()]
        assert handle.to_dict()["run_id"] == job.run_id
        assert handle.to_dict()["hourly_usd"] == hourly_rate("L4") * 2
        return PollResult(True, metrics={"train_tokens": 1})

    monkeypatch.setattr(jobs, "poll", poll)
    result = submit(spec(), on_handle=lambda handle: persisted.append(handle.to_dict()), attempt=3)
    assert result.ok
    payload = launch_state[0][1]["payload"]
    assert persisted[0]["cpu_uploads"] is True
    assert payload["handoff_volume"].startswith("flash-handoff-")
    assert payload["attempt"] == 3
    assert payload["flash_arm"] == "modal"
    assert payload["source_snapshot"] == valid_source_snapshot()
    assert payload["env"]["FLASH_WEIGHT_CACHE_DIR"] == "/weight-cache/hf-cache/hub"
    assert launch_state[0][1]["gpu_count"] == 2
    assert launch_state[-2:] == [("terminate", "sb-job"), ("cleanup", "sb-job")]


@pytest.mark.parametrize("failure_stage", ["persist", "poll"])
def test_failure_after_launch_always_stops_exact_sandbox(monkeypatch, launch_state, failure_stage):
    def fail(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(jobs, "poll", fail)
    callback = fail if failure_stage == "persist" else None
    with pytest.raises(RuntimeError, match="write failed"):
        submit(spec(), on_handle=callback)
    assert launch_state[-2:] == [("terminate", "sb-job"), ("cleanup", "sb-job")]


def test_operator_cache_bypass_survives_assigned_volume(monkeypatch, launch_state):
    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DISABLED", "1")
    monkeypatch.setattr(jobs, "poll", lambda *a, **k: PollResult(True, metrics={}))
    submit(spec())
    assert "FLASH_WEIGHT_CACHE_DIR" not in launch_state[0][1]["payload"]["env"]


def test_recovery_rejects_a_different_run_before_any_artifact_read(monkeypatch):
    from tests._helpers.modal import modal_remote

    handle = JobHandle.from_dict(modal_remote(run_id="another-run"))
    monkeypatch.setattr(jobs, "make_hf_text_reader", lambda *a, **k: pytest.fail("foreign run"))
    with pytest.raises(ValueError, match="does not belong"):
        jobs.poll(handle, spec(), deadline_at=time.time() + 60)


def test_successful_artifacts_are_recovered_without_provider_credentials(monkeypatch):
    import json

    from flash.runner.supervise.recovery import _completed_worker_metrics
    from tests._helpers.modal import modal_remote

    raw = modal_remote(run_id="recovered", attempt=2, started_ts=100)
    handle = ModalJobHandle.from_dict(raw)
    marker = {
        "run_id": handle.run_id,
        "attempt": 2,
        "ts": 110,
        "error": "",
        "retriable": False,
        "ok": True,
    }
    reads = []

    def reader(repo, path, **kwargs):
        reads.append((repo, path))
        value = marker if path.endswith("modal_attempt2.json") else {"train_tokens": 10}
        return lambda **kw: json.dumps(value)

    monkeypatch.setattr("flash.providers.artifacts.hf.make_hf_text_reader", reader)
    monkeypatch.delenv("MODAL_TOKEN_ID")
    monkeypatch.delenv("MODAL_TOKEN_SECRET")
    result = _completed_worker_metrics(raw, deadline_at=time.time())
    assert result == {"train_tokens": 10}
    assert reads == [
        ("org/artifacts", "sft/recovered/modal_attempt2.json"),
        ("org/artifacts", "sft/recovered/metrics.json"),
    ]


@pytest.mark.parametrize("changed_key", ["MODAL_TOKEN_ID", "MODAL_ENVIRONMENT"])
def test_handleless_recovery_cannot_prove_absence_in_a_different_scope(monkeypatch, changed_key):
    from types import SimpleNamespace

    import flash.server.platform.runtime as runtime
    from flash.providers.modal.auth import scope_fingerprint

    status = SimpleNamespace(
        submitted_modal_scope=scope_fingerprint(), submitted_instance_providers=["modal"]
    )
    monkeypatch.setattr(runtime, "get_status", lambda _: status)
    monkeypatch.setenv(changed_key, "another-owner")
    monkeypatch.setattr(
        "flash.providers.core.registry.configured_providers",
        lambda: pytest.fail("must reject before cleanup"),
    )
    assert runtime._confirm_run_clear(spec()) is False


def test_launch_keeps_original_owner_if_environment_changes_after_create(monkeypatch, launch_state):
    from flash.providers.modal.auth import scope_fingerprint

    original_scope = scope_fingerprint()
    create = api.create

    def change_environment(**kwargs):
        identity = create(**kwargs)
        monkeypatch.setenv("MODAL_ENVIRONMENT", "changed")
        return identity

    handles = []
    monkeypatch.setattr(api, "create", change_environment)
    monkeypatch.setattr(jobs, "poll", lambda *a, **kw: PollResult(False, failure="poll_error"))
    submit(spec(), on_handle=lambda h: handles.append(h.to_dict()))
    assert handles[0]["scope"] == original_scope
    assert launch_state[-2:] == [("terminate", "sb-job"), ("cleanup", "sb-job")]


def test_disabled_cache_cannot_authorize_cacheless_retry(monkeypatch):
    from types import SimpleNamespace

    from flash.runner.supervise.retry_decision import _managed_cache_mounted

    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DISABLED", "1")
    assert not _managed_cache_mounted(spec(), SimpleNamespace(provider="modal"))
