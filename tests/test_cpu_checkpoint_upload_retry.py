"""Final CPU checkpoint upload: retried on a fresh uploader, never reported as a preemption."""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from flash.core.spec import JobSpec, TrainSpec
from flash.providers.core.base import PollResult
from flash.providers.modal import api, checkpoint_upload, jobs
from flash.providers.modal.auth import scope_fingerprint
from flash.providers.modal.handles import ModalJobHandle
from flash.providers.modal.pricing import hourly_rate


def _handle(**changes):
    values = {
        "instance_id": "sb-test123",
        "gpu": "L4",
        "hourly_usd": hourly_rate("L4"),
        "attempt": 0,
        "started_ts": 100.0,
        "name": "flash-test-s0-a0",
        "run_id": "flash-test",
        "hf_repo": "org/artifacts",
        "phase": "sft",
        "scope": scope_fingerprint(),
        "gpu_count": 1,
        "cpu_uploads": True,
    }
    values.update(changes)
    return ModalJobHandle(**values)


def _spec():
    return JobSpec(
        run_id="flash-test",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        train=TrainSpec(hf_repo="org/artifacts"),
    )


class _Aio:
    def __init__(self, fn):
        self.aio = fn


class _NotFoundError(Exception):
    pass


class _FakeSandbox:
    def __init__(self, tags, exit_code):
        self.object_id = "sb-uploader"
        self.tags = tags
        self.exit_code = exit_code
        self.terminated = False

        async def wait():
            return None

        async def poll():
            return self.exit_code

        async def get_tags():
            return self.tags

        async def terminate(wait=False):
            self.terminated = True
            if self.exit_code is None:
                self.exit_code = -1

        self.wait = _Aio(wait)
        self.poll = _Aio(poll)
        self.get_tags = _Aio(get_tags)
        self.terminate = _Aio(terminate)


class _FakeModal:
    """Just enough of the Modal SDK for `finish` and `cleanup`; exit codes are scripted."""

    def __init__(self, exit_codes, existing=None):
        self.exit_codes = list(exit_codes)
        self.sandboxes = dict(existing or {})
        self.created = []
        self.environments = []
        self.parent_tags = {
            "flash_owner": "flash-training",
            "flash_run": "flash-test",
            "flash_attempt": "0",
            "flash_name": "flash-test-s0-a0",
            "flash_spec": hashlib.sha256(_spec().to_json().encode()).hexdigest(),
        }
        self.exception = SimpleNamespace(NotFoundError=_NotFoundError)

        async def from_name(_app, name, **_kwargs):
            if name not in self.sandboxes:
                raise _NotFoundError(name)
            return self.sandboxes[name]

        async def from_id(_identity, **_kwargs):
            return _FakeSandbox(self.parent_tags, 0)

        async def create(*_args, name, tags, **_kwargs):
            self.environments.append(_kwargs["secrets"][0])
            sandbox = _FakeSandbox(tags, self.exit_codes.pop(0))
            self.sandboxes[name] = sandbox
            self.created.append((name, tags))
            return sandbox

        async def lookup(*_args, **_kwargs):
            return object()

        async def delete_volume(*_args, **_kwargs):
            return None

        self.Sandbox = SimpleNamespace(
            from_name=_Aio(from_name), from_id=_Aio(from_id), create=_Aio(create)
        )
        self.App = SimpleNamespace(lookup=_Aio(lookup))
        self.Volume = SimpleNamespace(
            from_name=lambda *a, **k: object(),
            objects=SimpleNamespace(delete=_Aio(delete_volume)),
        )
        self.Image = SimpleNamespace(
            from_registry=lambda *_a, **_k: SimpleNamespace(pip_install=lambda *_a, **_k: object())
        )
        self.Secret = SimpleNamespace(from_dict=lambda env: env)


@pytest.fixture
def fake_modal(monkeypatch):
    sleeps = []

    def install(exit_codes, existing=None):
        modal = _FakeModal(exit_codes, existing)

        def call(operation, *, deadline_at=None, timeout=None):
            return asyncio.run(operation(modal, None))

        monkeypatch.setattr(api, "call", call)
        monkeypatch.setattr(api, "terminate", lambda _instance_id: None)
        monkeypatch.setattr(api, "environment", lambda: "test")
        modal.parent_tags["flash_handoff"] = checkpoint_upload.volume_name(
            _handle().scope, "flash-test", 0
        )
        monkeypatch.setattr(
            "flash.artifacts.config.runtime_environment", lambda: {"AZURE_STORAGE": "x"}
        )
        monkeypatch.setattr(
            "flash.providers._lifecycle.net.worker.worker_image_for_gpu", lambda _gpu: "image"
        )
        monkeypatch.setattr(checkpoint_upload.time, "sleep", sleeps.append)
        modal.sleeps = sleeps
        return modal

    return install


def test_failed_upload_is_retried_on_a_fresh_uploader(fake_modal):
    modal = fake_modal([1, 0])
    handle = _handle()

    notes = checkpoint_upload.finish(handle, deadline_at=time.time() + 3600)

    assert notes["cpu_upload_tries"] == 2
    assert [name for name, _ in modal.created] == ["upload-sb-test123", "upload-sb-test123-r1"]
    first_tags, retry_tags = (tags for _, tags in modal.created)
    assert "flash_upload_try" not in first_tags  # try 0 keeps the original identity
    assert retry_tags["flash_upload_try"] == "1"
    assert modal.sleeps == [15.0]


def test_every_try_failing_raises_a_distinct_upload_error(fake_modal):
    modal = fake_modal([1, 1, 1])

    with pytest.raises(checkpoint_upload.CheckpointUploadError, match="after 3 tries"):
        checkpoint_upload.finish(_handle(), deadline_at=time.time() + 3600)

    assert len(modal.created) == checkpoint_upload.UPLOAD_TRIES
    assert modal.sleeps == [15.0, 60.0]


def test_no_retry_once_the_run_deadline_is_spent(fake_modal, monkeypatch):
    modal = fake_modal([1, 0])
    deadline = time.time() + 3600

    real_call = api.call

    def call_then_expire(operation, **kwargs):
        try:
            return real_call(operation, **kwargs)
        finally:
            monkeypatch.setattr(checkpoint_upload.time, "time", lambda: deadline + 1)

    monkeypatch.setattr(api, "call", call_then_expire)

    with pytest.raises(checkpoint_upload.CheckpointUploadError, match="after 1 try"):
        checkpoint_upload.finish(_handle(), deadline_at=deadline)
    assert len(modal.created) == 1


def test_finished_uploader_from_an_earlier_poll_is_reconciled_not_relaunched(fake_modal):
    handle = _handle()
    done = _FakeSandbox(checkpoint_upload._uploader_tags(handle), 0)
    modal = fake_modal([], existing={"upload-sb-test123": done})

    notes = checkpoint_upload.finish(handle, deadline_at=time.time() + 3600)

    assert notes["cpu_upload_tries"] == 1
    assert modal.created == []


def test_cleanup_terminates_every_try(fake_modal):
    handle = _handle()
    tries = {
        checkpoint_upload._uploader_name(handle, i): _FakeSandbox(
            checkpoint_upload._uploader_tags(handle, i), None
        )
        for i in range(checkpoint_upload.UPLOAD_TRIES)
    }
    fake_modal([], existing=tries)

    checkpoint_upload.cleanup(handle)

    assert all(sandbox.terminated for sandbox in tries.values())


@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (
            checkpoint_upload.CheckpointUploadError(
                "CPU checkpoint upload failed after 3 tries; prior durable save is preserved"
            ),
            "CPU checkpoint upload failed after 3 tries; prior durable save is preserved",
        ),
        (
            TimeoutError("Modal operation exceeded the run deadline"),
            "Final CPU checkpoint publication failed",
        ),
    ],
)
def test_upload_failure_is_not_reported_as_preemption(monkeypatch, error, detail):
    monkeypatch.setattr(jobs, "poll_instance_job", lambda *_a, **_k: PollResult(True, metrics={}))
    monkeypatch.setattr(jobs, "make_artifact_text_reader", lambda *_a, **_k: lambda **_kw: None)
    monkeypatch.setattr(jobs, "start_time_reader", lambda *_a, **_k: lambda: None)
    monkeypatch.setattr(jobs, "heartbeat_reader_for", lambda *_a, **_k: None)

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(checkpoint_upload, "finish", fail)

    result = jobs.poll(_handle(), _spec(), deadline_at=time.time() + 3600)

    assert result.ok is False
    assert result.failure == "checkpoint_upload_failed"
    assert result.detail == detail


@pytest.mark.parametrize("warmstart", [False, True])
@pytest.mark.parametrize("algorithm", ["sft", "grpo"])
def test_recovered_upload_uses_launched_spec_not_remaining_wall_spec(
    fake_modal, warmstart, algorithm, tmp_path, monkeypatch
):
    from flash.engine.worker.io import artifacts, cpu_handoff
    from flash.engine.worker.runtime.state import WorkerConfig, WorkerContext
    from flash.runner.lifecycle import deadlines

    modal = fake_modal([0])
    original = replace(_spec(), algorithm=algorithm)
    if warmstart:
        original = replace(
            original,
            train=replace(
                original.train,
                init_from_adapter="org/source:sft/parent/checkpoints/step-10/adapter",
                init_from_adapter_revision="a" * 40,
                lora_rank=16,
                lora_alpha=64,
            ),
        )
    # Round-tripping is sound. Recovery changes the allowance independently of serialization.
    assert JobSpec.from_json(original.to_json()).to_json() == original.to_json()
    launched_hash = hashlib.sha256(original.to_json().encode()).hexdigest()
    modal.parent_tags["flash_spec"] = launched_hash
    monkeypatch.setattr(deadlines, "_remaining_run_wall_seconds", lambda *_a, **_k: 90)
    recovered = deadlines._spec_with_remaining_wall(original, require_provider_minimum=False)
    assert hashlib.sha256(recovered.to_json().encode()).hexdigest() != launched_hash

    monkeypatch.setattr(jobs, "poll_instance_job", lambda *_a, **_k: PollResult(True, metrics={}))
    monkeypatch.setattr(jobs, "make_artifact_text_reader", lambda *_a, **_k: lambda **_kw: None)
    monkeypatch.setattr(jobs, "start_time_reader", lambda *_a, **_k: lambda: None)
    monkeypatch.setattr(jobs, "heartbeat_reader_for", lambda *_a, **_k: None)
    result = jobs.poll(_handle(phase=original.phase), recovered, deadline_at=time.time() + 3600)
    assert result.ok

    assert modal.environments[0]["FLASH_EXPECTED_SPEC_SHA256"] == launched_hash
    # Feed the real staged GPU handoff and the uploader's actual environment into the CPU
    # entrypoint. Only artifact transport is fake; a wrong hash prevents every publication.
    root = tmp_path / "handoff"
    root.mkdir()
    source = tmp_path / "native"
    source.mkdir()
    (source / "model.pt").write_bytes(b"native checkpoint")
    monkeypatch.setattr(cpu_handoff, "MOUNT", str(root))
    context = WorkerContext(WorkerConfig(spec=original, run_id=original.run_id))
    cpu_handoff.stage_final_checkpoint(2, str(source), final_step=2, context=context)
    (root / "metrics.json").write_text('{"step":2}')
    (root / "DONE").write_text("done")
    (root / "terminal.json").write_text('{"ok":true}')
    monkeypatch.delenv("FLASH_JOB_SPEC_JSON", raising=False)
    for key, value in modal.environments[0].items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("FLASH_JOB_SPEC_PATH", str(root / "spec.json"))
    published = []
    monkeypatch.setattr(
        artifacts,
        "upload_resume_checkpoint",
        lambda *a, **k: artifacts.PublicationResult(
            resume=artifacts.ArtifactCommit("native", "revision")
        ),
    )
    monkeypatch.setattr(
        artifacts, "upload_artifact_file", lambda path, name, **k: published.append(name)
    )
    cpu_handoff.main()
    assert published == ["cpu_upload_attempt0.json", "metrics.json", "DONE", "modal_attempt0.json"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("flash_owner", "other-owner"),
        ("flash_run", "other-run"),
        ("flash_attempt", "1"),
        ("flash_name", "other-name"),
        ("flash_handoff", "other-volume"),
        ("flash_spec", None),
        ("flash_spec", "not-a-digest"),
    ],
)
def test_upload_rejects_foreign_or_invalid_launch_identity(fake_modal, key, value):
    modal = fake_modal([0])
    modal.parent_tags[key] = value

    with pytest.raises(checkpoint_upload.CheckpointUploadError, match="parent launch"):
        checkpoint_upload.finish(_handle(), deadline_at=time.time() + 3600)

    assert modal.created == []
