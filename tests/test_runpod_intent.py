"""Remote lease and CAS ownership contracts for RunPod controller intents."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from flash.core.spec import GpuSpec, JobSpec
from flash.providers.runpod.client import api as runpod_api
from flash.providers.runpod.client import pods as runpod_pods
from flash.providers.runpod.execution.hf_intent import HfRunpodIntentStore, IntentLeaseHeld
from flash.providers.runpod.execution.identity import PRE_POD_CREATE, RunpodPodHandle


class _CasConflict(RuntimeError):
    response = SimpleNamespace(status_code=409)


class _RemoteRepo:
    def __init__(self):
        self.record = None
        self.revision = 0
        self.race_record = None

    def repo_info(self, **kwargs):
        return SimpleNamespace(sha=f"r{self.revision}")

    def list_repo_files(self, **kwargs):
        return ["intent.json"] if self.record is not None else []

    def upload_file(self, **kwargs):
        if kwargs["parent_commit"] != f"r{self.revision}":
            raise _CasConflict("stale parent")
        if self.race_record is not None:
            self.record = self.race_record
            self.race_record = None
            self.revision += 1
            raise _CasConflict("raced")
        stream = kwargs["path_or_fileobj"]
        stream.seek(0)
        self.record = json.loads(stream.read())
        self.revision += 1
        return SimpleNamespace(oid=f"r{self.revision}")


def _spec() -> JobSpec:
    return JobSpec(
        model="Qwen/Qwen3.5-9B",
        run_id="intent-test",
        gpu=GpuSpec(type="RTX 4090", disk_gb=60),
    )


def _handle() -> RunpodPodHandle:
    label = "flash-preload-d1999999999-runpod-us-ca-2-a-0123456789abcdef-12345678"
    return RunpodPodHandle(
        instance_id=label,
        gpu="RTX 4090",
        hourly_usd=0.0,
        attempt=0,
        started_ts=time.time(),
        phase=PRE_POD_CREATE,
        label=label,
        key_fingerprint=runpod_api.key_fingerprint("key"),
        account_id="account",
        payload_secret_id="secret",
        payload_secret_name="FLASH_PAYLOAD_0123456789abcdef",
        data_center_id="US-CA-2",
        network_volume_id="volume",
        container_disk_gb=60,
        gpu_count=1,
    )


def _store(remote, owner):
    return HfRunpodIntentStore(
        remote,
        "owner/status",
        "intent.json",
        "token",
        "preload",
        "account:dc",
        owner,
    )


@pytest.fixture
def remote_download(monkeypatch, tmp_path):
    remote = _RemoteRepo()
    path = tmp_path / "intent.json"

    def download(**kwargs):
        path.write_text(json.dumps(remote.record))
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    return remote


def test_live_remote_owner_cannot_be_displaced_or_cleaned(remote_download, monkeypatch):
    from flash.providers.artifacts import preload_runpod

    owner = _store(remote_download, "owner-a")
    owner.publish_active("run-id", 0, _handle().to_dict())
    contender = _store(remote_download, "owner-b")
    terminated = []
    monkeypatch.setattr(
        preload_runpod,
        "terminate_handle",
        lambda *args, **kwargs: terminated.append(True),
    )
    with pytest.raises(IntentLeaseHeld, match="live remote owner"):
        preload_runpod._recover_target_intent(contender, 60)
    assert remote_download.record["owner"] == "owner-a"
    assert terminated == []


def test_expired_remote_owner_is_atomically_claimed(remote_download):
    owner = _store(remote_download, "owner-a")
    owner.publish_active("run-id", 0, _handle().to_dict())
    remote_download.record["lease_expires_at"] = time.time() - 1
    contender = _store(remote_download, "owner-b")
    claimed = contender.claim_expired()
    assert claimed["owner"] == "owner-b"
    assert claimed["lease_expires_at"] > time.time()
    assert remote_download.record["owner"] == "owner-b"


def test_launch_cancellation_guard_failure_still_terminates_and_preserves_original(monkeypatch):
    from flash.providers.runpod.execution import pods
    from flash.runner.supervise.errors import _TerminalHandleRace

    spec = _spec()
    persisted = []
    cleaned = []
    creates = []
    monkeypatch.setattr(
        runpod_pods,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: ("account", []),
    )
    monkeypatch.setattr(
        runpod_pods,
        "create_secret_for_fingerprint",
        lambda selected, name, value, **kwargs: runpod_pods.RunpodSecret("secret", name),
    )
    monkeypatch.setattr(pods, "terminate_handle", lambda handle, **kwargs: cleaned.append(handle))
    monkeypatch.setattr(
        pods,
        "create_or_adopt_pod",
        lambda *args, **kwargs: creates.append(True),
    )

    pre_pod_callbacks = []

    def persist(handle):
        if handle["phase"] == PRE_POD_CREATE:
            pre_pod_callbacks.append(1)
            if len(pre_pod_callbacks) == 2:
                raise _TerminalHandleRace("cancelled")
        persisted.append(handle)

    def guard():
        raise RuntimeError("intent renewal unavailable")

    with pytest.raises(_TerminalHandleRace, match="cancelled") as exc_info:
        pods.launch_payload_pod(
            spec,
            0,
            serialized_payload="opaque",
            fingerprint=runpod_api.key_fingerprint("key"),
            data_center_id="US-CA-2",
            network_volume_id="volume",
            on_handle=persist,
            cleanup_guard=guard,
            deadline_at=time.time() + 60,
        )
    assert persisted[-1]["phase"] == PRE_POD_CREATE
    assert persisted[-1]["payload_secret_id"] == "secret"
    assert pre_pod_callbacks == [1, 1]
    assert creates == []
    assert len(cleaned) == 1
    assert cleaned[0].payload_secret_id == "secret"
    assert any(
        "rollback guard also failed: RuntimeError: intent renewal unavailable" in note
        for note in exc_info.value.__notes__
    )


def test_launch_cancellation_preserved_when_guard_and_teardown_both_fail(monkeypatch):
    from flash.providers.runpod.execution import pods
    from flash.runner.supervise.errors import _TerminalHandleRace

    spec = _spec()
    creates = []
    teardown_calls = []
    monkeypatch.setattr(
        runpod_pods,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: ("account", []),
    )
    monkeypatch.setattr(
        runpod_pods,
        "create_secret_for_fingerprint",
        lambda selected, name, value, **kwargs: runpod_pods.RunpodSecret("secret", name),
    )
    monkeypatch.setattr(
        pods,
        "create_or_adopt_pod",
        lambda *args, **kwargs: creates.append(True),
    )

    def terminate(handle, **kwargs):
        teardown_calls.append(handle)
        raise runpod_api.RunpodApiError("teardown unavailable")

    monkeypatch.setattr(pods, "terminate_handle", terminate)
    pre_pod_callbacks = []

    def persist(handle):
        if handle["phase"] == PRE_POD_CREATE:
            pre_pod_callbacks.append(1)
            if len(pre_pod_callbacks) == 2:
                raise _TerminalHandleRace("cancelled")

    def guard():
        raise RuntimeError("intent renewal unavailable")

    with pytest.raises(_TerminalHandleRace, match="cancelled") as exc_info:
        pods.launch_payload_pod(
            spec,
            0,
            serialized_payload="opaque",
            fingerprint=runpod_api.key_fingerprint("key"),
            data_center_id="US-CA-2",
            network_volume_id="volume",
            on_handle=persist,
            cleanup_guard=guard,
            deadline_at=time.time() + 60,
        )

    assert pre_pod_callbacks == [1, 1]
    assert creates == []
    assert len(teardown_calls) == 1
    notes = exc_info.value.__notes__
    assert any(
        "rollback guard also failed: RuntimeError: intent renewal unavailable" in note
        for note in notes
    )
    assert any(
        "rollback teardown also failed: RunpodApiError: teardown unavailable" in note
        for note in notes
    )


def test_launch_owned_local_error_still_rolls_back(monkeypatch):
    from flash.providers.runpod.execution import pods

    spec = _spec()
    cleaned = []
    guarded = []
    monkeypatch.setattr(
        runpod_pods,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: ("account", []),
    )
    monkeypatch.setattr(
        runpod_pods,
        "create_secret_for_fingerprint",
        lambda selected, name, value, **kwargs: runpod_pods.RunpodSecret("secret", name),
    )
    monkeypatch.setattr(pods, "terminate_handle", lambda handle, **kwargs: cleaned.append(handle))

    pre_pod_callbacks = []

    def fail_after_secret(handle):
        if handle["phase"] == PRE_POD_CREATE:
            pre_pod_callbacks.append(1)
            if len(pre_pod_callbacks) == 2:
                raise RuntimeError("local callback failed")

    with pytest.raises(RuntimeError, match="local callback failed"):
        pods.launch_payload_pod(
            spec,
            0,
            serialized_payload="opaque",
            fingerprint=runpod_api.key_fingerprint("key"),
            data_center_id="US-CA-2",
            network_volume_id="volume",
            on_handle=fail_after_secret,
            cleanup_guard=lambda: guarded.append(True),
            deadline_at=time.time() + 60,
        )
    assert pre_pod_callbacks == [1, 1]
    assert guarded == [True]
    assert len(cleaned) == 1
    assert cleaned[0].payload_secret_id == "secret"


def test_claim_cas_race_reloads_and_respects_new_live_owner(remote_download):
    owner = _store(remote_download, "owner-a")
    owner.publish_active("run-id", 0, _handle().to_dict())
    remote_download.record["lease_expires_at"] = time.time() - 1
    raced = {
        **remote_download.record,
        "owner": "owner-c",
        "lease_expires_at": time.time() + 600,
    }
    remote_download.race_record = raced
    contender = _store(remote_download, "owner-b")
    with pytest.raises(IntentLeaseHeld, match="live remote owner"):
        contender.claim_expired()
    assert remote_download.record["owner"] == "owner-c"
