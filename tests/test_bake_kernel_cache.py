"""Contracts for the strict RunPod Pod kernel-cache bake controller."""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
BAKE_SCRIPT = ROOT / "docker" / "bake_kernel_cache.py"


def _load_bake():
    spec = importlib.util.spec_from_file_location("bake_kernel_cache", BAKE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bake = _load_bake()


def test_default_gpu_walk_covers_every_baked_arch():
    from flash.providers._lifecycle.net.worker import BAKED_PER_SM_ARCHES

    assert set(bake.GPU_WALK_BY_SM) == BAKED_PER_SM_ARCHES
    all_types = [gpu for choices in bake.GPU_WALK_BY_SM.values() for gpu in choices]
    assert all(bake.GPU_WALK_BY_SM.values())
    assert len(all_types) == len(set(all_types))


def test_capacity_only_walk_uses_shared_phase_aware_launcher(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.client import auth as runpod_auth
    from flash.providers.runpod.client import pods as runpod_pods
    from flash.providers.runpod.execution import pods

    attempts = []
    monkeypatch.setattr(runpod_auth, "ordered_keys", lambda: ["key-a", "key-b"])

    def callback(value):
        return None

    def guard():
        return None

    def launch(spec, seed, **kwargs):
        attempts.append(
            (
                kwargs["fingerprint"],
                kwargs["gpu_type_id_override"],
                kwargs["on_handle"],
                kwargs["cleanup_guard"],
            )
        )
        if len(attempts) < 3:
            raise runpod_pods.RunpodCapacityError("full")
        return SimpleNamespace(pod_id="pod-1", key_fingerprint=kwargs["fingerprint"])

    monkeypatch.setattr(pods, "launch_payload_pod", launch)
    handle, selected = bake._launch_with_gpu_walk(
        SimpleNamespace(seed=0),
        "{}",
        image="image@sha256:abc",
        gpu_type_ids=("gpu-a", "gpu-b"),
        allowed_cuda=("13.0",),
        deadline_at=time.time() + 60,
        on_handle=callback,
        cleanup_guard=guard,
        rounds=1,
    )
    assert handle.pod_id == "pod-1"
    assert selected == "gpu-a"
    assert attempts == [
        (runpod_api.key_fingerprint("key-a"), "gpu-a", callback, guard),
        (runpod_api.key_fingerprint("key-a"), "gpu-b", callback, guard),
        (runpod_api.key_fingerprint("key-b"), "gpu-a", callback, guard),
    ]


def test_non_capacity_failure_never_walks_to_another_gpu(monkeypatch):
    from flash.providers.runpod.client import auth as runpod_auth
    from flash.providers.runpod.execution import pods

    calls = []
    monkeypatch.setattr(runpod_auth, "ordered_keys", lambda: ["key-a"])

    def fail(*args, **kwargs):
        calls.append(kwargs["gpu_type_id_override"])
        raise RuntimeError("unauthorized")

    monkeypatch.setattr(pods, "launch_payload_pod", fail)
    with pytest.raises(RuntimeError, match="unauthorized"):
        bake._launch_with_gpu_walk(
            SimpleNamespace(seed=0),
            "{}",
            image="image",
            gpu_type_ids=("gpu-a", "gpu-b"),
            allowed_cuda=(),
            deadline_at=time.time() + 60,
            rounds=2,
            backoff_s=(0,),
        )
    assert calls == ["gpu-a"]


@pytest.mark.parametrize(
    ("sm", "allowed", "expected"),
    [
        ("sm80", None, ["12.8"]),
        ("sm86", None, ["12.8"]),
        ("sm89", None, ["12.8"]),
        ("sm90", None, ["12.8"]),
        ("sm100", ("13.0",), ["13.0"]),
        ("sm120", ("13.0",), ["13.0"]),
    ],
)
def test_bake_launch_boundary_cuda_floor(monkeypatch, sm, allowed, expected):
    from flash.providers.runpod.client import auth as runpod_auth
    from flash.providers.runpod.execution import pods
    from flash.providers.runpod.execution.identity import RunpodPodHandle, payload_for_handle

    monkeypatch.setattr(runpod_auth, "ordered_keys", lambda: ["key"])
    observed = []

    def launch(spec, seed, **kwargs):
        observed.append(kwargs)
        return SimpleNamespace(pod_id="pod-1", key_fingerprint=kwargs["fingerprint"])

    monkeypatch.setattr(pods, "launch_payload_pod", launch)

    def callback(value):
        return None

    def guard():
        return None

    bake._launch_with_gpu_walk(
        SimpleNamespace(seed=0),
        "{}",
        image="image",
        gpu_type_ids=(bake.GPU_WALK_BY_SM[sm][0],),
        allowed_cuda=allowed,
        deadline_at=time.time() + 60,
        on_handle=callback,
        cleanup_guard=guard,
        rounds=1,
    )
    assert observed[0]["allowed_cuda_versions"] == allowed
    assert observed[0]["on_handle"] is callback
    assert observed[0]["cleanup_guard"] is guard
    handle = RunpodPodHandle(
        instance_id="label",
        gpu="RTX 4090",
        hourly_usd=0.0,
        attempt=0,
        started_ts=time.time(),
        phase=pods.PRE_POD_CREATE,
        label="label",
        key_fingerprint="rpk-" + "0" * 64,
        account_id="account",
        payload_secret_id="secret",
        payload_secret_name="FLASH_PAYLOAD_0123456789abcdef",
        data_center_id=None,
        network_volume_id=None,
        container_disk_gb=60,
        gpu_count=1,
        image_name="image",
        gpu_type_id_override=bake.GPU_WALK_BY_SM[sm][0],
        allowed_cuda_versions=allowed,
    )
    assert payload_for_handle(handle)["allowedCudaVersions"] == expected


@pytest.mark.parametrize(
    ("repo_files", "pod_statuses", "expected", "expected_pod_calls"),
    [
        ([[], ["out/STATUS"]], ["RUNNING"], "done", 1),
        ([[]], ["DEAD", "DEAD"], "pod_died", 2),
        ([["out/STATUS"]], [], "done", 0),
    ],
)
def test_poll_bake_uses_strict_pod_client(
    monkeypatch, repo_files, pod_statuses, expected, expected_pod_calls
):
    from flash.providers.runpod.client import api as runpod_api
    from flash.providers.runpod.client import pods as runpod_pods

    observed = []

    def list_repo_files(repo, *, repo_type):
        assert repo == "owner/repo"
        assert repo_type == "dataset"
        if len(repo_files) > 1:
            return repo_files.pop(0)
        return repo_files[0]

    def get_pod(pod_id, fingerprint, *, deadline_at):
        observed.append((pod_id, fingerprint, deadline_at))
        return SimpleNamespace(desired_status=pod_statuses.pop(0))

    def wrong_owner(*args, **kwargs):
        raise AssertionError("strict Pod lookup must use client.pods")

    monkeypatch.setattr(runpod_pods, "get_pod_for_fingerprint", get_pod)
    monkeypatch.setattr(runpod_api, "get_pod_for_fingerprint", wrong_owner, raising=False)
    monkeypatch.setattr(bake.time, "sleep", lambda seconds: None)
    deadline_at = time.time() + 60
    handle = SimpleNamespace(pod_id="pod-1", key_fingerprint="rpk-fingerprint")

    assert (
        bake._poll_bake(
            SimpleNamespace(list_repo_files=list_repo_files),
            "owner/repo",
            handle,
            deadline_at,
        )
        == expected
    )
    assert observed == [("pod-1", "rpk-fingerprint", deadline_at)] * expected_pod_calls


def test_empty_allowed_cuda_means_no_override():
    assert bake._allowed_cuda_override("") is None
    assert bake._allowed_cuda_override("13.0") == ("13.0",)


def test_bake_payload_is_opaque_secret_content_not_process_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "ambient-secret")
    payload = bake._launch_payload("owner/repo", "9.0", "request-secret")
    assert '"mode":"kernel_bake"' in payload
    assert '"hf_token":"request-secret"' in payload
    assert "RUNPOD_API_KEY" not in payload


def test_bake_recovery_retains_intent_when_cleanup_is_unconfirmed(monkeypatch):
    from flash.providers.runpod.execution import pods

    pending = pods.RunpodPodHandle(
        instance_id="label-0123456789abcdef-12345678",
        gpu="RTX 4090",
        hourly_usd=0.0,
        attempt=0,
        started_ts=time.time(),
        phase=pods.POD_CREATE_PENDING,
        label="label-0123456789abcdef-12345678",
        key_fingerprint="rpk-" + "0" * 64,
        account_id="account",
        payload_secret_id="secret",
        payload_secret_name="FLASH_PAYLOAD_0123456789abcdef",
        data_center_id=None,
        network_volume_id=None,
        container_disk_gb=60,
        gpu_count=1,
    )
    record = {
        "owner": "bake-owner",
        "run_id": "run-id",
        "seed": 0,
        "handle": pending.to_dict(),
    }
    published = []
    cleared = []
    store = SimpleNamespace(
        owner="new-owner",
        claim_expired=lambda: record,
        renew=lambda: record,
        publish_active=lambda *args: published.append(args),
        clear=lambda: cleared.append(True),
    )
    exact = SimpleNamespace(to_dict=lambda: {**pending.to_dict(), "phase": pods.EXACT})
    monkeypatch.setattr(pods, "resolve_pending_handle", lambda *args, **kwargs: exact)
    monkeypatch.setattr(
        pods,
        "terminate_handle",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("secret still present")),
    )
    with pytest.raises(RuntimeError, match="secret still present"):
        bake._recover_bake_intent(store, 60)
    assert store.owner == "new-owner"
    assert published
    assert cleared == []


def test_bake_workflow_passes_stable_run_identity_and_documents_recovery():
    workflow = (ROOT / ".github" / "workflows" / "bake-kernel-cache.yml").read_text()
    assert '--workflow-id "$GITHUB_RUN_ID-${{ matrix.sm }}"' in workflow
    assert "retried job reconciles and terminates an interrupted prior" in workflow
    assert "leaving a billed pod" not in workflow.lower()


def test_bake_scripts_use_static_launcher_and_no_runpod_sdk():
    controller = BAKE_SCRIPT.read_text()
    launcher = (ROOT / "docker" / "runpod_pod_launcher.py").read_text()
    dockerfile = (ROOT / "Dockerfile.worker").read_text()
    intent_helper = (
        ROOT / "flash" / "providers" / "runpod" / "execution" / "hf_intent.py"
    ).read_text()
    assert "launch_payload_pod(" in controller
    assert "import runpod" not in controller
    assert "runpod.create_pod" not in controller
    assert 'parsed.get("mode") == "kernel_bake"' in launcher
    assert "docker/bake_pod_entry.py" in dockerfile
    assert "serialized_payload" not in intent_helper
    assert '"token"' not in intent_helper
