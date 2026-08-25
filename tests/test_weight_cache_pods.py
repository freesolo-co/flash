"""Strict REST and Pod contracts for the RunPod shared weight cache."""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

from flash.providers.runpod import api as runpod_api
from flash.providers.runpod import resources


def _fake_preload_intent(monkeypatch):
    published = []
    store = SimpleNamespace(
        repo="owner/status",
        path=".flash/runpod-intents/preload/test.json",
        renew=lambda: None,
        publish_active=lambda run_id, seed, handle: published.append((run_id, seed, handle)),
    )
    monkeypatch.setattr(
        "flash.providers.artifacts.preload_runpod._intent_store",
        lambda *args, **kwargs: store,
    )
    monkeypatch.setattr(
        "flash.providers.artifacts.preload_runpod._recover_target_intent",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "flash.providers.artifacts.preload_runpod._cleanup_owned_intent",
        lambda *args, **kwargs: None,
    )
    return published


def test_ensure_account_volume_creates_exact_catalog_datacenter(monkeypatch):
    fingerprint = runpod_api.key_fingerprint("key")
    created = runpod_api.RunpodNetworkVolume("vol-1", "flash-weights-us-ca-2", 250, "US-CA-2")
    monkeypatch.setattr(
        resources,
        "weight_cache_datacenters",
        lambda selected, **kwargs: ["US-CA-2"],
    )
    monkeypatch.setattr(
        runpod_api, "list_network_volumes_for_fingerprint", lambda *args, **kwargs: []
    )
    calls = []
    monkeypatch.setattr(
        runpod_api,
        "create_network_volume_for_fingerprint",
        lambda selected, **kwargs: calls.append((selected, kwargs)) or created,
    )
    actual = resources.ensure_account_volume(
        fingerprint,
        base="flash-weights",
        data_center_id="US-CA-2",
        size_gb=250,
        deadline_at=time.time() + 60,
    )
    assert actual == created
    assert calls[0][1]["data_center_id"] == "US-CA-2"
    assert calls[0][1]["name"] == "flash-weights-us-ca-2"


def test_ensure_account_volume_rejects_non_catalog_datacenter(monkeypatch):
    monkeypatch.setattr(resources, "weight_cache_datacenters", lambda *args, **kwargs: ["EU-RO-1"])
    with pytest.raises(runpod_api.RunpodApiError, match="does not support"):
        resources.ensure_account_volume(
            runpod_api.key_fingerprint("key"),
            base="flash-weights",
            data_center_id="US-CA-2",
            size_gb=250,
            deadline_at=time.time() + 60,
        )


def test_network_volume_delete_requires_confirmed_absence(monkeypatch):
    from flash.providers.runpod import pod_api

    fingerprint = runpod_api.key_fingerprint("key")
    monkeypatch.setattr(pod_api, "_key_for_fingerprint", lambda selected: "key")
    monkeypatch.setattr(pod_api, "_mutation_once", lambda *args, **kwargs: None)
    listings = iter(
        [
            [runpod_api.RunpodNetworkVolume("vol-1", "name", 250, "US-CA-2")],
            [],
        ]
    )
    monkeypatch.setattr(
        pod_api, "list_network_volumes_for_fingerprint", lambda *args, **kwargs: next(listings)
    )
    monkeypatch.setattr(pod_api.time, "sleep", lambda _seconds: None)
    pod_api.delete_network_volume_for_fingerprint(
        fingerprint, "vol-1", deadline_at=time.time() + 60
    )


def test_preload_uses_exact_volume_and_shared_phase_lifecycle(monkeypatch):
    from flash.providers.artifacts import preload_runpod, weight_cache
    from flash.providers.runpod import auth as runpod_auth

    monkeypatch.setattr(runpod_auth, "ordered_keys", lambda: ["key"])
    published = _fake_preload_intent(monkeypatch)
    volume = runpod_api.RunpodNetworkVolume("vol-1", "flash-weights-us-ca-2", 250, "US-CA-2")
    monkeypatch.setattr(resources, "ensure_account_volume", lambda *args, **kwargs: volume)
    handle = SimpleNamespace(
        pod_id="pod-1",
        key_fingerprint=runpod_api.key_fingerprint("key"),
        started_ts=time.time(),
    )
    launches = []
    monkeypatch.setattr(
        preload_runpod,
        "launch_payload_pod",
        lambda spec, seed, **kwargs: launches.append(kwargs) or handle,
    )
    monkeypatch.setattr(
        preload_runpod,
        "_poll_preload",
        lambda *args, **kwargs: {
            "preloaded": ["Qwen/Qwen3.5-0.8B"],
            "failed": {},
            "resolved_snapshots": {"Qwen/Qwen3.5-0.8B": "models--Qwen--Qwen3.5-0.8B/snapshots/abc"},
        },
    )
    monkeypatch.setattr(weight_cache, "_preload_status_repo", lambda: "owner/status")
    result = preload_runpod._preload_one_dc(
        0,
        runpod_api.key_fingerprint("key"),
        "US-CA-2",
        ["Qwen/Qwen3.5-0.8B"],
        "token",
        "RTX 4090",
        60,
        0,
    )
    assert result["status"] == "ok"
    assert result["account"] == "acct0"
    assert launches[0]["data_center_id"] == "US-CA-2"
    assert launches[0]["network_volume_id"] == "vol-1"
    assert callable(launches[0]["cleanup_guard"])
    assert callable(launches[0]["on_handle"])
    launches[0]["on_handle"]({"phase": "durable"})
    assert published[0][2] == {"phase": "durable"}
    payload = json.loads(launches[0]["serialized_payload"])
    assert payload["mode"] == "preload"
    assert payload["env"]["FLASH_WEIGHT_CACHE_DIR"] == "/runpod-volume/hf-cache/hub"


def test_preload_reports_unconfirmed_pod_or_secret_cleanup(monkeypatch):
    from flash.providers.artifacts import preload_runpod, weight_cache

    _fake_preload_intent(monkeypatch)
    volume = runpod_api.RunpodNetworkVolume("vol-1", "flash-weights-us-ca-2", 250, "US-CA-2")
    monkeypatch.setattr(resources, "ensure_account_volume", lambda *args, **kwargs: volume)
    handle = SimpleNamespace(
        pod_id="pod-1",
        key_fingerprint=runpod_api.key_fingerprint("key"),
        started_ts=time.time(),
    )
    monkeypatch.setattr(preload_runpod, "launch_payload_pod", lambda *args, **kwargs: handle)
    monkeypatch.setattr(
        preload_runpod,
        "_poll_preload",
        lambda *args, **kwargs: {"preloaded": ["model"], "failed": {}},
    )
    recoveries = []

    def recover(*args, **kwargs):
        recoveries.append(1)
        raise RuntimeError("secret still present")

    monkeypatch.setattr(preload_runpod, "_cleanup_owned_intent", recover)
    monkeypatch.setattr(weight_cache, "_preload_status_repo", lambda: "owner/status")
    result = preload_runpod._preload_one_dc(
        0,
        runpod_api.key_fingerprint("key"),
        "US-CA-2",
        ["model"],
        "token",
        "RTX 4090",
        60,
        0,
    )
    assert result["status"] == "error"
    assert result["error"] == "cleanup unconfirmed: secret still present"


def test_success_attempt_marker_without_result_evidence_fails(monkeypatch):
    from flash.providers.artifacts import preload_runpod, weight_cache

    monkeypatch.setattr(weight_cache, "_preload_status_repo", lambda: "owner/status")
    readers = iter(
        [
            lambda force=False: None,
            lambda force=False: json.dumps({"ok": True}),
        ]
    )
    monkeypatch.setattr(
        preload_runpod, "make_hf_text_reader", lambda *args, **kwargs: next(readers)
    )
    monkeypatch.setattr(
        runpod_api,
        "get_pod_for_fingerprint",
        lambda *args, **kwargs: SimpleNamespace(desired_status="TERMINATED"),
    )
    monkeypatch.setattr(preload_runpod.time, "sleep", lambda _seconds: None)
    spec = preload_runpod._preload_spec("RTX 4090", "run-id", 60)
    handle = SimpleNamespace(
        pod_id="pod-1",
        key_fingerprint=runpod_api.key_fingerprint("key"),
        started_ts=time.time(),
    )
    with pytest.raises(RuntimeError, match="without validated completion evidence"):
        preload_runpod._poll_preload(handle, spec, ["model"], 60, 0)


def test_preload_ambiguous_create_never_retries_the_target(monkeypatch):
    from flash.providers.artifacts import preload_runpod, weight_cache
    from flash.providers.base import UnreconciledCreateError

    _fake_preload_intent(monkeypatch)
    monkeypatch.setattr(
        resources,
        "ensure_account_volume",
        lambda *args, **kwargs: runpod_api.RunpodNetworkVolume(
            "vol-1", "flash-weights-us-ca-2", 250, "US-CA-2"
        ),
    )
    attempts = []

    def ambiguous(spec, seed, **kwargs):
        attempts.append(kwargs["fingerprint"])
        raise UnreconciledCreateError("unknown create")

    monkeypatch.setattr(preload_runpod, "launch_payload_pod", ambiguous)
    monkeypatch.setattr(weight_cache, "_preload_status_repo", lambda: "owner/status")
    fingerprint = runpod_api.key_fingerprint("key-a")
    result = preload_runpod._preload_one_dc(
        0, fingerprint, "US-CA-2", ["model"], "token", "RTX 4090", 60, 0
    )
    assert result["status"] == "error"
    assert attempts == [fingerprint]


def test_warm_weight_cache_launches_every_account_datacenter_target(monkeypatch):
    from flash.providers.artifacts import preload_runpod, weight_cache

    fp_a = runpod_api.key_fingerprint("key-a")
    fp_b = runpod_api.key_fingerprint("key-b")
    targets = [
        (0, fp_a, "US-CA-2"),
        (0, fp_a, "US-WA-1"),
        (1, fp_b, "US-CA-2"),
    ]
    monkeypatch.setattr(weight_cache, "_account_storage_targets", lambda: targets)
    monkeypatch.setattr(weight_cache, "_ensure_status_repo", lambda token: None)
    launched = []

    def one(account_index, fingerprint, dc_id, *args):
        launched.append((account_index, fingerprint, dc_id))
        return {
            "account": f"acct{account_index}",
            "datacenter": dc_id,
            "status": "ok",
        }

    monkeypatch.setattr(weight_cache, "_preload_one_dc", one)
    results = preload_runpod.warm_weight_cache(
        models=["model"], datacenters=["US-CA-2"], max_workers=2, token="token"
    )
    assert set(launched) == {(0, fp_a, "US-CA-2"), (1, fp_b, "US-CA-2")}
    assert {(result["account"], result["datacenter"]) for result in results} == {
        ("acct0", "US-CA-2"),
        ("acct1", "US-CA-2"),
    }


def test_account_target_discovery_fails_closed_when_any_account_catalog_fails(monkeypatch):
    from flash.providers.artifacts import preload_runpod
    from flash.providers.runpod import auth as runpod_auth

    monkeypatch.setattr(runpod_auth, "ordered_keys", lambda: ["key-a", "key-b"])
    fp_a = runpod_api.key_fingerprint("key-a")

    def discover(fingerprint, **kwargs):
        if fingerprint == fp_a:
            return ["US-CA-2"]
        raise runpod_api.RunpodApiError("catalog unavailable")

    monkeypatch.setattr(resources, "weight_cache_datacenters", discover)
    with pytest.raises(runpod_api.RunpodApiError, match="acct1"):
        preload_runpod._account_storage_targets()


def test_ambiguous_payload_secret_is_adopted_without_second_create(monkeypatch):
    from flash.providers.runpod import pods

    intent = pods.RunpodPodHandle(
        instance_id="flash-preload-d1999999999-runpod-us-ca-2-abc-s0-a0",
        gpu="RTX 4090",
        hourly_usd=0.0,
        attempt=0,
        started_ts=time.time(),
        phase=pods.SECRET_CREATE_PENDING,
        label="flash-preload-d1999999999-runpod-us-ca-2-abc-s0-a0",
        key_fingerprint=runpod_api.key_fingerprint("key"),
        account_id="account-1",
        payload_secret_id=None,
        payload_secret_name="FLASH_PAYLOAD_0123456789abcdef",
        data_center_id="US-CA-2",
        network_volume_id="vol-1",
        container_disk_gb=60,
        gpu_count=1,
    )
    creates = []
    monkeypatch.setattr(
        runpod_api,
        "create_secret_for_fingerprint",
        lambda *args, **kwargs: (
            creates.append(1)
            or (_ for _ in ()).throw(runpod_api.RunpodMutationAmbiguous("unknown"))
        ),
    )
    monkeypatch.setattr(
        runpod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: (
            intent.account_id,
            [runpod_api.RunpodSecret("secret-1", intent.payload_secret_name)],
        ),
    )
    adopted = pods._create_payload_secret(intent, "{}", deadline_at=time.time() + 60)
    assert adopted.payload_secret_id == "secret-1"
    assert creates == [1]


def test_killed_secret_create_recovers_exact_secret_without_second_mutation(monkeypatch):
    from flash.providers.artifacts import preload_runpod
    from flash.providers.runpod import pods

    spec = preload_runpod._preload_spec("RTX 4090", "run-id", 60)
    fingerprint = runpod_api.key_fingerprint("key")
    record = {}

    class Store:
        owner = "new-owner"

        def claim_expired(self):
            return record or None

        def renew(self):
            return record

        def publish_active(self, run_id, seed, handle):
            record.update({"owner": "test-owner", "run_id": run_id, "seed": seed, "handle": handle})

        def clear(self):
            record.clear()

    secrets = []
    creates = []
    deletes = []
    lists = []

    def list_secrets(*args, **kwargs):
        lists.append(1)
        if len(lists) == 2:
            raise KeyboardInterrupt
        return "account-1", list(secrets)

    monkeypatch.setattr(runpod_api, "list_secrets_for_fingerprint", list_secrets)

    def killed_create(selected, name, payload, **kwargs):
        creates.append(name)
        secrets.append(runpod_api.RunpodSecret("secret-1", name))
        raise runpod_api.RunpodMutationAmbiguous("response lost")

    monkeypatch.setattr(runpod_api, "create_secret_for_fingerprint", killed_create)
    monkeypatch.setattr(
        runpod_api,
        "delete_secret_for_fingerprint",
        lambda selected, secret_id, name, **kwargs: deletes.append((secret_id, name)),
    )
    with pytest.raises(KeyboardInterrupt):
        pods.launch_payload_pod(
            spec,
            spec.seed,
            serialized_payload="opaque",
            fingerprint=fingerprint,
            data_center_id="US-CA-2",
            network_volume_id="vol-1",
            on_handle=lambda handle: Store().publish_active(spec.run_id, spec.seed, handle),
            deadline_at=time.time() + 60,
        )
    assert record["handle"]["phase"] == pods.SECRET_CREATE_PENDING
    preload_runpod._recover_target_intent(Store(), 60)
    assert len(creates) == 1
    assert deletes == [("secret-1", creates[0])]
    assert record == {}


def test_pod_create_pending_cleanup_failure_retains_intent(monkeypatch):
    from flash.providers.artifacts import preload_runpod
    from flash.providers.runpod import pods

    pending = pods.RunpodPodHandle(
        instance_id="flash-preload-d1999999999-runpod-us-ca-2-abc-0123456789abcdef-12345678",
        gpu="RTX 4090",
        hourly_usd=0.0,
        attempt=0,
        started_ts=time.time(),
        phase=pods.POD_CREATE_PENDING,
        label="flash-preload-d1999999999-runpod-us-ca-2-abc-0123456789abcdef-12345678",
        key_fingerprint=runpod_api.key_fingerprint("key"),
        account_id="account-1",
        payload_secret_id="secret-1",
        payload_secret_name="FLASH_PAYLOAD_0123456789abcdef",
        data_center_id="US-CA-2",
        network_volume_id="vol-1",
        container_disk_gb=60,
        gpu_count=1,
    )
    record = {
        "owner": "test-owner",
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
    monkeypatch.setattr(preload_runpod, "resolve_pending_handle", lambda *args, **kwargs: exact)
    monkeypatch.setattr(
        preload_runpod,
        "terminate_handle",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("pod still present")),
    )
    with pytest.raises(RuntimeError, match="pod still present"):
        preload_runpod._recover_target_intent(store, 60)
    assert published
    assert cleared == []
    assert record["handle"]["phase"] == pods.POD_CREATE_PENDING


def test_prefetch_rejects_snapshot_outside_shared_mount(tmp_path, monkeypatch):
    import huggingface_hub

    from flash.engine.worker.io import hf

    mount = tmp_path / "runpod-volume"
    mount.mkdir()
    shared = mount / "hf-cache" / "hub"
    outside = tmp_path / "outside" / "snapshots" / "abc"
    outside.mkdir(parents=True)
    (outside / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DIR", str(shared))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(outside))
    monkeypatch.setattr(hf, "gpu_diagnostics", dict)
    monkeypatch.setattr(hf._w, "heartbeat", lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="outside"):
        hf.prefetch_model("org/model")


def test_prefetch_heartbeat_proves_real_under_mount_weights(tmp_path, monkeypatch):
    import huggingface_hub

    from flash.engine.worker.io import hf

    mount = tmp_path / "runpod-volume"
    mount.mkdir()
    shared = mount / "hf-cache" / "hub"
    snapshot = shared / "models--org--model" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setenv("FLASH_WEIGHT_CACHE_DIR", str(shared))
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(snapshot))
    monkeypatch.setattr(hf, "gpu_diagnostics", dict)
    heartbeats = []
    monkeypatch.setattr(
        hf._w, "heartbeat", lambda stage, **fields: heartbeats.append((stage, fields))
    )
    assert isinstance(hf.prefetch_model("org/model"), float)
    cache = heartbeats[-1][1]["cache"]
    assert cache == {
        "shared_mount": True,
        "snapshot_preexisting": True,
        "snapshot_under_mount": True,
        "weights_present": True,
        "snapshot_relative": "models--org--model/snapshots/abc",
    }
    assert str(tmp_path) not in json.dumps(cache)


def test_bootstrap_rejects_partial_index_even_with_standalone_weight(tmp_path):
    from flash.providers._lifecycle.bootstrap_preload import preload_snapshot_evidence

    cache = tmp_path / "hf-cache" / "hub"
    snapshot = cache / "models--org--model" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(b"standalone")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    with pytest.raises(RuntimeError, match="complete model weights"):
        preload_snapshot_evidence(str(snapshot), str(cache))


def test_bootstrap_rejects_under_mount_snapshot_without_weights(tmp_path, monkeypatch):
    import sys
    import types

    from flash.providers._lifecycle import bootstrap

    cache = tmp_path / "hf-cache" / "hub"
    snapshot = cache / "models--org--model" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = lambda **kwargs: str(snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    result = bootstrap.run_preload(
        {"env": {"FLASH_WEIGHT_CACHE_DIR": str(cache)}, "models": ["org/model"]}
    )
    assert result["preloaded"] == []
    assert "weights" in result["failed"]["org/model"]
    assert result["resolved_snapshots"] == {}


def test_cache_modules_do_not_import_runpod_sdk_or_serverless():
    root = os.path.dirname(os.path.dirname(__file__))
    paths = [
        "flash/providers/artifacts/preload_runpod.py",
        "docker/bake_kernel_cache.py",
        "docker/bake_pod_entry.py",
    ]
    for relative in paths:
        with open(os.path.join(root, relative), encoding="utf-8") as stream:
            source = stream.read()
        assert "runpod_flash" not in source
        assert "serverless" not in source.lower()
