"""Focused persistent RunPod Pod lifecycle tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from flash.core.spec import GpuSpec, JobSpec
from flash.providers.core.base import JobHandle, PollResult
from flash.providers.core.registry import INSTANCE_PROVIDERS
from flash.providers.runpod.client import api
from flash.providers.runpod.client import pods as pod_api
from flash.providers.runpod.execution import identity as pod_identity
from flash.providers.runpod.execution import pods
from flash.runner.supervise import recovery


@pytest.mark.parametrize(
    "modules",
    [
        ("flash.providers.runpod.client.pods", "flash.providers.runpod.client.api"),
        ("flash.providers.runpod.client.api", "flash.providers.runpod.client.pods"),
    ],
)
def test_runpod_client_modules_import_cold_in_either_order(modules):
    script = "; ".join(f"import {module}" for module in modules)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(autouse=True)
def _resolve_owner_key(monkeypatch):
    def resolver(fingerprint):
        return "owner-key"

    monkeypatch.setattr(api, "_key_for_fingerprint", resolver)
    monkeypatch.setattr(pod_api, "_key_for_fingerprint", resolver)


def _spec(*, gpu: str = "H100", count: int = 1, volume: str | None = None) -> JobSpec:
    return JobSpec(
        model="Qwen/Qwen3.5-9B",
        run_id="runpod-pod-test",
        gpu=GpuSpec(
            type=gpu,
            count=count,
            disk_gb=120,
            network_volume=volume,
            network_volume_gb=200,
        ),
    )


def _identity(
    *, count: int = 2, data_center: str | None = "US-KS-2", volume_id: str | None = "volume123"
) -> tuple[str, str]:
    spec = _spec(count=count)
    secret_name = "FLASH_PAYLOAD_0123456789abcdef"
    return (
        pod_identity.pod_label(
            spec,
            0,
            1,
            secret_name=secret_name,
            data_center_id=data_center,
            network_volume_id=volume_id,
            container_registry_auth_id=None,
        ),
        secret_name,
    )


def _pod(
    *,
    pod_id: str = "pod123456",
    name: str | None = None,
    status: str = "PENDING",
    gpu: str | None = "NVIDIA H100 80GB HBM3",
    count: int = 2,
    data_center: str | None = "US-KS-2",
    volume_id: str | None = "volume123",
    image: str = "ghcr.io/freesolo-co/flash-worker:cu128-sm90",
    disk: int = 120,
    rate: float | None = 6.58,
    complete: bool = True,
    registry_id: str | None = None,
) -> pod_api.RunpodPod:
    default_name, secret_name = _identity(count=count, data_center=data_center, volume_id=volume_id)
    name = name or default_name
    reference = pod_identity.secret_reference(secret_name)
    return pod_api.RunpodPod(
        id=pod_id,
        name=name,
        desired_status=status,
        image_name=image,
        gpu_type_id=gpu,
        gpu_count=count,
        data_center_id=data_center,
        container_disk_gb=disk,
        network_volume_id=volume_id,
        cost_per_hr=rate,
        docker_start_cmd=pod_identity.POD_LAUNCH_COMMAND if complete else None,
        payload_env_sha256=(
            hashlib.sha256(reference.encode("utf-8")).hexdigest() if complete else None
        ),
        secure_cloud=True if complete else None,
        interruptible=False if complete else None,
        support_public_ip=False if complete else None,
        volume_mount_path=pod_identity.NETWORK_VOLUME_MOUNT if complete else None,
        container_registry_auth_id=registry_id,
        payload_secret_name=secret_name if complete else None,
    )


def _handle(
    *,
    phase: str | None = None,
    exact: bool = False,
    count: int = 2,
    data_center: str | None = "US-KS-2",
    volume_id: str | None = "volume123",
) -> pods.RunpodPodHandle:
    label, secret_name = _identity(count=count, data_center=data_center, volume_id=volume_id)
    selected_phase = pods.EXACT if exact else (phase or pods.POD_CREATE_PENDING)
    return pods.RunpodPodHandle(
        instance_id="pod123456" if selected_phase == pods.EXACT else label,
        gpu="H100",
        hourly_usd=6.58,
        attempt=1,
        started_ts=1_700_000_000.0,
        phase=selected_phase,
        label=label,
        key_fingerprint=api.key_fingerprint("owner-key"),
        account_id="account123",
        payload_secret_id=(None if selected_phase == pods.SECRET_CREATE_PENDING else "secret123"),
        payload_secret_name=secret_name,
        data_center_id=data_center,
        network_volume_id=volume_id,
        container_disk_gb=120,
        gpu_count=count,
    )


def _api_pod_row(env: object) -> dict:
    return {
        "id": "pod123456",
        "name": "flash-0123456789ab-s0-a0",
        "desiredStatus": "RUNNING",
        "imageName": "image:tag",
        "gpuCount": 2,
        "containerDiskInGb": 120,
        "costPerHr": 5.0,
        "machine": {"gpuTypeId": "NVIDIA H100", "dataCenterId": "US-KS-2"},
        "networkVolume": {"id": "volume123"},
        "env": env,
    }


def test_api_parses_payload_only_env_identity():
    reference = "{{ RUNPOD_SECRET_FLASH_PAYLOAD_0123456789abcdef }}"
    parsed = pod_api._pod_rows([_api_pod_row({"FLASH_INSTANCE_PAYLOAD": reference})])
    assert len(parsed) == 1
    assert parsed[0].id == "pod123456"
    assert parsed[0].gpu_type_id == "NVIDIA H100"
    assert parsed[0].payload_env_sha256 == hashlib.sha256(reference.encode()).hexdigest()
    assert parsed[0].payload_secret_name == "FLASH_PAYLOAD_0123456789abcdef"
    assert reference not in repr(parsed)


def test_api_accepts_provider_managed_public_key_without_retaining_it():
    reference = "{{ RUNPOD_SECRET_FLASH_PAYLOAD_0123456789abcdef }}"
    public_key = "ssh-ed25519 provider-managed-raw-value"
    parsed = pod_api._pod_rows(
        [
            _api_pod_row(
                {
                    "FLASH_INSTANCE_PAYLOAD": reference,
                    "PUBLIC_KEY": public_key,
                }
            )
        ]
    )
    assert parsed[0].payload_env_sha256 == hashlib.sha256(reference.encode()).hexdigest()
    assert parsed[0].payload_secret_name == "FLASH_PAYLOAD_0123456789abcdef"
    assert public_key not in repr(parsed)


def test_api_rejects_unknown_env_key_without_exposing_value():
    raw_value = "provider-managed-sensitive-value"
    with pytest.raises(api.RunpodApiError) as exc_info:
        pod_api._pod_rows(
            [
                _api_pod_row(
                    {
                        "FLASH_INSTANCE_PAYLOAD": (
                            "{{ RUNPOD_SECRET_FLASH_PAYLOAD_0123456789abcdef }}"
                        ),
                        "UNKNOWN": raw_value,
                    }
                )
            ]
        )
    assert raw_value not in str(exc_info.value)
    assert raw_value not in repr(exc_info.value)


@pytest.mark.parametrize("public_key", [None, 1, [], {}, "", " ", " padded"])
def test_api_rejects_malformed_provider_managed_public_key(public_key):
    with pytest.raises(api.RunpodApiError, match="environment identity"):
        pod_api._pod_rows(
            [
                _api_pod_row(
                    {
                        "FLASH_INSTANCE_PAYLOAD": (
                            "{{ RUNPOD_SECRET_FLASH_PAYLOAD_0123456789abcdef }}"
                        ),
                        "PUBLIC_KEY": public_key,
                    }
                )
            ]
        )


@pytest.mark.parametrize("env", [[], {"FLASH_INSTANCE_PAYLOAD": "not-a-secret-reference"}])
def test_api_rejects_malformed_payload_env_identity(env):
    with pytest.raises(api.RunpodApiError, match="environment identity"):
        pod_api._pod_rows([_api_pod_row(env)])


@pytest.mark.parametrize("env", [None, {}, {"PUBLIC_KEY": "provider-managed-raw-value"}])
def test_api_preserves_pending_pod_with_incomplete_provider_env(env):
    pending = _handle()
    payload = pods._payload_for_handle(pending)
    row = _api_pod_row(env)
    row.update(
        {
            "name": pending.label,
            "desiredStatus": "PENDING",
            "imageName": payload["imageName"],
            "gpuCount": payload["gpuCount"],
            "containerDiskInGb": payload["containerDiskInGb"],
            "dockerStartCmd": payload["dockerStartCmd"],
            "interruptible": False,
            "supportPublicIp": False,
            "volumeMountPath": payload["volumeMountPath"],
        }
    )
    row["machine"] = {
        "gpuTypeId": payload["gpuTypeIds"][0],
        "dataCenterId": pending.data_center_id,
        "secureCloud": True,
    }
    observed = pod_api._pod_rows([row])[0]
    assert observed.payload_env_sha256 is None
    assert observed.payload_secret_name is None
    assert pods._pod_identity_is_incomplete(
        observed,
        payload,
        network_volume_id=pending.network_volume_id,
        data_center_id=pending.data_center_id,
        allow_preplacement=True,
    )
    assert not pods._pod_matches(
        observed,
        payload,
        network_volume_id=pending.network_volume_id,
        data_center_id=pending.data_center_id,
        allow_preplacement=True,
    )
    assert "provider-managed-raw-value" not in repr(observed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpuCount", True),
        ("gpuCount", 0),
        ("containerDiskInGb", "120"),
        ("desiredStatus", " running "),
    ],
)
def test_api_rejects_malformed_owned_pod_rows(field, value):
    row = {
        "id": "pod123456",
        "name": "flash-0123456789ab-s0-a0",
        "desiredStatus": "PENDING",
        "imageName": "image:tag",
        "gpuCount": 1,
        "containerDiskInGb": 120,
    }
    row[field] = value
    with pytest.raises(api.RunpodApiError):
        pod_api._pod_rows([row])


def test_api_parses_only_storage_capable_data_centers():
    assert pod_api._parse_storage_data_centers(
        {
            "data": {
                "dataCenters": [
                    {"id": "US-KS-2", "storageSupport": True},
                    {"id": "EU-RO-1", "storageSupport": False},
                    {"id": "US-CA-2", "storageSupport": True},
                ]
            }
        }
    ) == ["US-KS-2", "US-CA-2"]


@pytest.mark.parametrize(
    "response",
    [
        {"data": {"dataCenters": [None]}},
        {"data": {"dataCenters": [{"id": "US-KS-2", "storageSupport": None}]}},
        {"data": {"dataCenters": [{"id": "US-KS-2", "storageSupport": 1}]}},
        {"data": {"dataCenters": [{"id": "US-KS-2", "storageSupport": "true"}]}},
        {"errors": [{"message": "forbidden"}]},
    ],
)
def test_api_rejects_malformed_data_center_observations(response):
    with pytest.raises(api.RunpodApiError, match="data center"):
        pod_api._parse_storage_data_centers(response)


def test_api_rejects_duplicate_data_center_ids():
    response = {
        "data": {
            "dataCenters": [
                {"id": "US-KS-2", "storageSupport": True},
                {"id": "US-KS-2", "storageSupport": False},
            ]
        }
    }
    with pytest.raises(api.RunpodApiError, match="duplicate ids"):
        pod_api._parse_storage_data_centers(response)


def test_storage_data_center_discovery_uses_account_key_and_graphql(monkeypatch):
    fingerprint = api.key_fingerprint("owner-key")
    captured = {}

    def graphql_read(key, document, variables, *, deadline_at):
        captured.update(
            key=key,
            document=document,
            variables=variables,
            deadline_at=deadline_at,
        )
        return {"data": {"dataCenters": [{"id": "US-KS-2", "storageSupport": True}]}}

    monkeypatch.setattr(pod_api, "_graphql_read", graphql_read)
    deadline_at = time.time() + 60
    assert pod_api.list_storage_datacenters_for_fingerprint(
        fingerprint, deadline_at=deadline_at
    ) == ["US-KS-2"]
    assert captured == {
        "key": "owner-key",
        "document": pod_api._OBSERVE_STORAGE_DATA_CENTERS,
        "variables": {},
        "deadline_at": deadline_at,
    }


def test_pending_and_exact_handles_round_trip_with_full_owner_identity():
    pending = _handle()
    exact = _handle(exact=True)
    assert pending.pending
    assert pending.pod_id is None
    assert not exact.pending
    assert exact.pod_id == "pod123456"
    assert pods.RunpodPodHandle.from_dict(pending.to_dict()) == pending
    assert pods.RunpodPodHandle.from_dict(exact.to_dict()) == exact
    bad = exact.to_dict()
    bad["key_fingerprint"] = bad["key_fingerprint"][:16]
    with pytest.raises(ValueError, match="fingerprint"):
        pods.RunpodPodHandle.from_dict(bad)


def test_pod_payload_is_exact_secure_shape_and_secret_reference(monkeypatch):
    monkeypatch.setenv("RUNPOD_CONTAINER_REGISTRY_AUTH_ID", "registry123")
    payload = pod_identity.build_pod_payload(
        _spec(count=2),
        label="flash-runpod-pod-test-s0-a1",
        secret_name="FLASH_PAYLOAD_0123456789abcdef",
        data_center_id="US-KS-2",
        network_volume_id="volume123",
    )
    assert payload == {
        "allowedCudaVersions": ["12.8"],
        "cloudType": "SECURE",
        "containerDiskInGb": 120,
        "containerRegistryAuthId": "registry123",
        "dataCenterIds": ["US-KS-2"],
        "dockerStartCmd": ["python", "/opt/flash/runpod_pod_launcher.py"],
        "env": {"FLASH_INSTANCE_PAYLOAD": "{{ RUNPOD_SECRET_FLASH_PAYLOAD_0123456789abcdef }}"},
        "gpuCount": 2,
        "gpuTypeIds": ["NVIDIA H100 80GB HBM3"],
        "imageName": "ghcr.io/freesolo-co/flash-worker:cu128-sm90",
        "interruptible": False,
        "name": "flash-runpod-pod-test-s0-a1",
        "networkVolumeId": "volume123",
        "supportPublicIp": False,
        "volumeInGb": 0,
        "volumeMountPath": "/runpod-volume",
    }
    encoded = json.dumps(payload)
    assert "owner-key" not in encoded
    assert "hf_" not in encoded


def test_payload_secret_name_is_recoverable_from_final_pod_label():
    label, secret_name = _identity()
    assert pod_identity.payload_secret_name_from_pod_label(label) == secret_name


def test_payload_secret_names_are_fresh_random_identities():
    first = pods.fresh_payload_secret_name()
    second = pods.fresh_payload_secret_name()
    assert first != second
    assert first.startswith("FLASH_PAYLOAD_")
    assert len(first.removeprefix("FLASH_PAYLOAD_")) == 16


def test_secret_intent_skips_existing_same_name_without_reuse(monkeypatch):
    names = iter(
        [
            "FLASH_PAYLOAD_0000000000000000",
            "FLASH_PAYLOAD_1111111111111111",
        ]
    )
    seen = []
    monkeypatch.setattr(pods, "fresh_payload_secret_name", lambda: next(names))

    def list_secret(_fingerprint, *, name, deadline_at):
        seen.append(name)
        existing = [pod_api.RunpodSecret("stale", name)] if len(seen) == 1 else []
        return "account123", existing

    monkeypatch.setattr(pod_api, "list_secrets_for_fingerprint", list_secret)
    handle = pods._new_secret_intent(
        _spec(),
        0,
        1,
        api.key_fingerprint("owner-key"),
        container_registry_auth_id=None,
        started_ts=time.time(),
        deadline_at=time.time() + 60,
    )
    assert seen == [
        "FLASH_PAYLOAD_0000000000000000",
        "FLASH_PAYLOAD_1111111111111111",
    ]
    assert handle.payload_secret_name == "FLASH_PAYLOAD_1111111111111111"
    assert handle.payload_secret_id is None


def test_multi_card_shape_is_not_widened():
    payload = pod_identity.build_pod_payload(
        _spec(gpu="B200", count=4),
        label="flash-runpod-pod-test-s0-a1",
        secret_name="FLASH_PAYLOAD_0123456789abcdef",
        data_center_id=None,
        network_volume_id=None,
    )
    assert payload["gpuTypeIds"] == ["NVIDIA B200"]
    assert payload["gpuCount"] == 4
    assert payload["allowedCudaVersions"] == ["13.0"]


def test_create_ambiguity_adopts_one_and_terminates_duplicates(monkeypatch):
    pending = _handle()
    payload = pod_identity.build_pod_payload(
        _spec(count=2),
        label=pending.label,
        secret_name=pending.payload_secret_name,
        data_center_id=pending.data_center_id,
        network_volume_id=pending.network_volume_id,
    )
    duplicates = [_pod(pod_id="pod-a"), _pod(pod_id="pod-b")]
    deleted = []
    monkeypatch.setattr(
        pod_api,
        "create_pod_for_fingerprint",
        lambda *args, **kwargs: (_ for _ in ()).throw(pod_api.RunpodMutationAmbiguous("unknown")),
    )
    monkeypatch.setattr(
        "flash.providers.runpod.client.pods._key_for_fingerprint", lambda fingerprint: "owner-key"
    )
    monkeypatch.setattr(pod_api, "list_pods_for_key", lambda *args, **kwargs: duplicates)
    monkeypatch.setattr(
        pod_api,
        "delete_pod_for_fingerprint",
        lambda pod_id, *args, **kwargs: deleted.append(pod_id),
    )
    exact = pods.create_or_adopt_pod(pending, payload, deadline_at=time.time() + 60)
    assert exact.pod_id == "pod-a"
    assert deleted == ["pod-b"]


def test_ambiguous_create_adopts_matching_pod_and_terminates_conflicts(monkeypatch):
    pending = _handle()
    payload = pod_identity.build_pod_payload(
        _spec(count=2),
        label=pending.label,
        secret_name=pending.payload_secret_name,
        data_center_id=pending.data_center_id,
        network_volume_id=pending.network_volume_id,
    )
    matching = _pod(pod_id="pod-a")
    conflict = _pod(pod_id="pod-b", count=4)
    deleted = []
    monkeypatch.setattr(
        pod_api,
        "create_pod_for_fingerprint",
        lambda *args, **kwargs: (_ for _ in ()).throw(pod_api.RunpodMutationAmbiguous("unknown")),
    )
    monkeypatch.setattr(
        "flash.providers.runpod.client.pods._key_for_fingerprint", lambda fingerprint: "owner-key"
    )
    monkeypatch.setattr(
        pod_api,
        "list_pods_for_key",
        lambda *args, **kwargs: [matching, conflict],
    )
    monkeypatch.setattr(
        pod_api,
        "delete_pod_for_fingerprint",
        lambda pod_id, *args, **kwargs: deleted.append(pod_id),
    )
    exact = pods.create_or_adopt_pod(pending, payload, deadline_at=time.time() + 60)
    assert exact.pod_id == "pod-a"
    assert deleted == ["pod-b"]


def test_direct_created_response_allows_missing_preplacement_fields(monkeypatch):
    pending = _handle()
    payload = pods._payload_for_handle(pending)
    created = _pod(
        name=pending.label,
        status="CREATED",
        gpu=None,
        data_center=None,
        volume_id=None,
    )
    monkeypatch.setattr(pod_api, "create_pod_for_fingerprint", lambda *args, **kwargs: created)
    monkeypatch.setattr(pod_api, "list_pods_for_key", lambda *args, **kwargs: [created])
    exact = pods.create_or_adopt_pod(pending, payload, deadline_at=time.time() + 60)
    assert exact.pod_id == created.id


def test_created_reconciliation_allows_missing_preplacement_fields(monkeypatch):
    pending = _handle()
    payload = pods._payload_for_handle(pending)
    created = _pod(
        name=pending.label,
        status="CREATED",
        gpu=None,
        data_center=None,
        volume_id=None,
    )
    monkeypatch.setattr(
        pod_api,
        "create_pod_for_fingerprint",
        lambda *args, **kwargs: (_ for _ in ()).throw(pod_api.RunpodMutationAmbiguous("unknown")),
    )
    monkeypatch.setattr(pod_api, "list_pods_for_key", lambda *args, **kwargs: [created])
    exact = pods.create_or_adopt_pod(pending, payload, deadline_at=time.time() + 60)
    assert exact.pod_id == created.id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu", "NVIDIA B200"),
        ("data_center", "EU-RO-1"),
        ("volume_id", "volume-other"),
    ],
)
def test_created_reconciliation_rejects_present_wrong_placement(monkeypatch, field, value):
    pending = _handle()
    payload = pods._payload_for_handle(pending)
    observed = _pod(name=pending.label, status="CREATED", **{field: value})
    deleted = []
    monkeypatch.setattr(
        pod_api,
        "create_pod_for_fingerprint",
        lambda *args, **kwargs: (_ for _ in ()).throw(pod_api.RunpodMutationAmbiguous("unknown")),
    )
    monkeypatch.setattr(pod_api, "list_pods_for_key", lambda *args, **kwargs: [observed])
    monkeypatch.setattr(
        pod_api,
        "delete_pod_for_fingerprint",
        lambda pod_id, *args, **kwargs: deleted.append(pod_id),
    )
    with pytest.raises(pods.UnreconciledCreateError, match="conflicting"):
        pods.create_or_adopt_pod(pending, payload, deadline_at=time.time() + 60)
    assert deleted == [observed.id]


def test_ambiguous_create_rejects_unresolved_provider_shape(monkeypatch):
    pending = _handle()
    payload = pod_identity.build_pod_payload(
        _spec(count=2),
        label=pending.label,
        secret_name=pending.payload_secret_name,
        data_center_id=pending.data_center_id,
        network_volume_id=pending.network_volume_id,
    )
    unresolved = _pod(status="RUNNING", gpu=None)
    monkeypatch.setattr(
        pod_api,
        "create_pod_for_fingerprint",
        lambda *args, **kwargs: (_ for _ in ()).throw(pod_api.RunpodMutationAmbiguous("unknown")),
    )
    monkeypatch.setattr(
        "flash.providers.runpod.client.pods._key_for_fingerprint", lambda fingerprint: "owner-key"
    )
    monkeypatch.setattr(pod_api, "list_pods_for_key", lambda *args, **kwargs: [unresolved])
    with pytest.raises(pods.UnreconciledCreateError):
        pods.create_or_adopt_pod(pending, payload, deadline_at=time.time() + 60)


def test_capacity_refusal_does_not_hide_an_observed_matching_pod(monkeypatch):
    pending = _handle()
    payload = pod_identity.build_pod_payload(
        _spec(count=2),
        label=pending.label,
        secret_name=pending.payload_secret_name,
        data_center_id=pending.data_center_id,
        network_volume_id=pending.network_volume_id,
    )
    monkeypatch.setattr(
        pod_api,
        "create_pod_for_fingerprint",
        lambda *args, **kwargs: (_ for _ in ()).throw(pod_api.RunpodCapacityError("capacity")),
    )
    with pytest.raises(pod_api.RunpodCapacityError):
        pods.create_or_adopt_pod(pending, payload, deadline_at=time.time() + 60)


def test_exact_delete_requires_followup_absence(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "flash.providers.runpod.client.pods._key_for_fingerprint", lambda fingerprint: "owner-key"
    )
    monkeypatch.setattr(
        pod_api,
        "_mutation_once",
        lambda key, url, **kwargs: calls.append((key, url, kwargs["method"])),
    )
    monkeypatch.setattr(pod_api, "get_pod_for_fingerprint", lambda *args, **kwargs: None)
    pod_api.delete_pod_for_fingerprint(
        "pod123456", api.key_fingerprint("owner-key"), deadline_at=time.time() + 60
    )
    assert calls == [("owner-key", f"{api.REST_BASE}/pods/pod123456", "DELETE")]


def test_exact_delete_waits_for_delayed_absence(monkeypatch):
    observations = iter([_pod(), _pod(), None])
    monkeypatch.setattr(
        "flash.providers.runpod.client.pods._key_for_fingerprint", lambda fingerprint: "owner-key"
    )
    monkeypatch.setattr(
        "flash.providers.runpod.client.pods._mutation_once", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        pod_api,
        "get_pod_for_fingerprint",
        lambda *args, **kwargs: next(observations),
    )
    monkeypatch.setattr(pod_api.time, "sleep", lambda seconds: None)
    pod_api.delete_pod_for_fingerprint(
        "pod123456",
        api.key_fingerprint("owner-key"),
        deadline_at=time.time() + 60,
    )


def test_pending_cancel_deletes_all_matching_pods_before_secret(monkeypatch):
    handle = _handle()
    events = []
    observations = iter([[_pod(pod_id="pod-a"), _pod(pod_id="pod-b")], []])
    monkeypatch.setattr(
        "flash.providers.runpod.client.pods._key_for_fingerprint", lambda fingerprint: "owner-key"
    )
    monkeypatch.setattr(pod_api, "list_pods_for_key", lambda *args, **kwargs: next(observations))
    monkeypatch.setattr(
        pod_api,
        "delete_pod_for_fingerprint",
        lambda pod_id, *args, **kwargs: events.append(("pod", pod_id)),
    )
    monkeypatch.setattr(
        pod_api,
        "delete_secret_for_fingerprint",
        lambda *args, **kwargs: events.append(("secret", args[1])),
    )
    pods.terminate_handle(handle, deadline_at=time.time() + 60)
    assert events == [("pod", "pod-a"), ("pod", "pod-b"), ("secret", "secret123")]


def test_exact_cancel_proves_pod_absence_before_secret_cleanup(monkeypatch):
    handle = _handle(exact=True)
    events = []
    monkeypatch.setattr(
        pod_api,
        "delete_pod_for_fingerprint",
        lambda *args, **kwargs: events.append("pod-absent"),
    )
    monkeypatch.setattr(
        pod_api,
        "delete_secret_for_fingerprint",
        lambda *args, **kwargs: events.append("secret-absent"),
    )
    pods.terminate_handle(handle, deadline_at=time.time() + 60)
    assert events == ["pod-absent", "secret-absent"]


def test_run_listing_failure_fails_closed(monkeypatch):
    monkeypatch.setattr(
        pod_api, "list_pods_by_key", lambda: ({api.key_fingerprint("a"): []}, ["bad"])
    )
    with pytest.raises(api.RunpodApiError, match="incomplete"):
        pods.run_pods_remaining("runpod-pod-test")


def test_run_cleanup_recovers_nonce_backed_secret_identity(monkeypatch):
    fingerprint = api.key_fingerprint("owner-key")
    observed = _pod()
    expected_secret = pod_identity.payload_secret_name_from_pod_label(observed.name)
    events = []
    monkeypatch.setattr(pod_api, "list_pods_by_key", lambda: ({fingerprint: [observed]}, []))
    monkeypatch.setattr(
        pod_api,
        "delete_pod_for_fingerprint",
        lambda pod_id, *args, **kwargs: events.append(("pod", pod_id)),
    )

    def list_secret(_fingerprint, *, name, deadline_at):
        assert name == expected_secret
        return "account123", [pod_api.RunpodSecret("secret123", name)]

    monkeypatch.setattr(pod_api, "list_secrets_for_fingerprint", list_secret)
    monkeypatch.setattr(
        pod_api,
        "delete_secret_for_fingerprint",
        lambda _fingerprint, secret_id, *args, **kwargs: events.append(("secret", secret_id)),
    )
    assert pods.destroy_run_pods("runpod-pod-test") == [observed.id]
    assert events == [("pod", observed.id), ("secret", "secret123")]


def test_orphan_cleanup_recovers_nonce_backed_secret_identity(monkeypatch):
    fingerprint = api.key_fingerprint("owner-key")
    observed = _pod()
    secret_name = pod_identity.payload_secret_name_from_pod_label(observed.name)
    deleted = []
    monkeypatch.setattr(pod_api, "list_pods_by_key", lambda: ({fingerprint: [observed]}, []))
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: (
            "account123",
            [pod_api.RunpodSecret("secret123", secret_name)],
        ),
    )
    monkeypatch.setattr(
        pod_api,
        "delete_pod_for_fingerprint",
        lambda pod_id, *args, **kwargs: deleted.append(("pod", pod_id)),
    )
    monkeypatch.setattr(
        pod_api,
        "delete_secret_for_fingerprint",
        lambda _fingerprint, secret_id, *args, **kwargs: deleted.append(("secret", secret_id)),
    )
    assert pods.sweep_orphan_pods(active_labels=set(), known_labels={"runpod-pod-test"}) == [
        observed.id
    ]
    assert deleted == [("pod", observed.id), ("secret", "secret123")]


def test_expired_preload_orphan_is_reaped_outside_known_training_runs(monkeypatch):
    fingerprint = api.key_fingerprint("owner-key")
    run_id = "flash-preload-d1000000000-runpod-us-ca-2-x"
    secret_name = "FLASH_PAYLOAD_0123456789abcdef"
    reference = pod_identity.secret_reference(secret_name)
    label = f"{run_id}-s0-a0-0123456789abcdef-deadbeef"
    observed = pod_api.RunpodPod(
        id="pod-preload",
        name=label,
        desired_status="RUNNING",
        image_name="image:tag",
        gpu_type_id="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        data_center_id="US-CA-2",
        container_disk_gb=60,
        network_volume_id="volume123",
        cost_per_hr=1.0,
        payload_env_sha256=hashlib.sha256(reference.encode()).hexdigest(),
        payload_secret_name=secret_name,
    )
    deleted = []
    monkeypatch.setattr(pod_api, "list_pods_by_key", lambda: ({fingerprint: [observed]}, []))
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: ("account123", [pod_api.RunpodSecret("secret123", secret_name)]),
    )
    monkeypatch.setattr(
        pod_api,
        "delete_pod_for_fingerprint",
        lambda pod_id, *args, **kwargs: deleted.append(("pod", pod_id)),
    )
    monkeypatch.setattr(
        pod_api,
        "delete_secret_for_fingerprint",
        lambda _fingerprint, secret_id, *args, **kwargs: deleted.append(("secret", secret_id)),
    )
    assert pods.sweep_orphan_pods(active_labels=set(), known_labels={"other-run"}) == [
        "pod-preload"
    ]
    assert deleted == [("pod", "pod-preload"), ("secret", "secret123")]


def test_active_expired_preload_orphan_remains_protected(monkeypatch):
    fingerprint = api.key_fingerprint("owner-key")
    run_id = "flash-preload-d1000000000-runpod-us-ca-2-x"
    secret_name = "FLASH_PAYLOAD_0123456789abcdef"
    label = f"{run_id}-s0-a0-0123456789abcdef-deadbeef"
    observed = _pod(name=label)
    monkeypatch.setattr(pod_api, "list_pods_by_key", lambda: ({fingerprint: [observed]}, []))
    monkeypatch.setattr(
        pod_api, "list_secrets_for_fingerprint", lambda *args, **kwargs: ("account123", [])
    )
    monkeypatch.setattr(
        pod_api,
        "delete_pod_for_fingerprint",
        lambda *args, **kwargs: pytest.fail("active preload Pod must not be deleted"),
    )
    assert pods.sweep_orphan_pods(active_labels={run_id}, known_labels=set()) == []


def test_poll_uses_pod_state_and_shared_instance_kernel(monkeypatch):
    handle = _handle(exact=True)
    captured = {}
    monkeypatch.setattr(pods, "_make_hf_file_reader", lambda *args, **kwargs: lambda **kw: None)
    monkeypatch.setattr(
        pod_api,
        "get_pod_for_fingerprint",
        lambda *args, **kwargs: _pod(pod_id=handle.pod_id, status="RUNNING"),
    )

    def fake_poll(adapter, **kwargs):
        captured["adapter"] = adapter
        captured["kwargs"] = kwargs
        return PollResult(False, failure="stalled", detail="test")

    monkeypatch.setattr(pods, "poll_instance_job", fake_poll)
    result = pods.poll_runpod_pod(
        handle,
        _spec(count=2),
        0,
        heartbeat_reader=lambda **kwargs: None,
        deadline_at=time.time() + 60,
    )
    assert result.failure == "stalled"
    assert captured["adapter"].fetch_instance() == {"desired_status": "RUNNING"}
    assert captured["adapter"].running_status == "RUNNING"
    assert captured["adapter"].early_liveness_alive() is False


def test_launcher_writes_mode_0600_unsets_payload_and_executes_capsule(tmp_path, monkeypatch):
    path = Path(__file__).parents[1] / "docker" / "runpod_pod_launcher.py"
    spec = importlib.util.spec_from_file_location("runpod_pod_launcher_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload_path = tmp_path / "payload.json"
    module.PAYLOAD_PATH = payload_path
    module.CAPSULE_PATH = tmp_path / "capsule.pyz"
    monkeypatch.setenv(module.PAYLOAD_ENV, '{"run_id":"safe"}')
    calls = []
    monkeypatch.setattr(module.subprocess, "call", lambda argv: calls.append(argv) or 0)
    assert module.main() == 0
    assert stat.S_IMODE(payload_path.stat().st_mode) == 0o600
    assert json.loads(payload_path.read_text()) == {"run_id": "safe"}
    assert module.PAYLOAD_ENV not in os.environ
    assert calls == [[module.sys.executable, str(module.CAPSULE_PATH), "bootstrap"]]


def test_instance_payload_preserves_runpod_weight_cache_redirect():
    digest = "a" * 64
    payload = pods._build_instance_payload(
        _spec(volume="flash-weights"),
        42,
        1,
        None,
        {
            "kind": "flash-source-snapshot",
            "format_version": 1,
            "archive_path": f"source/{digest}/flash-source.zip",
            "sha256": digest,
            "size": 1,
            "revision": "b" * 40,
        },
        time.time() + 3600,
    )
    assert payload["env"]["FLASH_WEIGHT_CACHE_DIR"] == "/runpod-volume/hf-cache/hub"
    assert payload["flash_arm"] == "runpod"


def test_secret_delete_404_continues_through_same_name_absence(monkeypatch):
    fingerprint = api.key_fingerprint("owner-key")
    observations = []
    monkeypatch.setattr(api, "_key_for_fingerprint", lambda value: "owner-key")
    monkeypatch.setattr(
        pod_api,
        "_mutation_once",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            pod_api.RunpodRequestError("already deleted", status_code=404)
        ),
    )

    def list_absent(*args, **kwargs):
        observations.append(kwargs["name"])
        return "account123", []

    monkeypatch.setattr(pod_api, "list_secrets_for_fingerprint", list_absent)
    pod_api.delete_secret_for_fingerprint(
        fingerprint,
        "secret123",
        "FLASH_PAYLOAD_0123456789abcdef",
        deadline_at=time.time() + 60,
    )
    assert observations == ["FLASH_PAYLOAD_0123456789abcdef"]


def test_secret_delete_propagates_non_404_request_error(monkeypatch):
    fingerprint = api.key_fingerprint("owner-key")
    monkeypatch.setattr(api, "_key_for_fingerprint", lambda value: "owner-key")
    monkeypatch.setattr(
        pod_api,
        "_mutation_once",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            pod_api.RunpodRequestError("unauthorized", status_code=401)
        ),
    )
    with pytest.raises(pod_api.RunpodRequestError) as exc_info:
        pod_api.delete_secret_for_fingerprint(
            fingerprint,
            "secret123",
            "FLASH_PAYLOAD_0123456789abcdef",
            deadline_at=time.time() + 60,
        )
    assert exc_info.value.status_code == 401


def test_secret_delete_requires_same_name_absence(monkeypatch):
    fingerprint = api.key_fingerprint("owner-key")
    monkeypatch.setattr(api, "_key_for_fingerprint", lambda value: "owner-key")
    monkeypatch.setattr(
        pod_api,
        "_mutation_once",
        lambda *args, **kwargs: {"data": {"secretDelete": None}},
    )
    monkeypatch.setattr(pod_api.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: (
            "account123",
            [pod_api.RunpodSecret("secret-other", "FLASH_PAYLOAD_same")],
        ),
    )
    with pytest.raises(api.RunpodApiError, match="unconfirmed"):
        pod_api.delete_secret_for_fingerprint(
            fingerprint,
            "secret123",
            "FLASH_PAYLOAD_same",
            deadline_at=time.time() + 60,
        )


def test_recovery_reads_runpod_completion_from_shared_artifacts(monkeypatch):
    handle = _handle(exact=True)
    captured = {}

    def completed(spec, **kwargs):
        captured.update(kwargs)
        return {"loss": 1.0}

    monkeypatch.setattr(recovery, "_completed_attempt_metrics", completed)
    metrics = recovery._runpod_completed_metrics(
        handle,
        spec=_spec(count=2),
        deadline_at=time.time() + 60,
    )
    assert metrics == {"loss": 1.0}
    assert captured["provider"] == "runpod"
    assert captured["attempt"] == handle.attempt
    assert captured["launch_floor"] == handle.started_ts


def test_recovery_preserves_cleanup_target_when_secret_teardown_is_unconfirmed(monkeypatch):
    handle = _handle(exact=True)

    class Provider:
        def destroy(self, _selected):
            raise api.RunpodApiError("secret cleanup unconfirmed")

    monkeypatch.setattr("flash.providers.core.registry.get_provider", lambda name: Provider())
    monkeypatch.setattr(recovery, "_worker_provably_gone", lambda *args: True)
    with pytest.raises(RuntimeError, match="payload secret"):
        recovery._strict_teardown_handle(handle, "runpod-pod-test")


def test_submission_persists_every_create_phase_before_mutation(monkeypatch):
    key = "owner-key"
    fingerprint = api.key_fingerprint(key)
    observed_handles = []
    monkeypatch.setattr(pods.runpod_auth, "ordered_keys", lambda: [key])
    monkeypatch.setattr(pods, "_build_instance_payload", lambda *args, **kwargs: {"safe": True})
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: ("account123", []),
    )

    def create_secret(_fingerprint, name, _value, **kwargs):
        assert observed_handles[-1]["phase"] == pods.SECRET_CREATE_PENDING
        return pod_api.RunpodSecret("secret123", name)

    monkeypatch.setattr(pod_api, "create_secret_for_fingerprint", create_secret)
    monkeypatch.setattr(pods, "_volume_candidates", lambda *args, **kwargs: [(None, None)])

    def create(pending, payload, **kwargs):
        assert observed_handles[-1]["phase"] == pods.POD_CREATE_PENDING
        return pods._exact_handle(
            pending,
            _pod(
                pod_id="pod123456",
                name=pending.label,
                count=1,
                data_center=None,
                volume_id=None,
                rate=3.29,
            ),
        )

    monkeypatch.setattr(pods, "create_or_adopt_pod", create)
    monkeypatch.setattr(pods, "heartbeat_reader_for", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pods,
        "poll_runpod_pod",
        lambda *args, **kwargs: PollResult(True, metrics={"ok": True}),
    )
    monkeypatch.setattr(pods, "terminate_handle", lambda *args, **kwargs: None)
    result = pods.submit_runpod_pod(
        _spec(count=1),
        0,
        attempt=1,
        on_handle=observed_handles.append,
        deadline_at=time.time() + 3600,
    )
    assert result.ok
    assert [item["phase"] for item in observed_handles] == [
        pods.SECRET_CREATE_PENDING,
        pods.PRE_POD_CREATE,
        pods.PRE_POD_CREATE,
        pods.POD_CREATE_PENDING,
        pods.EXACT,
    ]
    assert observed_handles[0]["payload_secret_id"] is None
    assert observed_handles[1]["payload_secret_id"] == "secret123"
    assert observed_handles[-1]["instance_id"] == "pod123456"
    assert all(item["key_fingerprint"] == fingerprint for item in observed_handles)


def test_reattached_poll_preserves_success_and_records_unconfirmed_cleanup(monkeypatch):
    from flash.providers.runpod.execution.provider import PROVIDER

    handle = _handle(exact=True)
    monkeypatch.setattr(
        "flash.providers.artifacts.hf.heartbeat_reader_for",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        PROVIDER,
        "_poll_job",
        lambda *args, **kwargs: PollResult(True, metrics={"ok": True}),
    )
    monkeypatch.setattr(
        PROVIDER,
        "_teardown_reattached",
        lambda *args, **kwargs: (_ for _ in ()).throw(api.RunpodApiError("unconfirmed")),
    )
    recorded = []
    monkeypatch.setattr(
        "flash.runner.accounting.reconciliation._record_cleanup_remote",
        lambda run_id, remote: recorded.append((run_id, remote)) or True,
    )
    result = PROVIDER.poll(
        JobHandle.from_dict(handle.to_dict()),
        _spec(count=2),
        42,
        _deadline_at=time.time() + 60,
    )
    assert result.ok
    assert recorded == [("runpod-pod-test", handle.to_dict())]


def test_instance_provider_registry_and_non_runpod_commands_remain_unchanged():
    assert INSTANCE_PROVIDERS == ("runpod", "lambda", "vast")
    from flash.providers.lambda_.jobs.builders import build_user_data as lambda_user_data
    from flash.providers.vast.jobs.builders import build_onstart as vast_onstart

    assert callable(lambda_user_data)
    assert callable(vast_onstart)


def test_unconstrained_running_pod_accepts_actual_placement():
    spec = _spec(count=2)
    secret_name = "FLASH_PAYLOAD_0123456789abcdef"
    label = pod_identity.pod_label(
        spec,
        0,
        1,
        secret_name=secret_name,
        data_center_id=None,
        network_volume_id=None,
        container_registry_auth_id=None,
    )
    payload = pod_identity.build_pod_payload(
        spec,
        label=label,
        secret_name=secret_name,
        data_center_id=None,
        network_volume_id=None,
    )
    observed = _pod(
        name=label,
        status="RUNNING",
        data_center="US-TX-3",
        volume_id=None,
    )
    assert pods._pod_matches(observed, payload, network_volume_id=None, data_center_id=None)


def test_custom_pending_pod_recovers_from_its_exact_persisted_request(monkeypatch):
    handle = replace(
        _handle(
            phase=pods.POD_CREATE_PENDING,
            count=1,
            data_center=None,
            volume_id=None,
        ),
        image_name="image@sha256:abc",
        gpu_type_id_override="NVIDIA A100 80GB PCIe",
        allowed_cuda_versions=("12.8",),
    )
    payload = pod_identity.payload_for_handle(handle)
    label = pod_identity.pod_label_from_payload(
        pod_identity.pod_attempt_label_base("runpod-pod-test", 0, handle.attempt),
        handle.payload_secret_name,
        payload,
    )
    handle = replace(handle, instance_id=label, label=label)
    observed = _pod(
        name=label,
        status="PENDING",
        gpu=None,
        count=1,
        data_center=None,
        volume_id=None,
        image="image@sha256:abc",
    )
    monkeypatch.setattr(pod_api, "list_pods_for_key", lambda *args, **kwargs: [observed])
    exact = pods.resolve_pending_handle(handle, _spec(count=1), 0, deadline_at=time.time() + 60)
    assert exact.pod_id == observed.id
    assert exact.image_name == "image@sha256:abc"
    assert exact.gpu_type_id_override == "NVIDIA A100 80GB PCIe"


def test_secret_create_pending_recovers_exactly_one_identity(monkeypatch):
    handle = _handle(phase=pods.SECRET_CREATE_PENDING)
    secret = pod_api.RunpodSecret("secret123", handle.payload_secret_name)
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: (handle.account_id, [secret]),
    )
    resolved = pods.resolve_pending_handle(handle, _spec(count=2), 0, deadline_at=time.time() + 60)
    assert resolved.phase == pods.PRE_POD_CREATE
    assert resolved.payload_secret_id == secret.id


def test_secret_create_pending_zero_is_authoritative_absence(monkeypatch):
    handle = _handle(phase=pods.SECRET_CREATE_PENDING)
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: (handle.account_id, []),
    )
    monkeypatch.setattr(pods.time, "sleep", lambda seconds: None)
    with pytest.raises(pods.RunpodCreateAbsent):
        pods.resolve_pending_handle(handle, _spec(count=2), 0, deadline_at=time.time() + 60)


def test_secret_create_pending_duplicates_remain_unresolved(monkeypatch):
    handle = _handle(phase=pods.SECRET_CREATE_PENDING)
    secrets = [
        pod_api.RunpodSecret("secret-a", handle.payload_secret_name),
        pod_api.RunpodSecret("secret-b", handle.payload_secret_name),
    ]
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: (handle.account_id, secrets),
    )
    with pytest.raises(pods.UnreconciledCreateError, match="duplicate"):
        pods.resolve_pending_handle(handle, _spec(count=2), 0, deadline_at=time.time() + 60)


def test_pre_pod_create_is_authoritative_absence():
    handle = _handle(phase=pods.PRE_POD_CREATE)
    with pytest.raises(pods.RunpodCreateAbsent):
        pods.resolve_pending_handle(handle, _spec(count=2), 0, deadline_at=time.time() + 60)


def test_pending_recovery_adopts_one_complete_match_without_create(monkeypatch):
    handle = _handle()
    exact = _pod(name=handle.label)
    monkeypatch.setattr(
        "flash.providers.runpod.client.pods._key_for_fingerprint", lambda fingerprint: "owner-key"
    )
    monkeypatch.setattr(pod_api, "list_pods_for_key", lambda *args, **kwargs: [exact])
    resolved = pods.resolve_pending_handle(handle, _spec(count=2), 0, deadline_at=time.time() + 60)
    assert resolved is not None
    assert resolved.pod_id == exact.id


def test_pending_recovery_preserves_incomplete_identity(monkeypatch):
    handle = _handle()
    incomplete = _pod(name=handle.label, complete=False)
    deleted = []
    monkeypatch.setattr(
        "flash.providers.runpod.client.pods._key_for_fingerprint", lambda fingerprint: "owner-key"
    )
    monkeypatch.setattr(pod_api, "list_pods_for_key", lambda *args, **kwargs: [incomplete])
    monkeypatch.setattr(
        pod_api,
        "delete_pod_for_fingerprint",
        lambda pod_id, *args, **kwargs: deleted.append(pod_id),
    )
    with pytest.raises(pods.UnreconciledCreateError, match="incomplete"):
        pods.resolve_pending_handle(handle, _spec(count=2), 0, deadline_at=time.time() + 60)
    assert deleted == []


def test_pending_recovery_persists_exact_handle_before_poll(monkeypatch, tmp_path):
    from flash.runner.lifecycle import state as runner_state
    from flash.runner.lifecycle import status as runner_status
    from flash.runner.supervise import attach

    spec = _spec(count=2)
    pending = _handle()
    exact = _handle(exact=True)
    remote = {
        **pending.to_dict(),
        "seed": 0,
        "allocated_gpu": "H100",
        "allocated_gpu_count": 2,
    }
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    runner_state._save_status(
        runner_state.RunStatus(spec.run_id, "running", spec.to_dict(), remote=remote)
    )
    monkeypatch.setattr(pods, "resolve_pending_handle", lambda *args, **kwargs: exact)
    context = attach._AttachContext(
        spec, remote, JobHandle.from_dict(pending.to_dict()), 0, 1, 2, None
    )
    resolved = attach._resolve_pending_runpod_context(
        spec.run_id, context, deadline_at=time.time() + 60
    )
    persisted = runner_status.get_status(spec.run_id).remote
    assert resolved.handle.data["instance_id"] == exact.instance_id
    assert persisted["instance_id"] == exact.instance_id
    assert persisted["seed"] == 0
    assert persisted["allocated_gpu_count"] == 2


def test_direct_attach_resumes_after_authoritative_pending_absence(monkeypatch, tmp_path):
    from flash.runner.lifecycle import state as runner_state
    from flash.runner.supervise import attach

    spec = _spec(count=2)
    pending = _handle(phase=pods.PRE_POD_CREATE)
    remote = {**pending.to_dict(), "seed": 0}
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    runner_state._save_status(
        runner_state.RunStatus(spec.run_id, "running", spec.to_dict(), remote=remote)
    )
    context = attach._AttachContext(
        spec, remote, JobHandle.from_dict(pending.to_dict()), 0, 1, 2, None
    )
    monkeypatch.setattr(attach, "_build_attach_context", lambda *args: context)
    monkeypatch.setattr(
        "flash.runner.lifecycle.deadlines._spec_with_remaining_wall", lambda *args, **kwargs: spec
    )
    monkeypatch.setattr(
        attach,
        "_resolve_pending_runpod_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(attach._PendingRunpodCreateAbsent(context)),
    )
    resumed = []
    monkeypatch.setattr(
        attach,
        "_resume_reconciled_after_teardown",
        lambda *args, **kwargs: resumed.append(args),
    )
    status = attach.attach_run(spec.run_id)
    assert status.state == "running"
    assert resumed[0][2] == remote
    assert resumed[0][3] == 2


def test_background_reconciler_resumes_after_authoritative_pending_absence(monkeypatch, tmp_path):
    from flash.runner.lifecycle import state as runner_state
    from flash.runner.supervise import attach

    spec = _spec(count=2)
    pending = _handle(phase=pods.PRE_POD_CREATE)
    remote = {**pending.to_dict(), "seed": 0}
    monkeypatch.setattr(runner_state, "RUNS_DIR", str(tmp_path / "runs"))
    runner_state._save_status(
        runner_state.RunStatus(spec.run_id, "running", spec.to_dict(), remote=remote)
    )
    context = attach._AttachContext(
        spec, remote, JobHandle.from_dict(pending.to_dict()), 0, 1, 2, None
    )
    monkeypatch.setattr(attach, "_build_attach_context", lambda *args: context)
    monkeypatch.setattr(
        attach,
        "_resolve_pending_runpod_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(attach._PendingRunpodCreateAbsent(context)),
    )
    resumed = []
    monkeypatch.setattr(
        attach,
        "_resume_reconciled_after_teardown",
        lambda *args, **kwargs: resumed.append(args),
    )
    attach._reconcile_attached_remote(
        spec.run_id,
        remote,
        spec,
        2,
        None,
        None,
        "pending create unresolved",
    )
    assert resumed[0][2] == remote
    assert resumed[0][3] == 2


@pytest.mark.parametrize(
    "phase",
    [pods.SECRET_CREATE_PENDING, pods.PRE_POD_CREATE, pods.POD_CREATE_PENDING],
)
def test_provisional_handle_keeps_submission_fence_until_exact(monkeypatch, phase):
    from flash.runner.supervise import seed_submission

    class Lock:
        def __init__(self):
            self.releases = 0

        def release(self):
            self.releases += 1

    lock = Lock()
    context = seed_submission._SubmitContext(
        spec=_spec(count=2),
        seed=0,
        log=None,
        runtime_secrets=None,
        source_snapshot={},
        attempt_start=1,
        infra_budget=0,
        retry_budget=None,
        started_with_shared_cache=False,
        current_gpu={"provider": "runpod", "name": "H100", "count": 2},
        current_attempt=1,
        submission_lock=lock,
    )
    monkeypatch.setattr("flash.runner.lifecycle.status._update", lambda *args, **kwargs: True)
    context.on_handle(_handle(phase=phase).to_dict())
    assert lock.releases == 0
    assert context.submission_lock is lock
    context.on_handle(_handle(exact=True).to_dict())
    assert lock.releases == 1
    assert context.submission_lock is None


def test_account_create_request_failure_does_not_try_next_account(monkeypatch):
    keys = ["bad-key", "good-key"]
    attempted = []
    monkeypatch.setattr(pods.runpod_auth, "ordered_keys", lambda: keys)
    monkeypatch.setattr(pods, "_build_instance_payload", lambda *args, **kwargs: {"safe": True})
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: ("account123", []),
    )

    def create_secret(fingerprint, _name, _value, **kwargs):
        attempted.append(fingerprint)
        raise pod_api.RunpodRequestError("unauthorized", status_code=401)

    monkeypatch.setattr(pod_api, "create_secret_for_fingerprint", create_secret)
    monkeypatch.setattr(pods, "_volume_candidates", lambda *args, **kwargs: [(None, None)])

    with pytest.raises(pod_api.RunpodRequestError, match="unauthorized"):
        pods.submit_runpod_pod(
            _spec(count=1),
            0,
            attempt=1,
            deadline_at=time.time() + 3600,
        )

    assert attempted == [api.key_fingerprint(keys[0])]


def test_volume_candidates_reuse_and_grow_in_owning_account(monkeypatch):
    fingerprint = api.key_fingerprint("owner-key")
    existing = pod_api.RunpodNetworkVolume("vol1", "flash-weights-us-ks-2", 100, "US-KS-2")
    grown = pod_api.RunpodNetworkVolume("vol1", existing.name, 250, existing.data_center_id)
    listings = iter([[existing], [grown]])
    growth = []
    monkeypatch.setattr(
        "flash.providers.runpod.execution.resources.weight_cache_datacenters",
        lambda selected, **kwargs: ["US-KS-2"],
    )
    monkeypatch.setattr(
        pod_api, "list_network_volumes_for_fingerprint", lambda *args, **kwargs: next(listings)
    )
    monkeypatch.setattr(
        pod_api,
        "grow_network_volumes_for_fingerprint",
        lambda selected, wanted, **kwargs: growth.append((selected, wanted)) or wanted,
    )
    assert pods._volume_candidates(
        _spec(count=1, volume="flash-weights"),
        fingerprint,
        deadline_at=time.time() + 60,
    ) == [("US-KS-2", "vol1")]
    assert growth == [(fingerprint, {existing.name: 250})]


def test_volume_candidates_reject_duplicate_account_records(monkeypatch):
    fingerprint = api.key_fingerprint("owner-key")
    duplicate = pod_api.RunpodNetworkVolume("vol1", "flash-weights-us-ks-2", 250, "US-KS-2")
    monkeypatch.setattr(
        "flash.providers.runpod.execution.resources.weight_cache_datacenters",
        lambda selected, **kwargs: ["US-KS-2"],
    )
    monkeypatch.setattr(
        pod_api,
        "list_network_volumes_for_fingerprint",
        lambda *args, **kwargs: [
            duplicate,
            pod_api.RunpodNetworkVolume("vol2", duplicate.name, 250, "US-KS-2"),
        ],
    )
    with pytest.raises(api.RunpodApiError, match="duplicated"):
        pods._volume_candidates(
            _spec(count=1, volume="flash-weights"),
            fingerprint,
            deadline_at=time.time() + 60,
        )


def test_ambiguous_volume_create_adopts_only_one_complete_account_match(monkeypatch):
    fingerprint = api.key_fingerprint("owner-key")
    created = pod_api.RunpodNetworkVolume("vol1", "flash-weights-us-ks-2", 250, "US-KS-2")
    listings = iter([[], [created]])
    monkeypatch.setattr(
        "flash.providers.runpod.execution.resources.weight_cache_datacenters",
        lambda selected, **kwargs: ["US-KS-2"],
    )
    monkeypatch.setattr(
        pod_api, "list_network_volumes_for_fingerprint", lambda *args, **kwargs: next(listings)
    )
    monkeypatch.setattr(
        pod_api,
        "create_network_volume_for_fingerprint",
        lambda *args, **kwargs: (_ for _ in ()).throw(pod_api.RunpodMutationAmbiguous("unknown")),
    )
    assert pods._volume_candidates(
        _spec(count=1, volume="flash-weights"),
        fingerprint,
        deadline_at=time.time() + 60,
    ) == [("US-KS-2", "vol1")]


def test_handleless_runpod_completion_fails_closed_on_account_listing(monkeypatch):
    from types import SimpleNamespace

    from flash.server.platform import runtime

    status = SimpleNamespace(submitted_instance_providers=["runpod"], created_at=1.0)
    monkeypatch.setattr(
        "flash.runner.lifecycle.attempts._latest_reserved_attempt", lambda run_id: 1
    )

    class Provider:
        def run_instances_remaining(self, run_id):
            raise api.RunpodApiError("incomplete account listing")

    monkeypatch.setattr("flash.providers.core.registry.get_provider", lambda name: Provider())
    with pytest.raises(api.RunpodApiError, match="incomplete"):
        runtime._handleless_completed_metrics(_spec(count=1), status, deadline_at=time.time() + 60)


def test_terminal_race_on_secret_intent_never_reaches_any_create(monkeypatch):
    from flash.runner.supervise.errors import _TerminalHandleRace

    creates = []
    monkeypatch.setattr(pods.runpod_auth, "ordered_keys", lambda: ["owner-key"])
    monkeypatch.setattr(pods, "_build_instance_payload", lambda *args, **kwargs: {"safe": True})
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: ("account123", []),
    )
    monkeypatch.setattr(
        pod_api,
        "create_secret_for_fingerprint",
        lambda *args, **kwargs: creates.append("secret"),
    )
    monkeypatch.setattr(
        pods,
        "create_or_adopt_pod",
        lambda *args, **kwargs: creates.append("pod"),
    )

    def terminal(handle):
        assert handle["phase"] == pods.SECRET_CREATE_PENDING
        raise _TerminalHandleRace("cancelled")

    with pytest.raises(_TerminalHandleRace):
        pods.submit_runpod_pod(
            _spec(count=1),
            0,
            attempt=1,
            on_handle=terminal,
            deadline_at=time.time() + 3600,
        )
    assert creates == []


def test_terminal_callback_cleanup_failure_preserves_terminal_exception(monkeypatch):
    from flash.runner.supervise.errors import _TerminalHandleRace

    keys = ["owner-key", "unused-key"]
    handles = []
    secret_creates = []
    creates = []
    monkeypatch.setattr(pods.runpod_auth, "ordered_keys", lambda: keys)
    monkeypatch.setattr(pods, "_build_instance_payload", lambda *args, **kwargs: {"safe": True})
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: ("account123", []),
    )
    monkeypatch.setattr(
        pod_api,
        "create_secret_for_fingerprint",
        lambda fingerprint, name, _value, **kwargs: (
            secret_creates.append(fingerprint) or pod_api.RunpodSecret("secret123", name)
        ),
    )
    monkeypatch.setattr(pods, "_volume_candidates", lambda *args, **kwargs: [(None, None)])

    def persist_then_cancel(handle):
        handles.append(handle)
        if handle["phase"] == pods.PRE_POD_CREATE:
            raise _TerminalHandleRace("cancelled")

    def cleanup_fails(*args, **kwargs):
        raise api.RunpodApiError("cleanup unconfirmed")

    monkeypatch.setattr(pods, "terminate_handle", cleanup_fails)
    monkeypatch.setattr(
        pods,
        "create_or_adopt_pod",
        lambda *args, **kwargs: creates.append(True),
    )

    with pytest.raises(_TerminalHandleRace, match="cancelled") as exc_info:
        pods.submit_runpod_pod(
            _spec(count=1),
            0,
            attempt=1,
            on_handle=persist_then_cancel,
            deadline_at=time.time() + 3600,
        )

    assert secret_creates == [api.key_fingerprint(keys[0])]
    assert creates == []
    assert handles[-1]["phase"] == pods.PRE_POD_CREATE
    assert handles[-1]["payload_secret_id"] == "secret123"
    assert any("rollback teardown also failed" in note for note in exc_info.value.__notes__)


def test_non_capacity_launch_failure_does_not_try_next_candidate(monkeypatch):
    attempts = []
    monkeypatch.setattr(pods.runpod_auth, "ordered_keys", lambda: ["owner-key"])
    monkeypatch.setattr(pods, "_build_instance_payload", lambda *args, **kwargs: {"safe": True})
    monkeypatch.setattr(
        pods,
        "_volume_candidates",
        lambda *args, **kwargs: [("DC-1", "vol-1"), ("DC-2", "vol-2")],
    )

    def reject(*args, **kwargs):
        attempts.append(kwargs["data_center_id"])
        raise pod_api.RunpodRequestError("unauthorized", status_code=401)

    monkeypatch.setattr(pods, "launch_payload_pod", reject)

    with pytest.raises(pod_api.RunpodRequestError, match="unauthorized"):
        pods.submit_runpod_pod(
            _spec(count=1),
            0,
            attempt=1,
            deadline_at=time.time() + 3600,
        )

    assert attempts == ["DC-1"]


def test_capacity_launch_failure_advances_to_next_candidate(monkeypatch):
    attempts = []
    handle = _handle(exact=True, count=1, data_center="DC-2", volume_id="vol-2")
    monkeypatch.setattr(pods.runpod_auth, "ordered_keys", lambda: ["owner-key"])
    monkeypatch.setattr(pods, "_build_instance_payload", lambda *args, **kwargs: {"safe": True})
    monkeypatch.setattr(
        pods,
        "_volume_candidates",
        lambda *args, **kwargs: [("DC-1", "vol-1"), ("DC-2", "vol-2")],
    )

    def launch(*args, **kwargs):
        attempts.append(kwargs["data_center_id"])
        if kwargs["data_center_id"] == "DC-1":
            raise pod_api.RunpodCapacityError("full")
        return handle

    monkeypatch.setattr(pods, "launch_payload_pod", launch)
    monkeypatch.setattr(pods, "heartbeat_reader_for", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pods,
        "poll_runpod_pod",
        lambda *args, **kwargs: PollResult(True, metrics={"ok": True}),
    )
    monkeypatch.setattr(pods, "terminate_handle", lambda *args, **kwargs: None)

    result = pods.submit_runpod_pod(
        _spec(count=1),
        0,
        attempt=1,
        deadline_at=time.time() + 3600,
    )

    assert result.ok
    assert attempts == ["DC-1", "DC-2"]


def test_ambiguous_secret_create_preserves_secret_pending_handle(monkeypatch):
    handles = []
    monkeypatch.setattr(pods.runpod_auth, "ordered_keys", lambda: ["owner-key"])
    monkeypatch.setattr(pods, "_build_instance_payload", lambda *args, **kwargs: {"safe": True})
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: ("account123", []),
    )
    monkeypatch.setattr(
        pod_api,
        "create_secret_for_fingerprint",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            pod_api.RunpodMutationAmbiguous("secret outcome unknown")
        ),
    )
    with pytest.raises(pods.UnreconciledCreateError):
        pods.submit_runpod_pod(
            _spec(count=1),
            0,
            attempt=1,
            on_handle=handles.append,
            deadline_at=time.time() + 3600,
        )
    assert [handle["phase"] for handle in handles] == [pods.SECRET_CREATE_PENDING]
    assert handles[0]["payload_secret_id"] is None


def test_cancellation_after_secret_persistence_cleans_before_pod_create(monkeypatch):
    from flash.runner.supervise.errors import _TerminalHandleRace

    creates = []
    cleaned = []
    monkeypatch.setattr(pods.runpod_auth, "ordered_keys", lambda: ["owner-key"])
    monkeypatch.setattr(pods, "_build_instance_payload", lambda *args, **kwargs: {"safe": True})
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: ("account123", []),
    )
    monkeypatch.setattr(
        pod_api,
        "create_secret_for_fingerprint",
        lambda _fingerprint, name, _value, **kwargs: pod_api.RunpodSecret("secret123", name),
    )
    monkeypatch.setattr(
        pods,
        "create_or_adopt_pod",
        lambda *args, **kwargs: creates.append(True),
    )
    monkeypatch.setattr(
        pods,
        "terminate_handle",
        lambda handle, **kwargs: cleaned.append(handle),
    )

    def cancel_after_exact_secret(handle):
        if handle["phase"] == pods.PRE_POD_CREATE:
            raise _TerminalHandleRace("cancelled")

    with pytest.raises(_TerminalHandleRace):
        pods.submit_runpod_pod(
            _spec(count=1),
            0,
            attempt=1,
            on_handle=cancel_after_exact_secret,
            deadline_at=time.time() + 3600,
        )
    assert creates == []
    assert len(cleaned) == 1
    assert cleaned[0].payload_secret_id == "secret123"


def test_cancellation_after_pod_pending_persistence_never_posts(monkeypatch):
    from flash.runner.supervise.errors import _TerminalHandleRace

    creates = []
    cleaned = []
    monkeypatch.setattr(pods.runpod_auth, "ordered_keys", lambda: ["owner-key"])
    monkeypatch.setattr(pods, "_build_instance_payload", lambda *args, **kwargs: {"safe": True})
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: ("account123", []),
    )
    monkeypatch.setattr(
        pod_api,
        "create_secret_for_fingerprint",
        lambda _fingerprint, name, _value, **kwargs: pod_api.RunpodSecret("secret123", name),
    )
    monkeypatch.setattr(pods, "_volume_candidates", lambda *args, **kwargs: [(None, None)])
    monkeypatch.setattr(
        pods,
        "create_or_adopt_pod",
        lambda *args, **kwargs: creates.append(True),
    )
    monkeypatch.setattr(
        pods,
        "terminate_handle",
        lambda handle, **kwargs: cleaned.append(handle),
    )

    def cancel_before_post(handle):
        if handle["phase"] == pods.POD_CREATE_PENDING:
            raise _TerminalHandleRace("cancelled")

    with pytest.raises(_TerminalHandleRace):
        pods.submit_runpod_pod(
            _spec(count=1),
            0,
            attempt=1,
            on_handle=cancel_before_post,
            deadline_at=time.time() + 3600,
        )
    assert creates == []
    assert len(cleaned) == 1
    assert cleaned[0].phase == pods.POD_CREATE_PENDING


def test_ambiguous_pod_create_preserves_pending_cleanup_resources(monkeypatch):
    key = "owner-key"
    deleted = []
    handles = []
    monkeypatch.setattr(pods.runpod_auth, "ordered_keys", lambda: [key])
    monkeypatch.setattr(pods, "_build_instance_payload", lambda *args, **kwargs: {"safe": True})
    monkeypatch.setattr(
        pod_api,
        "list_secrets_for_fingerprint",
        lambda *args, **kwargs: ("account123", []),
    )
    monkeypatch.setattr(
        pod_api,
        "create_secret_for_fingerprint",
        lambda _fingerprint, name, _value, **kwargs: pod_api.RunpodSecret("secret123", name),
    )
    monkeypatch.setattr(pods, "_volume_candidates", lambda *args, **kwargs: [(None, None)])
    monkeypatch.setattr(
        pods,
        "create_or_adopt_pod",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            pods.UnreconciledCreateError("unknown create outcome")
        ),
    )
    monkeypatch.setattr(
        pod_api,
        "delete_secret_for_fingerprint",
        lambda *args, **kwargs: deleted.append(("secret", args[1])),
    )
    monkeypatch.setattr(
        pod_api,
        "delete_pod_for_fingerprint",
        lambda *args, **kwargs: deleted.append(("pod", args[0])),
    )
    with pytest.raises(pods.UnreconciledCreateError):
        pods.submit_runpod_pod(
            _spec(count=1),
            0,
            attempt=1,
            on_handle=handles.append,
            deadline_at=time.time() + 3600,
        )
    assert deleted == []
    assert handles[-1]["instance_id"] == handles[-1]["label"]
