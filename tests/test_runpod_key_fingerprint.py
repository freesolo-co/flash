"""RunPod account fingerprints and exact Pod ownership."""

from __future__ import annotations

import pytest


def _reset_pool(monkeypatch, value):
    monkeypatch.setenv("RUNPOD_API_KEY", value)
    from flash.providers.runpod import auth

    auth.reset()


def _pod_row(pod_id: str, name: str) -> dict:
    return {
        "id": pod_id,
        "name": name,
        "desiredStatus": "RUNNING",
        "imageName": "worker:latest",
        "gpuCount": 1,
        "containerDiskInGb": 120,
    }


def test_key_fingerprint_is_stable_complete_and_non_revealing():
    from flash.providers.runpod import api

    secret = "rpk-supersecret-value-123"
    fingerprint = api.key_fingerprint(secret)

    assert fingerprint == api.key_fingerprint(secret)
    assert secret not in fingerprint
    assert fingerprint.startswith("rpk-")
    assert len(fingerprint) == 68
    assert api._is_valid_key_fingerprint(fingerprint)
    assert api.key_fingerprint("a-different-key") != fingerprint


@pytest.mark.parametrize(
    "fingerprint",
    [
        "rpk-" + "a" * 12,
        "rpk-" + "A" * 64,
        "rpk-" + "g" * 64,
        "xpk-" + "a" * 64,
        "",
        None,
    ],
)
def test_incomplete_or_malformed_fingerprint_is_rejected(fingerprint):
    from flash.providers.runpod import api

    assert not api._is_valid_key_fingerprint(fingerprint)
    with pytest.raises(api.RunpodApiError, match="fingerprint is invalid"):
        api._key_for_fingerprint(fingerprint)


def test_key_lookup_rejects_unknown_fingerprint_without_leaking_credentials(monkeypatch):
    from flash.providers.runpod import api

    keys = ["secretA", "secretB"]
    monkeypatch.setattr(api._keys, "keys", lambda: keys)

    with pytest.raises(api.RunpodApiError, match="exactly one") as exc_info:
        api._key_for_fingerprint("rpk-" + "0" * 64)

    assert all(key not in str(exc_info.value) for key in keys)


def test_key_lookup_rejects_colliding_configured_fingerprints(monkeypatch):
    from flash.providers.runpod import api

    keys = ["secretA", "secretB"]
    fingerprint = "rpk-" + "a" * 64
    monkeypatch.setattr(api._keys, "keys", lambda: keys)
    monkeypatch.setattr(api, "key_fingerprint", lambda _key: fingerprint)

    with pytest.raises(api.RunpodApiError, match="exactly one") as exc_info:
        api._key_for_fingerprint(fingerprint)

    assert all(key not in str(exc_info.value) for key in keys)


def test_repeated_identical_pool_key_still_resolves_its_fingerprint(monkeypatch):
    from flash.providers.runpod import api

    _reset_pool(monkeypatch, "secretA,secretB,secretA")

    assert api._key_for_fingerprint(api.key_fingerprint("secretA")) == "secretA"
    assert api._key_for_fingerprint(api.key_fingerprint("secretB")) == "secretB"


def test_list_pods_by_key_returns_fingerprints_not_raw_keys(monkeypatch):
    from flash.providers.runpod import api

    _reset_pool(monkeypatch, "secretA,secretB")

    def request(key, _url, **_kwargs):
        if key == "secretA":
            return [_pod_row("pod-a", "flash-0123456789ab-s0-a0")]
        raise api.RunpodApiError("account listing failed")

    monkeypatch.setattr(api._CLIENT, "request_with_retries_for_key", request)

    by_fingerprint, failed = api.list_pods_by_key()

    rendered = repr(by_fingerprint) + repr(failed)
    assert "secretA" not in rendered
    assert "secretB" not in rendered
    fingerprint_a = api.key_fingerprint("secretA")
    assert [pod.id for pod in by_fingerprint[fingerprint_a]] == ["pod-a"]
    assert failed == [api.key_fingerprint("secretB")]


def test_training_fleet_ignores_serving_pod_and_keeps_managed_training_pod(monkeypatch):
    from flash.providers.runpod import api

    _reset_pool(monkeypatch, "secretA")
    serving = {
        "id": "serving-pod",
        "name": "flash-pod-customer-serving",
        "desiredStatus": "RUNNING",
        "imageName": "serving:latest",
        "gpuCount": 1,
        "containerDiskInGb": 120,
        "env": {"FLASH_SERVING_MANIFEST": "not-a-training-secret-reference"},
    }
    training = _pod_row("training-pod", "flash-0123456789ab-s0-a0")
    monkeypatch.setattr(
        api._CLIENT,
        "request_with_retries_for_key",
        lambda *_args, **_kwargs: [serving, training],
    )

    by_fingerprint, failed = api.list_pods_by_key()

    assert failed == []
    assert [pod.id for pod in by_fingerprint[api.key_fingerprint("secretA")]] == ["training-pod"]


def test_get_and_delete_pod_keep_exact_owning_key_after_active_key_rotation(monkeypatch):
    from flash.providers.runpod import api, auth, pod_api

    _reset_pool(monkeypatch, "secretA,secretB")
    owner = api.key_fingerprint("secretA")
    calls = []

    def request(key, url, **kwargs):
        calls.append((key, url, kwargs.get("method", "GET")))
        return [_pod_row("pod-1", "flash-0123456789ab-s0-a0")]

    monkeypatch.setattr(api._CLIENT, "request_with_retries_for_key", request)
    monkeypatch.setattr(
        pod_api,
        "_mutation_once",
        lambda key, url, **kwargs: calls.append((key, url, kwargs["method"])),
    )
    observations = iter([_pod_row("pod-1", "flash-0123456789ab-s0-a0"), None])
    monkeypatch.setattr(
        pod_api,
        "get_pod_for_fingerprint",
        lambda *_args, **_kwargs: (
            api._parse_pod(row) if (row := next(observations)) is not None else None
        ),
    )

    auth.advance_key()
    assert auth.active_key() == "secretB"

    pod = api.list_pods_for_key("secretA")[0]
    assert pod.id == "pod-1"
    api.delete_pod_for_fingerprint("pod-1", owner, deadline_at=4_000_000_000.0)

    assert calls[0][0] == "secretA"
    assert calls[1][0] == "secretA"


def test_strict_pod_handle_rejects_truncated_owner_identity():
    from flash.providers.runpod.pods import RunpodPodHandle

    payload = {
        "provider": "runpod",
        "instance_id": "pod-1",
        "phase": "exact",
        "label": "flash-run-s0-a0-0123456789abcdef-deadbeef",
        "key_fingerprint": "rpk-" + "a" * 12,
        "account_id": "account-1",
        "payload_secret_id": "secret-1",
        "payload_secret_name": "FLASH_PAYLOAD_0123456789abcdef",
        "data_center_id": "US-KS-2",
        "network_volume_id": None,
        "container_disk_gb": 120,
        "container_registry_auth_id": None,
        "gpu_count": 1,
        "image_name": None,
        "gpu_type_id_override": None,
        "allowed_cuda_versions": None,
        "docker_start_cmd": [],
        "gpu": "RTX 4090",
        "hourly_usd": 0.69,
        "attempt": 0,
        "started_ts": 1.0,
    }

    with pytest.raises(ValueError, match="key fingerprint is invalid"):
        RunpodPodHandle.from_dict(payload)

    payload["key_fingerprint"] = "rpk-" + "a" * 64
    assert RunpodPodHandle.from_dict(payload).key_fingerprint == payload["key_fingerprint"]
