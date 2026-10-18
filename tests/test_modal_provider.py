"""Training substrate contracts: ownership, lifecycle, resources, and artifact recovery."""

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from flash.providers.core.base import AllocationConstraints, JobHandle, UnreconciledCreateError
from flash.providers.core.registry import PROVIDER_NAMES, available_providers, get_provider
from flash.providers.modal import api
from flash.providers.modal.auth import scope_fingerprint
from flash.providers.modal.handles import ModalJobHandle
from flash.providers.modal.pricing import hourly_rate


def remote(**changes):
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
    }
    values.update(changes)
    return ModalJobHandle(**values).to_dict()


def test_registry_has_only_modal():
    assert PROVIDER_NAMES == available_providers() == ("modal",)
    assert get_provider("modal").name == "modal"
    with pytest.raises(KeyError):
        get_provider("unsupported")


@pytest.mark.parametrize("key", ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"])
def test_both_credentials_are_required(monkeypatch, key):
    monkeypatch.setenv(key, "   ")
    assert not get_provider("modal").is_configured()
    assert get_provider("modal").preflight(require_hf=False)


def test_candidates_respect_vram_and_per_class_card_limit():
    candidates = get_provider("modal").live_candidates(24, AllocationConstraints(max_gpu_count=8))
    assert candidates
    assert all(c.vram_gb >= 24 for c in candidates)
    assert max(c.gpu_count for c in candidates if c.gpu == "A10") == 4
    assert max(c.gpu_count for c in candidates if c.gpu == "RTX Pro 6000") == 1
    assert max(c.gpu_count for c in candidates if c.gpu == "H100") == 8


def test_unsupported_disk_is_rejected_before_launch():
    with pytest.raises(ValueError, match="512"):
        get_provider("modal").live_candidates(24, AllocationConstraints(disk_gb=513))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("instance_id", "not-a-sandbox"),
        ("attempt", -1),
        ("started_ts", float("nan")),
        ("hourly_usd", -1),
        ("gpu_count", True),
        ("scope", ""),
        ("run_id", ""),
        ("provider", "unknown"),
    ],
)
def test_invalid_handle_is_rejected(key, value):
    data = remote()
    data[key] = value
    with pytest.raises((ValueError, TypeError)):
        ModalJobHandle.from_dict(data)


def test_handle_roundtrip():
    data = remote(gpu_count=4)
    assert ModalJobHandle.from_dict(data).to_dict() == data


def test_cancellation_targets_exact_sandbox(monkeypatch):
    stopped = []
    monkeypatch.setattr(api, "terminate", stopped.append)
    get_provider("modal").cancel(JobHandle.from_dict(remote()))
    assert stopped == ["sb-test123"]


def test_changed_credentials_cannot_teardown_another_owner(monkeypatch):
    data = remote()
    monkeypatch.setenv("MODAL_TOKEN_ID", "other-workspace")
    monkeypatch.setattr(api, "terminate", lambda *_: pytest.fail("must fail before mutation"))
    with pytest.raises(RuntimeError, match="owner"):
        get_provider("modal").destroy(JobHandle.from_dict(data))


def test_orphan_sweep_rechecks_live_labels_after_listing(monkeypatch):
    active = set()

    def listing():
        active.add("flash-live")
        return [
            {"id": "sb-live", "tags": {"flash_run": "flash-live"}},
            {"id": "sb-dead", "tags": {"flash_run": "flash-dead"}},
            {"id": "sb-foreign", "tags": {"flash_run": "flash-foreign"}},
        ]

    stopped = []
    monkeypatch.setattr(api, "list_instances", listing)
    monkeypatch.setattr(api, "terminate", stopped.append)
    result = get_provider("modal").sweep_orphans(lambda: active, {"flash-live", "flash-dead"})
    assert result == stopped == ["sb-dead"]


def test_enumeration_failure_cannot_prove_worker_absence(monkeypatch):
    def failed():
        raise api.ModalApiError("unavailable")

    monkeypatch.setattr(api, "list_instances", failed)
    with pytest.raises(api.ModalApiError):
        get_provider("modal").run_instances_remaining("flash-test")


class AsyncMethod:
    def __init__(self, fn):
        self.fn = fn

    async def aio(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


@pytest.fixture
def sdk(monkeypatch):
    class Missing(Exception):
        pass

    state = SimpleNamespace(
        created=[], active=None, failures=False, terminated=[], secrets=[], volumes=[]
    )

    class Image:
        def run_commands(self, *args):
            return self

        def add_local_file(self, path, destination, **kwargs):
            assert destination.endswith("capsule.pyz")
            return self

    def create(*args, **kwargs):
        state.created.append((args, kwargs))
        state.active = SimpleNamespace(
            object_id="sb-created", get_tags=AsyncMethod(lambda: kwargs["tags"])
        )
        if state.failures:
            raise OSError("response lost")
        return state.active

    def lookup(*args, **kwargs):
        if state.active is None:
            raise Missing
        return state.active

    def secret(values):
        state.secrets.append(values)
        return values

    def volume(*args, **kwargs):
        state.volumes.append((args, kwargs))
        return object()

    module = SimpleNamespace(
        App=SimpleNamespace(lookup=AsyncMethod(lambda *a, **k: SimpleNamespace(app_id="ap-test"))),
        Image=SimpleNamespace(from_registry=lambda *a: Image()),
        Volume=SimpleNamespace(from_name=volume),
        Secret=SimpleNamespace(from_dict=secret),
        Sandbox=SimpleNamespace(create=AsyncMethod(create), from_name=AsyncMethod(lookup)),
        exception=SimpleNamespace(NotFoundError=Missing),
    )
    monkeypatch.setattr(
        api, "call", lambda operation, **kw: asyncio.run(operation(module, object()))
    )
    return state


def launch(**overrides):
    values = {
        "name": "flash-test-s0-a0",
        "payload": {
            "run_id": "flash-test",
            "attempt": 0,
            "job_spec_json": "{}",
            "env": {"FLASH_WEIGHT_CACHE_DIR": "/weight-cache/hf-cache/hub"},
        },
        "image_reference": "example/image:tag",
        "gpu": "H100",
        "gpu_count": 2,
        "capsule": b"capsule",
        "deadline_at": time.time() + 100000,
    }
    values.update(overrides)
    return api.create(**values)


def test_create_carries_secrets_outside_tags_and_command(sdk):
    assert launch() == "sb-created"
    args, kwargs = sdk.created[0]
    assert kwargs["gpu"] == "H100!:2"
    assert kwargs["timeout"] == 86400
    assert kwargs["cpu"] == (8, 8)
    assert kwargs["memory"] == (65536, 65536)
    assert "/weight-cache" in kwargs["volumes"]
    assert "FLASH_BOOTSTRAP_PAYLOAD" in sdk.secrets[0]
    assert "FLASH_BOOTSTRAP_PAYLOAD" not in str(args) + str(kwargs["tags"])


def test_lost_create_response_adopts_exact_named_attempt(sdk):
    sdk.failures = True
    assert launch() == "sb-created"
    assert len(sdk.created) == 1


def test_duplicate_submission_adopts_without_second_create(sdk):
    launch()
    assert launch() == "sb-created"
    assert len(sdk.created) == 1


def test_name_collision_with_different_spec_fails_closed(sdk):
    launch()
    with pytest.raises(UnreconciledCreateError):
        launch(
            payload={
                "run_id": "flash-test",
                "attempt": 0,
                "job_spec_json": '{"changed":true}',
                "env": {},
            }
        )
    assert len(sdk.created) == 1


def test_cold_retry_does_not_mount_or_create_volume(sdk):
    launch(payload={"run_id": "flash-test", "attempt": 0, "job_spec_json": "{}", "env": {}})
    assert sdk.created[0][1]["volumes"] == {}
    assert sdk.volumes == []


def test_bootstrap_consumes_secret_payload(monkeypatch):
    from flash.providers._lifecycle.bootstrapping.bootstrap import load_payload

    monkeypatch.setenv("FLASH_BOOTSTRAP_PAYLOAD", json.dumps({"flash_arm": "modal"}))
    assert load_payload() == {"flash_arm": "modal"}
    import os

    assert "FLASH_BOOTSTRAP_PAYLOAD" not in os.environ


def test_duplicate_with_different_source_snapshot_is_not_adopted(sdk):
    launch()
    with pytest.raises(UnreconciledCreateError, match="different launch"):
        launch(
            payload={
                "run_id": "flash-test",
                "attempt": 0,
                "job_spec_json": "{}",
                "env": {},
                "source_snapshot": {"sha256": "different-source"},
            }
        )
    assert len(sdk.created) == 1
