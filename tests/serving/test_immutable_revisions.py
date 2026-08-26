from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from flash.serve.contract.provenance import immutable_binding_fingerprint
from flash.serving.src.http.routing import AdapterRouter
from flash.serving.src.io.schemas import ImmutableCheckpointRegistration
from flash.serving.src.store import registration, undeploy

RUN_ID = "flash-1234567890-abcdef12"
SOURCE_REVISION = "a" * 40
ARTIFACT_DIGEST = "b" * 64


def _record(step: int | None = 20, **overrides):
    selector = "final" if step is None else f"step-{step}"
    checkpoint_id = f"{RUN_ID}/{selector}"
    payload = {
        "adapter_id": checkpoint_id,
        "repo_id": "org/run",
        "base_model": "Qwen/Qwen3.5-9B",
        "subfolder": f"checkpoints/{selector}",
        "repo_type": "model",
        "org_id": "org-1",
        "url": "https://huggingface.co/org/run",
        "checkpoint": checkpoint_id,
        "private": True,
        "thinking": False,
        "structured_outputs": None,
        "run_id": RUN_ID,
        "checkpoint_step": step,
        "artifact_revision": SOURCE_REVISION,
        "artifact_digest": ARTIFACT_DIGEST,
        "lora_rank": 16,
    }
    payload.update(overrides)
    payload["artifact_fingerprint"] = immutable_binding_fingerprint(payload)
    return ImmutableCheckpointRegistration.model_validate(payload).to_record()


def test_first_registration_inserts_one_exact_checkpoint(monkeypatch) -> None:
    checkpoint = _record()
    inserted = []

    async def get_stored(_org_id: str, _checkpoint_id: str):
        return None

    async def insert_or_read(record):
        inserted.append(record.adapter_id)
        return record, True

    monkeypatch.setattr(registration, "_get_stored", get_stored)
    monkeypatch.setattr(registration, "_insert_or_read", insert_or_read)

    stored = asyncio.run(registration.persist_checkpoint(checkpoint))

    assert stored == checkpoint
    assert inserted == [f"{RUN_ID}/step-20"]


def test_identical_registration_retry_is_idempotent(monkeypatch) -> None:
    checkpoint = _record()
    monkeypatch.setattr(
        registration, "_get_stored", lambda _org_id, _checkpoint_id: _async(checkpoint)
    )

    stored = asyncio.run(registration.persist_checkpoint(_record()))

    assert stored is checkpoint


@pytest.mark.parametrize(
    "changed",
    [
        _record(artifact_digest="d" * 64),
        _record(thinking=True),
    ],
)
def test_changed_immutable_registration_conflicts(monkeypatch, changed) -> None:
    existing = _record()
    monkeypatch.setattr(
        registration, "_get_stored", lambda _org_id, _checkpoint_id: _async(existing)
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(registration.persist_checkpoint(changed))

    assert raised.value.status_code == 409


def test_equal_checkpoint_string_cannot_cross_org_authorization(monkeypatch) -> None:
    existing = _record(org_id="org-a")
    monkeypatch.setattr(
        registration, "_get_stored", lambda _org_id, _checkpoint_id: _async(existing)
    )

    with pytest.raises(HTTPException) as raised:
        asyncio.run(registration.persist_checkpoint(_record(org_id="org-b")))

    assert raised.value.status_code == 404


def test_router_resolves_only_ready_exact_checkpoints() -> None:
    ready = _record().model_copy(update={"status": "ready"})
    disabled = _record(40)
    router = AdapterRouter([ready, disabled])

    assert router.resolve(ready.adapter_id, org_id=ready.org_id) == (ready, ready)
    assert router.resolve(disabled.adapter_id, org_id=disabled.org_id) is None
    assert router.resolve(RUN_ID) is None


def test_public_undeploy_requires_exact_checkpoint(monkeypatch) -> None:
    router = AdapterRouter()

    with pytest.raises(HTTPException) as raised:
        asyncio.run(undeploy.resolve_undeploy_target(router, "org-1", RUN_ID))

    assert raised.value.status_code == 404


def test_exact_undeploy_preserves_ready_sibling(monkeypatch) -> None:
    target = _record().model_copy(
        update={
            "status": "ready",
            "updated_at": "2026-08-26T00:00:00+00:00",
            "deployment_generation": "generation-a",
        }
    )
    sibling = _record(40).model_copy(
        update={
            "status": "ready",
            "updated_at": "2026-08-26T00:00:01+00:00",
            "deployment_generation": "generation-b",
        }
    )
    rows = {target.adapter_id: target, sibling.adapter_id: sibling}

    async def replace(record, *, expected_updated_at: str):
        current = rows.get(record.adapter_id)
        assert current is not None
        assert current.updated_at == expected_updated_at
        stored = record.model_copy(update={"updated_at": "2026-08-26T00:00:02+00:00"})
        rows[record.adapter_id] = stored
        return stored

    async def authoritative(checkpoint_id: str):
        return rows.get(checkpoint_id)

    monkeypatch.setattr(undeploy, "_replace_stored_cas", replace)
    result = asyncio.run(undeploy.disable_matched([target], get_authoritative=authoritative))

    assert result.disabled_checkpoints == [target.adapter_id]
    assert rows[target.adapter_id].status == "disabled"
    assert rows[sibling.adapter_id].status == "ready"
    assert rows[sibling.adapter_id].deployment_generation == "generation-b"


def test_already_disabled_checkpoint_is_an_idempotent_noop(monkeypatch) -> None:
    disabled = _record().model_copy(
        update={
            "status": "disabled",
            "updated_at": "2026-08-26T00:00:00+00:00",
            "deployment_generation": None,
        }
    )

    async def unexpected_authoritative(_checkpoint_id: str):
        raise AssertionError("settled disabled checkpoint must not be written again")

    result = asyncio.run(
        undeploy.disable_matched([disabled], get_authoritative=unexpected_authoritative)
    )

    assert result.disabled_checkpoints == []
    assert result.pending_teardown == []
    assert result.stuck_ready == []


def test_internal_run_cleanup_enumerates_exact_checkpoint_bindings(monkeypatch) -> None:
    checkpoints = [_record(), _record(40)]
    monkeypatch.setattr(
        undeploy,
        "list_authoritative_run",
        lambda _org_id, _run_id: _async(checkpoints),
    )

    assert asyncio.run(undeploy.resolve_run_cleanup_targets("org-1", RUN_ID)) == checkpoints


async def _async(value):
    return value
