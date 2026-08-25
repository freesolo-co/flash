"""Router edge coverage exercises rejected aliases, unsupported models, and streaming failures.

These cases stay hermetic by routing through FastAPI TestClient with small CPU-only engine pools.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from flash.serving.src.http.router import AdapterRouter
from flash.serving.src.http.router import build_offline_serving_app as build_serving_app
from flash.serving.src.io.schemas import AdapterRecord

QWEN = "Qwen/Qwen3.5-9B"
SHA = "a" * 40


async def _allow(_token: str, _adapter_id: str) -> str:
    return "org-1"


def _revision(
    run_id: str = "qa",
    *,
    base_model: str = QWEN,
    status: str = "ready",
) -> AdapterRecord:
    return AdapterRecord.model_validate(
        {
            "adapter_id": f"{run_id}@final.{SHA}",
            "repo_id": f"org/{run_id}",
            "base_model": base_model,
            "org_id": "org-1",
            "checkpoint": run_id,
            "status": status,
            "thinking": False,
            "metadata": {
                "record_type": "revision",
                "run_id": run_id,
                "checkpoint_step": None,
                "hf_revision": SHA,
            },
        }
    )


def _alias(revision: AdapterRecord) -> AdapterRecord:
    run_id = str(revision.metadata["run_id"])
    return revision.model_copy(
        update={
            "adapter_id": run_id,
            "checkpoint": None,
            "metadata": {
                "record_type": "alias",
                "run_id": run_id,
                "alias_of": revision.adapter_id,
            },
        }
    )


def _base_record(
    adapter_id: str,
    *,
    checkpoint: str | None = None,
    subfolder: str | None = None,
) -> AdapterRecord:
    return AdapterRecord.model_validate(
        {
            "adapter_id": adapter_id,
            "repo_id": f"org/{adapter_id}",
            "base_model": QWEN,
            "checkpoint": checkpoint,
            "subfolder": subfolder,
            "serve_base_model": True,
            "thinking": False,
        }
    )


def _client(pool: Any, router: AdapterRouter, **kwargs: Any) -> TestClient:
    return TestClient(
        build_serving_app(pool, router, chat_authorizer=_allow, **kwargs),
        headers={"Authorization": "Bearer test-key"},
    )


class _Pool:
    async def generate(
        self,
        _base_model: str,
        _payload: Any,
        _record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        return {"text": "ok", "finish_reason": "stop"}

    async def register(self, _base_model: str, _record: AdapterRecord) -> None:
        return None

    async def unregister(
        self,
        _base_model: str,
        _adapter_id: str,
        _expected_generation: str | None = None,
    ) -> None:
        return None


@pytest.mark.parametrize("target_state", ["missing", "disabled", "not_revision"])
def test_resolve_rejects_alias_with_unusable_target(target_state: str) -> None:
    revision = _revision()
    alias = _alias(revision)
    records = [alias]
    if target_state == "disabled":
        records.append(revision.model_copy(update={"status": "disabled"}))
    elif target_state == "not_revision":
        records.append(revision.model_copy(update={"checkpoint": None, "metadata": {}}))

    assert AdapterRouter(records).resolve(alias.adapter_id) is None


def test_unsupported_base_model_is_reported_and_rejected() -> None:
    unsupported = "example/uncataloged-model"
    revision = _revision(base_model=unsupported)
    router = AdapterRouter([revision])
    client = _client(_Pool(), router)

    health = client.get("/healthz").json()
    assert health["unsupported_base_models"] == [unsupported]
    assert health["base_models"] == [unsupported]
    assert health["gpus"] == 0

    response = client.post("/generate", json={"adapter_id": revision.adapter_id, "prompt": "hi"})
    assert response.status_code == 400
    assert f"Unsupported base model: {unsupported}" in response.json()["detail"]
    assert "Qwen/Qwen3.5-9B" in response.json()["detail"]


class _ReplayPool(_Pool):
    async def stream_generate(
        self,
        _base_model: str,
        _payload: Any,
        _record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "delta", "text": ""}
        yield {"type": "delta", "text": "hello"}
        yield {"type": "final", "finish_reason": "stop"}


@pytest.mark.parametrize(
    ("record", "expected_checkpoint"),
    [
        (_base_record("explicit", checkpoint="  explicit/ref  "), "explicit/ref"),
        (_base_record("empty"), None),
        (
            _base_record("stepped", subfolder="nested/checkpoints/step-17/adapter"),
            "stepped/step-17",
        ),
        (_base_record("folder", subfolder="adapter"), "folder"),
    ],
)
def test_stream_replay_derives_checkpoint_and_skips_empty_delta(
    record: AdapterRecord, expected_checkpoint: str | None
) -> None:
    client = _client(_ReplayPool(), AdapterRouter([record]))
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": record.adapter_id,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as response:
        assert response.status_code == 200
        body = response.read().decode()
        if expected_checkpoint is None:
            assert "x-freesolo-checkpoint" not in response.headers
        else:
            assert response.headers["x-freesolo-checkpoint"] == expected_checkpoint

    assert '"delta":{"content":"hello"}' in body
    assert '"delta":{"content":""}' not in body


class _ValueErrorPool(_Pool):
    def __init__(self, message: str, router: AdapterRouter | None = None) -> None:
        self.message = message
        self.router = router

    async def generate(
        self,
        _base_model: str,
        _payload: Any,
        _record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ) -> dict[str, Any]:
        if self.router is not None:
            alias = self.router.get("qa")
            assert alias is not None
            self.router.upsert(alias.model_copy(update={"status": "disabled"}))
        raise ValueError(self.message)


def test_checkpoint_mismatch_value_error_is_conflict() -> None:
    revision = _revision()
    alias = _alias(revision)
    response = _client(
        _ValueErrorPool("checkpoint mismatch: expected checkpoint differs"),
        AdapterRouter([revision, alias]),
    ).post("/generate", json={"adapter_id": "qa", "prompt": "hi"})

    assert response.status_code == 409
    assert response.json()["detail"] == "expected checkpoint differs"


def test_engine_error_after_adapter_is_disabled_is_not_found() -> None:
    revision = _revision()
    alias = _alias(revision)
    router = AdapterRouter([revision, alias])
    response = _client(_ValueErrorPool("adapter unavailable", router), router).post(
        "/generate", json={"adapter_id": "qa", "prompt": "hi"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown adapter id: qa"


class _RaisingEvents:
    def __aiter__(self) -> _RaisingEvents:
        return self

    async def __anext__(self) -> dict[str, Any]:
        raise ValueError("checkpoint mismatch: stream checkpoint differs")


class _RaisingStreamPool(_Pool):
    def stream_generate(
        self,
        _base_model: str,
        _payload: Any,
        _record: AdapterRecord,
        *,
        expected_checkpoint: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        return _RaisingEvents()


def test_stream_first_event_value_error_is_translated() -> None:
    revision = _revision()
    alias = _alias(revision)
    response = _client(_RaisingStreamPool(), AdapterRouter([revision, alias])).post(
        "/v1/chat/completions",
        json={"model": "qa", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "stream checkpoint differs"


def test_teardown_drops_a_row_that_vanished_from_persistence() -> None:
    """A matched row whose authoritative record is gone must leave routing.

    ``disable_matched`` reports ``current=None`` when the row disappeared (or stopped being
    ready) between enumeration and the compare-and-swap. Leaving it in the router would keep
    serving an adapter that no longer exists in persistence, and the next reload would not
    correct it: a reload only hydrates rows that ARE present, so it cannot remove one that is not.
    """
    from flash.serving.src.store.undeploy import apply_teardown

    revision = _revision()
    router = AdapterRouter([revision, _alias(revision)])
    assert router.get(revision.adapter_id) is not None

    cleanup = apply_teardown(router, [(revision, None, revision.deployment_generation)])

    assert router.get(revision.adapter_id) is None
    assert [record.adapter_id for record, _ in cleanup] == [revision.adapter_id]


def test_teardown_keeps_the_authoritative_row_when_one_came_back() -> None:
    """The disabled row returned by the CAS replaces the enumerated one in routing."""
    from flash.serving.src.store.undeploy import apply_teardown

    revision = _revision()
    disabled = revision.model_copy(update={"status": "disabled"})
    router = AdapterRouter([revision, _alias(revision)])

    cleanup = apply_teardown(router, [(revision, disabled, revision.deployment_generation)])

    assert router.get(revision.adapter_id) is not None
    assert router.get(revision.adapter_id).status == "disabled"
    assert [record.adapter_id for record, _ in cleanup] == [revision.adapter_id]
