"""router edge coverage for unsupported models and streaming failures.

these cases stay hermetic by routing through fastapi testclient with small cpu-only engine pools.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient

from flash.serving.src.http.router import AdapterRouter
from flash.serving.src.http.router import build_offline_serving_app as build_serving_app
from flash.serving.src.io.schemas import AdapterRecord
from tests.serving.checkpoint_fixtures import checkpoint_record
from tests.serving.conftest import attest

QWEN = "Qwen/Qwen3.5-9B"


async def _allow(_token: str, _adapter_id: str, _scope: dict | None = None) -> str:
    return "org-1"


def _revision(
    run_id: str = "qa",
    *,
    base_model: str = QWEN,
    status: str = "ready",
) -> AdapterRecord:
    return checkpoint_record(run_id, base_model, status=status)


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
        headers={"Authorization": "Bearer test-key", "X-Freesolo-Org-Id": "org-1"},
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
        return attest(
            _record,
            {
                "text": "ok",
                "finish_reason": "stop",
                "checkpoint": _record.adapter_id,
            },
        )

    async def register(self, _base_model: str, _record: AdapterRecord) -> None:
        return None

    async def unregister(
        self,
        _base_model: str,
        _org_id: str,
        _adapter_id: str,
        _expected_generation: str | None = None,
    ) -> None:
        return None


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


def test_base_model_stream_replay_skips_empty_delta_without_checkpoint_header() -> None:
    record = _base_record("empty")
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
        assert "x-freesolo-checkpoint" not in response.headers

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
            record = self.router.get("qa/final", org_id="org-1")
            assert record is not None
            self.router.upsert(record.model_copy(update={"status": "disabled"}))
        raise ValueError(self.message)


def test_checkpoint_mismatch_value_error_is_conflict() -> None:
    revision = _revision()
    response = _client(
        _ValueErrorPool("checkpoint mismatch: expected checkpoint differs"),
        AdapterRouter([revision]),
    ).post("/generate", json={"adapter_id": "qa/final", "prompt": "hi"})

    assert response.status_code == 409
    assert response.json()["detail"] == "expected checkpoint differs"


def test_engine_error_after_adapter_is_disabled_is_not_found() -> None:
    revision = _revision()
    router = AdapterRouter([revision])
    response = _client(_ValueErrorPool("Unknown adapter id on engine", router), router).post(
        "/generate", json={"adapter_id": "qa/final", "prompt": "hi"}
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown adapter id: qa/final"


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
    response = _client(_RaisingStreamPool(), AdapterRouter([revision])).post(
        "/v1/chat/completions",
        json={"model": "qa/final", "messages": [{"role": "user", "content": "hi"}], "stream": True},
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
    router = AdapterRouter([revision])
    assert router.get(revision.adapter_id, org_id="org-1") is not None

    cleanup = apply_teardown(router, [(revision, None, revision.deployment_generation)])

    assert router.get(revision.adapter_id, org_id="org-1") is None
    assert [record.adapter_id for record, _ in cleanup] == [revision.adapter_id]


def test_teardown_keeps_the_authoritative_row_when_one_came_back() -> None:
    """The disabled row returned by the CAS replaces the enumerated one in routing."""
    from flash.serving.src.store.undeploy import apply_teardown

    revision = _revision()
    disabled = revision.model_copy(update={"status": "disabled"})
    router = AdapterRouter([revision])

    cleanup = apply_teardown(router, [(revision, disabled, revision.deployment_generation)])

    assert router.get(revision.adapter_id, org_id="org-1") is not None
    assert router.get(revision.adapter_id, org_id="org-1").status == "disabled"
    assert [record.adapter_id for record, _ in cleanup] == [revision.adapter_id]
