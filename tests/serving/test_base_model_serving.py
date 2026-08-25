"""Serving a base model with NO LoRA adapter.

Each per-base-model engine already has its base weights loaded, so a base serve just generates with
no adapter. Base-model records are pre-seeded into the router in memory (one per served base model),
addressable by name. A base serve requires a valid API key (any org, not gated to an owner) and is
billed to the CALLING org — the backend authorizes it and returns the caller's org.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from flash.serving.src.router import AdapterRouter, build_serving_app
from flash.serving.src.schemas import AdapterRecord
from tests.serving.conftest import RecordingUsageStore, attest

QWEN = "Qwen/Qwen3.5-9B"
INTERNAL_KEY = "fs-internal"


def _lora_rec(run_id: str = "qa") -> AdapterRecord:
    sha = "a" * 40
    return AdapterRecord.model_validate(
        {
            "adapter_id": f"{run_id}@final.{sha}",
            "repo_id": f"org/{run_id}",
            "base_model": QWEN,
            "org_id": "org-A",
            "checkpoint": run_id,
            "status": "ready",
            "thinking": False,
            "metadata": {
                "record_type": "revision",
                "run_id": run_id,
                "checkpoint_step": None,
                "hf_revision": sha,
            },
        }
    )


def _lora_alias(revision: AdapterRecord) -> AdapterRecord:
    run_id = revision.run_id
    assert run_id is not None
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


def _base_rec(base_model: str = QWEN) -> AdapterRecord:
    return AdapterRecord(
        adapter_id=base_model,
        repo_id=base_model,
        base_model=base_model,
        serve_base_model=True,
        thinking=False,
        org_id=None,
        status="ready",
    )


class FakePool:
    async def generate(self, base_model, payload, record, *, expected_checkpoint=None):
        return attest(
            record,
            {
                "text": "hi",
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "cached_tokens_reported": False,
                "reasoning_tokens": 0,
                "request_id": payload.generation_id,
                "checkpoint": "",
            },
        )

    async def stream_generate(self, *a, **k):  # pragma: no cover
        yield {"type": "final", "finish_reason": "stop", "prompt_tokens": 3, "completion_tokens": 2}

    async def register(self, base_model, record) -> None:  # pragma: no cover
        pass

    async def unregister(
        self, base_model, adapter_id, expected_generation=None
    ) -> None:  # pragma: no cover
        pass


class FakeAuthorizer:
    def __init__(self, org="caller-org"):
        self.calls = []
        self._org = org

    async def __call__(self, token, adapter_id):
        self.calls.append((token, adapter_id))
        return self._org


def _build(records, *, authorizer=None):
    store = RecordingUsageStore()
    app = build_serving_app(
        FakePool(),
        AdapterRouter(records),
        internal_key=INTERNAL_KEY,
        chat_authorizer=authorizer,
        usage_store=store,
    )
    return TestClient(app), store


def _chat(client, model, **headers):
    return client.post(
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )


def test_base_model_requires_a_valid_api_key() -> None:
    auth = FakeAuthorizer()
    client, _ = _build([_base_rec()], authorizer=auth)
    assert _chat(client, QWEN).status_code == 401  # no key -> rejected
    assert auth.calls == []
    assert _chat(client, QWEN, Authorization="Bearer k").status_code == 200
    assert auth.calls == [("k", QWEN)]  # the key + base model id are handed to the backend


def test_base_model_serve_records_the_authorized_org_principal() -> None:
    client, store = _build([_base_rec()], authorizer=FakeAuthorizer(org="caller-org"))
    assert _chat(client, QWEN, Authorization="Bearer k").status_code == 200
    assert len(store.finalized) == 1
    event = store.finalized[0]
    assert event.principal.kind == "freesolo_org"
    assert event.principal.orgId == "caller-org"
    assert event.target.requested_adapter_id == QWEN
    assert event.target.base_model == QWEN


def test_base_model_via_internal_key_is_durably_unattributed() -> None:
    client, store = _build([_base_rec()], authorizer=FakeAuthorizer())
    resp = _chat(client, QWEN, **{"X-Freesolo-Internal-Key": INTERNAL_KEY})
    assert resp.status_code == 200
    assert len(store.finalized) == 1
    event = store.finalized[0]
    assert event.principal.kind == "trusted_internal"
    assert event.principal.orgId is None
    assert event.target.requested_adapter_id == QWEN


def test_internal_lora_is_explicitly_attributed_to_immutable_owner() -> None:
    revision = _lora_rec("qa")
    client, store = _build([revision, _lora_alias(revision)], authorizer=FakeAuthorizer())

    response = _chat(client, "qa", **{"X-Freesolo-Internal-Key": INTERNAL_KEY})

    assert response.status_code == 200
    assert len(store.finalized) == 1
    principal = store.finalized[0].principal
    assert principal.kind == "trusted_internal"
    assert principal.orgId == "org-A"
    assert principal.billingAttributionExplicit is True


def test_external_authorizer_none_fails_closed_before_dispatch() -> None:
    client, store = _build([_base_rec()], authorizer=FakeAuthorizer(org=None))

    response = _chat(client, QWEN, Authorization="Bearer k")

    assert response.status_code == 503
    assert response.json() == {"detail": "serving auth did not return an attributable principal"}
    assert store.finalized == []


def test_lora_adapter_still_requires_a_key_and_records_requested_identity() -> None:
    auth = FakeAuthorizer()
    revision = _lora_rec("qa")
    client, store = _build([revision, _lora_alias(revision)], authorizer=auth)
    assert _chat(client, "qa").status_code == 401
    assert _chat(client, "qa", Authorization="Bearer k").status_code == 200
    assert auth.calls == [("k", "qa")]
    assert len(store.finalized) == 1
    assert store.finalized[0].target.requested_adapter_id == "qa"
    assert store.finalized[0].principal.orgId == "caller-org"


def test_adapter_record_defaults_serve_base_model_false() -> None:
    assert _lora_rec().serve_base_model is False


# --- engine: a base-model record resolves to NO LoRA request (no download) -------------------


@pytest.fixture
def modal_app_module():
    modal_stub = MagicMock(name="modal")

    def _passthrough(*_a, **_k):
        def deco(obj):
            return obj

        return deco

    for attr in ("concurrent", "method", "enter", "asgi_app"):
        getattr(modal_stub, attr).side_effect = _passthrough
    modal_stub.parameter.return_value = None
    app_mock = MagicMock(name="app")
    for attr in ("cls", "function", "local_entrypoint"):
        getattr(app_mock, attr).side_effect = _passthrough
    modal_stub.App.return_value = app_mock
    modal_stub.Period.return_value = MagicMock()
    _MISSING = object()
    prev_modal = sys.modules.get("modal", _MISSING)
    prev_modal_app = sys.modules.get("flash.serving.modal_app", _MISSING)
    sys.modules["modal"] = modal_stub
    import flash.serving.modal_app as modal_app

    try:
        yield modal_app
    finally:
        for name, prev in (("modal", prev_modal), ("modal_app", prev_modal_app)):
            if prev is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


def test_base_model_records_seed_one_open_record_per_model(modal_app_module):
    from flash.serving.src.model_config import base_models

    recs = modal_app_module._base_model_records()
    assert {r.adapter_id for r in recs} == set(base_models())
    assert all(
        r.serve_base_model and r.org_id is None and r.adapter_id == r.base_model for r in recs
    )


def test_lora_request_returns_no_lora_for_base_model(modal_app_module):
    import asyncio

    from flash.serving.src.registry import AdapterRegistry

    engine = object.__new__(modal_app_module._LoraEngineImpl)
    engine.base_model = QWEN
    engine.registry = AdapterRegistry()
    engine.registry.hydrate([_base_rec()])
    engine._adapter_locks = {}
    engine._adapter_locks_guard = asyncio.Lock()

    def _boom(_record):
        raise AssertionError("base-model serve must not download a LoRA")

    engine._ensure_adapter_local_locked = _boom  # type: ignore[assignment]

    lora_request, record = asyncio.run(engine._lora_request(QWEN))
    assert lora_request is None
    assert record.serve_base_model is True
