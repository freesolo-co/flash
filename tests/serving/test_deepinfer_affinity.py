"""Replica hints preserve conversation locality without sharing tenant identity."""

import asyncio
import copy
import json

import httpx
import pytest

from flash.serving.deepinfer.app.affinity import conversation_affinity
from flash.serving.deepinfer.app.client import HostedDeepInfer
from flash.serving.deepinfer.contract.config import MODEL_ID
from flash.serving.src.io.openai_request import OpenAIGenerateRequest
from tests.serving.test_deepinfer_hosted import connection, record  # noqa: F401


def chat():
    return {
        "messages": [
            {"role": "system", "content": "Shared system prompt"},
            {"role": "user", "content": "Private opening question"},
        ],
        "chat_template_kwargs": {},
    }


def test_affinity_survives_later_turns_and_sampling_changes():
    original = chat()
    later = copy.deepcopy(original)
    later["messages"] += [
        {"role": "assistant", "content": "An answer"},
        {"role": "user", "content": "Continue"},
    ]
    later.update(max_tokens=350000, temperature=0.8, seed=42)
    key = conversation_affinity(original, "tenant-a")
    assert conversation_affinity(later, "tenant-a") == key
    assert len(key) == 64
    assert set(key) <= set("0123456789abcdef")
    assert original == chat()


def test_shared_system_prompt_does_not_merge_distinct_conversations():
    first = chat()
    second = chat()
    second["messages"][-1]["content"] = "Different opening question"
    assert conversation_affinity(first, "tenant-a") != conversation_affinity(second, "tenant-a")


@pytest.mark.parametrize("change", ["tenant", "system", "thinking", "revision"])
def test_cache_identity_changes_affinity(monkeypatch, change):
    body = chat()
    original = conversation_affinity(body, "tenant-a")
    namespace = "tenant-b" if change == "tenant" else "tenant-a"
    if change == "system":
        body["messages"][0]["content"] = "New system prompt"
    if change == "thinking":
        body["chat_template_kwargs"]["enable_thinking"] = True
    if change == "revision":
        monkeypatch.setattr("flash.serving.deepinfer.app.affinity.MODEL_REVISION", "new")
    assert conversation_affinity(body, namespace) != original


def test_missing_tenant_or_user_turn_omits_hint():
    assert conversation_affinity(chat(), None) is None
    body = chat()
    body["messages"].pop()
    assert conversation_affinity(body, "tenant-a") is None


@pytest.mark.parametrize("namespace", [None, "authorized-tenant"])
@pytest.mark.usefixtures("connection")
def test_hosted_bridge_sends_hint_only_with_authoritative_tenancy(namespace):
    calls = []

    async def handler(request):
        calls.append(request)
        sent = json.loads(request.content)
        assert sent["cache_namespace"] == namespace
        assert sent["chat"]["messages"] == chat()["messages"]
        return httpx.Response(
            200, text=json.dumps({"type": "final", "request_id": "affinity-test"}) + "\n"
        )

    async def run():
        pool = HostedDeepInfer(transport=httpx.MockTransport(handler))
        payload = OpenAIGenerateRequest(
            model_id=MODEL_ID, messages=chat()["messages"], generation_id="affinity-test"
        )
        payload._cache_namespace = namespace
        try:
            await pool.generate(MODEL_ID, payload, record())
        finally:
            await pool.aclose()

    asyncio.run(run())
    assert len(calls) == 1
    hint = calls[0].headers.get("Modal-Routing-Affinity-Key")
    assert hint == conversation_affinity(chat(), namespace)
