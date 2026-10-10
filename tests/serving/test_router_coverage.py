"""Extra coverage for src/router.py.

Offline & hermetic (matches the sibling suites): pure helpers are exercised directly and endpoints
run through a FastAPI TestClient against a fake engine pool.

Targets branches the existing suites miss:
  router: ``_usage_block`` bad-cached fallback, ``_response_format_to_spec`` non-dict/text,
  bearer/authorizer auth failures, and durable-versus-auxiliary shutdown behavior.
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest
from fastapi.testclient import TestClient

from flash.serving.src.http.router import AdapterRouter
from flash.serving.src.http.router import build_offline_serving_app as build_serving_app
from flash.serving.src.http.router import build_serving_app as build_durable_serving_app
from flash.serving.src.io.responses import (
    _openai_structured_outputs,
    _response_format_to_spec,
    _usage_block,
)
from flash.serving.src.io.schemas import AdapterRecord
from flash.serving.src.io.structured_outputs import StructuredOutputsError

QWEN = "Qwen/Qwen3.5-9B"


async def _allow(_token: str, _adapter_id: str, _scope: dict | None = None) -> str:
    return "org-1"


def _rec(adapter_id: str, base_model: str = QWEN, **overrides) -> AdapterRecord:
    run_id = adapter_id.split("/", 1)[0]
    checkpoint_id = adapter_id if "/" in adapter_id else f"{run_id}/final"
    checkpoint = checkpoint_id.split("/", 1)[1]
    data = {
        "adapter_id": checkpoint_id,
        "repo_id": f"org/{run_id}",
        "org_id": "org-1",
        "base_model": base_model,
        "status": "ready",
        "thinking": True,
        "checkpoint": checkpoint_id,
        "run_id": run_id,
        "checkpoint_step": None if checkpoint == "final" else int(checkpoint.removeprefix("step-")),
        "artifact_revision": hashlib.sha1(run_id.encode()).hexdigest(),
        "artifact_digest": hashlib.sha256(f"{run_id}-artifact".encode()).hexdigest(),
        "artifact_fingerprint": hashlib.sha256(f"{run_id}-binding".encode()).hexdigest(),
        "lora_rank": 16,
    }
    data.update(overrides)
    return AdapterRecord.model_validate(data)


# --------------------------------------------------------------------------------------------
# router pure helpers
# --------------------------------------------------------------------------------------------


def test_usage_block_clamps_cached_and_survives_a_bad_value():
    # Normal: total = prompt + completion, no cached-details block when cached is 0/absent.
    assert _usage_block(10, 5, 0) == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
    # cached clamped to prompt_tokens (never larger than the prompt).
    assert _usage_block(4, 1, 99)["prompt_tokens_details"] == {"cached_tokens": 4}
    # A partial cache hit is reported verbatim.
    assert _usage_block(10, 2, 6)["prompt_tokens_details"] == {"cached_tokens": 6}
    # A non-numeric cached value falls back to 0 (no details block), never raises.
    bad = _usage_block(10, 2, "not-a-number")
    assert "prompt_tokens_details" not in bad
    assert bad["total_tokens"] == 12
    # None cached -> 0 -> omitted.
    assert "prompt_tokens_details" not in _usage_block(3, 3, None)


def test_response_format_to_spec_translates_each_form():
    # Non-dict returned unchanged (the GenerateRequest validator normalizes it downstream).
    assert _response_format_to_spec("raw") == "raw"
    assert _response_format_to_spec(["a"]) == ["a"]
    # {"type": "text"} -> explicit "unconstrained".
    assert _response_format_to_spec({"type": "text"}) == {}
    assert _response_format_to_spec({"type": "json_object"}) == {"json_object": True}
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    # Nested json_schema.schema and the flattened schema both map to {"json": schema}.
    assert _response_format_to_spec({"type": "json_schema", "json_schema": {"schema": schema}}) == {
        "json": schema
    }
    assert _response_format_to_spec({"type": "json_schema", "schema": schema}) == {"json": schema}


def test_response_format_to_spec_rejects_malformed():
    with pytest.raises(StructuredOutputsError, match="requires a schema"):
        _response_format_to_spec({"type": "json_schema"})
    with pytest.raises(StructuredOutputsError, match="unsupported response_format type"):
        _response_format_to_spec({"type": "image_url"})


def test_openai_structured_output_fields_are_mutually_exclusive():
    with pytest.raises(
        StructuredOutputsError,
        match="structured_outputs and response_format cannot both be set",
    ):
        _openai_structured_outputs(
            {
                "structured_outputs": {"choice": ["a"]},
                "response_format": {"type": "json_object"},
            }
        )
    # response_format alone translates to the canonical spec.
    assert _openai_structured_outputs({"response_format": {"type": "json_object"}}) == {
        "json_object": True
    }
    # structured_outputs alone remains unchanged, including an explicit unconstrained marker.
    assert _openai_structured_outputs({"structured_outputs": {"choice": ["a"]}}) == {
        "choice": ["a"]
    }
    assert _openai_structured_outputs({"structured_outputs": {}}) == {}
    # neither field is unconstrained by the request.
    assert _openai_structured_outputs({"messages": []}) is None


# --------------------------------------------------------------------------------------------
# router auth branches
# --------------------------------------------------------------------------------------------


def test_inference_requires_a_bearer_api_key():
    """No Authorization header, or a non-bearer scheme, is a 401 before the engine is reached."""
    router = AdapterRouter([_rec("qa/final")])
    client = TestClient(build_serving_app(FakePool(), router, chat_authorizer=_allow))

    no_header = client.post("/generate", json={"adapter_id": "qa/final", "prompt": "hi"})
    assert no_header.status_code == 401
    assert "Missing Freesolo API key" in no_header.json()["detail"]

    non_bearer = client.post(
        "/generate",
        json={"adapter_id": "qa/final", "prompt": "hi"},
        headers={"Authorization": "Basic Zm9vOmJhcg=="},
    )
    assert non_bearer.status_code == 401  # scheme != bearer -> token treated as absent


def test_inference_fails_closed_when_no_authorizer_is_wired():
    """A bearer key with no chat_authorizer configured must fail closed (503), not serve open."""
    router = AdapterRouter([_rec("qa/final")])
    client = TestClient(build_serving_app(FakePool(), router, chat_authorizer=None))
    resp = client.post(
        "/generate",
        json={"adapter_id": "qa/final", "prompt": "hi"},
        headers={"Authorization": "Bearer k"},
    )
    assert resp.status_code == 503
    assert "serving auth is not configured" in resp.json()["detail"]


def test_chat_completions_rejects_a_missing_or_blank_model():
    """`model` must be a non-empty adapter id; anything else is a clean 400 (before auth/routing)."""
    router = AdapterRouter([_rec("qa/final")])
    client = TestClient(
        build_serving_app(FakePool(), router, chat_authorizer=_allow),
        headers={"Authorization": "Bearer k"},
    )
    for model in ("   ", "", None, 123):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 400, model
        assert "model must be the adapter id" in resp.json()["detail"]


# --------------------------------------------------------------------------------------------
# router streaming: replay path + checkpoint-ref + empty-delta skip + untokened-report drop
# --------------------------------------------------------------------------------------------


class FakePool:
    """Minimal engine pool: generate() returns no token counts; streaming yields NO leading
    'ready' event (so the router takes the replay path and derives the checkpoint from the
    record), skips an empty delta, and ends with a final that carries no token counts."""

    async def generate(self, base_model, payload, record, *, expected_checkpoint=None):
        return {"ok": True, "text": f"[{base_model}] reply", "finish_reason": "stop"}

    async def stream_generate(self, base_model, payload, record, *, expected_checkpoint=None):
        yield {"type": "delta", "text": ""}  # empty -> skipped by the SSE encoder
        yield {"type": "delta", "text": "hello"}
        yield {"type": "final", "finish_reason": "stop"}  # no token counts -> nothing to meter

    async def register(self, base_model, record) -> None:
        pass

    async def unregister(self, base_model, adapter_id, expected_generation=None) -> None:
        pass


# --------------------------------------------------------------------------------------------
# router registration: warmup fails AND the tombstone persist also fails -> still de-routed
# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------
# router lifespan: durable state errors are observable; auxiliary client cleanup is best effort
# --------------------------------------------------------------------------------------------


def test_exceptional_lifespan_exit_closes_usage_and_authorizer():
    from flash.serving.src.accounting.usage_outbox import OfflineUsageStore

    closed = []

    class Store(OfflineUsageStore):
        async def aclose(self):
            closed.append("usage")

    class Authorizer:
        async def __call__(self, _token, _adapter_id):
            return "org-1"

        async def aclose(self):
            closed.append("authorizer")

    app = build_durable_serving_app(
        FakePool(),
        AdapterRouter([_rec("qa/final")]),
        usage_store=Store(),
        chat_authorizer=Authorizer(),
    )

    async def fail_inside_lifespan():
        async with app.router.lifespan_context(app):
            raise RuntimeError("app failed")

    with pytest.raises(RuntimeError, match="app failed"):
        asyncio.run(fail_inside_lifespan())

    assert closed == ["usage", "authorizer"]


def test_shutdown_propagates_usage_store_close_failure():
    from flash.serving.src.accounting.usage_outbox import OfflineUsageStore, UsageOutboxError

    class Store(OfflineUsageStore):
        enabled = True

        async def aclose(self):
            raise UsageOutboxError("durable_close_failed")

    class Authorizer:
        def __init__(self):
            self.closed = False

        async def __call__(self, _token, _adapter_id):
            return "org-1"

        async def aclose(self):
            self.closed = True

    authorizer = Authorizer()
    app = build_durable_serving_app(
        FakePool(),
        AdapterRouter([_rec("qa/final")]),
        usage_store=Store(),
        chat_authorizer=authorizer,
    )
    with (
        pytest.raises(UsageOutboxError, match="durable_close_failed"),
        TestClient(app) as client,
    ):
        assert client.get("/healthz").status_code == 200
    assert authorizer.closed is True


def test_shutdown_swallows_authorizer_aclose_errors():
    from flash.serving.src.accounting.usage_outbox import OfflineUsageStore

    class Authorizer:
        def __init__(self):
            self.closed = False

        async def __call__(self, _token, _adapter_id):
            return "org-1"

        async def aclose(self):
            self.closed = True
            raise RuntimeError("client already closed")

    authorizer = Authorizer()
    app = build_durable_serving_app(
        FakePool(),
        AdapterRouter([_rec("qa/final")]),
        usage_store=OfflineUsageStore(),
        chat_authorizer=authorizer,
    )
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
    assert authorizer.closed is True
