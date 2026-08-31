"""Multi-base-model adapter routing: each adapter request reaches its base model's
engine (one GPU per base model), many adapters share a base model's engine, and
registration lands on the right engine. Offline — a fake engine pool stands in for the
per-base-model Modal GPU containers.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time

import pytest
from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.testclient import TestClient

from flash.serving.src.http.adapter_routes import remove_adapter
from flash.serving.src.http.context import ServingContext
from flash.serving.src.http.router import AdapterRouter
from flash.serving.src.http.router import build_offline_serving_app as build_serving_app
from flash.serving.src.http.router import build_serving_app as _metered_app
from flash.serving.src.io.schemas import AdapterRecord
from tests.serving.checkpoint_fixtures import (
    checkpoint_payload,
    checkpoint_record,
    checkpoint_registration_payload,
)
from tests.serving.conftest import attest


async def _allow(_token: str, _adapter_id: str, _scope: dict | None = None) -> str:
    """Permissive attributed authorizer for routing tests."""
    return "org-1"


def _serve(*args, **kwargs):
    """build_serving_app + TestClient wired for always-on chat auth: a permissive
    authorizer + a default Bearer header. Sends no X-Freesolo-Internal-Key, so the
    /adapters registration-auth tests still see 401-without-key / 200-with-key."""
    kwargs.setdefault("chat_authorizer", _allow)
    return TestClient(
        build_serving_app(*args, **kwargs),
        headers={"Authorization": "Bearer t", "X-Freesolo-Org-Id": "org-1"},
    )


QWEN = "Qwen/Qwen3.5-9B"
QWEN_35B = "Qwen/Qwen3.6-35B-A3B"


def _revision_id(run_id: str) -> str:
    return f"{run_id}/final"


def _rec(run_id: str, base_model: str, *, status: str = "ready") -> AdapterRecord:
    return checkpoint_record(run_id, base_model, status=status, thinking=True)


def _router_for(run_id: str, base_model: str, *, status: str = "ready") -> AdapterRouter:
    return AdapterRouter([_rec(run_id, base_model, status=status)])


def _adapter_payload(run_id: str, base_model: str = QWEN, **overrides: object) -> dict[str, object]:
    return checkpoint_payload(run_id, base_model, thinking=True, **overrides)


class FakePool:
    """Records which base-model engine each call was dispatched to."""

    def __init__(self) -> None:
        self.generated: list[tuple[str, str]] = []
        self.generated_records: list[AdapterRecord] = []
        self.registered: list[tuple[str, str]] = []
        self.unregistered: list[tuple[str, str]] = []
        self.template_kwargs: list = []  # chat_template_kwargs seen per generate call
        self.messages: list = []
        self.structured: list = []  # structured_outputs seen per generate call
        self.resolved_records: dict[str, AdapterRecord] = {}

    def _resolved_record(self, record: AdapterRecord) -> AdapterRecord:
        return self.resolved_records.get(record.adapter_id, record)

    def _active_checkpoint_ref(self, record: AdapterRecord) -> str:
        checkpoint = (record.checkpoint or "").strip()
        if checkpoint:
            return checkpoint
        subfolder = (record.subfolder or "").strip().strip("/")
        match = re.search(r"(?:^|/)checkpoints/(step-\d+)(?:/|$)", subfolder)
        if match:
            return f"{record.adapter_id}/{match.group(1)}"
        return record.adapter_id if subfolder else ""

    def _check_expected(self, record: AdapterRecord, expected_checkpoint: str | None) -> str:
        active = self._active_checkpoint_ref(record)
        if expected_checkpoint is not None and expected_checkpoint.strip() != active:
            expected = expected_checkpoint.strip()
            raise ValueError(
                "checkpoint mismatch: "
                f"adapter {record.adapter_id} is serving checkpoint "
                f"{active or '<none>'}, not the expected {expected or '<none>'}"
            )
        return active

    async def generate(
        self,
        base_model: str,
        payload,
        record,
        *,
        expected_checkpoint: str | None = None,
    ) -> dict:
        record = self._resolved_record(record)
        checkpoint = self._check_expected(record, expected_checkpoint)
        self.generated.append((base_model, payload.adapter_id))
        self.generated_records.append(record)
        self.template_kwargs.append(getattr(payload, "chat_template_kwargs", None))
        self.messages.append(getattr(payload, "messages", None))
        self.structured.append(getattr(payload, "structured_outputs", None))
        return attest(
            record,
            {
                # Snake_case, matching the real engine RPC contract (modal_app.py::_generate).
                "ok": True,
                "adapter_id": payload.adapter_id,
                "text": f"[{base_model}] reply",
                "finish_reason": "stop",
                "prompt_token_ids": [4, 5],
                "completion_token_ids": [1, 2, 3],
                "token_ids": [1, 2, 3],
                "inference_time_seconds": 0.01,
                "checkpoint": checkpoint,
            },
        )

    async def stream_generate(
        self,
        base_model: str,
        payload,
        record,
        *,
        expected_checkpoint: str | None = None,
    ):
        record = self._resolved_record(record)
        checkpoint = self._check_expected(record, expected_checkpoint)
        yield attest(record, {"type": "ready", "checkpoint": checkpoint})
        self.generated.append((base_model, payload.adapter_id))
        self.generated_records.append(record)
        self.messages.append(getattr(payload, "messages", None))
        self.structured.append(getattr(payload, "structured_outputs", None))
        yield {"type": "delta", "text": f"[{base_model}] "}
        yield {"type": "delta", "text": "reply"}
        yield {
            "type": "final",
            "finish_reason": "stop",
            "prompt_tokens": 2,
            "completion_tokens": 2,
            "inference_time_seconds": 0.01,
            "request_id": "req-stream",
            "checkpoint": checkpoint,
        }

    async def register(self, base_model: str, record: AdapterRecord) -> None:
        self.registered.append((base_model, record.adapter_id))

    async def unregister(
        self,
        base_model: str,
        org_id: str,
        adapter_id: str,
        expected_generation: str | None = None,
    ) -> None:
        del org_id, expected_generation
        self.unregistered.append((base_model, adapter_id))


@pytest.fixture
def app_setup():
    # two adapters share the 9b engine, while one adapter uses the separate 35b engine.
    revisions = [_rec("qa", QWEN), _rec("qb", QWEN), _rec("mc", QWEN_35B)]
    router = AdapterRouter(revisions)
    pool = FakePool()
    client = _serve(pool, router, internal_key="sekret")
    return client, pool, router


def test_unregister_safe_records_exact_gpu_cleanup_failure(capsys):
    class _FailingPool:
        async def unregister(self, base_model, org_id, adapter_id, expected_generation):
            raise RuntimeError(
                f"exact eviction failed for {base_model} {org_id} {adapter_id} "
                f"{expected_generation}"
            )

    context = object.__new__(ServingContext)
    context.pool = _FailingPool()

    asyncio.run(context.unregister_safe(QWEN, "org-1", "active/final", "generation-1"))

    assert (
        f"hosted adapter gpu cleanup failed for active/final on {QWEN}: "
        f"RuntimeError('exact eviction failed for {QWEN} org-1 active/final generation-1')"
        in capsys.readouterr().out
    )


def test_healthz_reports_one_gpu_per_base_model(app_setup):
    client, _, _ = app_setup
    body = client.get("/healthz").json()
    assert body["ok"] is True
    assert body["capabilities"] == [
        "permanent_checkpoint_identity",
        "thinking_structured_outputs_deferred_v1",
    ]
    assert body["base_models"] == [QWEN, QWEN_35B]  # sorted by model id
    assert body["gpus"] == 2  # two configured supported base-model engines
    assert body["gpu_by_model"] == {QWEN: "B200", QWEN_35B: "B200"}
    assert body["gpu_tiers"] == ["B200"]  # sorted(set(...)): both tiers share the card
    assert "configuredGpu" not in body  # the single-GPU field is gone (per-model now)
    assert body["adapters"] == 3


def test_healthz_reports_configured_deployment_identity() -> None:
    client = _serve(
        FakePool(),
        AdapterRouter([]),
        deployment_sha="abc123",
        deployment_id="456-2",
    )

    body = client.get("/healthz").json()
    assert body["deployment_sha"] == "abc123"
    assert body["deployment_id"] == "456-2"


class _UnhealthyUsageStore:
    """A store whose delivery worker has stopped, so nothing new can be settled."""

    enabled = True

    def assert_healthy(self) -> None:
        raise RuntimeError("usage_outbox_worker_stopped")

    async def start(self) -> None:
        return None

    async def capture(self, event) -> None:
        del event

    async def finalize(self, event) -> None:
        del event

    async def fail(self, event, code: str) -> None:
        del event, code

    def relinquish(self, request_id: str) -> None:
        del request_id

    async def snapshot(self):
        raise RuntimeError("usage_outbox_worker_stopped")

    async def recover_stale_in_progress(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def _metered_client(pool, router, store) -> TestClient:
    return TestClient(
        _metered_app(pool, router, usage_store=store, chat_authorizer=_allow),
        headers={"Authorization": "Bearer t", "X-Freesolo-Org-Id": "org-1"},
    )


def test_healthz_fails_when_accounting_cannot_settle() -> None:
    client = _metered_client(FakePool(), AdapterRouter([]), _UnhealthyUsageStore())

    response = client.get("/healthz")

    # a replica that cannot settle usage must leave rotation instead of taking chargeable traffic.
    assert response.status_code == 503
    body = response.json()
    assert body["ok"] is False
    assert body["accounting_ok"] is False


def test_healthz_reports_accounting_ok_when_settlement_is_live() -> None:
    client = _serve(FakePool(), AdapterRouter([]))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["accounting_ok"] is True


def test_generate_fails_closed_when_accounting_cannot_settle() -> None:
    # a non-thinking adapter, so the thinking-settlement guard cannot mask the accounting gate.
    router = AdapterRouter([checkpoint_record("qa", QWEN)])
    pool = FakePool()
    client = _metered_client(pool, router, _UnhealthyUsageStore())

    response = client.post("/generate", json={"adapter_id": "qa/final", "prompt": "hi"})

    assert response.status_code == 503
    assert response.json() == {"detail": "durable serving accounting unavailable"}
    assert pool.generated == []


def test_healthz_reports_unsupported_hydrated_base_models_without_routing_them():
    legacy = checkpoint_record("legacy", "unsupported/model", thinking=True)
    router = AdapterRouter([legacy])
    pool = FakePool()
    client = _serve(pool, router, internal_key="sekret")

    body = client.get("/healthz").json()
    assert body["ok"] is True
    assert body["base_models"] == ["unsupported/model"]
    assert body["unsupported_base_models"] == ["unsupported/model"]
    assert body["gpus"] == 0
    assert body["gpu_by_model"] == {}

    resp = client.post("/generate", json={"adapter_id": "legacy", "prompt": "hi"})
    assert resp.status_code == 404
    assert pool.generated == []

    cleanup = client.delete("/adapters/legacy", headers={"X-Freesolo-Internal-Key": "sekret"})
    assert cleanup.status_code == 404
    assert router.get("legacy/final", org_id="org-1") is not None


def test_chat_rejects_unknown_multimodal_content_block_before_dispatch(app_setup):
    client, pool, _ = app_setup
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("qa"),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "data:x"},
                        {"type": "image", "image": "data:image/png;base64,not-used"},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400
    assert "unsupported type" in response.json()["detail"]
    assert pool.generated == []


def test_chat_rejects_unknown_list_content_without_image_before_dispatch(app_setup):
    client, pool, _ = app_setup
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("qa"),
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "video", "video": "data:x"}],
                }
            ],
        },
    )
    assert response.status_code == 400
    assert "unsupported type" in response.json()["detail"]
    assert pool.generated == []


def test_generate_routes_to_the_adapters_base_model(app_setup):
    client, pool, _ = app_setup
    assert (
        client.post(
            "/generate", json={"adapter_id": _revision_id("qa"), "prompt": "hi"}
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/generate", json={"adapter_id": _revision_id("mc"), "prompt": "hi"}
        ).status_code
        == 200
    )
    # each adapter dispatches to its own active base-model engine.
    assert pool.generated == [(QWEN, _revision_id("qa")), (QWEN_35B, _revision_id("mc"))]


def test_chat_template_kwargs_forwarded_on_generate(app_setup):
    """Per-run thinking parity: chat_template_kwargs (e.g. enable_thinking=false) must reach the
    engine, not be dropped at the schema. Regression for the Qwen3.5-4B thinking-preamble bug."""
    client, pool, _ = app_setup
    resp = client.post(
        "/generate",
        json={
            "adapter_id": _revision_id("qa"),
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert resp.status_code == 200
    assert pool.template_kwargs[-1] == {"enable_thinking": False}


def test_chat_template_kwargs_forwarded_on_openai_chat(app_setup):
    """The OpenAI /v1/chat/completions path forwards chat_template_kwargs too (the eval / backend
    /api/sample proxy uses this endpoint)."""
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("mc"),
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert resp.status_code == 200
    assert pool.template_kwargs[-1] == {"enable_thinking": True}


def test_generate_without_chat_template_kwargs_is_none(app_setup):
    """Absent chat_template_kwargs stays None (template runs at its default) — no accidental key."""
    client, pool, _ = app_setup
    client.post("/generate", json={"adapter_id": _revision_id("qa"), "prompt": "hi"})
    assert pool.template_kwargs[-1] is None


# --- structured outputs (guided decoding) on the serving surface --------------------------------
# Every accepted request form must reach the engine as the SAME canonical spec (a dict of
# StructuredOutputsParams kwargs on payload.structured_outputs) — normalization happens once, in
# the GenerateRequest validator, so all entry points share it and 422 identically on bad specs.


_PERSON_SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}}


def test_thinking_logprobs_policy_runs_after_adapter_resolution(app_setup):
    client, pool, _ = app_setup
    context = client.app.state.serving_context
    original_resolve = context.lookup.resolve
    resolved = []

    async def tracked_resolve(adapter_id, **kwargs):
        result = await original_resolve(adapter_id, **kwargs)
        resolved.append(result[1].adapter_id)
        return result

    context.lookup.resolve = tracked_resolve
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("qa"),
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
        },
    )

    assert response.status_code == 422
    assert "thinking-enabled" in response.json()["detail"]
    assert resolved == [_revision_id("qa")]
    assert pool.generated == []


def test_tool_choice_none_is_inactive_on_unqualified_thinking_route() -> None:
    record = _rec("unqualified", QWEN_35B)
    pool = FakePool()
    client = _serve(pool, AdapterRouter([record]))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("unqualified"),
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools,
            "tool_choice": "none",
            "parallel_tool_calls": True,
        },
    )

    assert response.status_code == 200, response.text
    assert pool.generated == [(QWEN_35B, _revision_id("unqualified"))]


def test_base_model_false_thinking_override_allows_logprobs() -> None:
    record = AdapterRecord(
        adapter_id=QWEN,
        repo_id=QWEN,
        base_model=QWEN,
        serve_base_model=True,
        thinking=True,
        status="ready",
    )
    pool = FakePool()
    client = _serve(pool, AdapterRouter([record]))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": QWEN,
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
            "top_logprobs": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    assert response.status_code == 200, response.text
    assert pool.template_kwargs[-1]["enable_thinking"] is False


def test_immutable_adapter_ignores_false_thinking_override_for_logprobs(app_setup):
    client, pool, _ = app_setup
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("qa"),
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    assert response.status_code == 422
    assert "thinking-enabled" in response.json()["detail"]
    assert pool.generated == []


def test_unknown_adapter_does_not_reveal_thinking_logprobs_policy(app_setup):
    client, pool, _ = app_setup
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "missing",
            "messages": [{"role": "user", "content": "hi"}],
            "logprobs": True,
        },
    )

    assert response.status_code == 404
    assert "thinking" not in response.text.lower()
    assert pool.generated == []


def test_chat_rejects_unrelated_vllm_extensions(app_setup):
    """The canonical OpenAI grammar rejects unrelated vLLM extension fields."""
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("mc"),
            "messages": [{"role": "user", "content": "hi"}],
            "structured_outputs": {"choice": ["a", "b"]},
            "guided_regex": r"\d+",
        },
    )
    assert resp.status_code == 422
    assert not pool.structured


def test_chat_accepts_openai_response_format(app_setup):
    """OpenAI-SDK compatibility: response_format is honoured at /v1/chat/completions and
    translated to our canonical constraint (accepted at this endpoint only)."""
    client, pool, _ = app_setup
    for rf, expected in (
        ({"type": "json_object"}, {"json_object": True}),
        (
            {"type": "json_schema", "json_schema": {"name": "p", "schema": _PERSON_SCHEMA}},
            {"json": _PERSON_SCHEMA},
        ),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": _revision_id("mc"),
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": rf,
            },
        )
        assert resp.status_code == 200, resp.text
        assert pool.structured[-1] == expected


def test_chat_rejects_both_structured_output_fields(app_setup):
    """each field works alone, but supplying both is an invalid request."""
    client, pool, _ = app_setup
    base = {
        "model": _revision_id("mc"),
        "messages": [{"role": "user", "content": "hi"}],
    }

    extension = client.post(
        "/v1/chat/completions",
        json={**base, "structured_outputs": {"choice": ["a", "b"]}},
    )
    assert extension.status_code == 200
    assert pool.structured[-1] == {"choice": ["a", "b"]}

    standard = client.post(
        "/v1/chat/completions",
        json={**base, "response_format": {"type": "json_object"}},
    )
    assert standard.status_code == 200
    assert pool.structured[-1] == {"json_object": True}

    conflict = client.post(
        "/v1/chat/completions",
        json={
            **base,
            "structured_outputs": {"choice": ["a", "b"]},
            "response_format": {"type": "json_object"},
        },
    )
    assert conflict.status_code == 422
    assert conflict.json()["detail"] == "structured_outputs and response_format cannot both be set"
    assert len(pool.generated) == 2


def test_chat_response_format_json_schema_without_schema_is_422(app_setup):
    """A malformed response_format (json_schema with no schema) is a clean 422, not a 500."""
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("mc"),
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "p"}},
        },
    )
    assert resp.status_code == 422
    assert "response_format.json_schema is malformed" in resp.json()["detail"]
    assert pool.generated == []


def test_chat_response_format_unknown_type_is_422(app_setup):
    """An unknown response_format type is a clean 422, not silently reinterpreted as a schema."""
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("mc"),
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "image_url"},
        },
    )
    assert resp.status_code == 422
    assert "response_format type is not supported" in resp.json()["detail"]
    assert pool.generated == []


def test_guided_fields_are_rejected(app_setup):
    """vLLM guided_* request fields are rejected by the canonical grammar."""
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("mc"),
            "messages": [{"role": "user", "content": "hi"}],
            "guided_regex": r"\d+",
        },
    )
    assert resp.status_code == 422
    assert not pool.structured


def test_chat_without_structured_outputs_is_unconstrained(app_setup):
    """Freeform still works: a chat request with no structured-outputs field generates
    unconstrained (None reaches the engine)."""
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={"model": _revision_id("mc"), "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert pool.structured[-1] is None


def test_invalid_structured_outputs_is_422_with_detail(app_setup):
    client, pool, _ = app_setup
    two_constraints = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("mc"),
            "messages": [{"role": "user", "content": "hi"}],
            "structured_outputs": {"json": _PERSON_SCHEMA, "regex": r"\d+"},
        },
    )
    assert two_constraints.status_code == 422
    assert "exactly one constraint" in two_constraints.json()["detail"]
    unsupported = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("mc"),
            "messages": [{"role": "user", "content": "hi"}],
            "structured_outputs": {"grammar": 'root ::= "x"'},
        },
    )
    assert unsupported.status_code == 422
    assert "not supported" in unsupported.json()["detail"]
    # Neither bad spec ever reached the engine.
    assert pool.generated == []


def test_generate_passes_structured_outputs_through(app_setup):
    client, pool, _ = app_setup
    resp = client.post(
        "/generate",
        json={
            "adapter_id": _revision_id("qa"),
            "prompt": "hi",
            "structured_outputs": {"json": _PERSON_SCHEMA},
        },
    )
    assert resp.status_code == 200
    assert pool.structured[-1] == {"json": _PERSON_SCHEMA}


def test_per_adapter_generate_passes_structured_outputs_through(app_setup):
    # The untyped-dict variant builds a GenerateRequest from the body, so the field rides along.
    client, pool, _ = app_setup
    resp = client.post(
        f"/adapters/{_revision_id('qb')}/generate",
        json={"prompt": "hi", "structured_outputs": {"choice": ["a", "b"]}},
    )
    assert resp.status_code == 200
    assert pool.structured[-1] == {"choice": ["a", "b"]}


def test_absent_structured_outputs_stays_none(app_setup):
    """No spec anywhere -> None reaches the engine (which may then apply an adapter default)."""
    client, pool, _ = app_setup
    assert (
        client.post(
            "/generate", json={"adapter_id": _revision_id("qa"), "prompt": "hi"}
        ).status_code
        == 200
    )
    assert pool.structured[-1] is None


def test_streamed_chat_forwards_structured_outputs(app_setup):
    client, pool, _ = app_setup
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": _revision_id("mc"),
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "structured_outputs": {"json_object": True},
        },
    ) as resp:
        assert resp.status_code == 200
        resp.read()
    assert pool.structured[-1] == {"json_object": True}


def test_many_adapters_share_one_base_model_engine(app_setup):
    client, pool, _ = app_setup
    client.post("/generate", json={"adapter_id": _revision_id("qa"), "prompt": "x"})
    client.post("/generate", json={"adapter_id": _revision_id("qb"), "prompt": "y"})
    # Both Qwen adapters dispatched to the SAME base-model engine (one GPU, multi-LoRA).
    assert {bm for bm, _ in pool.generated} == {QWEN}
    assert {aid for _, aid in pool.generated} == {
        _revision_id("qa"),
        _revision_id("qb"),
    }


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/generate", {"adapter_id": _revision_id("qa"), "prompt": "hi"}),
        (f"/adapters/{_revision_id('qa')}/generate", {"prompt": "hi"}),
    ],
)
def test_raw_generate_responses_exclude_internal_fields(app_setup, path, payload):
    client, _, _ = app_setup
    body = client.post(path, json=payload).json()
    assert body["adapter_id"] == _revision_id("qa")
    assert body["finish_reason"] == "stop"
    assert body["token_ids"] == [1, 2, 3]
    assert body["text"] == f"[{QWEN}] reply"
    assert "prompt_token_ids" not in body
    assert "completion_token_ids" not in body
    # none of the old camelcase spellings survive anywhere in the response.
    for camel in ("adapterId", "finishReason", "tokenIds", "inferenceTimeSeconds", "requestId"):
        assert camel not in body


@pytest.mark.parametrize(
    "field",
    [
        "n",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "top_logprobs",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    ],
)
@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/generate", {"adapter_id": _revision_id("qa"), "prompt": "hi"}),
        (f"/adapters/{_revision_id('qa')}/generate", {"prompt": "hi"}),
    ],
)
def test_raw_generate_routes_reject_openai_only_sampling_fields(
    app_setup, field: str, path: str, payload: dict[str, object]
) -> None:
    client, pool, _ = app_setup
    value: object = True if field == "logprobs" else 1

    response = client.post(path, json={**payload, field: value})

    assert response.status_code == 422
    assert pool.generated == []


def test_openai_chat_completions_routes_and_shapes(app_setup):
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={"model": _revision_id("mc"), "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == _revision_id("mc")
    assert body["choices"][0]["message"]["content"] == f"[{QWEN_35B}] reply"
    assert pool.generated == [(QWEN_35B, _revision_id("mc"))]


def test_external_openai_chat_forwards_system_prompts(app_setup):
    client, pool, _ = app_setup
    messages = [
        {"role": "system", "content": "stay terse"},
        {"role": "user", "content": "hi"},
    ]
    resp = client.post(
        "/v1/chat/completions",
        json={"model": _revision_id("mc"), "messages": messages},
    )

    assert resp.status_code == 200, resp.text
    assert pool.generated == [(QWEN_35B, _revision_id("mc"))]
    assert pool.messages[-1] == messages


def test_internal_openai_chat_can_send_system_prompts(app_setup):
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        headers={"X-Freesolo-Internal-Key": "sekret"},
        json={
            "model": _revision_id("mc"),
            "messages": [
                {"role": "system", "content": "platform prompt"},
                {"role": "user", "content": "hi"},
            ],
        },
    )

    assert resp.status_code == 200, resp.text
    assert pool.generated == [(QWEN_35B, _revision_id("mc"))]


def test_external_generate_forwards_system_prompts(app_setup):
    client, pool, _ = app_setup
    messages = [
        {"role": "system", "content": "stay terse"},
        {"role": "user", "content": "hi"},
    ]
    resp = client.post(
        "/generate",
        json={"adapter_id": _revision_id("qa"), "messages": messages},
    )

    assert resp.status_code == 200, resp.text
    assert pool.generated == [(QWEN, _revision_id("qa"))]
    assert pool.messages[-1] == messages


def test_openai_chat_completions_streams_sse_chunks(app_setup):
    client, pool, _ = app_setup
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": _revision_id("mc"),
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["x-freesolo-lora-request-adapter"] == _revision_id("mc")
        text = resp.read().decode("utf-8")

    assert '"delta":{"role":"assistant"}' in text
    assert '"delta":{"content":"[' in text
    assert f"{QWEN_35B}] " in text
    assert '"delta":{"content":"reply"}' in text
    assert '"finish_reason":"stop"' in text
    assert "data: [DONE]" in text
    assert pool.generated == [(QWEN_35B, _revision_id("mc"))]


@pytest.mark.parametrize("attestation", [None, "wrong@final." + "b" * 40])
def test_openai_stream_rejects_unattested_revision_and_closes_engine(attestation):
    revision = _rec("attested", QWEN)

    class UnattestedPool(FakePool):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        async def stream_generate(
            self,
            base_model: str,
            payload,
            record,
            *,
            expected_checkpoint: str | None = None,
        ):
            del base_model, payload, expected_checkpoint
            event = {"type": "ready", "checkpoint": record.checkpoint}
            if attestation is not None:
                event["lora_request_adapter"] = attestation
            try:
                yield event
                await asyncio.Event().wait()
            finally:
                self.closed = True

    pool = UnattestedPool()
    client = _serve(pool, AdapterRouter([revision]))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": revision.adapter_id,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )

    assert response.status_code == 502
    assert "attest" in response.json()["detail"]
    assert "x-freesolo-adapter-revision" not in response.headers
    assert "x-freesolo-checkpoint" not in response.headers
    assert "x-freesolo-hf-revision" not in response.headers
    assert pool.closed


def test_openai_chat_completions_stream_can_include_usage(app_setup):
    client, pool, _ = app_setup
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": _revision_id("mc"),
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as resp:
        assert resp.status_code == 200
        text = resp.read().decode("utf-8")

    assert '"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}' in text
    assert pool.generated == [(QWEN_35B, _revision_id("mc"))]


def test_openai_chat_stream_sets_anti_buffering_headers(app_setup):
    # X-Accel-Buffering: no tells Nginx not to buffer the SSE response; without it each token
    # accumulates in Nginx's output buffer and the caller sees high TTFT for small completions.
    client, _, _ = app_setup
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": _revision_id("mc"),
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        headers = {k.lower(): v for k, v in resp.headers.items()}
    assert headers.get("x-accel-buffering") == "no"
    assert headers.get("cache-control") == "no-cache"


@pytest.mark.parametrize("stream", [1, "true"])
def test_openai_chat_rejects_non_boolean_stream_after_authorization(stream):
    authorizations: list[str] = []

    async def _authorize(_token: str, adapter_id: str, _scope: dict | None = None) -> str:
        authorizations.append(adapter_id)
        return "org-1"

    pool = FakePool()
    client = _serve(pool, _router_for("qa", QWEN), chat_authorizer=_authorize)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("qa"),
            "messages": [{"role": "user", "content": "hi"}],
            "stream": stream,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "stream must be a boolean"
    assert authorizations == [_revision_id("qa")]
    assert pool.generated == []


def test_openai_chat_completions_bad_payload_is_422_not_500(app_setup):
    # A non-numeric top_p (and likewise a malformed messages shape) makes the in-handler
    # GenerateRequest validation raise a Pydantic ValidationError. It must surface as a 4xx
    # client error, not escape the handler as a 500.
    client, pool, _ = app_setup
    bad_top_p = client.post(
        "/v1/chat/completions",
        json={
            "model": _revision_id("mc"),
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": "nope",
        },
    )
    assert bad_top_p.status_code == 422
    bad_messages = client.post(
        "/v1/chat/completions",
        json={"model": _revision_id("mc"), "messages": "not-a-list"},
    )
    assert bad_messages.status_code == 422
    # Never reached the engine.
    assert pool.generated == []


def test_per_adapter_generate_bad_payload_is_422_not_500(app_setup):
    client, pool, _ = app_setup
    resp = client.post(f"/adapters/{_revision_id('qb')}/generate", json={"top_p": "nope"})
    assert resp.status_code == 422
    assert pool.generated == []


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/generate", {"adapter_id": _revision_id("qb"), "prompt": "hi", "max_tokens": 0}),
        (f"/adapters/{_revision_id('qb')}/generate", {"prompt": "hi", "temperature": -0.1}),
        (
            "/v1/chat/completions",
            {
                "model": _revision_id("qb"),
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": 0,
            },
        ),
    ],
)
def test_inference_routes_reject_invalid_sampling_before_engine_dispatch(
    app_setup, path: str, payload: dict[str, object]
) -> None:
    client, pool, _ = app_setup

    response = client.post(path, json=payload)

    assert response.status_code == 422
    assert pool.generated == []


def test_per_adapter_generate_endpoint_routes(app_setup):
    client, pool, _ = app_setup
    assert (
        client.post(f"/adapters/{_revision_id('qb')}/generate", json={"prompt": "hi"}).status_code
        == 200
    )
    assert pool.generated == [(QWEN, _revision_id("qb"))]


def test_unknown_adapter_is_404_not_misrouted(app_setup):
    client, pool, _ = app_setup
    assert client.post("/generate", json={"adapter_id": "nope", "prompt": "hi"}).status_code == 404
    assert (
        client.post("/v1/chat/completions", json={"model": "nope", "messages": []}).status_code
        == 422
    )
    assert pool.generated == []


def test_engine_valueerror_is_400_when_adapter_still_ready(app_setup):
    # An engine ValueError while the router still has the adapter ready maps to 400, not 404.
    _, pool, router = app_setup

    async def boom(base_model, payload, record, **_kwargs):
        raise ValueError("invalid generated request")

    pool.generate = boom
    client = _serve(pool, router, internal_key="sekret")
    assert (
        client.post(
            "/generate", json={"adapter_id": _revision_id("qa"), "prompt": "hi"}
        ).status_code
        == 400
    )


def test_raced_undeploy_engine_valueerror_is_404(app_setup):
    # An adapter passes _lookup, then races out of the registry; the engine reports it as an
    # unknown adapter (ValueError). The router must surface 404 (same as _lookup), not 400.
    _, pool, router = app_setup

    async def vanish(base_model, payload, record, **_kwargs):
        router.remove(payload.adapter_id)  # concurrent undeploy after _lookup passed
        raise ValueError(f"Unknown adapter id on {base_model}: {payload.adapter_id}")

    pool.generate = vanish
    client = _serve(pool, router, internal_key="sekret")
    assert (
        client.post(
            "/generate", json={"adapter_id": _revision_id("qa"), "prompt": "hi"}
        ).status_code
        == 404
    )


def test_list_adapters_requires_internal_key(app_setup):
    client, _, _ = app_setup
    # Anonymous listing leaks repo_id/url (HF namespaces + adapter->tenant mapping), so it must be
    # gated the same as register/teardown.
    assert client.get("/adapters").status_code == 401
    ok = client.get("/adapters", headers={"X-Freesolo-Internal-Key": "sekret"})
    assert ok.status_code == 200
    assert {a["adapter_id"] for a in ok.json()["adapters"]} == {
        _revision_id("qa"),
        _revision_id("qb"),
        _revision_id("mc"),
    }


def test_list_adapters_refreshes_authoritative_persisted_state():
    persisted_revision = _rec("fresh", QWEN)
    reloads = 0

    def _reload():
        nonlocal reloads
        reloads += 1
        return [persisted_revision]

    router = _router_for("stale", QWEN)
    client = _serve(
        FakePool(),
        router,
        internal_key="sekret",
        reload_records=_reload,
    )

    assert client.get("/adapters").status_code == 401
    assert reloads == 0

    response = client.get(
        "/adapters",
        headers={"X-Freesolo-Internal-Key": "sekret"},
    )

    assert response.status_code == 200
    assert reloads == 1
    records = {record["adapter_id"]: record for record in response.json()["adapters"]}
    assert set(records) == {_revision_id("fresh")}
    assert not router.has("stale/final", org_id="org-1")


def test_adapters_fail_closed_when_no_internal_key_configured():
    # A serving app built without an internal key must NOT serve an open /adapters control plane:
    # register/teardown fail closed (503) rather than allowing unauthenticated mutation.
    router = _router_for("qa", QWEN)
    client = _serve(FakePool(), router)  # no internal_key
    rec = checkpoint_registration_payload("qz", QWEN, thinking=True)
    no_key = client.post("/adapters", json=rec)
    assert no_key.status_code == 503
    # Even presenting a key can't open it when none is configured (no key to compare against).
    with_key = client.post("/adapters", json=rec, headers={"X-Freesolo-Internal-Key": "anything"})
    assert with_key.status_code == 503
    teardown = client.delete(f"/adapters/{_revision_id('qa')}")
    assert teardown.status_code == 503
    resolved = router.resolve(_revision_id("qa"), org_id="org-1")
    assert resolved is not None
    assert resolved[1].base_model == QWEN


def test_base_model_delete_is_rejected_rather_than_faked() -> None:
    """A base model is not a deployed adapter, so DELETE must 404 instead of reporting success.

    `_base_model_records()` seeds these rows in memory and every replica re-adds them on each
    reload, so there is no durable row to remove. Removing it from one replica's memory and
    returning ok=True told the caller a teardown happened that the next reload silently undid --
    and evicted the shared engine's base weights out from under every other tenant on the way.
    """
    record = AdapterRecord(
        adapter_id=QWEN,
        repo_id=QWEN,
        base_model=QWEN,
        serve_base_model=True,
        thinking=True,
        status="ready",
    )
    router = AdapterRouter([record])
    pool = FakePool()
    app = build_serving_app(pool, router, internal_key="sekret", chat_authorizer=_allow)
    # call the handler directly rather than through the client: the point is that the response is
    # produced BEFORE the gpu unregister runs, which is only observable by holding the
    # BackgroundTasks and running it afterwards.
    request = Request(
        {
            "type": "http",
            "method": "DELETE",
            "path": f"/adapters/{QWEN}",
            "headers": [
                (b"x-freesolo-internal-key", b"sekret"),
                (b"x-freesolo-org-id", b"org-1"),
            ],
            "query_string": b"",
            "app": app,
        }
    )
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            remove_adapter(
                adapter_id=QWEN,
                request=request,
                background_tasks=background_tasks,
            )
        )

    assert excinfo.value.status_code == 404
    # nothing was mutated and no gpu eviction was scheduled: a rejected teardown must not evict
    # the base weights every other tenant on this engine is serving from.
    assert pool.unregistered == []
    assert background_tasks.tasks == []
    assert router.get(QWEN) is record


class _MeteringPool(FakePool):
    """A pool whose generate() returns token counts (what the real engine reports)."""

    async def generate(
        self,
        base_model: str,
        payload,
        record,
        *,
        expected_checkpoint: str | None = None,
    ) -> dict:
        record = self._resolved_record(record)
        checkpoint = self._check_expected(record, expected_checkpoint)
        self.generated.append((base_model, payload.adapter_id))
        return attest(
            record,
            {
                "ok": True,
                "adapter_id": payload.adapter_id,
                "text": f"[{base_model}] reply",
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "cached_tokens_reported": False,
                "inference_time_seconds": 0.25,
                "queue_wait_seconds": 0.025,
                "replica_in_flight_requests_at_admission": 2,
                "replica_boot_duration_seconds": 30.0,
                "replica_freshly_booted": True,
                "engine_replica_id": "replica-7",
                "checkpoint": checkpoint,
            },
        )


def test_openai_chat_completions_includes_usage_when_engine_reports_counts():
    router = _router_for("qa", QWEN)
    client = _serve(_MeteringPool(), router)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": _revision_id("qa"), "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 200
    assert resp.json()["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


def test_generate_response_strips_internal_cache_attribution():
    router = _router_for("qa", QWEN)
    client = _serve(_MeteringPool(), router)

    body = client.post("/generate", json={"adapter_id": _revision_id("qa"), "prompt": "hi"}).json()

    assert "cached_tokens_reported" not in body
    assert "engine_replica_id" not in body
    assert "queue_wait_seconds" not in body
    assert "replica_in_flight_requests_at_admission" not in body
    assert "replica_boot_duration_seconds" not in body
    assert "replica_freshly_booted" not in body


class _CachedMeteringPool(FakePool):
    """A pool whose generate()/stream report a prefix-cache hit (cached_tokens)."""

    async def generate(
        self,
        base_model: str,
        payload,
        record,
        *,
        expected_checkpoint: str | None = None,
    ) -> dict:
        record = self._resolved_record(record)
        checkpoint = self._check_expected(record, expected_checkpoint)
        self.generated.append((base_model, payload.adapter_id))
        return attest(
            record,
            {
                "ok": True,
                "adapter_id": payload.adapter_id,
                "text": f"[{base_model}] reply",
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "cached_tokens": 6,  # 6 of the 10 prompt tokens were served from the prefix cache
                "cached_tokens_reported": True,
                "inference_time_seconds": 0.25,
                "engine_replica_id": "replica-cached",
                "checkpoint": checkpoint,
            },
        )

    async def stream_generate(
        self,
        base_model: str,
        payload,
        record,
        *,
        expected_checkpoint: str | None = None,
    ):
        record = self._resolved_record(record)
        checkpoint = self._check_expected(record, expected_checkpoint)
        yield attest(record, {"type": "ready", "checkpoint": checkpoint})
        self.generated.append((base_model, payload.adapter_id))
        yield {"type": "delta", "text": "hi"}
        yield {
            "type": "final",
            "finish_reason": "stop",
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "cached_tokens": 6,
            "cached_tokens_reported": True,
            "inference_time_seconds": 0.25,
            "engine_replica_id": "replica-cached",
            "request_id": "req-cached",
            "checkpoint": checkpoint,
        }


def test_openai_usage_exposes_cached_tokens_in_prompt_details():
    """The client-facing OpenAI usage object surfaces cached tokens (prompt_tokens_details)."""
    router = _router_for("qa", QWEN)
    client = _serve(_CachedMeteringPool(), router)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": _revision_id("qa"), "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    usage = resp.json()["usage"]
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 3
    assert usage["total_tokens"] == 13
    assert usage["prompt_tokens_details"] == {"cached_tokens": 6}


def test_openai_usage_omits_prompt_details_when_no_cache_hit():
    """No cache hit -> no prompt_tokens_details block (matches the OpenAI shape for 0 cached)."""
    router = _router_for("qa", QWEN)
    client = _serve(_MeteringPool(), router)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": _revision_id("qa"), "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert "prompt_tokens_details" not in resp.json()["usage"]


def test_disabled_adapter_is_not_routable():
    router = _router_for("off", QWEN, status="disabled")
    assert router.resolve(_revision_id("off"), org_id="org-1") is None
    assert router.base_models() == []  # no ready adapters -> no GPU


def test_undeploy_unknown_checkpoint_is_idempotent(app_setup):
    client, pool, _ = app_setup
    resp = client.delete(
        "/adapters/nope/final",
        headers={"X-Freesolo-Internal-Key": "sekret", "X-Freesolo-Org-Id": "org-1"},
    )
    assert resp.status_code == 200
    assert resp.json()["checkpoint_id"] == "nope/final"
    assert resp.json()["disabled_checkpoints"] == []
    assert pool.unregistered == []


def test_generate_forwards_record_for_lazy_engine_load(app_setup):
    # The engine container may not have seen the adapter (registration can land on a
    # different / cold container), so the full record is forwarded on the generate path.
    client, pool, _router = app_setup
    captured: list = []

    async def _capture(base_model, payload, record, **_kwargs):
        captured.append((base_model, record.adapter_id, record.repo_id))
        return {
            "ok": True,
            "adapter_id": payload.adapter_id,
            "text": "x",
            "finish_reason": "stop",
            "token_ids": [],
            "inference_time_seconds": 0.0,
            # a checkpoint request is refused unless the engine attests exact identity
            "lora_request_adapter": record.adapter_id,
            "checkpoint": record.adapter_id,
        }

    pool.generate = _capture
    assert (
        client.post(
            "/generate", json={"adapter_id": _revision_id("qa"), "prompt": "hi"}
        ).status_code
        == 200
    )
    assert captured == [(QWEN, _revision_id("qa"), "org/qa")]


def test_generate_miss_reloads_from_shared_storage():
    # A router container that hasn't seen a just-registered adapter reloads the persisted
    # table once on a miss before 404-ing.
    reloaded = {"count": 0}

    def _reload():
        reloaded["count"] += 1
        revision = _rec("late", QWEN)
        return [revision]

    pool = FakePool()
    router = AdapterRouter([])
    client = _serve(pool, router, reload_records=_reload)
    # Unknown locally -> reload finds it -> 200, and it's now cached.
    assert (
        client.post(
            "/generate", json={"adapter_id": _revision_id("late"), "prompt": "hi"}
        ).status_code
        == 200
    )
    assert reloaded["count"] == 1
    assert (
        client.post(
            "/generate", json={"adapter_id": _revision_id("late"), "prompt": "hi"}
        ).status_code
        == 200
    )
    assert reloaded["count"] == 1  # permanent checkpoint hits use the hydrated record
    # Still-unknown after reload -> 404.
    assert client.post("/generate", json={"adapter_id": "ghost", "prompt": "hi"}).status_code == 404


def test_hit_refresh_failure_serves_cached_adapter():
    # A transient shared-storage failure on the TTL hit-refresh must NOT fail a request we can
    # still serve from the cached ready record; only a genuine miss should hard-fail.
    state = {"fail": False}

    def _reload():
        if state["fail"]:
            raise RuntimeError("supabase 503")
        revision = _rec("qa", QWEN)
        return [revision]

    pool = FakePool()
    router = _router_for("qa", QWEN)
    client = _serve(pool, router, reload_records=_reload, reload_interval_seconds=0.0)
    # Warm the cache through the mutable alias, then exercise the immutable revision. Immutable
    # records keep the background-refresh fallback because their target cannot drift.
    assert (
        client.post("/generate", json={"adapter_id": _revision_id("qa"), "prompt": "x"}).status_code
        == 200
    )
    state["fail"] = True
    revision_id = _revision_id("qa")
    assert (
        client.post("/generate", json={"adapter_id": revision_id, "prompt": "x"}).status_code == 200
    )
    assert pool.generated[-1] == (QWEN, revision_id)


def test_miss_refresh_failure_propagates():
    # On a genuine miss (nothing cached) a failed reload has nothing to fall back to -> error,
    # not a silent 404 that masks the storage outage.
    def _reload():
        raise RuntimeError("supabase 503")

    pool = FakePool()
    app = build_serving_app(pool, AdapterRouter([]), reload_records=_reload, chat_authorizer=_allow)
    # raise_server_exceptions=False so we observe the 500 response rather than re-raising.
    client = TestClient(app, raise_server_exceptions=False, headers={"Authorization": "Bearer t"})
    resp = client.post("/generate", json={"adapter_id": "ghost", "prompt": "x"})
    assert resp.status_code == 500


def test_concurrent_misses_hydrate_in_order_without_stampeding():
    """Two concurrent misses must not hydrate out of order or stampede shared storage.

    `reload()` suspends on `to_thread`, so without serialization a slow first fetch can land AFTER
    a fast second one and overwrite fresher records with staler ones -- then stamp a newer
    timestamp over the result, hiding the regression until the next TTL.
    """
    import asyncio

    from flash.serving.src.store.lookup import AdapterLookup

    revision = _rec("late", QWEN)
    fresh = [revision]
    # snapshot the expectation BEFORE production can touch these objects. the router stores the
    # instances it is handed, so comparing against the originals would compare a record with
    # itself: code that corrupts them after hydrate moves both sides of the assertion together
    # and passes while the caller receives the corruption.
    expected = (revision.model_copy(deep=True), revision.model_copy(deep=True))
    calls = {"count": 0}
    hydrated: list[int] = []

    def _reload():
        calls["count"] += 1
        # the FIRST caller is the slow one; a lock-free reload lets it hydrate last.
        if calls["count"] == 1:
            time.sleep(0.05)
            return []
        return fresh

    router = AdapterRouter([])
    original_hydrate = router.hydrate

    def _tracking_hydrate(records):
        hydrated.append(len(records))
        return original_hydrate(records)

    router.hydrate = _tracking_hydrate  # type: ignore[method-assign]
    lookup = AdapterLookup(router, _reload, reload_interval_seconds=30.0)

    async def _both():
        return await asyncio.gather(
            lookup.resolve(
                _revision_id("late"),
                org_id="org-1",
                require_supported_base_model=False,
            ),
            lookup.resolve(
                _revision_id("late"),
                org_id="org-1",
                require_supported_base_model=False,
            ),
            return_exceptions=True,
        )

    results = asyncio.run(_both())

    # bounded, not necessarily 1. the second caller's miss began AFTER the first fetch had already
    # snapshotted storage, so that snapshot cannot answer it -- an adapter committed in between
    # would be invisible. what must not happen is a fetch per waiter: the count stays at one
    # in-flight generation plus one follow-up no matter how many callers pile up.
    assert calls["count"] <= 2, f"{calls['count']} fetches; misses stampeded shared storage"
    # the empty fetch must not land after the fresh one. asserting the ORDER, not a fixed list:
    # requiring `[0]` would mean the later miss reused the earlier snapshot, which is the defect.
    assert hydrated == sorted(hydrated), f"a stale fetch hydrated after a fresher one: {hydrated}"
    # at least one caller must SEE the follow-up fetch. `any`, not `all`: the first caller
    # legitimately gets the empty snapshot and 404s, because its own fetch began before the adapter
    # was committed. requiring both would encode a false contract.
    resolved = [result for result in results if not isinstance(result, HTTPException)]
    assert resolved, "the follow-up fetch ran but no caller could resolve the adapter it read"
    # and every successful resolution must equal the immutable snapshot -- not merely share an
    # adapter id, and not merely be whatever the router currently holds.
    assert all(result == expected for result in resolved), (
        f"a caller resolved records that are not the ones storage returned: {resolved}"
    )


def test_reload_stampede_stays_bounded_as_callers_pile_up():
    """The coalescing must bound fetches by generation, not by caller count."""
    import asyncio

    from flash.serving.src.store.lookup import AdapterLookup

    calls = {"count": 0}

    def _reload():
        calls["count"] += 1
        return []

    lookup = AdapterLookup(AdapterRouter([]), _reload, reload_interval_seconds=30.0)

    async def _many():
        await asyncio.gather(*(lookup.reload() for _ in range(50)))

    asyncio.run(_many())

    assert calls["count"] <= 2, f"50 concurrent misses caused {calls['count']} fetches"


def test_a_reload_that_snapshotted_first_cannot_answer_a_later_miss():
    """Coalescing must compare against when the in-flight fetch STARTED, not when it finished.

    A reload can snapshot storage, stall, and complete after a second request begins. Comparing
    against its completion time let that stale snapshot satisfy the later request, so an adapter
    committed before that request arrived stayed invisible and the caller got a 404.
    """
    import asyncio

    from flash.serving.src.store.lookup import AdapterLookup

    revision = _rec("committed-late", QWEN)
    # snapshotted before production sees the records, for the same reason as above: the router
    # keeps the instances it is handed, so the originals cannot serve as an expectation.
    expected = (revision.model_copy(deep=True), revision.model_copy(deep=True))
    storage: list[AdapterRecord] = []
    snapshotted = threading.Event()
    release = threading.Event()
    calls = {"count": 0}

    def _reload():
        calls["count"] += 1
        snapshot = list(storage)
        if calls["count"] == 1:
            snapshotted.set()
            release.wait(timeout=5)
        return snapshot

    router = AdapterRouter([])
    lookup = AdapterLookup(router, _reload, reload_interval_seconds=30.0)

    async def _sequence():
        first = asyncio.create_task(lookup.reload())
        await asyncio.to_thread(snapshotted.wait, 5)
        # committed AFTER the first fetch snapshotted, BEFORE the second miss begins.
        storage.extend([revision])
        second = asyncio.create_task(lookup.reload())
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(first, second)

    asyncio.run(_sequence())

    assert calls["count"] == 2, "the later miss reused a snapshot taken before it began"
    # and the second fetch's records must actually land, as themselves. the fetch count only proves
    # storage was re-read; the point of re-reading it is that the adapter committed in between
    # becomes resolvable as exactly what was committed.
    assert router.resolve("committed-late/final", org_id="org-1") == expected, (
        "storage was re-read but did not resolve to the exact records it returned"
    )
