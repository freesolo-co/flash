"""Multi-base-model adapter routing: each adapter request reaches its base model's
engine (one GPU per base model), many adapters share a base model's engine, and
registration lands on the right engine. Offline — a fake engine pool stands in for the
per-base-model Modal GPU containers.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time

import pytest
from fastapi import BackgroundTasks, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse

from flash.serving.src.adapter_routes import remove_adapter
from flash.serving.src.router import AdapterRouter, build_serving_app
from flash.serving.src.schemas import AdapterRecord
from flash.serving.src.serving_io import _sse
from flash.serving.src.streaming import openai_chat_stream
from tests.serving.conftest import attest


async def _allow(_token: str, _adapter_id: str) -> None:
    """Permissive chat authorizer for routing tests (auth is always enforced; these
    tests exercise routing/metering, not auth)."""
    return


def _serve(*args, **kwargs):
    """build_serving_app + TestClient wired for always-on chat auth: a permissive
    authorizer + a default Bearer header. Sends no X-Freesolo-Internal-Key, so the
    /adapters registration-auth tests still see 401-without-key / 200-with-key."""
    kwargs.setdefault("chat_authorizer", _allow)
    return TestClient(build_serving_app(*args, **kwargs), headers={"Authorization": "Bearer t"})


QWEN = "Qwen/Qwen3.5-0.8B"
QWEN_2B = "Qwen/Qwen3.5-2B"


def _revision_id(run_id: str) -> str:
    return f"{run_id}@final.{hashlib.sha1(run_id.encode()).hexdigest()}"


def _rec(run_id: str, base_model: str, *, status: str = "ready") -> AdapterRecord:
    sha = hashlib.sha1(run_id.encode()).hexdigest()
    return AdapterRecord.model_validate(
        {
            "adapter_id": _revision_id(run_id),
            "repo_id": f"org/{run_id}",
            "org_id": "org-1",
            "base_model": base_model,
            "checkpoint": run_id,
            "status": status,
            "thinking": True,
            "created_at": "2026-07-14T00:00:00+00:00",
            "updated_at": "2026-07-14T00:00:01+00:00",
            "metadata": {
                "record_type": "revision",
                "run_id": run_id,
                "checkpoint_step": None,
                "hf_revision": sha,
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


def _router_for(run_id: str, base_model: str, *, status: str = "ready") -> AdapterRouter:
    revision = _rec(run_id, base_model, status=status)
    return AdapterRouter([revision, _alias(revision)])


def _adapter_payload(run_id: str, base_model: str = QWEN, **overrides: object) -> dict[str, object]:
    sha = hashlib.sha1(run_id.encode()).hexdigest()
    payload: dict[str, object] = {
        "adapter_id": _revision_id(run_id),
        "repo_id": f"org/{run_id}",
        "base_model": base_model,
        "org_id": "org-1",
        "checkpoint": run_id,
        "thinking": True,
        "metadata": {
            "record_type": "revision",
            "run_id": run_id,
            "checkpoint_step": None,
            "hf_revision": sha,
        },
    }
    payload.update(overrides)
    return payload


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
        yield {"type": "ready", "checkpoint": checkpoint}
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
        adapter_id: str,
        expected_generation: str | None = None,
    ) -> None:
        del expected_generation
        self.unregistered.append((base_model, adapter_id))


@pytest.fixture
def app_setup():
    # 2 adapters on the 0.8B base (share one GPU), 1 on the 2B base (its own GPU).
    revisions = [_rec("qa", QWEN), _rec("qb", QWEN), _rec("mc", QWEN_2B)]
    router = AdapterRouter([*revisions, *(_alias(revision) for revision in revisions)])
    pool = FakePool()
    client = _serve(pool, router, internal_key="sekret")
    return client, pool, router


def test_healthz_reports_one_gpu_per_base_model(app_setup):
    client, _, _ = app_setup
    body = client.get("/healthz").json()
    assert body["ok"] is True
    assert body["capabilities"] == [
        "immutable_adapter_revisions",
        "alias_compare_and_swap",
        "revision_provenance",
        "thinking_structured_outputs_deferred_v1",
    ]
    assert body["base_models"] == [QWEN, QWEN_2B]  # sorted by model id
    assert body["gpus"] == 2  # two configured supported base-model engines
    # Per-model GPU tier (replaces the misleading single configuredGpu): both test models are small
    # so they map to the cheap FP8-capable L4; gpu_tiers is the distinct set actually in use.
    assert body["gpu_by_model"] == {QWEN: "L4", QWEN_2B: "L4"}
    assert body["gpu_tiers"] == ["L4"]
    assert "configuredGpu" not in body  # the single-GPU field is gone (per-model now)
    assert body["adapters"] == 6


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


def test_healthz_reports_unsupported_hydrated_base_models_without_routing_them():
    legacy = AdapterRecord.model_validate(
        {
            "adapter_id": "legacy",
            "repo_id": "org/legacy",
            "base_model": "openai/gpt-oss-20b",
            "thinking": True,
        }
    )
    router = AdapterRouter([legacy])
    pool = FakePool()
    client = _serve(pool, router, internal_key="sekret")

    body = client.get("/healthz").json()
    assert body["ok"] is True
    assert body["base_models"] == []
    assert "unsupported_base_models" not in body
    assert body["gpus"] == 0
    assert body["gpu_by_model"] == {}

    resp = client.post("/generate", json={"adapter_id": "legacy", "prompt": "hi"})
    assert resp.status_code == 404
    assert pool.generated == []

    cleanup = client.delete("/adapters/legacy", headers={"X-Freesolo-Internal-Key": "sekret"})
    assert cleanup.status_code == 404
    assert router.get("legacy") is not None


def test_chat_rejects_unknown_multimodal_content_block_before_dispatch(app_setup):
    client, pool, _ = app_setup
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qa",
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
            "model": "qa",
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
    assert client.post("/generate", json={"adapter_id": "qa", "prompt": "hi"}).status_code == 200
    assert client.post("/generate", json={"adapter_id": "mc", "prompt": "hi"}).status_code == 200
    # qa -> 0.8B engine, mc -> 2B engine.
    assert pool.generated == [(QWEN, _revision_id("qa")), (QWEN_2B, _revision_id("mc"))]


def test_chat_template_kwargs_forwarded_on_generate(app_setup):
    """Per-run thinking parity: chat_template_kwargs (e.g. enable_thinking=false) must reach the
    engine, not be dropped at the schema. Regression for the Qwen3.5-4B thinking-preamble bug."""
    client, pool, _ = app_setup
    resp = client.post(
        "/generate",
        json={
            "adapter_id": "qa",
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
            "model": "mc",
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert resp.status_code == 200
    assert pool.template_kwargs[-1] == {"enable_thinking": False}


def test_generate_without_chat_template_kwargs_is_none(app_setup):
    """Absent chat_template_kwargs stays None (template runs at its default) — no accidental key."""
    client, pool, _ = app_setup
    client.post("/generate", json={"adapter_id": "qa", "prompt": "hi"})
    assert pool.template_kwargs[-1] is None


# --- structured outputs (guided decoding) on the serving surface --------------------------------
# Every accepted request form must reach the engine as the SAME canonical spec (a dict of
# StructuredOutputsParams kwargs on payload.structured_outputs) — normalization happens once, in
# the GenerateRequest validator, so all entry points share it and 422 identically on bad specs.


_PERSON_SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}}


def test_chat_uses_our_structured_outputs_extension(app_setup):
    """The chat endpoint honours our structured_outputs/structured_outputs extension; any vLLM
    guided_* field alongside it is ignored (guided_* is no longer accepted)."""
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mc",
            "messages": [{"role": "user", "content": "hi"}],
            "structured_outputs": {"choice": ["a", "b"]},
            "guided_regex": r"\d+",
        },
    )
    assert resp.status_code == 200
    assert pool.structured[-1] == {"choice": ["a", "b"]}


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
        ({"type": "json_schema", "schema": _PERSON_SCHEMA}, {"json": _PERSON_SCHEMA}),
    ):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "mc",
                "messages": [{"role": "user", "content": "hi"}],
                "response_format": rf,
            },
        )
        assert resp.status_code == 200, resp.text
        assert pool.structured[-1] == expected


def test_chat_extension_beats_response_format(app_setup):
    """Precedence: our structured_outputs extension wins over the OpenAI response_format."""
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mc",
            "messages": [{"role": "user", "content": "hi"}],
            "structured_outputs": {"choice": ["a", "b"]},
            "response_format": {"type": "json_object"},
        },
    )
    assert resp.status_code == 200
    assert pool.structured[-1] == {"choice": ["a", "b"]}


def test_chat_response_format_json_schema_without_schema_is_422(app_setup):
    """A malformed response_format (json_schema with no schema) is a clean 422, not a 500."""
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mc",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "p"}},
        },
    )
    assert resp.status_code == 422
    assert "requires a schema" in resp.json()["detail"]
    assert pool.generated == []


def test_chat_response_format_unknown_type_is_422(app_setup):
    """An unknown response_format type is a clean 422, not silently reinterpreted as a schema."""
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mc",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "image_url"},
        },
    )
    assert resp.status_code == 422
    assert "unsupported response_format type" in resp.json()["detail"]
    assert pool.generated == []


def test_guided_fields_are_ignored(app_setup):
    """vLLM guided_* request fields are no longer accepted; a request carrying only guided_*
    generates unconstrained."""
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "mc",
            "messages": [{"role": "user", "content": "hi"}],
            "guided_regex": r"\d+",
        },
    )
    assert resp.status_code == 200
    assert pool.structured[-1] is None


def test_chat_without_structured_outputs_is_unconstrained(app_setup):
    """Freeform still works: a chat request with no structured-outputs field generates
    unconstrained (None reaches the engine)."""
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "mc", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert pool.structured[-1] is None


def test_invalid_structured_outputs_is_422_with_detail(app_setup):
    client, pool, _ = app_setup
    two_constraints = client.post(
        "/v1/chat/completions",
        json={
            "model": "mc",
            "messages": [{"role": "user", "content": "hi"}],
            "structured_outputs": {"json": _PERSON_SCHEMA, "regex": r"\d+"},
        },
    )
    assert two_constraints.status_code == 422
    assert "exactly one constraint" in two_constraints.json()["detail"]
    unsupported = client.post(
        "/v1/chat/completions",
        json={
            "model": "mc",
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
        json={"adapter_id": "qa", "prompt": "hi", "structured_outputs": {"json": _PERSON_SCHEMA}},
    )
    assert resp.status_code == 200
    assert pool.structured[-1] == {"json": _PERSON_SCHEMA}


def test_per_adapter_generate_passes_structured_outputs_through(app_setup):
    # The untyped-dict variant builds a GenerateRequest from the body, so the field rides along.
    client, pool, _ = app_setup
    resp = client.post(
        "/adapters/qb/generate",
        json={"prompt": "hi", "structured_outputs": {"choice": ["a", "b"]}},
    )
    assert resp.status_code == 200
    assert pool.structured[-1] == {"choice": ["a", "b"]}


def test_absent_structured_outputs_stays_none(app_setup):
    """No spec anywhere -> None reaches the engine (which may then apply an adapter default)."""
    client, pool, _ = app_setup
    assert client.post("/generate", json={"adapter_id": "qa", "prompt": "hi"}).status_code == 200
    assert pool.structured[-1] is None


def test_streamed_chat_forwards_structured_outputs(app_setup):
    client, pool, _ = app_setup
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mc",
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
    client.post("/generate", json={"adapter_id": "qa", "prompt": "x"})
    client.post("/generate", json={"adapter_id": "qb", "prompt": "y"})
    # Both Qwen adapters dispatched to the SAME base-model engine (one GPU, multi-LoRA).
    assert {bm for bm, _ in pool.generated} == {QWEN}
    assert {aid for _, aid in pool.generated} == {
        _revision_id("qa"),
        _revision_id("qb"),
    }


def test_generate_endpoint_returns_snake_case_body(app_setup):
    """The client-facing /generate response is snake_case end-to-end: the engine RPC dict is now
    snake_case (see modal_app.py::_generate) and the router returns it verbatim, so no camelCase
    key can reach the caller."""
    client, _, _ = app_setup
    body = client.post("/generate", json={"adapter_id": "qa", "prompt": "hi"}).json()
    assert body["adapter_id"] == "qa"
    assert body["finish_reason"] == "stop"
    assert body["token_ids"] == [1, 2, 3]
    assert body["text"] == f"[{QWEN}] reply"
    # None of the old camelCase spellings survive anywhere in the response.
    for camel in ("adapterId", "finishReason", "tokenIds", "inferenceTimeSeconds", "requestId"):
        assert camel not in body


def test_openai_chat_completions_routes_and_shapes(app_setup):
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "mc", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "mc"
    assert body["choices"][0]["message"]["content"] == f"[{QWEN_2B}] reply"
    assert pool.generated == [(QWEN_2B, _revision_id("mc"))]


def test_external_openai_chat_forwards_system_prompts(app_setup):
    client, pool, _ = app_setup
    messages = [
        {"role": "system", "content": "stay terse"},
        {"role": "user", "content": "hi"},
    ]
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "mc", "messages": messages},
    )

    assert resp.status_code == 200, resp.text
    assert pool.generated == [(QWEN_2B, _revision_id("mc"))]
    assert pool.messages[-1] == messages


def test_internal_openai_chat_can_send_system_prompts(app_setup):
    client, pool, _ = app_setup
    resp = client.post(
        "/v1/chat/completions",
        headers={"X-Freesolo-Internal-Key": "sekret"},
        json={
            "model": "mc",
            "messages": [
                {"role": "system", "content": "platform prompt"},
                {"role": "user", "content": "hi"},
            ],
        },
    )

    assert resp.status_code == 200, resp.text
    assert pool.generated == [(QWEN_2B, _revision_id("mc"))]


def test_external_generate_forwards_system_prompts(app_setup):
    client, pool, _ = app_setup
    messages = [
        {"role": "system", "content": "stay terse"},
        {"role": "user", "content": "hi"},
    ]
    resp = client.post(
        "/generate",
        json={"adapter_id": "qa", "messages": messages},
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
            "model": "mc",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = resp.read().decode("utf-8")

    assert '"delta":{"role":"assistant"}' in text
    assert '"delta":{"content":"[' in text
    assert f"{QWEN_2B}] " in text
    assert '"delta":{"content":"reply"}' in text
    assert '"finish_reason":"stop"' in text
    assert "data: [DONE]" in text
    assert pool.generated == [(QWEN_2B, _revision_id("mc"))]


def test_openai_chat_completions_stream_can_include_usage(app_setup):
    client, pool, _ = app_setup
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mc",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as resp:
        assert resp.status_code == 200
        text = resp.read().decode("utf-8")

    assert '"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}' in text
    assert pool.generated == [(QWEN_2B, _revision_id("mc"))]


def test_streaming_usage_reporter_fires_and_is_not_gc_dropped(app_setup):
    # The streaming path meters via a bare asyncio.create_task. asyncio only keeps a WEAK ref to
    # such a task, so the router holds its own strong ref (a pending-tasks set) to keep the billing
    # report alive until it settles. Consuming the whole stream must deliver exactly one report.
    _, pool, router = app_setup
    reports: list[dict] = []

    async def reporter(usage: dict) -> None:
        reports.append(usage)

    client = _serve(pool, router, usage_reporter=reporter)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mc",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        resp.read()  # drain the full stream so the final chunk schedules the report

    assert len(reports) == 1
    assert reports[0]["promptTokens"] == 2
    assert reports[0]["completionTokens"] == 2
    assert reports[0]["requestId"] == "req-stream"


def _metered_chat_stream(events, reports):
    record = _rec("metered", QWEN)
    router = AdapterRouter([record])

    def schedule_usage(_record, final, _caller_org):
        reports.append(final.copy())

    return openai_chat_stream(
        router,
        schedule_usage,
        record=record,
        events=events,
        adapter_id=record.adapter_id,
        completion_id="chatcmpl-metered",
        created=123,
        include_usage=True,
        caller_org=None,
    )


def test_stream_disconnect_still_schedules_terminal_usage_once():
    async def scenario():
        release_final = asyncio.Event()
        sent_partial = asyncio.Event()
        disconnect_sent = asyncio.Event()
        reports = []

        async def events():
            yield {"type": "delta", "text": "partial"}
            await release_final.wait()
            yield {
                "type": "final",
                "finish_reason": "stop",
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "request_id": "req-disconnected",
            }

        response = StreamingResponse(_metered_chat_stream(events(), reports))

        async def receive():
            await sent_partial.wait()
            disconnect_sent.set()
            return {"type": "http.disconnect"}

        async def send(message):
            if (
                message["type"] == "http.response.body"
                and b'"content":"partial"' in message["body"]
            ):
                sent_partial.set()

        response_task = asyncio.create_task(
            response(
                {"type": "http", "asgi": {"spec_version": "2.3"}},
                receive,
                send,
            )
        )
        await disconnect_sent.wait()
        # hold the terminal event past the disconnect. the response must NOT complete here: the
        # shielded drain keeps billing inside the request's own shutdown order, so a container
        # stopping in this window still has a task to await rather than a silently dropped charge.
        await asyncio.sleep(0)
        assert not response_task.done(), (
            "the response completed while the terminal event was still pending -- "
            "the drain is orphaned and its usage can be lost"
        )
        assert reports == [], "usage cannot be scheduled before the terminal event arrives"
        release_final.set()
        await response_task
        return reports

    reports = asyncio.run(scenario())
    assert reports == [
        {
            "type": "final",
            "finish_reason": "stop",
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "request_id": "req-disconnected",
        }
    ]


def test_stream_normal_completion_schedules_usage_once_without_changing_bytes():
    async def scenario():
        reports = []
        source_tasks = []
        response_task = asyncio.current_task()

        async def events():
            source_tasks.append(asyncio.current_task())
            yield {"type": "delta", "text": "answer"}
            yield {
                "type": "final",
                "finish_reason": "stop",
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "request_id": "req-complete",
            }

        chunks = [chunk async for chunk in _metered_chat_stream(events(), reports)]
        assert len(source_tasks) == 1
        assert source_tasks[0] is not response_task
        return chunks, reports

    chunks, reports = asyncio.run(scenario())
    assert chunks == [
        _sse(
            {
                "id": "chatcmpl-metered",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": _revision_id("metered"),
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        ),
        _sse(
            {
                "id": "chatcmpl-metered",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": _revision_id("metered"),
                "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}],
            }
        ),
        _sse(
            {
                "id": "chatcmpl-metered",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": _revision_id("metered"),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        ),
        _sse("[DONE]"),
    ]
    assert reports == [
        {
            "type": "final",
            "finish_reason": "stop",
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "request_id": "req-complete",
        }
    ]


def test_stream_engine_error_reaches_connected_client_without_usage():
    async def scenario():
        reports = []
        source_tasks = []
        response_task = asyncio.current_task()

        async def events():
            source_tasks.append(asyncio.current_task())
            yield {"type": "delta", "text": "partial"}
            raise ValueError("engine stream failed")

        chunks = [chunk async for chunk in _metered_chat_stream(events(), reports)]
        assert len(source_tasks) == 1
        assert source_tasks[0] is not response_task
        return chunks, reports

    chunks, reports = asyncio.run(scenario())
    assert b'"content":"partial"' in chunks[-3]
    assert chunks[-2:] == [
        _sse(
            {
                "id": "chatcmpl-metered",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": _revision_id("metered"),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                "error": {
                    "message": "engine stream failed",
                    "type": "engine_error",
                    "code": 400,
                },
            }
        ),
        _sse("[DONE]"),
    ]
    assert reports == []


def test_openai_chat_stream_sets_anti_buffering_headers(app_setup):
    # X-Accel-Buffering: no tells Nginx not to buffer the SSE response; without it each token
    # accumulates in Nginx's output buffer and the caller sees high TTFT for small completions.
    client, _, _ = app_setup
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mc",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        headers = {k.lower(): v for k, v in resp.headers.items()}
    assert headers.get("x-accel-buffering") == "no"
    assert headers.get("cache-control") == "no-cache"


def test_openai_chat_completions_bad_payload_is_422_not_500(app_setup):
    # A non-numeric top_p (and likewise a malformed messages shape) makes the in-handler
    # GenerateRequest validation raise a Pydantic ValidationError. It must surface as a 4xx
    # client error, not escape the handler as a 500.
    client, pool, _ = app_setup
    bad_top_p = client.post(
        "/v1/chat/completions",
        json={"model": "mc", "messages": [{"role": "user", "content": "hi"}], "top_p": "nope"},
    )
    assert bad_top_p.status_code == 422
    bad_messages = client.post(
        "/v1/chat/completions",
        json={"model": "mc", "messages": "not-a-list"},
    )
    assert bad_messages.status_code == 422
    # Never reached the engine.
    assert pool.generated == []


def test_per_adapter_generate_bad_payload_is_422_not_500(app_setup):
    client, pool, _ = app_setup
    resp = client.post("/adapters/qb/generate", json={"top_p": "nope"})
    assert resp.status_code == 422
    assert pool.generated == []


def test_per_adapter_generate_endpoint_routes(app_setup):
    client, pool, _ = app_setup
    assert client.post("/adapters/qb/generate", json={"prompt": "hi"}).status_code == 200
    assert pool.generated == [(QWEN, _revision_id("qb"))]


def test_unknown_adapter_is_404_not_misrouted(app_setup):
    client, pool, _ = app_setup
    assert client.post("/generate", json={"adapter_id": "nope", "prompt": "hi"}).status_code == 404
    assert (
        client.post("/v1/chat/completions", json={"model": "nope", "messages": []}).status_code
        == 404
    )
    assert pool.generated == []


def test_engine_valueerror_is_400_when_adapter_still_ready(app_setup):
    # The engine raises ValueError for a genuine bad payload (e.g. missing prompt) while the
    # router still has the adapter ready -> 400, not 404.
    _, pool, router = app_setup

    async def boom(base_model, payload, record, **_kwargs):
        raise ValueError("prompt or messages is required")

    pool.generate = boom
    client = _serve(pool, router, internal_key="sekret")
    assert client.post("/generate", json={"adapter_id": "qa"}).status_code == 400


def test_raced_undeploy_engine_valueerror_is_404(app_setup):
    # An adapter passes _lookup, then races out of the registry; the engine reports it as an
    # unknown adapter (ValueError). The router must surface 404 (same as _lookup), not 400.
    _, pool, router = app_setup

    async def vanish(base_model, payload, record, **_kwargs):
        router.remove(payload.adapter_id)  # concurrent undeploy after _lookup passed
        raise ValueError(f"Unknown adapter id on {base_model}: {payload.adapter_id}")

    pool.generate = vanish
    client = _serve(pool, router, internal_key="sekret")
    assert client.post("/generate", json={"adapter_id": "qa", "prompt": "hi"}).status_code == 404


def test_list_adapters_requires_internal_key(app_setup):
    client, _, _ = app_setup
    # Anonymous listing leaks repo_id/url (HF namespaces + adapter->tenant mapping), so it must be
    # gated the same as register/teardown.
    assert client.get("/adapters").status_code == 401
    ok = client.get("/adapters", headers={"X-Freesolo-Internal-Key": "sekret"})
    assert ok.status_code == 200
    assert {a["adapter_id"] for a in ok.json()["adapters"]} == {
        "qa",
        "qb",
        "mc",
        _revision_id("qa"),
        _revision_id("qb"),
        _revision_id("mc"),
    }


def test_list_adapters_refreshes_authoritative_persisted_state():
    persisted_revision = _rec("fresh", QWEN)
    persisted_alias = _alias(persisted_revision)
    reloads = 0

    def _reload():
        nonlocal reloads
        reloads += 1
        return [persisted_revision, persisted_alias]

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
    assert set(records) == {"fresh", _revision_id("fresh")}
    assert records["fresh"]["metadata"]["alias_of"] == _revision_id("fresh")
    assert not router.has("stale")


def test_adapters_fail_closed_when_no_internal_key_configured():
    # A serving app built without an internal key must NOT serve an open /adapters control plane:
    # register/teardown fail closed (503) rather than allowing unauthenticated mutation.
    router = _router_for("qa", QWEN)
    client = _serve(FakePool(), router)  # no internal_key
    rec = _adapter_payload("qz")
    no_key = client.post("/adapters", json=rec)
    assert no_key.status_code == 503
    # Even presenting a key can't open it when none is configured (no key to compare against).
    with_key = client.post("/adapters", json=rec, headers={"X-Freesolo-Internal-Key": "anything"})
    assert with_key.status_code == 503
    teardown = client.delete("/adapters/qa")
    assert teardown.status_code == 503
    assert router.resolve("qa")[1].base_model == QWEN  # nothing was mutated


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
            "headers": [(b"x-freesolo-internal-key", b"sekret")],
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
                "engine_replica_id": "replica-7",
                "checkpoint": checkpoint,
            },
        )


def test_openai_chat_completions_includes_usage_when_engine_reports_counts():
    router = _router_for("qa", QWEN)
    client = _serve(_MeteringPool(), router)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qa", "messages": [{"role": "user", "content": "hi"}]},
    )

    assert resp.status_code == 200
    assert resp.json()["usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


def test_usage_reporter_fires_with_token_counts_and_gpu_time():
    """After a successful generation the router meters it (background task): the reporter is
    called once with token counts + gpu seconds, keyed by the adapter and its base model."""
    router = _router_for("qa", QWEN)
    reports: list[dict] = []

    async def reporter(usage: dict) -> None:
        reports.append(usage)

    client = _serve(_MeteringPool(), router, usage_reporter=reporter, deployment_id="deployment-3")
    resp = client.post("/generate", json={"adapter_id": "qa", "prompt": "hi"})
    assert resp.status_code == 200
    assert len(reports) == 1
    u = reports[0]
    assert u["adapterId"] == "qa"
    assert u["baseModel"] == QWEN
    assert u["promptTokens"] == 7
    assert u["completionTokens"] == 3
    # An engine that doesn't report cached tokens meters 0 (no discount), never a missing key.
    assert u["cachedTokens"] == 0
    assert u["cachedTokensReported"] is False
    assert u["gpuSeconds"] == 0.25
    assert u["requestId"]  # a stable idempotency key is generated per report
    assert u["engineReplicaId"] == "replica-7"
    assert u["servingDeploymentId"] == "deployment-3"


def test_generate_response_strips_internal_cache_attribution():
    router = _router_for("qa", QWEN)
    client = _serve(_MeteringPool(), router)

    body = client.post("/generate", json={"adapter_id": "qa", "prompt": "hi"}).json()

    assert "cached_tokens_reported" not in body
    assert "engine_replica_id" not in body


def test_usage_report_uses_engine_request_id_for_idempotency():
    """The report's requestId is the engine's stable per-generation id (so a future report retry
    dedupes via the (org_id, request_id) key), not a fresh uuid minted per delivery."""
    router = _router_for("qa", QWEN)
    reports: list[dict] = []

    class _IdPool(_MeteringPool):
        async def generate(
            self,
            base_model: str,
            payload,
            record,
            *,
            expected_checkpoint: str | None = None,
        ) -> dict:
            out = await super().generate(
                base_model,
                payload,
                record,
                expected_checkpoint=expected_checkpoint,
            )
            out["request_id"] = "gen-123"  # what the real engine returns
            return out

    async def reporter(usage: dict) -> None:
        reports.append(usage)

    client = _serve(_IdPool(), router, usage_reporter=reporter)
    assert client.post("/generate", json={"adapter_id": "qa", "prompt": "hi"}).status_code == 200
    assert reports[0]["requestId"] == "gen-123"


def test_usage_reporter_failure_does_not_break_serving():
    """Metering is fire-and-forget: a reporter that raises must not surface to the caller."""
    router = _router_for("qa", QWEN)

    async def boom(usage: dict) -> None:
        raise RuntimeError("backend down")

    client = _serve(_MeteringPool(), router, usage_reporter=boom)
    assert client.post("/generate", json={"adapter_id": "qa", "prompt": "hi"}).status_code == 200


def test_nonstream_usage_report_is_detached_from_response_lifecycle():
    import asyncio
    import threading

    router = _router_for("qa", QWEN)
    started = threading.Event()
    release = threading.Event()
    response_done = threading.Event()
    response: dict[str, object] = {}

    async def reporter(usage: dict) -> None:
        started.set()
        await asyncio.to_thread(release.wait)

    with _serve(_MeteringPool(), router, usage_reporter=reporter) as client:

        def _send() -> None:
            try:
                response["value"] = client.post(
                    "/generate", json={"adapter_id": "qa", "prompt": "hi"}
                )
            finally:
                response_done.set()

        worker = threading.Thread(target=_send)
        worker.start()
        assert started.wait(timeout=5)
        completed_while_reporter_blocked = response_done.wait(timeout=0.5)
        release.set()
        worker.join(timeout=5)

    assert completed_while_reporter_blocked is True
    assert response["value"].status_code == 200


def test_shutdown_drains_usage_reports_before_closing_reporter():
    import asyncio
    import threading

    router = _router_for("qa", QWEN)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    close_saw_finished = {"value": False}

    async def reporter(usage: dict) -> None:
        started.set()
        await asyncio.to_thread(release.wait)
        finished.set()

    async def _aclose() -> None:
        close_saw_finished["value"] = finished.is_set()

    reporter.aclose = _aclose
    timer = threading.Timer(0.1, release.set)
    with _serve(_MeteringPool(), router, usage_reporter=reporter) as client:
        assert (
            client.post("/generate", json={"adapter_id": "qa", "prompt": "hi"}).status_code == 200
        )
        assert started.wait(timeout=5)
        timer.start()
    timer.join(timeout=5)

    assert finished.is_set()
    assert close_saw_finished["value"] is True


def test_no_reporting_when_engine_omits_token_counts(app_setup):
    """The base FakePool returns no token counts -> nothing to meter, reporter is never called."""
    _, pool, router = app_setup
    reports: list[dict] = []

    async def reporter(usage: dict) -> None:
        reports.append(usage)

    client = _serve(pool, router, usage_reporter=reporter)
    assert client.post("/generate", json={"adapter_id": "qa", "prompt": "hi"}).status_code == 200
    assert reports == []


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
        yield {"type": "ready", "checkpoint": checkpoint}
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


def test_usage_reporter_forwards_cached_tokens():
    """A prefix-cache hit is metered: the reporter receives cachedTokens for the backend discount."""
    router = _router_for("qa", QWEN)
    reports: list[dict] = []

    async def reporter(usage: dict) -> None:
        reports.append(usage)

    client = _serve(_CachedMeteringPool(), router, usage_reporter=reporter)
    assert client.post("/generate", json={"adapter_id": "qa", "prompt": "hi"}).status_code == 200
    assert reports[0]["cachedTokens"] == 6
    assert reports[0]["cachedTokensReported"] is True
    assert reports[0]["engineReplicaId"] == "replica-cached"
    assert reports[0]["promptTokens"] == 10


def test_streaming_usage_reporter_forwards_cached_tokens():
    router = _router_for("qa", QWEN)
    reports: list[dict] = []

    async def reporter(usage: dict) -> None:
        reports.append(usage)

    client = _serve(_CachedMeteringPool(), router, usage_reporter=reporter)
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "qa", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        resp.read()
    assert len(reports) == 1
    assert reports[0]["cachedTokens"] == 6
    assert reports[0]["cachedTokensReported"] is True
    assert reports[0]["engineReplicaId"] == "replica-cached"


def test_openai_usage_exposes_cached_tokens_in_prompt_details():
    """The client-facing OpenAI usage object surfaces cached tokens (prompt_tokens_details)."""
    router = _router_for("qa", QWEN)
    client = _serve(_CachedMeteringPool(), router)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qa", "messages": [{"role": "user", "content": "hi"}]},
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
        json={"model": "qa", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert "prompt_tokens_details" not in resp.json()["usage"]


def test_disabled_adapter_is_not_routable():
    router = _router_for("off", QWEN, status="disabled")
    assert router.resolve("off") is None
    assert router.base_models() == []  # no ready adapters -> no GPU


def test_undeploy_unknown_adapter_is_404(app_setup):
    client, pool, _ = app_setup
    resp = client.delete("/adapters/nope", headers={"X-Freesolo-Internal-Key": "sekret"})
    assert resp.status_code == 404
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
            # a revision request is refused unless the engine attests what it served
            "lora_request_adapter": record.adapter_id,
        }

    pool.generate = _capture
    assert client.post("/generate", json={"adapter_id": "qa", "prompt": "hi"}).status_code == 200
    assert captured == [(QWEN, _revision_id("qa"), "org/qa")]


def test_generate_miss_reloads_from_shared_storage():
    # A router container that hasn't seen a just-registered adapter reloads the persisted
    # table once on a miss before 404-ing.
    reloaded = {"count": 0}

    def _reload():
        reloaded["count"] += 1
        revision = _rec("late", QWEN)
        return [revision, _alias(revision)]

    pool = FakePool()
    router = AdapterRouter([])
    client = _serve(pool, router, reload_records=_reload)
    # Unknown locally -> reload finds it -> 200, and it's now cached.
    assert client.post("/generate", json={"adapter_id": "late", "prompt": "hi"}).status_code == 200
    assert reloaded["count"] == 1
    assert client.post("/generate", json={"adapter_id": "late", "prompt": "hi"}).status_code == 200
    assert reloaded["count"] == 1  # cached now, no second reload
    # Still-unknown after reload -> 404.
    assert client.post("/generate", json={"adapter_id": "ghost", "prompt": "hi"}).status_code == 404


def test_stale_ready_record_refreshes_in_background():
    # Cross-container undeploy: this router still caches a ready row for an adapter another
    # container undeployed. With the TTL elapsed, a hit should not block on shared storage; it
    # serves the cached row, refreshes in the background, then stops routing it on later requests.
    import asyncio

    from httpx import ASGITransport, AsyncClient

    revision = _rec("qa", QWEN)
    shared = {"rows": [revision, _alias(revision)]}

    def _reload():
        return list(shared["rows"])

    pool = FakePool()
    router = _router_for("qa", QWEN)
    app = build_serving_app(
        pool, router, reload_records=_reload, reload_interval_seconds=0.0, chat_authorizer=_allow
    )

    async def _scenario():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as ac:
            assert (
                await ac.post(
                    "/generate",
                    json={"adapter_id": "qa", "prompt": "hi"},
                    headers={"Authorization": "Bearer t"},
                )
            ).status_code == 200
            # Another container undeploys qa: it drops out of the status=ready reload.
            shared["rows"] = []
            # This hit is stale, but still cached: serve it and schedule the refresh.
            assert (
                await ac.post(
                    "/generate",
                    json={"adapter_id": "qa", "prompt": "hi"},
                    headers={"Authorization": "Bearer t"},
                )
            ).status_code == 200
            for _ in range(50):
                if not router.has("qa"):
                    break
                await asyncio.sleep(0.01)
            assert not router.has("qa")
            # After the background refresh, it is no longer routed here.
            assert (
                await ac.post(
                    "/generate",
                    json={"adapter_id": "qa", "prompt": "hi"},
                    headers={"Authorization": "Bearer t"},
                )
            ).status_code == 404

    asyncio.run(_scenario())


def test_hit_refresh_failure_serves_cached_adapter():
    # A transient shared-storage failure on the TTL hit-refresh must NOT fail a request we can
    # still serve from the cached ready record; only a genuine miss should hard-fail.
    state = {"fail": False}

    def _reload():
        if state["fail"]:
            raise RuntimeError("supabase 503")
        revision = _rec("qa", QWEN)
        return [revision, _alias(revision)]

    pool = FakePool()
    router = _router_for("qa", QWEN)
    client = _serve(pool, router, reload_records=_reload, reload_interval_seconds=0.0)
    # Warm the cache (TTL=0 -> every subsequent hit triggers a refresh).
    assert client.post("/generate", json={"adapter_id": "qa", "prompt": "x"}).status_code == 200
    # Now refresh fails: the cached ready "qa" is still served.
    state["fail"] = True
    assert client.post("/generate", json={"adapter_id": "qa", "prompt": "x"}).status_code == 200
    assert pool.generated[-1] == (QWEN, _revision_id("qa"))


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


def test_usage_reporter_client_closed_on_shutdown():
    # The persistent httpx client must be closed on app shutdown (no leaked sockets). build_serving_app
    # wires the reporter's aclose into a FastAPI lifespan; TestClient's context manager runs it.
    router = _router_for("qa", QWEN)
    closed = {"v": False}

    async def reporter(usage: dict) -> None:
        return None

    async def _aclose() -> None:
        closed["v"] = True

    reporter.aclose = _aclose
    with _serve(FakePool(), router, usage_reporter=reporter):
        pass
    assert closed["v"] is True


def test_on_startup_is_invoked_in_background_on_app_start():
    """The optional startup callback runs through the FastAPI lifespan without blocking readiness."""
    import threading

    router = _router_for("qa", QWEN)
    called = threading.Event()

    async def _startup() -> None:
        called.set()

    with _serve(FakePool(), router, on_startup=_startup):
        assert called.wait(timeout=5), "on_startup was not invoked on app startup"


def test_on_startup_failure_does_not_crash_the_router():
    """A best-effort startup callback cannot prevent the router from serving traffic."""
    import threading

    router = _router_for("qa", QWEN)
    ran = threading.Event()

    async def _boom() -> None:
        ran.set()
        raise RuntimeError("warm exploded")

    with _serve(FakePool(), router, internal_key="sekret", on_startup=_boom) as client:
        assert ran.wait(timeout=5)
        # the router is unaffected by the startup failure; health and routing still work.
        assert client.get("/healthz").json()["ok"] is True


def test_concurrent_misses_reload_once_and_hydrate_in_order():
    """Two concurrent misses must not hydrate out of order or stampede shared storage.

    `reload()` suspends on `to_thread`, so without serialization a slow first fetch can land AFTER
    a fast second one and overwrite fresher records with staler ones -- then stamp a newer
    timestamp over the result, hiding the regression until the next TTL.
    """
    import asyncio

    from flash.serving.src.lookup import AdapterLookup

    revision = _rec("late", QWEN)
    fresh = [revision, _alias(revision)]
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
            lookup.resolve("late", require_supported_base_model=False),
            lookup.resolve("late", require_supported_base_model=False),
            return_exceptions=True,
        )

    results = asyncio.run(_both())

    assert calls["count"] == 1, "each concurrent miss fetched separately instead of coalescing"
    assert hydrated == [0], f"hydrate ran {len(hydrated)}x; a stale fetch can overwrite a fresh one"
    # both callers see the same outcome; neither observes a half-applied hydrate.
    assert all(isinstance(r, HTTPException) for r in results)
