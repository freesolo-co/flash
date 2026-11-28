"""Prefill mode returns exactly what the per-request path returns, from one engine batch.

The chain under test is the real one: score-batch chunk sender -> control-plane chat transport
-> serving router (``/v1/chat/completions`` and its ``/batch`` route, accounting included) ->
engine pool -> ``_LoraEngineImpl._generate``/``_generate_batch`` -> ``generation.generate``
over a fake vLLM engine. Only the vLLM engine, the tokenizer-side prompt build and adapter
resolution are faked.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from flash.serve.request import transport
from flash.server.domain.ops import score_batches
from flash.serving.lambda_cloud.proxy import LambdaEngineProxy
from flash.serving.lambda_cloud.transport import (
    BatchUnsupported,
    WorkerError,
    WorkerTransport,
    batch_item_outcome,
)
from flash.serving.lambda_cloud.worker import create_app
from flash.serving.src.engine.lora_engine import _LoraEngineImpl
from flash.serving.src.http.rate_limit import allow_offline_inference
from flash.serving.src.http.router import AdapterRouter, build_serving_app
from flash.serving.src.io.cache_namespace import internal_generation_payload
from flash.serving.src.io.generation_batch import envelope, outcome_of
from flash.serving.src.io.schemas import AdapterRecord, internal_adapter_payload
from tests.serving.conftest import RecordingUsageStore

MODEL = "Qwen/Qwen3.5-9B-Base"
INTERNAL_KEY = "internal-secret"


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _VllmEngine:
    """Deterministic per prompt; records how many generations were in the engine at once."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.sampling: list[dict[str, Any]] = []

    async def generate(self, prompt_input, sampling, request_id, **_kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.sampling.append(dict(vars(sampling)))
        try:
            await asyncio.sleep(0.01)  # siblings submitted together are in flight together
        finally:
            self.active -= 1
        ids = prompt_input["prompt_token_ids"]
        if "TOO LONG" in "".join(chr(i) for i in ids):
            raise ValueError("This model's maximum context length is 262144 tokens")
        seed = sum(ids)
        token = 65 + seed % 5
        position = {
            65 + k: SimpleNamespace(logprob=-((k + seed) % 7 + 1) / 8, decoded_token=chr(65 + k))
            for k in range(5)
        }
        choice = SimpleNamespace(
            index=0, text=chr(token), finish_reason="length", token_ids=[token], logprobs=[position]
        )
        yield SimpleNamespace(prompt_token_ids=ids, num_cached_tokens=0, outputs=[choice])


class _Engine(_LoraEngineImpl):
    """The hosted engine with vLLM, prompt rendering and adapter resolution faked."""

    def __init__(self, record: AdapterRecord) -> None:
        self.record = record
        self.base_model = record.base_model
        self.engine = _VllmEngine()
        # Base chat stops at <|im_end|>; the stop id is read from the tokenizer.
        self.tokenizer = SimpleNamespace(
            convert_tokens_to_ids=lambda token: 248046, unk_token_id=None
        )

    async def _lora_request(self, _adapter_id, _record_dict):
        return None, self.record

    def _thinking_default(self, _record, _payload):
        return False

    def _structured_outputs_state(self, _payload, _record, _thinking):
        return None, None, {}

    async def _prepare_prompt_input(self, payload, _thinking):
        rendered = json.dumps(payload.messages, sort_keys=True)
        return {"prompt_token_ids": [ord(c) for c in rendered]}

    def _close_prompt_images(self, _prompt_input):
        return None

    def _replica_identifier(self):
        return "replica-1"


class _Pool:
    """``_ModalEnginePool`` minus Modal: the same payload encoding and batch envelopes."""

    def __init__(self, engine: _Engine) -> None:
        self.engine = engine
        self.single_calls = 0
        self.batch_calls: list[int] = []

    async def generate(self, _base_model, payload, record, *, expected_checkpoint=None):
        self.single_calls += 1
        return await self.engine._generate(
            internal_generation_payload(payload),
            internal_adapter_payload(record),
            expected_checkpoint,
            payload.generation_id,
        )

    async def generate_batch(self, _base_model, payloads, record, *, expected_checkpoint=None):
        self.batch_calls.append(len(payloads))
        envelopes = await self.engine._generate_batch(
            [internal_generation_payload(payload) for payload in payloads],
            internal_adapter_payload(record),
            expected_checkpoint,
            [payload.generation_id for payload in payloads],
        )
        # Envelopes cross the Modal boundary as plain data.
        return [outcome_of(json.loads(json.dumps(item))) for item in envelopes]

    async def stream_generate(self, *_a, **_k):  # pragma: no cover
        yield {}

    async def register(self, _base_model, _record):  # pragma: no cover
        return None

    async def unregister(self, *_a, **_k):  # pragma: no cover
        return None


def _base_record() -> AdapterRecord:
    return AdapterRecord(
        model_id=MODEL,
        repo_id=MODEL,
        base_model=MODEL,
        serve_base_model=True,
        thinking=False,
        org_id=None,
        status="ready",
    )


@pytest.fixture
def serving(monkeypatch):
    record = _base_record()
    pool = _Pool(_Engine(record))
    store = RecordingUsageStore()
    app = build_serving_app(
        pool,
        AdapterRouter([record]),
        internal_key=INTERNAL_KEY,
        usage_store=store,
        rate_limiter=allow_offline_inference,
    )
    client = TestClient(app)
    monkeypatch.setenv("FREESOLO_INTERNAL_KEY", INTERNAL_KEY)
    monkeypatch.setattr(transport, "_chat_http_client", lambda: client)
    monkeypatch.setattr(transport, "serving_openai_base_url", lambda: "http://testserver/v1")
    monkeypatch.setattr(transport, "_CHAT_BATCH_UNSUPPORTED", {})
    return SimpleNamespace(pool=pool, store=store, client=client)


def _row() -> dict[str, Any]:
    return {
        "run_id": None,
        "target": MODEL,
        "org_id": "org-test",
        "thinking": None,
        "api_key_id": None,
    }


def _requests() -> list[dict[str, Any]]:
    requests = []
    for i in range(12):
        request = {
            "messages": [
                {"role": "system", "content": "Answer with one letter."},
                {"role": "user", "content": f"Question {i}: A, B, C, D or E?"},
            ],
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
        }
        if i % 3 == 0:
            request["logprob_token_ids"] = [65, 66, 67]
        else:
            request["top_logprobs"] = 1 + i % 5
        requests.append(request)
    # A request the engine rejects, and one the control plane rejects before sending: each is
    # rejected the same way inside a chunk.
    too_long = [{"role": "user", "content": "TOO LONG"}]
    requests.append({**requests[1], "messages": too_long})
    requests.append({**requests[1], "top_logprobs": 21})
    return requests


def _per_request(requests):
    out = []
    for request in requests:
        try:
            out.append(score_batches._send_request(_row(), {"id": 7}, request))
        except Exception as exc:
            out.append(exc)
    return out


def _comparable(outcome):
    if isinstance(outcome, httpx.HTTPStatusError):
        return ("error", outcome.response.status_code, outcome.response.json())
    if isinstance(outcome, BaseException):
        return ("raised", type(outcome), str(outcome))
    # Fresh per request on every path: the completion id and its timestamp.
    return {k: v for k, v in outcome.items() if k not in {"id", "created"}}


def test_prefill_chunk_matches_the_per_request_path_exactly(serving):
    requests = _requests()
    alone = _per_request(requests)
    engine = serving.pool.engine.engine
    alone_sampling = list(engine.sampling)
    sent = len(requests) - 1  # the locally rejected one never reaches serving
    assert serving.pool.single_calls == sent
    serving.pool.single_calls = 0
    engine.sampling.clear()
    engine.peak = 0

    chunked = score_batches._send_chunk(score_batches._send_request, _row(), {"id": 7}, requests)

    assert [_comparable(o) for o in chunked] == [_comparable(o) for o in alone]
    # Every content and logprob came from one engine batch: one router call, one pool batch,
    # all generations in the engine at once, with the very same sampling parameters.
    assert serving.pool.single_calls == 0
    assert serving.pool.batch_calls == [sent]
    assert engine.peak == sent
    assert engine.sampling == alone_sampling
    assert isinstance(chunked[-2], httpx.HTTPStatusError)
    assert chunked[-2].response.status_code == 400
    assert "maximum context length" in chunked[-2].response.json()["detail"]
    # Each request that ran is metered on its own, exactly as when sent alone.
    assert len(serving.store.finalized) == 2 * (sent - 1)


def test_serving_without_the_batch_route_gets_the_same_requests_one_by_one(serving, monkeypatch):
    requests = _requests()[:4]
    alone = _per_request(requests)
    serving.pool.single_calls = 0
    monkeypatch.setattr(transport, "CHAT_BATCH_PATH_SUFFIX", "/not-deployed")

    chunked = score_batches._send_chunk(score_batches._send_request, _row(), {"id": 7}, requests)

    assert [_comparable(o) for o in chunked] == [_comparable(o) for o in alone]
    assert serving.pool.batch_calls == []
    assert serving.pool.single_calls == len(requests)
    assert transport._CHAT_BATCH_UNSUPPORTED  # not asked again for a while


def test_the_batch_route_is_internal_only(serving):
    body = {"requests": [{**_requests()[1], "model": MODEL}]}
    assert serving.client.post("/v1/chat/completions/batch", json=body).status_code in (401, 403)
    response = serving.client.post(
        "/v1/chat/completions/batch",
        json=body,
        headers={"X-Freesolo-Internal-Key": INTERNAL_KEY, "X-Freesolo-Org-Id": "org-test"},
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["status_code"] == 200


@pytest.mark.anyio
async def test_engine_batch_is_each_single_generation():
    record = _base_record()
    engine = _Engine(record)
    payloads = [
        {
            "model_id": MODEL,
            "messages": [{"role": "user", "content": f"q{i}"}],
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 2,
        }
        for i in range(5)
    ]
    ids = [f"g{i}" for i in range(5)]
    alone = [
        await engine._generate(p, internal_adapter_payload(record), None, gid)
        for p, gid in zip(payloads, ids, strict=True)
    ]
    batched = await engine._generate_batch(payloads, internal_adapter_payload(record), None, ids)

    def strip(result):
        return {k: v for k, v in result.items() if k != "inference_time_seconds"}

    assert [strip(outcome_of(item)) for item in batched] == [strip(r) for r in alone]
    assert engine.engine.peak == 5


# worker protocol -------------------------------------------------------------------------------


def _worker_engine():
    async def generate(payload, record, expected_checkpoint, generation_id):
        if payload.get("fail") == "value":
            raise ValueError("checkpoint mismatch: stale")
        if payload.get("fail") == "timeout":
            raise TimeoutError
        if payload.get("fail") == "engine":
            raise RuntimeError("engine died")
        return {"request_id": generation_id, "text": payload["prompt"]}

    return SimpleNamespace(
        base_model="Qwen/Qwen3.5-0.8B",
        _load=AsyncMock(),
        _shutdown=AsyncMock(),
        _generate=generate,
        _health=lambda: {"ok": True, "base_model": "Qwen/Qwen3.5-0.8B"},
        engine=SimpleNamespace(abort=AsyncMock()),
    )


def test_worker_batch_answers_each_item_as_generate_would():
    headers = {"Authorization": "Bearer private"}
    items = [
        {"payload": {"prompt": "a"}, "generation_id": "b0"},
        {"payload": {"prompt": "b", "fail": "value"}, "generation_id": "b1"},
        {"payload": {"prompt": "c", "fail": "timeout"}, "generation_id": "b2"},
        {"payload": {"prompt": "d", "fail": "engine"}, "generation_id": "b3"},
        {"payload": {"prompt": "e"}, "generation_id": "b4"},
    ]
    with TestClient(create_app(_worker_engine(), "private"), raise_server_exceptions=False) as c:
        assert c.get("/health", headers=headers).json()["generate_batch"] is True
        ids = [item["generation_id"] for item in items]
        assert c.post("/lease", json={"generation_ids": ids}, headers=headers).status_code == 200
        batch = c.post("/generate_batch", json={"items": items}, headers=headers).json()
        alone = []
        for item in items:
            c.post("/lease", json={"generation_id": item["generation_id"]}, headers=headers)
            alone.append(c.post("/generate", json=item, headers=headers))

    def single_outcome(response, generation_id):
        try:
            from flash.serving.lambda_cloud.transport import check_response

            check_response(response)
        except (ValueError, WorkerError) as exc:
            return type(exc), str(exc), getattr(exc, "status_code", None)
        return response.json()

    def batch_outcome(item, generation_id):
        outcome = batch_item_outcome(item, generation_id)
        if isinstance(outcome, BaseException):
            return type(outcome), str(outcome), getattr(outcome, "status_code", None)
        return outcome

    assert [
        batch_outcome(item, i["generation_id"])
        for item, i in zip(batch["results"], items, strict=True)
    ] == [single_outcome(r, i["generation_id"]) for r, i in zip(alone, items, strict=True)]


def test_worker_batch_items_need_their_own_lease():
    headers = {"Authorization": "Bearer private"}
    items = [{"payload": {"prompt": "a"}, "generation_id": "unleased"}]
    with TestClient(create_app(_worker_engine(), "private")) as client:
        result = client.post("/generate_batch", json={"items": items}, headers=headers).json()
    assert result["results"] == [{"status": 504, "kind": "timeout"}]


@pytest.mark.anyio
async def test_proxy_uses_the_batch_route_only_on_workers_that_have_it():
    from flash.serving.lambda_cloud.bootstrap import identity

    paths = []

    async def handler(request):
        paths.append(request.url.path)
        body = json.loads(request.content or b"{}")
        if request.url.path == "/lease":
            return httpx.Response(200, json={"ok": True})
        if request.url.path == "/generate_batch":
            if batch_route == "missing":
                return httpx.Response(404, json={"detail": "Not Found"})
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"status": 200, "result": {"request_id": item["generation_id"]}}
                        for item in body["items"]
                    ]
                },
            )
        return httpx.Response(200, json={"request_id": body["generation_id"]})

    worker = WorkerTransport("8.8.8.8", identity(), transport=httpx.MockTransport(handler))
    proxy = LambdaEngineProxy()
    proxy.base_model = "Qwen/Qwen3.5-4B"  # admits 7 requests at once: 9 go as 7 + 2
    proxy._transport = worker
    payloads = [{"p": i} for i in range(9)]
    ids = [f"x{i}" for i in range(9)]
    try:
        batch_route = "present"
        proxy._health_cache = {"ok": True}
        old = await proxy._generate_batch(payloads, {}, None, ids)
        assert "/generate_batch" not in paths
        assert paths.count("/generate") == 9
        paths.clear()
        proxy._health_cache = {"ok": True, "generate_batch": True}
        new = await proxy._generate_batch(payloads, {}, None, ids)
        assert sorted(paths) == ["/generate_batch"] * 2 + ["/lease"] * 2
        assert old == new == [envelope({"request_id": gid}) for gid in ids]
        paths.clear()
        batch_route = "missing"
        again = await proxy._generate_batch(payloads, {}, None, ids)
        assert again == new
        assert proxy._health_cache["generate_batch"] is False
        with pytest.raises(BatchUnsupported):
            await worker.call_batch(
                [{"generation_id": g, "payload": p} for g, p in zip(ids, payloads, strict=True)]
            )
    finally:
        await worker.aclose()
