"""CPU contract tests for the private DeepInfer endpoint and rank bridge."""

import asyncio
import hashlib
import importlib
import json
import queue
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from flash.serving.deepinfer.app.http import build_app
from flash.serving.deepinfer.app.render import Decoder
from flash.serving.deepinfer.contract import config
from flash.serving.deepinfer.contract.protocol import parse_request
from flash.serving.deepinfer.runtime.bridge import Pending, Runtime
from flash.serving.src.http.stream_response import await_until_disconnect


def body(**kwargs):
    return {"model": config.MODEL_ID, "messages": [{"role": "user", "content": "Hello"}], **kwargs}


class Tokenizer:
    def token_to_id(self, text):
        return {"<think>": 10, "</think>": 11}.get(text, 1)

    def decode(self, tokens, **kwargs):
        values = {
            2: b"Hi",
            3: b" there",
            4: b"why",
            5: b"\xe2",
            6: b"\x82\xac",
            10: b"<think>",
            11: b"</think>",
        }
        return b"".join(values[token] for token in tokens).decode("utf8", errors="replace")


class FakeRuntime:
    tokenizer = Tokenizer()

    def __init__(self, events=None):
        self.events = events or [
            {"tokens": [2], "finish_reason": None},
            {"tokens": [3, 1], "finish_reason": "stop"},
        ]
        self.calls = 0
        self.closed = False

    def tokenize(self, request):
        return [0, 7]

    async def prepare_tokens(self, request):
        return self.tokenize(request)

    async def prefix_length(self, request, tokens):
        return 0

    def assert_healthy(self):
        pass

    async def generate(self, request, tokens):
        self.calls += 1
        try:
            for event in self.events:
                if isinstance(event, Exception):
                    raise event
                yield event
        finally:
            self.closed = True


@pytest.mark.parametrize(
    "change",
    [
        {"model": "deepseek-v4-flash"},
        {"n": 2},
        {"tools": []},
        {"stop": "END"},
        {"logprobs": True},
        {"frequency_penalty": 1},
        {"response_format": {"type": "json_object"}},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.org/image"}}
                    ],
                }
            ]
        },
        {"chat_template_kwargs": {"enable_thinking": "yes"}},
        {"max_tokens": -1},
        {"temperature": float("inf")},
    ],
)
def test_unsupported_requests_never_admit_gpu_work(change):
    runtime = FakeRuntime()
    with TestClient(build_app(runtime, manage_lifecycle=False)) as client:
        # JSON itself rejects infinity; directly test the same parser in that case.
        if change.get("temperature") == float("inf"):
            with pytest.raises(ValueError, match="temperature"):
                parse_request(body(**change))
        else:
            assert (
                client.post(
                    "/internal/generate", json={"request_id": "test", "chat": body(**change)}
                ).status_code
                == 422
            )
    assert runtime.calls == 0


def test_native_events_report_usage_and_close():
    runtime = FakeRuntime()
    with TestClient(build_app(runtime, manage_lifecycle=False)) as client:
        response = client.post("/internal/generate", json={"request_id": "test", "chat": body()})
    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["type"] == "ready"
    assert events[-1]["type"] == "final"
    assert events[-1]["text"] == "Hi there"
    assert events[-1]["completion_tokens"] == 3
    assert events[-1]["prompt_tokens"] == 2
    assert all(event["request_id"] == "test" for event in events)
    assert runtime.closed


def test_error_after_first_token_is_not_successful_completion():
    runtime = FakeRuntime([{"tokens": [2], "finish_reason": None}, RuntimeError("native failure")])
    with TestClient(build_app(runtime, manage_lifecycle=False)) as client:
        response = client.post("/internal/generate", json={"request_id": "test", "chat": body()})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1] == {"type": "error"}
    assert not any(event["type"] == "final" for event in events)
    assert runtime.closed


def test_capacity_error_before_stream_starts_returns_429():
    runtime = FakeRuntime([OverflowError()])
    with TestClient(build_app(runtime, manage_lifecycle=False)) as client:
        assert (
            client.post(
                "/internal/generate", json={"request_id": "test", "chat": body()}
            ).status_code
            == 429
        )
    assert runtime.closed


def test_reasoning_is_split_and_counted_without_retokenizing():
    decoder = Decoder(Tokenizer(), thinking=True)
    first = decoder.consume({"tokens": [10, 4], "finish_reason": None})
    second = decoder.consume({"tokens": [11, 2, 1], "finish_reason": "stop"})
    assert first == [{"reasoning_content": "why"}]
    assert second == [{"content": "Hi"}]
    assert decoder.completion_tokens == 5
    assert decoder.reasoning_tokens == 1


def test_decoder_retains_split_utf8_bytes():
    decoder = Decoder(Tokenizer(), thinking=False)
    assert decoder.consume({"tokens": [5], "finish_reason": None}) == []
    assert decoder.consume({"tokens": [6], "finish_reason": "length"}) == [{"content": "€"}]
    assert decoder.completion_tokens == 2


@pytest.mark.anyio
async def test_disconnect_cancels_generation_task():
    stopped = asyncio.Event()

    async def work():
        try:
            await asyncio.sleep(100)
        finally:
            stopped.set()

    async def disconnected():
        await asyncio.sleep(0)
        return {"type": "http.disconnect"}

    with pytest.raises(asyncio.CancelledError):
        await await_until_disconnect(SimpleNamespace(receive=disconnected), work())
    assert stopped.is_set()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("namespace", "prefix_tokens"), [(None, 0), ("a" * 64, 0), ("a" * 64, 128)]
)
async def test_rank_assignment_and_cancel_are_request_scoped(namespace, prefix_tokens):
    runtime = Runtime()
    runtime.incoming = [queue.Queue() for _ in range(config.GPU_COUNT)]
    tokens = list(range(130))
    affinity = (
        hashlib.sha256(((namespace or "") + ":" + str(tokens[:128])).encode()).digest()[0]
        % config.GPU_COUNT
    )
    idle = (affinity + 1) % config.GPU_COUNT
    initial_loads = [1] * config.GPU_COUNT
    initial_loads[idle] = 0
    runtime.loads = initial_loads.copy()
    rank = affinity if prefix_tokens else idle
    request = replace(parse_request(body()), cache_namespace=namespace, prefix_tokens=prefix_tokens)
    generator = runtime.generate(request, tokens)
    first = asyncio.create_task(anext(generator))
    await asyncio.sleep(0)
    command = runtime.incoming[rank].get_nowait()
    request_id = command["id"]
    runtime._deliver({"type": "tokens", "id": request_id, "tokens": [2], "finish_reason": None})
    assert (await first)["tokens"] == [2]
    closing = asyncio.create_task(generator.aclose())
    await asyncio.sleep(0)
    assert runtime.incoming[rank].get_nowait() == {"type": "cancel", "id": request_id}
    assert (
        runtime.loads[rank] == initial_loads[rank] + 1
    )  # Hold until native abort is acknowledged.
    runtime._deliver({"type": "cancelled", "id": request_id})
    await closing
    assert runtime.loads == initial_loads
    assert runtime.pending == {}
    assert all(channel.empty() for channel in runtime.incoming)


@pytest.mark.anyio
async def test_full_replica_does_not_enqueue_another_request():
    runtime = Runtime()
    runtime.loads = [config.SEQUENCES_PER_RANK] * config.GPU_COUNT
    with pytest.raises(OverflowError):
        await anext(runtime.generate(parse_request(body()), [0]))
    assert not runtime.pending


@pytest.mark.anyio
async def test_slow_reader_cancels_only_its_rank_and_reports_error():
    runtime = Runtime()
    runtime.incoming = [queue.Queue() for _ in range(config.GPU_COUNT)]
    pending = Pending(config.GPU_COUNT - 1, asyncio.Queue(maxsize=1), asyncio.Event())
    runtime.pending["slow"] = pending
    pending.events.put_nowait({"type": "tokens"})
    runtime._deliver({"type": "tokens", "id": "slow", "tokens": [2]})
    assert pending.events.get_nowait() == {"type": "fatal"}
    assert runtime.incoming[pending.rank].get_nowait() == {"type": "cancel", "id": "slow"}
    assert runtime.failure is None


def test_context_budget_includes_output_and_native_reserve():
    runtime = Runtime()
    runtime.encode_messages = lambda *a, **kw: "prompt"
    runtime.tokenizer = SimpleNamespace(
        encode=lambda *a, **kw: SimpleNamespace(ids=[0] * config.MAX_MODEL_LEN)
    )
    with pytest.raises(ValueError, match="serving context"):
        runtime.tokenize(parse_request(body()))


def test_native_image_patch_rejects_unknown_source():
    from flash.serving.deepinfer.native.image import patch_indexer

    with pytest.raises(RuntimeError, match="pinned release"):
        patch_indexer("unexpected upstream release")


def test_modal_private_two_gpu_autoscaling_contract(monkeypatch):
    captured = {}

    class Image:
        @classmethod
        def from_id(cls, image_id):
            captured["image"] = image_id
            return cls()

        def env(self, *a, **kw):
            return self

        def add_local_python_source(self, *a, **kw):
            return self

        def run_commands(self, *a, **kw):
            captured["build_commands"] = a
            return self

    class App:
        def __init__(self, *a, **kw):
            pass

        def function(self, **kwargs):
            captured["function"] = kwargs
            return lambda fn: fn

    def decorator(name):
        def factory(*args, **kwargs):
            captured[name] = kwargs
            return lambda fn: fn

        return factory

    stub = SimpleNamespace(
        Image=Image,
        App=App,
        Volume=SimpleNamespace(from_name=lambda *a, **kw: object()),
        concurrent=decorator("concurrent"),
        web_server=decorator("web"),
    )
    monkeypatch.setitem(sys.modules, "modal", stub)
    name = "flash.serving.deepinfer.app.modal_app"
    monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        launcher = importlib.import_module(name)
    finally:
        sys.modules.pop(name, None)
    assert captured["function"]["gpu"] == "B200:2"
    assert captured["function"]["memory"] == 524288
    assert captured["function"]["min_containers"] == 0
    assert captured["function"]["max_containers"] == 4
    assert captured["function"]["buffer_containers"] == 0
    assert captured["function"]["scaledown_window"] == 600
    assert captured["concurrent"] == {"max_inputs": 16, "target_inputs": 12}
    assert captured["web"]["requires_proxy_auth"] is True
    assert captured["build_commands"][-1] == (
        f"{config.RUNTIME_PYTHON} -m flash.serving.deepinfer.native.image"
    )
    process = object()

    def launch(command):
        captured["server_command"] = command
        return process

    def watch(**kwargs):
        captured["watcher"] = kwargs
        return SimpleNamespace(start=lambda: None)

    monkeypatch.setattr(launcher.subprocess, "Popen", launch)
    monkeypatch.setattr(launcher.threading, "Thread", watch)
    launcher.serve()
    assert captured["server_command"] == [
        config.RUNTIME_PYTHON,
        "-m",
        "flash.serving.deepinfer.app.http",
    ]
    assert captured["watcher"] == {
        "target": launcher._watch_server,
        "args": (process,),
        "daemon": True,
    }


@pytest.fixture
def anyio_backend():
    return "asyncio"
