"""atomic bootstrap, authenticated OpenAI HTTP, streaming, and provenance."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import httpx
import pytest

from flash.serve.app import __main__ as app_main
from flash.serve.app import bootstrap as bootstrap_module
from flash.serve.app import http as http_module
from flash.serve.app.bootstrap import (
    PublishedAdapter,
    ServingBootstrap,
    bootstrap_serving,
    engine_config_from_manifest,
)
from flash.serve.app.http import (
    MAX_CHAT_REQUEST_BYTES,
    _decimal_exceeds_limit,
    _stream_body,
    create_app,
)
from flash.serve.app.manifest import build_serving_manifest
from flash.serve.app.openai import ReasoningDeltaSplitter, split_reasoning
from flash.serve.runtime import (
    GenerationChoice,
    GenerationResult,
    PromptError,
    StreamChoiceFinished,
    StreamDelta,
    StreamFinished,
    StreamReady,
)
from tests.test_serve_app_manifest import _profile_spec_and_inputs, _spec_and_inputs

AUTH_TOKEN = "inference-token-sentinel"


class _FakeRuntime:
    def __init__(self, _config=None) -> None:
        self.started = False
        self.closed = False
        self.dead = False
        self.registered = []
        self.fail_registration_at: int | None = None
        self.generation_requests = []
        self.generate_error: BaseException | None = None
        self.result_thinking = True
        self.stream_events = []
        self.stream_closed = False

    async def start(self) -> None:
        self.started = True

    async def register_adapter(self, spec) -> bool:
        if self.fail_registration_at == len(self.registered):
            raise RuntimeError("registration failed with secret detail")
        self.registered.append(spec)
        return True

    async def close(self) -> None:
        self.closed = True

    def health(self):
        return SimpleNamespace(ok=self.started and not self.closed and not self.dead)

    async def generate(self, request):
        self.generation_requests.append(request)
        if self.generate_error is not None:
            raise self.generate_error
        return GenerationResult(
            request_id="request-1",
            adapter_id=request.adapter_id,
            incarnation=request.expected_incarnation,
            choices=(
                GenerationChoice(
                    index=0,
                    text="why</think>answer",
                    finish_reason="stop",
                    token_ids=(1, 2),
                ),
            ),
            prompt_tokens=5,
            completion_tokens=2,
            cached_tokens=3,
            cached_tokens_reported=True,
            thinking=self.result_thinking,
        )

    def stream(self, request):
        self.generation_requests.append(request)
        runtime = self

        async def events():
            try:
                for event in runtime.stream_events:
                    if isinstance(event, BaseException):
                        raise event
                    yield event
            finally:
                runtime.stream_closed = True

        return events()


def _manifest():
    return build_serving_manifest(*_spec_and_inputs())


def _published_owner(
    *, thinking_default: bool | None = None
) -> tuple[ServingBootstrap, _FakeRuntime]:
    manifest = _manifest()
    runtime = _FakeRuntime()
    runtime.started = True
    owner = ServingBootstrap(manifest, runtime)
    adapter = manifest.adapters[0]
    if thinking_default is not None:
        adapter = replace(adapter, thinking_default=thinking_default)
        runtime.result_thinking = thinking_default
    checkpoint = PublishedAdapter(adapter.checkpoint_id, adapter)
    owner._models = MappingProxyType({adapter.checkpoint_id: checkpoint})
    owner._ready = True
    return owner, runtime


async def _request(app, method: str, path: str, **kwargs):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        return await client.request(method, path, **kwargs)


def _auth() -> dict[str, str]:
    return {"authorization": f"Bearer {AUTH_TOKEN}"}


def _chat_body(**overrides):
    body = {
        "model": "run-1/final",
        "messages": [{"role": "user", "content": "hello"}],
    }
    body.update(overrides)
    return body


def _raw_chat_body(extra: str) -> str:
    """a chat body as raw text, because `json.dumps` emits the very tokens under test."""

    return (
        '{"model": "run-1/final", "messages": [{"role": "user", "content": "hi"}], ' + extra + "}"
    )


def _locked_paths(paths):
    @contextmanager
    def locked(*_args):
        yield paths

    return locked


def _sse_payloads(response: httpx.Response):
    payloads = []
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        payloads.append(data if data == "[DONE]" else json.loads(data))
    return payloads


def test_engine_config_loads_served_checkpoint_not_logical_provenance() -> None:
    manifest = _manifest()

    config = engine_config_from_manifest(manifest)

    assert config.model == manifest.engine.served_model
    assert config.effective_served_model == manifest.engine.served_model
    assert config.model != manifest.logical_base_model
    assert config.model_revision == manifest.engine.model_revision
    assert config.tool_parser == "qwen3_coder"


@pytest.mark.parametrize(
    "model_id",
    ["Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B", "Qwen/Qwen3.6-35B-A3B"],
)
def test_profile_fields_reach_the_runtime_engine_config(model_id: str) -> None:
    manifest = build_serving_manifest(*_profile_spec_and_inputs(model_id))

    config = engine_config_from_manifest(manifest)

    assert config.model == manifest.engine.served_model
    assert config.model_revision == manifest.engine.model_revision
    assert config.tokenizer_model == manifest.engine.tokenizer_model
    assert config.tokenizer_revision == manifest.engine.tokenizer_revision
    if model_id == "Qwen/Qwen3.8-27B":
        assert config.model_revision != config.tokenizer_revision
    assert config.engine_args["max_model_len"] == manifest.engine.max_model_len
    assert config.engine_args["max_num_seqs"] == manifest.engine.max_num_seqs
    assert config.max_loras == manifest.engine.max_loras
    assert config.max_cpu_loras == manifest.engine.max_cpu_loras
    assert config.max_lora_rank == manifest.engine.max_lora_rank
    assert config.image_limit == manifest.engine.image_limit
    assert config.enable_tower_connector_lora is manifest.engine.enable_tower_connector_lora
    assert config.tool_parser == (
        "qwen3_coder" if manifest.logical_base_model == "Qwen/Qwen3.5-9B" else None
    )
    if manifest.engine.max_num_batched_tokens is None:
        assert "max_num_batched_tokens" not in config.engine_args
    else:
        assert config.engine_args["max_num_batched_tokens"] == 4096


def test_unset_engine_knobs_are_omitted_rather_than_passed_as_none() -> None:
    """an optional knob left unset must not reach vllm as None.

    vllm types these as literals and validates them, so `kv_cache_dtype=None` raises a
    CacheConfig ValidationError ("Input should be 'auto', 'float16', ...") instead of meaning
    "use the default". that killed a live canary at engine construction, after the weights had
    already downloaded onto the GPU. omitting the key is what "unset" has to mean: vllm's own
    default for both of these is 'auto'.

    asserted over every optional named arg rather than the one that failed, because the next knob
    added here would reintroduce the same bug silently.
    """

    manifest = _manifest()
    assert manifest.engine.kv_cache_dtype is None, "fixture must leave this knob unset"

    config = engine_config_from_manifest(manifest)

    for name in ("kv_cache_dtype", "quantization", "max_num_batched_tokens"):
        assert config.engine_args.get(name, "__absent__") is not None, (
            f"{name} was forwarded to vllm as None; unset knobs must be omitted entirely"
        )


def test_bootstrap_registers_every_revision_before_atomic_publish(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    manifest = _manifest()
    paths = {
        adapter.checkpoint_id: tmp_path / adapter.checkpoint_id for adapter in manifest.adapters
    }
    monkeypatch.setattr(
        "flash.serve.app.bootstrap.locked_manifest_cache",
        _locked_paths(paths),
    )
    runtime = _FakeRuntime()

    owner = asyncio.run(
        bootstrap_serving(manifest, tmp_path, runtime_factory=lambda _config: runtime)
    )
    output = capsys.readouterr().out
    phases = [
        "engine-construction-starting",
        "engine-constructed",
        "adapters-registered",
    ]
    positions = [output.index(f"phase={phase}") for phase in phases]
    assert positions == sorted(positions)
    assert f'model="{manifest.engine.served_model}"' in output
    assert f'revision="{manifest.engine.model_revision}"' in output
    assert AUTH_TOKEN not in output

    assert owner.ready is True
    assert [spec.adapter_id for spec in runtime.registered] == [manifest.adapters[0].checkpoint_id]
    assert tuple(owner.models) == (manifest.adapters[0].checkpoint_id,)
    assert owner.models["run-1/final"].adapter.checkpoint_id == manifest.adapters[0].checkpoint_id
    assert not hasattr(owner, "token")
    asyncio.run(owner.close())
    assert runtime.closed is True
    assert owner.models == {}


def test_filesystem_usage_follows_engine_start_and_readiness_publish(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    paths = {
        adapter.checkpoint_id: tmp_path / adapter.source_revision for adapter in manifest.adapters
    }
    monkeypatch.setattr(bootstrap_module, "locked_manifest_cache", _locked_paths(paths))
    runtime = _FakeRuntime()
    events: list[str] = []
    owners: list[ServingBootstrap] = []
    real_owner_type = ServingBootstrap

    async def start() -> None:
        runtime.started = True
        events.append("runtime-started")

    async def register_adapter(spec) -> bool:
        events.append("adapter-registered")
        runtime.registered.append(spec)
        return True

    def build_owner(owner_manifest, owner_runtime):
        owner = real_owner_type(owner_manifest, owner_runtime)
        owners.append(owner)
        return owner

    def filesystem_usage(stage, cache_root) -> None:
        assert cache_root == tmp_path
        if stage == "engine-constructed":
            assert runtime.started is True
            assert runtime.registered == []
        if stage == "serving-ready":
            assert owners[0]._ready is True
            assert len(runtime.registered) == len(manifest.adapters)
        events.append(f"usage:{stage}")

    runtime.start = start
    runtime.register_adapter = register_adapter
    monkeypatch.setattr(bootstrap_module, "ServingBootstrap", build_owner)
    monkeypatch.setattr(bootstrap_module, "emit_filesystem_usage", filesystem_usage)

    owner = asyncio.run(
        bootstrap_serving(manifest, tmp_path, runtime_factory=lambda _config: runtime)
    )

    assert owner is owners[0]
    assert events == [
        "runtime-started",
        "usage:engine-constructed",
        "adapter-registered",
        "usage:serving-ready",
    ]


def test_bootstrap_validation_and_registration_fail_closed(monkeypatch, tmp_path: Path) -> None:
    manifest = _manifest()
    runtime = _FakeRuntime()

    def invalid_cache(*_args):
        raise RuntimeError("corrupt cache with credential sentinel")

    monkeypatch.setattr("flash.serve.app.bootstrap.locked_manifest_cache", invalid_cache)
    with pytest.raises(RuntimeError, match="corrupt cache"):
        asyncio.run(bootstrap_serving(manifest, tmp_path, runtime_factory=lambda _config: runtime))
    assert runtime.started is False
    assert runtime.registered == []
    assert runtime.closed is True

    runtime.closed = False
    monkeypatch.setattr(
        "flash.serve.app.bootstrap.locked_manifest_cache",
        _locked_paths({manifest.adapters[0].checkpoint_id: tmp_path / "adapter"}),
    )
    runtime.fail_registration_at = 0
    with pytest.raises(RuntimeError, match="registration failed"):
        asyncio.run(bootstrap_serving(manifest, tmp_path, runtime_factory=lambda _config: runtime))
    assert runtime.closed is True


def test_engine_death_handler_reaches_the_runtime(monkeypatch, tmp_path: Path) -> None:
    # when vllm's engine core dies after readiness the http process stays bound and answers 503
    # for every later request. the packaged app built the runtime with no on_engine_death
    # callback, so `_notify_engine_death` only marked the notification complete and neither the
    # modal container nor the runpod pod was ever replaced.
    manifest = _manifest()
    monkeypatch.setattr(
        "flash.serve.app.bootstrap.locked_manifest_cache",
        _locked_paths({manifest.adapters[0].checkpoint_id: tmp_path / "adapter"}),
    )
    seen: list[object] = []
    runtime = _FakeRuntime()

    def _factory(_config, **kwargs):
        seen.append(kwargs.get("on_engine_death"))
        return runtime

    async def _handler(_health: object) -> None:
        return None

    owner = asyncio.run(
        bootstrap_serving(manifest, tmp_path, runtime_factory=_factory, on_engine_death=_handler)
    )
    assert seen == [_handler]
    asyncio.run(owner.close())

    # a caller that supplies no handler must still construct the runtime the old way, so the
    # single-argument factories every other test uses keep working.
    seen.clear()
    plain: list[object] = []

    def _plain_factory(_config):
        plain.append(_config)
        return _FakeRuntime()

    owner = asyncio.run(bootstrap_serving(manifest, tmp_path, runtime_factory=_plain_factory))
    assert len(plain) == 1
    asyncio.run(owner.close())


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
def test_packaged_chat_rejects_active_tool_stop_marker_collisions(stream: bool) -> None:
    owner, runtime = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)
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

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers=_auth(),
            json=_chat_body(tools=tools, stop="</tool_call>", stream=stream),
        )
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert runtime.generation_requests == []


def test_health_auth_models_and_no_model_fallback() -> None:
    owner, _ = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)

    health = asyncio.run(_request(app, "GET", "/healthz"))
    unauthorized = asyncio.run(_request(app, "GET", "/v1/models"))
    models = asyncio.run(_request(app, "GET", "/v1/models", headers=_auth()))
    missing = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers=_auth(),
            json=_chat_body(model="missing"),
        )
    )

    assert health.status_code == 200
    assert health.json() == {"ok": True}
    assert unauthorized.status_code == 401
    assert unauthorized.headers["www-authenticate"] == "Bearer"
    model_payload = models.json()["data"]
    assert [item["id"] for item in model_payload] == sorted(owner.models)
    assert model_payload[0]["flash_provenance"]["served_checkpoint_revision"] == (
        owner.manifest.engine.model_revision
    )
    assert model_payload[0]["flash_provenance"]["tokenizer_model"] == (
        owner.manifest.engine.tokenizer_model
    )
    assert model_payload[0]["flash_provenance"]["tokenizer_revision"] == (
        owner.manifest.engine.tokenizer_revision
    )
    assert missing.status_code == 404
    assert AUTH_TOKEN not in models.text + missing.text


def test_chat_auth_rejection_closes_the_connection_before_reading_the_body(monkeypatch) -> None:
    owner, _ = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)

    async def fail_if_read(_request):
        pytest.fail("unauthorized request body was read")

    monkeypatch.setattr(http_module, "_read_request_body", fail_if_read)
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            content=b"an unread upload",
        )
    )

    assert response.status_code == 401
    assert response.headers["connection"] == "close"


def test_bearer_scheme_is_case_insensitive_and_duplicate_or_malformed_headers_fail() -> None:
    owner, _ = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)

    for scheme in ("bearer", "BeArEr", "BEARER"):
        accepted = asyncio.run(
            _request(
                app,
                "GET",
                "/v1/models",
                headers={"authorization": f"{scheme} {AUTH_TOKEN}"},
            )
        )
        assert accepted.status_code == 200

    duplicate = asyncio.run(
        _request(
            app,
            "GET",
            "/v1/models",
            headers=[
                ("authorization", f"Bearer {AUTH_TOKEN}"),
                ("authorization", f"Bearer {AUTH_TOKEN}"),
            ],
        )
    )
    assert duplicate.status_code == 401
    for value in (
        f"Bearer  {AUTH_TOKEN}",
        f"Bearer\t{AUTH_TOKEN}",
        f" Bearer {AUTH_TOKEN}",
        f"Bearer {AUTH_TOKEN} ",
        "Bearer",
    ):
        rejected = asyncio.run(_request(app, "GET", "/v1/models", headers={"authorization": value}))
        assert rejected.status_code == 401


def test_bearer_rejections_always_compare_one_fixed_length_digest(monkeypatch) -> None:
    owner, _ = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)
    calls: list[tuple[bytes, bytes]] = []
    real_compare = http_module.hmac.compare_digest

    def instrumented(left: bytes, right: bytes) -> bool:
        calls.append((left, right))
        return real_compare(left, right)

    monkeypatch.setattr(http_module.hmac, "compare_digest", instrumented)
    cases = (
        None,
        {"authorization": "Basic ignored"},
        {"authorization": f"Bearer  {AUTH_TOKEN}"},
    )
    for headers in cases:
        response = asyncio.run(_request(app, "GET", "/v1/models", headers=headers))
        assert response.status_code == 401
    assert len(calls) == len(cases)
    assert all(len(left) == len(right) == 32 for left, right in calls)
    assert len({left for left, _ in calls}) == 1


def test_request_body_ceiling_covers_current_encoded_image_contract() -> None:
    encoded_images = 4 * (((16 * 1024 * 1024) + 2) // 3)
    assert MAX_CHAT_REQUEST_BYTES == 24 * 1024 * 1024
    assert MAX_CHAT_REQUEST_BYTES - encoded_images >= 2 * 1024 * 1024
    assert _decimal_exceeds_limit(str(MAX_CHAT_REQUEST_BYTES), MAX_CHAT_REQUEST_BYTES) is False
    assert _decimal_exceeds_limit("9" * 5000, MAX_CHAT_REQUEST_BYTES) is True


def test_request_body_accepts_exact_limit_and_rejects_headers_or_streams_over_limit(
    monkeypatch,
) -> None:
    owner, _ = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)
    limit = 64
    monkeypatch.setattr(http_module, "MAX_CHAT_REQUEST_BYTES", limit)

    exact = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers=_auth(),
            content=b" " * (limit - 2) + b"{}",
        )
    )
    assert exact.status_code == 422

    over_header = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers={**_auth(), "content-length": str(limit + 1)},
            content=b"{}",
        )
    )
    assert over_header.status_code == 413
    assert over_header.json()["error"]["code"] == "request_too_large"

    async def streamed_body():
        yield b" " * limit
        yield b"x"

    over_stream = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers=_auth(),
            content=streamed_body(),
        )
    )
    assert over_stream.status_code == 413
    assert over_stream.json()["error"]["code"] == "request_too_large"

    # both branches reject before the body is drained, so the unread bytes make the connection
    # unusable for a following request. closing is what makes those bytes unreachable, and it is
    # what the hosted middleware already does for the same early rejection.
    assert over_header.headers["connection"] == "close"
    assert over_stream.headers["connection"] == "close"


@pytest.mark.parametrize("stream", [False, True])
def test_disconnect_cancels_generation_after_body_is_consumed(stream: bool) -> None:
    owner, runtime = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)
    body = json.dumps(_chat_body(stream=stream)).encode()

    async def scenario() -> None:
        generation_started = asyncio.Event()
        generation_cancelled = asyncio.Event()
        stream_closed = asyncio.Event()
        receive_count = 0

        if stream:

            class HangingStream:
                def __aiter__(self):
                    return self

                async def __anext__(self):
                    generation_started.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        generation_cancelled.set()
                        raise

                async def aclose(self):
                    stream_closed.set()

            def stream_request(request):
                runtime.generation_requests.append(request)
                return HangingStream()

            runtime.stream = stream_request
        else:

            async def generate(request):
                runtime.generation_requests.append(request)
                generation_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    generation_cancelled.set()
                    raise

            runtime.generate = generate

        async def receive():
            nonlocal receive_count
            receive_count += 1
            if receive_count == 1:
                return {"type": "http.request", "body": body, "more_body": False}
            await generation_started.wait()
            return {"type": "http.disconnect"}

        async def send(_message):
            raise AssertionError("a disconnected request must not send a response")

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"authorization", f"Bearer {AUTH_TOKEN}".encode()),
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("test", 80),
        }

        app_task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(generation_started.wait(), timeout=1)
            await asyncio.wait({app_task}, timeout=0.1)
            assert generation_cancelled.is_set(), (
                "generation continued after the request disconnected"
            )
            assert stream_closed.is_set() is stream
            assert receive_count == 2
            with pytest.raises(asyncio.CancelledError):
                await app_task
        finally:
            if not app_task.done():
                app_task.cancel()
                with suppress(asyncio.CancelledError):
                    await app_task

    asyncio.run(scenario())
    assert len(runtime.generation_requests) == 1


def test_nonstream_reasoning_accounting_provenance_and_structured_precedence() -> None:
    owner, runtime = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)
    image = "data:image/png;base64,aWdub3JlZA=="
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers=_auth(),
            json=_chat_body(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "look"},
                            {"type": "input_image", "image_url": image},
                        ],
                    }
                ],
                response_format={"type": "text"},
            ),
        )
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert "cache-control" not in response.headers
    assert "x-accel-buffering" not in response.headers
    payload = response.json()
    message = payload["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "answer", "reasoning_content": "why"}
    assert payload["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 3},
    }
    provenance = payload["flash_provenance"]
    assert provenance["requested_model"] == "run-1/final"
    assert provenance["checkpoint_id"] == owner.models["run-1/final"].adapter.checkpoint_id
    assert provenance["logical_base_revision"] == owner.manifest.logical_base_revision
    assert provenance["served_checkpoint_revision"] == owner.manifest.engine.model_revision
    assert provenance["tokenizer_model"] == owner.manifest.engine.tokenizer_model
    assert provenance["tokenizer_revision"] == owner.manifest.engine.tokenizer_revision
    assert provenance["served_checkpoint_revision"] != provenance["logical_base_revision"]
    assert response.headers["x-flash-manifest-id"] == owner.manifest.manifest_id
    assert response.headers["x-flash-served-checkpoint-revision"] == (
        owner.manifest.engine.model_revision
    )
    assert response.headers["x-flash-tokenizer-model"] == owner.manifest.engine.tokenizer_model
    assert response.headers["x-flash-tokenizer-revision"] == (
        owner.manifest.engine.tokenizer_revision
    )
    request = runtime.generation_requests[0]
    assert request.structured_outputs == {}
    assert request.messages[0]["content"][0]["type"] == "input_text"
    assert request.messages[0]["content"][1]["type"] == "input_image"
    assert AUTH_TOKEN not in response.text

    conflict = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers=_auth(),
            json=_chat_body(
                structured_outputs={"json_object": True},
                response_format={"type": "text"},
            ),
        )
    )
    assert conflict.status_code == 422


def test_reasoning_split_uses_first_close_and_preserves_non_thinking_literals() -> None:
    assert split_reasoning("why</think>answer</think>literal", thinking=True) == (
        "why",
        "answer</think>literal",
    )
    assert split_reasoning("<think>unfinished", thinking=True) == ("unfinished", "")
    literal = "<think>literal</think>answer"
    assert split_reasoning(literal, thinking=False) == (None, literal)

    splitter = ReasoningDeltaSplitter(thinking=True)
    events = splitter.feed("why</thi") + splitter.feed("nk>answer") + splitter.finish()
    assert "".join(value for key, value in events if key == "reasoning_content") == "why"
    assert "".join(value for key, value in events if key == "content") == "answer"


def test_stream_primes_before_200_splits_reasoning_and_emits_one_real_finish() -> None:
    owner, runtime = _published_owner()
    revision = owner.models["run-1/final"].adapter.checkpoint_id
    incarnation = owner.models["run-1/final"].adapter.aggregate_sha256
    runtime.stream_events = [
        StreamReady("request-2", "runtime", revision, incarnation, True),
        StreamDelta(0, "why</thi"),
        StreamDelta(0, "nk>answer"),
        StreamChoiceFinished(0, "why</think>answer", "stop", (1, 2)),
        StreamFinished(
            request_id="request-2",
            runtime_id="runtime",
            adapter_id=revision,
            incarnation=incarnation,
            choices=(GenerationChoice(0, "why</think>answer", "stop", (1, 2)),),
            prompt_tokens=5,
            completion_tokens=2,
            cached_tokens=1,
            cached_tokens_reported=True,
            thinking=True,
        ),
    ]
    app = create_app(owner, bearer_token=AUTH_TOKEN)
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers=_auth(),
            json=_chat_body(stream=True, stream_options={"include_usage": True}),
        )
    )
    payloads = _sse_payloads(response)

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.text.startswith("data: ")
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    deltas = [
        payload["choices"][0]["delta"]
        for payload in payloads
        if isinstance(payload, dict) and payload.get("choices")
    ]
    assert "".join(delta.get("reasoning_content", "") for delta in deltas) == "why"
    assert "".join(delta.get("content", "") for delta in deltas) == "answer"
    finishes = [
        payload
        for payload in payloads
        if isinstance(payload, dict)
        and payload.get("choices")
        and payload["choices"][0]["finish_reason"] is not None
    ]
    assert len(finishes) == 1
    assert finishes[0]["choices"][0]["finish_reason"] == "stop"
    assert payloads[-2]["usage"]["prompt_tokens_details"] == {"cached_tokens": 1}
    for payload in (payloads[0], finishes[0], payloads[-2]):
        provenance = payload["flash_provenance"]
        assert provenance["logical_base_revision"] == owner.manifest.logical_base_revision
        assert provenance["served_checkpoint_revision"] == owner.manifest.engine.model_revision
        assert provenance["tokenizer_model"] == owner.manifest.engine.tokenizer_model
        assert provenance["tokenizer_revision"] == owner.manifest.engine.tokenizer_revision
    assert payloads[-1] == "[DONE]"
    assert runtime.stream_closed is True


def test_stream_missing_duplicate_or_failed_terminal_is_sanitized_without_fake_stop() -> None:
    owner, runtime = _published_owner()
    revision = owner.models["run-1/final"].adapter.checkpoint_id
    incarnation = owner.models["run-1/final"].adapter.aggregate_sha256
    ready = StreamReady("request-3", "runtime", revision, incarnation, True)
    terminal = StreamFinished(
        request_id="request-3",
        runtime_id="runtime",
        adapter_id=revision,
        incarnation=incarnation,
        choices=(GenerationChoice(0, "answer", "stop", (1,)),),
        prompt_tokens=2,
        completion_tokens=1,
        cached_tokens=0,
        cached_tokens_reported=False,
        thinking=True,
    )
    mismatched_terminal = StreamFinished(
        request_id="request-3",
        runtime_id="other-runtime",
        adapter_id=revision,
        incarnation=incarnation,
        choices=(GenerationChoice(0, "answer", "stop", (1,)),),
        prompt_tokens=2,
        completion_tokens=1,
        cached_tokens=0,
        cached_tokens_reported=False,
        thinking=True,
    )
    choice_terminal = StreamChoiceFinished(0, "answer", "stop", (1,))
    scenarios = (
        [ready, StreamDelta(0, "partial")],
        [ready, choice_terminal, terminal, terminal],
        [ready, choice_terminal, mismatched_terminal],
        [ready, StreamDelta(0, "partial"), RuntimeError("secret engine failure")],
    )
    app = create_app(owner, bearer_token=AUTH_TOKEN)
    for events in scenarios:
        runtime.stream_events = list(events)
        response = asyncio.run(
            _request(
                app,
                "POST",
                "/v1/chat/completions",
                headers=_auth(),
                json=_chat_body(stream=True),
            )
        )
        payloads = _sse_payloads(response)
        assert payloads[-1]["error"]["code"] == "stream_terminated"
        assert "[DONE]" not in payloads
        assert sum(isinstance(item, dict) and "error" in item for item in payloads) == 1
        assert not any(
            isinstance(item, dict)
            and item.get("choices")
            and item["choices"][0]["finish_reason"] is not None
            for item in payloads
        )
        assert "secret engine failure" not in response.text


def test_a_chat_request_may_override_the_registered_grammar_for_its_own_call() -> None:
    """the registered grammar is a per-revision *default*, not an unbreakable constraint.

    The fixture adapter registers `{"json_object": true}`. A request-level `structured_outputs`
    replaces it for that call, and `response_format: {"type": "text"}` normalizes to `{}` -- the
    explicit "unconstrained for this call" marker, distinct from an absent field. Both leave the
    registered default untouched for the next request, which is what makes the override per-call
    rather than a mutation of an immutable revision.
    """

    owner, runtime = _published_owner(thinking_default=False)
    assert owner.models["run-1/final"].adapter.structured_outputs_default == {"json_object": True}
    app = create_app(owner, bearer_token=AUTH_TOKEN)
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

    for override, expected in (
        ({"structured_outputs": {"regex": "[ab]+"}}, {"regex": "[ab]+"}),
        ({"response_format": {"type": "text"}}, {}),
        (
            {"response_format": {"type": "text"}, "tools": tools},
            {},
        ),
        ({"structured_outputs": {}, "tools": tools}, {}),
        ({}, None),
    ):
        response = asyncio.run(
            _request(
                app,
                "POST",
                "/v1/chat/completions",
                headers=_auth(),
                json=_chat_body(**override),
            )
        )
        assert response.status_code == 200
        assert runtime.generation_requests[-1].structured_outputs == expected

    # the revision still carries its own default after every override above.
    assert owner.models["run-1/final"].adapter.structured_outputs_default == {"json_object": True}


def test_packaged_route_treats_tool_choice_none_as_inactive() -> None:
    owner, runtime = _published_owner(thinking_default=True)
    app = create_app(owner, bearer_token=AUTH_TOKEN)
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

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers=_auth(),
            json=_chat_body(
                tools=tools,
                tool_choice="none",
                parallel_tool_calls=True,
            ),
        )
    )

    assert response.status_code == 200
    assert runtime.generation_requests[-1].tool_choice == "none"


@pytest.mark.parametrize(
    "body",
    [
        _raw_chat_body('"chat_template_kwargs": {"nested": NaN}'),
        _raw_chat_body('"structured_outputs": {"json": {"maximum": Infinity}}'),
        _raw_chat_body('"temperature": -Infinity'),
    ],
)
def test_non_finite_json_constants_are_rejected_as_invalid_json(body: str) -> None:
    """`NaN` and `Infinity` are python spellings, not json, and must not reach the runtime.

    `json.loads` accepts them by default. `temperature` and `top_p` are guarded by `math.isfinite`,
    but nothing walks inside `chat_template_kwargs` or a structured-output schema -- so a nested
    non-finite reached the tokenizer or vllm's grammar compiler and came back 503, telling the
    caller the service is down about a body it should have called invalid. Sent as raw text
    because `json.dumps` emits these same tokens, so a dict fixture could not express the case.
    """

    owner, runtime = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers={**_auth(), "content-type": "application/json"},
            content=body,
        )
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"
    assert runtime.generation_requests == []


@pytest.mark.parametrize("number", ["1.0", "1e3", "9007199254740993.0", "1e-400"])
def test_packaged_raw_json_rejects_decimal_numeric_enums(number: str) -> None:
    owner, runtime = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)
    tools = (
        '"tools":[{"type":"function","function":{"name":"weather","parameters":'
        '{"type":"object","properties":{"value":{"type":"number","enum":['
        + number
        + ']}},"required":["value"],"additionalProperties":false}}}]'
    )

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers={**_auth(), "content-type": "application/json"},
            content=_raw_chat_body(tools),
        )
    )

    assert response.status_code == 422
    assert runtime.generation_requests == []


@pytest.mark.parametrize("stream", [False, True])
def test_rejected_prompt_is_400_not_a_retryable_503(stream: bool) -> None:
    """an over-length prompt is a caller error on both paths, so clients do not retry it.

    503 invites a retry that must fail identically, re-tokenizing and re-dispatching to the gpu
    each time. the runtime raises `PromptError` for a vllm rejection; this pins that it reaches the
    client as 400 rather than being swept into the catch-all below it.
    """

    owner, runtime = _published_owner()
    failure = PromptError("This model's maximum context length is 32768 tokens")
    if stream:
        runtime.stream_events = [failure]
    else:
        runtime.generate_error = failure
    app = create_app(owner, bearer_token=AUTH_TOKEN)
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers=_auth(),
            json=_chat_body(stream=stream),
        )
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_stream_failure_before_ready_returns_503_and_cancellation_closes_iterator() -> None:
    owner, runtime = _published_owner()
    runtime.stream_events = [RuntimeError("secret before ready")]
    app = create_app(owner, bearer_token=AUTH_TOKEN)
    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers=_auth(),
            json=_chat_body(stream=True),
        )
    )
    assert response.status_code == 503
    assert "secret before ready" not in response.text
    assert runtime.stream_closed is True

    class ClosingIterator:
        def __init__(self) -> None:
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(60)
            raise StopAsyncIteration

        async def aclose(self):
            self.closed = True

    iterator = ClosingIterator()
    ready = StreamReady("request-4", "runtime", None, None, False)
    resolved = owner.models["run-1/final"]

    async def cancel() -> None:
        body = _stream_body(iterator, ready, resolved, {}, include_usage=False)
        await anext(body)
        task = asyncio.create_task(anext(body))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel())
    assert iterator.closed is True


class _ClosableOwner:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize("failure_stage", ["auth", "config", "server"])
def test_serve_construction_failures_close_bootstrapped_runtime(
    failure_stage: str,
    monkeypatch,
) -> None:
    import uvicorn

    owner = _ClosableOwner()
    seen_digest: list[str] = []

    bootstrap_kwargs: list[object] = []

    async def bootstrap(*_args, **kwargs):
        # the serve entrypoint must hand the runtime a way to end the process when the engine
        # core dies; without it a dead engine serves 503 forever and the container is never
        # replaced.
        bootstrap_kwargs.append(kwargs.get("on_engine_death"))
        return owner

    def make_app(_owner, *, bearer_digest: str):
        seen_digest.append(bearer_digest)
        if failure_stage == "auth":
            raise RuntimeError("auth construction failed")
        return object()

    def make_config(*_args, **_kwargs):
        if failure_stage == "config":
            raise RuntimeError("config construction failed")
        return object()

    def make_server(_config):
        if failure_stage == "server":
            raise RuntimeError("server construction failed")
        raise AssertionError("server should not be constructed in this scenario")

    monkeypatch.setattr(app_main, "_read_inference_token", lambda: AUTH_TOKEN)
    monkeypatch.setattr(app_main, "bootstrap_serving", bootstrap)
    monkeypatch.setattr(app_main, "create_app", make_app)
    monkeypatch.setattr(uvicorn, "Config", make_config)
    monkeypatch.setattr(uvicorn, "Server", make_server)
    args = SimpleNamespace(cache_root="/cache", host="127.0.0.1", port=8000)

    with pytest.raises(RuntimeError, match="construction failed"):
        asyncio.run(app_main._serve(args, _manifest()))

    assert owner.close_calls == 1
    assert seen_digest == [hashlib.sha256(AUTH_TOKEN.encode()).hexdigest()]
    assert [callable(handler) for handler in bootstrap_kwargs] == [True]


def test_engine_death_asks_the_running_server_to_exit(monkeypatch, capsys) -> None:
    # wiring the handler is not enough: invoking it has to actually stop the http server, which
    # is what ends the container and lets the provider start a healthy replacement.
    import uvicorn

    owner = _ClosableOwner()
    handlers: list[object] = []

    async def bootstrap(*_args, **kwargs):
        handlers.append(kwargs.get("on_engine_death"))
        return owner

    class _Server:
        def __init__(self, _config) -> None:
            self.should_exit = False
            self.capture_signals = None

        async def serve(self) -> None:
            # the engine dies while the server is running; the handler must ask it to stop.
            await handlers[0](None)

    monkeypatch.setattr(app_main, "_read_inference_token", lambda: AUTH_TOKEN)
    monkeypatch.setattr(app_main, "bootstrap_serving", bootstrap)
    monkeypatch.setattr(app_main, "create_app", lambda _owner, *, bearer_digest: object())
    monkeypatch.setattr(uvicorn, "Config", lambda *_a, **_k: object())
    built: list[_Server] = []

    def _make_server(config):
        server = _Server(config)
        built.append(server)
        return server

    monkeypatch.setattr(uvicorn, "Server", _make_server)
    args = SimpleNamespace(cache_root="/cache", host="127.0.0.1", port=8000)

    asyncio.run(app_main._serve(args, _manifest()))
    output = capsys.readouterr().out

    assert 'phase=port-bind-starting host="127.0.0.1" port="8000"' in output
    assert AUTH_TOKEN not in output
    assert built[0].should_exit is True
    assert owner.close_calls == 1


def test_missing_inference_token_prevents_runtime_bootstrap(monkeypatch) -> None:
    bootstrap_calls = 0

    def missing_token():
        raise RuntimeError("inference token fd is not configured")

    async def bootstrap(*_args):
        nonlocal bootstrap_calls
        bootstrap_calls += 1
        raise AssertionError("bootstrap must not run without inference auth")

    monkeypatch.setattr(app_main, "_read_inference_token", missing_token)
    monkeypatch.setattr(app_main, "bootstrap_serving", bootstrap)
    args = SimpleNamespace(cache_root="/cache", host="127.0.0.1", port=8000)

    with pytest.raises(RuntimeError, match="inference token fd"):
        asyncio.run(app_main._serve(args, _manifest()))

    assert bootstrap_calls == 0


def test_the_packaged_app_exposes_exactly_its_documented_route_surface() -> None:
    # this app is immutable by construction: bootstrap_serving registers every adapter from the
    # manifest at boot and freezes the model map, so there is deliberately no /adapters surface.
    # docs/serving-contract.md documents the DYNAMIC contract (generated + hosted backends) and now
    # says so explicitly, because pointing tests/serving_conformance at a `flash serve deploy`
    # endpoint fails 26 registration/activation/alias tests against a perfectly healthy deployment.
    #
    # pinning the surface keeps those two facts from drifting apart. adding a route here without
    # updating the doc is exactly the drift that sent a live canary through a conformance run it
    # could never pass.
    owner, _ = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)

    served = {
        (route.path, method)
        for route in app.routes
        for method in getattr(route, "methods", set()) or set()
        if method not in {"HEAD", "OPTIONS"}
    }
    assert served == {
        ("/healthz", "GET"),
        ("/v1/models", "GET"),
        ("/v1/chat/completions", "POST"),
    }, f"packaged serving app route surface changed: {sorted(served)}"

    # the absence of /adapters is the load-bearing half, so assert it as a live response rather
    # than only as a route-table fact: a customer probing it must get 404, not a half-wired handler.
    registered = asyncio.run(
        _request(app, "POST", "/adapters", headers=_auth(), json={"adapter_id": "x"})
    )
    assert registered.status_code == 404, (
        "the packaged app must not accept dynamic adapter registration; its adapters come from "
        "the immutable manifest at boot"
    )


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("label", "messages"),
    [
        ("missing content", [{"role": "user"}]),
        ("missing role", [{"content": "hello"}]),
        ("unknown role", [{"role": "bogus", "content": "hello"}]),
        ("non-sequence content", [{"role": "user", "content": {"text": "hello"}}]),
        ("non-object block", [{"role": "user", "content": ["hello"]}]),
        ("unknown block type", [{"role": "user", "content": [{"type": "audio"}]}]),
        ("non-string text", [{"role": "user", "content": [{"type": "text", "text": 7}]}]),
        # `tool_calls` used to be accepted on key presence alone, so a scalar reached the chat
        # template and raised a jinja UndefinedError -- a 503 for input that can never render.
        (
            "scalar tool_calls",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": 1},
            ],
        ),
        (
            "null tool_calls",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": None},
            ],
        ),
        (
            "non-object tool call entries",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": [1, 2]},
            ],
        ),
        (
            "empty tool_calls",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": None, "tool_calls": []},
            ],
        ),
    ],
)
def test_a_malformed_message_is_rejected_before_the_runtime_is_reached(
    label: str, messages: list[dict], stream: bool
) -> None:
    """`type(item) is dict` is not a message check, and the runtime never made up the difference.

    `PromptPreparer.prepare` dispatches on `has_image_blocks`, so message shape was validated only
    for image-bearing requests. Text-only requests went straight to `apply_chat_template`, a jinja
    renderer that validates nothing: a missing `content` rendered empty and returned 200 having
    generated from an empty prompt, while a bad `role` or non-string `content` raised a
    `TemplateError` from outside `_rejection_as_prompt_error` and was answered 503 -- advertising a
    healthy service as down and inviting a retry that must fail identically.

    Asserted on the wire for both streaming and non-streaming, because the two take different
    branches through `http.py`, and asserted together with an untouched runtime: the point is not
    merely a better status code but that a request that can never succeed is refused before it is
    dispatched to the gpu.
    """
    owner, runtime = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers=_auth(),
            json=_chat_body(messages=messages, stream=stream),
        )
    )

    assert response.status_code == 422, (
        f"{label} (stream={stream}) was answered {response.status_code}; a message the runtime "
        "cannot honor must be refused as an invalid request"
    )
    assert response.headers["content-type"].startswith("application/json")
    assert runtime.generation_requests == [], (
        f"{label} (stream={stream}) reached the runtime; malformed messages must be rejected "
        "before dispatch"
    )


@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
def test_tool_result_text_parts_reach_follow_up_generation(stream: bool) -> None:
    owner, runtime = _published_owner(thinking_default=False)
    checkpoint = owner.models["run-1/final"].adapter.checkpoint_id
    incarnation = owner.models["run-1/final"].adapter.aggregate_sha256
    if stream:
        choice = GenerationChoice(0, "answer", "stop", (1,))
        runtime.stream_events = [
            StreamReady("request-history", "runtime", checkpoint, incarnation, False),
            StreamDelta(0, "answer"),
            StreamChoiceFinished(0, "answer", "stop", (1,)),
            StreamFinished(
                request_id="request-history",
                runtime_id="runtime",
                adapter_id=checkpoint,
                incarnation=incarnation,
                choices=(choice,),
                prompt_tokens=3,
                completion_tokens=1,
                cached_tokens=0,
                cached_tokens_reported=False,
                thinking=False,
            ),
        ]
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "input_text", "text": "sun"},
                {"type": "text", "text": "ny"},
            ],
        },
        {"role": "user", "content": "summarize"},
    ]
    app = create_app(owner, bearer_token=AUTH_TOKEN)

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers=_auth(),
            json=_chat_body(messages=messages, stream=stream),
        )
    )

    assert response.status_code == 200, response.text
    assert runtime.generation_requests[-1].messages == tuple(messages)


@pytest.mark.parametrize(
    ("label", "messages"),
    [
        ("plain string", [{"role": "user", "content": "hello"}]),
        ("text blocks", [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]),
        ("openai input_text", [{"role": "user", "content": [{"type": "input_text", "text": "x"}]}]),
        (
            "system then user",
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        ),
        ("developer alias", [{"role": "developer", "content": "s"}]),
        ("empty block list", [{"role": "user", "content": []}]),
        (
            "assistant tool calls",
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "result", "tool_call_id": "1"},
            ],
        ),
    ],
)
def test_valid_message_shapes_still_reach_the_runtime(label: str, messages: list[dict]) -> None:
    """the guard must not narrow what the app accepts.

    Every shape here is one the sdk and the openai clients genuinely send -- including the
    `developer` alias, an assistant turn carrying `tool_calls` instead of content, and a deliberately
    empty user turn. A validator that rejected any of these would trade a 503 for a 422 and break
    working callers, so they are pinned as reaching the runtime rather than merely returning 200.
    """
    owner, runtime = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)

    response = asyncio.run(
        _request(
            app,
            "POST",
            "/v1/chat/completions",
            headers=_auth(),
            json=_chat_body(messages=messages),
        )
    )

    assert response.status_code == 200, f"{label} was rejected: {response.text}"
    assert len(runtime.generation_requests) == 1, f"{label} did not reach the runtime"
    # the guard validates a *copy* and discards it, so the runtime must still receive the caller's
    # own messages -- a normalizer that rewrote `developer` to `system` here would silently change
    # what the model is asked, which is the parser's job to avoid.
    forwarded = runtime.generation_requests[0].messages
    assert list(forwarded) == messages, (
        f"{label} was mutated on the way to the runtime: {forwarded}"
    )
