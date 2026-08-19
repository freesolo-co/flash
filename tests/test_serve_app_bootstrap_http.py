"""atomic bootstrap, authenticated OpenAI HTTP, streaming, and provenance."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import httpx
import pytest

from flash.serve.app import __main__ as app_main
from flash.serve.app import http as http_module
from flash.serve.app.bootstrap import (
    PublishedAdapter,
    ServingBootstrap,
    bootstrap_serving,
    engine_config_from_manifest,
)
from flash.serve.app.http import (
    _MAX_CHAT_REQUEST_BYTES,
    _decimal_exceeds_limit,
    _stream_body,
    create_app,
)
from flash.serve.app.manifest import build_serving_manifest
from flash.serve.app.openai import ReasoningDeltaSplitter, split_reasoning
from flash.serve.runtime import (
    GenerationResult,
    StreamDelta,
    StreamFinished,
    StreamReady,
)
from tests.test_serve_app_manifest import _spec_and_inputs

AUTH_TOKEN = "inference-token-sentinel"


class _FakeRuntime:
    def __init__(self, _config=None) -> None:
        self.started = False
        self.closed = False
        self.dead = False
        self.registered = []
        self.fail_registration_at: int | None = None
        self.generation_requests = []
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
        return GenerationResult(
            request_id="request-1",
            runtime_id="runtime-1",
            adapter_id=request.adapter_id,
            incarnation=request.expected_incarnation,
            text="why</think>answer",
            finish_reason="stop",
            token_ids=(1, 2),
            prompt_tokens=5,
            completion_tokens=2,
            cached_tokens=3,
            cached_tokens_reported=True,
            duration_seconds=0.1,
            thinking=True,
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


def _published_owner() -> tuple[ServingBootstrap, _FakeRuntime]:
    manifest = _manifest()
    runtime = _FakeRuntime()
    runtime.started = True
    owner = ServingBootstrap(manifest, runtime)
    adapter = manifest.adapters[0]
    revision = PublishedAdapter(adapter.adapter_revision, adapter, Path("/cache/adapter"))
    alias = PublishedAdapter("run-1", adapter, Path("/cache/adapter"))
    owner._models = MappingProxyType({adapter.adapter_revision: revision, "run-1": alias})
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
        "model": "run-1",
        "messages": [{"role": "user", "content": "hello"}],
    }
    body.update(overrides)
    return body


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
    monkeypatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    paths = {
        adapter.adapter_revision: tmp_path / adapter.adapter_revision
        for adapter in manifest.adapters
    }
    monkeypatch.setattr(
        "flash.serve.app.bootstrap.locked_manifest_cache",
        _locked_paths(paths),
    )
    runtime = _FakeRuntime()

    owner = asyncio.run(
        bootstrap_serving(manifest, tmp_path, runtime_factory=lambda _config: runtime)
    )

    assert owner.ready is True
    assert [spec.adapter_id for spec in runtime.registered] == [
        manifest.adapters[0].adapter_revision
    ]
    assert tuple(owner.models) == tuple(sorted((manifest.adapters[0].adapter_revision, "run-1")))
    assert owner.models["run-1"].adapter_revision == manifest.adapters[0].adapter_revision
    assert not hasattr(owner, "token")
    asyncio.run(owner.close())
    assert runtime.closed is True
    assert owner.models == {}


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
        _locked_paths({manifest.adapters[0].adapter_revision: tmp_path / "adapter"}),
    )
    runtime.fail_registration_at = 0
    with pytest.raises(RuntimeError, match="registration failed"):
        asyncio.run(bootstrap_serving(manifest, tmp_path, runtime_factory=lambda _config: runtime))
    assert runtime.closed is True


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
    assert _MAX_CHAT_REQUEST_BYTES == 24 * 1024 * 1024
    assert _MAX_CHAT_REQUEST_BYTES - encoded_images >= 2 * 1024 * 1024
    assert _decimal_exceeds_limit(str(_MAX_CHAT_REQUEST_BYTES), _MAX_CHAT_REQUEST_BYTES) is False
    assert _decimal_exceeds_limit("9" * 5000, _MAX_CHAT_REQUEST_BYTES) is True


def test_request_body_accepts_exact_limit_and_rejects_headers_or_streams_over_limit(
    monkeypatch,
) -> None:
    owner, _ = _published_owner()
    app = create_app(owner, bearer_token=AUTH_TOKEN)
    limit = 64
    monkeypatch.setattr(http_module, "_MAX_CHAT_REQUEST_BYTES", limit)

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
    assert provenance["requested_model"] == "run-1"
    assert provenance["adapter_revision"] == owner.models["run-1"].adapter_revision
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
    revision = owner.models["run-1"].adapter_revision
    incarnation = owner.models["run-1"].incarnation
    runtime.stream_events = [
        StreamReady("request-2", "runtime", revision, incarnation, True),
        StreamDelta("why</thi"),
        StreamDelta("nk>answer"),
        StreamFinished(
            "request-2",
            "runtime",
            revision,
            incarnation,
            "why</think>answer",
            "stop",
            5,
            2,
            1,
            True,
            0.1,
            True,
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
    revision = owner.models["run-1"].adapter_revision
    incarnation = owner.models["run-1"].incarnation
    ready = StreamReady("request-3", "runtime", revision, incarnation, True)
    terminal = StreamFinished(
        "request-3",
        "runtime",
        revision,
        incarnation,
        "answer",
        "stop",
        2,
        1,
        0,
        False,
        0.1,
        True,
    )
    mismatched_terminal = StreamFinished(
        "request-3",
        "other-runtime",
        revision,
        incarnation,
        "answer",
        "stop",
        2,
        1,
        0,
        False,
        0.1,
        True,
    )
    scenarios = (
        [ready, StreamDelta("partial")],
        [ready, terminal, terminal],
        [ready, mismatched_terminal],
        [ready, StreamDelta("partial"), RuntimeError("secret engine failure")],
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
    resolved = owner.models["run-1"]

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

    async def bootstrap(*_args):
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
