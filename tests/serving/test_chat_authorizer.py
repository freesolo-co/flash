"""Status mapping for the serving->backend chat authorizer (modal_app._build_chat_authorizer).

The authorizer POSTs to the backend's /api/serving/authorize and must translate the backend's
HTTP status into the right serving-side HTTPException:

  - 200 + nonempty orgId      -> allow and return the billing org
  - malformed 200             -> 503 (fail closed; never authorize without a billing org)
  - 401 + code invalid_api_key-> 401 (the user's key is bad)
  - 401 without that code     -> 503 (our machine bearer was rejected: a serving misconfig, NOT the
                                 caller's key — must not be reported as an invalid user key)
  - 403 / 404                 -> 403 (collapse so we don't leak which adapters exist)
  - 503 / any other backend 5xx -> 503 (transient auth-lookup failure stays retryable; NEVER masked
                                 as a 502, which a client/LB may treat as permanent and not retry)
  - transport error           -> 503 (fail closed, never serve unauthorized)

It also caches a SUCCESSFUL authorization per API-key digest and adapter ID for a short TTL and
coalesces concurrent identical lookups into ONE backend call (single-flight), so an eval's many
same-key requests don't stampede the backend auth path into transient 5xx failures.

modal_app imports the `modal` SDK at module top (decorators run at import), which isn't installed
in the offline test env, so we stub it just enough to import and reach _build_chat_authorizer.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from flash.serving.src.http.router import (
    AdapterRouter,
    build_offline_serving_app,
    build_serving_app,
)
from flash.serving.src.io.schemas import AdapterRecord
from tests.serving.conftest import RecordingUsageStore, attest


def _passthrough_decorator(*_a: Any, **_k: Any):
    def deco(obj: Any) -> Any:
        return obj

    return deco


@pytest.fixture(scope="module")
def modal_app_module(load_modal_app_under_stub):
    modal_stub = MagicMock(name="modal")
    modal_stub.concurrent.side_effect = _passthrough_decorator
    modal_stub.method.side_effect = _passthrough_decorator
    modal_stub.enter.side_effect = _passthrough_decorator
    modal_stub.asgi_app.side_effect = _passthrough_decorator
    modal_stub.parameter.return_value = None
    app_mock = MagicMock(name="app")
    app_mock.cls.side_effect = _passthrough_decorator
    app_mock.function.side_effect = _passthrough_decorator
    app_mock.local_entrypoint.side_effect = _passthrough_decorator
    modal_stub.App.return_value = app_mock
    modal_stub.Period.return_value = MagicMock()
    return load_modal_app_under_stub(modal_stub)


class _FakeResp:
    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _authorizer(modal_app_module, monkeypatch, response: Any):
    """Build a real authorizer whose single httpx.AsyncClient.post returns/raises ``response``."""

    class _FakeClient:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        async def post(self, _url: str, json: Any, headers: Any = None) -> _FakeResp:
            if isinstance(response, Exception):
                raise response
            return response

        async def aclose(self) -> None:
            return None

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    settings = SimpleNamespace(
        backend_url="https://backend.example.com", internal_key="machine-key"
    )
    authorize = modal_app_module._build_chat_authorizer(settings)
    assert authorize is not None
    return authorize


def _run(authorize) -> None:
    import asyncio

    asyncio.run(authorize("fs-user-key", "adapter-1"))


def test_authorizer_none_when_unconfigured(modal_app_module):
    # No backend URL or no internal key -> no authorizer (router then fails closed).
    no_url = SimpleNamespace(backend_url="", internal_key="k")
    no_key = SimpleNamespace(backend_url="https://b", internal_key="")
    assert modal_app_module._build_chat_authorizer(no_url) is None
    assert modal_app_module._build_chat_authorizer(no_key) is None


def test_401_invalid_user_key_maps_to_401(modal_app_module, monkeypatch):
    from fastapi import HTTPException

    resp = _FakeResp(401, {"detail": {"code": "invalid_api_key"}})
    authorize = _authorizer(modal_app_module, monkeypatch, resp)
    with pytest.raises(HTTPException) as exc:
        _run(authorize)
    assert exc.value.status_code == 401


def test_401_rejected_machine_bearer_maps_to_503(modal_app_module, monkeypatch):
    # A bare 401 (no invalid_api_key code) is our machine bearer being rejected -> serving misconfig,
    # surfaced as 503, NOT as the caller's invalid key.
    from fastapi import HTTPException

    for payload in ({"detail": "Unauthorized"}, None, ValueError("not json")):
        authorize = _authorizer(modal_app_module, monkeypatch, _FakeResp(401, payload))
        with pytest.raises(HTTPException) as exc:
            _run(authorize)
        assert exc.value.status_code == 503


@pytest.mark.parametrize("code", [403, 404])
def test_403_and_404_collapse_to_403(modal_app_module, monkeypatch, code):
    from fastapi import HTTPException

    authorize = _authorizer(modal_app_module, monkeypatch, _FakeResp(code, {}))
    with pytest.raises(HTTPException) as exc:
        _run(authorize)
    assert exc.value.status_code == 403


def test_backend_503_stays_503(modal_app_module, monkeypatch):
    # The core fix: a retryable backend 503 must NOT be masked as a 502.
    from fastapi import HTTPException

    authorize = _authorizer(modal_app_module, monkeypatch, _FakeResp(503, {"detail": "down"}))
    with pytest.raises(HTTPException) as exc:
        _run(authorize)
    assert exc.value.status_code == 503


def test_other_5xx_maps_to_503(modal_app_module, monkeypatch):
    # A backend 5xx is a RETRYABLE 503, never a 502 — a 502 a client/LB reads as permanent is what
    # turned transient backend blips (Supabase overloaded under an eval's auth stampede) into failed
    # rows. The cache+single-flight below is what makes this failure rare in the first place.
    from fastapi import HTTPException

    authorize = _authorizer(modal_app_module, monkeypatch, _FakeResp(500, {"detail": "boom"}))
    with pytest.raises(HTTPException) as exc:
        _run(authorize)
    assert exc.value.status_code == 503


def test_transport_error_fails_closed_503(modal_app_module, monkeypatch):
    from fastapi import HTTPException

    authorize = _authorizer(modal_app_module, monkeypatch, RuntimeError("connection refused"))
    with pytest.raises(HTTPException) as exc:
        _run(authorize)
    assert exc.value.status_code == 503


# ── Auth cache + single-flight (the actual 502 fix: don't stampede the backend auth path) ────────


def _counting_client(monkeypatch, responder, *, delay: float = 0.0):
    """Patch httpx.AsyncClient with one whose .post counts calls and returns/raises ``responder(n)``
    for the n-th call. ``delay`` sleeps inside .post so concurrent callers overlap (single-flight)."""
    import asyncio

    import httpx

    calls = {"n": 0}

    class _Client:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        async def post(self, _url: str, json: Any, headers: Any = None) -> _FakeResp:
            calls["n"] += 1
            n = calls["n"]
            if delay:
                await asyncio.sleep(delay)
            r = responder(n)
            if isinstance(r, Exception):
                raise r
            return r

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    return calls


def _new_authorizer(modal_app_module):
    return modal_app_module._build_chat_authorizer(
        SimpleNamespace(backend_url="https://backend.example.com", internal_key="machine-key")
    )


_QWEN = "Qwen/Qwen3.5-9B"
_INTERNAL_KEY = "fs-internal"


class _CountingPool:
    def __init__(self) -> None:
        self.generate_calls = 0

    async def generate(self, _base_model, payload, record, *, expected_checkpoint=None):
        self.generate_calls += 1
        return attest(
            record,
            {
                "text": "hi",
                "finish_reason": "stop",
                "prompt_token_ids": [1],
                "completion_token_ids": [2],
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cached_tokens_reported": False,
                "reasoning_tokens": 0,
                "request_id": payload.generation_id,
                "checkpoint": "",
            },
        )

    async def stream_generate(self, *_a, **_k):  # pragma: no cover - unused here
        yield {"type": "final", "finish_reason": "stop", "prompt_tokens": 1, "completion_tokens": 1}

    async def register(self, _base_model, _record) -> None:  # pragma: no cover - unused here
        return None

    async def unregister(
        self, _base_model, _adapter_id, expected_generation=None
    ) -> None:  # pragma: no cover - unused here
        return None


def _base_model_client(authorize, pool, *, usage_store=None) -> TestClient:
    record = AdapterRecord(
        adapter_id=_QWEN,
        repo_id=_QWEN,
        base_model=_QWEN,
        serve_base_model=True,
        thinking=False,
        org_id=None,
        status="ready",
    )
    builder = build_serving_app if usage_store is not None else build_offline_serving_app
    kwargs = {"usage_store": usage_store} if usage_store is not None else {}
    return TestClient(
        builder(
            pool,
            AdapterRouter([record]),
            internal_key=_INTERNAL_KEY,
            chat_authorizer=authorize,
            **kwargs,
        )
    )


def _chat(client: TestClient, **headers: str):
    return client.post(
        "/v1/chat/completions",
        json={"model": _QWEN, "messages": [{"role": "user", "content": "hi"}]},
        headers=headers,
    )


@pytest.mark.parametrize(
    "malformed",
    [
        pytest.param({}, id="missing-org-id"),
        pytest.param({"orgId": None}, id="null-org-id"),
        pytest.param({"orgId": ""}, id="empty-org-id"),
        pytest.param({"orgId": 123}, id="non-string-org-id"),
        pytest.param(ValueError("invalid json"), id="invalid-json"),
    ],
)
def test_malformed_200_fails_closed_without_dispatch_or_cache(
    modal_app_module, monkeypatch, malformed
):
    calls = _counting_client(
        monkeypatch,
        lambda n: _FakeResp(200, malformed if n == 1 else {"orgId": "org-1"}),
    )
    authorize = _new_authorizer(modal_app_module)
    pool = _CountingPool()
    store = RecordingUsageStore()

    with _base_model_client(authorize, pool, usage_store=store) as client:
        denied = _chat(client, Authorization="Bearer fs-user-key")
        assert denied.status_code == 503
        assert pool.generate_calls == 0

        allowed = _chat(client, Authorization="Bearer fs-user-key")
        internal = _chat(
            client,
            **{"X-Freesolo-Internal-Key": _INTERNAL_KEY, "X-Freesolo-Org-Id": "org-2"},
        )

    assert allowed.status_code == 200
    assert internal.status_code == 200
    assert pool.generate_calls == 2
    assert calls["n"] == 2
    assert len(store.finalized) == 2
    assert store.finalized[0].principal.orgId == "org-1"
    assert store.finalized[1].principal.kind == "trusted_internal"
    assert store.finalized[1].principal.orgId == "org-2"


def test_cancelled_waiter_does_not_cancel_shared_authorization(modal_app_module, monkeypatch):
    import asyncio

    calls = _counting_client(monkeypatch, lambda _n: _FakeResp(200, {"orgId": "org-1"}), delay=0.05)
    authorize = _new_authorizer(modal_app_module)

    async def run():
        waiters = [asyncio.create_task(authorize("key", "ad")) for _ in range(10)]
        await asyncio.sleep(0)
        waiters[0].cancel()
        return await asyncio.gather(*waiters, return_exceptions=True)

    results = asyncio.run(run())
    assert isinstance(results[0], asyncio.CancelledError)
    assert results[1:] == ["org-1"] * 9
    assert calls["n"] == 1


def test_successful_authorization_is_cached_within_ttl(modal_app_module, monkeypatch):
    import asyncio

    calls = _counting_client(monkeypatch, lambda _n: _FakeResp(200, {"orgId": "org-1"}))
    authorize = _new_authorizer(modal_app_module)

    async def run():
        first = await authorize("key", "ad")
        second = await authorize("key", "ad")  # cache hit -> no second backend call
        return first, second

    first, second = asyncio.run(run())
    assert first == second == "org-1"
    assert calls["n"] == 1


def test_concurrent_identical_lookups_coalesce_to_one_call(modal_app_module, monkeypatch):
    import asyncio

    # 50 concurrent identical lookups; the slow backend guarantees they all overlap in-flight.
    calls = _counting_client(monkeypatch, lambda _n: _FakeResp(200, {"orgId": "org-1"}), delay=0.05)
    authorize = _new_authorizer(modal_app_module)

    async def run():
        return await asyncio.gather(*[authorize("key", "ad") for _ in range(50)])

    orgs = asyncio.run(run())
    assert orgs == ["org-1"] * 50
    assert calls["n"] == 1  # single-flight collapsed the stampede into ONE backend call


def test_distinct_keys_are_not_coalesced(modal_app_module, monkeypatch):
    import asyncio

    calls = _counting_client(
        monkeypatch, lambda n: _FakeResp(200, {"orgId": f"org-{n}"}), delay=0.05
    )
    authorize = _new_authorizer(modal_app_module)

    async def run():
        return await asyncio.gather(authorize("k1", "ad"), authorize("k2", "ad"))

    asyncio.run(run())
    assert calls["n"] == 2  # different callers -> not coalesced


def test_failed_authorization_is_not_cached(modal_app_module, monkeypatch):
    import asyncio

    from fastapi import HTTPException

    # First lookup 500 (-> 503), second lookup 200: the failure must NOT be cached, so the second
    # call re-checks the backend and succeeds (a transient blip doesn't poison the cache).
    calls = _counting_client(
        monkeypatch,
        lambda n: _FakeResp(500, {}) if n == 1 else _FakeResp(200, {"orgId": "org-1"}),
    )
    authorize = _new_authorizer(modal_app_module)

    async def run():
        with pytest.raises(HTTPException) as exc:
            await authorize("key", "ad")
        assert exc.value.status_code == 503
        return await authorize("key", "ad")

    org = asyncio.run(run())
    assert org == "org-1"
    assert calls["n"] == 2


_EMPTY_SCOPE_DIGEST = hashlib.sha256(b"").hexdigest()


def test_expired_key_is_evicted_before_reauthorization(modal_app_module, monkeypatch):
    import asyncio
    import inspect

    from fastapi import HTTPException

    _counting_client(
        monkeypatch,
        lambda n: _FakeResp(200, {"orgId": "org-1"}) if n == 1 else _FakeResp(500, {}),
    )
    authorize = _new_authorizer(modal_app_module)
    cache = inspect.getclosurevars(authorize).nonlocals["_cache"]
    request_key = ("synthetic-expired-key", "adapter-1")
    cache_key = (
        hashlib.sha256(request_key[0].encode("utf-8")).hexdigest(),
        request_key[1],
        _EMPTY_SCOPE_DIGEST,
    )

    async def run() -> None:
        await authorize(*request_key)
        _, org_id = cache[cache_key]
        cache[cache_key] = (float("-inf"), org_id)
        with pytest.raises(HTTPException):
            await authorize(*request_key)

    asyncio.run(run())

    assert cache_key not in cache


def test_prune_removes_expired_keys_below_capacity(modal_app_module, monkeypatch):
    import asyncio
    import inspect

    _counting_client(monkeypatch, lambda _n: _FakeResp(200, {"orgId": "org-1"}))
    authorize = _new_authorizer(modal_app_module)
    closure = inspect.getclosurevars(authorize).nonlocals
    cache = closure["_cache"]
    clock = [0.0]
    monkeypatch.setattr(closure["time"], "monotonic", lambda: clock[0])
    expired_request = ("synthetic-expired-key", "adapter-1")
    current_request = ("synthetic-current-key", "adapter-2")
    expired_key = (
        hashlib.sha256(expired_request[0].encode("utf-8")).hexdigest(),
        expired_request[1],
        _EMPTY_SCOPE_DIGEST,
    )
    current_key = (
        hashlib.sha256(current_request[0].encode("utf-8")).hexdigest(),
        current_request[1],
        _EMPTY_SCOPE_DIGEST,
    )

    async def run() -> None:
        await authorize(*expired_request)
        clock[0] = modal_app_module._AUTH_CACHE_TTL_SECONDS + 1
        await authorize(*current_request)

    asyncio.run(run())

    assert expired_key not in cache
    assert current_key in cache


def test_cache_entry_expires_after_ttl(modal_app_module, monkeypatch):
    import asyncio

    # TTL 0 -> every cache entry is immediately stale, forcing a re-check on the next call.
    monkeypatch.setattr(modal_app_module, "_AUTH_CACHE_TTL_SECONDS", 0.0)
    calls = _counting_client(monkeypatch, lambda _n: _FakeResp(200, {"orgId": "org-1"}))
    authorize = _new_authorizer(modal_app_module)

    async def run():
        await authorize("key", "ad")
        await authorize("key", "ad")  # prior entry already expired -> a second backend call

    asyncio.run(run())
    assert calls["n"] == 2


def test_engine_secret_contains_only_engine_credentials(modal_app_module, monkeypatch: Any) -> None:
    for name in (
        "HF_TOKEN",
        "PLATFORM_BACKEND_URL",
        "FREESOLO_INTERNAL_KEY",
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
    ):
        monkeypatch.setenv(name, f"value-for-{name}")

    modal_app_module.modal.Secret.from_dict.reset_mock()
    modal_app_module._engine_secret()
    engine_values = modal_app_module.modal.Secret.from_dict.call_args.args[0]

    assert "FREESOLO_INTERNAL_KEY" not in engine_values
    assert "PLATFORM_BACKEND_URL" not in engine_values
    assert "SUPABASE_URL" not in engine_values
    assert "SUPABASE_SERVICE_ROLE_KEY" not in engine_values

    modal_app_module.modal.Secret.from_dict.reset_mock()
    modal_app_module._runtime_secret()
    router_values = modal_app_module.modal.Secret.from_dict.call_args.args[0]

    assert router_values["FREESOLO_INTERNAL_KEY"] == "value-for-FREESOLO_INTERNAL_KEY"
    assert router_values["PLATFORM_BACKEND_URL"] == "value-for-PLATFORM_BACKEND_URL"
    assert router_values["SUPABASE_URL"] == "value-for-SUPABASE_URL"
    assert router_values["SUPABASE_SERVICE_ROLE_KEY"] == "value-for-SUPABASE_SERVICE_ROLE_KEY"
    assert all(
        call.kwargs["secrets"] is modal_app_module.engine_secrets
        for call in modal_app_module.app.cls.call_args_list
    ), "engine classes must not receive the full router secret"
    assert (
        modal_app_module.app.function.call_args.kwargs["secrets"]
        is modal_app_module.runtime_secrets
    )
