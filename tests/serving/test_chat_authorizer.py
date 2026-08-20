"""Status mapping for the serving->backend chat authorizer (modal_app._build_chat_authorizer).

The authorizer POSTs to the backend's /api/serving/authorize and must translate the backend's
HTTP status into the right serving-side HTTPException:

  - 200                       -> allow (no raise)
  - 401 + code invalid_api_key-> 401 (the user's key is bad)
  - 401 without that code     -> 503 (our machine bearer was rejected: a serving misconfig, NOT the
                                 caller's key — must not be reported as an invalid user key)
  - 403 / 404                 -> 403 (collapse so we don't leak which adapters exist)
  - 503 / any other backend 5xx -> 503 (transient auth-lookup failure stays retryable; NEVER masked
                                 as a 502, which a client/LB may treat as permanent and not retry)
  - transport error           -> 503 (fail closed, never serve unauthorized)

It also caches a SUCCESSFUL authorization per (api_key, adapter_id) for a short TTL and coalesces
concurrent identical lookups into ONE backend call (single-flight), so an eval's many same-key
requests don't stampede the backend auth path into transient 5xx failures.

modal_app imports the `modal` SDK at module top (decorators run at import), which isn't installed
in the offline test env, so we stub it just enough to import and reach _build_chat_authorizer.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


def _passthrough_decorator(*_a: Any, **_k: Any):
    def deco(obj: Any) -> Any:
        return obj

    return deco


@pytest.fixture(scope="module")
def modal_app_module():
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

        async def post(self, _url: str, json: Any) -> _FakeResp:
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


def test_200_allows(modal_app_module, monkeypatch):
    authorize = _authorizer(modal_app_module, monkeypatch, _FakeResp(200, {"ok": True}))
    _run(authorize)  # must not raise


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

        async def post(self, _url: str, json: Any) -> _FakeResp:
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
