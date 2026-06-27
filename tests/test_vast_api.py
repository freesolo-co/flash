"""Vast.ai REST client: auth, query shape, retry matrix (CPU-only; urllib mocked)."""

from __future__ import annotations

import io
import json
import urllib.error

import pytest


def _http_error(code: int, body: bytes = b""):
    return urllib.error.HTTPError(
        url="https://console.vast.ai/api/v0/x", code=code, msg="err", hdrs=None, fp=io.BytesIO(body)
    )


class _FakeResponse:
    def __init__(self, payload: dict):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_urlopen(monkeypatch, responses):
    """Mock urlopen; returns the list of (method, url, body_dict, auth_header) captured."""
    from flash.providers.vast import api as vast_api

    calls = []
    seq = iter(responses)

    def fake_urlopen(req, timeout=None):
        out = next(seq)
        calls.append(
            (
                req.get_method(),
                req.full_url,
                json.loads(req.data) if req.data else None,
                req.get_header("Authorization"),
            )
        )
        if isinstance(out, Exception):
            raise out
        return _FakeResponse(out)

    monkeypatch.setattr(vast_api.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(vast_api.time, "sleep", lambda s: None)
    return calls


def test_api_key_env_only(monkeypatch):
    from flash.providers.vast.api import VastApiError, _api_key

    monkeypatch.delenv("VAST_API_KEY", raising=False)
    with pytest.raises(VastApiError, match="VAST_API_KEY"):
        _api_key()
    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    assert _api_key() == "vk-test"


def test_search_offers_query_shape(monkeypatch):
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(monkeypatch, [{"offers": [{"id": 1}]}])
    out = vast_api.search_offers(24576, min_disk_gb=60, min_reliability=0.97, limit=10)
    assert out == [{"id": 1}]
    method, url, body, auth = calls[0]
    assert method == "PUT"
    assert url.endswith("/search/asks/")
    assert auth == "Bearer vk-test"
    q = body["q"]
    # The non-negotiable filters: verified DATACENTER hosts, rentable, 1 GPU. Community/marketplace
    # hosts are filtered out server-side (datacenter==True) — they're rejected downstream anyway
    # (run secrets ship to the box), so filtering server-side keeps the price-sorted page full of
    # usable datacenter offers; hosting_type + the reliability floor are still re-checked downstream.
    assert q["verified"] == {"eq": True}
    assert q["datacenter"] == {"eq": True}
    assert q["rentable"] == {"eq": True}
    assert q["num_gpus"] == {"eq": 1}
    assert q["gpu_ram"] == {"gte": 24576}
    assert q["disk_space"] == {"gte": 60.0}
    assert q["reliability2"] == {"gte": 0.97}
    assert q["order"] == [["dph_total", "asc"]]


def test_request_retries_5xx_and_429_then_succeeds(monkeypatch):
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(monkeypatch, [_http_error(503), _http_error(429), {"ok": True}])
    assert vast_api.request_with_retries("/instances/") == {"ok": True}
    assert len(calls) == 3


def test_request_4xx_raises_immediately_with_body(monkeypatch):
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(monkeypatch, [_http_error(400, b'{"msg": "no such ask"}')])
    with pytest.raises(vast_api.VastApiError, match="no such ask"):
        vast_api.request_with_retries("/asks/123/", method="PUT", body={})
    assert len(calls) == 1  # no retry on 4xx


def test_create_instance_success_and_rejection(monkeypatch):
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(
        monkeypatch, [{"success": True, "new_contract": 777}, {"success": False, "error": "taken"}]
    )
    kwargs = {
        "image": "img",
        "disk_gb": 60,
        "env": {"A": "1"},
        "onstart": "#!/bin/bash",
        "label": "flash-x",
    }
    assert vast_api.create_instance(123, **kwargs) == 777
    method, url, body, _ = calls[0]
    assert method == "PUT"
    assert url.endswith("/asks/123/")
    assert body["label"] == "flash-x"
    assert body["env"] == {"A": "1"}
    with pytest.raises(vast_api.VastApiError, match="rejected"):
        vast_api.create_instance(123, **kwargs)


def test_create_instance_is_not_retried(monkeypatch):
    """Fix #4: ``PUT /asks/{id}`` is non-idempotent (each success rents a NEW instance),
    so a transient failure must NOT be blindly retried — a retry on a timeout where Vast
    already accepted the first request would double-provision (a billing leak)."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    # First call raises a transient 503; if create were retried it would hit the second
    # (success) response and silently double-provision. It must surface the failure on
    # the FIRST attempt instead.
    calls = _capture_urlopen(
        monkeypatch, [_http_error(503), {"success": True, "new_contract": 999}]
    )
    kwargs = {
        "image": "img",
        "disk_gb": 60,
        "env": {},
        "onstart": "#!/bin/bash",
        "label": "flash-x",
    }
    with pytest.raises(vast_api.VastApiError):
        vast_api.create_instance(123, **kwargs)
    assert len(calls) == 1  # exactly one create attempt, never a retry


def test_get_instance_none_on_404(monkeypatch):
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [_http_error(404)])
    assert vast_api.get_instance(777) is None


def test_destroy_instance_never_raises(monkeypatch):
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [{}])
    assert vast_api.destroy_instance(777) is True
    _capture_urlopen(monkeypatch, [_http_error(500)] * 3)
    assert vast_api.destroy_instance(777) is False
