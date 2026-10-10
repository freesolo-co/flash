"""Vast.ai REST client: auth, query shape, retry matrix (CPU-only; urllib mocked)."""

from __future__ import annotations

import http.client
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
    from flash.providers.vast.client import api as vast_api

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
    from flash.providers.vast.client.api import _CLIENT, VastApiError

    monkeypatch.delenv("VAST_API_KEY", raising=False)
    with pytest.raises(VastApiError, match="VAST_API_KEY"):
        _CLIENT.api_key()
    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    assert _CLIENT.api_key() == "vk-test"


def test_search_offers_query_shape(monkeypatch):
    from flash.providers.vast.client import api as vast_api

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
    # No duration filter unless a deadline is threaded in (default off).
    assert "duration" not in q


def test_search_offers_applies_exact_gpu_names_server_side(monkeypatch):
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(monkeypatch, [{"offers": []}])
    vast_api.search_offers(
        81920,
        max_vram_mb=81920,
        gpu_names=("H100 SXM", "H100 PCIE"),
    )

    query = calls[0][2]["q"]
    assert query["gpu_ram"] == {"gte": 81920, "lte": 81920}
    assert query["gpu_name"] == {"in": ["H100 SXM", "H100 PCIE"]}


def test_search_offers_applies_duration_filter(monkeypatch):
    # a run whose wall cap exceeds an offer's remaining availability must not rent a short-lived
    # offer. When min_duration_seconds is set, the search adds Vast's documented `duration` filter
    # (seconds, "available at least this long from now") in the same operator-dict form as the other
    # numeric filters; 0/unset leaves it off.
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(monkeypatch, [{"offers": []}, {"offers": []}])
    vast_api.search_offers(24576, min_duration_seconds=7200.0)
    assert calls[0][2]["q"]["duration"] == {"gte": 7200.0}
    vast_api.search_offers(24576, min_duration_seconds=0)
    assert "duration" not in calls[1][2]["q"]


def test_request_retries_5xx_and_429_then_succeeds(monkeypatch):
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(monkeypatch, [_http_error(503), _http_error(429), {"ok": True}])
    assert vast_api.request_with_retries("/instances/") == {"ok": True}
    assert len(calls) == 3


def test_request_4xx_raises_immediately_without_provider_body(monkeypatch):
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(monkeypatch, [_http_error(400, b'{"msg": "no such ask"}')])
    with pytest.raises(vast_api.VastApiError) as exc_info:
        vast_api.request_with_retries("/asks/123/", method="PUT", body={})
    assert len(calls) == 1  # no retry on 4xx
    assert "HTTP 400" in str(exc_info.value)
    assert "no such ask" not in str(exc_info.value)


def test_create_instance_success_and_rejection(monkeypatch):
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(
        monkeypatch,
        [
            {"success": True, "new_contract": 777},
            {"success": False, "error": "provider body secret"},
        ],
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
    with pytest.raises(vast_api.VastApiError, match="rejected") as exc_info:
        vast_api.create_instance(123, **kwargs)
    assert vast_api.create_error_is_ambiguous(exc_info.value) is False
    assert "provider body secret" not in str(exc_info.value)


@pytest.mark.parametrize("status", [404, 410])
def test_create_instance_non_2xx_explicit_false_is_rejected(monkeypatch, status):
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(
        monkeypatch,
        [_http_error(status, b'{"success": false, "error": "offer unavailable"}')],
    )
    kwargs = {
        "image": "img",
        "disk_gb": 60,
        "env": {},
        "onstart": "#!/bin/bash",
        "label": "flash-x",
    }

    with pytest.raises(vast_api.VastCreateRejected) as exc_info:
        vast_api.create_instance(123, **kwargs)
    assert vast_api.create_error_is_ambiguous(exc_info.value) is False


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (404, b"not-json"),
        (503, b'{"success": false, "error": "upstream unavailable"}'),
    ],
)
def test_create_instance_non_2xx_uncertain_response_is_ambiguous(monkeypatch, status, body):
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [_http_error(status, body)])
    kwargs = {
        "image": "img",
        "disk_gb": 60,
        "env": {},
        "onstart": "#!/bin/bash",
        "label": "flash-x",
    }

    with pytest.raises(vast_api.VastApiError) as exc_info:
        vast_api.create_instance(123, **kwargs)
    assert vast_api.create_error_is_ambiguous(exc_info.value) is True


@pytest.mark.parametrize("contract", [777, "not-an-int", [7], {"id": 7}])
def test_create_instance_non_2xx_contradictory_response_is_ambiguous(monkeypatch, contract):
    # any non-empty new_contract evidence in a false body is contradictory: the contract may
    # exist and bill, so it must be ambiguous even when the id is unparseable.
    import json as _json

    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    body = _json.dumps({"success": False, "new_contract": contract}).encode()
    _capture_urlopen(monkeypatch, [_http_error(410, body)])
    kwargs = {
        "image": "img",
        "disk_gb": 60,
        "env": {},
        "onstart": "#!/bin/bash",
        "label": "flash-x",
    }

    with pytest.raises(vast_api.VastAmbiguousCreate) as exc_info:
        vast_api.create_instance(123, **kwargs)
    assert vast_api.create_error_is_ambiguous(exc_info.value) is True
    if contract == 777:
        assert exc_info.value.contract_id == 777


def test_create_instance_non_2xx_long_false_body_is_still_rejected(monkeypatch):
    # restclient truncates the display message at 500 chars; the full body is preserved as
    # structured metadata, so a long definitive rejection must still classify as rejected.
    import json as _json

    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    body = _json.dumps({"success": False, "error": "x" * 800}).encode()
    _capture_urlopen(monkeypatch, [_http_error(410, body)])
    kwargs = {
        "image": "img",
        "disk_gb": 60,
        "env": {},
        "onstart": "#!/bin/bash",
        "label": "flash-x",
    }

    with pytest.raises(vast_api.VastCreateRejected) as exc_info:
        vast_api.create_instance(123, **kwargs)
    assert vast_api.create_error_is_ambiguous(exc_info.value) is False


def test_create_instance_missing_key_is_not_sent(monkeypatch):
    # a missing api key raises locally before any http request: the create was provably never
    # sent, so it must surface as an actionable config error, not a possibly-billed create.
    from flash.providers.vast.client import api as vast_api

    monkeypatch.delenv("VAST_API_KEY", raising=False)
    kwargs = {
        "image": "img",
        "disk_gb": 60,
        "env": {},
        "onstart": "#!/bin/bash",
        "label": "flash-x",
    }

    with pytest.raises(vast_api.VastCreateNotSent) as exc_info:
        vast_api.create_instance(123, **kwargs)
    assert vast_api.create_error_is_ambiguous(exc_info.value) is False


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"detail": "offer state unknown"},
        {"error": "upstream response lost"},
        {"success": None},
        {"success": 0},
        {"success": False, "new_contract": 777},
        {"success": 1, "new_contract": 777},
        {"success": "true", "new_contract": 777},
    ],
)
def test_create_instance_malformed_response_is_ambiguous(monkeypatch, body):
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [body])
    kwargs = {
        "image": "img",
        "disk_gb": 60,
        "env": {},
        "onstart": "#!/bin/bash",
        "label": "flash-x",
    }

    with pytest.raises(vast_api.VastAmbiguousCreate) as exc_info:
        vast_api.create_instance(123, **kwargs)
    assert vast_api.create_error_is_ambiguous(exc_info.value) is True
    assert "provider body secret" not in str(exc_info.value)


def test_create_instance_diagnostics_never_render_opaque_response_body(monkeypatch):
    import traceback

    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    private = "opaque-private-sentinel-7f3c"
    kwargs = {
        "image": "img",
        "disk_gb": 60,
        "env": {},
        "onstart": "#!/bin/bash",
        "label": "flash-x",
    }

    for response, expected in (
        ({"success": False, "error": private}, vast_api.VastCreateRejected),
        ({"success": None, "error": private}, vast_api.VastAmbiguousCreate),
    ):
        _capture_urlopen(monkeypatch, [response])
        with pytest.raises(expected) as exc_info:
            vast_api.create_instance(123, **kwargs)
        detail = str(exc_info.value)
        assert private not in detail
        assert "create_instance(123)" in detail

    private_http = urllib.error.HTTPError(
        "https://console.vast.ai/api/v0/asks/123/",
        409,
        private,
        None,
        io.BytesIO(json.dumps({"error": private}).encode()),
    )
    _capture_urlopen(monkeypatch, [private_http])
    with pytest.raises(vast_api.VastAmbiguousCreate) as exc_info:
        vast_api.create_instance(123, **kwargs)
    detail = str(exc_info.value)
    formatted = "".join(traceback.format_exception(exc_info.type, exc_info.value, exc_info.tb))
    assert private not in detail
    assert private not in formatted
    assert "HTTP 409" in detail
    assert "/v0/asks/123/" in detail


def test_create_instance_is_not_retried(monkeypatch):
    """Fix #4: ``PUT /asks/{id}`` is non-idempotent (each success rents a NEW instance),
    so a transient failure must NOT be blindly retried — a retry on a timeout where Vast
    already accepted the first request would double-provision (a billing leak)."""
    from flash.providers.vast.client import api as vast_api

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


def test_create_instance_unreadable_response_is_ambiguous(monkeypatch):
    # a 200 with a truncated / non-JSON body on the NON-IDEMPOTENT create may mean the host already
    # billed a contract while the RESPONSE leg failed. Such a failure (JSONDecodeError /
    # IncompleteRead — neither an OSError, so the _http retry wrapper doesn't catch/wrap them) must
    # surface as a VastApiError the walk classifies AMBIGUOUS, NOT escape raw past
    # deploy_and_submit's `except VastApiError` and leak the contract.
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast_api.time, "sleep", lambda s: None)
    kwargs = {
        "image": "img",
        "disk_gb": 60,
        "env": {},
        "onstart": "#!/bin/bash",
        "label": "flash-x",
    }

    class _NonJsonResp:  # 200 with a non-JSON body -> json.loads raises JSONDecodeError
        def read(self):
            return b"<html>502 Bad Gateway (truncated)</html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _ReadFailsResp:  # 200 whose body read truncates -> http.client.IncompleteRead
        def read(self):
            raise http.client.IncompleteRead(b"partial")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _InvalidUtf8Resp:  # 200 with invalid UTF-8 -> json.loads(bytes) raises UnicodeDecodeError
        # UnicodeDecodeError is a SIBLING of JSONDecodeError under ValueError, so the
        # JSONDecodeError clause alone would miss it and let it escape raw past the ambiguous
        # reconcile.
        def read(self):
            return b'\xff\xfe{"new_contract": 1}'  # leading invalid-UTF8 bytes

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    for resp in (_NonJsonResp(), _ReadFailsResp(), _InvalidUtf8Resp()):
        monkeypatch.setattr(
            vast_api.urllib.request, "urlopen", lambda req, timeout=None, _r=resp: _r
        )
        with pytest.raises(vast_api.VastApiError, match="unreadable") as ei:
            vast_api.create_instance(123, **kwargs)
        # the wrapped error is classified AMBIGUOUS so deploy_and_submit reconciles by label
        assert vast_api.create_error_is_ambiguous(ei.value) is True


def test_create_instance_unparseable_new_contract_is_ambiguous(monkeypatch):
    # Codex: a 200 ``success`` body whose new_contract is TRUTHY but non-numeric (malformed id) on the
    # NON-IDEMPOTENT create means Vast may have accepted/billed a contract we can't use as a handle. A
    # bare int() ValueError here would escape past deploy_and_submit's ``except VastApiError`` and skip
    # the ambiguous-create reconcile (leaking the contract). It must surface as a VastApiError the walk
    # classifies AMBIGUOUS, exactly like a ``success`` body that carried no contract id at all.
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    kwargs = {
        "image": "img",
        "disk_gb": 60,
        "env": {},
        "onstart": "#!/bin/bash",
        "label": "flash-x",
    }
    # "²" is a unicode digit str.isdigit() accepts but int() rejects — the parser must stay
    # total and non-throwing so it flows to ambiguous instead of escaping as a valueerror.
    for bad in ("not-a-number", {"id": 1}, [7], "²", True, -3, 0):
        _capture_urlopen(monkeypatch, [{"success": True, "new_contract": bad}])
        with pytest.raises(vast_api.VastAmbiguousCreate, match="no usable instance id") as ei:
            vast_api.create_instance(123, **kwargs)
        assert vast_api.create_error_is_ambiguous(ei.value) is True


def test_get_instance_none_on_404(monkeypatch):
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [_http_error(404)])
    assert vast_api.get_instance(777) is None


def test_get_instance_reraises_non_404_with_404ish_body(monkeypatch):
    # a non-404 4xx whose body embeds an id like "4040" must NOT be misread as a
    # disappearance/preemption. get_instance keys off the chained HTTPError status (is_not_found),
    # not a bare "404" substring, so this raises instead of returning None.
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    private_body = "bad request for instance 4040"
    _capture_urlopen(
        monkeypatch,
        [_http_error(400, json.dumps({"error": private_body}).encode())],
    )
    with pytest.raises(vast_api.VastApiError) as exc_info:
        vast_api.get_instance(4040)
    assert private_body not in str(exc_info.value)


def test_get_instance_raises_on_success_false_envelope(monkeypatch):
    # Codex 3519040494: a 200 body that is a success:false error envelope (no "instances" key) must
    # NOT be returned as instance detail — the poller would read it as a live-but-"unknown" instance
    # (.get("actual_status") is None) and RESET its missing-streak, masking a real disappearance. It
    # raises so the poll loop counts it as a bounded, retryable poll_error instead of a false read.
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    private_key = "opaque_provider_key_sentinel"
    private_value = "opaque-provider-response-sentinel"
    _capture_urlopen(monkeypatch, [{"success": False, private_key: private_value}])
    with pytest.raises(vast_api.VastApiError) as exc_info:
        vast_api.get_instance(4040)
    detail = str(exc_info.value)
    assert private_key not in detail
    assert private_value not in detail
    assert "get_instance(4040)" in detail
    assert "classification=error_envelope" in detail
    assert "returned_type=dict" in detail


def test_get_instance_returns_dict_and_gone_signal(monkeypatch):
    # The success:false raise is narrow: a real {"instances": {...}} detail still returns the inst,
    # and the {"instances": null} "gone" signal still returns None (not an error).
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [{"instances": {"id": 5, "actual_status": "running"}}])
    assert vast_api.get_instance(5) == {"id": 5, "actual_status": "running"}
    _capture_urlopen(monkeypatch, [{"instances": None}])
    assert vast_api.get_instance(5) is None


def test_create_error_is_ambiguous_classification():
    # classification is now purely by type: only the typed definitive outcomes (explicit
    # success:false rejection, local not-sent failure) permit walking to another offer.
    # representative untyped errors stand in for the http/socket/decode failures that used to
    # be cause-inspected — all of them must stay ambiguous because the request may have landed.
    from flash.providers.vast.client import api as vast_api

    def err(cause=None, msg="x"):
        e = vast_api.VastApiError(msg)
        if cause is not None:
            e.__cause__ = cause
        return e

    assert vast_api.create_error_is_ambiguous(vast_api.VastCreateRejected("success:false")) is False
    assert vast_api.create_error_is_ambiguous(vast_api.VastCreateNotSent("no api key")) is False
    assert (
        vast_api.create_error_is_ambiguous(vast_api.VastAmbiguousCreate("no instance id")) is True
    )
    assert vast_api.create_error_is_ambiguous(err(msg="untyped rejection")) is True
    assert (
        vast_api.create_error_is_ambiguous(
            err(urllib.error.HTTPError("u", 503, "m", None, io.BytesIO(b"")))
        )
        is True
    )
    assert vast_api.create_error_is_ambiguous(err(TimeoutError("read timed out"))) is True
    assert (
        vast_api.create_error_is_ambiguous(json.JSONDecodeError("Expecting value", "x", 0)) is True
    )


def test_destroy_instance_never_raises(monkeypatch):
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [{"success": True}])
    assert vast_api.destroy_instance(777) is True
    _capture_urlopen(monkeypatch, [_http_error(500)] * 3)
    assert vast_api.destroy_instance(777) is False


def test_destroy_instance_respects_success_flag(monkeypatch):
    """only explicit success:true confirms deletion; explicit false remains unconfirmed."""
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [{"success": True}])
    assert vast_api.destroy_instance(5) is True
    _capture_urlopen(monkeypatch, [{"success": False}])
    assert vast_api.destroy_instance(5) is False


@pytest.mark.parametrize(
    "body",
    [
        {},
        [],
        {"detail": "delete state unknown"},
        {"error": "permission denied"},
        {"success": None},
        {"success": 0},
        {"success": 1},
        {"success": "true"},
    ],
)
def test_destroy_instance_malformed_response_is_unconfirmed(monkeypatch, body):
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [body, {"instances": {"id": 5, "actual_status": "running"}}])
    assert vast_api.destroy_instance(5) is False


def test_destroy_instance_exact_followup_can_confirm_absence(monkeypatch):
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(monkeypatch, [{}, {"instances": None}])
    assert vast_api.destroy_instance(5) is True
    assert [call[0] for call in calls] == ["DELETE", "GET"]


def test_destroy_instance_followup_error_envelope_is_unconfirmed(monkeypatch):
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [{}, {"instances": None, "detail": "lookup failed"}])
    assert vast_api.destroy_instance(5) is False


def test_destroy_instance_unreadable_body_is_unconfirmed(monkeypatch):
    from flash.providers.vast.client import api as vast_api

    class _UnreadableResponse:
        def read(self):
            return b"<html>upstream error</html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    responses = iter(
        [
            _UnreadableResponse(),
            _UnreadableResponse(),
            _UnreadableResponse(),
            _FakeResponse({"instances": {"id": 5}}),
        ]
    )
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.get_method())
        return next(responses)

    monkeypatch.setattr(vast_api.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(vast_api.time, "sleep", lambda seconds: None)
    assert vast_api.destroy_instance(5) is False
    assert calls == ["DELETE", "DELETE", "DELETE", "GET"]


def test_destroy_instance_permission_failure_is_unconfirmed(monkeypatch):
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(monkeypatch, [_http_error(403, b'{"detail":"forbidden"}')])
    assert vast_api.destroy_instance(9) is False
    assert len(calls) == 1


def test_destroy_instance_404_is_confirmed_gone(monkeypatch):
    """Codex: a 404 DELETE means the instance no longer exists (already destroyed / host-preempted) —
    a CONFIRMED non-billing state, so destroy_instance reports True. Otherwise the retry loop's
    pre-launch destroy would raise "unconfirmed teardown" on an already-gone box and fail the run
    instead of retrying. A non-404 4xx whose body merely embeds "404" stays unconfirmed (False)."""
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [_http_error(404)] * 3)
    assert vast_api.destroy_instance(9) is True  # genuine 404 -> gone -> confirmed
    # status-CODE-authoritative: a 400 whose body embeds an id like "4040" is NOT a 404 -> still False
    _capture_urlopen(monkeypatch, [_http_error(400, b'{"error":"bad id 4040"}')] * 3)
    assert vast_api.destroy_instance(4040) is False


def test_list_instances_paginates_every_page(monkeypatch):
    """The v1 instances list is keyset-paginated (limit max 25; pass the prior page's
    `next_token` as `after_token`; `next_token` is null on the last page). list_instances must walk
    EVERY page — a flash orphan on a later page would otherwise never be seen by adoption / destroy /
    sweep and bill forever."""
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(
        monkeypatch,
        [
            {"instances": [{"id": 1}, {"id": 2}], "next_token": "tok2"},
            {"instances": [{"id": 3}], "next_token": None},
        ],
    )
    out = vast_api.list_instances()
    assert [i["id"] for i in out] == [1, 2, 3]  # both pages collected
    assert len(calls) == 2  # stopped once next_token went null
    assert calls[0][1].endswith("/v1/instances/")  # page 1 is the bare path
    assert "after_token=tok2" in calls[1][1]  # page 2 carries the cursor


def test_list_instances_returns_partial_on_later_page_error(monkeypatch):
    """A LATER page failing must not discard pages already fetched — adoption/teardown/
    sweep should still act on what we saw (a single-page list would have). A FIRST-page failure has
    nothing useful, so it re-raises and the callers' existing try/except skips, exactly as before."""
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    # page 1 ok, page 2 errors after exhausting retries -> partial list of page 1
    _capture_urlopen(
        monkeypatch,
        [{"instances": [{"id": 1}], "next_token": "tok2"}] + [_http_error(500)] * 5,
    )
    assert [i["id"] for i in vast_api.list_instances()] == [1]
    # first page errors -> nothing collected -> propagates (callers catch and skip)
    _capture_urlopen(monkeypatch, [_http_error(500)] * 5)
    with pytest.raises(vast_api.VastApiError):
        vast_api.list_instances()


def test_list_instances_strict_raises_on_truncated_listing(monkeypatch):
    """Cursor: a caller that draws a conclusion from an ABSENCE (run_instances_remaining: "no instance
    for this run remains") must NOT accept a partial page set — an unseen later page could hide the very
    instance it is ruling out. strict=True raises on ANY incompleteness instead of returning a truncated
    list, so the caller treats it as "could not confirm" (and defers) rather than a false clear."""
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    # page 1 ok, page 2 errors: the LENIENT default returns partial, but strict must RAISE.
    _capture_urlopen(
        monkeypatch,
        [{"instances": [{"id": 1}], "next_token": "tok2"}] + [_http_error(500)] * 5,
    )
    with pytest.raises(vast_api.VastApiError):
        vast_api.list_instances(strict=True)
    # a COMPLETE listing (next_token exhausted) still returns normally under strict.
    _capture_urlopen(monkeypatch, [{"instances": [{"id": 1}], "next_token": None}])
    assert [i["id"] for i in vast_api.list_instances(strict=True)] == [1]


def test_list_instances_strict_rejects_non_list_instances_page(monkeypatch):
    """Codex: a 200 whose body lacks an `instances` LIST (e.g. an error envelope `{"success": false}`
    with no `next_token`) must NOT fall through as a complete empty page in strict mode — a strict caller
    would read it as 'no instance remains' and act on that false clear. strict raises; the lenient
    default still treats it as an (empty) page so the best-effort sweeps are unchanged."""
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [{"success": False}])  # no 'instances' list, no next_token
    with pytest.raises(vast_api.VastApiError):
        vast_api.list_instances(strict=True)
    # lenient default: a missing 'instances' list is just an empty page (prior best-effort behavior)
    _capture_urlopen(monkeypatch, [{"success": False}])
    assert vast_api.list_instances() == []


def test_search_offers_num_gpus_threads_into_query(monkeypatch):
    """num_gpus > 1 filters to multi-card hosts; the default (1) is exercised elsewhere."""
    from flash.providers.vast.client import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(monkeypatch, [{"offers": [{"id": 1}]}])
    vast_api.search_offers(24576, num_gpus=2, limit=10)
    _method, _url, body, _auth = calls[0]
    assert body["q"]["num_gpus"] == {"eq": 2}
