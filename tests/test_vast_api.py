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
    # No duration filter unless a deadline is threaded in (default off).
    assert "duration" not in q


def test_search_offers_applies_duration_filter(monkeypatch):
    # Codex Msvb0: a run whose wall cap exceeds an offer's remaining availability must not rent a
    # short-lived offer. When min_duration_seconds is set, the search adds Vast's documented
    # `duration` filter (seconds, "available at least this long from now") in the same operator-dict
    # form as the other numeric filters; 0/unset leaves it off.
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    calls = _capture_urlopen(monkeypatch, [{"offers": []}, {"offers": []}])
    vast_api.search_offers(24576, min_duration_seconds=7200.0)
    assert calls[0][2]["q"]["duration"] == {"gte": 7200.0}
    vast_api.search_offers(24576, min_duration_seconds=0)
    assert "duration" not in calls[1][2]["q"]


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


def test_create_instance_unreadable_response_is_ambiguous(monkeypatch):
    # Codex Msvbz: a 200 with a truncated / non-JSON body on the NON-IDEMPOTENT create may mean the
    # host already billed a contract while the RESPONSE leg failed. Such a failure (JSONDecodeError /
    # IncompleteRead — neither an OSError, so the _http retry wrapper doesn't catch/wrap them) must
    # surface as a VastApiError the walk classifies AMBIGUOUS, NOT escape raw past deploy_and_submit's
    # `except VastApiError` and leak the contract.
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    monkeypatch.setattr(vast_api.time, "sleep", lambda s: None)
    kwargs = {"image": "img", "disk_gb": 60, "env": {}, "onstart": "#!/bin/bash", "label": "flash-x"}

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
        # Codex MtrgJ: UnicodeDecodeError is a SIBLING of JSONDecodeError under ValueError, so the
        # JSONDecodeError clause alone would miss it and let it escape raw past the ambiguous reconcile.
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


def test_get_instance_none_on_404(monkeypatch):
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [_http_error(404)])
    assert vast_api.get_instance(777) is None


def test_get_instance_reraises_non_404_with_404ish_body(monkeypatch):
    # Codex Mr4sO: a non-404 4xx whose body embeds an id like "4040" must NOT be misread as a
    # disappearance/preemption. get_instance keys off the chained HTTPError status (is_not_found),
    # not a bare "404" substring, so this raises instead of returning None.
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [_http_error(400, b'{"error":"bad request for instance 4040"}')])
    with pytest.raises(vast_api.VastApiError):
        vast_api.get_instance(4040)


def test_create_error_is_ambiguous_classification():
    # Codex Mr72L: classify create_instance failures so the walk only reconciles-by-label on the
    # AMBIGUOUS ones (a billed contract may exist), and walks straight past DEFINITIVE rejections.
    from flash.providers.vast import api as vast_api

    def err(cause=None, msg="x"):
        e = vast_api.VastApiError(msg)
        if cause is not None:
            e.__cause__ = cause
        return e

    def http(code):
        return urllib.error.HTTPError("u", code, "m", None, io.BytesIO(b""))

    # DEFINITIVE (created nothing) -> not ambiguous
    assert vast_api.create_error_is_ambiguous(err(http(404))) is False
    assert vast_api.create_error_is_ambiguous(err(http(400))) is False
    assert vast_api.create_error_is_ambiguous(err(msg="rejected: {'success': False}")) is False
    # AMBIGUOUS (a contract may exist) -> reconcile by label
    assert vast_api.create_error_is_ambiguous(err(http(503))) is True
    assert vast_api.create_error_is_ambiguous(err(http(429))) is True  # Cursor MsA6e: rate-limit
    assert vast_api.create_error_is_ambiguous(err(urllib.error.URLError("timed out"))) is True
    assert vast_api.create_error_is_ambiguous(err(msg="...: no instance id in response: {}")) is True
    # Codex MtrgJ: json.loads on bytes with invalid UTF-8 raises UnicodeDecodeError (a SIBLING of
    # JSONDecodeError under ValueError, NOT caught by the JSONDecodeError clause) — an unreadable
    # response on the non-idempotent create, so it MUST be ambiguous (else the contract leaks).
    _utf8_err = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    assert vast_api.create_error_is_ambiguous(err(_utf8_err)) is True
    assert vast_api.create_error_is_ambiguous(_utf8_err) is True  # bare cause too (defensive)
    # Codex MsMPk: the _http RestClient also chains BARE socket errors (TimeoutError ==
    # socket.timeout, ConnectionError, generic OSError) — a RESPONSE-leg timeout of the non-idempotent
    # PUT /asks AFTER the host billed a contract surfaces as one of these, NOT a URLError. They MUST be
    # ambiguous (else the walk treats a real billed instance as a clean rejection and leaks it).
    assert vast_api.create_error_is_ambiguous(err(TimeoutError("read timed out"))) is True
    assert vast_api.create_error_is_ambiguous(err(ConnectionResetError("peer reset"))) is True
    assert vast_api.create_error_is_ambiguous(err(OSError("socket error"))) is True
    # Codex Msvbz: a 200 whose body is unreadable (truncated read / non-JSON) on the non-idempotent
    # create -> AMBIGUOUS. JSONDecodeError (a ValueError) and IncompleteRead (an HTTPException) are
    # NOT OSErrors, so they miss the branches above and must be classified explicitly — both when
    # create_instance wraps them as a VastApiError-from-cause AND if a bare one ever reaches here.
    # bare name: the local http() helper above shadows the module
    from http.client import IncompleteRead

    jde = json.JSONDecodeError("Expecting value", "x", 0)
    assert vast_api.create_error_is_ambiguous(err(jde)) is True  # wrapped (cause is JSONDecodeError)
    assert vast_api.create_error_is_ambiguous(jde) is True  # bare
    inc = IncompleteRead(b"partial")
    assert vast_api.create_error_is_ambiguous(err(inc)) is True  # wrapped (cause is IncompleteRead)
    assert vast_api.create_error_is_ambiguous(inc) is True  # bare


def test_destroy_instance_never_raises(monkeypatch):
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [{}])
    assert vast_api.destroy_instance(777) is True
    _capture_urlopen(monkeypatch, [_http_error(500)] * 3)
    assert vast_api.destroy_instance(777) is False


def test_destroy_instance_respects_success_flag(monkeypatch):
    """Codex MsXoJ: Vast's 200 DELETE carries a `success` bool — `success: false` means the box is
    still billable, so it must NOT be reported destroyed (destroy_run_instances/sweep_orphans would
    count it reaped and stop the immediate cleanup). A body without the key stays success (prior shape)."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [{"success": True}])
    assert vast_api.destroy_instance(5) is True
    _capture_urlopen(monkeypatch, [{"success": False}])
    assert vast_api.destroy_instance(5) is False
    _capture_urlopen(monkeypatch, [{"detail": "ok, no success key"}])
    assert vast_api.destroy_instance(5) is True


def test_destroy_instance_404_is_confirmed_gone(monkeypatch):
    """Codex: a 404 DELETE means the instance no longer exists (already destroyed / host-preempted) —
    a CONFIRMED non-billing state, so destroy_instance reports True. Otherwise the retry loop's
    pre-launch destroy would raise "unconfirmed teardown" on an already-gone box and fail the run
    instead of retrying. A non-404 4xx whose body merely embeds "404" stays unconfirmed (False)."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [_http_error(404)] * 3)
    assert vast_api.destroy_instance(9) is True  # genuine 404 -> gone -> confirmed
    # status-CODE-authoritative: a 400 whose body embeds an id like "4040" is NOT a 404 -> still False
    _capture_urlopen(monkeypatch, [_http_error(400, b'{"error":"bad id 4040"}')] * 3)
    assert vast_api.destroy_instance(4040) is False


def test_list_instances_paginates_every_page(monkeypatch):
    """Codex MsXoI: the v1 instances list is keyset-paginated (limit max 25; pass the prior page's
    `next_token` as `after_token`; `next_token` is null on the last page). list_instances must walk
    EVERY page — a flash orphan on a later page would otherwise never be seen by adoption / destroy /
    sweep and bill forever."""
    from flash.providers.vast import api as vast_api

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
    """Cursor MsaAk: a LATER page failing must not discard pages already fetched — adoption/teardown/
    sweep should still act on what we saw (a single-page list would have). A FIRST-page failure has
    nothing useful, so it re-raises and the callers' existing try/except skips, exactly as before."""
    from flash.providers.vast import api as vast_api

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
    from flash.providers.vast import api as vast_api

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
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [{"success": False}])  # no 'instances' list, no next_token
    with pytest.raises(vast_api.VastApiError):
        vast_api.list_instances(strict=True)
    # lenient default: a missing 'instances' list is just an empty page (prior best-effort behavior)
    _capture_urlopen(monkeypatch, [{"success": False}])
    assert vast_api.list_instances() == []
