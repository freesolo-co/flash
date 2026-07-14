"""Extra Vast.ai REST client coverage: instance_logs polling, and the malformed-response
edge branches of create_instance / get_instance / list_instances (CPU-only; urllib mocked)."""

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
    """A 200 whose body is the JSON encoding of ``payload`` (dict OR list)."""

    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture_urlopen(monkeypatch, responses):
    """Mock the REST transport urlopen; each response is a payload to return or an Exception to raise."""
    from flash.providers.vast import api as vast_api

    calls = []
    seq = iter(responses)

    def fake_urlopen(req, timeout=None):
        out = next(seq)
        calls.append((req.get_method(), req.full_url, json.loads(req.data) if req.data else None))
        if isinstance(out, Exception):
            raise out
        return _FakeResponse(out)

    monkeypatch.setattr(vast_api.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(vast_api.time, "sleep", lambda s: None)
    return calls


class _LogResp:
    """A log-fetch HTTP response usable as a context manager (bytes body)."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_log_urlopen(monkeypatch, seq):
    """Mock the direct ``urlopen(result_url)`` log poll; items are bytes (a body) or an Exception."""
    from flash.providers.vast import api as vast_api

    it = iter(seq)

    def fake(url, timeout=None):
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return _LogResp(item)

    monkeypatch.setattr(vast_api.urllib.request, "urlopen", fake)
    monkeypatch.setattr(vast_api.time, "sleep", lambda s: None)


# ---------------------------------------------------------------------------
# create_instance: success body carrying no usable id (line 150)
# ---------------------------------------------------------------------------
def test_create_instance_success_body_without_contract_is_ambiguous(monkeypatch):
    """A ``success: True`` body whose ``new_contract`` is falsy/missing (0 / None / absent) on the
    NON-IDEMPOTENT create means Vast accepted the create but returned no handle — a contract may be
    billing. It must raise VastAmbiguousCreate (classified ambiguous), not a plain VastApiError, so
    the caller reconciles by label instead of leaking the box."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    kwargs = {
        "image": "img",
        "disk_gb": 60,
        "env": {},
        "onstart": "#!/bin/bash",
        "label": "flash-x",
    }
    for body in (
        {"success": True},
        {"success": True, "new_contract": 0},
        {"success": True, "new_contract": None},
    ):
        _capture_urlopen(monkeypatch, [body])
        with pytest.raises(vast_api.VastApiError, match="returned no instance id") as ei:
            vast_api.create_instance(123, **kwargs)
        assert isinstance(ei.value, vast_api.VastAmbiguousCreate)
        assert vast_api.create_error_is_ambiguous(ei.value) is True


# ---------------------------------------------------------------------------
# get_instance: non-envelope dict without "instances", and non-dict body (lines 215-216)
# ---------------------------------------------------------------------------
def test_get_instance_returns_bare_dict_without_instances_key(monkeypatch):
    """A 200 dict that has no ``instances`` key and is NOT a ``success: false`` error envelope is
    passed through as-is (best-effort detail) rather than raising."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [{"foo": "bar"}])
    assert vast_api.get_instance(7) == {"foo": "bar"}
    # success is truthy (not the False error-envelope sentinel) -> still returned, not raised
    _capture_urlopen(monkeypatch, [{"success": True, "actual_status": "loading"}])
    assert vast_api.get_instance(7) == {"success": True, "actual_status": "loading"}


def test_get_instance_none_when_response_not_a_dict(monkeypatch):
    """A 200 whose top-level JSON is not an object (e.g. a bare list) is treated as 'no detail' ->
    None, never crashing on a ``.get`` of a non-dict."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [[1, 2, 3]])
    assert vast_api.get_instance(7) is None


# ---------------------------------------------------------------------------
# list_instances: a non-dict page (lines 251-253) and the page-cap runaway guard (line 266)
# ---------------------------------------------------------------------------
def test_list_instances_non_dict_page_strict_raises(monkeypatch):
    """A page whose body is not a dict (e.g. a bare JSON list) is an incomplete listing: strict must
    raise rather than silently treating it as 'nothing more'."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(monkeypatch, [[1, 2, 3]])
    with pytest.raises(vast_api.VastApiError, match="non-dict page"):
        vast_api.list_instances(strict=True)


def test_list_instances_non_dict_page_lenient_breaks_with_partial(monkeypatch):
    """The lenient default stops at a non-dict later page and returns what was already collected
    (a best-effort sweep still acts on the instances it saw)."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    _capture_urlopen(
        monkeypatch,
        [{"instances": [{"id": 1}], "next_token": "tok2"}, [9, 9, 9]],
    )
    assert [i["id"] for i in vast_api.list_instances()] == [1]


def test_list_instances_strict_raises_when_page_cap_exceeded(monkeypatch):
    """If the keyset walk never exhausts ``next_token`` and falls off the 200-page runaway guard with
    more pages pending, a strict caller must treat the listing as incomplete and raise."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setenv("VAST_API_KEY", "vk-test")
    pages = {"n": 0}

    def always_more(req, timeout=None):
        pages["n"] += 1
        return _FakeResponse({"instances": [], "next_token": "keep-going"})

    monkeypatch.setattr(vast_api.urllib.request, "urlopen", always_more)
    monkeypatch.setattr(vast_api.time, "sleep", lambda s: None)
    with pytest.raises(vast_api.VastApiError, match="exceeded the page cap"):
        vast_api.list_instances(strict=True)
    assert pages["n"] == 200  # bounded by the runaway guard, not infinite


# ---------------------------------------------------------------------------
# instance_logs: request -> poll the result URL (lines 276-299)
# ---------------------------------------------------------------------------
def test_instance_logs_returns_body_on_first_poll(monkeypatch):
    """Happy path: request_logs yields a result_url, the first fetch has a non-empty body -> it is
    returned verbatim (decoded)."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setattr(
        vast_api, "request_with_retries", lambda *a, **k: {"result_url": "http://logs/x"}
    )
    _fake_log_urlopen(monkeypatch, [b"boot ok\ntrainer started\n"])
    assert vast_api.instance_logs(42) == "boot ok\ntrainer started\n"


def test_instance_logs_none_without_result_url(monkeypatch):
    """No usable result_url -> None. Covers both an empty/absent url in the dict and a non-dict
    response from the logs request."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setattr(vast_api, "request_with_retries", lambda *a, **k: {"other": "field"})
    assert vast_api.instance_logs(42) is None
    monkeypatch.setattr(vast_api, "request_with_retries", lambda *a, **k: None)
    assert vast_api.instance_logs(42) is None


def test_instance_logs_none_on_non_404_http_error(monkeypatch):
    """A non-404 HTTP error while fetching the result URL is terminal (not 'not materialized yet')
    -> stop polling and return None."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setattr(
        vast_api, "request_with_retries", lambda *a, **k: {"result_url": "http://logs/x"}
    )
    _fake_log_urlopen(monkeypatch, [_http_error(500)])
    assert vast_api.instance_logs(42) is None


def test_instance_logs_polls_past_404_and_empty_then_returns_body(monkeypatch):
    """404 means 'log file not materialized yet' and an empty body means 'nothing yet' — both keep
    polling; a later non-empty body is returned. Exercises the 404-continue and empty-body branches."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setattr(
        vast_api, "request_with_retries", lambda *a, **k: {"result_url": "http://logs/x"}
    )
    # poll 1: 404 (not ready) -> continue; poll 2: whitespace-only body -> continue; poll 3: real logs
    _fake_log_urlopen(monkeypatch, [_http_error(404), b"   \n", b"the real logs"])
    assert vast_api.instance_logs(42) == "the real logs"


def test_instance_logs_caps_requests_and_sleep_at_run_deadline(monkeypatch):
    from flash.providers.vast import api as vast_api

    clock = {"now": 100.0}
    request_kwargs = {}
    fetch_timeouts = []
    sleeps = []

    def request(*_args, **kwargs):
        request_kwargs.update(kwargs)
        return {"result_url": "http://logs/x"}

    def urlopen(_url, timeout=None):
        fetch_timeouts.append(timeout)
        return _LogResp(b"")

    def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(vast_api, "request_with_retries", request)
    monkeypatch.setattr(vast_api.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(vast_api.time, "time", lambda: clock["now"])
    monkeypatch.setattr(vast_api.time, "sleep", sleep)

    assert vast_api.instance_logs(42, deadline_at=101.0) is None
    assert request_kwargs["deadline_at"] == 101.0
    assert fetch_timeouts == [1.0]
    assert sleeps == [1.0]


def test_instance_logs_none_when_deadline_exhausted(monkeypatch):
    """If the 20s poll deadline has already elapsed before the first fetch, the loop body never runs
    and the function returns None (no fetch attempted)."""
    from flash.providers.vast import api as vast_api

    monkeypatch.setattr(
        vast_api, "request_with_retries", lambda *a, **k: {"result_url": "http://logs/x"}
    )

    # First time() call sets deadline = now + 20; the loop-condition time() jumps well past it.
    stamps = [1000.0, 9999.0]

    def fake_time():
        return stamps.pop(0) if len(stamps) > 1 else stamps[0]

    monkeypatch.setattr(vast_api.time, "time", fake_time)
    fetched = {"n": 0}

    def never(url, timeout=None):
        fetched["n"] += 1
        return _LogResp(b"unreached")

    monkeypatch.setattr(vast_api.urllib.request, "urlopen", never)
    monkeypatch.setattr(vast_api.time, "sleep", lambda s: None)
    assert vast_api.instance_logs(42) is None
    assert fetched["n"] == 0  # deadline already past -> no fetch


def test_instance_logs_never_raises_when_request_fails(monkeypatch):
    """instance_logs is best-effort: if the underlying request itself raises, it swallows the error
    and returns None rather than propagating."""
    from flash.providers.vast import api as vast_api

    def boom(*a, **k):
        raise vast_api.VastApiError("request_logs failed")

    monkeypatch.setattr(vast_api, "request_with_retries", boom)
    assert vast_api.instance_logs(42) is None
