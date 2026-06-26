"""RunPod multi-account key pool + quota waterfall.

CPU-only; the REST layer drives a real urllib HTTPError through request_with_retries so
the failover predicate's __cause__ inspection is genuinely exercised. The SDK deploy path
is exercised against fake runpod_flash mocks.
"""

from __future__ import annotations

import io
import urllib.error

import pytest

# Captured at module-load time (before conftest._offline's autouse fixture fires).
# conftest stubs `runpod_api.list_endpoints` to `lambda: []` for every test; tests that
# exercise the real aggregation implementation restore it via this reference.
from flash.providers.runpod.api import list_endpoints as _real_list_endpoints


def _http_error(code: int, body: bytes = b"{}"):
    return urllib.error.HTTPError(
        url="https://rest.runpod.io/v1/endpoints",
        code=code,
        msg="err",
        hdrs=None,
        fp=io.BytesIO(body),
    )


# ---------------------------------------------------------------------------
# keys.py — parsing, ordering, failover advance, env collapse
# ---------------------------------------------------------------------------
def test_single_key_pool(monkeypatch):
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-solo")
    keys.reset()
    assert keys.keys() == ["rk-solo"]
    assert keys.active_key() == "rk-solo"
    assert keys.ordered_keys() == ["rk-solo"]
    assert keys.advance_key() is False  # nothing to fail over to
    assert keys.key_count() == 1


def test_comma_separated_pool_parsed_and_trimmed(monkeypatch):
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", " rk-a , rk-b ,, rk-c ")
    keys.reset()
    assert keys.keys() == ["rk-a", "rk-b", "rk-c"]
    assert keys.key_count() == 3


def test_advance_collapses_env_and_reorders(monkeypatch):
    import os

    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-a,rk-b")
    keys.reset()
    assert keys.active_key() == "rk-a"
    assert keys.ordered_keys() == ["rk-a", "rk-b"]

    assert keys.advance_key() is True
    assert keys.active_key() == "rk-b"
    # ordered_keys is active-first, then the rest (so a non-preferred account is still tried).
    assert keys.ordered_keys() == ["rk-b", "rk-a"]
    # advance collapses RUNPOD_API_KEY to the active key for the runpod_flash SDK.
    assert os.environ["RUNPOD_API_KEY"] == "rk-b"

    # Pool cycles: the pointer wraps back to key #0 so recovered quota there is reused. advance_key
    # ALWAYS advances (and returns True) for a multi-key pool — the return value is NOT an
    # exhaustion signal (that depends on where a failover started, which advance_key can't know), so
    # callers bound failovers by key_count() instead. See deploy_train_endpoint.
    assert keys.advance_key() is True
    assert keys.active_key() == "rk-a"
    assert os.environ["RUNPOD_API_KEY"] == "rk-a"


def test_advance_key_count_bound_visits_every_account_from_any_start(monkeypatch):
    """advance_key() always advances (wrapping), so a COUNT bound of key_count()-1 visits every
    OTHER account exactly once — from ANY starting account, including mid-pool (a prior run may have
    left the pointer advanced). This is the contract deploy_train_endpoint relies on; a wrap-to-0
    'exhaustion' heuristic would wrongly stop a mid-pool failover before the wrapped-over keys."""
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-a,rk-b,rk-c")
    keys.reset()

    # Simulate a prior run that already failed over once: the active key starts at rk-b (mid-pool).
    assert keys.advance_key() is True
    assert keys.active_key() == "rk-b"

    # A fresh failover bounded by key_count()-1 must now visit BOTH remaining accounts (rk-c, then
    # wrap to rk-a) exactly once — never stopping early at the index-0 wrap.
    failovers_left = keys.key_count() - 1  # == 2
    visited = [keys.active_key()]
    while failovers_left > 0:
        assert keys.advance_key() is True  # multi-key pool: always advances
        visited.append(keys.active_key())
        failovers_left -= 1
    assert visited == ["rk-b", "rk-c", "rk-a"]  # every account tried exactly once


def test_select_active_collapses_env(monkeypatch):
    import os

    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-a,rk-b")
    keys.reset()
    assert keys.select_active() == "rk-a"
    assert os.environ["RUNPOD_API_KEY"] == "rk-a"  # collapsed; pool still cached
    assert keys.keys() == ["rk-a", "rk-b"]


def test_no_key_pool(monkeypatch):
    import flash.providers.runpod.keys as keys

    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    keys.reset()
    assert keys.keys() == []
    assert keys.active_key() is None
    assert keys.ordered_keys() == []
    assert keys.select_active() is None
    assert keys.advance_key() is False


def test_is_failover_error(monkeypatch):
    import flash.providers.runpod.keys as keys

    def with_cause(code):
        try:
            raise _http_error(code)
        except urllib.error.HTTPError as cause:
            exc = RuntimeError(f"HTTP {code}")
            exc.__cause__ = cause
            return exc

    # auth / payment / forbidden / not-found / quota-rate => try the next account
    for code in (401, 402, 403, 404, 429):
        assert keys.is_failover_error(with_cause(code)) is True
    # a hard, key-agnostic 4xx and a 5xx server error are the same on every account => no failover
    for code in (400, 409, 422, 500, 503):
        assert keys.is_failover_error(with_cause(code)) is False

    # Non-HTTP failures are account-agnostic (the per-key retry loop already absorbed transient
    # blips), so they do NOT fail over — neither a chained network error nor a bare exception.
    net = RuntimeError("connection reset")
    net.__cause__ = urllib.error.URLError("connection reset")
    assert keys.is_failover_error(net) is False
    assert keys.is_failover_error(RuntimeError("Max workers across all endpoints ... quota")) is False


# ---------------------------------------------------------------------------
# REST client waterfall (flash.providers.runpod.api)
# ---------------------------------------------------------------------------
def _fake_urlopen_by_key(monkeypatch, behavior):
    """Route urlopen by the bearer key. ``behavior(key) -> bytes | Exception``."""
    from flash.providers import _http

    seen = []

    def fake_urlopen(req, timeout=None):
        key = req.get_header("Authorization").split(" ", 1)[1]
        seen.append(key)
        out = behavior(key)
        if isinstance(out, Exception):
            raise out
        return io.BytesIO(out) if not hasattr(out, "read") else out

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(_http.time, "sleep", lambda s: None)
    return seen


def test_rest_waterfall_fails_over_on_401(monkeypatch):
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-bad,rk-good")
    keys.reset()
    # First account is rejected (401); second serves the request.
    seen = _fake_urlopen_by_key(
        monkeypatch,
        lambda k: _http_error(401) if k == "rk-bad" else b'[{"id": "ep1"}]',
    )
    out = runpod_api.request_with_retries(f"{runpod_api.REST_BASE}/endpoints")
    assert out == [{"id": "ep1"}]
    assert seen == ["rk-bad", "rk-good"]  # tried bad first, then failed over


def test_rest_waterfall_single_key_does_not_swallow_4xx(monkeypatch):
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-only")
    keys.reset()
    _fake_urlopen_by_key(monkeypatch, lambda k: _http_error(403, b'{"error":"forbidden"}'))
    # One key, no failover: the 403 surfaces (delete_endpoint relies on this to report failure).
    with pytest.raises(runpod_api.RunpodApiError, match="403"):
        runpod_api.request_with_retries(f"{runpod_api.REST_BASE}/endpoints")


def test_rest_waterfall_hard_4xx_not_retried_across_keys(monkeypatch):
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-a,rk-b")
    keys.reset()
    # A 400 is the same on every account -> raise immediately, do NOT try the second key.
    seen = _fake_urlopen_by_key(monkeypatch, lambda k: _http_error(400, b'{"error":"bad"}'))
    with pytest.raises(runpod_api.RunpodApiError, match="400"):
        runpod_api.request_with_retries(f"{runpod_api.REST_BASE}/endpoints")
    assert seen == ["rk-a"]  # never tried rk-b


def test_rest_waterfall_all_keys_exhausted_raises(monkeypatch):
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-a,rk-b")
    keys.reset()
    seen = _fake_urlopen_by_key(monkeypatch, lambda k: _http_error(401))
    with pytest.raises(runpod_api.RunpodApiError):
        runpod_api.request_with_retries(f"{runpod_api.REST_BASE}/endpoints")
    assert seen == ["rk-a", "rk-b"]  # both accounts attempted


def test_rest_waterfall_persistent_429_fails_over(monkeypatch):
    """A 429 that survives the retry loop (no fast-fail) still fails over to the next key.

    Regression: the exhausted-retries raise must chain the HTTPError as __cause__ so the
    failover predicate can see the 429 status — otherwise it stops on the first account.
    """
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-a,rk-b")
    keys.reset()
    seen = _fake_urlopen_by_key(
        monkeypatch,
        lambda k: _http_error(429) if k == "rk-a" else b'{"ok": true}',
    )
    out = runpod_api.request_with_retries(f"{runpod_api.REST_BASE}/endpoints", retries=1)
    assert out == {"ok": True}
    assert "rk-b" in seen  # failed over after rk-a's 429 exhausted its retries


def test_rest_waterfall_persistent_5xx_does_not_fail_over(monkeypatch):
    """A 5xx is a server-side error (same on every account) -> do NOT try the next key."""
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-a,rk-b")
    keys.reset()
    seen = _fake_urlopen_by_key(monkeypatch, lambda k: _http_error(500, b"boom"))
    with pytest.raises(runpod_api.RunpodApiError):
        runpod_api.request_with_retries(f"{runpod_api.REST_BASE}/endpoints", retries=1)
    assert "rk-b" not in seen  # never failed over on a server error


# NOTE: the deploy_train_endpoint account-failover test lives in test_jobs.py (next to the
# _patch_deploy_deps / _make_runpod_flash_mocks helpers it shares with the other deploy tests),
# and the idle-endpoint reaper / run-aware protection tests live in test_idle_endpoint_reaper.py
# and test_jobs.py (the proactive reclaim mechanism from #190).


# ---------------------------------------------------------------------------
# list_endpoints — multi-key aggregation and raise-on-failure contract
# ---------------------------------------------------------------------------
def test_list_endpoints_aggregates_across_all_keys(monkeypatch):
    """Endpoints from every account are merged into a single list."""
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-a,rk-b")
    keys.reset()

    responses = {
        "rk-a": [{"id": "ep-a1"}, {"id": "ep-a2"}],
        "rk-b": [{"id": "ep-b1"}],
    }

    def fake_for_key(key, target, **_kw):
        return responses[key]

    monkeypatch.setattr(runpod_api._CLIENT, "request_with_retries_for_key", fake_for_key)
    # conftest._offline stubs list_endpoints → restore the real aggregating implementation.
    monkeypatch.setattr(runpod_api, "list_endpoints", _real_list_endpoints)

    result = runpod_api.list_endpoints()
    ids = {e["id"] for e in result}
    assert ids == {"ep-a1", "ep-a2", "ep-b1"}


def test_list_endpoints_raises_when_any_key_fails(monkeypatch):
    """A per-key failure propagates so callers don't act on a partial view."""
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-a,rk-b")
    keys.reset()

    def fake_for_key(key, target, **_kw):
        if key == "rk-a":
            return [{"id": "ep-a1"}]
        raise runpod_api.RunpodApiError("GET .../endpoints -> HTTP 500: Internal Server Error")

    monkeypatch.setattr(runpod_api._CLIENT, "request_with_retries_for_key", fake_for_key)
    monkeypatch.setattr(runpod_api, "list_endpoints", _real_list_endpoints)

    with pytest.raises(runpod_api.RunpodApiError):
        runpod_api.list_endpoints()


def test_list_endpoints_raises_when_no_key_configured(monkeypatch):
    """No RUNPOD_API_KEY -> raise instead of returning [] without any authenticated call. Callers
    (slot-reconcile/teardown) treat [] as 'fleet confirmed empty' and would act on a false view."""
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.keys as keys

    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    keys.reset()
    monkeypatch.setattr(
        runpod_api._CLIENT,
        "request_with_retries_for_key",
        lambda *a, **k: pytest.fail("must not hit RunPod with no key configured"),
    )
    monkeypatch.setattr(runpod_api, "list_endpoints", _real_list_endpoints)

    with pytest.raises(runpod_api.RunpodApiError, match="not set"):
        runpod_api.list_endpoints()


def test_list_endpoints_raises_on_non_list_response(monkeypatch):
    """A 200 whose body isn't a list is NOT an empty account — silently skipping it would yield a
    partial fleet view callers trust as complete, so raise."""
    import flash.providers.runpod.api as runpod_api
    import flash.providers.runpod.keys as keys

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-a")
    keys.reset()
    monkeypatch.setattr(
        runpod_api._CLIENT, "request_with_retries_for_key", lambda *a, **k: {"unexpected": "shape"}
    )
    monkeypatch.setattr(runpod_api, "list_endpoints", _real_list_endpoints)

    with pytest.raises(runpod_api.RunpodApiError, match="unexpected /endpoints response"):
        runpod_api.list_endpoints()
