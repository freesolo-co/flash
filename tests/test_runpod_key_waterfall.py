"""RunPod multi-account key pool + quota waterfall.

CPU-only; the REST layer drives a real urllib HTTPError through request_with_retries so
the failover predicate's __cause__ inspection is genuinely exercised. The SDK deploy path
is exercised against fake runpod_flash mocks.
"""

from __future__ import annotations

import io
import urllib.error

import pytest


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
    import os

    assert os.environ["RUNPOD_API_KEY"] == "rk-b"

    assert keys.advance_key() is False  # pool exhausted


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

    # auth / payment / forbidden / not-found / quota-rate => try next account
    for code in (401, 402, 403, 404, 429):
        assert keys.is_failover_error(with_cause(code)) is True
    # a hard, key-agnostic 4xx is NOT a failover trigger
    for code in (400, 409, 422):
        assert keys.is_failover_error(with_cause(code)) is False
    # quota wording with no HTTP cause still fails over
    assert keys.is_failover_error(RuntimeError("Max workers across all endpoints ... quota")) is True
    # a bare network failure (no cause, no "http") is worth another account
    assert keys.is_failover_error(RuntimeError("connection reset")) is True


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


# NOTE: the deploy_train_endpoint account-failover test lives in test_jobs.py, next to the
# _patch_deploy_deps / _make_runpod_flash_mocks helpers it shares with the other deploy tests.


# ---------------------------------------------------------------------------
# sweep_orphans (proactive idle reclaim with active-run protection)
# ---------------------------------------------------------------------------
def test_sweep_orphans_no_key_is_noop(monkeypatch):
    from flash.providers import get_provider

    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    assert get_provider("runpod").sweep_orphans(active_labels={"flash-x"}) == []


def test_sweep_orphans_reclaims_unowned_idle_endpoints(monkeypatch):
    import flash.providers.runpod.api as runpod_api
    from flash.providers import get_provider
    from flash.providers.runpod.train import _run_suffix

    monkeypatch.setenv("RUNPOD_API_KEY", "rk")
    import flash.providers.runpod.keys as keys

    keys.reset()

    active_run = "flash-1782-live"
    suffix = _run_suffix(active_run)
    endpoints = [
        {"id": "ep-active", "name": f"live-flash-a100-{suffix}"},   # active run -> protected
        {"id": "ep-orphan", "name": "live-flash-a100-deadbeef99"},  # no owner -> reclaim
    ]
    idle_done = {"workers": {"running": 0, "ready": 1, "idle": 1, "initializing": 0},
                 "jobs": {"inQueue": 0, "inProgress": 0, "completed": 1, "failed": 0}}
    deleted = []
    monkeypatch.setattr(runpod_api, "list_endpoints", lambda: endpoints)
    monkeypatch.setattr(runpod_api, "endpoint_health", lambda eid: idle_done)
    monkeypatch.setattr(runpod_api, "delete_endpoint", lambda eid: deleted.append(eid) or True)

    reaped = get_provider("runpod").sweep_orphans(active_labels={active_run})
    assert reaped == ["ep-orphan"]
    assert deleted == ["ep-orphan"]
