from __future__ import annotations

import io
import json
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest

from flash.server.platform import auth

_IDENTITY = {
    "email": "user@example.com",
    "key_prefix": "fslo_test",
    "org_id": "org-1",
    "org_slug": "acme",
}


class _Response:
    status = 200

    def __init__(self, identity: dict[str, str] | None = None):
        self._body = json.dumps(identity or _IDENTITY).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


@pytest.fixture(autouse=True)
def _clear_verify_state():
    with auth._verify_cache_lock:
        auth._verify_cache.clear()
        auth._verify_inflight.clear()
    yield
    with auth._verify_cache_lock:
        auth._verify_cache.clear()
        auth._verify_inflight.clear()


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.freesolo.co/api/auth/verify",
        code,
        "error",
        {},
        io.BytesIO(b"{}"),
    )


def test_same_token_verification_is_single_flight_and_shares_identity(monkeypatch) -> None:
    worker_count = 16
    start = threading.Barrier(worker_count)
    calls_lock = threading.Lock()
    call_count = 0

    def fake_urlopen(req, timeout=None):
        nonlocal call_count
        with calls_lock:
            call_count += 1
        with auth._verify_cache_lock:
            assert "shared-token" not in auth._verify_inflight
            assert auth._verify_key_digest("shared-token") in auth._verify_inflight
            state_bytes = repr((auth._verify_cache, auth._verify_inflight)).encode()
            assert b"shared-token" not in state_bytes
        time.sleep(0.1)
        return _Response()

    monkeypatch.setattr(auth.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(auth.db, "lookup_key", lambda token: {"id": 1, "key_prefix": "stored"})

    def verify() -> dict[str, object] | None:
        start.wait()
        return auth.authenticate("Bearer shared-token")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(lambda _: verify(), range(worker_count)))

    assert all(result is not None and result["org_slug"] == "acme" for result in results)
    assert call_count == 1


def test_http_error_raised_on_response_exit_is_not_cached(monkeypatch) -> None:
    class _ExitFailureResponse(_Response):
        def __exit__(self, *args: object) -> bool:
            raise _http_error(401)

    responses = [_ExitFailureResponse(), _Response()]
    monkeypatch.setattr(
        auth.urllib.request,
        "urlopen",
        lambda req, timeout=None: responses.pop(0),
    )

    token = "revoked-on-exit"
    digest = auth._verify_key_digest(token)
    assert auth._freesolo_verify(token) is None
    assert digest not in auth._verify_cache
    assert auth._freesolo_verify(token) == _IDENTITY
    assert responses == []


def test_first_request_after_completed_flight_revalidates(monkeypatch) -> None:
    call_count = 0

    def fake_urlopen(req, timeout=None):
        nonlocal call_count
        call_count += 1
        return _Response()

    monkeypatch.setattr(auth.urllib.request, "urlopen", fake_urlopen)

    assert auth._freesolo_verify("shared-token") == _IDENTITY
    assert auth._freesolo_verify("shared-token") == _IDENTITY
    assert call_count == 2
    assert auth._verify_key_digest("shared-token") not in auth._verify_cache


def test_allow_then_upstream_401_is_rejected_on_next_request(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(auth.db, "DB_PATH", str(tmp_path / "server.db"))
    responses: list[object] = [_Response(), _http_error(401)]

    def fake_urlopen(req, timeout=None):
        response = responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(auth.urllib.request, "urlopen", fake_urlopen)

    first = auth.authenticate("Bearer fslo-live")
    assert first is not None
    assert first["org_slug"] == "acme"
    assert auth.authenticate("Bearer fslo-live") is None
    assert responses == []


def test_distinct_tokens_verify_independently(monkeypatch) -> None:
    calls_lock = threading.Lock()
    both_entered = threading.Event()
    calls: dict[str, int] = {}

    def fake_urlopen(req, timeout=None):
        token = req.get_header("Authorization").removeprefix("Bearer ")
        with calls_lock:
            calls[token] = calls.get(token, 0) + 1
            if len(calls) == 2:
                both_entered.set()
        assert both_entered.wait(timeout=1.0)
        return _Response({**_IDENTITY, "org_slug": token})

    monkeypatch.setattr(auth.urllib.request, "urlopen", fake_urlopen)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(auth._freesolo_verify, ("token-a", "token-b")))

    assert [result["org_slug"] for result in results if result is not None] == [
        "token-a",
        "token-b",
    ]
    assert calls == {"token-a": 1, "token-b": 1}


def test_wall_clock_jumps_do_not_expire_negative_cache(monkeypatch) -> None:
    calls = 0
    monotonic_now = 100.0

    def fake_urlopen(req, timeout=None):
        nonlocal calls
        calls += 1
        raise _http_error(401)

    monkeypatch.setattr(auth.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(auth.time, "monotonic", lambda: monotonic_now)

    assert auth._freesolo_verify("revoked") is None
    monkeypatch.setattr(auth.time, "time", lambda: 10**12)
    assert auth._freesolo_verify("revoked") is None
    monkeypatch.setattr(auth.time, "time", lambda: -(10**12))
    assert auth._freesolo_verify("revoked") is None
    assert calls == 1


def test_negative_cache_retains_only_a_token_digest(monkeypatch) -> None:
    token = "fslo_secret_bearer_material"
    digest = auth._verify_key_digest(token)
    monkeypatch.setattr(
        auth.urllib.request,
        "urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(_http_error(401)),
    )

    assert auth._freesolo_verify(token) is None
    with auth._verify_cache_lock:
        assert set(auth._verify_cache) == {digest}
        state_bytes = repr((auth._verify_cache, auth._verify_inflight)).encode()
        assert token.encode() not in state_bytes


def test_negative_cache_expires_on_monotonic_time(monkeypatch) -> None:
    calls = 0
    monotonic_now = 100.0

    def fake_urlopen(req, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _http_error(403)
        return _Response()

    monkeypatch.setattr(auth.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(auth.time, "monotonic", lambda: monotonic_now)

    assert auth._freesolo_verify("rotated") is None
    monotonic_now += auth._VERIFY_CACHE_NEG_TTL_S - 0.1
    assert auth._freesolo_verify("rotated") is None
    assert calls == 1
    monotonic_now += 0.2
    assert auth._freesolo_verify("rotated") == _IDENTITY
    assert calls == 2


@pytest.mark.parametrize(
    "failure",
    [
        urllib.error.URLError("connection timed out"),
        _http_error(429),
        _http_error(503),
    ],
    ids=["transport", "rate-limit", "server-error"],
)
def test_transient_failures_are_not_cached(monkeypatch, failure: BaseException) -> None:
    calls = 0

    def fake_urlopen(req, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise failure
        return _Response()

    monkeypatch.setattr(auth.urllib.request, "urlopen", fake_urlopen)

    assert auth._freesolo_verify("recovering") is None
    assert auth._verify_key_digest("recovering") not in auth._verify_cache
    assert auth._freesolo_verify("recovering") == _IDENTITY
    assert calls == 2


def test_negative_cache_is_bounded_and_prunes_expired_entries(monkeypatch) -> None:
    monotonic_now = 100.0

    monkeypatch.setattr(auth.time, "monotonic", lambda: monotonic_now)
    monkeypatch.setattr(auth, "_VERIFY_CACHE_MAX", 8)
    monkeypatch.setattr(
        auth.urllib.request,
        "urlopen",
        lambda req, timeout=None: (_ for _ in ()).throw(_http_error(401)),
    )

    stale_digest = auth._verify_key_digest("stale")
    auth._verify_cache[stale_digest] = monotonic_now - 1.0
    for index in range(50):
        assert auth._freesolo_verify(f"revoked-{index}") is None
        assert len(auth._verify_cache) <= auth._VERIFY_CACHE_MAX
    assert stale_digest not in auth._verify_cache


def test_local_disabled_key_stays_rejected_after_upstream_allow(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(auth.db, "DB_PATH", str(tmp_path / "server.db"))
    monkeypatch.setattr(auth.urllib.request, "urlopen", lambda req, timeout=None: _Response())

    first = auth.authenticate("Bearer fslo-disabled")
    assert first is not None
    with auth.db._connect() as conn:
        conn.execute(
            "UPDATE api_keys SET disabled = 1 WHERE key_hash = ?",
            (auth.db.hash_key("fslo-disabled"),),
        )

    assert auth.authenticate("Bearer fslo-disabled") is None
