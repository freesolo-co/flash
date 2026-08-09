from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from flash.server.platform import auth


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return b"{}"


@pytest.fixture(autouse=True)
def _clear_verify_state():
    with auth._verify_cache_lock:
        auth._verify_cache.clear()
        auth._verify_inflight.clear()
    yield
    with auth._verify_cache_lock:
        auth._verify_cache.clear()
        auth._verify_inflight.clear()


def test_same_token_verification_is_single_flight(monkeypatch) -> None:
    worker_count = 16
    start = threading.Barrier(worker_count)
    calls_lock = threading.Lock()
    call_count = 0

    def fake_urlopen(req, timeout=None):
        nonlocal call_count
        with calls_lock:
            call_count += 1
        time.sleep(0.1)
        return _Response()

    monkeypatch.setattr(auth.urllib.request, "urlopen", fake_urlopen)

    def verify() -> bool:
        start.wait()
        return auth._freesolo_verify("shared-token")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(lambda _: verify(), range(worker_count)))

    assert results == [True] * worker_count
    assert call_count == 1


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
        return _Response()

    monkeypatch.setattr(auth.urllib.request, "urlopen", fake_urlopen)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(auth._freesolo_verify, ("token-a", "token-b")))

    assert results == [True, True]
    assert calls == {"token-a": 1, "token-b": 1}
