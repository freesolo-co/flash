"""The managed teacher shares a bounded standard HTTP pool across scoring batches."""

import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from flash.engine.worker.teacher.client import _TeacherHttpTransport


def test_pool_survives_multiple_scoring_executors_and_closes(monkeypatch):
    clients = []
    seen = []
    barrier = threading.Barrier(4)
    real_client = httpx.Client

    def handle(request):
        barrier.wait(timeout=5)
        seen.append(request)
        return httpx.Response(200, content=b'{"ok":true}')

    def make_client(**kwargs):
        assert kwargs["limits"].max_connections == kwargs["limits"].max_keepalive_connections
        client = real_client(transport=httpx.MockTransport(handle), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "Client", make_client)
    transport = _TeacherHttpTransport()

    def score(_):
        req = urllib.request.Request("https://teacher.test/score", data=b"{}")
        with transport.urlopen(req, timeout=2) as response:
            return response.read()

    try:
        for _ in range(2):
            with ThreadPoolExecutor(max_workers=4) as pool:
                assert list(pool.map(score, range(4))) == [b'{"ok":true}'] * 4
        assert len(clients) == 1
        assert len(seen) == 8
        assert not clients[0].is_closed
    finally:
        transport.close()
    assert clients[0].is_closed


@pytest.mark.parametrize("status", [302, 429, 503])
def test_status_and_body_are_preserved_without_redirects_or_automatic_retries(monkeypatch, status):
    calls = []
    real_client = httpx.Client

    def handle(request):
        calls.append(request)
        return httpx.Response(
            status, headers={"Location": "https://elsewhere.test/"}, content=b"error"
        )

    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs),
    )
    transport = _TeacherHttpTransport()
    req = urllib.request.Request(
        "https://teacher.test/score", data=b"{}", headers={"X-Flash-Teacher-Request-Id": "same-id"}
    )
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            transport.urlopen(req, timeout=1)
        assert exc.value.code == status
        assert exc.value.read() == b"error"
        assert len(calls) == 1
        assert calls[0].headers["x-flash-teacher-request-id"] == "same-id"
    finally:
        transport.close()


def test_transport_failure_keeps_existing_retry_classifier_contract(monkeypatch):
    real_client = httpx.Client

    def fail(request):
        assert request.extensions["timeout"]["read"] == 0.5
        raise httpx.ReadTimeout("timeout", request=request)

    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(fail), **kwargs)
    )
    transport = _TeacherHttpTransport()
    try:
        with pytest.raises(urllib.error.URLError):
            transport.urlopen(urllib.request.Request("https://teacher.test/score"), timeout=0.5)
    finally:
        transport.close()
