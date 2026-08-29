"""high-level authenticated clients reject redirects without contacting the sink."""

from __future__ import annotations

import logging
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from flash._internal.http import _urlopen_no_redirect
from flash.client.freesolo_api import _freesolo_request, verify_freesolo_key
from flash.client.http import ApiClient, ApiError
from flash.server.billing.charges import BillingError, _post_billing
from flash.server.platform import auth
from flash.server.platform import internal_client as ic


@pytest.fixture
def redirect_servers():
    source_seen: list[tuple[str, str | None]] = []
    sink_seen: list[tuple[str, str | None]] = []

    class SinkHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            sink_seen.append((self.command, self.headers.get("Authorization")))
            self.send_response(200)
            self.end_headers()

        do_POST = do_GET

        def log_message(self, *_args):
            pass

    sink = ThreadingHTTPServer(("127.0.0.1", 0), SinkHandler)
    sink_url = f"http://127.0.0.1:{sink.server_address[1]}"

    class SourceHandler(BaseHTTPRequestHandler):
        def _redirect(self):
            source_seen.append((self.command, self.headers.get("Authorization")))
            self.send_response(302)
            self.send_header("Location", f"{sink_url}/steal")
            self.end_headers()

        do_GET = _redirect
        do_POST = _redirect
        do_DELETE = _redirect

        def log_message(self, *_args):
            pass

    source = ThreadingHTTPServer(("127.0.0.1", 0), SourceHandler)
    source_url = f"http://127.0.0.1:{source.server_address[1]}"
    threads = [
        threading.Thread(target=sink.serve_forever, daemon=True),
        threading.Thread(target=source.serve_forever, daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        yield source_url, source_seen, sink_seen
    finally:
        source.shutdown()
        sink.shutdown()
        source.server_close()
        sink.server_close()


@pytest.mark.parametrize(
    "call",
    [
        lambda client: client.health(),
        lambda client: client._request_bytes("GET", "/bytes"),
        lambda client: list(client.chat_stream("run-a/final", [])),
    ],
    ids=["json", "bytes", "chat-stream"],
)
def test_api_client_paths_keep_api_error_classification_and_do_not_reach_redirect_sink(
    redirect_servers, call
) -> None:
    source_url, source_seen, sink_seen = redirect_servers
    client = ApiClient(source_url, "client-secret")

    with pytest.raises(ApiError) as exc_info:
        call(client)

    assert exc_info.value.status == 302
    assert source_seen[-1][1] == "Bearer client-secret"
    assert sink_seen == []


@pytest.mark.parametrize(
    "call",
    [
        lambda source_url: verify_freesolo_key("freesolo-secret", source_url),
        lambda source_url: _freesolo_request(
            "POST", "/api/projects", "freesolo-secret", source_url, body={"name": "n"}
        ),
    ],
    ids=["verify", "json"],
)
def test_freesolo_client_paths_keep_api_error_classification_and_do_not_reach_redirect_sink(
    redirect_servers, call
) -> None:
    source_url, source_seen, sink_seen = redirect_servers

    with pytest.raises(ApiError) as exc_info:
        call(source_url)

    assert exc_info.value.status == 302
    assert source_seen[-1][1] == "Bearer freesolo-secret"
    assert sink_seen == []


def test_server_verifier_returns_false_and_does_not_reach_redirect_sink(
    monkeypatch, redirect_servers
) -> None:
    source_url, source_seen, sink_seen = redirect_servers
    monkeypatch.setenv(auth.FREESOLO_BASE_URL_ENV, source_url)
    auth._verify_cache.clear()
    auth._verify_inflight.clear()

    assert auth._freesolo_verify("server-secret") is False
    assert source_seen == [("GET", "Bearer server-secret")]
    assert sink_seen == []


def test_a_rejected_redirect_is_not_cached_as_a_failed_verification(
    monkeypatch, redirect_servers
) -> None:
    """a redirect is not a verdict on the token, so it must not populate the negative cache.

    `_urlopen_no_redirect` raises the 3xx instead of following the hop, which means the backend
    never judged this token. caching that alongside a real 401 would keep a valid key failing for
    the whole negative TTL after the redirect condition is gone.
    """
    source_url, source_seen, sink_seen = redirect_servers
    monkeypatch.setenv(auth.FREESOLO_BASE_URL_ENV, source_url)
    auth._verify_cache.clear()
    auth._verify_inflight.clear()

    assert auth._freesolo_verify("redirected-secret") is False
    assert sink_seen == []
    assert "redirected-secret" not in auth._verify_cache

    class _Ok:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self) -> bytes:
            return b"{}"

    # the redirect condition clears and the backend answers for real. a cached negative from the
    # redirect would shadow this and keep returning False without a request.
    monkeypatch.setattr(auth.urllib.request, "urlopen", lambda req, timeout=None: _Ok())

    assert auth._freesolo_verify("redirected-secret") is True
    assert len(source_seen) == 1


def test_internal_client_returns_false_and_does_not_reach_redirect_sink(
    monkeypatch, redirect_servers
) -> None:
    source_url, source_seen, sink_seen = redirect_servers
    monkeypatch.setenv(auth.FREESOLO_BASE_URL_ENV, source_url)
    monkeypatch.setenv(auth.INTERNAL_KEY_ENV, "internal-secret")
    monkeypatch.delenv(auth.STANDALONE_ENV, raising=False)

    assert (
        ic.post_internal_json(
            "/api/internal",
            {"ok": True},
            subject="report test",
            logger=logging.getLogger("test.authenticated_redirects"),
        )
        is False
    )
    assert source_seen == [("POST", "Bearer internal-secret")]
    assert sink_seen == []


def test_billing_keeps_billing_error_classification_and_does_not_reach_redirect_sink(
    monkeypatch, redirect_servers
) -> None:
    source_url, source_seen, sink_seen = redirect_servers
    monkeypatch.setenv(auth.FREESOLO_BASE_URL_ENV, source_url)

    with pytest.raises(BillingError) as exc_info:
        _post_billing(token="billing-secret", path="/api/billing", body={"runId": "run-a"})

    assert exc_info.value.status_code == 302
    assert source_seen == [("POST", "Bearer billing-secret")]
    assert sink_seen == []


def test_a_replaced_global_transport_is_called_directly(monkeypatch, redirect_servers) -> None:
    """a test that swaps `urllib.request.urlopen` must reach its fake, not the opener stack.

    the transport is resolved per call rather than bound at import. a replaced transport is not
    part of the stdlib opener stack, so routing it through a no-redirect opener would silently
    bypass it and send a real request instead.
    """
    source_url, _source_seen, sink_seen = redirect_servers
    calls: list[str] = []

    class _Ok:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> bool:
            return False

        def read(self) -> bytes:
            return b"{}"

    def _fake(request, timeout=None):
        calls.append(request.full_url)
        return _Ok()

    monkeypatch.setattr(urllib.request, "urlopen", _fake)

    req = urllib.request.Request(f"{source_url}/hop", headers={"Authorization": "Bearer t"})
    with _urlopen_no_redirect(req, timeout=5) as resp:
        assert resp.status == 200

    assert calls == [f"{source_url}/hop"]
    assert sink_seen == []


def test_a_redirect_does_not_mutate_the_process_global_opener(redirect_servers) -> None:
    """rejecting a redirect must not install an opener that changes unrelated urllib callers."""
    source_url, _source_seen, sink_seen = redirect_servers
    before = urllib.request._opener

    req = urllib.request.Request(f"{source_url}/hop", headers={"Authorization": "Bearer t"})
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _urlopen_no_redirect(req, timeout=5)

    assert exc_info.value.code == 302
    assert sink_seen == []
    assert urllib.request._opener is before
