"""high-level authenticated clients reject redirects without contacting the sink."""

from __future__ import annotations

import logging
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from flash._internal.http import (
    _move_credentials_to_the_unredirected_bag,
    _urlopen_no_redirect,
)
from flash.client.freesolo_api import _freesolo_request, verify_freesolo_key
from flash.client.http import ApiClient, ApiError
from flash.server.billing.charges import BillingError, _post_billing
from flash.server.platform import auth
from flash.server.platform import internal_client as ic
from tests._helpers.wire_headers import sent_headers


@contextmanager
def _ok_server():
    """serve 200 with an empty JSON body, for asserting a later call is really made."""

    class OkHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), OkHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


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
    assert source_seen == [("GET", "Bearer redirected-secret")]
    assert sink_seen == []
    assert "redirected-secret" not in auth._verify_cache

    # the redirect condition clears and the backend answers for real. a cached negative from the
    # redirect would shadow this and keep returning False without a request.
    with _ok_server() as ok_url:
        monkeypatch.setenv(auth.FREESOLO_BASE_URL_ENV, ok_url)
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


def test_a_redirect_following_injected_transport_still_loses_the_credential(
    redirect_servers,
) -> None:
    """an injected transport that delegates to the stdlib must not become a credential leak.

    the seam honours a caller-supplied transport as given and cannot know whether it follows
    redirects: an identity check could not tell a test fake from an APM or compatibility wrapper
    that calls the real stdlib underneath. so the protection is structural instead. the hop is
    followed here, but the credential lives in the header bag `redirect_request` does not copy,
    and the sink sees none of it.
    """
    source_url, source_seen, sink_seen = redirect_servers
    stdlib_urlopen = urllib.request.urlopen

    def _delegating_wrapper(request, **kwargs):
        return stdlib_urlopen(request, **kwargs)

    req = urllib.request.Request(f"{source_url}/hop", headers={"Authorization": "Bearer wrapped"})
    with _urlopen_no_redirect(req, timeout=5, urlopen=_delegating_wrapper) as resp:
        resp.read()

    assert source_seen == [("GET", "Bearer wrapped")]
    assert sink_seen == [("GET", None)]


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


def test_the_callers_credential_survives_two_redirect_hops_through_a_delegating_transport(
    monkeypatch,
) -> None:
    """flash's own credential must not reach a redirect target, however many hops it takes.

    two hops rather than one, because the stdlib builds each hop as a fresh `Request`. anything
    scoped to the object we were handed is gone by the second hop, so a one-hop test cannot tell a
    real containment guarantee from one that only holds for the request we happened to touch.
    """
    sink_seen: list[str | None] = []

    class SinkHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            sink_seen.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_args):
            pass

    sink = ThreadingHTTPServer(("127.0.0.1", 0), SinkHandler)
    sink_url = f"http://127.0.0.1:{sink.server_address[1]}"

    hops: list[str | None] = []

    class ProxyHandlerServer(BaseHTTPRequestHandler):
        def do_GET(self):
            hops.append(self.headers.get("Authorization"))
            # hop one redirects to a second proxied host; hop two out to the exempt sink.
            location = "http://second.invalid/y" if len(hops) == 1 else f"{sink_url}/steal"
            self.send_response(302)
            self.send_header("Location", location)
            self.end_headers()

        def log_message(self, *_args):
            pass

    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ProxyHandlerServer)
    proxy_port = proxy.server_address[1]
    for server in (sink, proxy):
        threading.Thread(target=server.serve_forever, daemon=True).start()

    monkeypatch.setenv("http_proxy", f"http://proxyuser:proxypw@127.0.0.1:{proxy_port}/")
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    # a wrapper that follows redirects and honours the proxy env, standing in for an APM or
    # compatibility shim. `urlopen`'s cached global opener would not pick the env up here.
    def _delegating_wrapper(request, **kwargs):
        return urllib.request.build_opener().open(request, **kwargs)

    try:
        req = urllib.request.Request(
            "http://proxied.invalid/x", headers={"Authorization": "Bearer flash-secret"}
        )
        with _urlopen_no_redirect(req, timeout=5, urlopen=_delegating_wrapper) as resp:
            resp.read()
    finally:
        for server in (proxy, sink):
            server.shutdown()
            server.server_close()

    assert hops == ["Bearer flash-secret", None]
    assert sink_seen == [None]


def test_relocation_keeps_the_credential_the_wire_was_already_carrying() -> None:
    """a duplicate across both bags must not change which principal is authenticated.

    `do_open` gives `unredirected_hdrs` precedence, so an entry already there is the value being
    sent. promoting the redirectable duplicate over it would silently swap the credential.
    """
    request = urllib.request.Request("http://example.invalid/x")
    request.add_unredirected_header("Authorization", "Bearer canonical")
    request.add_header("Authorization", "Bearer other")

    _move_credentials_to_the_unredirected_bag(request)

    assert sent_headers(request)["Authorization"] == "Bearer canonical"
    assert "Authorization" not in request.headers


def test_sent_headers_collapses_case_colliding_entries_the_way_do_open_does() -> None:
    """`do_open` title-cases after merging, so a case collision resolves to the `headers` value.

    the exact-key merge gives `unredirected_hdrs` precedence, but the `.title()` pass runs after it
    and collapses the two spellings onto one key. a helper that stopped before that step would
    report both entries surviving and would score a mixed-casing regression as safe.
    """
    request = urllib.request.Request("http://example.invalid/x")
    request.add_unredirected_header("Authorization", "Bearer canonical")
    # not `add_header`: it capitalizes, and the collision only exists for unequal exact keys.
    request.headers["authorization"] = "Bearer redirectable"

    assert sent_headers(request) == {"Authorization": "Bearer redirectable"}
