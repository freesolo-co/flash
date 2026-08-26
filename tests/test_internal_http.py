"""redirect containment for authenticated stdlib requests."""

from __future__ import annotations

import builtins
import copyreg
import dis
import email.message
import functools
import gc
import http.cookiejar
import io
import os
import ssl
import subprocess
import sys
import textwrap
import threading
import types
import urllib.error
import urllib.request
import urllib.response
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

import flash._internal.http as http_transport
from flash._internal.http import _build_no_redirect_opener, _urlopen_no_redirect


@pytest.fixture(autouse=True)
def _restore_http_globals(monkeypatch):
    opener = urllib.request._opener
    default = http_transport._DEFAULT_NO_REDIRECT_OPENER
    cached = http_transport._INSTALLED_OPENER_CACHE
    for key in ("HTTPS_PROXY", "https_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(key, raising=False)
    http_transport._DEFAULT_NO_REDIRECT_OPENER = None
    http_transport._INSTALLED_OPENER_CACHE = None
    try:
        yield
    finally:
        urllib.request._opener = opener
        http_transport._DEFAULT_NO_REDIRECT_OPENER = default
        http_transport._INSTALLED_OPENER_CACHE = cached


def _response(url: str, body: bytes = b"ok"):
    response = urllib.response.addinfourl(io.BytesIO(body), email.message.Message(), url, code=200)
    response.msg = "ok"
    return response


def _redirect_opener(*, source: str, target: str, status: int, observed: list[dict[str, object]]):
    class RedirectSource(urllib.request.BaseHandler):
        def default_open(self, request):
            observed.append(
                {
                    "url": request.full_url,
                    "method": request.get_method(),
                    "authorization": request.get_header("Authorization"),
                }
            )
            if request.full_url != source:
                return urllib.response.addinfourl(
                    io.BytesIO(b"ok"), email.message.Message(), request.full_url, code=200
                )
            headers = email.message.Message()
            headers["Location"] = target
            response = urllib.response.addinfourl(
                io.BytesIO(b""), headers, request.full_url, code=status
            )
            response.msg = "redirect"
            return response

    return _build_no_redirect_opener(RedirectSource())


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_no_redirect_opener_rejects_get_redirects_before_the_sink(status: int) -> None:
    source = "https://source.invalid/data"
    target = "https://sink.invalid/steal"
    observed: list[dict[str, object]] = []
    opener = _redirect_opener(source=source, target=target, status=status, observed=observed)
    request = urllib.request.Request(source, headers={"Authorization": "Bearer secret"})

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        opener.open(request, timeout=1.0)

    assert exc_info.value.code == status
    assert observed == [{"url": source, "method": "GET", "authorization": "Bearer secret"}]


def test_no_redirect_opener_rejects_post_redirect_before_method_rewrite() -> None:
    source = "https://source.invalid/write"
    target = "https://sink.invalid/steal"
    observed: list[dict[str, object]] = []
    opener = _redirect_opener(source=source, target=target, status=302, observed=observed)
    request = urllib.request.Request(
        source,
        data=b"payload",
        method="POST",
        headers={"Authorization": "Bearer secret"},
    )

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        opener.open(request, timeout=1.0)

    assert exc_info.value.code == 302
    assert observed == [{"url": source, "method": "POST", "authorization": "Bearer secret"}]


def test_no_redirect_opener_rejects_https_to_http_before_the_sink() -> None:
    source = "https://source.invalid/data"
    target = "http://sink.invalid/steal"
    observed: list[dict[str, object]] = []
    opener = _redirect_opener(source=source, target=target, status=302, observed=observed)
    request = urllib.request.Request(source, headers={"Authorization": "Bearer secret"})

    with pytest.raises(urllib.error.HTTPError):
        opener.open(request, timeout=1.0)

    assert observed == [{"url": source, "method": "GET", "authorization": "Bearer secret"}]


def test_building_and_using_private_opener_does_not_mutate_global_opener(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(urllib.request, "_opener", sentinel)
    observed: list[dict[str, object]] = []
    opener = _redirect_opener(
        source="https://source.invalid/data",
        target="https://sink.invalid/steal",
        status=302,
        observed=observed,
    )

    with pytest.raises(urllib.error.HTTPError):
        opener.open(
            urllib.request.Request(
                "https://source.invalid/data", headers={"Authorization": "Bearer secret"}
            ),
            timeout=1.0,
        )

    assert urllib.request._opener is sentinel


def test_installed_https_handler_copy_preserves_context_and_state() -> None:
    context = ssl.create_default_context()
    observed: list[tuple[object, object, int]] = []

    class RecordingHttpsHandler(urllib.request.HTTPSHandler):
        def https_open(self, request):
            observed.append((self.parent, self._context, self.debuglevel))
            return _response(request.full_url)

    handler = RecordingHttpsHandler(context=context)
    handler.debuglevel = 7
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)
    original_parent = handler.parent
    original_state = handler.__dict__.copy()

    with _urlopen_no_redirect(
        urllib.request.Request("https://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    copied, copied_context, copied_debuglevel = observed.pop()
    assert copied is not handler
    assert copied_context is context
    assert copied_debuglevel == 7
    assert urllib.request._opener is opener
    assert handler.parent is original_parent
    assert handler.__dict__ == original_state


def test_installed_proxy_and_auth_handler_state_is_preserved() -> None:
    proxies = {"custom": "custom://proxy.invalid:8080"}
    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password(None, "https://source.invalid", "user", "pass")
    proxy_calls: list[tuple[object, object]] = []

    class StatefulProxyHandler(urllib.request.ProxyHandler):
        def proxy_open(self, request, proxy, proxy_type):
            proxy_calls.append((self, self.marker))
            return super().proxy_open(request, proxy, proxy_type)

    marker = object()
    proxy = StatefulProxyHandler(proxies)
    proxy.marker = marker
    auth = urllib.request.HTTPBasicAuthHandler(password_manager)
    observed: list[tuple[object, object, object]] = []

    class TerminalHandler(urllib.request.BaseHandler):
        handler_order = 150

        def custom_open(self, request):
            installed = self.parent
            copied_proxy = next(
                item for item in installed.handlers if isinstance(item, urllib.request.ProxyHandler)
            )
            copied_auth = next(
                item
                for item in installed.handlers
                if isinstance(item, urllib.request.HTTPBasicAuthHandler)
            )
            observed.append((copied_proxy, copied_auth, request))
            return _response(request.full_url)

    terminal = TerminalHandler()
    opener = urllib.request.build_opener(proxy, auth, terminal)
    urllib.request.install_opener(opener)
    original_parents = {handler: handler.parent for handler in (proxy, auth, terminal)}
    original_states = {handler: handler.__dict__.copy() for handler in (proxy, auth, terminal)}

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    copied_proxy, copied_auth, request = observed.pop()
    assert copied_proxy is not proxy
    assert copied_proxy.proxies is proxies
    assert proxy_calls == [(copied_proxy, marker)]
    assert copied_auth is not auth
    assert copied_auth.passwd is password_manager
    assert request.full_url == "custom://source.invalid/data"
    assert request.host == "proxy.invalid:8080"
    assert urllib.request._opener is opener
    assert {handler: handler.parent for handler in original_parents} == original_parents
    assert {handler: handler.__dict__ for handler in original_states} == original_states


def test_default_opener_discovers_proxy_environment_on_first_request(monkeypatch) -> None:
    monkeypatch.setattr(urllib.request, "_opener", None)
    monkeypatch.setenv("HTTPS_PROXY", "http://late-proxy.invalid:8080")
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.delenv("no_proxy", raising=False)
    observed: list[tuple[str, str | None]] = []

    def fake_https_open(self, request):
        observed.append((request.host, request._tunnel_host))
        return _response(request.full_url)

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", fake_https_open)

    with _urlopen_no_redirect(
        urllib.request.Request("https://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    assert observed == [("late-proxy.invalid:8080", "source.invalid")]


def test_installed_cookie_processor_with_rlock_state_is_supported() -> None:
    cookie_jar = http.cookiejar.CookieJar()
    observed: list[object] = []

    class TerminalHandler(urllib.request.BaseHandler):
        handler_order = 1000

        def custom_open(self, request):
            copied_cookie = next(
                item
                for item in self.parent.handlers
                if isinstance(item, urllib.request.HTTPCookieProcessor)
            )
            observed.append(copied_cookie.cookiejar)
            return _response(request.full_url)

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(cookie_jar),
        TerminalHandler(),
    )
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    assert observed == [cookie_jar]


def test_installed_addheaders_are_copied_and_isolated() -> None:
    observed: list[str | None] = []

    class TerminalHandler(urllib.request.AbstractHTTPHandler):
        custom_request = urllib.request.AbstractHTTPHandler.do_request_

        def custom_open(self, request):
            observed.append(request.get_header("X-installed"))
            return _response(request.full_url)

    opener = urllib.request.build_opener(TerminalHandler())
    opener.addheaders = [("X-installed", "original")]
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    private = http_transport._INSTALLED_OPENER_CACHE.private
    assert observed == ["original"]
    assert private.addheaders == opener.addheaders
    assert private.addheaders is not opener.addheaders
    private.addheaders.append(("X-private", "value"))
    assert opener.addheaders == [("X-installed", "original")]


def test_installed_addheader_item_iterator_does_not_execute_before_rejection() -> None:
    iterator_calls: list[str] = []
    contacted: list[str] = []

    class MaliciousPair:
        def __iter__(self):
            iterator_calls.append("called")
            raise AssertionError("iterator executed")

    class TerminalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    opener = urllib.request.build_opener(TerminalHandler())
    opener.addheaders = [MaliciousPair()]
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib opener cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert iterator_calls == []
    assert contacted == []


def test_installed_addheaders_mutation_updates_cached_private_opener() -> None:
    observed: list[str | None] = []

    class TerminalHandler(urllib.request.AbstractHTTPHandler):
        custom_request = urllib.request.AbstractHTTPHandler.do_request_

        def custom_open(self, request):
            observed.append(request.get_header("X-installed"))
            return _response(request.full_url)

    opener = urllib.request.build_opener(TerminalHandler())
    opener.addheaders = [("X-installed", "first")]
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/a"), timeout=1.0):
        pass
    private = http_transport._INSTALLED_OPENER_CACHE.private
    opener.addheaders[0] = ("X-installed", "second")
    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/b"), timeout=1.0):
        pass
    opener.addheaders = [("X-installed", "third")]
    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/c"), timeout=1.0):
        pass

    assert observed == ["first", "second", "third"]
    assert http_transport._INSTALLED_OPENER_CACHE.private is private
    assert private.addheaders == [("X-installed", "third")]
    assert private.addheaders is not opener.addheaders


def test_installed_handler_list_mutation_rebuilds_private_opener() -> None:
    observed: list[str] = []

    class FirstHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            observed.append("first")
            return _response(request.full_url)

    class SecondHandler(urllib.request.BaseHandler):
        handler_order = 50

        def custom_open(self, request):
            observed.append("second")
            return _response(request.full_url)

    opener = urllib.request.build_opener(FirstHandler())
    urllib.request.install_opener(opener)
    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/a"), timeout=1.0):
        pass
    first_private = http_transport._INSTALLED_OPENER_CACHE.private
    opener.add_handler(SecondHandler())
    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/b"), timeout=1.0):
        pass

    assert observed == ["first", "second"]
    assert http_transport._INSTALLED_OPENER_CACHE.private is not first_private


def test_installed_https_context_replacement_rebuilds_private_opener() -> None:
    first_context = ssl.create_default_context()
    second_context = ssl.create_default_context()
    observed: list[object] = []

    class RecordingHttpsHandler(urllib.request.HTTPSHandler):
        def https_open(self, request):
            observed.append(self._context)
            return _response(request.full_url)

    handler = RecordingHttpsHandler(context=first_context)
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(urllib.request.Request("https://source.invalid/first"), timeout=1.0):
        pass
    first_private = http_transport._INSTALLED_OPENER_CACHE.private
    handler._context = second_context
    with _urlopen_no_redirect(urllib.request.Request("https://source.invalid/second"), timeout=1.0):
        pass

    assert observed == [first_context, second_context]
    assert http_transport._INSTALLED_OPENER_CACHE.private is not first_private
    assert handler.parent is opener
    assert handler._context is second_context


def test_installed_scalar_configuration_change_rebuilds_private_opener() -> None:
    observed: list[str] = []

    class GatewayHandler(urllib.request.BaseHandler):
        def __init__(self, token: str):
            self.token = token

        def custom_open(self, request):
            observed.append(self.token)
            return _response(request.full_url)

    handler = GatewayHandler("token-a")
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/a"), timeout=1.0):
        pass
    first_private = http_transport._INSTALLED_OPENER_CACHE.private
    handler.token = "token-b"
    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/b"), timeout=1.0):
        pass

    assert observed == ["token-a", "token-b"]
    assert http_transport._INSTALLED_OPENER_CACHE.private is not first_private
    assert handler.parent is opener
    assert handler.token == "token-b"


def test_installed_container_configuration_mutation_rebuilds_private_opener() -> None:
    observed: list[tuple[str, ...]] = []

    class GatewayHandler(urllib.request.BaseHandler):
        def __init__(self):
            self.config = {"tokens": ["token-a"]}

        def custom_open(self, request):
            observed.append(tuple(self.config["tokens"]))
            return _response(request.full_url)

    handler = GatewayHandler()
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/a"), timeout=1.0):
        pass
    first_private = http_transport._INSTALLED_OPENER_CACHE.private
    handler.config["tokens"].append("token-b")
    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/b"), timeout=1.0):
        pass

    assert observed == [("token-a",), ("token-a", "token-b")]
    assert http_transport._INSTALLED_OPENER_CACHE.private is not first_private
    assert handler.parent is opener
    assert handler.config == {"tokens": ["token-a", "token-b"]}


def test_concurrent_installed_configuration_refresh_is_singleton() -> None:
    barrier = threading.Barrier(8)
    observed: list[tuple[str, object]] = []
    observed_lock = threading.Lock()

    class GatewayHandler(urllib.request.BaseHandler):
        def __init__(self, token: str):
            self.token = token

        def custom_open(self, request):
            with observed_lock:
                observed.append((self.token, self.parent))
            return _response(request.full_url)

    handler = GatewayHandler("token-a")
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)
    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/first"), timeout=1.0):
        pass
    observed.clear()
    first_private = http_transport._INSTALLED_OPENER_CACHE.private
    handler.token = "token-b"

    def request(index: int) -> bytes:
        barrier.wait()
        with _urlopen_no_redirect(
            urllib.request.Request(f"custom://source.invalid/{index}"), timeout=1.0
        ) as response:
            return response.read()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(request, range(8)))

    parents = {id(parent) for _token, parent in observed}
    assert results == [b"ok"] * 8
    assert [token for token, _parent in observed] == ["token-b"] * 8
    assert len(parents) == 1
    assert observed[0][1] is http_transport._INSTALLED_OPENER_CACHE.private
    assert observed[0][1] is not first_private


def test_installed_slot_configuration_change_rebuilds_private_opener() -> None:
    observed: list[str] = []

    class SlotGatewayHandler(urllib.request.BaseHandler):
        __slots__ = ("token",)

        def custom_open(self, request):
            observed.append(self.token)
            return _response(request.full_url)

    handler = SlotGatewayHandler()
    handler.token = "token-a"
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/a"), timeout=1.0):
        pass
    first_private = http_transport._INSTALLED_OPENER_CACHE.private
    handler.token = "token-b"
    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/b"), timeout=1.0):
        pass

    assert observed == ["token-a", "token-b"]
    assert http_transport._INSTALLED_OPENER_CACHE.private is not first_private
    assert handler.parent is opener
    assert handler.token == "token-b"


def test_installed_slot_container_mutation_rebuilds_private_opener() -> None:
    observed: list[tuple[str, ...]] = []

    class SlotGatewayHandler(urllib.request.BaseHandler):
        __slots__ = ("config",)

        def custom_open(self, request):
            observed.append(tuple(self.config["tokens"]))
            return _response(request.full_url)

    handler = SlotGatewayHandler()
    handler.config = {"tokens": ["token-a"]}
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/a"), timeout=1.0):
        pass
    first_private = http_transport._INSTALLED_OPENER_CACHE.private
    handler.config["tokens"].append("token-b")
    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/b"), timeout=1.0):
        pass

    assert observed == [("token-a",), ("token-a", "token-b")]
    assert http_transport._INSTALLED_OPENER_CACHE.private is not first_private
    assert handler.parent is opener
    assert handler.config == {"tokens": ["token-a", "token-b"]}


def test_inherited_and_mangled_slot_changes_rebuild_private_opener() -> None:
    observed: list[tuple[str, str]] = []

    class BaseGatewayHandler(urllib.request.BaseHandler):
        __slots__ = ("__base_token",)

    class SlotGatewayHandler(BaseGatewayHandler):
        __slots__ = ("token",)

        def custom_open(self, request):
            observed.append((self._BaseGatewayHandler__base_token, self.token))
            return _response(request.full_url)

    handler = SlotGatewayHandler()
    handler._BaseGatewayHandler__base_token = "base-a"
    handler.token = "token-a"
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/a"), timeout=1.0):
        pass
    first_private = http_transport._INSTALLED_OPENER_CACHE.private
    handler._BaseGatewayHandler__base_token = "base-b"
    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/b"), timeout=1.0):
        pass
    second_private = http_transport._INSTALLED_OPENER_CACHE.private
    handler.token = "token-b"
    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/c"), timeout=1.0):
        pass

    assert observed == [
        ("base-a", "token-a"),
        ("base-b", "token-a"),
        ("base-b", "token-b"),
    ]
    assert second_private is not first_private
    assert http_transport._INSTALLED_OPENER_CACHE.private is not second_private


def test_unchanged_installed_slot_configuration_reuses_private_opener() -> None:
    class SlotGatewayHandler(urllib.request.BaseHandler):
        __slots__ = ("token",)

        def custom_open(self, request):
            return _response(request.full_url, self.token.encode())

    handler = SlotGatewayHandler()
    handler.token = "token-a"
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)
    private_openers: list[object] = []

    for suffix in ("a", "b"):
        with _urlopen_no_redirect(
            urllib.request.Request(f"custom://source.invalid/{suffix}"), timeout=1.0
        ) as response:
            assert response.read() == b"token-a"
        private_openers.append(http_transport._INSTALLED_OPENER_CACHE.private)

    assert private_openers[0] is private_openers[1]
    assert handler.parent is opener
    assert handler.token == "token-a"


def test_unsupported_slot_descriptor_fails_before_transport() -> None:
    contacted: list[str] = []
    descriptor_calls: list[str] = []

    class SlotGatewayHandler(urllib.request.BaseHandler):
        __slots__ = ("token",)

        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = SlotGatewayHandler()
    handler.token = "token-a"

    def malicious(_self):
        descriptor_calls.append("called")
        return "malicious"

    SlotGatewayHandler.token = property(malicious)
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)
    original_parent = handler.parent

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []
    assert descriptor_calls == []
    assert urllib.request._opener is opener
    assert handler.parent is original_parent


def test_installed_digest_handler_state_survives_across_requests() -> None:
    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password("realm", "https://source.invalid", "user", "pass")
    digest = urllib.request.HTTPDigestAuthHandler(password_manager)
    seen_nonce_counts: list[int] = []

    class SlotConfiguration(urllib.request.BaseHandler):
        __slots__ = ("token",)

        def https_request(self, request):
            return request

    slot_configuration = SlotConfiguration()
    slot_configuration.token = "unchanged"

    class DigestSource(urllib.request.BaseHandler):
        handler_order = 100

        def https_open(self, request):
            authorization = request.get_header("Authorization")
            if authorization:
                seen_nonce_counts.append(
                    next(
                        item
                        for item in self.parent.handlers
                        if isinstance(item, urllib.request.HTTPDigestAuthHandler)
                    ).nonce_count
                )
                return _response(request.full_url)
            headers = email.message.Message()
            headers["WWW-Authenticate"] = (
                'Digest realm="realm", nonce="nonce-a", algorithm="MD5", qop="auth"'
            )
            response = urllib.response.addinfourl(
                io.BytesIO(b""), headers, request.full_url, code=401
            )
            response.msg = "unauthorized"
            return response

    opener = urllib.request.build_opener(DigestSource(), slot_configuration, digest)
    urllib.request.install_opener(opener)

    private_openers: list[object] = []
    for suffix in ("a", "b"):
        with _urlopen_no_redirect(
            urllib.request.Request(f"https://source.invalid/{suffix}"), timeout=1.0
        ) as response:
            assert response.read() == b"ok"
        private_openers.append(http_transport._INSTALLED_OPENER_CACHE.private)

    private_digest = next(
        handler
        for handler in http_transport._INSTALLED_OPENER_CACHE.private.handlers
        if isinstance(handler, urllib.request.HTTPDigestAuthHandler)
    )
    assert seen_nonce_counts == [1, 2]
    assert private_openers[0] is private_openers[1]
    assert private_digest.nonce_count == 2
    assert private_digest.last_nonce == "nonce-a"
    assert digest.nonce_count == 0
    assert digest.last_nonce is None
    assert slot_configuration.token == "unchanged"


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_generic_redirect_error_handler_cannot_reach_sink(status: int) -> None:
    source = "https://source.invalid/data"
    target = "https://sink.invalid/steal"
    observed: list[str] = []
    redirect_calls: list[str] = []

    class RedirectSource(urllib.request.BaseHandler):
        def default_open(self, request):
            observed.append(request.full_url)
            if request.full_url == source:
                headers = email.message.Message()
                headers["Location"] = target
                response = urllib.response.addinfourl(
                    io.BytesIO(b""), headers, request.full_url, code=status
                )
                response.msg = "redirect"
                return response
            return _response(request.full_url)

    class GenericRedirectHandler(urllib.request.BaseHandler):
        handler_order = -100

        def http_error_redirect(self, request, fp, code, msg, headers):
            redirect_calls.append(request.full_url)
            return self.parent.open(headers["Location"], timeout=request.timeout)

        http_error_301 = http_error_redirect
        http_error_302 = http_error_redirect
        http_error_303 = http_error_redirect
        http_error_307 = http_error_redirect
        http_error_308 = http_error_redirect

    opener = urllib.request.build_opener(RedirectSource(), GenericRedirectHandler())
    urllib.request.install_opener(opener)

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _urlopen_no_redirect(
            urllib.request.Request(source, headers={"Authorization": "Bearer secret"}),
            timeout=1.0,
        )

    assert exc_info.value.code == status
    assert observed == [source]
    assert redirect_calls == []


@pytest.mark.parametrize("protocol", ["http", "https"])
@pytest.mark.parametrize("status", [300, 301, 302, 303, 304, 305, 306, 307, 308, 399])
def test_early_response_processor_cannot_follow_redirect(protocol: str, status: int) -> None:
    source = f"{protocol}://source.invalid/data"
    target = f"{protocol}://sink.invalid/steal"
    observed: list[str] = []
    processor_calls: list[str] = []

    class RedirectSource(urllib.request.BaseHandler):
        def default_open(self, request):
            observed.append(request.full_url)
            if request.full_url == source:
                headers = email.message.Message()
                headers["Location"] = target
                response = urllib.response.addinfourl(
                    io.BytesIO(b""), headers, request.full_url, code=status
                )
                response.msg = "redirect"
                return response
            return _response(request.full_url)

    class EarlyResponseProcessor(urllib.request.BaseHandler):
        handler_order = -1000

        def process(self, request, response):
            processor_calls.append(request.full_url)
            if 300 <= response.code < 400:
                redirected = urllib.request.Request(
                    response.headers["Location"],
                    headers={"Authorization": request.get_header("Authorization")},
                )
                return self.parent.open(redirected, timeout=request.timeout)
            return response

        http_response = process
        https_response = process

    opener = urllib.request.build_opener(RedirectSource(), EarlyResponseProcessor())
    urllib.request.install_opener(opener)

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _urlopen_no_redirect(
            urllib.request.Request(source, headers={"Authorization": "Bearer secret"}),
            timeout=1.0,
        )

    assert exc_info.value.code == status
    assert observed == [source]
    assert processor_calls == []


@pytest.mark.parametrize("protocol", ["http", "https"])
def test_success_response_processors_keep_their_relative_order(protocol: str) -> None:
    observed: list[str] = []

    class TerminalHandler(urllib.request.BaseHandler):
        def default_open(self, request):
            return _response(request.full_url)

    class FirstProcessor(urllib.request.BaseHandler):
        handler_order = -1000

        def process(self, request, response):
            observed.append("first")
            return response

        http_response = process
        https_response = process

    class SecondProcessor(urllib.request.BaseHandler):
        handler_order = -2000

        def process(self, request, response):
            observed.append("second")
            return response

        http_response = process
        https_response = process

    opener = urllib.request.build_opener(TerminalHandler(), FirstProcessor(), SecondProcessor())
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(
        urllib.request.Request(f"{protocol}://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    assert observed == ["second", "first"]


@pytest.mark.parametrize("protocol", ["http", "https"])
def test_non_redirect_error_reaches_later_response_processors(protocol: str) -> None:
    processor_calls: list[int] = []

    class ErrorSource(urllib.request.BaseHandler):
        def default_open(self, request):
            response = urllib.response.addinfourl(
                io.BytesIO(b"error"), email.message.Message(), request.full_url, code=418
            )
            response.msg = "teapot"
            return response

    class ErrorProcessor(urllib.request.BaseHandler):
        handler_order = -1000

        def process(self, request, response):
            processor_calls.append(response.code)
            return response

        http_response = process
        https_response = process

    opener = urllib.request.build_opener(ErrorSource(), ErrorProcessor())
    urllib.request.install_opener(opener)

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _urlopen_no_redirect(
            urllib.request.Request(f"{protocol}://source.invalid/error"), timeout=1.0
        )

    assert exc_info.value.code == 418
    assert processor_calls == [418]


def test_concurrent_custom_opener_cache_construction_is_singleton() -> None:
    barrier = threading.Barrier(8)
    seen_parents: list[object] = []
    seen_lock = threading.Lock()

    class TerminalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            with seen_lock:
                seen_parents.append(self.parent)
            return _response(request.full_url)

    opener = urllib.request.build_opener(TerminalHandler())
    urllib.request.install_opener(opener)

    def request(index: int) -> bytes:
        barrier.wait()
        with _urlopen_no_redirect(
            urllib.request.Request(f"custom://source.invalid/{index}"), timeout=1.0
        ) as response:
            return response.read()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(request, range(8)))

    assert results == [b"ok"] * 8
    assert len({id(parent) for parent in seen_parents}) == 1
    assert seen_parents[0] is http_transport._INSTALLED_OPENER_CACHE.private


def test_installed_redirect_handler_cannot_reach_sink() -> None:
    source = "https://source.invalid/data"
    target = "https://sink.invalid/steal"
    observed: list[str] = []

    class RedirectSource(urllib.request.BaseHandler):
        def default_open(self, request):
            observed.append(request.full_url)
            if request.full_url == source:
                headers = email.message.Message()
                headers["Location"] = target
                response = urllib.response.addinfourl(
                    io.BytesIO(b""), headers, request.full_url, code=302
                )
                response.msg = "redirect"
                return response
            return _response(request.full_url)

    class PermissiveRedirect(urllib.request.HTTPRedirectHandler):
        pass

    redirect = PermissiveRedirect()
    opener = urllib.request.build_opener(RedirectSource(), redirect)
    urllib.request.install_opener(opener)
    original_parent = redirect.parent

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        _urlopen_no_redirect(
            urllib.request.Request(source, headers={"Authorization": "Bearer secret"}),
            timeout=1.0,
        )

    assert exc_info.value.code == 302
    assert observed == [source]
    assert redirect.parent is original_parent
    assert urllib.request._opener is opener


def test_installed_opener_subclass_override_fails_before_transport() -> None:
    contacted: list[str] = []

    class OverrideOpener(urllib.request.OpenerDirector):
        def open(self, fullurl, data=None, timeout=None):
            contacted.append(str(fullurl))
            return _response("custom://source.invalid/data")

    opener = OverrideOpener()
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib opener cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []
    assert urllib.request._opener is opener


def test_installed_opener_internal_open_override_fails_before_transport() -> None:
    contacted: list[str] = []

    class OverrideOpener(urllib.request.OpenerDirector):
        def _open(self, request, data=None):
            contacted.append(request.full_url)
            return _response(request.full_url)

    opener = OverrideOpener()
    request = urllib.request.Request("custom://source.invalid/data")
    with opener.open(request, timeout=1.0) as response:
        assert response.read() == b"ok"
    contacted.clear()
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib opener cannot be copied safely"
    ):
        _urlopen_no_redirect(request, timeout=1.0)

    assert contacted == []
    assert urllib.request._opener is opener


def test_installed_opener_dynamic_open_override_fails_before_transport() -> None:
    dispatch_calls: list[str] = []
    handler_calls: list[str] = []

    class DynamicOpener(urllib.request.OpenerDirector):
        def __getattribute__(self, name):
            if name == "_open":

                def override(request, data=None):
                    dispatch_calls.append(request.full_url)
                    return _response(request.full_url)

                return override
            return super().__getattribute__(name)

    class UnexpectedHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            handler_calls.append(request.full_url)
            return _response(request.full_url)

    opener = DynamicOpener()
    opener.add_handler(UnexpectedHandler())
    request = urllib.request.Request("custom://source.invalid/data")
    with opener.open(request, timeout=1.0) as response:
        assert response.read() == b"ok"
    dispatch_calls.clear()
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib opener cannot be copied safely"
    ):
        _urlopen_no_redirect(request, timeout=1.0)

    assert dispatch_calls == []
    assert handler_calls == []
    assert urllib.request._opener is opener


def test_installed_opener_instance_override_fails_before_transport() -> None:
    contacted: list[str] = []
    opener = urllib.request.build_opener()

    def override(fullurl, data=None, timeout=None):
        contacted.append(str(fullurl))
        return _response("custom://source.invalid/data")

    opener.open = override
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib opener cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []
    assert urllib.request._opener is opener
    assert opener.open is override


def test_manually_assembled_opener_does_not_gain_default_handlers() -> None:
    contacted: list[str] = []

    class RecordingHttpHandler(urllib.request.HTTPHandler):
        def http_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    original_http_handler = urllib.request.HTTPHandler
    urllib.request.HTTPHandler = RecordingHttpHandler
    try:
        opener = urllib.request.OpenerDirector()
        urllib.request.install_opener(opener)

        response = _urlopen_no_redirect(
            urllib.request.Request("http://source.invalid/data"), timeout=1.0
        )
    finally:
        urllib.request.HTTPHandler = original_http_handler

    assert response is None
    assert contacted == []
    assert urllib.request._opener is opener
    assert not any(isinstance(handler, RecordingHttpHandler) for handler in opener.handlers)


def test_installed_opener_inheriting_standard_open_is_supported() -> None:
    observed: list[str] = []

    class StandardOpener(urllib.request.OpenerDirector):
        pass

    class TerminalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            observed.append(request.full_url)
            return _response(request.full_url)

    opener = StandardOpener()
    opener.add_handler(TerminalHandler())
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    assert observed == ["custom://source.invalid/data"]
    assert urllib.request._opener is opener


def test_resetting_installed_opener_releases_cached_original() -> None:
    class TerminalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            return _response(request.full_url)

    opener = urllib.request.build_opener(TerminalHandler())
    opener_ref = weakref.ref(opener)
    urllib.request.install_opener(opener)
    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/first"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"
    assert http_transport._INSTALLED_OPENER_CACHE.installed is opener

    urllib.request._opener = None
    del opener
    with _urlopen_no_redirect(
        urllib.request.Request("data:text/plain,ok"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"
    gc.collect()

    assert http_transport._INSTALLED_OPENER_CACHE is None
    assert opener_ref() is None


def test_non_none_install_releases_cached_openers_immediately() -> None:
    class TerminalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            return _response(request.full_url)

    first = urllib.request.build_opener(TerminalHandler())
    urllib.request.install_opener(first)
    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/first"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"
    first_private = http_transport._INSTALLED_OPENER_CACHE.private
    first_ref = weakref.ref(first)
    first_private_ref = weakref.ref(first_private)

    urllib.request.install_opener(urllib.request.build_opener(TerminalHandler()))
    del first
    del first_private
    gc.collect()

    assert http_transport._INSTALLED_OPENER_CACHE is None
    assert first_ref() is None
    assert first_private_ref() is None


def test_saved_install_alias_replacement_does_not_reuse_stale_cache() -> None:
    observed: list[str] = []

    class TerminalHandler(urllib.request.BaseHandler):
        def __init__(self, name: str):
            self.name = name

        def custom_open(self, request):
            observed.append(self.name)
            return _response(request.full_url)

    saved_install_opener = http_transport._STDLIB_INSTALL_OPENER
    first = urllib.request.build_opener(TerminalHandler("first"))
    second = urllib.request.build_opener(TerminalHandler("second"))
    urllib.request.install_opener(first)
    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/first"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"
    first_private = http_transport._INSTALLED_OPENER_CACHE.private

    saved_install_opener(second)
    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/second"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    assert observed == ["first", "second"]
    assert http_transport._INSTALLED_OPENER_CACHE.installed is second
    assert http_transport._INSTALLED_OPENER_CACHE.private is not first_private


def test_replaced_installed_opener_is_observed() -> None:
    observed: list[str] = []

    class TerminalHandler(urllib.request.BaseHandler):
        def __init__(self, name: str):
            self.name = name

        def default_open(self, request):
            observed.append(self.name)
            return _response(request.full_url)

    first = urllib.request.build_opener(TerminalHandler("first"))
    second = urllib.request.build_opener(TerminalHandler("second"))
    urllib.request.install_opener(first)
    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/first"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"
    first_private = http_transport._INSTALLED_OPENER_CACHE.private
    urllib.request.install_opener(second)
    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/second"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    assert observed == ["first", "second"]
    assert http_transport._INSTALLED_OPENER_CACHE.private is not first_private
    assert urllib.request._opener is second


@pytest.mark.parametrize(
    ("replacement_kind", "spoof_metadata"),
    [("function", False), ("lambda", False), ("function", True)],
)
def test_urlopen_replaced_before_helper_import_is_called(
    replacement_kind: str, spoof_metadata: bool
) -> None:
    script = textwrap.dedent(
        f"""
        import urllib.request

        calls = []

        class UnexpectedTransport(urllib.request.BaseHandler):
            def https_open(self, request):
                calls.append(("unexpected", request.full_url))
                raise AssertionError("real transport was used")

        urllib.request.install_opener(urllib.request.build_opener(UnexpectedTransport()))

        def replacement_function(request, timeout):
            calls.append(("replacement", request.full_url, timeout))
            return "replacement-response"

        replacement_lambda = lambda request, timeout: (
            calls.append(("replacement", request.full_url, timeout)),
            "replacement-response",
        )[1]
        replacement = (
            replacement_function
            if {replacement_kind!r} == "function"
            else replacement_lambda
        )

        if {spoof_metadata!r}:
            replacement.__module__ = urllib.request.__name__
            replacement.__name__ = "urlopen"

        urllib.request.urlopen = replacement

        from flash._internal.http import _urlopen_no_redirect

        response = _urlopen_no_redirect(
            urllib.request.Request("https://source.invalid/data"), timeout=3.0
        )
        assert response == "replacement-response"
        assert calls == [("replacement", "https://source.invalid/data", 3.0)]
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.dirname(__file__)),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_unknown_stdlib_urlopen_fails_closed_before_transport(monkeypatch) -> None:
    contacted: list[str] = []

    class UnexpectedTransport(urllib.request.BaseHandler):
        def https_open(self, request):
            contacted.append(request.full_url)
            raise AssertionError("transport was contacted")

    original = urllib.request.urlopen
    code = original.__code__.replace(co_names=(*original.__code__.co_names, "vendor_hook"))
    vendor_urlopen = types.FunctionType(
        code,
        original.__globals__,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    vendor_urlopen.__kwdefaults__ = original.__kwdefaults__
    vendor_urlopen.__module__ = original.__module__
    vendor_urlopen.__qualname__ = original.__qualname__
    monkeypatch.setattr(urllib.request, "urlopen", vendor_urlopen)
    urllib.request.install_opener(urllib.request.build_opener(UnexpectedTransport()))

    with pytest.raises(urllib.error.URLError) as exc_info:
        _urlopen_no_redirect(urllib.request.Request("https://source.invalid/data"), timeout=3.0)

    assert exc_info.value.reason == "stdlib urllib transport cannot be classified safely"
    assert str(exc_info.value) == (
        "<urlopen error stdlib urllib transport cannot be classified safely>"
    )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "vendor_hook" not in str(exc_info.value)
    assert contacted == []


def test_sourceless_stdlib_urlopen_fails_closed_before_transport(monkeypatch) -> None:
    contacted: list[str] = []

    class UnexpectedTransport(urllib.request.BaseHandler):
        def https_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    monkeypatch.setattr(urllib.request, "__file__", f"{urllib.request.__file__}c")
    urllib.request.install_opener(urllib.request.build_opener(UnexpectedTransport()))

    with pytest.raises(urllib.error.URLError) as exc_info:
        _urlopen_no_redirect(urllib.request.Request("https://source.invalid/data"), timeout=3.0)

    assert exc_info.value.reason == "stdlib urllib transport cannot be classified safely"
    assert contacted == []


def test_urlopen_helper_preserves_late_bound_monkeypatch(monkeypatch) -> None:
    seen = []
    response = object()

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, timeout))
        return response

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    actual = _urlopen_no_redirect(
        urllib.request.Request("https://source.invalid/data"), timeout=3.0
    )

    assert actual is response
    assert seen == [("https://source.invalid/data", 3.0)]


def test_urlopen_helper_preserves_explicit_injected_transport() -> None:
    seen = []
    response = object()

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, timeout))
        return response

    actual = _urlopen_no_redirect(
        urllib.request.Request("https://source.invalid/data"),
        timeout=3.0,
        urlopen=fake_urlopen,
    )

    assert actual is response
    assert seen == [("https://source.invalid/data", 3.0)]


def test_explicit_injected_stdlib_code_clone_is_called_directly() -> None:
    original = urllib.request.urlopen
    calls: list[tuple[str, float]] = []
    globals_copy = dict(original.__globals__)

    class RecordingOpener:
        def open(self, request, data=None, timeout=None):
            calls.append((request.full_url, timeout))
            return "injected-response"

    globals_copy["_opener"] = RecordingOpener()
    replacement = types.FunctionType(
        original.__code__,
        globals_copy,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    replacement.__kwdefaults__ = original.__kwdefaults__
    replacement.__module__ = original.__module__
    replacement.__qualname__ = original.__qualname__

    actual = _urlopen_no_redirect(
        urllib.request.Request("https://source.invalid/data"),
        timeout=3.0,
        urlopen=replacement,
    )

    assert actual == "injected-response"
    assert calls == [("https://source.invalid/data", 3.0)]


def test_explicit_global_urlopen_is_compared_against_one_snapshot(monkeypatch) -> None:
    explicit = urllib.request.urlopen
    calls: list[str] = []
    reads = 0

    class TerminalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            calls.append("opener")
            return _response(request.full_url)

    def replacement(request, timeout):
        calls.append("replacement")
        return _response(request.full_url)

    class SwitchingRequestModule(types.ModuleType):
        def __getattribute__(self, name):
            nonlocal reads
            if name == "urlopen":
                reads += 1
                return explicit if reads == 1 else replacement
            return super().__getattribute__(name)

    monkeypatch.setattr(urllib.request, "__class__", SwitchingRequestModule)
    urllib.request.install_opener(urllib.request.build_opener(TerminalHandler()))

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"),
        timeout=3.0,
        urlopen=explicit,
    ) as response:
        assert response.read() == b"ok"

    assert reads == 1
    assert calls == ["opener"]


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_builtin_bound_method_retaining_target_fails_before_transport(target_kind: str) -> None:
    contacted: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = CallbackHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener
    holder = [target]
    handler.callback = holder.append
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []
    assert holder == [target]


def test_copied_handler_callback_bound_to_original_fails_before_transport() -> None:
    callback_calls: list[str] = []

    class InstanceCallbackHandler(urllib.request.BaseHandler):
        def __init__(self):
            self.custom_open = self.open_custom

        def open_custom(self, request):
            callback_calls.append(request.full_url)
            return _response(request.full_url)

    handler = InstanceCallbackHandler()
    opener = urllib.request.build_opener(handler)
    request = urllib.request.Request("custom://source.invalid/data")
    with opener.open(request, timeout=1.0) as response:
        assert response.read() == b"ok"
    callback_calls.clear()
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(request, timeout=1.0)

    assert callback_calls == []
    assert handler.parent is opener
    assert urllib.request._opener is opener


@pytest.mark.parametrize("callback_kind", ["bound", "lambda", "partial"])
def test_copied_handler_nested_callback_fails_before_transport(callback_kind: str) -> None:
    callback_calls: list[str] = []

    class NestedCallbackHandler(urllib.request.BaseHandler):
        def __init__(self):
            if callback_kind == "lambda":

                def callback(request):
                    return self.open_custom(request)
            elif callback_kind == "partial":
                callback = functools.partial(self.open_custom)
            else:
                callback = self.open_custom
            self.config = {"callbacks": [callback]}

        def custom_open(self, request):
            return self.config["callbacks"][0](request)

        def open_custom(self, request):
            callback_calls.append(request.full_url)
            return _response(request.full_url)

    handler = NestedCallbackHandler()
    opener = urllib.request.build_opener(handler)
    request = urllib.request.Request("custom://source.invalid/data")
    with opener.open(request, timeout=1.0) as response:
        assert response.read() == b"ok"
    callback_calls.clear()
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(request, timeout=1.0)

    assert callback_calls == []
    assert handler.parent is opener


def test_copied_handler_callback_in_inherited_mangled_slot_fails_before_transport() -> None:
    callback_calls: list[str] = []

    class CallbackSlots(urllib.request.BaseHandler):
        __slots__ = ("__callback",)

        def bind_callback(self):
            self.__callback = self.open_custom

        def dispatch(self, request):
            return self.__callback(request)

    class SlottedCallbackHandler(CallbackSlots):
        def __init__(self):
            self.bind_callback()

        def custom_open(self, request):
            return self.dispatch(request)

        def open_custom(self, request):
            callback_calls.append(request.full_url)
            return _response(request.full_url)

    handler = SlottedCallbackHandler()
    assert not any(type(value) is types.MethodType for value in handler.__dict__.values())
    opener = urllib.request.build_opener(handler)
    request = urllib.request.Request("custom://source.invalid/data")
    with opener.open(request, timeout=1.0) as response:
        assert response.read() == b"ok"
    callback_calls.clear()
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(request, timeout=1.0)

    assert callback_calls == []
    assert handler.parent is opener
    assert urllib.request._opener is opener


def test_copied_handler_state_retaining_installed_opener_open_fails_before_transport() -> None:
    contacted: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = CallbackHandler()
    opener = urllib.request.build_opener(handler)
    handler.callback = opener.open
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("container_kind", ["list", "partial"])
def test_copied_handler_callable_subclass_fails_before_transport(
    container_kind: str,
) -> None:
    contacted: list[str] = []

    class CallbackList(list):
        pass

    class CallbackPartial(functools.partial):
        pass

    class CallbackHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = CallbackHandler()
    if container_kind == "list":
        handler.callback = CallbackList([handler.custom_open])
    else:
        handler.callback = CallbackPartial(handler.custom_open)
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("protocol_name", http_transport._HANDLER_COPY_METHODS)
@pytest.mark.parametrize("shadow_location", ["class", "instance"])
def test_malicious_copy_protocol_descriptor_does_not_execute(
    protocol_name: str,
    shadow_location: str,
) -> None:
    descriptor_calls: list[str] = []
    contacted: list[str] = []

    class MaliciousDescriptor:
        def __get__(self, instance, owner=None):
            descriptor_calls.append(protocol_name)
            raise AssertionError("descriptor executed")

    class ProtocolHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = ProtocolHandler()
    opener = urllib.request.build_opener(handler)
    if shadow_location == "class":
        setattr(ProtocolHandler, protocol_name, MaliciousDescriptor())
    else:
        handler.__dict__[protocol_name] = MaliciousDescriptor()
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert descriptor_calls == []
    assert contacted == []


def test_overridden_new_does_not_execute_during_handler_copy() -> None:
    new_calls: list[str] = []
    contacted: list[str] = []

    class NewHandler(urllib.request.BaseHandler):
        def __new__(cls):
            new_calls.append("called")
            return super().__new__(cls)

        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = NewHandler()
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)
    new_calls.clear()

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    assert new_calls == []
    assert contacted == ["custom://source.invalid/data"]


def test_custom_metaclass_getattribute_does_not_execute_during_handler_copy() -> None:
    metaclass_calls: list[str] = []
    contacted: list[str] = []

    class RecordingMeta(type):
        def __getattribute__(cls, name):
            metaclass_calls.append(name)
            return super().__getattribute__(name)

    class MetaHandler(urllib.request.BaseHandler, metaclass=RecordingMeta):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = MetaHandler()
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)
    metaclass_calls.clear()

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert metaclass_calls == []
    assert contacted == []


def test_copyreg_reducer_does_not_execute_during_handler_copy(monkeypatch) -> None:
    reducer_calls: list[str] = []
    contacted: list[str] = []

    class ReducerHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    def reducer(handler):
        reducer_calls.append("called")
        return type(handler), ()

    handler = ReducerHandler()
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)
    monkeypatch.setitem(copyreg.dispatch_table, ReducerHandler, reducer)

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    assert reducer_calls == []
    assert contacted == ["custom://source.invalid/data"]


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("container", [False, True])
def test_copied_function_global_target_fails_before_transport(
    target_kind: str,
    container: bool,
) -> None:
    contacted: list[str] = []

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener
    retained = {"target": target} if container else target

    module_code = compile(
        "def callback():\n    return retained_global\n",
        "<handler-global-callback>",
        "exec",
    )
    callback_code = next(item for item in module_code.co_consts if type(item) is types.CodeType)
    handler.callback = types.FunctionType(
        callback_code,
        {"retained_global": retained},
        "callback",
    )
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("nested", [False, True])
def test_copied_function_global_holder_target_fails_before_transport(
    target_kind: str,
    nested: bool,
) -> None:
    contacted: list[str] = []

    class Holder:
        pass

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener
    inner = Holder()
    inner.target = target
    holder = Holder()
    if nested:
        holder.target = {"nested": [inner]}
    else:
        holder.target = target
    module_code = compile(
        "def callback():\n    return retained_holder.target\n",
        "<handler-global-holder-callback>",
        "exec",
    )
    callback_code = next(item for item in module_code.co_consts if type(item) is types.CodeType)
    handler.callback = types.FunctionType(
        callback_code,
        {"retained_holder": holder},
        "callback",
    )
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("namespace_kind", ["module", "class"])
@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("nested", [False, True])
def test_copied_function_global_namespace_attribute_fails_before_transport(
    namespace_kind: str,
    target_kind: str,
    nested: bool,
) -> None:
    contacted: list[str] = []

    class Namespace:
        pass

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener
    retained = {"nested": [target]} if nested else target
    if namespace_kind == "module":
        namespace = types.ModuleType("handler_namespace")
        types.ModuleType.__getattribute__(namespace, "__dict__")["target"] = retained
    else:
        namespace = Namespace
        type.__setattr__(namespace, "target", retained)
    module_code = compile(
        "def callback():\n    return retained_namespace.target\n",
        "<handler-global-namespace-callback>",
        "exec",
    )
    callback_code = next(item for item in module_code.co_consts if type(item) is types.CodeType)
    handler.callback = types.FunctionType(
        callback_code,
        {"retained_namespace": namespace},
        "callback",
    )
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("namespace_kind", ["module", "class"])
@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("binding_kind", ["default", "closure"])
@pytest.mark.parametrize("nested", [False, True])
def test_copied_function_bound_namespace_attribute_fails_before_transport(
    namespace_kind: str,
    target_kind: str,
    binding_kind: str,
    nested: bool,
) -> None:
    contacted: list[str] = []

    class Namespace:
        pass

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener
    retained = {"nested": [target]} if nested else target
    if namespace_kind == "module":
        namespace = types.ModuleType("bound_handler_namespace")
        types.ModuleType.__getattribute__(namespace, "__dict__")["target"] = retained
    else:
        namespace = Namespace
        type.__setattr__(namespace, "target", retained)
    if binding_kind == "default":

        def callback(bound_namespace=namespace):
            return bound_namespace.target

    else:
        bound_namespace = namespace

        def callback():
            return bound_namespace.target

    handler.callback = callback
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("namespace_kind", ["module", "class"])
@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("binding_kind", ["default", "closure"])
@pytest.mark.parametrize("nested_kind", ["lambda", "function"])
@pytest.mark.parametrize("nested_container", [False, True])
def test_copied_nested_function_bound_namespace_fails_before_transport(
    namespace_kind: str,
    target_kind: str,
    binding_kind: str,
    nested_kind: str,
    nested_container: bool,
) -> None:
    contacted: list[str] = []

    class Namespace:
        pass

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener
    retained = {"nested": [target]} if nested_container else target
    if namespace_kind == "module":
        namespace = types.ModuleType("nested_bound_handler_namespace")
        types.ModuleType.__getattribute__(namespace, "__dict__")["target"] = retained
    else:
        namespace = Namespace
        type.__setattr__(namespace, "target", retained)
    if binding_kind == "default":
        if nested_kind == "lambda":

            def callback(bound_namespace=namespace):
                return lambda: bound_namespace.target

        else:

            def callback(bound_namespace=namespace):
                def nested():
                    return bound_namespace.target

                return nested

    else:
        bound_namespace = namespace
        if nested_kind == "lambda":

            def callback():
                return lambda: bound_namespace.target

        else:

            def callback():
                def nested():
                    return bound_namespace.target

                return nested

    handler.callback = callback
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("dynamic_name", ["eval", "exec"])
@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("binding_kind", ["default", "closure"])
@pytest.mark.parametrize("nested_kind", ["lambda", "function"])
def test_copied_nested_function_bound_dynamic_builtin_fails_before_transport(
    dynamic_name: str,
    target_kind: str,
    binding_kind: str,
    nested_kind: str,
) -> None:
    contacted: list[str] = []

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener
    namespace = builtins
    if binding_kind == "default":
        if nested_kind == "lambda":
            if dynamic_name == "eval":

                def callback(namespace=namespace):
                    return lambda: namespace.eval("retained_global")

            else:

                def callback(namespace=namespace):
                    return lambda: namespace.exec("sink = retained_global")

        elif dynamic_name == "eval":

            def callback(namespace=namespace):
                def nested():
                    return namespace.eval("retained_global")

                return nested

        else:

            def callback(namespace=namespace):
                def nested():
                    return namespace.exec("sink = retained_global")

                return nested

    elif nested_kind == "lambda":
        if dynamic_name == "eval":

            def callback():
                return lambda: namespace.eval("retained_global")

        else:

            def callback():
                return lambda: namespace.exec("sink = retained_global")

    elif dynamic_name == "eval":

        def callback():
            def nested():
                return namespace.eval("retained_global")

            return nested

    else:

        def callback():
            def nested():
                return namespace.exec("sink = retained_global")

            return nested

    callback = types.FunctionType(
        callback.__code__,
        {"retained_global": target},
        "callback",
        callback.__defaults__,
        callback.__closure__,
    )
    handler.callback = callback
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("dynamic_name", ["eval", "exec"])
@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("binding_kind", ["default", "closure"])
def test_copied_function_bound_dynamic_builtin_fails_before_transport(
    dynamic_name: str,
    target_kind: str,
    binding_kind: str,
) -> None:
    contacted: list[str] = []

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener
    expression = (
        'namespace.eval("retained_global")'
        if dynamic_name == "eval"
        else 'namespace.exec("sink = retained_global")'
    )
    if binding_kind == "default":
        module_code = compile(
            f"def callback(namespace=None):\n    return {expression}\n",
            "<handler-default-dynamic-namespace-callback>",
            "exec",
        )
        callback_code = next(item for item in module_code.co_consts if type(item) is types.CodeType)
        callback = types.FunctionType(
            callback_code,
            {"retained_global": target},
            "callback",
            (builtins,),
        )
    else:
        namespace = builtins
        if dynamic_name == "eval":

            def bound_callback():
                return namespace.eval("retained_global")

        else:

            def bound_callback():
                return namespace.exec("sink = retained_global")

        callback = types.FunctionType(
            bound_callback.__code__,
            {"retained_global": target},
            "callback",
            closure=bound_callback.__closure__,
        )
    handler.callback = callback
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize(
    "namespace_kind",
    ["instance_getattribute", "instance_descriptor", "module_subclass", "class_metaclass"],
)
@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_unsafe_namespace_lookup_hooks_do_not_execute(
    namespace_kind: str,
    target_kind: str,
) -> None:
    hook_calls: list[str] = []
    contacted: list[str] = []

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener

    class DynamicHolder:
        def __getattribute__(self, name):
            if name == "target":
                hook_calls.append("instance_getattribute")
                return target
            return object.__getattribute__(self, name)

    class TargetDescriptor:
        def __get__(self, instance, owner=None):
            hook_calls.append("instance_descriptor")
            return target

        def __set__(self, instance, value):
            hook_calls.append("instance_descriptor_set")

    class DescriptorHolder:
        target = TargetDescriptor()

    class DynamicModule(types.ModuleType):
        def __getattribute__(self, name):
            if name == "target":
                hook_calls.append("module_subclass")
                return target
            return types.ModuleType.__getattribute__(self, name)

    class DynamicMeta(type):
        def __getattribute__(cls, name):
            if name == "target":
                hook_calls.append("class_metaclass")
                return target
            return type.__getattribute__(cls, name)

    class DynamicClass(metaclass=DynamicMeta):
        target = object()

    if namespace_kind == "instance_getattribute":
        namespace = DynamicHolder()
        object.__setattr__(namespace, "target", object())
    elif namespace_kind == "instance_descriptor":
        namespace = DescriptorHolder()
        object.__setattr__(namespace, "target", object())
    elif namespace_kind == "module_subclass":
        namespace = DynamicModule("dynamic_handler_namespace")
        types.ModuleType.__getattribute__(namespace, "__dict__")["target"] = object()
    else:
        namespace = DynamicClass
    hook_calls.clear()

    def callback(bound_namespace=namespace):
        return bound_namespace.target

    handler.callback = callback
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert hook_calls == []
    assert contacted == []


@pytest.mark.parametrize("namespace_kind", ["module", "class"])
@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("binding_kind", ["default", "closure"])
@pytest.mark.parametrize("nested_kind", ["lambda", "function"])
@pytest.mark.parametrize("nested_container", [False, True])
def test_copied_nested_parameter_default_provenance_fails_before_transport(
    namespace_kind: str,
    target_kind: str,
    binding_kind: str,
    nested_kind: str,
    nested_container: bool,
) -> None:
    contacted: list[str] = []

    class Namespace:
        pass

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener
    retained = {"nested": [target]} if nested_container else target
    if namespace_kind == "module":
        namespace = types.ModuleType("nested_parameter_namespace")
        types.ModuleType.__getattribute__(namespace, "__dict__")["target"] = retained
    else:
        namespace = Namespace
        type.__setattr__(namespace, "target", retained)
    if binding_kind == "default":
        if nested_kind == "lambda":

            def callback(outer=namespace):
                return lambda child=outer: child.target

        else:

            def callback(outer=namespace):
                def nested(child=outer):
                    return child.target

                return nested

    else:
        outer = namespace
        if nested_kind == "lambda":

            def callback():
                return lambda child=outer: child.target

        else:

            def callback():
                def nested(child=outer):
                    return child.target

                return nested

    handler.callback = callback
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("dynamic_name", ["eval", "exec"])
@pytest.mark.parametrize("binding_kind", ["default", "closure"])
def test_copied_nested_parameter_default_dynamic_builtin_fails_before_transport(
    dynamic_name: str,
    binding_kind: str,
) -> None:
    contacted: list[str] = []

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()
    opener = urllib.request.build_opener(handler)
    if binding_kind == "default":
        if dynamic_name == "eval":

            def callback(outer=builtins):
                return lambda child=outer: child.eval("retained_global")

        else:

            def callback(outer=builtins):
                return lambda child=outer: child.exec("sink = retained_global")

    else:
        outer = builtins
        if dynamic_name == "eval":

            def callback():
                return lambda child=outer: child.eval("retained_global")

        else:

            def callback():
                return lambda child=outer: child.exec("sink = retained_global")

    callback = types.FunctionType(
        callback.__code__,
        {"retained_global": handler},
        "callback",
        callback.__defaults__,
        callback.__closure__,
    )
    handler.callback = callback
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


def test_malformed_kwdefaults_key_fails_before_transport() -> None:
    contacted: list[str] = []

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()

    def callback(*, namespace="safe"):
        local = object()
        return (local, namespace)[0]

    callback.__kwdefaults__ = {"local": handler}
    handler.callback = callback
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


def test_legitimate_keyword_only_default_remains_supported() -> None:
    contacted: list[str] = []

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()

    def callback(*, token="safe"):
        return token

    handler.callback = callback
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    assert contacted == ["custom://source.invalid/data"]


def test_copied_function_dynamic_globals_lookup_fails_before_transport() -> None:
    contacted: list[str] = []

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()
    opener = urllib.request.build_opener(handler)

    def callback():
        return globals()["retained_global"]

    handler.callback = types.FunctionType(
        callback.__code__,
        {"retained_global": handler},
        callback.__name__,
    )
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("dynamic_name", ["eval", "exec"])
@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_copied_function_dynamic_namespace_builtin_fails_before_transport(
    dynamic_name: str,
    target_kind: str,
) -> None:
    contacted: list[str] = []

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener
    expression = (
        'eval("retained_global")' if dynamic_name == "eval" else 'exec("sink = retained_global")'
    )
    module_code = compile(
        f"def callback():\n    return {expression}\n",
        "<handler-dynamic-namespace-callback>",
        "exec",
    )
    callback_code = next(item for item in module_code.co_consts if type(item) is types.CodeType)
    handler.callback = types.FunctionType(
        callback_code,
        {"retained_global": target},
        "callback",
    )
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("dynamic_name", ["eval", "exec"])
@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_copied_function_qualified_dynamic_builtin_fails_before_transport(
    dynamic_name: str,
    target_kind: str,
) -> None:
    contacted: list[str] = []

    class GlobalHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = GlobalHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener
    expression = (
        'builtins.eval("retained_global")'
        if dynamic_name == "eval"
        else 'builtins.exec("sink = retained_global")'
    )
    module_code = compile(
        f"def callback():\n    return {expression}\n",
        "<handler-qualified-dynamic-namespace-callback>",
        "exec",
    )
    callback_code = next(item for item in module_code.co_consts if type(item) is types.CodeType)
    handler.callback = types.FunctionType(
        callback_code,
        {"builtins": builtins, "retained_global": target},
        "callback",
    )
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


def test_custom_add_parent_does_not_execute_during_private_registration() -> None:
    add_parent_calls: list[object] = []
    contacted: list[str] = []

    class ParentHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = ParentHandler()
    opener = urllib.request.build_opener(handler)

    def custom_add_parent(self, parent):
        add_parent_calls.append(parent)
        self.parent = parent

    ParentHandler.add_parent = custom_add_parent
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert add_parent_calls == []
    assert contacted == []


def test_custom_add_parent_cannot_retain_original_opener() -> None:
    contacted: list[str] = []

    class StickyParentHandler(urllib.request.BaseHandler):
        def add_parent(self, parent):
            if not hasattr(self, "parent"):
                self.parent = parent

        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = StickyParentHandler()
    opener = urllib.request.build_opener(handler)
    request = urllib.request.Request("custom://source.invalid/data")
    with opener.open(request, timeout=1.0) as response:
        assert response.read() == b"ok"
    contacted.clear()
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(request, timeout=1.0)

    assert contacted == []
    assert handler.parent is opener


def test_reference_alias_budget_is_independent_of_dict_insertion_order(
    monkeypatch,
) -> None:
    target = object()
    shared = ["leaf"]
    first = {"a": shared, "b": shared, "c": shared}
    second = {"c": shared, "b": shared, "a": shared}
    monkeypatch.setattr(http_transport, "_TRAVERSAL_NODES_MAX", 6)

    results = [http_transport._references_target(graph, (target,)) for graph in (first, second)]

    assert results == [False, False]


def test_alias_rich_handler_snapshot_stays_within_global_budget(monkeypatch) -> None:
    class AliasHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            return _response(request.full_url)

    shared: object = []
    for _ in range(8):
        shared = [shared, shared]
    handler = AliasHandler()
    handler.config = shared
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)
    monkeypatch.setattr(http_transport, "_TRAVERSAL_NODES_MAX", 32)

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"


def test_handler_snapshot_global_budget_fails_before_transport(monkeypatch) -> None:
    contacted: list[str] = []

    class BroadHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = BroadHandler()
    handler.config = [[index] for index in range(64)]
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)
    monkeypatch.setattr(http_transport, "_TRAVERSAL_NODES_MAX", 32)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib opener cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


def test_handler_state_snapshot_limit_fails_before_transport() -> None:
    contacted: list[str] = []

    class BroadHolder:
        pass

    class BroadHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    holder = BroadHolder()
    for index in range(257):
        holder.__dict__[f"state_{index}"] = index
    handler = BroadHandler()
    handler.holder = holder
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


def test_unsnapshotable_installed_handler_fails_before_transport() -> None:
    contacted: list[str] = []

    class CyclicHandler(urllib.request.BaseHandler):
        def __init__(self):
            self.config = []
            self.config.append(self.config)

        def default_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = CyclicHandler()
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)
    original_parent = handler.parent

    with pytest.raises(
        urllib.error.URLError, match="installed urllib opener cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []
    assert urllib.request._opener is opener
    assert handler.parent is original_parent
    assert len(handler.config) == 1
    assert handler.config[0] is handler.config


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("nested", [False, True])
def test_passive_holder_state_retaining_target_fails_before_transport(
    target_kind: str,
    nested: bool,
) -> None:
    contacted: list[str] = []

    class Holder:
        pass

    class HolderHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = HolderHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener
    inner = Holder()
    inner.target = target
    holder = Holder()
    holder.state = {"nested": [inner]} if nested else target
    handler.holder = holder
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_passive_holder_bound_method_retaining_target_fails_before_transport(
    target_kind: str,
) -> None:
    contacted: list[str] = []
    hook_calls: list[str] = []

    class Holder:
        def callback(self, request):
            hook_calls.append("called")
            return _response(request.full_url)

    class HolderHandler(urllib.request.BaseHandler):
        pass

    handler = HolderHandler()
    opener = urllib.request.build_opener()
    holder = Holder()
    holder.target = handler if target_kind == "handler" else opener
    handler.custom_open = holder.callback
    opener.add_handler(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert hook_calls == []
    assert contacted == []


def test_cached_class_callback_dependency_mutation_fails_before_transport() -> None:
    contacted: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        retained = None

        def custom_open(self, request):
            retained = self.retained
            contacted.append(request.full_url)
            return _response(request.full_url, str(retained).encode())

    handler = CallbackHandler()
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/first"), timeout=1.0
    ) as response:
        assert response.read() == b"None"
    first_private = http_transport._INSTALLED_OPENER_CACHE.private
    type.__setattr__(CallbackHandler, "retained", opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/second"), timeout=1.0)

    assert contacted == ["custom://source.invalid/first"]
    assert http_transport._INSTALLED_OPENER_CACHE.private is first_private


def test_cached_private_opener_dependency_cannot_enable_redirect_transport() -> None:
    safe = "https://safe.invalid/data"
    source = "https://source.invalid/data"
    sink = "https://sink.invalid/steal"
    observed: list[tuple[str, str | None]] = []

    class CallbackHandler(urllib.request.BaseHandler):
        handler_order = 100
        retained = None

        def https_open(self, request):
            observed.append((request.full_url, request.get_header("Authorization")))
            if request.full_url == source:
                private = self.retained
                blocker = next(
                    handler
                    for handler in private.handlers
                    if isinstance(handler, http_transport._NoRedirectHandler)
                )
                private.handlers.remove(blocker)
                private.process_response["https"].remove(blocker)
                for status in http_transport._REDIRECT_STATUSES:
                    private.handle_error["http"][status].remove(blocker)
                private.add_handler(urllib.request.HTTPRedirectHandler())
                headers = email.message.Message()
                headers["Location"] = sink
                response = urllib.response.addinfourl(
                    io.BytesIO(b""), headers, request.full_url, code=302
                )
                response.msg = "redirect"
                return response
            return _response(request.full_url)

    opener = urllib.request.build_opener(CallbackHandler())
    urllib.request.install_opener(opener)
    with _urlopen_no_redirect(urllib.request.Request(safe), timeout=1.0) as response:
        assert response.read() == b"ok"
    type.__setattr__(
        CallbackHandler,
        "retained",
        http_transport._INSTALLED_OPENER_CACHE.private,
    )

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(
            urllib.request.Request(source, headers={"Authorization": "Bearer secret"}),
            timeout=1.0,
        )

    assert observed == [(safe, None)]
    assert all(url != sink for url, _authorization in observed)


@pytest.mark.parametrize("capture_kind", ["default", "kwdefault", "closure"])
def test_cached_private_opener_capture_cannot_enable_redirect_transport(
    capture_kind: str,
) -> None:
    safe = "https://safe.invalid/data"
    source = "https://source.invalid/data"
    sink = "https://sink.invalid/steal"
    observed: list[tuple[str, str | None]] = []

    def dispatch(request, private):
        observed.append((request.full_url, request.get_header("Authorization")))
        if request.full_url == source:
            blocker = next(
                handler
                for handler in private.handlers
                if isinstance(handler, http_transport._NoRedirectHandler)
            )
            private.handlers.remove(blocker)
            private.process_response["https"].remove(blocker)
            for status in http_transport._REDIRECT_STATUSES:
                private.handle_error["http"][status].remove(blocker)
            private.add_handler(urllib.request.HTTPRedirectHandler())
            headers = email.message.Message()
            headers["Location"] = sink
            response = urllib.response.addinfourl(
                io.BytesIO(b""), headers, request.full_url, code=302
            )
            response.msg = "redirect"
            return response
        return _response(request.full_url)

    if capture_kind == "default":

        def callback(self, request, retained=None):
            return dispatch(request, retained)

    elif capture_kind == "kwdefault":

        def callback(self, request, *, retained=None):
            return dispatch(request, retained)

    else:
        retained = None

        def callback(self, request):
            return dispatch(request, retained)

    class CallbackHandler(urllib.request.BaseHandler):
        handler_order = 100
        https_open = callback

    opener = urllib.request.build_opener(CallbackHandler())
    urllib.request.install_opener(opener)
    with _urlopen_no_redirect(urllib.request.Request(safe), timeout=1.0) as response:
        assert response.read() == b"ok"
    private = http_transport._INSTALLED_OPENER_CACHE.private
    if capture_kind == "default":
        callback.__defaults__ = (private,)
    elif capture_kind == "kwdefault":
        callback.__kwdefaults__ = {"retained": private}
    else:
        retained_index = callback.__code__.co_freevars.index("retained")
        callback.__closure__[retained_index].cell_contents = private

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(
            urllib.request.Request(source, headers={"Authorization": "Bearer secret"}),
            timeout=1.0,
        )

    assert observed == [(safe, None)]
    assert all(url != sink for url, _authorization in observed)


@pytest.mark.parametrize("capture_kind", ["default", "kwdefault", "closure"])
@pytest.mark.parametrize("indirection", ["container", "holder"])
def test_indirect_cached_private_opener_capture_cannot_enable_redirect_transport(
    capture_kind: str,
    indirection: str,
) -> None:
    safe = "https://safe.invalid/data"
    source = "https://source.invalid/data"
    sink = "https://sink.invalid/steal"
    observed: list[tuple[str, str | None]] = []

    class Holder:
        pass

    def dispatch(request, retained):
        observed.append((request.full_url, request.get_header("Authorization")))
        if request.full_url == source:
            private = retained["private"] if indirection == "container" else retained.private
            blocker = next(
                handler
                for handler in private.handlers
                if isinstance(handler, http_transport._NoRedirectHandler)
            )
            private.handlers.remove(blocker)
            private.process_response["https"].remove(blocker)
            for status in http_transport._REDIRECT_STATUSES:
                private.handle_error["http"][status].remove(blocker)
            private.add_handler(urllib.request.HTTPRedirectHandler())
            headers = email.message.Message()
            headers["Location"] = sink
            response = urllib.response.addinfourl(
                io.BytesIO(b""), headers, request.full_url, code=302
            )
            response.msg = "redirect"
            return response
        return _response(request.full_url)

    if capture_kind == "default":

        def callback(self, request, retained=None):
            return dispatch(request, retained)

    elif capture_kind == "kwdefault":

        def callback(self, request, *, retained=None):
            return dispatch(request, retained)

    else:
        retained = None

        def callback(self, request):
            return dispatch(request, retained)

    class CallbackHandler(urllib.request.BaseHandler):
        handler_order = 100
        https_open = callback

    opener = urllib.request.build_opener(CallbackHandler())
    urllib.request.install_opener(opener)
    with _urlopen_no_redirect(urllib.request.Request(safe), timeout=1.0) as response:
        assert response.read() == b"ok"
    private = http_transport._INSTALLED_OPENER_CACHE.private
    if indirection == "container":
        retained_value = {"private": private}
    else:
        retained_value = Holder()
        retained_value.private = private
    if capture_kind == "default":
        callback.__defaults__ = (retained_value,)
    elif capture_kind == "kwdefault":
        callback.__kwdefaults__ = {"retained": retained_value}
    else:
        retained_index = callback.__code__.co_freevars.index("retained")
        callback.__closure__[retained_index].cell_contents = retained_value

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(
            urllib.request.Request(source, headers={"Authorization": "Bearer secret"}),
            timeout=1.0,
        )

    assert observed == [(safe, None)]
    assert all(url != sink for url, _authorization in observed)


@pytest.mark.parametrize("change", ["add", "remove"])
def test_redirect_callback_eligibility_change_rebuilds_cached_topology(change: str) -> None:
    observed: list[str] = []

    class MutableHandler(urllib.request.BaseHandler):
        handler_order = 100

        def custom_open(self, request):
            observed.append("mutable")
            return _response(request.full_url)

    class TerminalHandler(urllib.request.BaseHandler):
        handler_order = 200

        def custom_open(self, request):
            observed.append("terminal")
            return _response(request.full_url)

    def redirect(self, request, fp, code, msg, headers):
        raise AssertionError("redirect callback executed")

    if change == "remove":
        type.__setattr__(MutableHandler, "http_error_302", redirect)
    opener = urllib.request.build_opener(MutableHandler(), TerminalHandler())
    urllib.request.install_opener(opener)
    with _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/first"), timeout=1.0):
        pass
    first_private = http_transport._INSTALLED_OPENER_CACHE.private

    if change == "add":
        type.__setattr__(MutableHandler, "http_error_302", redirect)
    else:
        type.__delattr__(MutableHandler, "http_error_302")
    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/second"), timeout=1.0
    ):
        pass

    expected = ["mutable", "terminal"] if change == "add" else ["terminal", "mutable"]
    assert observed == expected
    assert http_transport._INSTALLED_OPENER_CACHE.private is not first_private


@pytest.mark.parametrize("component", ["code", "defaults", "kwdefaults"])
def test_mutated_stdlib_callback_implementation_fails_before_transport(
    monkeypatch,
    component: str,
) -> None:
    contacted: list[str] = []
    callback = urllib.request.HTTPHandler.http_open

    if component == "code":
        monkeypatch.setitem(callback.__globals__, "retained_contacts", contacted)

        replacement_module = compile(
            "def replacement(self, request):\n    retained_contacts.append(request.full_url)\n",
            "<mutated-stdlib-callback>",
            "exec",
        )
        replacement_code = next(
            value for value in replacement_module.co_consts if type(value) is types.CodeType
        )
        monkeypatch.setattr(callback, "__code__", replacement_code)
    elif component == "defaults":
        monkeypatch.setattr(callback, "__defaults__", (contacted,))
    else:
        monkeypatch.setattr(callback, "__kwdefaults__", {"retained": contacted})
    opener = urllib.request.build_opener(urllib.request.HTTPHandler())
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("http://source.invalid/data"), timeout=1.0)

    assert contacted == []


def test_stdlib_http_callback_helper_override_fails_before_transport() -> None:
    contacted: list[str] = []

    class HelperOverrideHandler(urllib.request.HTTPHandler):
        def do_open(self, http_class, request, retained=None, **kwargs):
            contacted.append(request.full_url)
            return _response(request.full_url, str(retained).encode())

    handler = HelperOverrideHandler()
    opener = urllib.request.build_opener(handler)
    HelperOverrideHandler.do_open.__defaults__ = (opener,)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("http://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("spoof_kind", ["class", "function"])
def test_spoofed_stdlib_class_callback_provenance_fails_before_transport(
    spoof_kind: str,
) -> None:
    contacted: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        pass

    handler = CallbackHandler()
    opener = urllib.request.build_opener()

    def callback(self, request, retained=opener):
        contacted.append(request.full_url)
        return _response(request.full_url, str(retained).encode())

    if spoof_kind == "class":
        type.__setattr__(CallbackHandler, "__module__", urllib.request.__name__)
    else:
        callback.__module__ = urllib.request.__name__
    type.__setattr__(CallbackHandler, "custom_open", callback)
    opener.add_handler(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("capture_kind", ["default", "kwdefault", "closure"])
def test_unused_class_callback_capture_fails_before_transport(
    target_kind: str,
    capture_kind: str,
) -> None:
    contacted: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        pass

    handler = CallbackHandler()
    opener = urllib.request.build_opener()
    target = handler if target_kind == "handler" else opener
    if capture_kind == "default":

        def callback(self, request, retained=target):
            contacted.append(request.full_url)
            return _response(request.full_url)

    elif capture_kind == "kwdefault":

        def callback(self, request, *, retained=target):
            contacted.append(request.full_url)
            return _response(request.full_url)

    else:
        retained = target

        def callback(self, request):
            if False:
                return retained
            contacted.append(request.full_url)
            return _response(request.full_url)

    type.__setattr__(CallbackHandler, "custom_open", callback)
    opener.add_handler(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("capture_kind", ["global", "default", "closure", "namespace"])
def test_class_registered_callback_retaining_target_fails_before_transport(
    target_kind: str,
    capture_kind: str,
) -> None:
    contacted: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        pass

    handler = CallbackHandler()
    opener = urllib.request.build_opener()
    target = handler if target_kind == "handler" else opener
    if capture_kind == "global":
        code = compile(
            "def custom_open(self, request):\n    return retained_target\n",
            "<class-handler-global-callback>",
            "exec",
        )
        callback_code = next(item for item in code.co_consts if type(item) is types.CodeType)
        callback = types.FunctionType(
            callback_code,
            {"retained_target": target},
            "custom_open",
        )
    elif capture_kind == "default":

        def callback(self, request, retained_target=target):
            return retained_target

    elif capture_kind == "closure":
        retained_target = target

        def callback(self, request):
            return retained_target

    else:
        namespace = types.ModuleType("class_handler_namespace")
        types.ModuleType.__getattribute__(namespace, "__dict__")["target"] = target

        def callback(self, request, retained_namespace=namespace):
            return retained_namespace.target

    type.__setattr__(CallbackHandler, "custom_open", callback)
    opener.add_handler(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


def test_saved_install_alias_reset_rediscovers_https_proxy(monkeypatch) -> None:
    observed: list[tuple[str, str | None]] = []

    def fake_https_open(self, request):
        observed.append((request.host, request._tunnel_host))
        return _response(request.full_url)

    saved_install_opener = http_transport._STDLIB_INSTALL_OPENER
    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", fake_https_open)
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("HTTPS_PROXY", "http://first-proxy.invalid:8080")
    with _urlopen_no_redirect(urllib.request.Request("https://source.invalid/first"), timeout=1.0):
        pass

    saved_install_opener(None)
    monkeypatch.setenv("HTTPS_PROXY", "http://second-proxy.invalid:8080")
    with _urlopen_no_redirect(urllib.request.Request("https://source.invalid/second"), timeout=1.0):
        pass

    assert observed == [
        ("first-proxy.invalid:8080", "source.invalid"),
        ("second-proxy.invalid:8080", "source.invalid"),
    ]


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("state_kind", ["direct", "inherited", "mangled"])
def test_class_callback_bound_self_state_retaining_target_fails_before_transport(
    target_kind: str,
    state_kind: str,
) -> None:
    contacted: list[str] = []

    class CallbackBase(urllib.request.BaseHandler):
        def custom_open(self, request):
            retained = self.retry_state
            contacted.append(request.full_url)
            raise AssertionError(retained)

        def privateopen(self, request):
            retained = self.__retry_state
            contacted.append(request.full_url)
            raise AssertionError(retained)

    class CallbackHandler(CallbackBase):
        pass

    handler = CallbackHandler()
    opener = urllib.request.build_opener()
    target = handler if target_kind == "handler" else opener
    if state_kind == "direct":
        type.__setattr__(CallbackHandler, "retry_state", target)
    elif state_kind == "inherited":
        type.__setattr__(CallbackBase, "retry_state", target)
    else:
        type.__setattr__(CallbackBase, "_CallbackBase__retry_state", target)
        type.__setattr__(CallbackBase, "custom_open", CallbackBase.privateopen)
    opener.add_handler(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


def test_class_callback_direct_self_attribute_is_supported() -> None:
    class CallbackHandler(urllib.request.BaseHandler):
        token = b"safe"

        def custom_open(self, request):
            return _response(request.full_url, self.token)

    opener = urllib.request.build_opener(CallbackHandler())
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"safe"


def test_class_callback_instance_state_shadows_non_data_descriptor() -> None:
    descriptor_calls: list[str] = []

    class ShadowedDescriptor:
        def __get__(self, instance, owner=None):
            descriptor_calls.append("get")
            return b"unsafe"

    class CallbackHandler(urllib.request.BaseHandler):
        token = ShadowedDescriptor()

        def custom_open(self, request):
            return _response(request.full_url, self.token)

    handler = CallbackHandler()
    object.__getattribute__(handler, "__dict__")["token"] = b"safe"
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"safe"

    assert descriptor_calls == []


def test_class_callback_direct_self_parent_is_supported() -> None:
    observed: list[object] = []

    class CallbackHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            observed.append(self.parent)
            return _response(request.full_url)

    handler = CallbackHandler()
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    assert observed == [http_transport._INSTALLED_OPENER_CACHE.private]


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_class_callback_parent_data_descriptor_fails_without_execution(
    target_kind: str,
) -> None:
    descriptor_calls: list[str] = []
    callback_calls: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            callback_calls.append(request.full_url)
            return self.parent

    handler = CallbackHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener

    class ParentTrap:
        def __get__(self, instance, owner=None):
            descriptor_calls.append("get")
            return target

        def __set__(self, instance, value):
            descriptor_calls.append("set")

    type.__setattr__(CallbackHandler, "parent", ParentTrap())
    descriptor_calls.clear()
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert descriptor_calls == []
    assert callback_calls == []


@pytest.mark.parametrize("callback_kind", ["bare", "attribute"])
@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_class_callback_conditional_delete_bound_self_fails_before_transport(
    callback_kind: str,
    target_kind: str,
) -> None:
    contacted: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        if callback_kind == "bare":

            def custom_open(self, request):
                if request is None:
                    del self
                contacted.append(request.full_url)
                return self

        else:

            def custom_open(self, request):
                if request is None:
                    del self
                contacted.append(request.full_url)
                return self.retry_state

    handler = CallbackHandler()
    opener = urllib.request.build_opener()
    target = handler if target_kind == "handler" else opener
    type.__setattr__(CallbackHandler, "retry_state", target)
    opener.add_handler(handler)
    urllib.request.install_opener(opener)

    if sys.version_info >= (3, 12):
        assert "LOAD_FAST_CHECK" in {
            instruction.opname for instruction in dis.get_instructions(CallbackHandler.custom_open)
        }

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("nested", [False, True])
def test_class_callback_bound_self_alias_retaining_target_fails_before_transport(
    target_kind: str,
    nested: bool,
) -> None:
    contacted: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        if nested:

            def custom_open(self, request):
                alias = self

                def read_state():
                    return alias.retry_state

                contacted.append(request.full_url)
                return read_state()

        else:

            def custom_open(self, request):
                alias = self
                contacted.append(request.full_url)
                return alias.retry_state

    handler = CallbackHandler()
    opener = urllib.request.build_opener()
    target = handler if target_kind == "handler" else opener
    type.__setattr__(CallbackHandler, "retry_state", target)
    opener.add_handler(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_class_callback_bound_self_chained_alias_fails_before_transport(
    target_kind: str,
) -> None:
    contacted: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            first = second = self
            contacted.append(request.full_url)
            return (first, second)[0].retry_state

    handler = CallbackHandler()
    opener = urllib.request.build_opener()
    target = handler if target_kind == "handler" else opener
    type.__setattr__(CallbackHandler, "retry_state", target)
    opener.add_handler(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_class_callback_bound_self_nested_positional_default_fails_before_transport(
    target_kind: str,
) -> None:
    contacted: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            def read_state(bound=self):
                return bound.retry_state

            contacted.append(request.full_url)
            return read_state()

    handler = CallbackHandler()
    opener = urllib.request.build_opener()
    target = handler if target_kind == "handler" else opener
    type.__setattr__(CallbackHandler, "retry_state", target)
    opener.add_handler(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_class_callback_bound_self_alias_loop_backedge_fails_before_transport(
    target_kind: str,
) -> None:
    contacted: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            initial = object()
            alias = initial
            for alias in (initial, self):
                if alias is initial:
                    continue
            contacted.append(request.full_url)
            return alias.retry_state

    handler = CallbackHandler()
    opener = urllib.request.build_opener()
    target = handler if target_kind == "handler" else opener
    type.__setattr__(CallbackHandler, "retry_state", target)
    opener.add_handler(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("loop_position", ["inside", "after"])
def test_class_callback_bound_self_loop_alias_with_exception_fails_before_transport(
    target_kind: str,
    loop_position: str,
) -> None:
    contacted: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        if loop_position == "inside":

            def custom_open(self, request):
                try:
                    for alias in (object(), self):
                        marker = alias
                finally:
                    marker = None
                contacted.append(request.full_url)
                return (alias, marker)[0].retry_state

        else:

            def custom_open(self, request):
                try:
                    marker = None
                except Exception:
                    marker = object()
                for alias in (object(), self):
                    marker = alias
                contacted.append(request.full_url)
                return (alias, marker)[0].retry_state

    handler = CallbackHandler()
    opener = urllib.request.build_opener()
    target = handler if target_kind == "handler" else opener
    type.__setattr__(CallbackHandler, "retry_state", target)
    opener.add_handler(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_class_callback_bound_self_nested_kwdefault_fails_before_transport(
    target_kind: str,
) -> None:
    contacted: list[str] = []

    class CallbackHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            def read_state(*, bound=self):
                return bound.retry_state

            contacted.append(request.full_url)
            return read_state()

    handler = CallbackHandler()
    opener = urllib.request.build_opener()
    target = handler if target_kind == "handler" else opener
    type.__setattr__(CallbackHandler, "retry_state", target)
    opener.add_handler(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_class_callback_bound_self_alias_descriptor_fails_without_execution(
    target_kind: str,
) -> None:
    descriptor_calls: list[str] = []
    contacted: list[str] = []

    class TrapDescriptor:
        def __get__(self, instance, owner=None):
            descriptor_calls.append("get")
            return handler if target_kind == "handler" else opener

    class CallbackHandler(urllib.request.BaseHandler):
        retry_state = TrapDescriptor()

        def custom_open(self, request):
            alias = self
            contacted.append(request.full_url)
            return alias.retry_state

    handler = CallbackHandler()
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert descriptor_calls == []
    assert contacted == []


def test_class_callback_bound_self_descriptor_fails_without_execution() -> None:
    descriptor_calls: list[str] = []
    contacted: list[str] = []

    class TrapDescriptor:
        def __get__(self, instance, owner=None):
            descriptor_calls.append("get")
            return instance

    class CallbackHandler(urllib.request.BaseHandler):
        retry_state = TrapDescriptor()

        def custom_open(self, request):
            contacted.append(request.full_url)
            return self.retry_state

    handler = CallbackHandler()
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert descriptor_calls == []
    assert contacted == []


def test_non_none_install_releases_active_default_private_opener() -> None:
    urllib.request.install_opener(None)
    with _urlopen_no_redirect(
        urllib.request.Request("data:text/plain,default-private"), timeout=1.0
    ) as response:
        assert response.read() == b"default-private"
    default_private = http_transport._DEFAULT_NO_REDIRECT_OPENER
    default_private_ref = weakref.ref(default_private)

    replacement = urllib.request.build_opener()
    urllib.request.install_opener(replacement)
    del default_private
    gc.collect()

    assert default_private_ref() is None
    assert http_transport._DEFAULT_NO_REDIRECT_OPENER is None
    assert http_transport._INSTALLED_OPENER_CACHE is None


def test_default_opener_reset_rediscovers_environment_and_releases_identity(monkeypatch) -> None:
    observed: list[tuple[str, str | None]] = []

    def fake_https_open(self, request):
        observed.append((request.host, request._tunnel_host))
        return _response(request.full_url)

    monkeypatch.setattr(urllib.request.HTTPSHandler, "https_open", fake_https_open)
    urllib.request.install_opener(None)
    monkeypatch.setenv("HTTPS_PROXY", "http://first-proxy.invalid:8080")
    monkeypatch.setenv("NO_PROXY", "")
    with _urlopen_no_redirect(urllib.request.Request("https://source.invalid/first"), timeout=1.0):
        pass
    first = http_transport._DEFAULT_NO_REDIRECT_OPENER
    first_ref = weakref.ref(first)

    urllib.request.install_opener(None)
    monkeypatch.setenv("HTTPS_PROXY", "http://second-proxy.invalid:8080")
    del first
    gc.collect()
    with _urlopen_no_redirect(urllib.request.Request("https://source.invalid/second"), timeout=1.0):
        pass
    second = http_transport._DEFAULT_NO_REDIRECT_OPENER

    assert observed == [
        ("first-proxy.invalid:8080", "source.invalid"),
        ("second-proxy.invalid:8080", "source.invalid"),
    ]
    assert second is not first_ref()
    assert first_ref() is None
    assert http_transport._INSTALLED_OPENER_CACHE is None


@pytest.mark.parametrize("declaration_type", [set, frozenset])
def test_set_declared_slot_holder_safe_state_is_supported(declaration_type) -> None:
    class SlotHolder:
        __slots__ = declaration_type({"token"})

    class HolderHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            return _response(request.full_url, self.holder.token.encode())

    holder = SlotHolder()
    holder.token = "safe"
    handler = HolderHandler()
    handler.holder = holder
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"safe"


@pytest.mark.parametrize("declaration_type", [set, frozenset])
@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_set_declared_slot_holder_retaining_target_fails_before_transport(
    declaration_type,
    target_kind: str,
) -> None:
    contacted: list[str] = []

    class SlotHolder:
        __slots__ = declaration_type({"target"})

    class HolderHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = HolderHandler()
    opener = urllib.request.build_opener(handler)
    holder = SlotHolder()
    holder.target = handler if target_kind == "handler" else opener
    handler.holder = holder
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_all_underscore_owner_private_slot_retaining_target_fails_before_transport(
    target_kind: str,
) -> None:
    contacted: list[str] = []

    holder_type = type("_", (), {"__slots__": ("__target",)})

    class HolderHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = HolderHandler()
    opener = urllib.request.build_opener(handler)
    holder = holder_type()
    holder.__target = handler if target_kind == "handler" else opener
    handler.holder = holder
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


def test_all_underscore_owner_private_slot_safe_state_is_supported() -> None:
    holder_type = type("_", (), {"__slots__": ("__token",)})

    class HolderHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            return _response(request.full_url, b"safe")

    handler = HolderHandler()
    holder = holder_type()
    holder.__token = "safe"
    handler.holder = holder
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"safe"


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("nested", [False, True])
def test_slot_only_holder_state_retaining_target_fails_before_transport(
    target_kind: str,
    nested: bool,
) -> None:
    contacted: list[str] = []

    class SlotBase:
        __slots__ = ("__target",)

        def __init__(self, target):
            self.__target = target

    class SlotHolder(SlotBase):
        __slots__ = ()

    class NestedHolder:
        __slots__ = ("state",)

    class HolderHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = HolderHandler()
    opener = urllib.request.build_opener(handler)
    target = handler if target_kind == "handler" else opener
    retained = SlotHolder(target)
    if nested:
        holder = NestedHolder()
        holder.state = {"nested": [retained]}
    else:
        holder = retained
    handler.holder = holder
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
def test_slot_only_holder_bound_method_retaining_target_fails_before_transport(
    target_kind: str,
) -> None:
    hook_calls: list[str] = []

    class SlotHolder:
        __slots__ = ("target",)

        def callback(self, request):
            hook_calls.append(request.full_url)
            return _response(request.full_url)

    class HolderHandler(urllib.request.BaseHandler):
        pass

    handler = HolderHandler()
    opener = urllib.request.build_opener()
    holder = SlotHolder()
    holder.target = handler if target_kind == "handler" else opener
    handler.custom_open = holder.callback
    opener.add_handler(handler)
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert hook_calls == []


@pytest.mark.parametrize("target_kind", ["handler", "opener"])
@pytest.mark.parametrize("capture_kind", ["global", "default", "closure"])
def test_function_slot_only_holder_retaining_target_fails_before_transport(
    target_kind: str,
    capture_kind: str,
) -> None:
    contacted: list[str] = []

    class SlotHolder:
        __slots__ = ("target",)

    class HolderHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = HolderHandler()
    opener = urllib.request.build_opener(handler)
    holder = SlotHolder()
    holder.target = handler if target_kind == "handler" else opener
    if capture_kind == "global":
        module_code = compile(
            "def callback():\n    return retained_holder\n",
            "<handler-global-slot-holder-callback>",
            "exec",
        )
        callback_code = next(item for item in module_code.co_consts if type(item) is types.CodeType)
        callback = types.FunctionType(
            callback_code,
            {"retained_holder": holder},
            "callback",
        )
    elif capture_kind == "default":

        def callback(retained_holder=holder):
            return retained_holder

    else:
        retained_holder = holder

        def callback():
            return retained_holder

    handler.callback = callback
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert contacted == []


@pytest.mark.parametrize("hook_name", ["__getattribute__", "__getattr__"])
def test_slot_only_holder_lookup_hooks_fail_closed_without_execution(hook_name: str) -> None:
    contacted: list[str] = []
    hook_calls: list[str] = []

    def lookup_hook(self, name):
        hook_calls.append(name)
        if hook_name == "__getattribute__":
            return object.__getattribute__(self, name)
        raise AttributeError(name)

    class SlotHolder:
        __slots__ = ("target",)

    type.__setattr__(SlotHolder, hook_name, lookup_hook)

    class HolderHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = HolderHandler()
    opener = urllib.request.build_opener(handler)
    holder = SlotHolder()
    holder.target = handler
    handler.holder = holder
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert hook_calls == []
    assert contacted == []


def test_shadowed_slot_descriptor_fails_closed_without_execution() -> None:
    contacted: list[str] = []
    descriptor_calls: list[str] = []

    class TrapDescriptor:
        def __get__(self, instance, owner=None):
            descriptor_calls.append("get")
            return instance

        def __set__(self, instance, value):
            descriptor_calls.append("set")

    class SlotHolder:
        __slots__ = ("target",)

    class HolderHandler(urllib.request.BaseHandler):
        def custom_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = HolderHandler()
    opener = urllib.request.build_opener(handler)
    holder = SlotHolder()
    holder.target = handler
    type.__setattr__(SlotHolder, "target", TrapDescriptor())
    handler.holder = holder
    urllib.request.install_opener(opener)

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ):
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert descriptor_calls == []
    assert contacted == []


def test_uncopyable_installed_handler_fails_before_transport() -> None:
    contacted: list[str] = []
    copy_calls: list[str] = []

    class UncopyableHandler(urllib.request.BaseHandler):
        def __copy__(self):
            copy_calls.append("called")
            raise RuntimeError("private copy detail")

        def default_open(self, request):
            contacted.append(request.full_url)
            return _response(request.full_url)

    handler = UncopyableHandler()
    opener = urllib.request.build_opener(handler)
    urllib.request.install_opener(opener)
    original_parent = handler.parent
    original_state = handler.__dict__.copy()

    with pytest.raises(
        urllib.error.URLError, match="installed urllib handler cannot be copied safely"
    ) as exc_info:
        _urlopen_no_redirect(urllib.request.Request("custom://source.invalid/data"), timeout=1.0)

    assert "private copy detail" not in str(exc_info.value)
    assert copy_calls == []
    assert contacted == []
    assert urllib.request._opener is opener
    assert handler.parent is original_parent
    assert handler.__dict__ == original_state
