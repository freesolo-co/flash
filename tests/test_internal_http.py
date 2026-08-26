"""redirect containment for authenticated stdlib requests."""

from __future__ import annotations

import email.message
import io
import os
import ssl
import threading
import urllib.error
import urllib.request
import urllib.response
from concurrent.futures import ThreadPoolExecutor

import pytest

import flash._internal.http as http_transport
from flash._internal.http import _build_no_redirect_opener, _urlopen_no_redirect


@pytest.fixture(autouse=True)
def _restore_http_globals():
    opener = urllib.request._opener
    default = http_transport._DEFAULT_NO_REDIRECT_OPENER
    cached = http_transport._INSTALLED_OPENER_CACHE
    proxy_env = {key: os.environ.get(key) for key in ("HTTPS_PROXY", "https_proxy", "NO_PROXY")}
    http_transport._DEFAULT_NO_REDIRECT_OPENER = None
    http_transport._INSTALLED_OPENER_CACHE = None
    try:
        yield
    finally:
        urllib.request._opener = opener
        http_transport._DEFAULT_NO_REDIRECT_OPENER = default
        http_transport._INSTALLED_OPENER_CACHE = cached
        for key, value in proxy_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
            observed.append((self, self._context, self.debuglevel))
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
    monkeypatch.setenv("HTTPS_PROXY", "http://late-proxy.invalid:8080")
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.setenv("NO_PROXY", "")
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


def test_installed_digest_handler_state_survives_across_requests() -> None:
    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    password_manager.add_password("realm", "https://source.invalid", "user", "pass")
    digest = urllib.request.HTTPDigestAuthHandler(password_manager)
    seen_nonce_counts: list[int] = []

    class DigestSource(urllib.request.BaseHandler):
        handler_order = 100

        def https_open(self, request):
            authorization = request.get_header("Authorization")
            if authorization:
                seen_nonce_counts.append(digest_copy(self).nonce_count)
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

    def digest_copy(handler):
        return next(
            item
            for item in handler.parent.handlers
            if isinstance(item, urllib.request.HTTPDigestAuthHandler)
        )

    opener = urllib.request.build_opener(DigestSource(), digest)
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
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
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
            if response.code in (301, 302, 303, 307, 308):
                return self.parent.open(response.headers["Location"], timeout=request.timeout)
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


def test_uncopyable_installed_handler_fails_before_transport() -> None:
    contacted: list[str] = []

    class UncopyableHandler(urllib.request.BaseHandler):
        def __copy__(self):
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
    assert contacted == []
    assert urllib.request._opener is opener
    assert handler.parent is original_parent
    assert handler.__dict__ == original_state
