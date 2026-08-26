"""redirect containment for authenticated stdlib requests."""

from __future__ import annotations

import email.message
import io
import ssl
import urllib.error
import urllib.request
import urllib.response

import pytest

from flash._internal.http import _build_no_redirect_opener, _urlopen_no_redirect


@pytest.fixture(autouse=True)
def _restore_global_opener():
    opener = urllib.request._opener
    try:
        yield
    finally:
        urllib.request._opener = opener


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
    urllib.request.install_opener(second)

    with _urlopen_no_redirect(
        urllib.request.Request("custom://source.invalid/data"), timeout=1.0
    ) as response:
        assert response.read() == b"ok"

    assert observed == ["second"]
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
