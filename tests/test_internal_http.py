"""redirect containment for authenticated stdlib requests."""

from __future__ import annotations

import email.message
import io
import urllib.error
import urllib.request
import urllib.response

import pytest

from flash._internal.http import _build_no_redirect_opener, _urlopen_no_redirect


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
