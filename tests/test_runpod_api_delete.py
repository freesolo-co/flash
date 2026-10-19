"""delete_endpoint_for_key: a 404 under the owning account is a clean teardown.

CPU-only; urllib mocked. Drives a real HTTPError through request_with_retries so the
__cause__ chaining used by key-addressed deletion is genuinely exercised.
"""

from __future__ import annotations

import io
import urllib.error

import pytest


def _http_error(code: int, body: bytes = b""):
    return urllib.error.HTTPError(
        url="https://rest.runpod.io/v1/endpoints/x",
        code=code,
        msg="err",
        hdrs=None,
        fp=io.BytesIO(body),
    )


def _patch_urlopen(monkeypatch, error):
    from flash.providers._lifecycle.net import http as _http

    def fake_urlopen(req, timeout=None):
        raise error

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(_http.time, "sleep", lambda s: None)


def test_delete_endpoint_404_reports_success(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api

    _patch_urlopen(monkeypatch, _http_error(404, b'{"error": "endpoint does not exist"}'))
    # already gone == desired end state reached == clean teardown.
    assert runpod_api.delete_endpoint_for_key("ep-gone", "rk-test") is True


def test_delete_endpoint_empty_success_response_reports_success(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api

    monkeypatch.setattr(
        runpod_api._CLIENT,
        "request_with_retries_for_key",
        lambda *_args, **_kwargs: None,
    )

    assert runpod_api.delete_endpoint_for_key("ep-deleted", "rk-test") is True


def test_delete_endpoint_other_error_reports_failure(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api

    _patch_urlopen(monkeypatch, _http_error(403, b'{"error": "forbidden"}'))
    assert runpod_api.delete_endpoint_for_key("ep-denied", "rk-test") is False


def test_delete_endpoint_5xx_exhausted_reports_failure(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api

    # 5xx retries then raises "failed after n attempts" (no __cause__ httperror, no 404 text).
    _patch_urlopen(monkeypatch, _http_error(500, b"boom"))
    assert runpod_api.delete_endpoint_for_key("ep-flaky", "rk-test") is False


def test_delete_endpoint_403_with_not_found_text_reports_failure(monkeypatch):
    """A 403 whose body says "does not exist" is still a 403 — must NOT be swallowed."""
    from flash.providers.runpod.client import api as runpod_api

    _patch_urlopen(monkeypatch, _http_error(403, b'{"error": "endpoint does not exist"}'))
    assert runpod_api.delete_endpoint_for_key("ep-denied", "rk-test") is False


def test_exact_endpoint_lookup_accepts_only_explicit_owner_404(monkeypatch):
    from flash.providers.runpod.client import api as runpod_api

    key = "rk-owner-secret"
    monkeypatch.setattr(runpod_api._keys, "keys", lambda: [key])
    _patch_urlopen(
        monkeypatch,
        _http_error(404, b'{"error": "provider response body must stay private"}'),
    )

    assert (
        runpod_api.endpoint_absent_for_fingerprint("ep-gone", runpod_api.key_fingerprint(key))
        is True
    )


@pytest.mark.parametrize(
    "response",
    [
        {"id": "ep-present", "secret": "provider response body"},
        None,
        "malformed provider response body",
    ],
    ids=["endpoint-present", "empty-response", "malformed-response"],
)
def test_exact_endpoint_lookup_rejects_every_success_response(monkeypatch, response):
    from flash.providers.runpod.client import api as runpod_api

    key = "rk-owner-secret"
    calls = []
    monkeypatch.setattr(runpod_api._keys, "keys", lambda: [key])
    monkeypatch.setattr(
        runpod_api._CLIENT,
        "request_with_retries_for_key",
        lambda *args, **kwargs: calls.append((args, kwargs)) or response,
    )

    with pytest.raises(runpod_api.RunpodApiError, match="cleanup unconfirmed") as exc_info:
        runpod_api.endpoint_absent_for_fingerprint("ep-present", runpod_api.key_fingerprint(key))

    assert calls == [
        (
            (key, f"{runpod_api.REST_BASE}/endpoints/ep-present"),
            {"retries": 2},
        )
    ]
    error = str(exc_info.value)
    assert key not in error
    assert "provider response body" not in error


@pytest.mark.parametrize(
    "error",
    [
        _http_error(403, b'{"error": "provider response body"}'),
        OSError("network error with provider response body"),
    ],
    ids=["non-404", "network-error"],
)
def test_exact_endpoint_lookup_rejects_errors_without_leaking_details(monkeypatch, error):
    from flash.providers.runpod.client import api as runpod_api

    key = "rk-owner-secret"
    monkeypatch.setattr(runpod_api._keys, "keys", lambda: [key])
    _patch_urlopen(monkeypatch, error)

    with pytest.raises(runpod_api.RunpodApiError, match="cleanup unconfirmed") as exc_info:
        runpod_api.endpoint_absent_for_fingerprint(
            "ep-unconfirmed", runpod_api.key_fingerprint(key)
        )

    message = str(exc_info.value)
    assert key not in message
    assert "provider response body" not in message


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
