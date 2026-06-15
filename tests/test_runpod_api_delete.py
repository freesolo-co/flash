"""delete_endpoint: a 404 (endpoint already gone) is a clean teardown, not a failure.

CPU-only; urllib mocked. Drives a real HTTPError through request_with_retries so the
__cause__ chaining delete_endpoint relies on is genuinely exercised.
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
    from autoslm.providers import _http

    def fake_urlopen(req, timeout=None):
        raise error

    monkeypatch.setattr(_http.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(_http.time, "sleep", lambda s: None)


def test_delete_endpoint_404_reports_success(monkeypatch):
    from autoslm.providers.runpod import api as runpod_api

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-test")
    _patch_urlopen(monkeypatch, _http_error(404, b'{"error": "endpoint does not exist"}'))
    # Already gone == desired end state reached == clean teardown.
    assert runpod_api.delete_endpoint("ep-gone") is True


def test_delete_endpoint_other_error_reports_failure(monkeypatch):
    from autoslm.providers.runpod import api as runpod_api

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-test")
    _patch_urlopen(monkeypatch, _http_error(403, b'{"error": "forbidden"}'))
    assert runpod_api.delete_endpoint("ep-denied") is False


def test_delete_endpoint_5xx_exhausted_reports_failure(monkeypatch):
    from autoslm.providers.runpod import api as runpod_api

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-test")
    # 5xx retries then raises "failed after N attempts" (no __cause__ HTTPError, no 404 text).
    _patch_urlopen(monkeypatch, _http_error(500, b"boom"))
    assert runpod_api.delete_endpoint("ep-flaky") is False


def test_delete_endpoint_403_with_not_found_text_reports_failure(monkeypatch):
    """A 403 whose body says "does not exist" is still a 403 — must NOT be swallowed."""
    from autoslm.providers.runpod import api as runpod_api

    monkeypatch.setenv("RUNPOD_API_KEY", "rk-test")
    _patch_urlopen(monkeypatch, _http_error(403, b'{"error": "endpoint does not exist"}'))
    assert runpod_api.delete_endpoint("ep-denied") is False


def test_is_not_found_uses_cause_code_not_message():
    from autoslm.providers.runpod import api as runpod_api

    # Genuine 404 cause -> True even with a terse message.
    err_404 = runpod_api.RunpodApiError("DELETE x -> HTTP 404: Not Found")
    err_404.__cause__ = _http_error(404)
    assert runpod_api._is_not_found(err_404)

    # 403 cause whose message mentions "does not exist" -> still False (cause is authoritative).
    err_403 = runpod_api.RunpodApiError("DELETE x -> HTTP 403: endpoint does not exist")
    err_403.__cause__ = _http_error(403)
    assert not runpod_api._is_not_found(err_403)

    # No HTTPError cause: fall back to an unambiguous 404 text match only.
    assert runpod_api._is_not_found(runpod_api.RunpodApiError("DELETE x -> HTTP 404: Not Found"))
    assert not runpod_api._is_not_found(runpod_api.RunpodApiError("endpoint does not exist"))
    assert not runpod_api._is_not_found(runpod_api.RunpodApiError("HTTP 500: boom"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
