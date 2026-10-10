from __future__ import annotations

import pytest

from tests import capture_qwen38_27b_metadata as capture


class _Response:
    def __init__(self, status: int, content_range: str, body: bytes) -> None:
        self.status = status
        self.headers = {"Content-Range": content_range}
        self.body = body
        self.read_sizes: list[int | None] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def getcode(self) -> int:
        return self.status

    def read(self, size: int | None = None) -> bytes:
        self.read_sizes.append(size)
        return self.body if size is None else self.body[:size]


def _install(monkeypatch, response: _Response) -> None:
    monkeypatch.setattr(capture.urllib.request, "urlopen", lambda *_args, **_kwargs: response)


def test_ranged_read_rejects_http_200(monkeypatch):
    response = _Response(200, "", b"abc")
    _install(monkeypatch, response)

    with pytest.raises(ValueError, match="returned HTTP 200, expected 206"):
        capture._read("https://example.invalid/model", end=2)

    assert response.read_sizes == []


def test_ranged_read_rejects_malformed_content_range(monkeypatch):
    response = _Response(206, "bytes 1-3/10", b"abc")
    _install(monkeypatch, response)

    with pytest.raises(ValueError, match="invalid Content-Range"):
        capture._read("https://example.invalid/model", end=2)

    assert response.read_sizes == []


def test_ranged_read_rejects_excess_body_after_one_sentinel_byte(monkeypatch):
    response = _Response(206, "bytes 0-2/10", b"abcd-and-more")
    _install(monkeypatch, response)

    with pytest.raises(ValueError, match="returned more than 3 bytes"):
        capture._read("https://example.invalid/model", end=2)

    assert response.read_sizes == [4]


def test_ranged_read_accepts_exact_http_206_response(monkeypatch):
    response = _Response(206, "bytes 0-2/10", b"abc")
    _install(monkeypatch, response)

    assert capture._read("https://example.invalid/model", end=2) == b"abc"
    assert response.read_sizes == [4]
