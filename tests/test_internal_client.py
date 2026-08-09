"""internal key whitespace normalization for shared internal-backend requests."""

from __future__ import annotations

from flash.server.platform import internal_client as ic
from flash.server.platform.auth import INTERNAL_KEY_ENV


def test_internal_key_strips_whitespace(monkeypatch):
    monkeypatch.setenv(INTERNAL_KEY_ENV, "  sk-abc \n")
    assert ic.internal_key() == "sk-abc"


def test_blank_internal_key_is_none(monkeypatch):
    for blank in ("", "   ", "\n", "\t "):
        monkeypatch.setenv(INTERNAL_KEY_ENV, blank)
        assert ic.internal_key() is None  # whitespace-only -> normalized to None


def test_unset_internal_key_is_none(monkeypatch):
    monkeypatch.delenv(INTERNAL_KEY_ENV, raising=False)
    assert ic.internal_key() is None
