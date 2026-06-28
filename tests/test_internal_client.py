"""flash.server._internal_client key gate: whitespace-strip/normalize + shared enabled() definition.

A stray trailing newline or a whitespace-only INTERNAL key must not masquerade as "enabled" (which
would emit an invalid ``Authorization: Bearer <whitespace>`` header), and ``enabled()`` must share ONE
definition with ``internal_key()`` so the two can't disagree on what counts as a usable key.
"""

from __future__ import annotations

from flash.server import _internal_client as ic
from flash.server.auth import INTERNAL_KEY_ENV


def test_internal_key_strips_whitespace(monkeypatch):
    monkeypatch.setenv(INTERNAL_KEY_ENV, "  sk-abc \n")
    assert ic.internal_key() == "sk-abc"
    assert ic.enabled() is True


def test_blank_internal_key_is_none_and_disabled(monkeypatch):
    for blank in ("", "   ", "\n", "\t "):
        monkeypatch.setenv(INTERNAL_KEY_ENV, blank)
        assert ic.internal_key() is None  # whitespace-only -> normalized to None
        assert ic.enabled() is False  # ... and therefore NOT enabled (shared definition)


def test_unset_internal_key_is_none_and_disabled(monkeypatch):
    monkeypatch.delenv(INTERNAL_KEY_ENV, raising=False)
    assert ic.internal_key() is None
    assert ic.enabled() is False
