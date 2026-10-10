"""Supabase REST coverage pins missing credentials and bounded HTTP error diagnostics.

The tests construct local settings and responses so no Supabase request is made.
"""

from __future__ import annotations

import httpx
import pytest

from flash.serving.src.store.settings import Settings
from flash.serving.src.store.supabase_rest import raise_for_supabase, supabase_headers


def test_supabase_headers_requires_service_role_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    settings = Settings(_env_file=None)

    with pytest.raises(RuntimeError, match="Missing SUPABASE_SERVICE_ROLE_KEY"):
        supabase_headers(settings, "flash")


def test_supabase_headers_rejects_legacy_service_role_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "eyJhbGciOiJIUzI1NiJ9.test.signature")

    with pytest.raises(RuntimeError, match="must use the sb_secret_ format"):
        supabase_headers(Settings(_env_file=None), "flash")


def test_supabase_headers_omit_bearer_for_opaque_secret_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service_role_key = "sb_secret_test"
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", service_role_key)

    headers = supabase_headers(Settings(_env_file=None), "flash")

    assert headers == {
        "apikey": service_role_key,
        "Content-Type": "application/json",
        "Accept-Profile": "flash",
        "Content-Profile": "flash",
    }


def test_raise_for_supabase_includes_action_status_and_bounded_body() -> None:
    body = "x" * 1000 + "excluded-sentinel"
    response = httpx.Response(
        503,
        content=body,
        request=httpx.Request("POST", "https://example.supabase.co/rest/v1/adapters"),
    )

    with pytest.raises(RuntimeError) as caught:
        raise_for_supabase(response, "insert adapter")

    message = str(caught.value)
    assert message.startswith("Failed to insert adapter: 503 ")
    assert "x" * 1000 in message
    assert "excluded-sentinel" not in message
