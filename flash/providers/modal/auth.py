"""Modal training credential handling on the control-plane host."""

from __future__ import annotations

from flash.providers._lifecycle.auth import load_provider_key

_TOKEN_ID_ENV = "MODAL_TOKEN_ID"
_TOKEN_SECRET_ENV = "MODAL_TOKEN_SECRET"


def load_credentials() -> tuple[str, str] | None:
    """Return the Modal token pair only when both environment values are present."""
    token_id = load_provider_key(_TOKEN_ID_ENV)
    token_secret = load_provider_key(_TOKEN_SECRET_ENV)
    if not token_id or not token_secret:
        return None
    return token_id, token_secret
