"""Explicit operator credentials for Modal training."""

from flash.providers._lifecycle.net.auth import load_provider_key


def credentials() -> tuple[str, str] | None:
    token_id = load_provider_key("MODAL_TOKEN_ID")
    secret = load_provider_key("MODAL_TOKEN_SECRET")
    return (token_id, secret) if token_id and secret else None


def scope_fingerprint() -> str:
    import hashlib

    from flash.providers.modal.api import environment

    auth = credentials()
    if auth is None:
        raise RuntimeError("Modal credentials are missing")
    return hashlib.sha256(f"{auth[0]}:{environment()}".encode()).hexdigest()


def require_recorded_scope(status) -> None:
    """A lost launch must be reconciled in the same workspace and environment."""
    spot_scope = getattr(status, "submitted_spot_scope", None)
    if spot_scope is not None:
        from flash.providers.nebius.config import scope_fingerprint as spot_scope_fingerprint

        if spot_scope != spot_scope_fingerprint():
            raise RuntimeError("Nebius project or region differs from this run's owner")
    recorded = getattr(status, "submitted_modal_scope", None)
    if recorded is not None and recorded != scope_fingerprint():
        raise RuntimeError("Modal credentials or environment differ from this run's owner")
