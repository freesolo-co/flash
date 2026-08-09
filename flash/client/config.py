"""Client-side credential storage: Flash API key + control-plane URL.

Stored in ``config.json`` (0600) under the Flash data dir, ``~/.flash`` unless
``FLASH_DATA_DIR`` says otherwise. ``FREESOLO_API_KEY`` / ``FLASH_API_URL`` override.
"""

from __future__ import annotations

import os
from pathlib import Path

from flash._internal.channel import CHANNEL, CLI_NAME
from flash._internal.fileio import read_json_or_empty, secure_json_write
from flash._internal.paths import data_dir

PROD_API_URL = "https://flash.freesolo.co"
DEV_API_URL = "https://flash-dev.freesolo.co"


def default_api_url(channel: str = CHANNEL) -> str:
    """Default control-plane URL for the given release channel."""
    return DEV_API_URL if channel == "dev" else PROD_API_URL


DEFAULT_API_URL = default_api_url()

# Resolved at import so the CLI reports one stable path for the life of a command: these appear
# in user-facing messages ("the login saved in ..."), and a value that changed between the read
# and the message would name a file the user was not told about.
CONFIG_DIR = data_dir()
CONFIG_PATH = CONFIG_DIR / "config.json"


def _read_config() -> dict:
    return read_json_or_empty(CONFIG_PATH)


def load_credentials_with_source() -> tuple[str, str | None, str | None]:
    """Resolve (api_url, api_key, key_source); key/source are None when logged out."""
    cfg = _read_config()
    api_url = (os.environ.get("FLASH_API_URL") or cfg.get("api_url") or DEFAULT_API_URL).rstrip("/")
    env_key = os.environ.get("FREESOLO_API_KEY")
    if env_key:
        return api_url, env_key, "FREESOLO_API_KEY"
    if cfg.get("api_key"):
        return api_url, cfg["api_key"], str(CONFIG_PATH)
    return api_url, None, None


def load_credentials() -> tuple[str, str | None]:
    """Resolve (api_url, api_key); the key is None when the user hasn't logged in."""
    api_url, api_key, _source = load_credentials_with_source()
    return api_url, api_key


def shadowed_login_warning() -> str | None:
    """Warn when ``FREESOLO_API_KEY`` shadows a *different* saved login, else None.

    ``~/.flash/config.json`` is shared mutable state and the env var silently wins over it, so an
    inherited or exported key can point a command at a different organization than the one the user
    logged into. Both keys are valid, so nothing fails to authenticate: projects, runs, and
    deployments simply land in the other org. Only flag a genuine mismatch: an env var equal to the
    saved key is the common `source .env` case and warning on it would train users to ignore this.
    """
    env_key = os.environ.get("FREESOLO_API_KEY")
    if not env_key:
        return None
    saved = _read_config().get("api_key")
    if not saved or saved == env_key:
        return None
    return (
        f"FREESOLO_API_KEY is set and overrides the login saved in {CONFIG_PATH}; "
        "this command runs against the env var's organization. "
        f"Run `{CLI_NAME} whoami` to confirm the org, "
        "or unset FREESOLO_API_KEY to use the saved login."
    )


def save_credentials(api_key: str, api_url: str | None = None) -> Path:
    """Persist the key (and optionally a non-default URL) with private permissions."""
    cfg = _read_config()
    cfg["api_key"] = api_key
    if api_url:
        # Drop default URL rather than pin it — clears any stale custom URL from a previous login.
        if api_url.rstrip("/") == DEFAULT_API_URL.rstrip("/"):
            cfg.pop("api_url", None)
        else:
            cfg["api_url"] = api_url.rstrip("/")
    secure_json_write(CONFIG_PATH, cfg)
    return CONFIG_PATH
