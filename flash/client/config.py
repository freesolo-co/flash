"""Client-side credential storage: the Flash API key + control-plane URL.

Stored in ``~/.flash/config.json`` (dir 0700, file 0600 — it holds a secret).
Environment variables take precedence so CI/agents can inject credentials without
touching the file: ``FREESOLO_API_KEY`` for the key, ``FLASH_API_URL`` for the URL.
"""

from __future__ import annotations

import os
from pathlib import Path

from .._channel import CHANNEL
from .._fileio import read_json_or_empty, secure_json_write

# The default control plane follows the installed channel (see flash/_channel.py): the prod
# package (freesolo-flash / `flash`) targets the production plane, the dev-channel package
# (freesolo-flash-dev / `flash-dev`) targets the staging plane. Either is overridable via
# FLASH_API_URL or `flash login --api-url`.
PROD_API_URL = "https://flash.freesolo.co"
DEV_API_URL = "https://flash-dev.freesolo.co"


def default_api_url(channel: str = CHANNEL) -> str:
    """Default control-plane URL for the given release channel."""
    return DEV_API_URL if channel == "dev" else PROD_API_URL


DEFAULT_API_URL = default_api_url()

CONFIG_DIR = Path.home() / ".flash"
CONFIG_PATH = CONFIG_DIR / "config.json"


def _read_config() -> dict:
    return read_json_or_empty(CONFIG_PATH)


def load_credentials_with_source() -> tuple[str, str | None, str | None]:
    """Resolve (api_url, api_key, key_source); key/source are None when logged out."""
    cfg = _read_config()
    api_url = os.environ.get("FLASH_API_URL") or cfg.get("api_url") or DEFAULT_API_URL
    env_key = os.environ.get("FREESOLO_API_KEY")
    if env_key:
        return api_url.rstrip("/"), env_key, "FREESOLO_API_KEY"
    if cfg.get("api_key"):
        return api_url.rstrip("/"), cfg["api_key"], str(CONFIG_PATH)
    return api_url.rstrip("/"), None, None


def load_credentials() -> tuple[str, str | None]:
    """Resolve (api_url, api_key); the key is None when the user hasn't logged in."""
    api_url, api_key, _source = load_credentials_with_source()
    return api_url, api_key


def save_credentials(api_key: str, api_url: str | None = None) -> Path:
    """Persist the key (and optionally a non-default URL) with private permissions."""
    cfg = _read_config()
    cfg["api_key"] = api_key
    if api_url:
        # Record the plane actually authenticated against. When it's the default, drop any
        # stored url instead of pinning it — this also clears a stale custom url from a
        # previous custom FLASH_API_URL login so later commands don't keep hitting the old host.
        if api_url.rstrip("/") == DEFAULT_API_URL.rstrip("/"):
            cfg.pop("api_url", None)
        else:
            cfg["api_url"] = api_url.rstrip("/")
    secure_json_write(CONFIG_PATH, cfg)
    return CONFIG_PATH
