"""Client-side credential storage: the AutoSLM API key + control-plane URL.

Stored in ``~/.autoslm/config.json`` (dir 0700, file 0600 — it holds a secret).
Environment variables take precedence so CI/agents can inject credentials without
touching the file: ``FREESOLO_API_KEY`` for the key, ``AUTOSLM_API_URL`` for the URL.
"""

from __future__ import annotations

import os
from pathlib import Path

from .._fileio import read_json_or_empty, secure_json_write

DEFAULT_API_URL = "https://flash.freesolo.co"

CONFIG_DIR = Path.home() / ".autoslm"
CONFIG_PATH = CONFIG_DIR / "config.json"


def _read_config() -> dict:
    return read_json_or_empty(CONFIG_PATH)


def load_credentials() -> tuple[str, str | None]:
    """Resolve (api_url, api_key); the key is None when the user hasn't logged in."""
    cfg = _read_config()
    api_url = os.environ.get("AUTOSLM_API_URL") or cfg.get("api_url") or DEFAULT_API_URL
    api_key = os.environ.get("FREESOLO_API_KEY") or cfg.get("api_key")
    return api_url.rstrip("/"), api_key


def save_credentials(api_key: str, api_url: str | None = None) -> Path:
    """Persist the key (and optionally a non-default URL) with private permissions."""
    cfg = _read_config()
    cfg["api_key"] = api_key
    if api_url:
        # Record the plane actually authenticated against. When it's the default, drop any
        # stored url instead of pinning it — this also clears a stale custom url from a
        # previous custom AUTOSLM_API_URL login so later commands don't keep hitting the old host.
        if api_url.rstrip("/") == DEFAULT_API_URL.rstrip("/"):
            cfg.pop("api_url", None)
        else:
            cfg["api_url"] = api_url.rstrip("/")
    secure_json_write(CONFIG_PATH, cfg)
    return CONFIG_PATH
