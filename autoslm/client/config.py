"""Client-side credential storage: the AutoSLM API key + control-plane URL.

Stored in ``~/.autoslm/config.json`` (dir 0700, file 0600 — it holds a secret).
Environment variables take precedence so CI/agents can inject credentials without
touching the file: ``AUTOSLM_API_KEY`` for the key, ``AUTOSLM_API_URL`` for the URL.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

DEFAULT_API_URL = "https://api.autoslm.dev"

CONFIG_DIR = Path.home() / ".autoslm"
CONFIG_PATH = CONFIG_DIR / "config.json"


def _read_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text())
    except (OSError, ValueError):
        return {}


def load_credentials() -> tuple[str, str | None]:
    """Resolve (api_url, api_key); the key is None when the user hasn't logged in."""
    cfg = _read_config()
    api_url = os.environ.get("AUTOSLM_API_URL") or cfg.get("api_url") or DEFAULT_API_URL
    api_key = os.environ.get("AUTOSLM_API_KEY") or cfg.get("api_key")
    return api_url.rstrip("/"), api_key


def save_credentials(api_key: str, api_url: str | None = None) -> Path:
    """Persist the key (and optionally a non-default URL) with private permissions."""
    cfg = _read_config()
    cfg["api_key"] = api_key
    if api_url:
        # Record the plane actually authenticated against. When it's the default, drop any
        # stored url instead of pinning it — this also clears a stale custom url from a
        # previous self-hosted login so later commands don't keep hitting the old host.
        if api_url.rstrip("/") == DEFAULT_API_URL.rstrip("/"):
            cfg.pop("api_url", None)
        else:
            cfg["api_url"] = api_url.rstrip("/")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(CONFIG_DIR, 0o700)
    # Create/truncate with 0600 from the start so the key is never briefly world-readable.
    # O_NOFOLLOW (where available): refuse to follow a symlink planted at CONFIG_PATH, so
    # saving the key can't be redirected to clobber an arbitrary file.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(CONFIG_PATH, flags, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    with contextlib.suppress(OSError):
        os.chmod(CONFIG_PATH, 0o600)
    return CONFIG_PATH
