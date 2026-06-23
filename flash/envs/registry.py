"""Environment registry used by specs, worker, CLI, and server.

Every managed run names a Freesolo SDK environment by a published environment id.
The canonical generated environment entrypoint is
``freesolo/environment.py:load_environment``.
"""

from __future__ import annotations

import os
from pathlib import Path

from .._fileio import read_json_or_empty, secure_json_write
from .base import Environment

# Manifest of local Freesolo environment ids (written by `flash env install`).
INSTALLED_MANIFEST = Path(
    os.environ.get("FLASH_ENVS_MANIFEST", str(Path.home() / ".flash" / "envs.json"))
)


def load_installed_manifest() -> dict:
    return read_json_or_empty(INSTALLED_MANIFEST)


def list_installed_environments() -> list[str]:
    """Freesolo environment ids recorded via `flash env install`."""
    return sorted(load_installed_manifest())


def record_installed_env(env_id: str, package: str, extras: dict | None = None) -> None:
    manifest = load_installed_manifest()
    manifest[env_id] = {"package": package, **(extras or {})}
    # The manifest can hold a credentialed --extra-index-url, so write it with private perms.
    secure_json_write(INSTALLED_MANIFEST, manifest)


def worker_pip_for_env(env_id: str) -> list[str]:
    """Pip deps the GPU worker needs to run a Freesolo environment."""
    return ["freesolo"]


def worker_hub_env_ids(env_id: str, params: dict | None = None) -> list[str]:
    """Backward-compatible name for worker environment refs.

    The worker resolves Freesolo environment ids lazily through :func:`load_environment`,
    so there is nothing to pre-install here.
    """
    return []


def load_environment(env_id: str, params: dict | None = None) -> Environment:
    """Load a Freesolo SDK environment and wrap it in Flash's protocol."""
    params = params or {}
    from .adapter import load_freesolo_environment

    if not env_id:
        raise ValueError(
            "no environment specified: set [environment] id to the id returned by `flash env push`"
        )
    return load_freesolo_environment(env_id, **params)
