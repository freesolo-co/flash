"""Environment registry used by specs, worker, CLI, and server.

Every managed run names a Freesolo SDK environment by Hub slug.
The canonical generated environment entrypoint is ``environment.py:load_environment``.
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


def load_environment(
    env_id: str, params: dict | None = None, resolved_sha: str | None = None
) -> Environment:
    """Load a Freesolo SDK environment and wrap it in Flash's protocol.

    ``resolved_sha`` is the optional resolve-once hint (the control-plane-pinned commit sha for the
    env's GitHub ref). None/"" preserves today's behavior — the adapter resolves the ref itself.
    """
    params = params or {}
    from .adapter import load_freesolo_environment

    if not env_id:
        raise ValueError(
            "no environment specified: set [environment] id to the id returned by "
            "`flash env push --name <name>` (for example 'your-name/your-env')"
        )
    # resolved_sha is our reserved control-plane kwarg; never let an env-supplied param shadow it
    # (a stray params["resolved_sha"] would otherwise raise "multiple values for keyword argument").
    _params = {k: v for k, v in params.items() if k != "resolved_sha"}
    return load_freesolo_environment(env_id, resolved_sha=resolved_sha or None, **_params)
