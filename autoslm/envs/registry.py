"""Environment registry used by specs, worker, CLI, and server.

Verifiers-only: every environment is a Prime Intellect ``verifiers`` env. There are no
built-in task environments and no local-file environment mode (``schema.py`` rejects a
``path`` key outright). ``load_environment`` resolves a single source: an installed /
Prime Hub verifiers env referenced by its slug (``env_id``, ``owner/name``), resolvable
by ``verifiers`` (installed via ``slm env install owner/name`` and recorded in the
manifest below).
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path

from .base import Environment

# Manifest of installed verifiers / Prime Hub environments (written by `slm env install`).
INSTALLED_MANIFEST = Path(
    os.environ.get("AUTOSLM_ENVS_MANIFEST", str(Path.home() / ".autoslm" / "envs.json"))
)


def load_installed_manifest() -> dict:
    try:
        return json.loads(INSTALLED_MANIFEST.read_text())
    except (OSError, ValueError):
        return {}


def list_installed_verifiers_envs() -> list[str]:
    """Names of verifiers/Hub environments installed via `slm env install`."""
    return sorted(load_installed_manifest())


def record_installed_env(env_id: str, package: str, extras: dict | None = None) -> None:
    manifest = load_installed_manifest()
    manifest[env_id] = {"package": package, **(extras or {})}
    INSTALLED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    # The manifest can hold a credentialed --extra-index-url. Create/truncate with 0600
    # from the start (not write_text + chmod, which leaves it umask-readable in between);
    # O_NOFOLLOW refuses a symlink planted at the path. chmod after covers a pre-existing
    # file created before this code path.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(INSTALLED_MANIFEST, flags, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    with contextlib.suppress(OSError):
        os.chmod(INSTALLED_MANIFEST, 0o600)


def _bare_wheel_name(env_ref: str) -> str:
    """``owner/name`` Hub slug -> the bare pip wheel name (``name``)."""
    return env_ref.split("/", 1)[1] if "/" in env_ref else env_ref


def worker_pip_for_env(env_id: str) -> list[str]:
    """Pip deps the GPU worker needs to run ``env_id`` (a verifiers/Hub env): just ``verifiers``.

    The environment itself (and any separate eval env) is installed on the worker via the
    authenticated ``prime env install`` (see :func:`worker_hub_env_ids`), not pip — the public
    pip index does not serve private env wheels. Override with ``[environment] pip`` if a run
    needs extra packages.
    """
    return ["verifiers"]


def worker_hub_env_ids(env_id: str, params: dict | None = None) -> list[str]:
    """The Prime Hub env ids the worker must ``prime env install`` for this run.

    The training env plus a separate **eval** Hub env (``[environment.params] eval_env_id``,
    alias ``eval_env``) when configured. ``prime env install`` is authenticated by
    ``PRIME_API_KEY`` and installs public and private envs alike.
    """
    params = params or {}
    ids = [env_id, params.get("eval_env_id") or params.get("eval_env")]
    return list(dict.fromkeys(str(i) for i in ids if i))


def load_environment(env_id: str, params: dict | None = None) -> Environment:
    """Load a verifiers environment and wrap it in AutoSLM's protocol.

    ``env_id`` is resolved as an installed / Prime Hub verifiers env slug.
    """
    params = params or {}
    from .adapter import load_verifiers_environment

    if not env_id:
        raise ValueError(
            "no environment specified: set [environment] id to a verifiers/Prime Hub env "
            "slug (e.g. 'owner/name')"
        )
    return load_verifiers_environment(env_id, **params)
