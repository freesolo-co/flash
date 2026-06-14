"""Environment registry used by specs, worker, CLI, and server.

Verifiers-only: every environment is a Prime Intellect ``verifiers`` env. There are no
built-in task environments. ``load_environment`` resolves an env from one of two sources:

  * a LOCAL verifiers env module (``path``) — a directory or ``*.py`` exposing
    ``load_environment(**kwargs) -> verifiers.Environment``; or
  * an INSTALLED / Hub verifiers env (``env_id``) resolvable by ``verifiers``
    (installed via ``slm env install owner/name``, recorded in the manifest below).
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


def list_environments() -> list[str]:
    """All known environments — just the installed verifiers/Hub envs (no builtins)."""
    return list_installed_verifiers_envs()


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


def worker_pip_for_env(env_id: str, params: dict | None = None) -> list[str]:
    """Pip requirements the GPU worker needs to run ``env_id`` (a verifiers/Hub env).

    Always installs ``verifiers``; the env wheel is added only for an env RECORDED in the
    local install manifest (``slm env install`` captures its package name and the Prime
    Intellect ``extra_index_url``). An unrecorded Hub id ships ``verifiers`` only — to run a
    published Hub env on the managed worker, ``slm env install`` it first (so the wheel +
    index are known) or set ``[environment] pip`` explicitly. Also installs a separate
    **eval** Hub env (``[environment.params] eval_env_id``, alias ``eval_env``) when
    configured, and carries any recorded ``extra_index_url`` through to the worker's
    ``pip install``.

    A local-path env is for LOCAL runs only — it is NOT uploaded to a managed worker (Flash
    ships only the ``autoslm`` package, not the client project tree), so the managed CLI
    rejects ``[environment] path`` before submit and managed runs must reference a published
    Hub ``id``. For local execution it just needs ``verifiers`` installed.
    """
    params = params or {}
    manifest = load_installed_manifest()
    refs = [env_id]
    eval_ref = params.get("eval_env_id") or params.get("eval_env")
    if eval_ref:
        refs.append(str(eval_ref))
    deps = ["verifiers"]
    indexes: list[str] = []
    for ref in refs:
        entry = manifest.get(ref) or {}
        pkg = entry.get("package") or (_bare_wheel_name(ref) if ref in manifest else None)
        if pkg:
            deps.append(pkg)
        idx = entry.get("extra_index_url")
        if idx and idx not in indexes:
            indexes.append(idx)
    out = list(dict.fromkeys(deps))  # dedupe, preserve order
    for idx in indexes:
        out += ["--extra-index-url", idx]
    return out


def load_environment(
    env_id: str, params: dict | None = None, path: str | None = None
) -> Environment:
    """Load a verifiers environment and wrap it in AutoSLM's protocol.

    ``path`` -> a LOCAL verifiers env module (dir or ``*.py``); otherwise ``env_id`` is
    resolved as an installed / Prime Hub verifiers env.
    """
    params = params or {}
    from .verifiers_adapter import (
        load_local_verifiers_environment,
        load_verifiers_environment,
    )

    if path:
        return load_local_verifiers_environment(path, env_id=env_id or path, **params)
    if not env_id:
        raise ValueError(
            "no environment specified: set [environment] id to a verifiers/Prime Hub env "
            "slug (e.g. 'owner/name'), or [environment] path to a local verifiers env module"
        )
    return load_verifiers_environment(env_id, **params)
