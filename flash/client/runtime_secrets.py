"""Client-side runtime secrets for managed runs.

These values are read on the user's machine and sent only with the submit request. They are not
part of JobSpec/TOML, and the control plane must not persist them in run status or artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

RUNTIME_SECRET_KEYS = frozenset({"WANDB_API_KEY"})


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        lines = path.read_text(errors="ignore").splitlines()
    except OSError:
        return out
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        key = key.strip()
        if key not in RUNTIME_SECRET_KEYS:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            out[key] = value
    return out


def runtime_secrets_from_local_env(config_path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Collect supported run secrets from the user's local env.

    Process environment wins. As a convenience for local project workflows, also read `.env` and
    `.env.local` in the current directory and next to the config file. This deliberately does not
    scan arbitrary parent directories or serialize secrets into the run spec.
    """

    secrets = {key: value for key in RUNTIME_SECRET_KEYS if (value := os.environ.get(key))}
    candidates = [Path.cwd() / ".env", Path.cwd() / ".env.local"]
    if config_path:
        cfg_dir = Path(config_path).expanduser().resolve().parent
        candidates.extend([cfg_dir / ".env", cfg_dir / ".env.local"])
    for path in candidates:
        for key, value in _read_env_file(path).items():
            secrets.setdefault(key, value)
    return secrets
