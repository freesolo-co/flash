"""Client-side runtime secrets for managed runs.

These values are read on the user's machine and sent only with the submit request. They are not
part of JobSpec/TOML, and the control plane must not persist them in run status or artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_RUNTIME_SECRET_KEYS = frozenset({"WANDB_API_KEY"})


def _read_env_file(path: Path, keys: set[str]) -> dict[str, str]:
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
        if key not in keys:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            out[key] = value
    return out


def runtime_secrets_from_local_env(
    config_path: str | os.PathLike[str] | None = None,
    keys: tuple[str, ...] | list[str] | set[str] | None = None,
) -> dict[str, str]:
    """Collect supported run secrets from the user's local env.

    Process environment wins. As a convenience for local project workflows, also read `.env` and
    `.env.local` in the current directory and next to the config file. This deliberately does not
    scan arbitrary parent directories or serialize secrets into the run spec.
    """

    required = {str(key) for key in (keys or ())}
    wanted = set(DEFAULT_RUNTIME_SECRET_KEYS) | required
    secrets = {key: value for key in wanted if (value := os.environ.get(key))}
    candidates = [Path.cwd() / ".env", Path.cwd() / ".env.local"]
    if config_path:
        cfg_dir = Path(config_path).expanduser().resolve().parent
        candidates.extend([cfg_dir / ".env", cfg_dir / ".env.local"])
    for path in candidates:
        for key, value in _read_env_file(path, wanted).items():
            secrets.setdefault(key, value)
    missing = sorted(required - set(secrets))
    if missing:
        raise ValueError(
            "missing declared environment secret(s): "
            f"{', '.join(missing)}. Set them in your shell or local .env file before submitting; "
            "do not put secret values in TOML."
        )
    return secrets


def resolve_hf_token(explicit: str | None = None) -> str | None:
    """Resolve the HuggingFace token `flash export` writes the destination repo with.

    Order: an explicit value (the ``--api-key`` flag) > the process environment (HF_TOKEN) > a local
    ``.env`` / ``.env.local`` in the cwd. Only HF_TOKEN is accepted — the convention the rest of flash
    uses — not the huggingface_hub aliases, so the token source is unambiguous.
    Returns ``None`` when none is set. Read on the user's machine and sent only with the export
    request — never persisted in the run spec (same contract as the run secrets above).
    """
    if explicit and explicit.strip():
        return explicit.strip()
    value = os.environ.get("HF_TOKEN")
    if value and value.strip():
        return value.strip()
    for path in (Path.cwd() / ".env", Path.cwd() / ".env.local"):
        found = _read_env_file(path, {"HF_TOKEN"})
        if found.get("HF_TOKEN"):
            return found["HF_TOKEN"]
    return None
