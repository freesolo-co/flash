"""Client-side runtime secrets — read locally, sent with submit, never persisted in the run spec."""

from __future__ import annotations

import os
from pathlib import Path

# The global runtime-secret allow-list: keys collected/gated/injected for EVERY run regardless of
# algorithm. Only WANDB_API_KEY belongs here (optional experiment logging, useful to all arms). The
# opd teacher key (FIREWORKS_API_KEY) is NOT global: opd auto-declares it as a required environment
# secret in spec_from_dict, and each gate unions the spec's declared secrets on top of this set
# (client collection below; server create gate server/_deps.py; worker-env injection
# providers/runpod/train/deps.py). Keeping it out of the default means SFT/GRPO workers never
# receive the teacher key they don't need.
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
    """Collect run secrets from the local env; process env wins over .env/.env.local files."""

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
    """Resolve HF_TOKEN for flash export: --api-key > env > .env. Returns None if unset."""
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
