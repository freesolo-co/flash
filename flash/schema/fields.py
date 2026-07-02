"""Field-level validators/coercers for Flash TOML config parsing."""

from __future__ import annotations

import math
from typing import Any

from flash.envs.adapter import is_freesolo_environment_id
from flash.spec import WandbSpec


def _train_int(train_raw: dict, key: str, *, minimum: int) -> int | None:
    """Validate an optional integer [train] knob (>= minimum) -> ConfigError (HTTP 400)."""
    v = train_raw.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"train.{key} must be an integer")
    # int(inf) raises OverflowError (500), not ValueError; check finiteness first.
    if not math.isfinite(v) or float(v) != int(v):
        raise ConfigError(f"train.{key} must be a finite integer")
    v = int(v)
    if v < minimum:
        raise ConfigError(f"train.{key} must be >= {minimum}")
    return v


def _train_float(
    train_raw: dict,
    key: str,
    *,
    minimum: float,
    exclusive: bool = False,
    maximum: float | None = None,
) -> float | None:
    """Validate an optional float [train] knob -> ConfigError (HTTP 400). None stays None."""
    v = train_raw.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"train.{key} must be a number")
    v = float(v)
    # nan/inf slip past range checks (nan compares false, inf passes any minimum).
    if not math.isfinite(v):
        raise ConfigError(f"train.{key} must be a finite number")
    if exclusive and v <= minimum:
        raise ConfigError(f"train.{key} must be > {minimum}")
    if not exclusive and v < minimum:
        raise ConfigError(f"train.{key} must be >= {minimum}")
    if maximum is not None and v > maximum:
        raise ConfigError(f"train.{key} must be between {minimum} and {maximum}")
    return v


def _train_str(train_raw: dict, key: str, *, choices: tuple[str, ...] | None = None) -> str:
    """Validate an optional string [train] knob -> ConfigError. Missing/None -> "" (worker default).

    An empty string is treated as unset (deferred to the recipe default), so a `choices` gate only
    fires on a non-empty value.
    """
    v = train_raw.get(key)
    if v is None:
        return ""
    if not isinstance(v, str):
        raise ConfigError(f"train.{key} must be a string")
    v = v.strip()
    if v and choices is not None and v not in choices:
        raise ConfigError(f"train.{key} must be one of {', '.join(choices)} (got {v!r})")
    return v


def _train_stops(train_raw: dict) -> tuple[str, ...]:
    """Validate stop_sequences -> ConfigError. A string is ONE stop (never char-split);
    a list must hold strings; empties are dropped; anything else is rejected."""
    v = train_raw.get("stop_sequences")
    if v is None:
        return ()
    if isinstance(v, str):
        return (v,) if v else ()
    if not isinstance(v, (list, tuple)):
        raise ConfigError("train.stop_sequences must be a string or a list of strings")
    for s in v:
        if not isinstance(s, str):
            raise ConfigError("train.stop_sequences entries must be strings")
    return tuple(s for s in v if s)


class ConfigError(ValueError):
    pass


def _require_environment_ref(value: str, message: str) -> None:
    """Require a Freesolo environment id."""
    if not is_freesolo_environment_id(value):
        raise ConfigError(message)


def _coerce_scalar(value: str):
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _validate_env_var_names(names, context: str) -> None:
    bad_names = sorted(repr(k) for k in names if (not k) or any(c in k for c in "=\0 \t\n\r"))
    if bad_names:
        raise ConfigError(
            f"{context} has invalid environment variable name(s): {', '.join(bad_names)}; an "
            "env var name must be non-empty and contain no '=', whitespace, or NUL byte"
        )


_RESERVED_ENVIRONMENT_SECRET_KEYS = frozenset(
    {
        "RUNPOD_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "GITHUB_TOKEN",
        "FREESOLO_API_KEY",
        "FREESOLO_INTERNAL_KEY",
        "RUN_ID",
        "HF_REPO",
        "FLASH_ARM",
    }
)


def _environment_secrets(raw: Any) -> tuple[str, ...]:
    """Parse [environment].secrets as declared worker env-var secret names."""
    if raw is None:
        return ()
    if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
        raise ConfigError("[environment] secrets must be a list of environment variable names")
    if not all(isinstance(name, str) for name in raw):
        raise ConfigError("[environment] secrets entries must be strings")
    secrets = tuple(dict.fromkeys(raw))
    _validate_env_var_names(secrets, "[environment] secrets")
    reserved = sorted(set(secrets) & _RESERVED_ENVIRONMENT_SECRET_KEYS)
    if reserved:
        raise ConfigError(
            "[environment] secrets must not include platform-managed key(s): "
            f"{', '.join(reserved)}"
        )
    return secrets


def _worker_env(raw: Any) -> dict[str, str]:
    """Parse the optional [worker_env] table: per-run worker env overrides (string-valued)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("[worker_env] must be a table of string key/values")
    env = {str(k): str(v) for k, v in raw.items()}
    _validate_env_var_names(env, "[worker_env]")
    # [worker_env] is serialized into job_spec_json (persisted + logged) — must not carry secrets.
    # Match by word components (not substring): KEY only flagged when qualified by a credential context.
    _secret_words = {
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "PASSPHRASE",
        "CREDENTIAL",
        "CREDENTIALS",
        "APIKEY",
        "PRIVATEKEY",
        "PAT",
    }
    _key_qualifiers = {
        "API",
        "SECRET",
        "PRIVATE",
        "ACCESS",
        "INTERNAL",
        "AUTH",
        "SIGNING",
        "ENCRYPTION",
        "SSH",
        "DEPLOY",
        "GPG",
        "PGP",
        "RSA",
        "PEM",
        "SSL",
        "TLS",
    }

    def _is_secret_key(name: str) -> bool:
        words = set(name.upper().split("_"))
        return bool(words & _secret_words) or ("KEY" in words and bool(words & _key_qualifiers))

    secrets = sorted(k for k in env if _is_secret_key(k))
    if secrets:
        raise ConfigError(
            f"[worker_env] must not contain secret-bearing keys ({', '.join(secrets)}); these are "
            "serialized into run artifacts; use provider process env or supported runtime secrets "
            "instead"
        )
    return env


_WANDB_KEYS = ("project", "run_name")


def _wandb_spec(raw: Any) -> WandbSpec:
    """Parse the optional ``[wandb]`` table into a typed ``WandbSpec`` (project / run_name)."""
    if raw is None:
        return WandbSpec()
    if not isinstance(raw, dict):
        raise ConfigError('[wandb] must be a table (e.g. project = "my-project")')
    unknown = sorted(set(raw) - set(_WANDB_KEYS))
    if unknown:
        raise ConfigError(
            f"[wandb] unknown key(s): {', '.join(unknown)} (allowed: {', '.join(_WANDB_KEYS)})"
        )
    values: dict[str, str] = {}
    for key in _WANDB_KEYS:
        val = raw.get(key)
        # null = "unset"; JobSpec round-trips unset fields as null so re-parsing must accept it.
        if val is None:
            continue
        if not isinstance(val, str) or not val.strip():
            raise ConfigError(f"[wandb] {key} must be a non-empty string")
        values[key] = val.strip()
    return WandbSpec(**values)
