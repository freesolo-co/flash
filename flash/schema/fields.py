"""Field-level validators/coercers for Flash TOML config parsing.

Leaf helpers split out of ``flash.schema``: the ``ConfigError`` type, the [train] scalar
validators, the slug/worker-env/wandb validators, and the ``--set`` scalar coercer. None
reference the rest of the schema package; the package ``__init__`` re-exports them.
"""

from __future__ import annotations

import math
import re
import urllib.parse
from typing import Any

from flash.spec import WandbSpec

_GITHUB_SAFE_PART_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _train_int(train_raw: dict, key: str, *, minimum: int) -> int | None:
    """Validate an optional integer [train] knob (>= minimum) -> ConfigError (HTTP 400).

    None stays None (recipe default). Rejects bools, non-numbers, non-integers, and
    out-of-range values at parse time instead of letting them reach a provisioned worker.
    """
    v = train_raw.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"train.{key} must be an integer")
    # Check finiteness BEFORE int(v): int(inf) raises OverflowError and int(nan) ValueError
    # (the former would be a 500); reject both as a clean 400.
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
    # nan/inf slip past the range checks below (nan compares false, inf passes any minimum)
    # and would reach TRL optimizer/sampling settings; reject them as a 400 here.
    if not math.isfinite(v):
        raise ConfigError(f"train.{key} must be a finite number")
    if exclusive and v <= minimum:
        raise ConfigError(f"train.{key} must be > {minimum}")
    if not exclusive and v < minimum:
        raise ConfigError(f"train.{key} must be >= {minimum}")
    if maximum is not None and v > maximum:
        raise ConfigError(f"train.{key} must be between {minimum} and {maximum}")
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


def _require_slug(value: str, message: str) -> None:
    """Require an ``owner/name`` slug."""
    text = (value or "").strip()
    if not text or ":" in text:
        raise ConfigError(message)
    parsed = urllib.parse.urlparse(text)
    if parsed.scheme or parsed.netloc:
        raise ConfigError(message)
    parts = text.split("/")
    if len(parts) != 2 or not _is_safe_github_path_parts(parts):
        raise ConfigError(message)


def _require_environment_ref(value: str, message: str) -> None:
    """Require a Freesolo environment id."""
    try:
        _require_slug(value, message)
        return
    except ConfigError:
        pass
    if value.startswith("github:"):
        body = value[len("github:") :]
        repo_ref, sep, path = body.partition(":")
        repo, at, ref = repo_ref.partition("@")
        if at and not ref:
            raise ConfigError(message)
        owner_repo = repo.split("/")
        if (
            len(owner_repo) == 2
            and _is_safe_github_path_parts(owner_repo)
            and (not at or _is_safe_github_path_parts([ref]))
            and (not sep or _is_safe_environment_path(path))
        ):
            return
        raise ConfigError(message)
    if value.startswith("https://github.com/") or value.startswith("http://github.com/"):
        parsed = urllib.parse.urlparse(value)
        if parsed.scheme in {"http", "https"} and parsed.netloc.lower() == "github.com":
            parts = [
                part for part in urllib.parse.unquote(parsed.path).strip("/").split("/") if part
            ]
            if len(parts) < 2:
                raise ConfigError(message)
            owner, repo = parts[0], parts[1]
            repo = repo[:-4] if repo.endswith(".git") else repo
            if len(parts) == 2:
                if not _is_safe_github_path_parts([owner, repo]):
                    raise ConfigError(message)
            elif len(parts) >= 5 and parts[2] in {"blob", "tree"}:
                ref = parts[3]
                if not _is_safe_github_path_parts([ref]):
                    raise ConfigError(message)
                raw_path = "/".join(parts[4:])
                if not _is_safe_environment_path(raw_path):
                    raise ConfigError(message)
                if not _is_safe_github_path_parts([owner, repo, ref]):
                    raise ConfigError(message)
            else:
                raise ConfigError(message)
            return
    raise ConfigError(message)


def _is_safe_environment_path(path: str) -> bool:
    if not path:
        return True
    raw = path.strip().replace("\\", "/")
    if raw.startswith("/"):
        return False
    parts = [part for part in raw.split("/") if part]
    if not parts:
        return True
    return not any(part in {".", ".."} for part in parts)


def _is_safe_github_path_parts(parts: list[str]) -> bool:
    if any(part in {".", "..", ""} for part in parts):
        return False
    return all(_GITHUB_SAFE_PART_RE.fullmatch(part) for part in parts)


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


def _worker_env(raw: Any) -> dict[str, str]:
    """Parse the optional [worker_env] table: per-run worker env overrides (string-valued)."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError("[worker_env] must be a table of string key/values")
    env = {str(k): str(v) for k, v in raw.items()}
    # Env var NAMES must be usable by subprocess.Popen(env=...) on the worker, which raises
    # ValueError for an empty name or one containing '=' or a NUL byte (and whitespace breaks most
    # shells). Reject these at parse time so a malformed [worker_env] (e.g. a TOML quoted key like
    # "BAD=KEY", or an empty key) fails on config load — not after a worker has been provisioned.
    bad_names = sorted(repr(k) for k in env if (not k) or any(c in k for c in "=\0 \t\n\r"))
    if bad_names:
        raise ConfigError(
            f"[worker_env] has invalid environment variable name(s): {', '.join(bad_names)}; an "
            "env var name must be non-empty and contain no '=', whitespace, or NUL byte"
        )
    # [worker_env] is serialized into job_spec_json (persisted + logged), so it must NOT carry
    # secrets — they would leak into run artifacts. Reject secret-looking keys; operators set
    # those as real process environment variables (forwarded to the worker out-of-band) instead.
    # Detect by `_`-delimited WORD components (not substring): flag a secret WORD, or `KEY`
    # qualified by a credential context. This catches HF_TOKEN, *_API_KEY, SECRET_KEY, INTERNAL_KEY,
    # CREDENTIAL, AWS_SECRET_ACCESS_KEY, GITHUB_PAT (PAT word), and credential keys like SSH_KEY /
    # DEPLOY_KEY / GPG_KEY (KEY qualified by a credential context) — while allowing legit knobs whose
    # names merely contain a marker (RL_VLLM_MAX_BATCHED_TOKENS -> word TOKENS, not TOKEN; a bare
    # SORT_KEY -> KEY without a secret qualifier).
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
        "PAT",  # personal access token (e.g. GITHUB_PAT, GH_PAT)
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
        # credential-key contexts: SSH_KEY, DEPLOY_KEY, GPG_KEY, RSA_KEY, TLS/SSL/PEM keys, etc.
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


# Allowed [wandb] config keys -> typed JobSpec.wandb fields (first-class spec config, NOT env vars).
_WANDB_KEYS = ("project", "run_name")


def _wandb_spec(raw: Any) -> WandbSpec:
    """Parse the optional ``[wandb]`` table into a typed ``WandbSpec`` (project / run_name).

    These are non-secret W&B naming labels carried as first-class spec config (round-tripped in
    the job-spec JSON the worker reads), NOT environment variables. The worker honors them in
    ``engine.worker.wandb_report_to`` / ``wandb_run_name``, so a run can land in its own W&B
    project under its own run name instead of the hardcoded ``flash`` / ``flash-…`` defaults.
    Settable in TOML (``[wandb] project = …``) or via ``flash train cfg.toml --set
    wandb.project=… --set wandb.run_name=…``. The actual W&B credential (WANDB_API_KEY) stays an
    env-var secret — only the naming config lives here."""
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
        # Absent OR null means "unset". A serialized JobSpec round-trips unset wandb fields as
        # null (``asdict`` emits ``{"project": null, "run_name": null}``), so re-parsing a spec —
        # which is exactly what the control plane does on submit, ``spec_from_dict(spec.to_dict())``
        # — must accept null without demanding a value, or every run that omits ``[wandb]`` is
        # rejected. Only an explicitly-set value is validated: a bare ""/whitespace is a real
        # config mistake worth flagging.
        if val is None:
            continue
        if not isinstance(val, str) or not val.strip():
            raise ConfigError(f"[wandb] {key} must be a non-empty string")
        values[key] = val.strip()
    return WandbSpec(**values)
