"""Field-level validators/coercers for Flash TOML config parsing."""

from __future__ import annotations

import json
import math
from typing import Any

from flash.engine.structured_outputs import CONSTRAINT_KEYS as _SO_CONSTRAINT_KEYS
from flash.envs.adapter import is_freesolo_environment_id
from flash.spec import WandbSpec, validate_worker_env_reserved


def _section_int(
    section_raw: dict, section: str, key: str, *, minimum: int | None = None
) -> int | None:
    """validate an optional section-qualified integer knob."""
    v = section_raw.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{section}.{key} must be an integer")
    # int(inf) raises overflowerror, so check finiteness before conversion.
    if not math.isfinite(v) or float(v) != int(v):
        raise ConfigError(f"{section}.{key} must be a finite integer")
    v = int(v)
    if minimum is not None and v < minimum:
        raise ConfigError(f"{section}.{key} must be >= {minimum}")
    return v


def _train_int(train_raw: dict, key: str, *, minimum: int) -> int | None:
    """Validate an optional integer [train] knob (>= minimum) -> ConfigError (HTTP 400)."""
    return _section_int(train_raw, "train", key, minimum=minimum)


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


def _train_teacher(train_raw: dict) -> str:
    """Validate [train] teacher_model against the managed OPD teacher allow-list -> ConfigError (400).

    Missing/None/blank -> "" (the worker then uses the default GLM 5.2 teacher). A supported value is
    resolved to its canonical Fireworks model id and stored, so a spaced ("GLM 5.2") or alias form is
    canonicalized once here and every downstream layer reads one representation; an unsupported teacher
    is rejected at PARSE time (before a paid GPU is provisioned), listing the allowed aliases."""
    v = train_raw.get("teacher_model")
    if v is None:
        return ""
    if not isinstance(v, str):
        raise ConfigError("train.teacher_model must be a string")
    if not v.strip():
        return ""
    # Imported lazily: recipe is dependency-free, but keep fields.py's import graph minimal.
    from flash.engine.recipe import resolve_teacher

    # resolve_teacher owns the allow-list + its enumeration; reuse its message so the choices are
    # listed in exactly one place (recipe.py).
    try:
        return resolve_teacher(v).model_id
    except ValueError as exc:
        raise ConfigError(
            f"train.{exc}. The teacher is a managed Fireworks model — "
            f"bring-your-own teachers are not supported."
        ) from None


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


# vLLM StructuredOutputsParams surface: exactly one constraint field (_SO_CONSTRAINT_KEYS, imported
# above from flash.engine.structured_outputs as the single source of truth) plus these options.
# vLLM also offers grammar/structural_tag, but Flash does not support them; reject explicitly so a
# `{"grammar": ...}` table isn't silently swallowed by the bare-JSON-schema fallback below.
_SO_REMOVED_KEYS = frozenset({"grammar", "structural_tag"})
_SO_OPTION_KEYS = ("disable_any_whitespace", "disable_additional_properties", "whitespace_pattern")
# Common TOML/JSON spellings folded onto the vLLM field names, so users can write whichever they know.
_SO_ALIASES = {"json_schema": "json", "schema": "json", "choices": "choice"}


def _so_error(msg: str) -> ConfigError:
    return ConfigError(f"train.structured_outputs {msg}")


def _so_canonical(value: Any) -> dict | None:
    """Fold one structured_outputs value into canonical StructuredOutputsParams kwargs.

    Accepts (flexibility over one blessed spelling): a canonical/aliased constraint table, a bare
    JSON-schema table, a JSON string of either, or the "json"/"json_object" shorthand. Returns
    None for the explicit "unconstrained" forms (false/""/"none")."""
    if value is None or value is False:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() == "none":
            return None
        if s.lower() in ("json", "json_object"):
            return {"json_object": True}
        try:
            parsed = json.loads(s)
        except ValueError as exc:
            raise _so_error(
                'string form must be JSON (a schema/constraint object) or the "json_object" '
                f"shorthand; got unparseable {s[:80]!r} ({exc})"
            ) from exc
        if not isinstance(parsed, dict):
            raise _so_error("JSON string form must decode to an object")
        return _so_canonical(parsed)
    if not isinstance(value, dict):
        raise _so_error("must be a table, a JSON string, or false")
    removed = sorted(k for k in value if k in _SO_REMOVED_KEYS)
    if removed:
        raise _so_error(
            f"{', '.join(removed)} constraint(s) are not supported; use one of "
            f"{', '.join(_SO_CONSTRAINT_KEYS)}"
        )

    folded = {_SO_ALIASES.get(k, k): v for k, v in value.items()}
    if len(folded) < len(value):
        raise _so_error(f"sets the same constraint twice via aliases: {sorted(value)}")
    constraints = [k for k in _SO_CONSTRAINT_KEYS if folded.get(k) is not None]
    if not constraints:
        # A stray backend option (disable_any_whitespace / whitespace_pattern / ...) with no
        # constraint is a misconfiguration, not a schema: reject it here rather than wrapping it
        # into a vacuous {"json": {...}} that silently swallows the option (mirrors the worker's
        # own "no constraint" guard in parse_structured_outputs).
        stray_options = sorted(k for k in folded if k in _SO_OPTION_KEYS)
        if stray_options:
            raise _so_error(
                f"sets backend option(s) {', '.join(stray_options)} without a constraint; "
                f"options only apply alongside one of {', '.join(_SO_CONSTRAINT_KEYS)}"
            )
        # No constraint key at all: the whole table is an inline JSON schema (its own
        # "type"/"properties" keys are schema vocabulary, not ours).
        return {"json": value}
    unknown = sorted(set(folded) - set(_SO_CONSTRAINT_KEYS) - set(_SO_OPTION_KEYS))
    if unknown:
        raise _so_error(
            f"unknown key(s): {', '.join(unknown)} (constraints: "
            f"{', '.join(_SO_CONSTRAINT_KEYS)}; options: {', '.join(_SO_OPTION_KEYS)}; "
            f"aliases: {', '.join(sorted(_SO_ALIASES))})"
        )
    if len(constraints) > 1:
        raise _so_error(f"must set exactly ONE constraint, got: {', '.join(constraints)}")

    kind = constraints[0]
    val = folded[kind]
    canonical: dict[str, Any] = {}
    if kind == "json":
        if isinstance(val, str):
            try:
                val = json.loads(val)
            except ValueError as exc:
                raise _so_error(f"json schema string is not valid JSON: {exc}") from exc
        if not isinstance(val, dict):
            raise _so_error("json must be a schema table or a JSON-object string")
        canonical["json"] = val
    elif kind == "choice":
        if (
            not isinstance(val, (list, tuple))
            or not val
            or not all(isinstance(c, str) and c for c in val)
        ):
            raise _so_error("choice must be a non-empty list of non-empty strings")
        canonical["choice"] = list(val)
    elif kind == "json_object":
        if val is not True:
            raise _so_error("json_object must be true (omit the key for unconstrained output)")
        canonical["json_object"] = True
    else:  # regex
        if not isinstance(val, str) or not val.strip():
            raise _so_error(f"{kind} must be a non-empty string")
        canonical[kind] = val

    for opt in _SO_OPTION_KEYS:
        v = folded.get(opt)
        if v is None:
            continue
        if opt == "whitespace_pattern":
            if not isinstance(v, str) or not v:
                raise _so_error("whitespace_pattern must be a non-empty string")
        elif not isinstance(v, bool):
            raise _so_error(f"{opt} must be a boolean")
        canonical[opt] = v
    return canonical


def _train_structured_outputs(train_raw: dict) -> str:
    """Validate/normalize [train] structured_outputs to canonical JSON ("" = unconstrained).

    The canonical string is exactly the kwargs of vLLM's StructuredOutputsParams, so the worker
    applies it with zero re-interpretation (json.loads -> StructuredOutputsParams(**spec))."""
    canonical = _so_canonical(train_raw.get("structured_outputs"))
    if canonical is None:
        return ""
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


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
        "SEED",
    }
)


def _environment_secrets(raw: Any) -> tuple[str, ...]:
    """Parse [environment].secrets as declared worker env-var secret names."""
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ConfigError("[environment] secrets must be a list of environment variable names")
    if not all(isinstance(name, str) for name in raw):
        raise ConfigError("[environment] secrets entries must be strings")
    secrets = tuple(dict.fromkeys(raw))
    _validate_env_var_names(secrets, "[environment] secrets")
    reserved = sorted(set(secrets) & _RESERVED_ENVIRONMENT_SECRET_KEYS)
    if reserved:
        raise ConfigError(
            f"[environment] secrets must not include platform-managed key(s): {', '.join(reserved)}"
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
    try:
        validate_worker_env_reserved(env)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
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
