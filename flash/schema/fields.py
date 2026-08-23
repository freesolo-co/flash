"""Field-level validators/coercers for Flash TOML config parsing."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from flash.content.structured_outputs import CONSTRAINT_KEYS as _SO_CONSTRAINT_KEYS
from flash.core.grpo import GRPO_CONTROL_PLANE_OWNED_ENV_KEYS
from flash.core.spec import (
    CONTROL_PLANE_OWNED_ENV_KEYS,
    CREDIT_ASSIGNMENTS,
    DEFAULT_CREDIT_ASSIGNMENT,
    CreditAssignment,
    WandbSpec,
)
from flash.envs.loading.adapter import is_freesolo_environment_id


def _section_int(
    section_raw: dict,
    section: str,
    key: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
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
    if minimum is not None and maximum is not None and (v < minimum or v > maximum):
        raise ConfigError(f"{section}.{key} must be between {minimum} and {maximum}")
    if minimum is not None and v < minimum:
        raise ConfigError(f"{section}.{key} must be >= {minimum}")
    if maximum is not None and v > maximum:
        raise ConfigError(f"{section}.{key} must be <= {maximum}")
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
    stored as its canonical friendly alias. Provider and repository identifiers are never accepted as
    user input. An unsupported teacher is rejected before a paid GPU is provisioned."""
    v = train_raw.get("teacher_model")
    if v is None:
        return ""
    if not isinstance(v, str):
        raise ConfigError("train.teacher_model must be a string")
    if not v.strip():
        return ""
    # Imported lazily: recipe is dependency-free, but keep fields.py's import graph minimal.
    from flash.engine.plan.recipe import resolve_teacher

    # resolve_teacher owns the allow-list + its enumeration; reuse its message so the choices are
    # listed in exactly one place (recipe.py).
    try:
        return resolve_teacher(v).alias
    except ValueError as exc:
        raise ConfigError(
            f"train.{exc}. The teacher is platform-managed; bring-your-own teachers are not supported."
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
# above from flash.content.structured_outputs as the single source of truth) plus these options.
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


def _train_credit_assignment(train_raw: dict) -> CreditAssignment:
    """Validate the multi-turn GRPO credit-assignment mode."""
    value = train_raw.get("credit_assignment")
    if value is None:
        return DEFAULT_CREDIT_ASSIGNMENT
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return DEFAULT_CREDIT_ASSIGNMENT
        for mode in CREDIT_ASSIGNMENTS:
            if normalized == mode:
                return mode
    allowed = '" or "'.join(CREDIT_ASSIGNMENTS)
    raise ConfigError(f'train.credit_assignment must be "{allowed}"; got {value!r}')


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
    # ',' is rejected because declared secret names travel to every redactor as the comma-joined
    # FLASH_SECRET_ENV_KEYS list (flash._internal.diagnostics): a name containing a comma is
    # indistinguishable from two names there, so the real key goes unrecognized and its value
    # reaches diagnostics verbatim. Rejecting it here keeps that channel unambiguous by
    # construction rather than needing an escape.
    bad_names = sorted(repr(k) for k in names if (not k) or any(c in k for c in "=,\0 \t\n\r"))
    if bad_names:
        raise ConfigError(
            f"{context} has invalid environment variable name(s): {', '.join(bad_names)}; an "
            "env var name must be non-empty and contain no '=', ',', whitespace, or NUL byte"
        )


_RESERVED_ENVIRONMENT_SECRET_KEYS = frozenset(
    {
        "RUNPOD_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "GITHUB_TOKEN",
        "FREESOLO_API_KEY",
        "FREESOLO_INTERNAL_KEY",
        *CONTROL_PLANE_OWNED_ENV_KEYS,
    }
)


# `scheme://userinfo@host` in a direct or VCS requirement, where userinfo is anything before the
# authority's `@`. ANY nonempty userinfo is rejected, not just the `user:password` shape: a github
# token is conventionally supplied username-only (`https://ghp_xxx@github.com/...`), and a colon can
# also arrive percent-encoded, so matching on the colon would miss the most likely leak. Anchored on
# `://` and stopping at `/` so a bare `pkg@1.2` or a PEP 508 `name @ https://host/x.whl` is untouched.
_PIP_URL_USERINFO_RE = re.compile(r"://[^/@\s]+@")

# a query string on a requirement URL (`...whl?private_token=x`, a presigned `?X-Amz-Signature=...`).
# rejected wholesale rather than by parameter name: the credential-carrying names are unbounded, and
# naming a package needs no query, so there is nothing legitimate to preserve by allowing some.
_PIP_URL_QUERY_RE = re.compile(r"://[^\s]*\?")


def _pip_echo(value: Any) -> str:
    """Quote a rejected pip value back to the author, redacting anything URL-shaped.

    Every rejection message here is printed by the CLI and returned verbatim as the server's HTTP
    error detail, so a value quoted into one lands in terminals, CI logs and API logs. A URL is the
    only pip syntax that can carry a credential, and this runs on values that failed validation --
    including before the credential guard is reached -- so URL-shaped input is never echoed at all.
    Redacting the whole value rather than the userinfo keeps this sound for credential shapes the
    guard does not model yet, which is the failure mode that produced this helper.
    """
    return "<redacted url>" if "://" in str(value) else repr(value)


def _environment_pip(raw: Any) -> tuple[str, ...]:
    """Parse [environment].pip as the scorer's own third-party requirements.

    A malformed entry would otherwise reach the worker's pip invocation, where it fails mid-install
    after the GPU is already allocated and billing. A bare string is called out separately: TOML has
    no implicit one-element list, so `pip = "pymongo"` is the natural first mistake to make here.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        # the suggestion is meant to be pasted straight into the toml, so quote it the way toml
        # does rather than the way repr() does.
        suggestion = '"<redacted url>"' if "://" in raw else f'"{raw}"'
        raise ConfigError(
            f"[environment] pip must be a list of requirement strings, not a string: "
            f"use [{suggestion}]"
        )
    # tuple as well as list: to_dict() emits the spec's own tuple, and that payload is re-parsed
    # here on the submit round trip (mirrors _environment_secrets).
    if not isinstance(raw, (list, tuple)):
        raise ConfigError("[environment] pip must be a list of requirement strings")
    requirements = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(
                f"[environment] pip entries must be non-empty requirement strings "
                f"(got: {_pip_echo(item)})"
            )
        requirement = item.strip()
        # entries are spliced straight into `python -m pip install` on the worker, so an option
        # flag is not just an odd requirement: `--no-deps` or `--target=...` would change how the
        # mandatory freesolo worker requirement installs, from a field that only names packages.
        # `--extra-index-url=https://user:token@host` also makes this a credential-bearing branch,
        # which is why the echo is redacted here rather than after the URL guard below.
        if requirement.startswith("-"):
            raise ConfigError(
                "[environment] pip entries must be requirements, not pip options "
                f"(got: {_pip_echo(requirement)})"
            )
        # A spec is not a secret store: pip is persisted verbatim in ``RunStatus.spec`` and uploaded
        # inside the worker's ``metrics.json`` job_spec, so a token embedded in a direct or VCS URL
        # would be written to disk and to the run log in plaintext.
        if _PIP_URL_USERINFO_RE.search(requirement) or _PIP_URL_QUERY_RE.search(requirement):
            raise ConfigError(
                "[environment] pip entries must not embed credentials in a URL: the spec is stored "
                "and uploaded in plaintext. Use [environment] secrets for the credential and an "
                "unauthenticated requirement URL without a query string."
            )
        requirements.append(requirement)
    return tuple(requirements)


def _environment_secrets(raw: Any, algorithm: str) -> tuple[str, ...]:
    """Parse [environment].secrets as declared worker env-var secret names."""
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raise ConfigError("[environment] secrets must be a list of environment variable names")
    if not all(isinstance(name, str) for name in raw):
        raise ConfigError("[environment] secrets entries must be strings")
    secrets = tuple(dict.fromkeys(raw))
    _validate_env_var_names(secrets, "[environment] secrets")
    # matched case-insensitively even though linux env names are case-sensitive: build_worker_env
    # tests ownership on the UPPERCASED name, so a declared `flash_secret_env_keys` would pass this
    # check and then be silently dropped from the worker env, launching the job without the secret
    # it declared as required. reserving the whole case-space keeps parse and dispatch agreed.
    reserved_keys = _RESERVED_ENVIRONMENT_SECRET_KEYS
    if algorithm == "grpo":
        reserved_keys |= GRPO_CONTROL_PLANE_OWNED_ENV_KEYS
    reserved = sorted(k for k in secrets if k.upper() in reserved_keys)
    if reserved:
        raise ConfigError(
            f"[environment] secrets must not include platform-managed key(s): {', '.join(reserved)}"
        )
    return secrets


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
