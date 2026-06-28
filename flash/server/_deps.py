"""Shared FastAPI request dependencies and spec/secret parsing for the route modules.

Only import lazily from ``create_app()`` — never at ``flash.server.app`` import time.
``owned_run`` resolves ``get_status`` via the module so test patches on ``app.get_status`` are honored.
"""

from __future__ import annotations

from fastapi import Header, HTTPException

from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS
from flash.schema import ConfigError, spec_from_dict
from flash.spec import JobSpec

from . import app as _app
from . import auth, db

_RUNTIME_SECRET_KEYS = DEFAULT_RUNTIME_SECRET_KEYS


def require_key(authorization: str | None = Header(default=None)) -> dict:
    key = auth.authenticate(authorization)
    if key is None:
        raise HTTPException(
            status_code=401,
            detail="invalid or missing API key; log in with `flash login` using your "
            "freesolo API key",
        )
    return key


def owned_run(run_id: str, key: dict):
    """Load a run's status iff `key` owns it; 404 otherwise (don't leak existence)."""
    if db.run_owner(run_id) != key["id"]:
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    try:
        return _app.get_status(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _require_bool(payload: dict, field: str, default: bool) -> bool:
    """Read ``payload[field]`` as a real JSON boolean; 400 on non-bool. Never coerce — "false" (str) is truthy."""
    value = payload.get(field, default)
    if not isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"{field} must be a boolean")
    return value


def _parse_spec(payload: dict, run_id: str) -> JobSpec:
    # Use get()+None check, not `or {}` — falsy non-objects must still hit the type check below.
    spec_raw = payload.get("spec")
    if spec_raw is None:
        spec_raw = {}
    if not isinstance(spec_raw, dict):
        raise HTTPException(status_code=400, detail="spec must be a JSON object")
    env_raw = spec_raw.get("environment")
    if env_raw is None:
        env_raw = {}
    if not isinstance(env_raw, dict):
        raise HTTPException(
            status_code=400, detail="spec.environment must be a JSON object"
        )
    if env_raw.get("path"):
        raise HTTPException(
            status_code=400,
            detail="local environment paths are not supported on the managed service; "
            "publish the environment with `flash env push --name <name>`, then reference it "
            "by the returned environment id",
        )
    try:
        return spec_from_dict(spec_raw, run_id=run_id)
    except (ConfigError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _runtime_secrets(
    payload: dict, spec: JobSpec, *, require_environment_secrets: bool
) -> dict[str, str]:
    # Use get()+None check, not `or {}` — falsy non-objects must still hit the type check below.
    raw = payload.get("runtime_secrets")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="runtime_secrets must be a JSON object")
    allowed = set(_RUNTIME_SECRET_KEYS) | set(spec.environment.secrets)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                "unsupported runtime secret(s): "
                f"{', '.join(unknown)} (allowed: {', '.join(sorted(allowed))})"
            ),
        )
    out: dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if not isinstance(value, str):
            raise HTTPException(status_code=400, detail=f"runtime_secrets.{key} must be a string")
        value = value.strip()
        if value:
            out[key] = value
    if require_environment_secrets:
        missing = sorted(set(spec.environment.secrets) - set(out))
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    "missing runtime secret(s) required by [environment] secrets: "
                    f"{', '.join(missing)}"
                ),
            )
    return out
