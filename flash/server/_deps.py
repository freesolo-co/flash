"""Shared FastAPI request dependencies and spec/secret parsing for the route modules.

Imports fastapi, so this module must only be imported lazily from ``create_app()`` (inside
the server-extras guard) — never at ``flash.server.app`` import time. The route modules that
import it are themselves only imported from ``create_app()``, so that invariant holds.

``owned_run`` resolves ``get_status`` through the ``flash.server.app`` module (not a direct
import) so a test that patches ``app.get_status`` is honored by the handlers.
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


def _parse_spec(payload: dict, run_id: str) -> JobSpec:
    spec_raw = payload.get("spec") or {}
    # A non-object JSON value for `spec` (list/string/number) has no `.get`, so guard before
    # touching it — a bad payload is a 400 request-validation error, not an unhandled 500.
    if not isinstance(spec_raw, dict):
        raise HTTPException(status_code=400, detail="spec must be a JSON object")
    env_raw = spec_raw.get("environment") or {}
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
    raw = payload.get("runtime_secrets") or {}
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
