"""Shared FastAPI request dependencies and spec/secret parsing for the route modules.

Only import lazily from ``create_app()`` — never at ``flash.server.app`` import time.
``owned_run`` resolves ``get_status`` via the module so test patches on ``app.get_status`` are honored.
"""

from __future__ import annotations

from fastapi import Header, HTTPException

from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS
from flash.schema import ConfigError, spec_from_dict
from flash.spec import JobSpec, require_project_id

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


def _load_status(run_id: str):
    """Resolve `run_id` to its status via the module (so `app.get_status` patches are honored)."""
    try:
        return _app.get_status(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def owned_run(run_id: str, key: dict):
    """Load a run's status iff `key` owns it; 404 otherwise (don't leak existence)."""
    if db.run_owner(run_id) != key["id"]:
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    return _load_status(run_id)


def _internal_org_run(run_id: str, key: dict, org_id: str | None = None):
    org = (org_id or "").strip()
    if key.get("auth_kind") == "internal" and org:
        from flash.runner import _status_org_id

        status = _load_status(run_id)
        if _status_org_id(status) == org:
            return status
    raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")


def readable_run(run_id: str, key: dict, org_id: str | None = None):
    """Load a run's status for READ-only endpoints (`/logs`, `/worker`) via one of two paths.

    Mirrors the env-delete posture (a user key carries its own identity; the org-agnostic
    internal key is the trusted platform caller):
      * exact-key owner -- the API key that created the run. This is the `flash runs log` CLI path.
      * internal/service key + matching ``X-Freesolo-Org-Id`` -- the platform web proxy. The
        browser has no per-run API key, so the platform authenticates the user, checks org
        membership on the mirror row, then calls with the internal key and the run's org here.
        The header is honored only for the internal key and only when it matches the run's
        PERSISTED org, so a proxy bug can never cross orgs. A non-owner user key does reach this
        check but fails the ``auth_kind == "internal"`` gate, so its org header is never honored.
    404 (never 403) on every failure so we never leak whether a run exists.
    """
    if db.run_owner(run_id) == key["id"]:
        return _load_status(run_id)
    return _internal_org_run(run_id, key, org_id)


def manageable_run(
    run_id: str,
    key: dict,
    org_id: str | None = None,
    project_id: str | None = None,
):
    """Load a run for deployment management by its exact owner or matching internal scope."""
    if key.get("auth_kind") == "internal":
        status = _internal_org_run(run_id, key, org_id)
        persisted_project = status.spec.get("project") if isinstance(status.spec, dict) else None
        try:
            project = require_project_id(project_id)
            persisted_project = require_project_id(persisted_project)
        except (TypeError, ValueError):
            pass
        else:
            if persisted_project == project:
                return status
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    return owned_run(run_id, key)


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
        raise HTTPException(status_code=400, detail="spec.environment must be a JSON object")
    if env_raw.get("path"):
        raise HTTPException(
            status_code=400,
            detail="local environment paths are not supported on the managed service; "
            "publish the environment with `flash env push --project <project-uuid> --name <name>`, "
            "then reference it "
            "by the returned environment id",
        )
    try:
        spec = spec_from_dict(spec_raw, run_id=run_id, project_required=True)
    except (ConfigError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _authorize_model_policy(spec)
    return spec


def _authorize_model_policy(spec: JobSpec) -> None:
    """Reject an open-model run unless THIS plane is self-hosted.

    The parser accepts ``model_policy`` from any caller because it also runs client-side, where
    ``FLASH_STANDALONE`` is invisible. This is the authorization half, and it runs only on the
    control plane: a managed run trains curated models on Freesolo's billing, so `allow` -- which
    accepts any HuggingFace model -- must not be reachable by asking for it in a config.
    """
    if spec.model_policy != "allow" or auth.standalone():
        return
    raise HTTPException(
        status_code=403,
        detail=(
            'model_policy = "allow" is available on self-hosted control planes only. This managed '
            "plane trains the curated catalog; choose a listed model (`flash models list`)."
        ),
    )


def _runtime_secrets(payload: dict, spec: JobSpec) -> dict[str, str]:
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
    missing = sorted(set(spec.environment.secrets) - set(out))
    if missing:
        raise HTTPException(
            status_code=400,
            detail=(
                f"missing runtime secret(s) required by [environment] secrets: {', '.join(missing)}"
            ),
        )
    return out
