"""Shared FastAPI request dependencies and spec/secret parsing for the route modules.

Only import lazily from ``create_app()`` — never at ``flash.server.asgi.app`` import time.
``owned_run`` resolves ``get_status`` via the module so test patches on ``app.get_status`` are honored.
"""

from __future__ import annotations

from fastapi import Header, HTTPException

from flash.client.runtime_secrets import DEFAULT_RUNTIME_SECRET_KEYS
from flash.core.spec import JobSpec, require_project_id
from flash.schema import ConfigError, spec_from_dict
from flash.server.asgi import app as _app
from flash.server.platform import auth, db

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
        from flash.runner.lifecycle.preparation import _status_org_id

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
    """Load a run for deployment management by its exact owner or matching internal scope.

    A STANDALONE plane resolves its single operator credential to ``auth_kind == "internal"``
    against the key-independent standalone-owner row, so the internal branch below sent the plane's
    own run owner into ``_internal_org_run``, which requires an ``X-Freesolo-Org-Id`` header. The
    CLI's deploy, poll, and undeploy calls send no such header and a standalone plane has no
    organization directory to take one from, so the exact owner got 404 from every deployment
    command.

    The org/project scoping stays mandatory on a MANAGED plane, where the internal key is the
    platform proxy rather than the user: it owns runs it submitted on a user's behalf, and
    ``test_internal_owned_run_still_requires_matching_org_for_deployment_management`` pins that
    even then it must name a matching org AND project to manage a deployment. Only standalone --
    which is single-tenant by definition and cannot distinguish organizations at all -- takes the
    exact-owner path.
    """
    if auth.standalone() and db.run_owner(run_id) == key["id"]:
        return _load_status(run_id)
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


def _require_supported_environment_form(env_raw: dict) -> None:
    """The environment id must name a source THIS plane can actually fetch.

    The two deployments have opposite, non-overlapping environment sources, so each has to reject
    the other's form:

    The managed hub is the only repo the managed service can vouch for: it is the one
    ``flash env push`` writes to, the one whose packages carry a validated project association, and
    the one whose slugs the submit route canonicalizes. A ``github:``/browser-URL ref names a repo
    that plane has no relationship with, so accepting it would train against unreviewed code under
    a Freesolo run.

    A self-hosted plane is the exact inverse: the hub is ``freesolo-co/environment-hub``, which is
    private and hardcoded with no override, and ``managed_slug_to_github_ref`` maps EVERY slug onto
    it. So a slug names the one repo such a plane provably cannot read -- not an environment it
    happens to lack. accepted, it passes submit, survives the best-effort sha preflight, and fails
    later during controller staging with a github 404 naming a repo the operator has never heard of.
    rejecting it at submit keeps that failure synchronous and explained. the explicit github forms
    remain a self-hosted plane's only environment source (a local ``path`` is already rejected above).

    Checked through ``canonical_managed_environment_slug`` rather than ``is_managed_...`` because
    the hub has four spellings that all resolve to the same private repo -- the bare slug, the
    ``github:`` ref, the browser URL, and case variants of each. It returns the slug for all of
    them, and raises for a ref that targets the hub but is malformed, which is equally unfetchable.

    Enforced here rather than in the schema because ``FLASH_STANDALONE`` lives on the server and
    the CLI cannot see it -- a client-side check would reject the self-hosted operator's own refs.
    """
    env_id = env_raw.get("id")
    if not isinstance(env_id, str) or not env_id.strip():
        return  # absent/malformed ids are the schema's error to report, with its own message
    from flash.envs.loading.adapter import (
        canonical_managed_environment_slug,
        is_managed_environment_slug,
    )

    env_id = env_id.strip()
    if auth.standalone():
        try:
            names_hub = canonical_managed_environment_slug(env_id) is not None
        except ValueError:
            # targets the hub repo but is malformed: still the unreachable repo, not the
            # operator's own, so it gets this explanation rather than a shape complaint.
            names_hub = True
        if not names_hub:
            return
        raise HTTPException(
            status_code=400,
            detail=(
                f"[environment] id {env_id!r} names Freesolo's managed environment hub "
                "(freesolo-co/environment-hub), which is private and unreadable from a "
                "self-hosted plane -- the run would fail on a rented GPU with a GitHub 404. "
                "Reference your own repository instead, as "
                "`github:OWNER/REPO@REF:path/to/environment.py`, and set GITHUB_TOKEN on this "
                "plane if it is private."
            ),
        )
    if is_managed_environment_slug(env_id):
        return
    raise HTTPException(
        status_code=400,
        detail=(
            f"[environment] id {env_id!r} is not a Freesolo environment id; the managed service "
            'runs environments from its hub only, addressed as "namespace/project/name". Publish yours '
            "with `flash env push --project <project-uuid> --name <name>` and use the returned id. "
            "Direct GitHub references are supported only on a self-hosted plane."
        ),
    )


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
    _require_supported_environment_form(env_raw)
    try:
        return spec_from_dict(spec_raw, run_id=run_id, project_required=True)
    except (ConfigError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
