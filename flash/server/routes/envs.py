"""Environment publishing endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from flash._logging import get_logger
from flash.server._deps import require_key

logger = get_logger("flash.server.routes.envs")
router = APIRouter()


@router.post("/v1/envs")
def publish_env(payload: dict, key: Annotated[dict, Depends(require_key)]):
    # Publish a client-built Freesolo environment package to the managed
    # environment repository. Users never need direct repository credentials.
    from flash.server import envs

    # Default to "" only when the key is missing/None — pass a present-but-falsy
    # non-string (0, False, []) THROUGH so publish_package's type checks reject it with
    # the right 400, instead of `or ""` silently coercing it to a valid-looking empty string.
    _pkg = payload.get("package_b64")
    _name = payload.get("name")
    try:
        slug = envs.publish_package(
            package_b64="" if _pkg is None else _pkg,
            name="" if _name is None else _name,
            key=key,
        )
    except envs.EnvPublishError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    from flash.server.environment_registry import record_published_environment

    # Record the same normalized name passed to publish_package — "" for an unnamed upload, NOT
    # str(None) == "None", which would pollute platform metadata for unnamed uploads.
    # Best-effort metadata write: the environment is ALREADY published (GitHub is the source of
    # truth), so a backend-reporting failure — including ones the function's own guard misses, e.g. a
    # misconfigured FREESOLO_BASE_URL making urllib raise ValueError — must never turn this into a 500.
    try:
        record_published_environment(slug=slug, name="" if _name is None else _name, key=key)
    except Exception as exc:
        # exc_info=True keeps the full traceback for diagnosing unexpected failures (e.g. the
        # urllib ValueError noted above) while the publish stays non-fatal.
        logger.warning("record_published_environment failed (non-fatal): %s", exc, exc_info=True)
    return {"id": slug}


@router.delete("/v1/envs/{env_id:path}")
def delete_env(env_id: str, key: Annotated[dict, Depends(require_key)]):
    # Delete a published Freesolo environment package from the managed hub. ``env_id`` is the
    # ``namespace/name`` slug and carries a slash, so the route uses the ``:path`` converter.
    # Authorization (own-namespace for user keys, any for the internal key) lives in
    # ``delete_package`` so it can't be bypassed.
    from flash.server import envs

    # Normalize/validate ONCE here so the id used for deletion, the metadata-mirror drop, and the
    # response are the SAME canonical ``namespace/name`` — never a non-canonical variant (trailing
    # slash, padding from URL-decoding) that could delete one slug while recording/returning another.
    try:
        env_id = envs.canonical_env_id(env_id)
        deleted = envs.delete_package(slug=env_id, key=key)
    except envs.EnvPublishError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc

    # GitHub (the package store) is the source of truth and is already updated; dropping the
    # web-UI metadata mirror is best-effort and must never turn a successful delete into a 500
    # (same contract as the publish path's record_published_environment write).
    from flash.server.environment_registry import record_deleted_environment

    try:
        record_deleted_environment(slug=env_id, key=key)
    except Exception as exc:
        logger.warning("record_deleted_environment failed (non-fatal): %s", exc, exc_info=True)
    return {"id": env_id, "deleted": deleted}
