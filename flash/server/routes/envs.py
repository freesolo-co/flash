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
    from flash.server import envs

    # Use `if x is None` not `x or ""` so non-string falsy values reach publish_package's type checks.
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

    # Best-effort: env is already published (GitHub is source of truth); don't 500 on metadata failure.
    try:
        record_published_environment(slug=slug, name="" if _name is None else _name, key=key)
    except Exception as exc:
        logger.warning("record_published_environment failed (non-fatal): %s", exc, exc_info=True)
    return {"id": slug}
