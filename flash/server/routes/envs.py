"""Environment publishing endpoint."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from flash.server._deps import require_key

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
    record_published_environment(slug=slug, name="" if _name is None else _name, key=key)
    return {"id": slug}
