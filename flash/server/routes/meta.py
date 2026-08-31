"""Meta endpoints: liveness, caller identity, and the public model catalog."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from flash import __version__
from flash.core.catalog import public_model_rows
from flash.server.platform.auth import _IDENTITY_PASSTHROUGH_FIELDS
from flash.server.platform.deps import require_key

router = APIRouter()


# NOTE: must stay `async def`. Sync (`def`) route handlers share one CapacityLimiter (default 40
# tokens); when slow sync handlers (chat->Modal, auth/billing/HF calls) saturate it under upstream
# degradation, a sync health handler gets queued and its Docker HEALTHCHECK times out -> container
# marked unhealthy -> 502, even though the process is fine.
@router.get("/v1/health")
async def health():
    return {
        "ok": True,
        "service": "flash",
        "version": __version__,
        "capabilities": [],
    }


@router.get("/v1/me")
def me(key: Annotated[dict, Depends(require_key)]):
    payload = {
        "kind": "internal" if key.get("auth_kind") == "internal" else "freesolo_api_key",
        "key_prefix": key["key_prefix"],
    }
    for field in ("email", *_IDENTITY_PASSTHROUGH_FIELDS):
        if key.get(field):
            payload[field] = key[field]
    return payload


@router.get("/v1/models")
def models(_: Annotated[dict, Depends(require_key)]):
    return {"models": public_model_rows()}
