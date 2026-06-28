"""Meta endpoints: liveness, caller identity, and the public model catalog."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from flash import __version__
from flash.catalog import public_model_rows
from flash.server._deps import require_key

router = APIRouter()


@router.get("/v1/health")
def health():
    return {"ok": True, "service": "flash", "version": __version__}


@router.get("/v1/me")
def me(key: Annotated[dict, Depends(require_key)]):
    payload = {
        "kind": "internal" if key.get("auth_kind") == "internal" else "freesolo_api_key",
        "key_prefix": key["key_prefix"],
    }
    for field in (
        "email",
        "user_id",
        "org_id",
        "org_slug",
        "org_name",
        "api_key_id",
        "training_agent_job_id",
        "project_id",
    ):
        if key.get(field):
            payload[field] = key[field]
    return payload


@router.get("/v1/models")
def models(_: Annotated[dict, Depends(require_key)]):
    return {"models": public_model_rows()}
