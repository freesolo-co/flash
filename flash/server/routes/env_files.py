"""Environment file browser and internal create-if-absent provisioning."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from flash.envs.package.direct_tokens import DirectTokenScanError, package_contains_direct_token
from flash.server.domain.registry import blob_envs, env_files, envs
from flash.server.platform.deps import require_key

router = APIRouter()


@router.get("/v1/env-files/{env_id:path}")
def get_files(env_id: str, key: Annotated[dict, Depends(require_key)], file: str | None = None):
    try:
        env_id = envs.canonical_env_id(env_id)
        if file is not None:
            return Response(env_files.raw(env_id, key, file), media_type="application/octet-stream")
        return env_files.tree(env_id, key)
    except envs.EnvPublishError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@router.post("/v1/env-files/{env_id:path}")
def preview_files(env_id: str, payload: dict, key: Annotated[dict, Depends(require_key)]):
    try:
        return env_files.contents(envs.canonical_env_id(env_id), key, payload.get("paths"))
    except envs.EnvPublishError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc


@router.put("/v1/env-seeds/{env_id:path}")
def seed_environment(env_id: str, payload: dict, key: Annotated[dict, Depends(require_key)]):
    # The platform provisioner owns the authenticated DB claim and target slug. End-user
    # publishes still use /v1/envs with its project association and ownership checks.
    if key.get("auth_kind") != "internal":
        raise HTTPException(status_code=403, detail="internal service key required")
    try:
        slug = envs.canonical_env_id(env_id)
        if not blob_envs.enabled():
            raise envs.EnvPublishError("Azure environment registry is not enabled", status=503)
        package = envs.validate_publish_inputs(
            package_b64=payload.get("package_b64"), name=slug.split("/")[-1]
        )
        with tempfile.TemporaryDirectory(prefix="flash-env-seed-") as tmp:
            root = Path(tmp)
            envs._safe_extract(package, root)
            if package_contains_direct_token(root):
                raise envs.EnvPublishError("environment package contains a direct access token")
            blob_envs.publish(slug, root, create_only=True)
        return {"id": slug}
    except DirectTokenScanError as exc:
        raise HTTPException(
            status_code=400, detail="environment package could not be scanned safely"
        ) from exc
    except envs.EnvPublishError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
