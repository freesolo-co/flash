"""Authenticated checkpoint downloads; source storage stays behind the control plane."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from flash.runner.lifecycle.status import effective_spec_from_status
from flash.runner.results.deployments import require_checkpoint
from flash.serve.contract.errors import ServingError
from flash.serve.deployment.download import build_archive
from flash.server.platform.deps import require_key, require_owned_run
from flash.server.platform.locks import _deploy_lock
from flash.server.routes.serving_revisions import _resolve_deployable_target

router = APIRouter()


class AdapterResponse(FileResponse):
    """Own the temporary archive through completion, range responses, and disconnects."""

    def __init__(self, path, workspace, **kwargs):
        super().__init__(path, **kwargs)
        self.workspace = workspace

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.workspace.cleanup()


@router.get("/v1/runs/{run_id}/adapter", response_class=FileResponse)
def download_adapter(
    run_id: str,
    checkpoint_id: str,
    key: Annotated[dict, Depends(require_key)],
):
    require_owned_run(run_id, key)  # unauthorized requests cannot queue behind a run lock.
    try:
        require_checkpoint(run_id, checkpoint_id)
    except ValueError as error:
        raise HTTPException(400, str(error)) from None
    workspace = tempfile.TemporaryDirectory(prefix="flash-adapter-download-")
    try:
        with _deploy_lock(run_id):
            status = require_owned_run(run_id, key)
            try:
                spec = effective_spec_from_status(status)
            except ValueError as error:
                raise HTTPException(409, str(error)) from None
            _, _, prefix = _resolve_deployable_target(
                run_id,
                spec,
                status,
                checkpoint_id,
                action="download",
                enforce_state=True,
            )
            if not spec.train.hf_repo:
                raise HTTPException(409, "checkpoint artifact storage is unavailable")
            try:
                archive, digest = build_archive(
                    source_repo=spec.train.hf_repo,
                    source_subfolder=f"{prefix}/adapter",
                    run_id=run_id,
                    checkpoint_id=checkpoint_id,
                    base_model=spec.model,
                    base_model_revision=spec.model_revision,
                    directory=Path(workspace.name),
                )
            except ValueError:
                raise HTTPException(404, "checkpoint has no valid downloadable adapter") from None
            except (ServingError, OSError):
                raise HTTPException(502, "checkpoint adapter download preparation failed") from None
        return AdapterResponse(
            archive,
            workspace,
            media_type="application/zip",
            filename="adapter.zip",
            headers={
                "Cache-Control": "private, no-store",
                "X-Flash-Run-Id": run_id,
                "X-Flash-Checkpoint-Id": checkpoint_id,
                "X-Flash-Adapter-Sha256": digest,
            },
        )
    except BaseException:
        workspace.cleanup()
        raise
