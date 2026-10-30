"""Owner-only connection capability for a managed custom GPU session."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from flash.custom.managed import session_token
from flash.server.platform.deps import require_key, require_owned_run

router = APIRouter()


@router.get("/v1/runs/{run_id}/custom")
def connection(run_id: str, response: Response, key: Annotated[dict, Depends(require_key)]):
    from flash.providers.modal import api
    from flash.providers.modal.handles import ModalJobHandle, require_scope
    from flash.runner.lifecycle.state import TERMINAL_STATES

    status = require_owned_run(run_id, key)
    response.headers["Cache-Control"] = "no-store"
    if status.spec.get("algorithm") != "custom":
        raise HTTPException(400, "run does not use custom training")
    if status.state in TERMINAL_STATES:
        raise HTTPException(409, "custom run ended; resume its durable checkpoint in a new run")
    if not status.remote:
        raise HTTPException(425, "custom GPU is provisioning")
    handle = ModalJobHandle.from_dict(status.remote)
    require_scope(handle)
    return {
        "url": api.custom_tunnel(handle.instance_id),
        "api_key": session_token(run_id, handle.attempt),
        "run_id": run_id,
    }
