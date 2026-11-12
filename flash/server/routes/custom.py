"""Owner-only connection capability for a managed custom GPU session."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from flash.custom.managed import session_tls, session_token
from flash.server.platform.deps import require_key, require_owned_run

router = APIRouter()


def _modal_connection(run_id, remote):
    from flash.providers.modal import api
    from flash.providers.modal.handles import ModalJobHandle, require_scope

    handle = ModalJobHandle.from_dict(remote)
    require_scope(handle)
    return {
        "url": api.custom_tunnel(handle.instance_id),
        "api_key": session_token(run_id, handle.attempt),
        "run_id": run_id,
    }


def _spot_connection(run_id, remote):
    """A Spot VM is reached directly: its public IP, the attempt's token and pinned certificate."""
    from flash.providers.mithril import api
    from flash.providers.mithril.handles import SpotJobHandle, require_scope

    try:
        handle = SpotJobHandle.from_dict(remote)
        require_scope(handle)
    except (ValueError, RuntimeError):
        # e.g. a handle from the retired Nebius Spot provider, or other provider credentials:
        # this plane cannot reach that session, so stop the controller instead of a retried 500.
        raise HTTPException(
            409, "custom Spot attempt belongs to a retired provider or other credentials"
        ) from None
    if handle.run_id != run_id:
        raise HTTPException(409, "custom Spot attempt does not belong to this run")
    address = api.public_ip(handle.instance_id)
    if address is None:
        raise HTTPException(425, "custom Spot GPU has no public address yet")
    return {
        "url": f"https://{address}:{api.CUSTOM_SESSION_PORT}",
        "api_key": session_token(run_id, handle.attempt),
        "tls_cert": session_tls(run_id, handle.attempt)["cert"],
        "run_id": run_id,
    }


@router.get("/v1/runs/{run_id}/custom")
def connection(run_id: str, response: Response, key: Annotated[dict, Depends(require_key)]):
    from flash.runner.lifecycle.state import TERMINAL_STATES

    status = require_owned_run(run_id, key)
    response.headers["Cache-Control"] = "no-store"
    if status.spec.get("algorithm") != "custom":
        raise HTTPException(400, "run does not use custom training")
    if status.state in TERMINAL_STATES:
        raise HTTPException(409, "custom run ended; resume its durable checkpoint in a new run")
    if not status.remote:
        raise HTTPException(425, "custom GPU is provisioning")
    if status.remote.get("provider") == "spot":
        return _spot_connection(run_id, status.remote)
    return _modal_connection(run_id, status.remote)
