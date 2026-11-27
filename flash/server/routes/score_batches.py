"""Score batch endpoints: submit many chat requests, poll once, get every result together.

See flash.server.domain.ops.score_batches. Batches run only on Lambda and wait for capacity.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Depends, Header

from flash.server.domain.ops import score_batches
from flash.server.platform.deps import require_key

router = APIRouter()


@router.post("/v1/score-batches", status_code=202)
def submit_score_batch(
    payload: Annotated[dict, Body()],
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    # Sync handler: validation reads run status for each request, off the event loop.
    return score_batches.submit(payload, key, x_freesolo_org_id, x_freesolo_project_id)


@router.get("/v1/score-batches/{batch_id}")
def score_batch(
    batch_id: str,
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    """Progress while active; every result, in request order, once the batch is terminal."""
    row = score_batches.owned_row(batch_id, key, x_freesolo_org_id, x_freesolo_project_id)
    return score_batches.public(row, include_results=True)


@router.delete("/v1/score-batches/{batch_id}")
def cancel_score_batch(
    batch_id: str,
    key: Annotated[dict, Depends(require_key)],
    x_freesolo_org_id: Annotated[str | None, Header()] = None,
    x_freesolo_project_id: Annotated[str | None, Header()] = None,
):
    return score_batches.cancel(batch_id, key, x_freesolo_org_id, x_freesolo_project_id)
