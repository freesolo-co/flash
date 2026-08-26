from __future__ import annotations

import pytest
from fastapi import HTTPException

from flash.serve.contract.provenance import (
    CheckpointProvenance,
    decode_flash_body,
    decode_freesolo_headers,
    validate_body_provenance,
)
from flash.server.routes.serving_revisions import _authorized_chat_checkpoint

RUN_ID = "flash-1234567890-abcdef12"
CHECKPOINT_ID = f"{RUN_ID}/final"


def _ready_deployment() -> dict[str, str]:
    return {"state": "ready", "checkpoint_id": CHECKPOINT_ID, "openai_model": CHECKPOINT_ID}


def test_managed_chat_authorizes_one_explicit_verified_checkpoint() -> None:
    assert (
        _authorized_chat_checkpoint(
            RUN_ID,
            _ready_deployment(),
            CHECKPOINT_ID,
            {CHECKPOINT_ID},
        )
        == CHECKPOINT_ID
    )


@pytest.mark.parametrize(
    ("checkpoint_id", "detail"),
    [
        (None, "checkpoint_id must"),
        (RUN_ID, "checkpoint_id must"),
        (RUN_ID + "@final." + "a" * 40, "checkpoint_id must"),
        ("other-run/final", "belongs to run"),
    ],
)
def test_managed_chat_rejects_missing_bare_composite_and_cross_run_targets(
    checkpoint_id: object, detail: str
) -> None:
    with pytest.raises(HTTPException, match=detail):
        _authorized_chat_checkpoint(
            RUN_ID,
            _ready_deployment(),
            checkpoint_id,
            {CHECKPOINT_ID},
        )


def test_managed_chat_rejects_unverified_checkpoint() -> None:
    with pytest.raises(HTTPException, match="has not passed"):
        _authorized_chat_checkpoint(
            RUN_ID,
            _ready_deployment(),
            CHECKPOINT_ID,
            set(),
        )


def test_managed_chat_rejects_checkpoint_other_than_ready_deployment_record() -> None:
    sibling = f"{RUN_ID}/step-20"
    with pytest.raises(HTTPException, match="not the active managed deployment record"):
        _authorized_chat_checkpoint(
            RUN_ID,
            _ready_deployment(),
            sibling,
            {CHECKPOINT_ID, sibling},
        )


def test_managed_provenance_contains_only_checkpoint_identity() -> None:
    provenance = CheckpointProvenance(CHECKPOINT_ID)
    assert provenance.freesolo_body() == {"checkpoint_id": CHECKPOINT_ID}
    assert provenance.freesolo_headers() == {"X-Freesolo-Checkpoint": CHECKPOINT_ID}
    assert decode_flash_body({"checkpoint_id": CHECKPOINT_ID}) == provenance
    assert decode_freesolo_headers({"x-freesolo-checkpoint": CHECKPOINT_ID}) == provenance

    response = validate_body_provenance(
        {"flash_provenance": {"checkpoint_id": CHECKPOINT_ID}},
        provenance,
    )
    assert response["freesolo"] == {"checkpoint_id": CHECKPOINT_ID}
    assert "artifact_revision" not in str(response)
    assert "hf_revision" not in str(response)
