"""Schema edge coverage verifies strict identities and empty-value validation.

Each assertion uses public Pydantic validation so failures match the API-facing contract.
"""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from flash.serving.src.schemas import (
    AdapterRecord,
    GenerateRequest,
    ImmutableRevisionMetadata,
    PersistedAdapterRecord,
)

SHA = "a" * 40
RUN_ID = "flash-1234567890-abcdef12"
REVISION_ID = f"{RUN_ID}@step-20.{SHA}"


def _persisted_revision(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "adapter_id": REVISION_ID,
        "repo_id": "org/run",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "org_id": "org-1",
        "checkpoint": f"{RUN_ID}/step-20",
        "thinking": False,
        "status": "ready",
        "updated_at": "2026-07-14T00:00:00+00:00",
        "metadata": {
            "record_type": "revision",
            "run_id": RUN_ID,
            "checkpoint_step": 20,
            "hf_revision": SHA,
        },
    }
    payload.update(overrides)
    return payload


def test_required_record_field_rejects_whitespace_only() -> None:
    with pytest.raises(ValidationError, match="value must not be empty"):
        AdapterRecord.model_validate(
            {
                "adapter_id": "   ",
                "repo_id": "org/run",
                "base_model": "Qwen/Qwen3.5-0.8B",
                "thinking": False,
            }
        )


def test_adapter_record_before_validator_returns_non_dict_input() -> None:
    with pytest.raises(ValidationError, match="valid dictionary"):
        AdapterRecord.model_validate("not a record")


def test_revision_checkpoint_step_rejects_negative_integer() -> None:
    with pytest.raises(
        ValidationError,
        match="checkpoint_step must be a non-negative integer or null",
    ):
        ImmutableRevisionMetadata.model_validate(
            {
                "record_type": "revision",
                "run_id": RUN_ID,
                "checkpoint_step": -1,
                "hf_revision": SHA,
            }
        )


def test_revision_checkpoint_step_guard_rejects_bool_directly() -> None:
    """Exercise the guard directly because public model_validate coerces bool before it runs."""
    with pytest.raises(
        ValueError,
        match="checkpoint_step must be a non-negative integer or null",
    ):
        ImmutableRevisionMetadata.validate_checkpoint_step(True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"serve_base_model": True}, "base-model records must not be persisted"),
        ({"org_id": "   "}, "persisted adapter records require org_id"),
        ({"updated_at": None}, "persisted adapter records require updated_at"),
        ({"adapter_id": "other"}, "persisted revision adapter_id does not match metadata"),
        (
            {"checkpoint": f"{RUN_ID}/step-21"},
            "persisted revision checkpoint does not match metadata",
        ),
    ],
)
def test_persisted_revision_identity_failures(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PersistedAdapterRecord.model_validate(_persisted_revision(**overrides))


def test_persisted_alias_id_must_equal_run_id() -> None:
    payload = _persisted_revision(
        adapter_id="other",
        checkpoint=None,
        metadata={
            "record_type": "alias",
            "run_id": RUN_ID,
            "alias_of": REVISION_ID,
        },
    )

    with pytest.raises(
        ValidationError,
        match=re.escape("persisted alias adapter_id must equal metadata.run_id"),
    ):
        PersistedAdapterRecord.model_validate(payload)


def test_generate_request_rejects_whitespace_adapter_id() -> None:
    with pytest.raises(ValidationError, match="adapter_id must not be empty"):
        GenerateRequest.model_validate({"adapter_id": "   ", "prompt": "hi"})
