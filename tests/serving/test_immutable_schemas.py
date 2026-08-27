from __future__ import annotations

import pytest
from pydantic import ValidationError

from flash.serve.contract.provenance import immutable_binding_fingerprint
from flash.serving.src.io.schemas import (
    AdapterRecord,
    ImmutableCheckpointRegistration,
    PersistedAdapterRecord,
)

RUN_ID = "flash-1234567890-abcdef12"
CHECKPOINT_ID = f"{RUN_ID}/step-20"
SOURCE_REVISION = "a" * 40
ARTIFACT_DIGEST = "b" * 64


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "adapter_id": CHECKPOINT_ID,
        "repo_id": "org/run",
        "base_model": "Qwen/Qwen3.5-9B",
        "subfolder": "checkpoints/step-20",
        "repo_type": "model",
        "org_id": "org-1",
        "url": "https://huggingface.co/org/run",
        "checkpoint": CHECKPOINT_ID,
        "private": True,
        "thinking": False,
        "structured_outputs": None,
        "run_id": RUN_ID,
        "checkpoint_step": 20,
        "artifact_revision": SOURCE_REVISION,
        "artifact_digest": ARTIFACT_DIGEST,
        "lora_rank": 16,
    }
    fallback_fingerprint = immutable_binding_fingerprint(payload)
    payload.update(overrides)
    binding = payload.copy()
    if isinstance(subfolder := binding.get("subfolder"), str):
        binding["subfolder"] = subfolder.strip() or None
    try:
        fingerprint = immutable_binding_fingerprint(binding)
    except ValueError:
        fingerprint = fallback_fingerprint
    payload.setdefault("artifact_fingerprint", fingerprint)
    return payload


@pytest.mark.parametrize("sha", ["a" * 39, "a" * 41, "A" * 40, "g" * 40, "main"])
def test_registration_requires_canonical_private_source_commit(sha: str) -> None:
    with pytest.raises(ValidationError, match="canonical 40-character"):
        ImmutableCheckpointRegistration.model_validate(_payload(artifact_revision=sha))


@pytest.mark.parametrize("field", ["artifact_digest", "artifact_fingerprint"])
def test_registration_requires_canonical_private_artifact_digests(field: str) -> None:
    with pytest.raises(ValidationError, match="canonical sha-256"):
        ImmutableCheckpointRegistration.model_validate(_payload(**{field: "a" * 63}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("adapter_id", f"{RUN_ID}/step-21"),
        ("checkpoint", f"{RUN_ID}/step-21"),
        ("run_id", "other-run"),
        ("checkpoint_step", 21),
    ],
)
def test_registration_requires_one_canonical_checkpoint_identity(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match="canonical checkpoint identity"):
        ImmutableCheckpointRegistration.model_validate(_payload(**{field: value}))


def test_registration_rejects_extra_and_server_lifecycle_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ImmutableCheckpointRegistration.model_validate(_payload(status="ready"))


@pytest.mark.parametrize("subfolder", ["/absolute", "../escape", "safe/../escape"])
def test_subfolder_rejects_absolute_and_traversal_paths(subfolder: str) -> None:
    with pytest.raises(ValidationError, match="subfolder"):
        ImmutableCheckpointRegistration.model_validate(_payload(subfolder=subfolder))


@pytest.mark.parametrize(
    ("subfolder", "expected"),
    [(None, None), ("", None), ("   ", None), ("  checkpoints/step-20  ", "checkpoints/step-20")],
)
def test_subfolder_normalizes_registration_input(
    subfolder: str | None, expected: str | None
) -> None:
    registration = ImmutableCheckpointRegistration.model_validate(_payload(subfolder=subfolder))
    assert registration.subfolder == expected


@pytest.mark.parametrize(
    "run_id",
    ["", ".run", "-run", "run/one", "run@one", "a" * 97, "run.", "run-"],
)
def test_run_id_uses_route_safe_namespace(run_id: str) -> None:
    with pytest.raises(ValidationError, match=r"run_id|canonical checkpoint identity"):
        ImmutableCheckpointRegistration.model_validate(_payload(run_id=run_id))


def test_registration_builds_disabled_checkpoint_record_with_private_fields_excluded() -> None:
    record = ImmutableCheckpointRegistration.model_validate(_payload()).to_record()

    assert record.status == "disabled"
    assert record.is_checkpoint
    assert record.immutable_fingerprint()
    public = record.model_dump(mode="json")
    for field in (
        "org_id",
        "run_id",
        "checkpoint_step",
        "artifact_revision",
        "artifact_digest",
        "artifact_fingerprint",
        "lora_rank",
        "deployment_generation",
    ):
        assert field not in public


def test_persisted_rows_require_exact_checkpoint_and_timestamp() -> None:
    row = {
        **_payload(),
        "status": "ready",
        "created_at": "2026-08-26T00:00:00+00:00",
        "updated_at": "2026-08-26T00:00:00+00:00",
    }
    persisted = PersistedAdapterRecord.model_validate(row)
    assert persisted.adapter_id == CHECKPOINT_ID

    with pytest.raises(ValidationError, match="updated_at"):
        PersistedAdapterRecord.model_validate({**row, "updated_at": None})


def test_base_model_record_remains_a_separate_identity_kind() -> None:
    base = AdapterRecord.model_validate(
        {
            "adapter_id": "Qwen/Qwen3.5-9B",
            "repo_id": "Qwen/Qwen3.5-9B",
            "base_model": "Qwen/Qwen3.5-9B",
            "serve_base_model": True,
            "thinking": True,
        }
    )
    assert not base.is_checkpoint
