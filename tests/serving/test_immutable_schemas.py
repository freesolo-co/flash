from __future__ import annotations

import pytest
from pydantic import ValidationError

from flash.serving.src.schemas import AdapterRecord, ImmutableAdapterRegistration, PersistedAdapterRecord

SHA = "a" * 40
RUN_ID = "flash-1234567890-abcdef12"
REVISION_ID = f"{RUN_ID}@step-20.{SHA}"


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "adapter_id": REVISION_ID,
        "repo_id": "org/run",
        "base_model": "Qwen/Qwen3.5-0.8B",
        "subfolder": "checkpoints/step-20",
        "repo_type": "model",
        "org_id": "org-1",
        "url": "https://huggingface.co/org/run",
        "checkpoint": f"{RUN_ID}/step-20",
        "private": True,
        "thinking": False,
        "structured_outputs": None,
        "metadata": {
            "record_type": "revision",
            "run_id": RUN_ID,
            "checkpoint_step": 20,
            "hf_revision": SHA,
        },
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("sha", ["a" * 39, "a" * 41, "A" * 40, "g" * 40, "main"])
def test_registration_requires_canonical_full_hub_sha(sha: str) -> None:
    payload = _payload()
    payload["metadata"] = {**payload["metadata"], "hf_revision": sha}
    with pytest.raises(ValidationError, match="canonical 40-character"):
        ImmutableAdapterRegistration.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("adapter_id", f"{RUN_ID}@step-21.{SHA}", "adapter_id must match"),
        ("checkpoint", f"{RUN_ID}/step-21", "checkpoint must match"),
    ],
)
def test_registration_requires_id_and_checkpoint_consistency(
    field: str, value: str, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        ImmutableAdapterRegistration.model_validate(_payload(**{field: value}))


def test_registration_rejects_extra_fields_and_server_metadata() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ImmutableAdapterRegistration.model_validate(_payload(status="ready"))

    payload = _payload()
    payload["metadata"] = {**payload["metadata"], "lifecycle_state": "ready"}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ImmutableAdapterRegistration.model_validate(payload)


@pytest.mark.parametrize("subfolder", ["/absolute", "../escape", "safe/../escape"])
def test_subfolder_rejects_absolute_and_traversal_paths(subfolder: str) -> None:
    payload = _payload(subfolder=subfolder)
    with pytest.raises(ValidationError, match="subfolder"):
        AdapterRecord.model_validate(payload)
    with pytest.raises(ValidationError, match="subfolder"):
        ImmutableAdapterRegistration.model_validate(payload)


@pytest.mark.parametrize(
    ("subfolder", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("  checkpoints/step-20  ", "checkpoints/step-20"),
    ],
)
def test_subfolder_normalizes_shared_record_and_registration_input(
    subfolder: str | None, expected: str | None
) -> None:
    payload = _payload(subfolder=subfolder)
    assert AdapterRecord.model_validate(payload).subfolder == expected
    assert ImmutableAdapterRegistration.model_validate(payload).subfolder == expected


@pytest.mark.parametrize(
    "run_id",
    ["", ".run", "-run", "run/one", "run@one", "a" * 97, "run.", "run-"],
)
def test_run_id_uses_route_safe_deployment_namespace(run_id: str) -> None:
    payload = _payload(
        adapter_id=f"{run_id}@step-20.{SHA}",
        checkpoint=f"{run_id}/step-20",
    )
    payload["metadata"] = {**payload["metadata"], "run_id": run_id}
    with pytest.raises(ValidationError, match="run_id"):
        ImmutableAdapterRegistration.model_validate(payload)


def test_persisted_rows_reject_legacy_and_extra_metadata() -> None:
    row = {
        **_payload(),
        "status": "ready",
        "created_at": "2026-07-14T00:00:00+00:00",
        "updated_at": "2026-07-14T00:00:00+00:00",
    }
    PersistedAdapterRecord.model_validate(row)

    legacy = {**row, "metadata": {}}
    with pytest.raises(ValidationError, match="revision or alias"):
        PersistedAdapterRecord.model_validate(legacy)

    extra = {**row, "metadata": {**row["metadata"], "serving_generation": "old"}}
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PersistedAdapterRecord.model_validate(extra)


def test_persisted_alias_requires_minimal_metadata_and_null_checkpoint() -> None:
    revision = ImmutableAdapterRegistration.model_validate(_payload()).to_record()
    alias = {
        **revision.model_dump(mode="json"),
        "org_id": revision.org_id,
        "adapter_id": RUN_ID,
        "checkpoint": None,
        "metadata": {
            "record_type": "alias",
            "run_id": RUN_ID,
            "alias_of": REVISION_ID,
        },
        "updated_at": "2026-07-14T00:00:00+00:00",
    }
    PersistedAdapterRecord.model_validate(alias)
    with pytest.raises(ValidationError, match="checkpoint must be null"):
        PersistedAdapterRecord.model_validate({**alias, "checkpoint": RUN_ID})
