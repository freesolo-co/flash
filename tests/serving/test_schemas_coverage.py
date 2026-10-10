"""Schema edge coverage for permanent checkpoint identities and request validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from flash.serving.src.io.schemas import (
    AdapterRecord,
    GenerateRequest,
    ImmutableCheckpointRegistration,
    PersistedAdapterRecord,
)

SHA = "a" * 40
DIGEST = "b" * 64
FINGERPRINT = "c" * 64
RUN_ID = "flash-1234567890-abcdef12"
CHECKPOINT_ID = f"{RUN_ID}/step-20"


def _registration(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "adapter_id": CHECKPOINT_ID,
        "repo_id": "org/run",
        "base_model": "Qwen/Qwen3.5-9B",
        "org_id": "org-1",
        "checkpoint": CHECKPOINT_ID,
        "checkpoint_step": 20,
        "run_id": RUN_ID,
        "artifact_revision": SHA,
        "artifact_digest": DIGEST,
        "artifact_fingerprint": FINGERPRINT,
        "lora_rank": 16,
        "thinking": False,
    }
    payload.update(overrides)
    return payload


def _persisted(**overrides: object) -> dict[str, object]:
    payload = {
        **_registration(),
        "status": "ready",
        "updated_at": "2026-07-14T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def test_required_record_field_rejects_whitespace_only() -> None:
    with pytest.raises(ValidationError, match="value must not be empty"):
        AdapterRecord.model_validate(
            {
                **_registration(),
                "adapter_id": "   ",
            }
        )


def test_adapter_record_before_validator_rejects_non_dict_input() -> None:
    with pytest.raises(ValidationError, match="valid dictionary"):
        AdapterRecord.model_validate("not a record")


@pytest.mark.parametrize("checkpoint_step", [-1, True])
def test_checkpoint_step_rejects_negative_and_boolean_values(checkpoint_step: object) -> None:
    with pytest.raises(ValidationError, match=r"checkpoint step|canonical checkpoint identity"):
        ImmutableCheckpointRegistration.model_validate(
            _registration(checkpoint_step=checkpoint_step)
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"serve_base_model": True}, "base-model records must not be persisted"),
        ({"org_id": "   "}, "checkpoint records require run_id and org_id"),
        ({"updated_at": None}, "persisted checkpoint records require updated_at"),
        ({"adapter_id": f"{RUN_ID}/step-21"}, "checkpoint identity does not match"),
        ({"checkpoint": f"{RUN_ID}/step-21"}, "checkpoint identity does not match"),
    ],
)
def test_persisted_checkpoint_identity_failures(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PersistedAdapterRecord.model_validate(_persisted(**overrides))


def test_generate_request_rejects_whitespace_adapter_id() -> None:
    with pytest.raises(ValidationError, match="value must not be empty"):
        GenerateRequest.model_validate({"adapter_id": "   ", "prompt": "hi"})


@pytest.mark.parametrize(
    "sources",
    [
        {},
        {"prompt": "   "},
        {"messages": []},
        {"prompt": "hi", "messages": []},
        {"prompt": "   ", "messages": [{"role": "user", "content": "hello"}]},
        {"prompt": "hi", "messages": [{"role": "user", "content": "hello"}]},
    ],
)
def test_generate_request_requires_exactly_one_nonempty_prompt_source(
    sources: dict[str, object],
) -> None:
    with pytest.raises(
        ValidationError,
        match="exactly one nonempty prompt or messages source is required",
    ):
        GenerateRequest.model_validate({"adapter_id": CHECKPOINT_ID, **sources})


def test_serving_readme_uses_generate_request_field_names() -> None:
    readme = Path(__file__).parents[2] / "flash" / "serving" / "app" / "README.md"
    text = readme.read_text()

    assert "`POST /generate` with `adapter_id`" in text
    assert '"max_tokens": 512' in text
    assert '"structured_outputs": {' in text
    assert "adapterId" not in text
    assert "maxTokens" not in text
    assert "structuredOutputs" not in text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_tokens", 0),
        ("max_tokens", -1),
        ("max_tokens", True),
        ("max_tokens", "1"),
        ("temperature", -0.1),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", True),
        ("temperature", "0"),
        ("top_p", 0),
        ("top_p", -0.1),
        ("top_p", 1.1),
        ("top_p", float("nan")),
        ("top_p", float("inf")),
        ("top_p", True),
        ("top_p", "1"),
    ],
)
def test_generate_request_rejects_sampling_outside_runtime_contract(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        GenerateRequest.model_validate({"adapter_id": CHECKPOINT_ID, "prompt": "hi", field: value})
