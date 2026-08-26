from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from flash.schema import format_checkpoint_ref, parse_checkpoint_ref
from flash.serve.contract.provenance import (
    CheckpointKey,
    immutable_binding_fingerprint,
    record_key,
)
from flash.serve.runtime.types import (
    validate_generation_max_tokens,
    validate_generation_temperature,
    validate_generation_top_p,
)
from flash.serving.src.engine.model_config import reasoning_parser_for
from flash.serving.src.io.structured_outputs import normalize_structured_outputs


def _require_non_empty(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError("value must not be empty")
    return stripped


def normalize_adapter_subfolder(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    path = PurePosixPath(cleaned)
    if path.is_absolute():
        raise ValueError("subfolder must be a relative POSIX path")
    if ".." in path.parts:
        raise ValueError("subfolder must not contain '..'")
    return cleaned


class AdapterRecord(BaseModel):
    """one base model or one exact run-backed checkpoint used by hosted serving."""

    adapter_id: str
    repo_id: str
    base_model: str
    subfolder: str | None = None
    repo_type: Literal["model", "dataset"] = "model"
    org_id: str | None = Field(default=None, exclude=True)
    url: str | None = None
    checkpoint: str | None = None
    private: StrictBool = True
    serve_base_model: bool = False
    thinking: StrictBool
    structured_outputs: dict[str, Any] | None = Field(default=None)
    status: Literal["ready", "disabled"] = "ready"
    created_at: str | None = None
    updated_at: str | None = None
    deployment_generation: str | None = Field(default=None, exclude=True)
    run_id: str | None = Field(default=None, exclude=True)
    checkpoint_step: int | None = Field(default=None, exclude=True)
    artifact_revision: str | None = Field(default=None, exclude=True)
    artifact_digest: str | None = Field(default=None, exclude=True)
    artifact_fingerprint: str | None = Field(default=None, exclude=True)
    lora_rank: int | None = Field(default=None, exclude=True)

    @field_validator("structured_outputs", mode="before")
    @classmethod
    def normalize_structured_outputs_default(cls, value: Any) -> dict[str, Any] | None:
        return normalize_structured_outputs(value) or None

    @field_validator("adapter_id", "repo_id", "base_model")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("subfolder")
    @classmethod
    def validate_subfolder(cls, value: str | None) -> str | None:
        return normalize_adapter_subfolder(value)

    @field_validator("org_id")
    @classmethod
    def normalize_org_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def validate_record(self) -> AdapterRecord:
        if self.thinking and self.structured_outputs is not None:
            try:
                parser = reasoning_parser_for(self.base_model)
            except ValueError:
                parser = None
            if parser is None:
                raise ValueError(
                    "structured outputs with thinking require a base model with a reasoning parser"
                )
        if self.serve_base_model:
            return self
        if self.run_id is None or self.org_id is None:
            raise ValueError("checkpoint records require run_id and org_id")
        expected = format_checkpoint_ref(self.run_id, self.checkpoint_step)
        if self.adapter_id != expected or self.checkpoint != expected:
            raise ValueError("checkpoint identity does not match run_id and checkpoint_step")
        if (
            self.artifact_revision is None
            or re.fullmatch(r"[0-9a-f]{40}", self.artifact_revision) is None
        ):
            raise ValueError("artifact_revision must be a canonical 40-character commit sha")
        for name in ("artifact_digest", "artifact_fingerprint"):
            value = getattr(self, name)
            if value is None or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"{name} must be a canonical sha-256 digest")
        if (
            isinstance(self.lora_rank, bool)
            or not isinstance(self.lora_rank, int)
            or self.lora_rank <= 0
        ):
            raise ValueError("lora_rank must be a positive integer")
        return self

    @property
    def is_checkpoint(self) -> bool:
        return not self.serve_base_model and parse_checkpoint_ref(self.adapter_id) is not None

    @property
    def storage_key(self) -> CheckpointKey:
        return record_key(self)

    def immutable_fingerprint(self) -> str:
        return immutable_binding_fingerprint(self)


class ImmutableCheckpointRegistration(BaseModel):
    """strict internal registration input for one permanent checkpoint binding."""

    model_config = ConfigDict(extra="forbid")

    adapter_id: str
    repo_id: str
    base_model: str
    subfolder: str | None = None
    repo_type: Literal["model", "dataset"] = "model"
    org_id: str
    url: str | None = None
    checkpoint: str
    private: StrictBool = True
    thinking: StrictBool
    structured_outputs: dict[str, Any] | None = Field(default=None)
    run_id: str
    checkpoint_step: int | None
    artifact_revision: str
    artifact_digest: str
    artifact_fingerprint: str
    lora_rank: int

    @field_validator("adapter_id", "repo_id", "base_model", "org_id", "checkpoint")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("subfolder")
    @classmethod
    def validate_subfolder(cls, value: str | None) -> str | None:
        return normalize_adapter_subfolder(value)

    @field_validator("structured_outputs", mode="before")
    @classmethod
    def normalize_structured_outputs_default(cls, value: Any) -> dict[str, Any] | None:
        return normalize_structured_outputs(value) or None

    @model_validator(mode="after")
    def validate_identity(self) -> ImmutableCheckpointRegistration:
        expected = format_checkpoint_ref(self.run_id, self.checkpoint_step)
        if self.adapter_id != expected or self.checkpoint != expected:
            raise ValueError(
                "adapter_id and checkpoint must equal the canonical checkpoint identity"
            )
        if re.fullmatch(r"[0-9a-f]{40}", self.artifact_revision) is None:
            raise ValueError("artifact_revision must be a canonical 40-character commit sha")
        for name in ("artifact_digest", "artifact_fingerprint"):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError(f"{name} must be a canonical sha-256 digest")
        if self.artifact_fingerprint != immutable_binding_fingerprint(self):
            raise ValueError("artifact_fingerprint does not match the immutable binding")
        if isinstance(self.checkpoint_step, bool) or (
            self.checkpoint_step is not None and self.checkpoint_step < 0
        ):
            raise ValueError("checkpoint_step must be a non-negative integer or null")
        if isinstance(self.lora_rank, bool) or self.lora_rank <= 0:
            raise ValueError("lora_rank must be a positive integer")
        return self

    def to_record(self) -> AdapterRecord:
        return AdapterRecord.model_validate({**self.model_dump(mode="json"), "status": "disabled"})


class PersistedAdapterRecord(AdapterRecord):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_persisted_identity(self) -> PersistedAdapterRecord:
        if self.serve_base_model:
            raise ValueError("base-model records must not be persisted")
        if not self.updated_at:
            raise ValueError("persisted checkpoint records require updated_at")
        return self


def internal_adapter_payload(record: AdapterRecord) -> dict[str, Any]:
    return {
        **record.model_dump(mode="json"),
        "org_id": record.org_id,
        "deployment_generation": record.deployment_generation,
        "run_id": record.run_id,
        "checkpoint_step": record.checkpoint_step,
        "artifact_revision": record.artifact_revision,
        "artifact_digest": record.artifact_digest,
        "artifact_fingerprint": record.artifact_fingerprint,
        "lora_rank": record.lora_rank,
    }


class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter_id: str
    # the cpu front door overwrites this before every production dispatch. direct offline engine
    # tests may omit it, in which case the engine supplies a test-only id before calling vllm.
    generation_id: str | None = None
    prompt: str | None = None
    messages: list[dict[str, Any]] | None = None
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 0.95
    # Extra kwargs forwarded to the tokenizer's chat template after sanitization. ``enable_thinking``
    # is not caller-controlled for adapters with a persisted trained value; the engine overwrites it
    # from the adapter record.
    chat_template_kwargs: dict[str, Any] | None = None
    # Structured outputs (guided decoding) for THIS call, normalized to canonical
    # StructuredOutputsParams kwargs (see src/structured_outputs.py). Accepts every flexible form
    # (raw JSON schema, constraint dict, str) under "structured_outputs".
    # Post-validation: None = not specified (the engine falls back to the adapter's registered
    # default), {} = explicitly unconstrained (overrides the adapter default), non-empty dict =
    # the constraint to apply.
    structured_outputs: dict[str, Any] | None = Field(default=None)
    # stop sequences for this call, forwarded to the engine as part of the strict raw contract.
    stop: str | list[str] | None = Field(default=None)

    @field_validator("max_tokens", mode="before")
    @classmethod
    def validate_max_tokens(cls, value: Any) -> int:
        return validate_generation_max_tokens(value)

    @field_validator("temperature", mode="before")
    @classmethod
    def validate_temperature(cls, value: Any) -> float:
        return validate_generation_temperature(value)

    @field_validator("top_p", mode="before")
    @classmethod
    def validate_top_p(cls, value: Any) -> float:
        return validate_generation_top_p(value)

    @field_validator("stop", mode="before")
    @classmethod
    def normalize_stop(cls, value: Any) -> str | list[str] | None:
        # `None` is the one spelling of "no stop constraint". an empty LIST means the caller
        # supplied no sequences, so it normalizes to that. an empty STRING is different: it is a
        # sequence the caller authored that can never terminate generation, and accepting it would
        # dispatch an unconstrained run under a constraint the caller believes they set. it is
        # already refused inside a list, and the flash-owned runtime validator refuses it too, so
        # accepting the bare form was the odd one out rather than a deliberate allowance.
        if value is None:
            return None
        if isinstance(value, str):
            if not value:
                raise ValueError("stop must not be an empty string")
            return value
        if isinstance(value, list):
            if not all(isinstance(item, str) and item for item in value):
                raise ValueError("stop entries must be non-empty strings")
            return value or None
        raise ValueError("stop must be a string or a list of strings")

    @field_validator("structured_outputs", mode="before")
    @classmethod
    def normalize_structured(cls, value: Any) -> dict[str, Any] | None:
        # StructuredOutputsError is a ValueError subclass, so pydantic surfaces it as a 422 with the
        # normalizer's message intact — no manual re-raise needed.
        return normalize_structured_outputs(value)

    @model_validator(mode="after")
    def validate_prompt_source(self) -> GenerateRequest:
        # an omitted source and a present-but-empty source are both "no source": messages=[]
        # must fail here rather than reaching the engine as a zero-turn conversation.
        invalid_prompt = self.prompt is not None and not self.prompt.strip()
        invalid_messages = self.messages is not None and not self.messages
        has_prompt = self.prompt is not None
        has_messages = self.messages is not None
        if invalid_prompt or invalid_messages or has_prompt == has_messages:
            raise ValueError("exactly one nonempty prompt or messages source is required")
        return self

    @field_validator("generation_id")
    @classmethod
    def validate_generation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped or len(stripped) > 512:
            raise ValueError("generation_id must be 1-512 characters")
        return stripped

    @field_validator("adapter_id")
    @classmethod
    def strip_adapter_id(cls, value: str) -> str:
        # normalize once so every entry point authorizes and routes the same nonempty selector.
        # checkpoint-only control and managed-run boundaries enforce permanent checkpoint grammar;
        # the serving router distinguishes those records from supported base-model records.
        return _require_non_empty(value)
