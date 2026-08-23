from __future__ import annotations

import json
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

from flash.serve.runtime.types import (
    validate_generation_max_tokens,
    validate_generation_temperature,
    validate_generation_top_p,
)
from flash.serving.src.engine.model_config import reasoning_parser_for
from flash.serving.src.io.structured_outputs import normalize_structured_outputs

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")


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


def normalize_run_id(value: str) -> str:
    cleaned = value.strip()
    if not _RUN_ID_RE.fullmatch(cleaned):
        raise ValueError(
            "run_id must be 1-96 characters, start with a letter or number, and contain only "
            "letters, numbers, '.', '_', or '-'"
        )
    if cleaned.endswith((".", "-")):
        raise ValueError("run_id must not end with '.' or '-'")
    return cleaned


class AdapterRecord(BaseModel):
    adapter_id: str
    repo_id: str
    base_model: str
    # Path within ``repo_id`` to the directory that actually holds ``adapter_config.json``.
    # AutoSLM runs upload the trained LoRA to ``<phase>/<run_id>/seed<N>/adapter`` inside the
    # run's HF repo (see autoslm worker ``hf_upload_folder(adapter_dir, "adapter")``), so the
    # served LoRA must point at that ``.../adapter`` subfolder, not the seed dir or repo root.
    # None = repo root.
    subfolder: str | None = None
    # HF repo kind for ``repo_id``. AutoSLM publishes run artifacts (including adapters) to
    # *dataset* repos, so serving an AutoSLM adapter requires ``"dataset"`` — otherwise
    # ``snapshot_download`` queries the model namespace and 404s. Plain model repos use "model".
    repo_type: Literal["model", "dataset"] = "model"
    # Owning org for this adapter (``hosted_lora_adapters.org_id``). Used to authorize external
    # chat callers: a request bearing a Freesolo API key may only reach an adapter its org owns.
    # None = unattributed (legacy rows / a registrar that didn't send it); chat auth treats an
    # unattributed adapter as not authorizable for a user key.
    # exclude=True keeps it OUT of serialized responses: even though `GET /adapters` now requires
    # the internal key, this is defense in depth so a listing never emits org ids (which would leak
    # the adapter->tenant mapping). It's still accepted on input and persisted via the explicit row
    # mapping in persistence.py.
    org_id: str | None = Field(default=None, exclude=True)
    url: str | None = None
    checkpoint: str | None = None
    private: bool = True
    # A base-model, no-LoRA serve: the engine generates against the base weights it already has
    # loaded (no adapter is downloaded or applied). Any valid API key may reach it (not gated to an
    # owner) and it is billed to the CALLING org. These records are pre-seeded into the router
    # in-memory (one per served base model) — never persisted, never org-owned. Defaults False so
    # LoRA records are unaffected.
    serve_base_model: bool = False
    # Per-adapter thinking default: the ``enable_thinking`` value the run was trained with. The
    # engine applies this value regardless of caller chat_template_kwargs, so raw OpenAI clients get
    # the think/no-think behavior the adapter was trained for.
    thinking: StrictBool
    # Per-adapter structured-outputs (guided decoding) DEFAULT: canonical StructuredOutputsParams
    # kwargs (see src/structured_outputs.py), applied by the engine when a request doesn't carry its
    # own spec. None = no default. Unlike ``thinking`` this is caller-overridable per request — a
    # per-call spec replaces it, and a per-call ``{}`` explicitly disables it for that call.
    structured_outputs: dict[str, Any] | None = Field(default=None)
    status: Literal["ready", "disabled"] = "ready"
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    # internal lifecycle token used to reject stale background unregister work.
    deployment_generation: str | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _strip_metadata_stored_fields(cls, data: Any) -> Any:
        # ``metadata.thinking`` / ``metadata.structured_outputs`` are the database representation
        # only (persistence packs the first-class fields into the metadata JSON column). API callers
        # must send the first-class fields; strip duplicate metadata copies so the in-memory model
        # has one authoritative source of truth — a stray metadata copy would otherwise be persisted
        # verbatim and resurrected as the field value on the next hydration.
        if not isinstance(data, dict):
            return data
        meta = data.get("metadata")
        if not isinstance(meta, dict) or not ({"thinking", "structured_outputs"} & meta.keys()):
            return data
        new_meta = {k: v for k, v in meta.items() if k not in ("thinking", "structured_outputs")}
        return {**data, "metadata": new_meta}

    @field_validator("structured_outputs", mode="before")
    @classmethod
    def normalize_structured_outputs_default(cls, value: Any) -> dict[str, Any] | None:
        # mode="before": accept every flexible input form (raw schema, str, bool) ahead of the
        # ``dict | None`` type check. Explicit-off ({}) collapses to None here:
        # "no default" and "default: unconstrained" are the same thing for a record, and None keeps
        # the persisted metadata free of a meaningless marker.
        # StructuredOutputsError is a ValueError subclass, so pydantic already surfaces it as a
        # field validation error (FastAPI -> 422) with the message intact — no re-raise needed.
        return normalize_structured_outputs(value) or None

    @field_validator("adapter_id", "repo_id", "base_model")
    @classmethod
    def require_non_empty(cls, value: str) -> str:
        return _require_non_empty(value)

    @field_validator("subfolder")
    @classmethod
    def validate_subfolder(cls, value: str | None) -> str | None:
        return normalize_adapter_subfolder(value)

    @model_validator(mode="after")
    def validate_thinking_structured_outputs(self) -> AdapterRecord:
        if self.thinking and self.structured_outputs is not None:
            try:
                parser = reasoning_parser_for(self.base_model)
            except ValueError:
                parser = None
            if parser is None:
                raise ValueError(
                    "structured outputs with thinking require a base model with a reasoning parser"
                )
        return self

    @field_validator("org_id")
    @classmethod
    def normalize_org_id(cls, value: str | None) -> str | None:
        # Treat a blank/whitespace-only org id as absent: persisting "   " into
        # ``hosted_lora_adapters.org_id`` would read as "attributed" yet never match a real org,
        # silently breaking auth + billing attribution. Strip; collapse empty to None.
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @property
    def is_revision(self) -> bool:
        try:
            metadata = ImmutableRevisionMetadata.model_validate(self.metadata)
        except ValueError:
            return False
        suffix = "final" if metadata.checkpoint_step is None else f"step-{metadata.checkpoint_step}"
        checkpoint = (
            metadata.run_id
            if metadata.checkpoint_step is None
            else f"{metadata.run_id}/step-{metadata.checkpoint_step}"
        )
        return (
            self.adapter_id == f"{metadata.run_id}@{suffix}.{metadata.hf_revision}"
            and self.checkpoint == checkpoint
        )

    @property
    def is_alias(self) -> bool:
        try:
            metadata = ImmutableAliasMetadata.model_validate(self.metadata)
        except ValueError:
            return False
        return self.adapter_id == metadata.run_id

    @property
    def run_id(self) -> str | None:
        value = self.metadata.get("run_id")
        return value if isinstance(value, str) and value else None

    @property
    def alias_of(self) -> str | None:
        value = self.metadata.get("alias_of")
        return value if isinstance(value, str) and value else None

    @property
    def hf_revision(self) -> str | None:
        value = self.metadata.get("hf_revision")
        return value if isinstance(value, str) and value else None

    def immutable_fingerprint(self) -> tuple[Any, ...]:
        return (
            self.adapter_id,
            self.repo_id,
            self.base_model,
            self.subfolder,
            self.repo_type,
            self.org_id,
            self.url,
            self.checkpoint,
            self.private,
            self.thinking,
            json.dumps(self.structured_outputs, sort_keys=True, separators=(",", ":")),
            json.dumps(self.metadata, sort_keys=True, separators=(",", ":")),
        )


class ImmutableRevisionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_type: Literal["revision"]
    run_id: str
    checkpoint_step: int | None
    hf_revision: str

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return normalize_run_id(value)

    @field_validator("checkpoint_step")
    @classmethod
    def validate_checkpoint_step(cls, value: int | None) -> int | None:
        if value is not None and (isinstance(value, bool) or value < 0):
            raise ValueError("checkpoint_step must be a non-negative integer or null")
        return value

    @field_validator("hf_revision")
    @classmethod
    def validate_hf_revision(cls, value: str) -> str:
        stripped = value.strip()
        if re.fullmatch(r"[0-9a-f]{40}", stripped) is None:
            raise ValueError("hf_revision must be a canonical 40-character Hub commit SHA")
        return stripped


class ImmutableAliasMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    record_type: Literal["alias"]
    run_id: str
    alias_of: str

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        return normalize_run_id(value)

    @field_validator("alias_of")
    @classmethod
    def require_non_empty_alias_of(cls, value: str) -> str:
        return _require_non_empty(value)


class ImmutableAdapterRegistration(BaseModel):
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
    metadata: ImmutableRevisionMetadata

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
    def validate_identity(self) -> ImmutableAdapterRegistration:
        suffix = (
            "final"
            if self.metadata.checkpoint_step is None
            else f"step-{self.metadata.checkpoint_step}"
        )
        expected_id = f"{self.metadata.run_id}@{suffix}.{self.metadata.hf_revision}"
        if self.adapter_id != expected_id:
            raise ValueError(
                "adapter_id must match metadata run_id, checkpoint_step, and hf_revision"
            )
        expected_checkpoint = (
            self.metadata.run_id
            if self.metadata.checkpoint_step is None
            else f"{self.metadata.run_id}/step-{self.metadata.checkpoint_step}"
        )
        if self.checkpoint != expected_checkpoint:
            raise ValueError("checkpoint must match metadata run_id and checkpoint_step")
        return self

    def to_record(self) -> AdapterRecord:
        return AdapterRecord.model_validate(
            {
                **self.model_dump(mode="json"),
                "status": "disabled",
                "metadata": self.metadata.model_dump(mode="json"),
            }
        )


class AdapterActivationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_adapter_revision: str | None


class PersistedAdapterRecord(AdapterRecord):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_persisted_identity(self) -> PersistedAdapterRecord:
        if self.serve_base_model:
            raise ValueError("base-model records must not be persisted")
        if self.org_id is None:
            raise ValueError("persisted adapter records require org_id")
        if not self.updated_at:
            raise ValueError("persisted adapter records require updated_at")
        if self.metadata.get("record_type") == "revision":
            metadata = ImmutableRevisionMetadata.model_validate(self.metadata)
            suffix = (
                "final" if metadata.checkpoint_step is None else f"step-{metadata.checkpoint_step}"
            )
            if self.adapter_id != f"{metadata.run_id}@{suffix}.{metadata.hf_revision}":
                raise ValueError("persisted revision adapter_id does not match metadata")
            checkpoint = (
                metadata.run_id
                if metadata.checkpoint_step is None
                else f"{metadata.run_id}/step-{metadata.checkpoint_step}"
            )
            if self.checkpoint != checkpoint:
                raise ValueError("persisted revision checkpoint does not match metadata")
        elif self.metadata.get("record_type") == "alias":
            metadata = ImmutableAliasMetadata.model_validate(self.metadata)
            if self.adapter_id != metadata.run_id:
                raise ValueError("persisted alias adapter_id must equal metadata.run_id")
            if self.checkpoint is not None:
                raise ValueError("persisted alias checkpoint must be null")
        else:
            raise ValueError("persisted records require revision or alias metadata")
        return self


def internal_adapter_payload(record: AdapterRecord) -> dict[str, Any]:
    return {**record.model_dump(mode="json"), "org_id": record.org_id}


class GenerateRequest(BaseModel):
    adapter_id: str
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
    # stop sequences for THIS call, forwarded to the engine. the serving contract documents `stop`
    # as an accepted field, and pydantic drops undeclared keys silently, so an undeclared `stop`
    # would be accepted and then ignored -- billing the caller for tokens it asked to stop before.
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

    @field_validator("adapter_id")
    @classmethod
    def strip_adapter_id(cls, value: str) -> str:
        # Normalize once so every entry point (/generate, /v1/chat/completions, the per-adapter
        # route) authorizes against and routes to the same adapter — a stray "  qa  " resolves to
        # "qa" rather than auth'ing one value and looking up another.
        stripped = value.strip()
        if not stripped:
            raise ValueError("adapter_id must not be empty")
        return stripped
