"""strict internal hosted OpenAI generation envelope."""

from __future__ import annotations

from typing import Any

from pydantic import field_validator, model_validator

from flash.serve.request.tool_calls import (
    normalize_tools,
    tools_active,
    tools_wire,
    validate_tool_control_presence,
    validate_tool_history_replay,
    validate_tool_stop_sequences,
)
from flash.serve.request.validation import MAX_SOURCE_CHARS, has_image_blocks, normalize_messages
from flash.serve.runtime.sampling import (
    validate_choice_count,
    validate_logprobs,
    validate_penalty,
    validate_sampling_relationships,
    validate_seed,
    validate_top_logprobs,
)
from flash.serving.src.io.schemas import GenerateRequest


class OpenAIGenerateRequest(GenerateRequest):
    """OpenAI-only sampling fields kept out of the raw generate contract."""

    n: int = 1
    seed: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logprobs: bool = False
    top_logprobs: int = 0
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    parallel_tool_calls: bool | None = None

    @field_validator("n", mode="before")
    @classmethod
    def validate_n(cls, value: Any) -> int:
        return validate_choice_count(value)

    @field_validator("seed", mode="before")
    @classmethod
    def validate_seed_value(cls, value: Any) -> int | None:
        return validate_seed(value)

    @field_validator("frequency_penalty", "presence_penalty", mode="before")
    @classmethod
    def validate_penalty_value(cls, value: Any, info: Any) -> float:
        return validate_penalty(value, info.field_name)

    @field_validator("logprobs", mode="before")
    @classmethod
    def validate_logprobs_value(cls, value: Any) -> bool:
        return validate_logprobs(value)

    @field_validator("top_logprobs", mode="before")
    @classmethod
    def validate_top_logprobs_value(cls, value: Any) -> int:
        return validate_top_logprobs(value)

    @model_validator(mode="before")
    @classmethod
    def validate_tools(cls, value: Any) -> Any:
        if not isinstance(value, dict) or value.get("tools") is None:
            return value
        normalized = normalize_tools(value["tools"])
        updated = dict(value)
        updated["tools"] = tools_wire(normalized)
        if updated.get("tool_choice") not in {"auto", "none"}:
            raise ValueError("tool_choice must be auto or none")
        if updated.get("parallel_tool_calls") is not True:
            raise ValueError("parallel_tool_calls must be true")
        return updated

    @model_validator(mode="after")
    def validate_sampling(self) -> OpenAIGenerateRequest:
        validate_sampling_relationships(
            n=self.n,
            temperature=self.temperature,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
        )
        validate_tool_control_presence(
            self.tools,
            self.tool_choice,
            self.parallel_tool_calls,
        )
        if self.messages is not None:
            normalize_messages(
                self.messages,
                sequence_types=list,
                sequence_error="messages must be a nonempty array of objects",
                error_type=ValueError,
                max_source_chars=MAX_SOURCE_CHARS,
            )
        normalized_tools = None if self.tools is None else normalize_tools(self.tools)
        validate_tool_history_replay(
            self.messages or (),
            normalized_tools if tools_active(self.tools, self.tool_choice) else None,
        )
        if tools_active(self.tools, self.tool_choice):
            if self.messages is None:
                raise ValueError("tools require chat messages")
            if has_image_blocks(self.messages, sequence_types=list):
                raise ValueError("tools cannot be combined with image messages")
            if self.logprobs or self.structured_outputs:
                raise ValueError("tools cannot be combined with logprobs or structured outputs")
        validate_tool_stop_sequences(
            () if self.stop is None else (self.stop,) if isinstance(self.stop, str) else self.stop,
            tools=normalized_tools,
            tool_choice=self.tool_choice,
        )
        return self
