"""strict internal hosted OpenAI generation envelope."""

from __future__ import annotations

from typing import Any

from pydantic import field_validator, model_validator

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

    @model_validator(mode="after")
    def validate_sampling(self) -> OpenAIGenerateRequest:
        validate_sampling_relationships(
            n=self.n,
            temperature=self.temperature,
            logprobs=self.logprobs,
            top_logprobs=self.top_logprobs,
        )
        return self
