"""Canonical request parsing and checkpoint header handling for serving routes."""

# do not add `from __future__ import annotations`: `_parse_generate` is annotated with the same
# pydantic body model the fastapi handlers use, and deferred annotations cause silent 422 responses.

from typing import Any

from fastapi import HTTPException, Request, status
from pydantic import ValidationError

from flash.serving.src.engine.model_config import base_models, is_supported_base_model
from flash.serving.src.io.openai_request import OpenAIGenerateRequest
from flash.serving.src.io.schemas import GenerateRequest


def _parse_generate(data: dict[str, Any]) -> GenerateRequest:
    # untyped dict body -> surface a bad shape as 422, not 500.
    try:
        return GenerateRequest.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


def _parse_openai_generate(data: dict[str, Any]) -> OpenAIGenerateRequest:
    try:
        return OpenAIGenerateRequest.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


def _assert_supported_base_model(base_model: str) -> None:
    if is_supported_base_model(base_model):
        return
    allowed = ", ".join(base_models())
    raise HTTPException(
        status.HTTP_400_BAD_REQUEST,
        f"Unsupported base model: {base_model}. Supported base models: {allowed}",
    )


def _expected_checkpoint(request: Request) -> str | None:
    value = request.headers.get("X-Freesolo-Expected-Checkpoint")
    return value.strip() if value is not None else None
