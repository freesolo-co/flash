"""Dependency-light values shared by serving clients and generated backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PERMANENT_CHECKPOINT_IDENTITY_CAPABILITY = "permanent_checkpoint_identity"
THINKING_STRUCTURED_OUTPUTS_CAPABILITY = "thinking_structured_outputs_deferred_v1"
# the serving backend echoes back WHICH LoRA answered a request. like revision provenance
# this is produced by the serving image, not by the run, so a client cannot make it appear
# by changing the adapter, the config, or the training.
LORA_REQUEST_ATTESTATION_CAPABILITY = "lora_request_attestation"
# 16 mib of compressed images expands below 22 mib in base64, leaving over 2 mib for json and text.
MAX_CHAT_REQUEST_BYTES = 24 * 1024 * 1024
# every spelling of a text and an image content block. request validation, tool-history detachment
# and template rendering each classify blocks, so the sets live here rather than in any of them
# importing another.
TEXT_TYPES = frozenset({"text", "input_text"})
IMAGE_TYPES = frozenset({"image_url", "input_image", "image"})

REQUIRED_SERVING_CAPABILITIES = frozenset({PERMANENT_CHECKPOINT_IDENTITY_CAPABILITY})
PREFERRED_SERVING_CAPABILITIES: frozenset[str] = frozenset()


ServingHealthErrorCode = Literal[
    "non_object",
    "capabilities_not_list",
    "capabilities_not_strings",
]


def reject_non_finite_json_constant(constant: str) -> None:
    """refuse constants such as nan and infinity that json does not define."""

    raise ValueError(f"json does not define {constant}")


class ServingHealthError(ValueError):
    """A malformed serving health response with a stable machine-readable reason."""

    def __init__(self, code: ServingHealthErrorCode):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ServingHealth:
    """The dependency-free portion of the serving ``/healthz`` contract."""

    capabilities: tuple[str, ...]
    base_models: tuple[str, ...]
    requires_key: bool | None
    ok: bool | None


def parse_serving_health(payload: object) -> ServingHealth:
    """Validate and normalize one serving ``/healthz`` response."""
    if not isinstance(payload, dict):
        raise ServingHealthError("non_object")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list):
        raise ServingHealthError("capabilities_not_list")
    if not all(isinstance(capability, str) for capability in capabilities):
        raise ServingHealthError("capabilities_not_strings")
    raw_models = payload.get("base_models")
    base_models = tuple(str(model) for model in raw_models) if isinstance(raw_models, list) else ()
    requires_key = payload.get("requires_key")
    ok = payload.get("ok")
    return ServingHealth(
        capabilities=tuple(capabilities),
        base_models=base_models,
        requires_key=requires_key if isinstance(requires_key, bool) else None,
        ok=ok if isinstance(ok, bool) else None,
    )
