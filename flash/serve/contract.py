"""Dependency-light values shared by serving clients and generated backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

IMMUTABLE_ADAPTER_REVISIONS_CAPABILITY = "immutable_adapter_revisions"
ALIAS_COMPARE_AND_SWAP_CAPABILITY = "alias_compare_and_swap"
REVISION_PROVENANCE_CAPABILITY = "revision_provenance"
THINKING_STRUCTURED_OUTPUTS_CAPABILITY = "thinking_structured_outputs_deferred_v1"

REQUIRED_SERVING_CAPABILITIES = frozenset(
    {
        IMMUTABLE_ADAPTER_REVISIONS_CAPABILITY,
        ALIAS_COMPARE_AND_SWAP_CAPABILITY,
    }
)
PREFERRED_SERVING_CAPABILITIES = frozenset({REVISION_PROVENANCE_CAPABILITY})
ADAPTER_REVISION_PATTERN = (
    r"(?P<run_id>[A-Za-z0-9][A-Za-z0-9._-]{0,127})@"
    r"(?:final|step-(?P<step>0|[1-9]\d{0,17}))\."
    r"(?P<hf_revision>[0-9a-f]{40})"
)

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
