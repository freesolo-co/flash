"""errors raised by the policy-neutral serving runtime."""

from __future__ import annotations


class ServingRuntimeError(RuntimeError):
    """base error for serving runtime failures."""


class RuntimeConfigurationError(ServingRuntimeError, ValueError):
    """invalid runtime configuration."""


class RuntimeNotReadyError(ServingRuntimeError):
    """the runtime could not provide a live engine."""


class EngineDeadError(RuntimeNotReadyError):
    """the vllm engine core has died and cannot recover in process."""


class AdapterError(ServingRuntimeError):
    """base error for adapter lifecycle failures."""


class AdapterNotFoundError(AdapterError, LookupError):
    """the requested adapter is not registered."""


class AdapterConflictError(AdapterError):
    """one incarnation token was reused for different adapter state."""


class StaleIncarnationError(AdapterError):
    """an operation targeted an adapter incarnation that is no longer current."""


class AdapterPathError(AdapterError, ValueError):
    """a local adapter directory is incomplete or invalid."""


class PromptError(ServingRuntimeError, ValueError):
    """a generation prompt is invalid for this runtime."""


class MultimodalRequestError(PromptError):
    """a multimodal request failed bounded validation."""


class StructuredOutputsError(PromptError):
    """a structured-output specification is invalid."""
