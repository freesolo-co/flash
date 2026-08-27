"""Exception types raised by the serving path.

These live in a leaf module with no `flash.serve` imports of their own so deployment and request
modules can raise them without importing the deployment orchestrator back. Every caller imports
these errors from this canonical owner.
"""

from __future__ import annotations


class ServingError(RuntimeError):
    """Serving backend rejected a request or was unreachable; carries the upstream status."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class RetryableServingUnavailable(ServingError):
    """a recognized serving cold-start envelope that may be retried within a caller deadline."""

    def __init__(self, code: str, retry_after_seconds: float):
        super().__init__(
            f"serving_retryable_unavailable: {code}",
            status_code=503,
        )
        self.code = code
        self.retry_after_seconds = retry_after_seconds


class AdapterConfigMissing(ServingError):
    """The adapter's adapter_config.json could not be read from HF (artifact likely absent)."""


class AdapterTensorMissing(ServingError):
    """The adapter artifact has metadata but no loadable LoRA tensor file."""
