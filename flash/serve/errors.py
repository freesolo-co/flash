"""Exception types raised by the serving path.

These live in a leaf module with no `flash.serve` imports of their own so that the modules split
out of `flash.serve.deploy` can raise them without importing `deploy` back. `deploy` re-exports
every name here, so `from flash.serve.deploy import ServingError` keeps resolving.
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


class ActivationOutcomeUnknown(ServingError):
    """Alias activation cannot be resolved to the attempted or previous revision."""

    def __init__(self, run_id: str, attempted_revision: str, *, detail: str | None = None):
        reason = detail or "authoritative alias readback failed"
        super().__init__(
            "alias_activation_unknown: serving may have activated "
            f"{attempted_revision} for {run_id}, but {reason}; retry deployment reconciliation "
            "before treating either revision as authoritative"
        )
        self.run_id = run_id
        self.attempted_revision = attempted_revision


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
