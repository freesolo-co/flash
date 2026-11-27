"""Let a trusted internal caller pin one request to the Lambda GPU pool.

Normal traffic picks Lambda or the scale-to-zero Modal GPU fallback per request. Score batches
(flash.server.domain.serving.score_batches) must run on Lambda only: they wait for Lambda capacity
instead of booting a Modal GPU. The router reads the header during authorization and records the
pin in a context variable, so the engine pool sees it for the rest of the same request without
threading a flag through every generation layer. No modal/vllm imports.
"""

from contextvars import ContextVar

from fastapi import HTTPException, Request, status

from flash.serve.contract.protocol import LAMBDA_SERVING_PROVIDER as LAMBDA_PROVIDER
from flash.serve.contract.protocol import SERVING_PROVIDER_HEADER

_pinned_provider: ContextVar[str | None] = ContextVar("flash_serving_provider", default=None)


def pin_from_request(request: Request, *, trusted_internal: bool) -> None:
    """Record this request's provider pin; only trusted internal callers may set one."""

    raw = request.headers.get(SERVING_PROVIDER_HEADER)
    if raw is None:
        _pinned_provider.set(None)
        return
    value = raw.strip().lower()
    if not trusted_internal:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"{SERVING_PROVIDER_HEADER} is reserved for internal Flash callers",
        )
    if value != LAMBDA_PROVIDER:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{SERVING_PROVIDER_HEADER} must be {LAMBDA_PROVIDER!r}",
        )
    _pinned_provider.set(value)


def lambda_pinned() -> bool:
    return _pinned_provider.get() == LAMBDA_PROVIDER
