"""Map an engine dispatch failure onto a client-meaningful HTTP status.

Split out of router.py's app builder. The only app state these need is the adapter router, so it
is passed in rather than captured, which keeps the mapping testable without building the app.
"""

from collections.abc import AsyncIterator
from typing import Any

from fastapi import HTTPException, status

from flash.serving.src.http.routing import AdapterRouter


def value_error_http(router: AdapterRouter, adapter_id: str, exc: ValueError) -> HTTPException:
    message = str(exc)
    if message.startswith("checkpoint mismatch: "):
        return HTTPException(
            status.HTTP_409_CONFLICT, message.removeprefix("checkpoint mismatch: ")
        )
    if message.startswith("Unknown adapter id"):
        return HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown adapter id: {adapter_id}")
    return HTTPException(status.HTTP_400_BAD_REQUEST, message)


def engine_error_http(
    router: AdapterRouter, adapter_id: str, exc: Exception
) -> HTTPException | None:
    """Map an engine dispatch failure to a client-meaningful status.

    Returns ``None`` when ``exc`` is not a failure this layer can identify, so the caller
    re-raises it unchanged. That distinction matters: rewriting every exception into a 502
    would mask router-side defects (a ``TypeError`` in the pool wrapper) and would downgrade
    an ``HTTPException`` that already carries a deliberate status, such as the authorizer's
    403, into a bare upstream failure.

    Engine work crosses a Modal RPC boundary, so an identifiable failure arrives in one of two
    shapes. A vLLM/engine ``ValueError`` (including ``VLLMValidationError``, which subclasses
    it) unpickles with its type intact and keeps its 400/404/409 meaning. Modal's own
    infrastructure errors -- ``FunctionTimeoutError`` when an input outlives the function
    timeout, ``RemoteError``/``ExecutionError`` when the remote exception cannot be
    deserialized -- subclass ``modal.exception.Error``, NOT ``ValueError``, so before this they
    escaped the sole ``except ValueError`` and surfaced as unhandled 500s, blaming the server
    for what is usually a caller-side condition (an oversized prompt).

    ``modal`` is imported lazily so this module keeps the no-modal-at-import-scope contract in
    router.py's docstring, which is what lets the routing layer be unit-tested against a fake pool.
    """
    if isinstance(exc, ValueError):
        return value_error_http(router, adapter_id, exc)
    from modal.exception import Error as ModalError
    from modal.exception import FunctionTimeoutError

    if isinstance(exc, FunctionTimeoutError):
        # checked before the ModalError branch below, which it subclasses. 504, not 500: the
        # request was well-formed and the upstream engine simply did not finish inside its
        # per-input timeout.
        return HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT,
            "The engine did not complete this request before its timeout. Retry with a "
            "shorter prompt or a smaller max_tokens.",
        )
    if isinstance(exc, ModalError):
        return HTTPException(
            status.HTTP_502_BAD_GATEWAY, "The serving engine failed to handle this request."
        )
    return None


def raise_if_engine_error(router: AdapterRouter, adapter_id: str, exc: Exception) -> None:
    """Re-raise ``exc`` as its mapped ``HTTPException``, or unchanged if it is not mapped."""
    failure = engine_error_http(router, adapter_id, exc)
    if failure is None:
        raise exc
    raise failure from exc


async def terminating_on_engine_error(
    router: AdapterRouter, events: AsyncIterator[dict[str, Any]], adapter_id: str
) -> AsyncIterator[dict[str, Any]]:
    """Convert a mid-stream engine failure into a terminal ``error`` event.

    Only advancing ``events`` is guarded: an exception raised by the *consumer* while rendering
    an event propagates through this generator untouched, so a router-side defect stays an
    unhandled server error instead of being reported to the caller as an upstream engine
    failure. That is why this is a wrapper rather than a ``try`` around the consuming loop.

    An unidentified failure is re-raised for the same reason. The caller then gets a truncated
    stream, as it did before this existed -- worse than a clean termination, but diagnosable,
    whereas a fabricated engine error would point every investigation at the wrong subsystem.

    ``CancelledError``/``GeneratorExit`` are ``BaseException``, so a client disconnect or a
    shutdown keeps unwinding rather than being mistaken for an engine failure.
    """
    try:
        async for event in events:
            yield event
    except Exception as exc:
        failure = engine_error_http(router, adapter_id, exc)
        if failure is None:
            raise
        yield {"type": "error", "message": str(failure.detail), "code": failure.status_code}
