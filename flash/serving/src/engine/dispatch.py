"""Internal pre-header dispatch deadlines for hosted generation."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Mapping
from typing import Any

PRE_HEADER_DISPATCH_TIMEOUT_SECONDS = 120.0
CAPACITY_RETRY_AFTER_SECONDS = 1


class PreHeaderDispatchExpired(RuntimeError):
    """The request waited too long to begin gpu work."""


def new_pre_header_dispatch_deadline(*, clock: Callable[[], float] | None = None) -> float:
    return (clock or time.time)() + PRE_HEADER_DISPATCH_TIMEOUT_SECONDS


def require_pre_header_dispatch_time(
    deadline: float | None,
    *,
    clock: Callable[[], float] | None = None,
) -> None:
    if deadline is None:
        return
    if not isinstance(deadline, (int, float)) or isinstance(deadline, bool):
        raise ValueError("pre-header dispatch deadline must be a finite timestamp")
    normalized = float(deadline)
    if not math.isfinite(normalized):
        raise ValueError("pre-header dispatch deadline must be a finite timestamp")
    if (clock or time.time)() >= normalized:
        raise PreHeaderDispatchExpired("request expired before gpu generation began")


_PROTOCOL_VERSION = 1
_ACK_FIELDS = frozenset(
    {
        "protocol_version",
        "kind",
        "generation_id",
        "invocation_nonce",
        "function_call_id",
    }
)


class AdmissionProtocolError(RuntimeError):
    """the remote admission acknowledgement was not exact and trustworthy."""


def admission_acknowledgement(
    *,
    generation_id: str,
    invocation_nonce: str,
    function_call_id: str,
) -> dict[str, Any]:
    return {
        "protocol_version": _PROTOCOL_VERSION,
        "kind": "admitted",
        "generation_id": generation_id,
        "invocation_nonce": invocation_nonce,
        "function_call_id": function_call_id,
    }


def validate_admission_acknowledgement(
    value: Any,
    *,
    generation_id: str,
    invocation_nonce: str,
    function_call_id: str,
) -> None:
    if not isinstance(value, Mapping) or frozenset(value) != _ACK_FIELDS:
        raise AdmissionProtocolError("invalid non-streaming admission acknowledgement")
    expected = admission_acknowledgement(
        generation_id=generation_id,
        invocation_nonce=invocation_nonce,
        function_call_id=function_call_id,
    )
    if dict(value) != expected:
        raise AdmissionProtocolError("mismatched non-streaming admission acknowledgement")


async def publish_admission_acknowledgement(
    queue_id: str,
    *,
    generation_id: str,
    invocation_nonce: str,
    deadline: float,
) -> None:
    import modal

    function_call_id = modal.current_function_call_id()
    if not isinstance(function_call_id, str) or not function_call_id:
        raise AdmissionProtocolError("function call id is unavailable for admission")
    remaining = deadline - time.time()
    if remaining <= 0:
        raise PreHeaderDispatchExpired("request expired before gpu generation began")
    queue = modal.Queue.from_id(queue_id)
    try:
        await asyncio.wait_for(
            queue.put.aio(
                admission_acknowledgement(
                    generation_id=generation_id,
                    invocation_nonce=invocation_nonce,
                    function_call_id=function_call_id,
                )
            ),
            timeout=remaining,
        )
    except TimeoutError as exc:
        raise PreHeaderDispatchExpired("request expired before gpu generation began") from exc
