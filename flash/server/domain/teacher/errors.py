"""Error and request types shared by the teacher broker and its request validator.

These live apart from both so the validation half and the dispatch half can import them without
importing each other. `teacher_broker` re-exports them, so `teacher_broker.TeacherBrokerError`
stays the name callers and tests use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flash.teacher.provider_status import (
    BROKER_TRANSIENT_STATUSES,
    DEFAULT_TRANSIENT_STATUS,
    validated_provider_status,
)


class TeacherBrokerError(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        status_code: int,
        retryable: bool = False,
        request_id: str | None = None,
        provider_status: int | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        # preserve broker statuses whose structured classification makes them retryable. all other
        # retryable failures use an unambiguous status so the signal survives a replaced body.
        self.status_code = (
            status_code
            if not retryable or status_code in BROKER_TRANSIENT_STATUSES
            else DEFAULT_TRANSIENT_STATUS
        )
        self.retryable = retryable
        self.request_id = request_id
        self.provider_status = validated_provider_status(provider_status)

    def payload(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "classification": "transient" if self.retryable else "permanent",
        }
        if self.request_id is not None:
            error["request_id"] = self.request_id
        if self.provider_status is not None:
            error["provider_status"] = self.provider_status
        return {"error": error}


@dataclass(frozen=True)
class ValidatedCompletionRequest:
    body: dict[str, Any]
    canonical_body: bytes
    score_items: int
