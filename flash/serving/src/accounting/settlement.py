"""Provider settlement value types exchanged with the reconciliation rpcs.

These are the shapes a provider adapter produces and the reconciliation rpcs return. They carry no
outbox state and reach the database only through `DurableUsageOutbox`, so they live apart from the
worker that uses them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol


@dataclass(frozen=True)
class AuthoritativeProviderDay:
    provider: str
    usage_date: date
    source: str
    source_version: str
    attestation_evidence: Mapping[str, Any]

    def rpc_payload(self) -> dict[str, Any]:
        return {
            "p_provider": self.provider,
            "p_usage_date": self.usage_date.isoformat(),
            "p_source": self.source,
            "p_source_version": self.source_version,
            "p_attestation_evidence": dict(self.attestation_evidence),
        }


@dataclass(frozen=True)
class ReconciliationDayResult:
    reconciliation_day_id: str
    reconciliation_state: str
    status_reason: str | None
    replay: bool


@dataclass(frozen=True)
class ReconciliationResult:
    input_ordinal: int
    outbox_id: str | None
    reconciliation_status: str
    dispute_code: str | None
    replay: bool


@dataclass(frozen=True)
class ProviderSettlementRecord:
    provider: str
    usage_date: date
    source: str
    source_version: str
    provider_record_id: str
    request_id: str | None
    provider_amount_micro_usd: int | None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    evidence: Mapping[str, Any] | None = None
    attestation_evidence: Mapping[str, Any] | None = None

    def rpc_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "usage_date": self.usage_date.isoformat(),
            "source": self.source,
            "source_version": self.source_version,
            "provider_record_id": self.provider_record_id,
            "request_id": self.request_id,
            "provider_amount_micro_usd": self.provider_amount_micro_usd,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cached_tokens": self.cached_tokens,
            "evidence": dict(self.evidence or {}),
            "attestation_evidence": dict(self.attestation_evidence or {}),
        }


@dataclass(frozen=True)
class ProviderSettlementPage:
    records: tuple[ProviderSettlementRecord, ...]
    next_cursor: str | None


class ProviderSettlementAdapter(Protocol):
    provider: str
    authoritative: bool
    exact_reasoning_tokens: bool

    async def fetch_day(
        self, usage_date: date, *, cursor: str | None, limit: int
    ) -> ProviderSettlementPage: ...
