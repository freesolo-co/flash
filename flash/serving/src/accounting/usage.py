"""Request-scoped durable usage event construction for hosted serving."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from fastapi import Request

from flash.serving.src.accounting.usage_facts import usage_facts
from flash.serving.src.accounting.usage_outbox import (
    CapturedPrice,
    FreesoloOrgTrafficPrincipal,
    ImmutableTarget,
    RequestIdentity,
    ServingTrafficPrincipal,
    TrustedInternalTrafficPrincipal,
    UsageEvent,
    UsageStore,
)
from flash.serving.src.io.schemas import AdapterRecord

FREESOLO_PRICING_SOURCE = "freesolo_backend_catalog"
FREESOLO_PRICING_VERSION = "2026-08-27.1"
# (prompt, completion, cached prompt) usd per million tokens, one entry per active hosted model.
# these are the charged customer rates, set 5% below the lowest healthy comparable openrouter fp8
# listing for each model and token category. they are NOT cost-plus: no markup is applied, and they
# currently sit below the recorded cost basis, so do not reintroduce a multiplier here until measured
# b200 economics establish a lower truthful cost basis.
_FREESOLO_USD_PER_MTOK: dict[str, tuple[str, str, str]] = {
    "Qwen/Qwen3.5-9B": ("0.095", "0.1425", "0.0276"),
    "Qwen/Qwen3.8-27B": ("0.3325", "2.4225", "0.03325"),
    "Qwen/Qwen3.6-35B-A3B": ("0.095", "0.9025", "0.0475"),
}
_USD_PER_MTOK_DIVISOR = Decimal("1000000")


@dataclass(frozen=True)
class AuthorizedTraffic:
    principal: ServingTrafficPrincipal


@dataclass(frozen=True)
class TrustedInternalAuthorization:
    """A trusted server-to-server caller that has not yet been attributed to an organization."""

    org_id: str | None = None


InferenceAuthorization = AuthorizedTraffic | TrustedInternalAuthorization


@dataclass(frozen=True)
class UsageSession:
    store: UsageStore
    identity: RequestIdentity
    principal: ServingTrafficPrincipal
    target: ImmutableTarget
    price: CapturedPrice
    captured_at: datetime
    deployment_id: str | None
    serving_release: str | None
    attested_adapter: str | None

    def event(self, result: dict[str, Any]) -> UsageEvent:
        attested_adapter = result.get("lora_request_adapter") or self.attested_adapter
        evidence = (
            {"checkpoint_id": attested_adapter}
            if isinstance(attested_adapter, str) and attested_adapter
            else {}
        )
        return UsageEvent(
            identity=self.identity,
            principal=self.principal,
            target=self.target,
            price=self.price,
            captured_at=self.captured_at,
            deployment_id=self.deployment_id,
            serving_release=self.serving_release,
            tokenizer_identity=_optional_text(result.get("tokenizer_identity")),
            tokenizer_version=_optional_text(result.get("tokenizer_version")),
            attestation_evidence=evidence,
            facts=usage_facts(result),
        )

    async def capture(self, result: dict[str, Any]) -> None:
        if self.store.enabled:
            await self.store.capture(self.event(result))

    async def finalize(self, result: dict[str, Any]) -> None:
        if self.store.enabled:
            await self.store.finalize(self.event(result))

    async def fail(self, result: dict[str, Any], code: str) -> None:
        if self.store.enabled:
            await self.store.fail(self.event(result), code)

    def relinquish(self) -> None:
        if self.store.enabled:
            self.store.relinquish(self.identity.request_id)


def new_generation_id() -> str:
    return f"fsgen-{uuid.uuid4().hex}"


def new_request_identity(
    request: Request,
    *,
    openai_completion_id: str | None = None,
) -> RequestIdentity:
    request_id = new_generation_id()
    supplied_correlation = _bounded_header(request, "X-Correlation-ID")
    correlation_id = supplied_correlation or str(uuid.uuid4())
    return RequestIdentity(
        request_id=request_id,
        correlation_id=correlation_id,
        openai_completion_id=openai_completion_id,
    )


def build_usage_session(
    store: UsageStore,
    identity: RequestIdentity,
    principal: ServingTrafficPrincipal,
    requested: AdapterRecord,
    target: AdapterRecord,
    result: dict[str, Any],
    *,
    deployment_id: str,
    serving_release: str,
    captured_at: datetime,
) -> UsageSession:
    immutable_target = ImmutableTarget(
        public_model_id=requested.adapter_id,
        base_model=target.base_model,
        checkpoint_id=target.adapter_id if target.is_checkpoint else None,
        artifact_fingerprint=target.artifact_fingerprint,
    )
    price = freesolo_price(target.base_model)
    return UsageSession(
        store=store,
        identity=identity,
        principal=principal,
        target=immutable_target,
        price=price,
        captured_at=captured_at,
        deployment_id=deployment_id or None,
        serving_release=serving_release or None,
        attested_adapter=_optional_text(result.get("lora_request_adapter")),
    )


def freesolo_price(base_model: str) -> CapturedPrice:
    try:
        prompt_mtok, completion_mtok, cached_mtok = _FREESOLO_USD_PER_MTOK[base_model]
    except KeyError as exc:
        raise ValueError(f"no durable serving price for base model {base_model!r}") from exc

    def per_token(rate: str) -> str:
        return format(Decimal(rate) / _USD_PER_MTOK_DIVISOR, "f")

    return CapturedPrice(
        source=FREESOLO_PRICING_SOURCE,
        version=FREESOLO_PRICING_VERSION,
        snapshot={
            "prompt_token_usd": per_token(prompt_mtok),
            "cached_prompt_token_usd": per_token(cached_mtok),
            "completion_token_usd": per_token(completion_mtok),
        },
    )


def principal_for_external_org(org_id: str) -> FreesoloOrgTrafficPrincipal:
    return FreesoloOrgTrafficPrincipal(orgId=org_id)


def principal_for_trusted_internal(org_id: str) -> TrustedInternalTrafficPrincipal:
    return TrustedInternalTrafficPrincipal(orgId=org_id)


def captured_now() -> datetime:
    return datetime.now(UTC)


def _bounded_header(request: Request, name: str) -> str | None:
    value = request.headers.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped[:512] if stripped else None


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
