"""Request-scoped durable usage event construction for hosted serving."""

from __future__ import annotations

import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, DecimalException
from types import MappingProxyType
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
    UsageOutboxError,
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
_DECIMAL_PRICE_RE = re.compile(r"^(0|[1-9][0-9]*)(\.[0-9]+)?$")
_FREESOLO_PRICE_KEYS = frozenset(
    {"prompt_token_usd", "cached_prompt_token_usd", "completion_token_usd"}
)
_OPENROUTER_REQUIRED_PRICE_KEYS = frozenset({"promptTokenUsd", "completionTokenUsd"})
_OPENROUTER_PRICE_KEYS = _OPENROUTER_REQUIRED_PRICE_KEYS | frozenset(
    {"cachedPromptTokenUsd", "requestUsd"}
)
_ADMISSION_RESULT = MappingProxyType(
    {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cached_tokens": 0,
        "cached_tokens_reported": False,
        "reasoning_tokens": 0,
        "thinking": False,
    }
)


@dataclass(frozen=True)
class AuthorizedTraffic:
    principal: ServingTrafficPrincipal
    openrouter_request_id: str | None = None
    openrouter_generation_id: str | None = None
    upstream_id: str | None = None


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

    async def admit(self) -> None:
        if self.store.enabled:
            await self.store.capture(self.event(dict(_ADMISSION_RESULT)))

    async def finalize(self, result: dict[str, Any]) -> None:
        if self.store.enabled:
            await self.store.finalize(self.event(result))

    async def fail(self, result: dict[str, Any], code: str) -> None:
        if self.store.enabled:
            await self.store.fail(self.event(result), code)

    async def fail_admission(self, code: str) -> None:
        await self.fail(dict(_ADMISSION_RESULT), code)

    def with_attestation(self, result: dict[str, Any]) -> UsageSession:
        attested = _optional_text(result.get("lora_request_adapter"))
        return replace(self, attested_adapter=attested or self.attested_adapter)

    def relinquish(self) -> None:
        if self.store.enabled:
            self.store.relinquish(self.identity.request_id)


def new_generation_id() -> str:
    return f"fsgen-{uuid.uuid4().hex}"


def new_request_identity(
    request: Request,
    *,
    openai_completion_id: str | None = None,
    traffic: AuthorizedTraffic | None = None,
) -> RequestIdentity:
    request_id = new_generation_id()
    supplied_correlation = _bounded_header(request, "X-Correlation-ID")
    correlation_id = supplied_correlation or str(uuid.uuid4())
    return RequestIdentity(
        request_id=request_id,
        correlation_id=correlation_id,
        openai_completion_id=openai_completion_id,
        openrouter_request_id=traffic.openrouter_request_id if traffic else None,
        openrouter_generation_id=traffic.openrouter_generation_id if traffic else None,
        upstream_id=traffic.upstream_id if traffic else None,
    )


def build_usage_session(
    store: UsageStore,
    identity: RequestIdentity,
    principal: ServingTrafficPrincipal,
    requested: AdapterRecord,
    target: AdapterRecord,
    *,
    price: CapturedPrice,
    deployment_id: str,
    serving_release: str,
    captured_at: datetime,
) -> UsageSession:
    public_model_id = (
        principal.publicModelId if principal.kind == "openrouter" else requested.adapter_id
    )
    immutable_target = ImmutableTarget(
        public_model_id=public_model_id,
        base_model=target.base_model,
        checkpoint_id=target.adapter_id if target.is_checkpoint else None,
        artifact_fingerprint=target.artifact_fingerprint,
    )
    if principal.kind == "trusted_internal" and target.org_id is not None:
        principal = TrustedInternalTrafficPrincipal(
            orgId=target.org_id,
            billingAttributionExplicit=True,
        )
    return UsageSession(
        store=store,
        identity=identity,
        principal=principal,
        target=immutable_target,
        price=price,
        captured_at=captured_at,
        deployment_id=deployment_id or None,
        serving_release=serving_release or None,
        attested_adapter=None,
    )


def capture_authoritative_price(
    principal: ServingTrafficPrincipal, target: AdapterRecord
) -> CapturedPrice:
    try:
        if principal.kind == "openrouter":
            price = CapturedPrice(
                source="openrouter_admission",
                version=principal.providerCatalogDigest,
                snapshot=principal.acceptedPriceSnapshot.model_dump(mode="json"),
            )
            _validate_price(
                price,
                required_keys=_OPENROUTER_REQUIRED_PRICE_KEYS,
                allowed_keys=_OPENROUTER_PRICE_KEYS,
                optional_none_keys=_OPENROUTER_PRICE_KEYS - _OPENROUTER_REQUIRED_PRICE_KEYS,
            )
            return replace(price, snapshot=MappingProxyType(dict(price.snapshot)))

        price = freesolo_price(target.base_model)
        _validate_price(
            price,
            required_keys=_FREESOLO_PRICE_KEYS,
            allowed_keys=_FREESOLO_PRICE_KEYS,
        )
        return price
    except (AttributeError, DecimalException, KeyError, TypeError, ValueError) as exc:
        raise UsageOutboxError("durable_serving_price_unavailable") from exc


def freesolo_price(base_model: str) -> CapturedPrice:
    try:
        prompt_mtok, completion_mtok, cached_mtok = _FREESOLO_USD_PER_MTOK[base_model]
    except KeyError as exc:
        raise ValueError(f"no durable serving price for base model {base_model!r}") from exc

    def per_token(rate: Any) -> str:
        # dev sets the launch rates below market, so the table is already the customer rate.
        # the decimal parse stays: a malformed table entry must not reach a durable price.
        return format(_price_decimal(rate) / _USD_PER_MTOK_DIVISOR, "f")

    return CapturedPrice(
        source=FREESOLO_PRICING_SOURCE,
        version=FREESOLO_PRICING_VERSION,
        snapshot=MappingProxyType(
            {
                "prompt_token_usd": per_token(prompt_mtok),
                "cached_prompt_token_usd": per_token(cached_mtok),
                "completion_token_usd": per_token(completion_mtok),
            }
        ),
    )


def _validate_price(
    price: CapturedPrice,
    *,
    required_keys: frozenset[str],
    allowed_keys: frozenset[str],
    optional_none_keys: frozenset[str] = frozenset(),
) -> None:
    if not isinstance(price.source, str) or not price.source:
        raise ValueError("price source is missing")
    if not isinstance(price.version, str) or not price.version:
        raise ValueError("price version is missing")
    if not isinstance(price.snapshot, Mapping):
        raise TypeError("price snapshot must be a mapping")
    keys = frozenset(price.snapshot)
    if not required_keys <= keys or not keys <= allowed_keys:
        raise ValueError("price snapshot keys are invalid")
    for key, raw in price.snapshot.items():
        if raw is None and key in optional_none_keys:
            continue
        _price_decimal(raw)


def _price_decimal(raw: Any) -> Decimal:
    if not isinstance(raw, str):
        raise TypeError("price must be a string")
    value = Decimal(raw)
    if not value.is_finite() or value < 0:
        raise ValueError("price must be finite and nonnegative")
    if _DECIMAL_PRICE_RE.fullmatch(raw) is None:
        raise ValueError("price must be a canonical decimal string")
    return value


def principal_for_external_org(org_id: str) -> FreesoloOrgTrafficPrincipal:
    return FreesoloOrgTrafficPrincipal(orgId=org_id)


def principal_for_trusted_internal(org_id: str | None = None) -> TrustedInternalTrafficPrincipal:
    return TrustedInternalTrafficPrincipal(
        orgId=org_id,
        billingAttributionExplicit=org_id is not None,
    )


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
