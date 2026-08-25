"""Digest-only authorization for the provisional OpenRouter serving principal."""

from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from flash.serving.src.accounting.usage import AuthorizedTraffic

OPENROUTER_CURRENT_DIGEST_ENV = "OPENROUTER_INFERENCE_KEY_SHA256_CURRENT"
OPENROUTER_PREVIOUS_DIGEST_ENV = "OPENROUTER_INFERENCE_KEY_SHA256_PREVIOUS"
OPENROUTER_SETTLEMENT_ORG_ENV = "OPENROUTER_SETTLEMENT_ORG_ID"

_DIGEST_RE = re.compile(r"[0-9a-fA-F]{64}")
_DISABLED_PREVIOUS_DIGEST = bytes(32)


def _decode_digest(name: str, value: str) -> bytes:
    if _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    decoded = bytes.fromhex(value)
    if len(decoded) != 32:
        raise ValueError(f"{name} must decode to exactly 32 bytes")
    return decoded


@dataclass(frozen=True, slots=True)
class OpenRouterPrincipal:
    """The single capability granted to a matched OpenRouter credential."""

    settlement_org_id: str
    capability: ClassVar[Literal["canonical_hosted_base_chat"]] = "canonical_hosted_base_chat"

    def authorized_traffic(self) -> AuthorizedTraffic:
        from flash.serving.src.accounting.usage import (
            AuthorizedTraffic,
            principal_for_external_org,
        )

        return AuthorizedTraffic(
            principal=principal_for_external_org(self.settlement_org_id),
            credential_principal="openrouter",
        )


@dataclass(frozen=True, slots=True)
class OpenRouterAuthorization:
    """An immutable matcher that retains only configured credential digests."""

    current_digest: bytes = field(repr=False)
    previous_digest: bytes = field(repr=False)
    previous_enabled: bool
    principal: OpenRouterPrincipal

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str],
        *,
        internal_key: str | None,
    ) -> OpenRouterAuthorization | None:
        names = (
            OPENROUTER_CURRENT_DIGEST_ENV,
            OPENROUTER_PREVIOUS_DIGEST_ENV,
            OPENROUTER_SETTLEMENT_ORG_ENV,
        )
        if not any(name in environ for name in names):
            return None

        current_value = environ.get(OPENROUTER_CURRENT_DIGEST_ENV, "")
        settlement_value = environ.get(OPENROUTER_SETTLEMENT_ORG_ENV, "")
        if not current_value:
            raise ValueError(
                f"{OPENROUTER_CURRENT_DIGEST_ENV} is required when OpenRouter authorization is configured"
            )
        if not settlement_value.strip():
            raise ValueError(
                f"{OPENROUTER_SETTLEMENT_ORG_ENV} is required when OpenRouter authorization is configured"
            )

        current_digest = _decode_digest(OPENROUTER_CURRENT_DIGEST_ENV, current_value)
        previous_value = environ.get(OPENROUTER_PREVIOUS_DIGEST_ENV, "")
        previous_enabled = bool(previous_value)
        previous_digest = (
            _decode_digest(OPENROUTER_PREVIOUS_DIGEST_ENV, previous_value)
            if previous_enabled
            else _DISABLED_PREVIOUS_DIGEST
        )
        if previous_enabled and hmac.compare_digest(current_digest, previous_digest):
            raise ValueError(
                f"{OPENROUTER_CURRENT_DIGEST_ENV} and {OPENROUTER_PREVIOUS_DIGEST_ENV} must differ"
            )
        if not previous_enabled and hmac.compare_digest(current_digest, previous_digest):
            raise ValueError(
                f"{OPENROUTER_CURRENT_DIGEST_ENV} must not equal the disabled previous-slot digest"
            )

        if internal_key:
            internal_digest = hashlib.sha256(internal_key.encode("utf-8")).digest()
            if hmac.compare_digest(current_digest, internal_digest):
                raise ValueError(
                    f"{OPENROUTER_CURRENT_DIGEST_ENV} must not equal SHA-256(FREESOLO_INTERNAL_KEY)"
                )
            if previous_enabled and hmac.compare_digest(previous_digest, internal_digest):
                raise ValueError(
                    f"{OPENROUTER_PREVIOUS_DIGEST_ENV} must not equal SHA-256(FREESOLO_INTERNAL_KEY)"
                )

        return cls(
            current_digest=current_digest,
            previous_digest=previous_digest,
            previous_enabled=previous_enabled,
            principal=OpenRouterPrincipal(settlement_org_id=settlement_value.strip()),
        )

    def matches(self, token: str) -> bool:
        presented_digest = hashlib.sha256(token.encode("utf-8")).digest()
        current_match = hmac.compare_digest(presented_digest, self.current_digest)
        previous_match = hmac.compare_digest(presented_digest, self.previous_digest)
        return current_match or (self.previous_enabled and previous_match)
