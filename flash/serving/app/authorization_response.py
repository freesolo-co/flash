"""Validate backend authorization responses before billing and dispatch."""

from typing import TYPE_CHECKING, Any

from flash.schema import parse_checkpoint_ref

if TYPE_CHECKING:
    from flash.serving.src.accounting.usage import AuthorizedTraffic


def _authorize_response(
    resp: Any, model_id: str, *, require_api_key_id: bool = False
) -> "AuthorizedTraffic":
    """Return the approved billing org and checkpoint generation, or fail closed."""
    from fastapi import HTTPException, status

    from flash.serving.src.accounting.usage import AuthorizedTraffic, principal_for_external_org

    if resp.status_code == 200:
        # Bind billing and checkpoint dispatch to the same backend authorization.
        try:
            body = resp.json()
            org_id = body.get("orgId")
            if not isinstance(org_id, str) or not org_id.strip():
                raise ValueError("missing billing organization")
            api_key_id = body.get("apiKeyId")
            if require_api_key_id or api_key_id is not None:
                from uuid import UUID

                api_key_id = str(UUID(api_key_id))
            generation = body.get("deploymentGeneration")
            if parse_checkpoint_ref(model_id) is not None and (
                not isinstance(generation, str) or not generation.strip()
            ):
                raise ValueError("missing verified checkpoint generation")
        except Exception as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "serving auth backend returned malformed response",
            ) from exc
        return AuthorizedTraffic(principal_for_external_org(org_id), generation, api_key_id)
    if resp.status_code == 401:
        # The backend returns 401 for BOTH an invalid *user* key (our authorize route, body
        # code "invalid_api_key") AND a rejected serving *machine* bearer (require_internal_token,
        # decided before the user key is even read). Only the former is the caller's fault; the
        # latter is a serving misconfiguration we must not report as the user's bad key.
        code = ""
        try:
            detail = resp.json().get("detail")
            if isinstance(detail, dict):
                code = detail.get("code") or ""
        except Exception:  # a non-JSON 401 is an internal-auth failure (below)
            code = ""
        if code == "invalid_api_key":
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid Freesolo API key")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "serving auth is misconfigured")
    # Org has no budget to serve (zero balance and no card+auto-topup). The backend gates this
    # before generation so a broke org can't serve for free; surface it as a clean 402 rather
    # than letting it fall through to the retryable 503 below.
    if resp.status_code == 402:
        raise HTTPException(
            status.HTTP_402_PAYMENT_REQUIRED,
            "insufficient balance: top up or enable auto-topup to serve this model",
        )
    # Unknown adapter (404) and org mismatch (403) both collapse to 403 so we don't leak
    # which adapters exist to an unauthorized caller.
    if resp.status_code in (403, 404):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "API key is not authorized for this adapter")
    # Any backend 5xx (500/502/503/504) or other unexpected status is a transient auth-lookup
    # infra failure (Supabase down, backend overloaded). Surface it as a RETRYABLE 503, never a
    # 502: a client/load-balancer must be free to retry rather than treat it as a permanent
    # upstream error. The cache+single-flight above is what makes this failure rare under load.
    raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "serving auth backend unavailable")
