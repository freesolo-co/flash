"""One quota RPC per request, independent of prompt size and cached decisions."""

import json

import httpx
import pytest
from fastapi import HTTPException

from flash.serving.src.http.key_rate_limit import admit_key
from tests.serving.test_key_rate_limit import ALLOWED, DENIED, KEY
from tests.serving.test_rate_limit import ORG


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_checks_observed_usage_and_live_limits_on_every_request():
    calls = []
    responses = iter(
        [ALLOWED, DENIED, {**ALLOWED, "tpm_limit": None, "tokens_remaining": None}, ALLOWED]
    )

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=[next(responses)])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:

        async def admit():
            await admit_key(client, "https://db/rest/v1/rpc/org", ORG, KEY)

        await admit()
        with pytest.raises(HTTPException) as exc:
            await admit()
        assert exc.value.status_code == 429
        await admit()
        await admit()
    assert calls == [{"p_org_id": ORG, "p_api_key_id": KEY}] * 4


@pytest.mark.anyio
async def test_does_not_retry_ambiguous_admission_or_reuse_another_org():
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        raise httpx.ReadTimeout("reply lost", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        for org in (ORG, "another-org"):
            with pytest.raises(HTTPException) as exc:
                await admit_key(client, "https://db/rest/v1/rpc/org", org, KEY)
            assert exc.value.status_code == 503
    assert calls == [{"p_org_id": org, "p_api_key_id": KEY} for org in (ORG, "another-org")]
