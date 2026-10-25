"""Preflight hints save a read without caching permission, limits, or reservations."""

import json

import httpx
import pytest
from fastapi import HTTPException

from flash.serving.src.http.key_rate_limit import admit_key
from tests.serving.test_key_rate_limit import ALLOWED, KEY, PREFLIGHT
from tests.serving.test_rate_limit import ORG


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_hint_still_checks_changed_limits_and_removed_overrides():
    calls = []
    responses = iter(
        [
            PREFLIGHT,
            ALLOWED,
            {**ALLOWED, "allowed": False, "tokens_remaining": 0},
            {**ALLOWED, "tpm_limit": None, "tokens_remaining": None},
            PREFLIGHT,
            ALLOWED,
        ]
    )

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=[next(responses)])

    counts = []

    async def count():
        counts.append(1)
        return 300

    hints = {}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:

        async def admit():
            await admit_key(
                client, "https://db/rest/v1/rpc/org", ORG, KEY, count, token_budget_keys=hints
            )

        await admit()
        assert (ORG, KEY) in hints
        with pytest.raises(HTTPException) as exc:
            await admit()
        assert exc.value.status_code == 429
        await admit()
        assert (ORG, KEY) not in hints
        await admit()
    assert [c.get("p_tokens") for c in calls] == [None, 300, 300, 300, None, 300]
    assert len(counts) == 4


@pytest.mark.anyio
async def test_hint_does_not_retry_ambiguous_admission_or_cross_organizations():
    calls = []
    hints = {(ORG, KEY): None}

    def handle(request):
        calls.append(json.loads(request.content))
        raise httpx.ReadTimeout("reply lost", request=request)

    async def count():
        return 99

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        for org in (ORG, "another-org"):
            with pytest.raises(HTTPException) as exc:
                await admit_key(
                    client, "https://db/rest/v1/rpc/org", org, KEY, count, token_budget_keys=hints
                )
            assert exc.value.status_code == 503
    assert len(calls) == 2
    assert calls[0]["p_tokens"] == 99
    assert "p_tokens" not in calls[1]
