"""A settled release can promote while older quarantine records retain their audit state."""

import asyncio
import json

import httpx
import pytest

from flash.serving.promotion.accounting_smoke_test import read_accounting
from flash.serving.promotion.evidence import parse_accounting, verify_accounting

RPC = "https://database.invalid/rest/v1/rpc/serving_promotion_accounting_snapshot"
SCOPE = {
    "p_org_id": "canary-org",
    "p_correlation_id": "fspromo-123-1",
    "p_serving_deployment_id": "123-1",
    "p_serving_release": "a" * 40,
    "p_public_model_id": "canary-model",
}
NOW = "2026-10-29T00:58:18+00:00"
HEADERS = {"apikey": "sb_secret_fixture", "Authorization": "Bearer sb_secret_fixture"}


def _snapshot(**overrides):
    return {
        "captured_at": NOW,
        "states": {"delivered": 224629, "quarantined": 239},
        "canary_settled": True,
        "expired_leases": 0,
        "oldest_undelivered_age_seconds": 84992,
        **overrides,
    }


def _read(*, snapshot=None, oldest=None, lease=None, table_error=None):
    seen = []
    snapshot = _snapshot() if snapshot is None else snapshot

    def respond(request):
        seen.append(request)
        if request.method == "POST":
            assert str(request.url) == RPC
            assert json.loads(request.content) == SCOPE
            return httpx.Response(200, json=snapshot)
        assert request.method == "GET"
        assert request.headers["Accept-Profile"] == "internal"
        assert request.url.path == "/rest/v1/serving_usage_outbox"
        params = request.url.params
        assert params["serving_deployment_id"] == "eq.123-1"
        assert params["serving_release"] == "eq." + "a" * 40
        assert params["limit"] == "1"
        # Every organization/model in this release is covered, not only the canary.
        assert "org_id" not in params
        assert "public_model_id" not in params
        if table_error:
            return httpx.Response(table_error)
        if params["select"] == "captured_at":
            assert params["state"] == "in.(pending,leased,quarantined,disputed)"
            assert params["or"] == "(quarantine_stage.is.null,quarantine_stage.neq.generation)"
            assert params["order"] == "captured_at.asc"
            return httpx.Response(200, json=[] if oldest is None else oldest)
        assert params["state"] == "eq.leased"
        assert params["order"] == "lease_expires_at.asc"
        return httpx.Response(200, json=[] if lease is None else lease)

    async def read():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await read_accounting(client, RPC, HEADERS, SCOPE)

    return asyncio.run(read()), seen


def test_historical_quarantine_does_not_roll_back_a_settled_release():
    original = _snapshot()
    assert verify_accounting(parse_accounting(original)).reason == "accounting_stalled"
    result, seen = _read(snapshot=original)
    assert verify_accounting(parse_accounting(result)).ok
    assert result["global_backlog"] == original
    assert len(seen) == 3


@pytest.mark.parametrize(
    ("oldest", "lease"),
    [
        ([{"captured_at": "2026-10-29T00:50:00+00:00"}], None),
        (None, [{"lease_expires_at": "2026-10-29T00:58:17+00:00"}]),
    ],
)
def test_current_release_stalls_still_block_promotion(oldest, lease):
    result, _ = _read(oldest=oldest, lease=lease)
    assert verify_accounting(parse_accounting(result)).reason == "accounting_stalled"


def test_unsettled_canary_is_not_hidden_by_a_clear_scoped_backlog():
    result, _ = _read(snapshot=_snapshot(canary_settled=False))
    assert verify_accounting(parse_accounting(result)).reason == "accounting_canary_unsettled"


def test_in_flight_rows_and_unexpired_leases_can_settle_within_the_budget():
    result, _ = _read(
        oldest=[{"captured_at": "2026-10-29T00:58:00+00:00"}],
        lease=[{"lease_expires_at": "2026-10-29T00:59:00+00:00"}],
    )
    assert result["oldest_undelivered_age_seconds"] == 18
    assert verify_accounting(parse_accounting(result)).ok


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_unreadable_scoped_tables_fail_closed(status):
    with pytest.raises(httpx.HTTPStatusError):
        _read(table_error=status)


@pytest.mark.parametrize(
    "rows",
    [{}, [None], [{}], [{"captured_at": None}], [{"captured_at": "2026-10-29T00:50:00"}]],
)
def test_malformed_scoped_rows_do_not_look_like_an_empty_backlog(rows):
    with pytest.raises(ValueError, match="accounting"):
        _read(oldest=rows)


@pytest.mark.parametrize("value", [None, "bad", "2026-10-29T00:58:18"])
def test_missing_or_naive_database_time_fails_closed(value):
    with pytest.raises(ValueError, match="accounting"):
        _read(snapshot=_snapshot(captured_at=value))
