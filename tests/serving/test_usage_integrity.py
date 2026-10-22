import asyncio
import json

import httpx
import pytest

from flash.serving.src.accounting.usage_facts import usage_facts
from flash.serving.src.accounting.usage_outbox import DurableUsageOutbox, UsageOutboxError
from tests.serving.test_usage_outbox import _claimed_row, _outbox_settings, _QueuedClient


@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens"])
@pytest.mark.parametrize("value", [True, False])
def test_boolean_token_counts_are_not_billable(field, value):
    result = {"prompt_tokens": 2, "completion_tokens": 1, "thinking": False, field: value}
    with pytest.raises(UsageOutboxError):
        usage_facts(result)


@pytest.mark.parametrize("value", ["invalid", {}, 10**400, float("inf"), -1])
def test_invalid_duration_uses_usage_error(value):
    with pytest.raises(UsageOutboxError, match="usage_duration_invalid"):
        usage_facts(
            {
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "thinking": False,
                "inference_time_seconds": value,
            }
        )


@pytest.mark.parametrize("field", ["exactCostMicroUsd", "billedCents"])
@pytest.mark.parametrize("value", [True, -1, 1.5, float("inf")])
def test_invalid_settlement_amount_is_quarantined_without_stopping_worker(field, value):
    result = {
        "usageId": "usage-1",
        "ledgerId": "ledger-1",
        "priceVersion": "v1",
        "exactCostMicroUsd": 12,
        "billedCents": 1,
        "replay": False,
        field: value,
    }
    response = httpx.Response(200, content=json.dumps(result).encode())
    client = _QueuedClient([response, (200, None)])
    outbox = DurableUsageOutbox(_outbox_settings(), client=client, worker_id="worker-1")
    row = _claimed_row()
    asyncio.run(outbox._deliver(row))
    url, payload = client.calls[1]
    assert url.endswith("/rpc/quarantine_serving_usage")
    assert payload["p_reason"] == "invalid_settlement_response"
    assert outbox._active_leases == set()
