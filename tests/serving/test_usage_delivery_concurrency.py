"""Bounded settlement must preserve leases, failure isolation and shutdown."""

import asyncio

import httpx
import pytest

from flash.serving.src.accounting.usage_outbox import DurableUsageOutbox, UsageOutboxError
from tests.serving.test_usage_outbox import _claimed_row, _outbox_settings, _QueuedClient


class SettlementClient:
    def __init__(self, *, fail_ack=False):
        self.rows = [_claimed_row(outbox_id=f"outbox-{i}") for i in range(8)]
        self.claimed = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.delivered = asyncio.Event()
        self.active = 0
        self.peak = 0
        self.acks = []
        self.rescheduled = []
        self.fail_ack = fail_ack
        self.claim_count = 0

    async def post(self, url, *, headers, json):
        status, result = 200, []
        if url.endswith("/claim_serving_usage_batch"):
            self.claim_count += 1
            assert json["p_limit"] == 8
            result = [] if self.claimed else self.rows
            self.claimed = True
        elif url.endswith("/durable"):
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.active == 8:
                self.entered.set()
            try:
                await self.release.wait()
            finally:
                self.active -= 1
            result = {
                "usageId": "usage-" + json["outboxId"],
                "ledgerId": None,
                "priceVersion": "captured-v1",
                "exactCostMicroUsd": 0,
                "billedCents": 0,
                "replay": False,
            }
        elif url.endswith("/acknowledge_serving_usage_delivered"):
            if self.fail_ack and json["p_outbox_id"] == "outbox-0":
                status = 400
            else:
                self.acks.append(json["p_outbox_id"])
                if len(self.acks) == 8:
                    self.delivered.set()
        elif url.endswith("/reschedule_serving_usage"):
            assert self.active == 0, "Delivery must end before releasing its lease"
            self.rescheduled.append(json["p_outbox_id"])
        return httpx.Response(status, json=result, request=httpx.Request("POST", url))


@pytest.mark.parametrize("shutdown_during_delivery", [False, True])
def test_bounded_batch_delivers_or_releases_every_lease(shutdown_during_delivery):
    async def run():
        client = SettlementClient()
        outbox = DurableUsageOutbox(
            _outbox_settings(), client=client, worker_id="worker", poll_seconds=60
        )
        await outbox.start()
        await asyncio.wait_for(client.entered.wait(), 1)
        assert len(outbox._active_leases) == 8
        assert client.claim_count == 1
        if not shutdown_during_delivery:
            client.release.set()
            await asyncio.wait_for(client.delivered.wait(), 1)
        await outbox.aclose()
        assert client.peak == 8
        assert client.active == 0
        assert not outbox._active_leases
        expected = {row["id"] for row in client.rows}
        assert set(client.rescheduled if shutdown_during_delivery else client.acks) == expected
        assert len(client.rescheduled) + len(client.acks) == 8

    asyncio.run(run())


def test_one_permanent_failure_does_not_orphan_other_settlements():
    async def run():
        client = SettlementClient(fail_ack=True)
        outbox = DurableUsageOutbox(
            _outbox_settings(), client=client, worker_id="worker", poll_seconds=60
        )
        await outbox.start()
        await asyncio.wait_for(client.entered.wait(), 1)
        client.release.set()
        await asyncio.wait_for(outbox._worker, 1)
        assert len(client.acks) == 7
        assert client.claim_count == 1
        with pytest.raises(UsageOutboxError, match="background_failure"):
            outbox.assert_healthy()
        with pytest.raises(UsageOutboxError, match="shutdown_failed"):
            await outbox.aclose()
        assert client.rescheduled == ["outbox-0"]
        assert client.active == 0

    asyncio.run(run())


@pytest.mark.parametrize("code", ["serving_usage_lease_lost", "serving_usage_identity_mismatch"])
def test_lease_loss_is_not_a_permanent_billing_conflict(code):
    async def run():
        client = _QueuedClient([(409, {"detail": {"code": code}}), (200, None)])
        outbox = DurableUsageOutbox(_outbox_settings(), client=client, worker_id="worker")
        row = _claimed_row()
        await outbox._deliver(row)
        assert not outbox._active_leases
        if code == "serving_usage_lease_lost":
            assert len(client.calls) == 1
            # Shutdown cannot reschedule work that now belongs to a different owner.
            await outbox.aclose()
            assert len(client.calls) == 1
        else:
            assert client.calls[1][0].endswith("/quarantine_serving_usage")
            assert client.calls[1][1]["p_reason"] == "permanent_http_409"

    asyncio.run(run())


@pytest.mark.parametrize(
    "body", [None, [], {"detail": None}, {"detail": []}, {"code": "serving_usage_lease_lost"}]
)
def test_malformed_conflict_response_does_not_bypass_quarantine(body):
    async def run():
        client = _QueuedClient([(409, body), (200, None)])
        outbox = DurableUsageOutbox(_outbox_settings(), client=client, worker_id="worker")
        await outbox._deliver(_claimed_row())
        assert client.calls[1][0].endswith("/quarantine_serving_usage")
        assert not outbox._active_leases

    asyncio.run(run())
