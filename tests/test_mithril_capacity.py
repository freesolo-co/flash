"""Quota exhaustion waits without provisioning or spending a failure retry."""

import io

import pytest

from flash.providers.mithril import capacity, queue
from flash.providers.mithril.client import MithrilApiError
from flash.providers.mithril.prices import Offer


@pytest.mark.parametrize(
    ("total", "used", "available"), [(5, 4, True), (5, 5, False), (0, 0, False)]
)
def test_project_quota_counts_instances_from_all_jobs(monkeypatch, total, used, available):
    monkeypatch.setattr(capacity, "settings", lambda: {"project_id": "proj_test"})

    def call(method, path, *, params, deadline_at):
        assert (method, path, params, deadline_at) == (
            "GET",
            "/quotas",
            {"project": "proj_test"},
            500,
        )
        return [
            {
                "project": "proj_test",
                "instance_type": "it_8gpu",
                "product_type": "Spot",
                "units": "Instances",
                "total_quantity": total,
                "used_quantity": used,
            }
        ]

    monkeypatch.setattr(capacity, "call", call)
    assert capacity.quota_available("it_8gpu", deadline_at=500) is available
    assert capacity.quota_available("it_unlisted", deadline_at=500) is True


@pytest.mark.parametrize(
    "rows", [None, [None], [{"instance_type": "it_test", "product_type": "Spot"}]]
)
def test_invalid_quotas_fail_closed(monkeypatch, rows):
    monkeypatch.setattr(capacity, "settings", lambda: {"project_id": "proj_test"})
    monkeypatch.setattr(capacity, "call", lambda *a, **kw: rows)
    with pytest.raises(MithrilApiError):
        capacity.quota_available("it_test")


@pytest.mark.parametrize("first", [False, "unavailable"])
def test_exhausted_or_unknown_quota_waits_then_rechecks_price(monkeypatch, first):
    from types import SimpleNamespace

    from flash.runner.lifecycle import status

    now = [100.0]
    states, checks = [], []
    monkeypatch.setattr(queue.time, "time", lambda: now[0])
    monkeypatch.setattr(queue.time, "sleep", lambda n: now.__setitem__(0, now[0] + n))
    monkeypatch.setattr(queue, "settings", lambda: {"region": "test"})
    monkeypatch.setattr(status, "get_status", lambda _: SimpleNamespace(state="queued"))
    monkeypatch.setattr(status, "_update", lambda _, state: states.append(state))
    offer = Offer("H200", 8, "it_8gpu", 1, 1500)
    monkeypatch.setattr(
        queue, "affordable_offers", lambda *a, **kw: checks.append(now[0]) or [offer]
    )
    answers = iter([first, True])

    def quota(*args, **kwargs):
        answer = next(answers)
        if answer == "unavailable":
            raise MithrilApiError
        return answer

    monkeypatch.setattr(queue, "quota_available", quota)
    assert (
        queue.wait_for_price("H200", "run", gpu_count=8, deadline_at=300, log=io.StringIO())
        == offer
    )
    assert checks == [100, 160]
    assert states == ["queued", "provisioning"]
