import pytest

from flash.providers.realized import realized_cost_for_remote


def test_modal_estimate_uses_resource_rate_and_actual_end():
    result = realized_cost_for_remote(
        {"provider": "modal", "instance_id": "sb-test", "started_ts": 100, "hourly_usd": 3.6},
        start=100,
        end=3800,
        run_end=3700,
    )
    assert result.realized_usd == pytest.approx(3.6)
    assert result.source["basis"] == "reserved_resource_estimate"


@pytest.mark.parametrize("remote", [None, {}, {"provider": "unknown"}, {"provider": "modal"}])
def test_incomplete_identity_has_no_cost(remote):
    assert realized_cost_for_remote(remote, start=1, end=2) is None
