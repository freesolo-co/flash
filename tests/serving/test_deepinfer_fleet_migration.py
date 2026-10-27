"""Legacy migration never guesses whether pre-fence requests have finished."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flash.serving.deepinfer.app.fleet import reconcile_fleet
from flash.serving.deepinfer.app.fleet_deploy import require_fleet_migration, wait_for_fleet
from flash.serving.deepinfer.app.fleet_migration import (
    authorize_legacy_retirement,
    reserve_legacy,
)
from tests.serving.test_deepinfer_fleet import Ops, Store


def legacy():
    server = Mock()
    server.autoscaler.settings.return_value = SimpleNamespace(max_containers=4)
    server.get_url.return_value = "https://legacy.modal.direct"
    server._experimental_list_containers.return_value = [SimpleNamespace(container_id="old")]
    return server


def test_unmigrated_deployment_fails_without_modifying_existing_pool(monkeypatch):
    server = legacy()
    monkeypatch.setattr("modal.Server.from_name", lambda *args: server)
    with pytest.raises(RuntimeError, match="legacy monolithic"):
        require_fleet_migration(Store())
    assert server.mock_calls == [("get_url", (), {})]


def test_migration_reserves_full_legacy_cap_and_never_retires_it_automatically():
    store = Store(desired={"version": "new", "image_id": "im-new"})
    reserve_legacy(store, legacy(), app_id="ap-legacy")
    ops = Ops(store)
    reconcile_fleet(store, ops, now=0)
    assert "routes" not in store  # First candidate still warming: preserve old discovery.
    candidate = store["fleet"]["replicas"][-1]
    ops.loads[candidate["name"]] = 128
    for now in (5, 7201, 100000):
        result = reconcile_fleet(store, ops, now=now)
        assert result["reserved"] == 5
    assert len(store["routes"]["endpoints"]) == 1
    assert ops.stops == []
    assert ops.drains == []


@pytest.mark.parametrize(
    ("samples", "disabled"),
    [
        ({}, True),
        ({"old": {"at": 0, "active": 0}}, True),
        ({"old": {"at": 100, "active": 1}}, True),
        ({"old": {"at": 100, "active": 0}}, False),
    ],
)
def test_migration_refuses_unknown_busy_or_unfenced_legacy(samples, disabled):
    store = Store(routes={"endpoints": ["https://new.modal.direct"]})
    server = legacy()
    reserve_legacy(store, server, app_id="ap-legacy")
    with pytest.raises(RuntimeError):
        authorize_legacy_retirement(store, server, samples, now=100, ingress_disabled=disabled)
    assert store["fleet"]["replicas"][0]["phase"] == "legacy"


def test_confirmed_idle_legacy_remains_reserved_until_platform_confirms_exit():
    store = Store(routes={"endpoints": ["https://new.modal.direct"]})
    server = legacy()
    reserve_legacy(store, server, app_id="ap-legacy")
    authorize_legacy_retirement(
        store, server, {"old": {"at": 100, "active": 0}}, now=100, ingress_disabled=True
    )
    assert store["fleet"]["replicas"][0]["phase"] == "legacy_stopping"
    assert store["fleet"]["replicas"][0]["slots"] == 4


def test_rollout_timeout_never_stops_undrained_replicas():
    store = Store(fleet={"replicas": [{"version": "old", "phase": "draining"}]})
    with pytest.raises(TimeoutError, match="retains all undrained"):
        wait_for_fleet(store, "new", timeout=0)
    assert store["fleet"]["replicas"][0]["phase"] == "draining"
