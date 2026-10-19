"""The CLI and managed API expose every deployment, not a selected run summary."""

import flash.runner.lifecycle.status as run_status
from tests._helpers import deployments as fixtures
from tests.test_server_api import _bearer, _login, _make_run
from tests.test_server_api import api as api


def test_deployment_listing_and_exact_lookup_preserve_siblings(api, monkeypatch):
    import flash.serve.deployment.deploy as serving

    key = _login()
    run_id = _make_run(api, key, "done")
    targets = [f"{run_id}/final", f"{run_id}/step-20", f"{run_id}/step-40"]
    for target in targets:
        fixtures.seed_ready_checkpoint(run_id, target)
    headers = _bearer(key)
    before = api.get(f"/v1/runs/{run_id}/deploy", headers=headers).json()["deployments"]
    rows = api.get("/v1/deployments", headers=headers).json()["deployments"]
    assert {row["deployment"]["checkpoint_id"] for row in rows} == set(targets)
    for target in targets:
        response = api.get(
            f"/v1/runs/{run_id}/deploy", params={"checkpoint_id": target}, headers=headers
        )
        assert response.json()["checkpoint_id"] == target
        assert response.json()["verified_at"] == before[target]["verified_at"]
    monkeypatch.setattr(
        serving, "undeploy_adapter", lambda target, **_: {"disabled_checkpoints": [target]}
    )
    removed = api.delete(
        f"/v1/runs/{run_id}/deploy", params={"checkpoint_id": targets[1]}, headers=headers
    )
    assert removed.status_code == 200
    saved = run_status.get_status(run_id).deployments
    assert saved[targets[0]] == before[targets[0]]
    assert saved[targets[2]] == before[targets[2]]
    rows = api.get("/v1/deployments", headers=headers).json()["deployments"]
    assert {row["deployment"]["checkpoint_id"] for row in rows} == {targets[0], targets[2]}
    run = api.get(f"/v1/runs/{run_id}", headers=headers).json()
    assert "verified_checkpoints" not in run
    assert "deployment" not in run
    assert run["deployments"] == saved
