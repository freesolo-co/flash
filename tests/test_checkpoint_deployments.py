"""Per-checkpoint authority survives independent lifecycle changes and stale completions."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from flash.runner.lifecycle import state, status
from flash.runner.results.deployments import is_ready
from flash.runner.supervise import transitions


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "RUNS_DIR", str(tmp_path))
    state._save_status(state.RunStatus(run_id="run-a", state="done", spec={}))
    return "run-a"


def queued(run, step, generation=1):
    return {
        "checkpoint_id": f"{run}/step-{step}",
        "checkpoint_step": step,
        "state": "queued",
        "generation": generation,
        "requested_at": float(generation),
    }


def ready(run, step):
    record = queued(run, step)
    transitions.mark_deployment_pending(run, record)
    record = {
        **record,
        "state": "ready",
        "verified_at": 123.0,
        "deployment_generation": "generation-a",
        "verify_sample": str(step),
    }
    return transitions.mark_checkpoint_deployed(run, record).deployments[record["checkpoint_id"]]


def test_exact_removal_retains_all_sibling_metadata(run):
    first, second = ready(run, 20), ready(run, 40)
    updated = transitions.mark_undeployed(run, first["checkpoint_id"])
    assert updated.deployments[second["checkpoint_id"]] == second
    assert updated.deployments[first["checkpoint_id"]]["state"] == "undeployed"
    assert not is_ready(run, updated.deployments[first["checkpoint_id"]])
    assert updated.state == "deployed"
    assert transitions.mark_undeployed(run, second["checkpoint_id"]).state == "done"


def test_failed_sibling_does_not_replace_ready_checkpoint(run):
    sibling = ready(run, 20)
    attempt = queued(run, 40)
    transitions.mark_deployment_pending(run, attempt)
    result = transitions.mark_deployment_failed(run, {**attempt, "error": "smoke failed"})
    assert result.deployments[sibling["checkpoint_id"]] == sibling
    assert result.deployments[attempt["checkpoint_id"]]["error"] == "smoke failed"


@pytest.mark.parametrize("action", ["undeploy", "revoke"])
def test_late_completion_cannot_restore_revoked_checkpoint(run, action):
    attempt = queued(run, 20)
    transitions.mark_deployment_pending(run, attempt)
    if action == "undeploy":
        fenced = transitions.mark_undeployed(run, attempt["checkpoint_id"])
    else:
        fenced = transitions.mark_deployment_cleanup_pending(
            run, "pending", attempt["checkpoint_id"]
        )
    late = {
        **attempt,
        "state": "ready",
        "verified_at": 10.0,
        "deployment_generation": "generation-a",
    }
    assert transitions.mark_checkpoint_deployed(run, late).deployments == fenced.deployments
    assert transitions.mark_deployment_failed(run, attempt).deployments == fenced.deployments
    assert (
        transitions.mark_deployment_pending(run, attempt, owner_deployment=attempt).deployments
        == fenced.deployments
    )


def test_new_attempt_generation_survives_old_completion(run):
    old = queued(run, 20)
    transitions.mark_deployment_pending(run, old)
    new = queued(run, 20, 2)
    transitions.mark_deployment_pending(run, new)
    late = {**old, "state": "ready", "verified_at": 10.0, "deployment_generation": "generation-a"}
    assert transitions.mark_checkpoint_deployed(run, late).deployments[old["checkpoint_id"]] == new


def test_concurrent_sibling_writes_preserve_complete_inventory(run):
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda step: ready(run, step), range(8)))
    saved = status.get_status(run)
    assert len(saved.deployments) == 8
    assert all(is_ready(run, record) for record in saved.deployments.values())


def test_ready_without_verification_cannot_be_committed(run):
    attempt = queued(run, 20)
    transitions.mark_deployment_pending(run, attempt)
    with pytest.raises(ValueError, match="verification"):
        transitions.mark_checkpoint_deployed(run, {**attempt, "state": "ready"})
    assert not is_ready(run, status.get_status(run).deployments[attempt["checkpoint_id"]])


def test_roundtrip_is_the_only_authority_store(run, tmp_path):
    record = ready(run, 20)
    assert status.get_status(run).to_dict()["deployments"] == {record["checkpoint_id"]: record}
    assert not list(tmp_path.glob("*.verified-checkpoints*"))
    assert "deployment" not in status.get_status(run).to_dict()


@pytest.mark.parametrize(
    "records",
    [
        {"other/final": {"checkpoint_id": "other/final"}},
        {"run-a/final": {"checkpoint_id": "run-a/step-20"}},
        {"run-a/step-20": {"checkpoint_id": "run-a/step-20", "checkpoint_step": 40}},
    ],
)
def test_malformed_inventory_fails_closed(records):
    with pytest.raises(ValueError, match="deployment"):
        status._run_status_from_json(
            {"run_id": "run-a", "state": "done", "spec": {}, "deployments": records}
        )


@pytest.mark.parametrize("serving_generation", [None, "", "   ", 7])
def test_ready_without_serving_generation_cannot_be_committed(run, serving_generation):
    attempt = queued(run, 20)
    transitions.mark_deployment_pending(run, attempt)
    ready_record = {**attempt, "state": "ready", "verified_at": 123.0}
    if serving_generation is not None:
        ready_record["deployment_generation"] = serving_generation
    with pytest.raises(ValueError, match="verification"):
        transitions.mark_checkpoint_deployed(run, ready_record)
    assert status.get_status(run).deployments[attempt["checkpoint_id"]]["state"] == "queued"


@pytest.mark.parametrize("initial_state", ["deployed", "training"])
@pytest.mark.parametrize("with_ready_sibling", [False, True])
def test_failed_legacy_reverification_reconciles_parent_state(
    run, initial_state, with_ready_sibling
):
    if with_ready_sibling:
        ready(run, 40)
    saved = status.get_status(run)
    saved.state = initial_state
    legacy = {**queued(run, 20), "state": "ready", "verified_at": 1.0}
    saved.deployments[legacy["checkpoint_id"]] = legacy
    state._save_status(saved)

    attempt = queued(run, 20, generation=2)
    transitions.mark_deployment_pending(run, attempt, expect_state=initial_state)
    result = transitions.mark_deployment_failed(run, {**attempt, "error": "smoke failed"})

    expected = "done" if initial_state == "deployed" and not with_ready_sibling else initial_state
    assert result.state == status.get_status(run).state == expected
    assert result.deployments[attempt["checkpoint_id"]]["state"] == "failed"
    assert (
        any(is_ready(run, record) for record in result.deployments.values()) == with_ready_sibling
    )
