"""Fixtures for checkpoint inventories; all authority lives in the persisted run status."""

from flash.runner.lifecycle import state, status
from flash.runner.results.deployments import is_ready
from flash.schema import format_checkpoint_ref, parse_checkpoint_ref


def records(run_id, deployment):
    if not deployment:
        return {}
    record = dict(deployment)
    checkpoint_id = record.setdefault(
        "checkpoint_id", format_checkpoint_ref(run_id, record.get("checkpoint_step"))
    )
    parsed = parse_checkpoint_ref(checkpoint_id)
    record.setdefault("checkpoint_step", parsed[1] if parsed else None)
    return {checkpoint_id: record}


def only_deployment(run):
    assert len(run.deployments) <= 1, "select the exact checkpoint when testing siblings"
    return next(iter(run.deployments.values()), None)


def ready_checkpoints(run_id):
    return frozenset(
        key
        for key, record in status.get_status(run_id).deployments.items()
        if is_ready(run_id, record)
    )


def seed_ready_checkpoint(run_id, checkpoint_id):
    current = status.get_status(run_id)
    record = current.deployments.get(
        checkpoint_id,
        {"checkpoint_id": checkpoint_id, "checkpoint_step": parse_checkpoint_ref(checkpoint_id)[1]},
    )
    current.deployments[checkpoint_id] = {**record, "state": "ready", "verified_at": 1.0}
    state._save_status(current)
    return True


def complete_deployment(run_id, record, expect_state=None):
    from flash.runner.results.deployments import generation
    from flash.runner.supervise import transitions

    record = next(iter(records(run_id, record).values()))
    current = status.get_status(run_id).deployments.get(record["checkpoint_id"], {})
    record = {
        **record,
        "generation": generation(current) + 1,
        "requested_at": record.get("requested_at", 1.0),
    }
    transitions.mark_deployment_pending(run_id, {**record, "state": "queued"}, expect_state)
    return transitions.mark_checkpoint_deployed(
        run_id, {**record, "verified_at": record.get("verified_at", 1.0)}, expect_state
    )
