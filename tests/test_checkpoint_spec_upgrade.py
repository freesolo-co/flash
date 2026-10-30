import copy
import hashlib
import json

import pytest
from fastapi import HTTPException

from flash.core.spec import JobSpec
from flash.runner.lifecycle.state import RunStatus
from flash.schema import spec_from_dict
from flash.server.routes.serving import _validate_deploy_request


def persisted_run():
    spec = JobSpec(
        run_id="completed-sft",
        model="Qwen/Qwen3.5-9B",
        algorithm="sft",
        model_revision="a" * 40,
        model_revision_auto=True,
    )
    public, worker = spec.to_dict(), spec.to_internal_dict()
    public["train"]["teacher_model"] = ""
    worker["train"]["teacher_model"] = ""
    # The old writer's canonical payload, computed independently of today's digest helper.
    hashed_public, hashed_worker = copy.deepcopy(public), copy.deepcopy(worker)
    if not hashed_public["environment"].get("pip"):
        hashed_public["environment"].pop("pip", None)
    for key in (
        "model_revision_auto",
        "model_revision_force_pin",
        "gpu_count_auto",
        "workload_profile_input_digest",
        "workload_profile_producer_version",
        "workload_profile",
    ):
        if not hashed_worker.get(key):
            hashed_worker.pop(key, None)
    payload = {
        "version": 1,
        "public_spec": hashed_public,
        "worker_spec": hashed_worker,
        "adapter_identity": None,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return RunStatus(
        run_id=spec.run_id,
        state="done",
        spec=public,
        effective_preparation={
            "version": 1,
            "worker_spec": worker,
            "adapter_identity": None,
            "preparation_digest": digest,
        },
    )


def test_completed_checkpoint_deployment_survives_removal_of_unused_teacher_default():
    record = persisted_run()
    original = copy.deepcopy(record)
    spec, _ = _validate_deploy_request(record.run_id, record, {}, False)
    assert spec.model == "Qwen/Qwen3.5-9B"
    assert spec.model_revision == "a" * 40
    assert record == original
    assert "teacher_model" not in spec.to_dict()["train"]


@pytest.mark.parametrize("mutation", ["revision", "digest", "teacher", "unknown"])
def test_old_checkpoint_compatibility_does_not_bypass_integrity(mutation):
    record = persisted_run()
    worker = record.effective_preparation["worker_spec"]
    if mutation == "revision":
        worker["model_revision"] = "b" * 40
    elif mutation == "digest":
        record.effective_preparation["preparation_digest"] = "0" * 64
    elif mutation == "teacher":
        worker["train"]["teacher_model"] = "actual-teacher"
    else:
        worker["train"]["invented_option"] = ""
    with pytest.raises(HTTPException) as error:
        _validate_deploy_request(record.run_id, record, {}, False)
    assert error.value.status_code == 409


def test_new_authored_requests_still_reject_removed_teacher_option():
    with pytest.raises(ValueError, match="teacher_model"):
        spec_from_dict(persisted_run().spec)
